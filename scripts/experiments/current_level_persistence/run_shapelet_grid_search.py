"""Grid-search learnable-shapelet models for current-level persistence."""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.current_level_persistence.train_learnable_shapelets import (
    CurrentLevelPersistenceDataset,
    build_shapelet_config,
    evaluate_split,
    learned_shapelets_numpy,
    load_npz,
    make_loader,
    make_loss,
    prediction_frame,
    prepare_split_arrays,
    project_path,
    scalar_context_metadata,
    unpack_batch,
)
from src.tasks.current_level_persistence.models.learnable_shapelets import (
    SUPPORTED_MODEL_IDS,
    build_learnable_shapelet_model,
)
from src.tasks.current_level_persistence.plots.diagnostics import (
    plot_duration_distribution,
    plot_error_by_event,
    plot_learned_shapelets,
    plot_residuals,
    plot_true_vs_predicted_duration,
)
from src.tasks.current_level_persistence.utils.paths import (
    RUN_INDEX_COLUMNS,
    grid_search_dir,
    make_run_id,
    model_dir,
    model_selection_dir,
    run_dir,
    run_index_path,
)
from src.tuning.grid_search import (
    apply_flat_overrides,
    deep_merge,
    expand_parameter_grid,
    make_trial_id,
    select_best_trial,
)
from src.tuning.parallel_trials import (
    choose_trial_devices,
    iter_parallel_trial_results,
    prepare_trial_device,
)
from src.tuning.trial_logging import format_trial_start
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.device import select_device
from src.utils.reproducibility import set_seed
from src.utils.results_paths import (
    GRID_SEARCH_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_results_index_dir,
    relative_project_path,
    sanitize_id,
    upsert_index_row,
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/current_level_persistence/shapelet_grid_search_delta_0p5.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse grid-search options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Current-level persistence shapelet grid-search config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the planned grid without training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing search and final best-run artifacts.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override training.device for all trials, e.g. auto, cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable multi-GPU trial scheduling for this invocation.",
    )
    parser.add_argument(
        "--max-trials-per-model",
        type=int,
        default=None,
        help="Temporary safety-limit override for search.max_trials_per_model.",
    )
    return parser.parse_args()


def model_base_config(config: Mapping[str, Any], model_spec: Mapping[str, Any]) -> dict:
    """Build the single-model base config for one grid entry."""

    model_id = str(model_spec["model_id"])
    if model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(f"Unsupported current-level persistence model_id: {model_id}")
    base = {
        "task_name": config["task_name"],
        "model_family": config["model_family"],
        "model_id": model_id,
        "selection_id": config["selection_id"],
        "dataset": copy.deepcopy(config["dataset"]),
        "features": copy.deepcopy(config.get("features", {})),
        "scalar_encoder": copy.deepcopy(config.get("scalar_encoder", {})),
        "target": copy.deepcopy(config.get("target", {})),
        "training": copy.deepcopy(config["training"]),
        "shapelets": copy.deepcopy(config["shapelets"]),
        "model": {},
        "evaluation": {
            "primary_selection_metric": config["search"]["selection_metric"],
        },
    }
    return deep_merge(base, model_spec.get("fixed", {}))


def build_trial_config(
    config: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict:
    """Build one fully resolved single-model trial config."""

    return apply_flat_overrides(
        model_base_config(config, model_spec),
        parameters,
    )


def expand_model_trials(
    config: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand one model-specific grid into resolved trial configs."""

    candidates = expand_parameter_grid(model_spec["parameter_grid"])
    return [
        {
            "model_id": str(model_spec["model_id"]),
            "parameters": parameters,
            "config": build_trial_config(config, model_spec, parameters),
        }
        for parameters in candidates
    ]


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate grid-search config structure before starting expensive work."""

    if config.get("task_name") != "current_level_persistence":
        raise ValueError("Config task_name must be current_level_persistence.")
    if config.get("model_family") != "learnable_shapelets":
        raise ValueError("Config model_family must be learnable_shapelets.")
    required = [
        "search_id",
        "selection_id",
        "dataset",
        "training",
        "shapelets",
        "search",
        "models",
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Grid config is missing sections: {missing}")
    search = config["search"]
    if search.get("selection_mode") not in {"min", "max"}:
        raise ValueError("search.selection_mode must be 'min' or 'max'.")
    if not str(search.get("selection_metric", "")).startswith("val_"):
        raise ValueError("search.selection_metric must be a validation metric.")
    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list.")
    seen = set()
    max_trials = int(search["max_trials_per_model"])
    for model_spec in models:
        model_id = str(model_spec.get("model_id", ""))
        if model_id in seen:
            raise ValueError(f"Duplicate model_id in grid config: {model_id}")
        seen.add(model_id)
        if model_id not in SUPPORTED_MODEL_IDS:
            raise ValueError(f"Unsupported model_id in grid config: {model_id}")
        if "parameter_grid" not in model_spec:
            raise ValueError(f"Missing parameter_grid for {model_id}.")
        count = len(expand_parameter_grid(model_spec["parameter_grid"]))
        if count > max_trials:
            raise ValueError(
                f"Configured {model_id} grid has {count} trials, exceeding "
                f"search.max_trials_per_model={max_trials}. Reduce the grid or "
                "explicitly increase the safety limit; no combinations were skipped."
            )
    if not bool(search.get("save_trial_checkpoints", True)):
        raise ValueError(
            "search.save_trial_checkpoints must be true because final best-model "
            "materialization needs the selected validation checkpoint."
        )
    features = config.get("features", {})
    if bool(features.get("use_scalar_context", False)):
        dataset = config["dataset"]
        for key in ("raw_input_key", "scalar_context_key"):
            if key not in dataset:
                raise ValueError(
                    f"dataset.{key} is required when scalar context is enabled."
                )


def _metric_without_split(metric: str) -> str:
    if not metric.startswith("val_"):
        raise ValueError(f"Selection metric must start with val_: {metric}")
    return metric.removeprefix("val_")


def _state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _autocast_context(*, amp_enabled: bool):
    return torch.autocast(device_type="cuda") if amp_enabled else nullcontext()


def _make_scaler(*, amp_enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler,
    amp_enabled: bool,
    gradient_clip_norm: float | None,
) -> float:
    model.train()
    train_loss_sum = 0.0
    n_train = 0
    for batch in loader:
        inputs, scalar_inputs, targets = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(amp_enabled=amp_enabled):
            predictions = model(inputs, scalar_inputs)
            loss = loss_fn(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                    error_if_nonfinite=True,
                )
            except TypeError:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise RuntimeError(f"Non-finite gradient norm: {grad_norm}")
        scaler.step(optimizer)
        scaler.update()
        batch_size = len(targets)
        train_loss_sum += float(loss.detach().cpu().item()) * batch_size
        n_train += batch_size
    return train_loss_sum / max(n_train, 1)


def _build_model(config: dict[str, Any], context_length: int, device: torch.device):
    model = build_learnable_shapelet_model(
        str(config["model_id"]),
        build_shapelet_config(config, context_length),
    )
    return model.to(device)


def _build_optimizer(config: Mapping[str, Any], model: nn.Module):
    training = config["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def _loader_kwargs(training: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }


def build_split_loaders(
    split_arrays: Mapping[str, Mapping[str, np.ndarray]],
    *,
    config: Mapping[str, Any],
    device: torch.device,
    include_test: bool,
) -> dict[str, DataLoader]:
    """Build train/validation/test loaders for a resolved config."""

    training = config["training"]
    kwargs = _loader_kwargs(training, device)
    input_key = str(config["dataset"]["input_key"])
    target_key = str(config["dataset"]["target_key"])
    scalar_context_key = (
        str(
            config.get("features", {}).get(
                "scalar_context_key",
                config["dataset"].get("scalar_context_key"),
            )
        )
        if bool(config.get("features", {}).get("use_scalar_context", False))
        else None
    )
    splits = ("train", "val", "test") if include_test else ("train", "val")
    return {
        split: make_loader(
            split_arrays[split],
            input_key=input_key,
            target_key=target_key,
            scalar_context_key=scalar_context_key,
            shuffle=split == "train",
            **kwargs,
        )
        for split in splits
    }


def build_full_development_loader(
    split_arrays: Mapping[str, Mapping[str, np.ndarray]],
    *,
    config: Mapping[str, Any],
    device: torch.device,
) -> DataLoader:
    """Build a shuffled train+validation loader for final retraining."""

    training = config["training"]
    input_key = str(config["dataset"]["input_key"])
    target_key = str(config["dataset"]["target_key"])
    scalar_context_key = (
        str(
            config.get("features", {}).get(
                "scalar_context_key",
                config["dataset"].get("scalar_context_key"),
            )
        )
        if bool(config.get("features", {}).get("use_scalar_context", False))
        else None
    )
    dataset = ConcatDataset(
        [
            CurrentLevelPersistenceDataset(
                split_arrays["train"],
                input_key,
                target_key,
                scalar_context_key,
            ),
            CurrentLevelPersistenceDataset(
                split_arrays["val"],
                input_key,
                target_key,
                scalar_context_key,
            ),
        ]
    )
    generator = torch.Generator()
    generator.manual_seed(int(training.get("seed", 42)))
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )


def train_trial_model(
    *,
    config: dict[str, Any],
    split_arrays: Mapping[str, Mapping[str, np.ndarray]],
    val_metadata: pd.DataFrame,
    context_length: int,
    selection_metric: str,
    selection_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    """Train one validation-only trial and return its best state and metrics."""

    training = config["training"]
    set_seed(int(training.get("seed", 42)))
    loaders = build_split_loaders(
        split_arrays,
        config=config,
        device=device,
        include_test=False,
    )
    model = _build_model(config, context_length, device)
    loss_fn = make_loss(str(training.get("loss", "huber")))
    optimizer = _build_optimizer(config, model)
    amp_enabled = bool(training.get("mixed_precision", False)) and device.type == "cuda"
    scaler = _make_scaler(amp_enabled=amp_enabled)
    gradient_clip_norm = training.get("gradient_clip_norm")
    gradient_clip_norm = (
        float(gradient_clip_norm) if gradient_clip_norm is not None else None
    )
    patience = int(training["early_stopping_patience"])
    best_metric = np.inf if selection_mode == "min" else -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    history = []
    epochs_without_improvement = 0
    metric_key = _metric_without_split(selection_metric)

    for epoch in range(1, int(training["max_epochs"]) + 1):
        train_loss = _train_epoch(
            model,
            loaders["train"],
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_clip_norm=gradient_clip_norm,
        )
        val_metrics, _, _ = evaluate_split(
            model,
            loaders["val"],
            val_metadata,
            loss_fn=loss_fn,
            device=device,
        )
        if metric_key not in val_metrics:
            raise ValueError(f"Selection metric is not available: {selection_metric}")
        current_metric = float(val_metrics[metric_key])
        improved = (
            current_metric < best_metric
            if selection_mode == "min"
            else current_metric > best_metric
        )
        history.append(
            {
                "epoch": epoch,
                "train_log_loss": train_loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            best_state = _state_dict_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")
    return {
        "best_epoch": best_epoch,
        "best_state_dict": best_state,
        "best_metrics": best_metrics,
        "history": history,
        "model_config": copy.deepcopy(config),
    }


def train_fixed_epochs(
    *,
    config: dict[str, Any],
    loader: DataLoader,
    context_length: int,
    epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train one final model for a fixed number of epochs on train+validation."""

    training = config["training"]
    set_seed(int(training.get("seed", 42)))
    model = _build_model(config, context_length, device)
    loss_fn = make_loss(str(training.get("loss", "huber")))
    optimizer = _build_optimizer(config, model)
    amp_enabled = bool(training.get("mixed_precision", False)) and device.type == "cuda"
    scaler = _make_scaler(amp_enabled=amp_enabled)
    gradient_clip_norm = training.get("gradient_clip_norm")
    gradient_clip_norm = (
        float(gradient_clip_norm) if gradient_clip_norm is not None else None
    )
    history = []
    for epoch in range(1, int(epochs) + 1):
        train_loss = _train_epoch(
            model,
            loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            gradient_clip_norm=gradient_clip_norm,
        )
        history.append({"epoch": epoch, "train_log_loss": train_loss})
    return {
        "model": model,
        "optimizer": optimizer,
        "history": history,
    }


def load_dataset_bundle(dataset_path: Path) -> tuple[dict, dict, dict]:
    """Load split arrays, split metadata, and dataset metadata."""

    split_arrays = {
        split: load_npz(dataset_path / f"{split}.npz")
        for split in ("train", "val", "test")
    }
    split_metadata = {
        split: pd.read_parquet(dataset_path / f"{split}_metadata.parquet")
        for split in ("train", "val", "test")
    }
    dataset_metadata = load_yaml_config(dataset_path / "dataset_metadata.yaml")
    return split_arrays, split_metadata, dataset_metadata


def trial_display_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact effective parameters for progress logging."""

    return {
        "model": {
            "architecture": config["model_id"],
            "context_length": config.get("context_length"),
            **dict(config.get("shapelets", {})),
            **dict(config.get("model", {})),
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
        },
        "training": {
            key: config["training"].get(key)
            for key in (
                "batch_size",
                "max_epochs",
                "early_stopping_patience",
                "gradient_clip_norm",
                "mixed_precision",
                "seed",
                "loss",
            )
            if key in config["training"]
        },
    }


def run_trial_job(job: dict[str, Any]) -> dict[str, Any]:
    """Train one current-level persistence trial on its assigned worker device."""

    config = job["config"]
    trial_id = str(job["trial_id"])
    model_id = str(job["model_id"])
    checkpoint_path = Path(job["checkpoint_path"])
    history_path = Path(job["history_path"])
    device = prepare_trial_device(str(job["device"]))
    start = time.perf_counter()
    print(
        format_trial_start(
            index=int(job["index"]),
            total=int(job["total"]),
            run_id=str(job["target_run_id"]),
            trial_id=trial_id,
            device=device,
            parameters=trial_display_parameters(config),
        ),
        flush=True,
    )
    try:
        split_arrays, split_metadata, dataset_metadata = load_dataset_bundle(
            Path(job["dataset_path"])
        )
        split_arrays, scalar_scaler = prepare_split_arrays(
            split_arrays,
            config,
            delta=float(dataset_metadata["delta"]),
        )
        result = train_trial_model(
            config=config,
            split_arrays=split_arrays,
            val_metadata=split_metadata["val"],
            context_length=int(job["context_length"]),
            selection_metric=str(job["selection_metric"]),
            selection_mode=str(job["selection_mode"]),
            device=device,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if bool(job["save_trial_checkpoints"]):
            torch.save(
                {
                    "trial_id": trial_id,
                    "model_id": model_id,
                    "best_epoch": result["best_epoch"],
                    "best_state_dict": result["best_state_dict"],
                    "best_metrics": result["best_metrics"],
                    "history": result["history"],
                    "config": result["model_config"],
                    "scalar_context_scaler": scalar_scaler,
                },
                checkpoint_path,
            )
        if bool(job["save_trial_history"]):
            history_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(result["history"]).to_csv(history_path, index=False)
        row = {
            "trial_id": trial_id,
            "status": "complete",
            "model_id": model_id,
            **job["parameters"],
            **{
                f"val_{key}": value
                for key, value in result["best_metrics"].items()
            },
            "best_epoch": int(result["best_epoch"]),
            "epochs_run": len(result["history"]),
            "duration_seconds": time.perf_counter() - start,
            "checkpoint_path": relative_project_path(checkpoint_path),
            "device": str(device),
            "error": "",
        }
        print(
            f"  completed {trial_id} device={device} "
            f"{job['selection_metric']}={row[job['selection_metric']]:.6g} "
            f"epoch={row['best_epoch']}",
            flush=True,
        )
        return row
    except Exception as error:
        print(f"  failed {trial_id} device={device}: {error}", flush=True)
        return {
            "trial_id": trial_id,
            "status": "failed",
            "model_id": model_id,
            **job["parameters"],
            str(job["selection_metric"]): np.nan,
            "best_epoch": np.nan,
            "epochs_run": np.nan,
            "duration_seconds": time.perf_counter() - start,
            "checkpoint_path": relative_project_path(checkpoint_path),
            "device": str(device),
            "error": repr(error),
        }


def _save_trial_tables(
    search_dir: Path,
    rows: list[dict[str, Any]],
    model_ids: list[str],
) -> None:
    for model_id in model_ids:
        model_rows = [row for row in rows if row.get("model_id") == model_id]
        if not model_rows:
            continue
        table_dir = search_dir / sanitize_id(model_id) / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(model_rows).to_csv(table_dir / "trials.csv", index=False)


def _load_trial_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_canonical_best_run(
    *,
    best_trial: pd.Series,
    trial_checkpoint: Mapping[str, Any],
    search_id: str,
    dataset_path: Path,
    split_arrays: Mapping[str, Mapping[str, np.ndarray]],
    split_metadata: Mapping[str, pd.DataFrame],
    dataset_metadata: Mapping[str, Any],
    config: dict[str, Any],
    output: Mapping[str, Any],
    final_training: Mapping[str, Any],
    selection_metric: str,
    device: torch.device,
    scalar_context_scaler: Mapping | None,
) -> dict[str, Any]:
    """Save the selected model and its canonical test outputs."""

    model_id = str(config["model_id"])
    delta = float(dataset_metadata["delta"])
    context_length = int(dataset_metadata["context_length"])
    selection_id = str(config.get("selection_id", dataset_metadata["selection_id"]))
    run_id = make_run_id(
        delta=delta,
        context_length=context_length,
        model_id=model_id,
        selection_id=selection_id,
        scalar_context=bool(
            config.get("features", {}).get("use_scalar_context", False)
        ),
    )
    result_dir = run_dir(run_id)
    checkpoint_dir = model_dir(run_id)
    if bool(output.get("overwrite", False)):
        shutil.rmtree(result_dir, ignore_errors=True)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    elif result_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"Canonical best-run artifacts already exist: {run_id}")

    result_paths = ensure_results_subdirs(
        result_dir,
        ("metrics", "predictions", "figures", "tables"),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training = config["training"]
    loss_fn = make_loss(str(training.get("loss", "huber")))
    loaders = build_split_loaders(
        split_arrays,
        config=config,
        device=device,
        include_test=True,
    )
    final_retraining_enabled = bool(
        final_training.get("retrain_on_full_development", False)
    )
    if final_retraining_enabled:
        if not final_training.get("use_best_epoch_from_grid", False):
            raise ValueError("Final retraining requires use_best_epoch_from_grid=true.")
        if final_training.get("early_stopping", True):
            raise ValueError("Final retraining must use early_stopping=false.")
        final_result = train_fixed_epochs(
            config=config,
            loader=build_full_development_loader(
                split_arrays,
                config=config,
                device=device,
            ),
            context_length=context_length,
            epochs=int(best_trial["best_epoch"]),
            device=device,
        )
        model = final_result["model"]
        optimizer = final_result["optimizer"]
        history = final_result["history"]
        final_training_epochs = int(best_trial["best_epoch"])
    else:
        model = _build_model(config, context_length, device)
        model.load_state_dict(trial_checkpoint["best_state_dict"])
        optimizer = _build_optimizer(config, model)
        history = list(trial_checkpoint["history"])
        final_training_epochs = len(history)

    test_metrics, test_true, test_pred = evaluate_split(
        model,
        loaders["test"],
        split_metadata["test"],
        loss_fn=loss_fn,
        device=device,
    )
    test_predictions = prediction_frame(
        split_metadata["test"],
        test_true,
        test_pred,
        model_id=model_id,
        run_id=run_id,
        split="test",
    )
    if bool(output.get("save_best_predictions", True)):
        test_predictions.to_parquet(
            result_paths["predictions"] / "test_predictions.parquet",
            index=False,
        )

    val_metric_columns = {
        column: best_trial[column]
        for column in best_trial.index
        if str(column).startswith("val_")
    }
    metrics_summary = {
        "run_id": run_id,
        "scalar_context_scaler": dict(scalar_context_scaler)
        if scalar_context_scaler is not None
        else None,
        "model_id": model_id,
        "best_trial_id": best_trial["trial_id"],
        "best_epoch": int(best_trial["best_epoch"]),
        "final_training_epochs": final_training_epochs,
        "final_retrained_on_full_development": final_retraining_enabled,
        **{key: python_scalar(value) for key, value in val_metric_columns.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    pd.DataFrame([metrics_summary]).to_csv(
        result_paths["metrics"] / "metrics_summary.csv",
        index=False,
    )
    pd.DataFrame([{f"test_{key}": value for key, value in test_metrics.items()}]).to_csv(
        result_paths["metrics"] / "test_metrics.csv",
        index=False,
    )
    pd.DataFrame(history).to_csv(
        result_paths["tables"] / "training_history.csv",
        index=False,
    )

    if bool(output.get("save_best_plots", True)):
        plot_true_vs_predicted_duration(
            test_predictions,
            result_paths["figures"] / "y_true_vs_y_pred_test.png",
        )
        plot_residuals(test_predictions, result_paths["figures"] / "residuals_test.png")
        plot_error_by_event(
            test_predictions,
            result_paths["figures"] / "error_by_event_test.png",
        )
        plot_duration_distribution(
            test_predictions,
            result_paths["figures"] / "predicted_vs_true_distribution_test.png",
        )
        shapelets = learned_shapelets_numpy(model)
        plot_learned_shapelets(shapelets, result_paths["figures"])
        for length, values in shapelets.items():
            np.save(result_paths["tables"] / f"learned_shapelets_L{length}.npy", values)

    created_at = datetime.now(timezone.utc).isoformat()
    checkpoint = {
        "model_state_dict": _state_dict_cpu(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "grid_search_id": search_id,
        "best_trial_id": best_trial["trial_id"],
        "best_epoch": int(best_trial["best_epoch"]),
        "final_retrained_on_full_development": final_retraining_enabled,
        "config": config,
        "run_id": run_id,
        "dataset_build_fingerprint": config_fingerprint(dataset_metadata),
    }
    torch.save(checkpoint, checkpoint_dir / "best_model.pt")

    metadata = {
        "run_id": run_id,
        "task_name": "current_level_persistence",
        "model_family": "learnable_shapelets",
        "model_id": model_id,
        "selection_id": selection_id,
        "delta": delta,
        "context_length": context_length,
        "dataset_path": relative_project_path(dataset_path),
        "dataset_build_fingerprint": config_fingerprint(dataset_metadata),
        "target_search_scope": dataset_metadata.get("target_search_scope"),
        "censoring_policy": dataset_metadata.get("censoring_policy"),
        "grid_search_id": search_id,
        "best_trial_id": best_trial["trial_id"],
        "selection_metric": selection_metric,
        "best_epoch": int(best_trial["best_epoch"]),
        "final_training": {
            "retrain_on_full_development": final_retraining_enabled,
            "train_split": "train+val" if final_retraining_enabled else "train",
            "excluded_split": "test",
            "epochs": final_training_epochs,
        },
        "device": str(device),
        "results_path": relative_project_path(result_dir),
        "checkpoint_path": relative_project_path(checkpoint_dir),
        "created_at": created_at,
        **scalar_context_metadata(config, scalar_context_scaler),
    }
    save_yaml(result_dir / "metadata.yaml", metadata)
    save_yaml(result_dir / "config_resolved.yaml", config)
    save_yaml(checkpoint_dir / "metadata.yaml", metadata)
    save_yaml(checkpoint_dir / "config_resolved.yaml", config)

    upsert_index_row(
        run_index_path(),
        {
            "run_id": run_id,
            "task_name": "current_level_persistence",
            "model_family": "learnable_shapelets",
            "model_id": model_id,
            "delta": delta,
            "context_length": context_length,
            "selection_id": selection_id,
            "dataset_path": relative_project_path(dataset_path),
            "best_epoch": int(best_trial["best_epoch"]),
            "best_val_mae_seconds": float(best_trial["val_mae_seconds"]),
            "best_val_rmse_seconds": float(best_trial["val_rmse_seconds"]),
            "test_mae_seconds": test_metrics["mae_seconds"],
            "test_rmse_seconds": test_metrics["rmse_seconds"],
            "test_median_ae_seconds": test_metrics["median_ae_seconds"],
            "created_at": created_at,
            "status": "complete",
        },
        id_column="run_id",
        columns=RUN_INDEX_COLUMNS,
    )
    return {
        "run_id": run_id,
        "model_id": model_id,
        "best_trial_id": best_trial["trial_id"],
        "best_val_mae_seconds": float(best_trial["val_mae_seconds"]),
        "best_val_rmse_seconds": float(best_trial["val_rmse_seconds"]),
        "test_mae_seconds": test_metrics["mae_seconds"],
        "test_rmse_seconds": test_metrics["rmse_seconds"],
        "results_path": relative_project_path(result_dir),
        "model_path": relative_project_path(checkpoint_dir),
        "created_at": created_at,
    }


def python_scalar(value: Any) -> Any:
    """Convert NumPy/Pandas scalars to YAML/CSV-safe Python scalars."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def build_jobs(
    *,
    config: dict[str, Any],
    dataset_path: Path,
    delta: float,
    context_length: int,
    search_dir: Path,
) -> list[dict[str, Any]]:
    """Build serializable trial jobs for all configured models."""

    jobs = []
    total = sum(len(expand_model_trials(config, spec)) for spec in config["models"])
    index = 1
    for model_spec in config["models"]:
        model_id = str(model_spec["model_id"])
        target_run_id = make_run_id(
            delta=delta,
            context_length=context_length,
            model_id=model_id,
            selection_id=str(config["selection_id"]),
            scalar_context=bool(
                config.get("features", {}).get("use_scalar_context", False)
            ),
        )
        for trial_index, trial in enumerate(
            expand_model_trials(config, model_spec),
            start=1,
        ):
            trial_id = make_trial_id(
                {"model_id": model_id, **trial["parameters"]},
                trial_index,
            )
            trial_config = copy.deepcopy(trial["config"])
            trial_config["context_length"] = context_length
            trial_dir = search_dir / sanitize_id(model_id) / "trials" / trial_id
            jobs.append(
                {
                    "index": index,
                    "total": total,
                    "trial_id": trial_id,
                    "model_id": model_id,
                    "target_run_id": target_run_id,
                    "parameters": trial["parameters"],
                    "config": trial_config,
                    "dataset_path": str(dataset_path),
                    "context_length": context_length,
                    "selection_metric": config["search"]["selection_metric"],
                    "selection_mode": config["search"]["selection_mode"],
                    "checkpoint_path": str(trial_dir / "best_model.pt"),
                    "history_path": str(trial_dir / "history.csv"),
                    "save_trial_checkpoints": bool(
                        config["search"].get("save_trial_checkpoints", True)
                    ),
                    "save_trial_history": bool(
                        config["search"].get("save_trial_history", True)
                    ),
                }
            )
            index += 1
    return jobs


def main() -> None:
    """Run the configured validation-only grid search and final best evaluations."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if args.device is not None:
        config["training"]["device"] = str(args.device)
    if args.max_trials_per_model is not None:
        config["search"]["max_trials_per_model"] = int(args.max_trials_per_model)
    if args.no_parallel:
        config.setdefault("parallel", {})["enabled"] = False
    validate_config(config)

    dataset_path = project_path(config["dataset"]["path"])
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}. Build it first."
        )
    split_arrays, split_metadata, dataset_metadata = load_dataset_bundle(dataset_path)
    split_arrays, scalar_scaler = prepare_split_arrays(
        split_arrays,
        config,
        delta=float(dataset_metadata["delta"]),
    )
    context_length = int(dataset_metadata["context_length"])
    selection_id = str(config.get("selection_id", dataset_metadata["selection_id"]))
    search_id = str(config["search_id"])
    search_dir = grid_search_dir(selection_id=selection_id, search_id=search_id)

    output_overwrite = bool(config.get("output", {}).get("overwrite", False)) or args.force

    requested_device = str(config["training"].get("device", "auto"))
    fallback_device = select_device() if requested_device == "auto" else requested_device
    devices = choose_trial_devices(
        config.get("parallel", {}),
        fallback_device=fallback_device,
    )
    jobs = build_jobs(
        config=config,
        dataset_path=dataset_path,
        delta=float(dataset_metadata["delta"]),
        context_length=context_length,
        search_dir=search_dir,
    )
    model_ids = [str(model["model_id"]) for model in config["models"]]
    model_trial_counts = {
        model_id: sum(job["model_id"] == model_id for job in jobs)
        for model_id in model_ids
    }

    print(
        "=== Current-Level Persistence Shapelet Grid Search ===\n"
        f"Config: {relative_project_path(config_path)}\n"
        f"Search ID: {search_id}\n"
        f"Dataset: {relative_project_path(dataset_path)}\n"
        f"Selection: {selection_id}\n"
        f"Selection metric: {config['search']['selection_metric']} "
        f"({config['search']['selection_mode']})\n"
        f"Trial counts: {model_trial_counts}\n"
        f"Total trials: {len(jobs)}\n"
        f"Devices: {devices}\n"
        "Trial phase uses train+validation only; test is evaluated only for "
        "the selected model per architecture.",
        flush=True,
    )
    if args.dry_run:
        print("Dry-run completed. No training was started.")
        return

    if search_dir.exists() and output_overwrite:
        shutil.rmtree(search_dir)
    elif search_dir.exists():
        raise FileExistsError(
            f"Grid-search folder already exists: {relative_project_path(search_dir)}. "
            "Use --force or set output.overwrite=true."
        )
    search_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(
        search_dir / "metadata.yaml",
        {
            "search_id": search_id,
            "task_name": "current_level_persistence",
            "model_family": "learnable_shapelets",
            "selection_id": selection_id,
            "dataset_path": relative_project_path(dataset_path),
            "dataset_build_fingerprint": config_fingerprint(dataset_metadata),
            "target_search_scope": dataset_metadata.get("target_search_scope"),
            "censoring_policy": dataset_metadata.get("censoring_policy"),
            "config_path": relative_project_path(config_path),
            "config_fingerprint": config_fingerprint(config),
            "trial_counts": model_trial_counts,
            "total_trials": len(jobs),
            "devices": devices,
            "selection_metric": config["search"]["selection_metric"],
            "selection_mode": config["search"]["selection_mode"],
            "test_policy": "test split evaluated only after validation selection",
            **scalar_context_metadata(config, scalar_scaler),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    save_yaml(search_dir / "config_resolved.yaml", config)

    rows = []
    for row in iter_parallel_trial_results(jobs, run_trial_job, devices):
        rows.append(row)
        _save_trial_tables(search_dir, rows, model_ids)

    final_device = prepare_trial_device(devices[0])
    final_rows = []
    for model_id in model_ids:
        model_dir_name = sanitize_id(model_id)
        trials_path = search_dir / model_dir_name / "tables" / "trials.csv"
        trials = pd.read_csv(trials_path)
        try:
            best_trial = select_best_trial(
                trials,
                metric=str(config["search"]["selection_metric"]),
                mode=str(config["search"]["selection_mode"]),
            )
        except ValueError as error:
            save_yaml(
                search_dir / model_dir_name / "best_trial_error.yaml",
                {
                    "model_id": model_id,
                    "status": "failed",
                    "reason": str(error),
                    "selection_metric": str(config["search"]["selection_metric"]),
                },
            )
            print(
                f"Skipping final best run for {model_id}: {error}",
                flush=True,
            )
            continue
        checkpoint_path = PROJECT_ROOT / str(best_trial["checkpoint_path"])
        trial_checkpoint = _load_trial_checkpoint(checkpoint_path)
        best_config = dict(trial_checkpoint["config"])
        save_yaml(
            search_dir / model_dir_name / "best_trial.yaml",
            {
                key: python_scalar(value)
                for key, value in best_trial.to_dict().items()
            },
        )
        save_yaml(
            search_dir / model_dir_name / "best_config.yaml",
            best_config,
        )
        final_row = save_canonical_best_run(
            best_trial=best_trial,
            trial_checkpoint=trial_checkpoint,
            search_id=search_id,
            dataset_path=dataset_path,
            split_arrays=split_arrays,
            split_metadata=split_metadata,
            dataset_metadata=dataset_metadata,
            config=best_config,
            output={**config.get("output", {}), "overwrite": output_overwrite},
            final_training=config.get("final_training", {}),
            selection_metric=str(config["search"]["selection_metric"]),
            device=final_device,
            scalar_context_scaler=scalar_scaler,
        )
        final_rows.append(final_row)
        upsert_index_row(
            get_results_index_dir() / "grid_searches.csv",
            {
                "search_id": f"{search_id}_{model_dir_name}",
                "model_family": "learnable_shapelets",
                "target_run_id": final_row["run_id"],
                "selection_id": selection_id,
                "selection_metric": str(config["search"]["selection_metric"]),
                "best_trial_id": final_row["best_trial_id"],
                "results_path": relative_project_path(search_dir / model_dir_name),
                "best_model_path": final_row["model_path"],
                "status": "complete",
                "created_at": final_row["created_at"],
            },
            id_column="search_id",
            columns=GRID_SEARCH_INDEX_COLUMNS,
        )

    best_runs = pd.DataFrame(final_rows)
    (search_dir / "tables").mkdir(parents=True, exist_ok=True)
    best_runs.to_csv(search_dir / "tables" / "best_runs.csv", index=False)
    selection_dir = model_selection_dir(
        selection_id=selection_id,
        search_id=search_id,
    )
    (selection_dir / "tables").mkdir(parents=True, exist_ok=True)
    best_runs.to_csv(selection_dir / "tables" / "best_runs.csv", index=False)
    save_yaml(
        selection_dir / "metadata.yaml",
        {
            "search_id": search_id,
            "task_name": "current_level_persistence",
            "model_family": "learnable_shapelets",
            "selection_id": selection_id,
            "selection_metric": str(config["search"]["selection_metric"]),
            "selection_mode": str(config["search"]["selection_mode"]),
            "source_grid_search_path": relative_project_path(search_dir),
            "summary_table": relative_project_path(
                selection_dir / "tables" / "best_runs.csv"
            ),
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    if not final_rows:
        raise RuntimeError("No model produced a completed finite validation trial.")
    print("=== Selected best runs ===")
    for row in final_rows:
        print(
            f"{row['model_id']} | run={row['run_id']} | "
            f"val MAE={row['best_val_mae_seconds']:.6g} | "
            f"test MAE={row['test_mae_seconds']:.6g} | "
            f"test RMSE={row['test_rmse_seconds']:.6g}"
        )
    print(f"Grid results: {search_dir}")


if __name__ == "__main__":
    main()
