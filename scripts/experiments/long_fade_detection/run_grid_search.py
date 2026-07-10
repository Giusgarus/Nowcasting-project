"""Grid-search first-stage long-fade detection classifiers."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import shutil
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.long_fade_detection.run_model import (  # noqa: E402
    LongFadeDataset,
    evaluate_predictions,
    load_npz,
    neural_predictions,
    positive_weight,
    predict_neural,
    predict_sklearn,
    prediction_frame,
    project_path,
    save_outputs,
    xgboost_classifier,
)
from src.tasks.long_fade_detection.models.classifiers import (  # noqa: E402
    MODEL_ID_SHAPELET_CONV,
    MODEL_ID_TCN,
    MODEL_ID_XGBOOST,
    SUPPORTED_MODEL_IDS,
    build_neural_classifier,
)
from src.tasks.long_fade_detection.utils.paths import (  # noqa: E402
    grid_search_dir,
    make_run_id,
    model_dir,
    model_selection_dir,
    run_dir,
)
from src.tuning.grid_search import (  # noqa: E402
    expand_parameter_grid,
    make_trial_id,
    select_best_trial,
)
from src.tuning.parallel_trials import (  # noqa: E402
    choose_trial_devices,
    iter_parallel_trial_results,
    prepare_trial_device,
)
from src.tuning.trial_logging import format_trial_start  # noqa: E402
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.device import select_device  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402
from src.utils.results_paths import (  # noqa: E402
    GRID_SEARCH_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_results_index_dir,
    relative_project_path,
    sanitize_id,
    upsert_index_row,
)

CONFIG_PATH = (
    PROJECT_ROOT / "configs/long_fade_detection/grid_search_threshold10_duration300.yaml"
)
SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    """Parse grid-search options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--max-trials-per-model", type=int, default=None)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help=(
            "Skip trial execution and use the existing trials.csv to refit/save "
            "the selected best runs."
        ),
    )
    return parser.parse_args()


INTEGER_PARAMETER_NAMES = {
    "model.n_estimators",
    "model.max_depth",
    "model.n_jobs",
    "model.hidden_channels",
    "model.num_blocks",
    "model.kernel_size",
    "model.n_shapelets_per_length",
    "shapelets.n_shapelets_per_length",
    "model.conv_channels",
    "model.num_conv_layers",
    "model.scalar_context_dim",
    "model.scalar_hidden_dim",
    "training.batch_size",
    "training.max_epochs",
    "training.early_stopping_patience",
    "training.seed",
}


def deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``base`` recursively updated by ``update``."""

    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_flat_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict:
    """Return a copy of config with dotted-key overrides applied."""

    result = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        target = result
        parts = str(dotted_key).split(".")
        if not parts or any(part == "" for part in parts):
            raise ValueError(f"Invalid override key: {dotted_key!r}")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = copy.deepcopy(value)
    return result


def model_base_config(config: Mapping[str, Any], model_spec: Mapping[str, Any]) -> dict:
    """Build the single-run base config for one model family."""

    model_id = str(model_spec["model_id"])
    if model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(f"Unsupported long-fade model_id: {model_id}")
    base = {
        "task_name": config["task_name"],
        "model_id": model_id,
        "event_definition": copy.deepcopy(config["event_definition"]),
        "source": copy.deepcopy(config["source"]),
        "dataset": copy.deepcopy(config["dataset"]),
        "split": copy.deepcopy(config["split"]),
        "training": copy.deepcopy(config.get("training_defaults", {})),
        "shapelets": {},
        "model": {},
        "output": copy.deepcopy(config.get("output", {})),
    }
    return deep_merge(base, model_spec.get("fixed", {}))


def build_trial_config(
    config: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict:
    """Build one fully resolved trial config."""

    return apply_flat_overrides(model_base_config(config, model_spec), parameters)


def expand_model_trials(
    config: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand one model-specific parameter grid."""

    candidates = expand_parameter_grid(model_spec["parameter_grid"])
    return [
        {
            "model_id": str(model_spec["model_id"]),
            "parameters": parameters,
            "config": build_trial_config(config, model_spec, parameters),
        }
        for parameters in candidates
    ]


def validate_config(config: Mapping[str, Any]) -> dict[str, int]:
    """Validate the long-fade grid config and return trial counts."""

    if config.get("task_name") != "long_fade_detection":
        raise ValueError("Config task_name must be long_fade_detection.")
    required = [
        "search_id",
        "selection_id",
        "event_definition",
        "source",
        "dataset",
        "split",
        "search",
        "models",
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Long-fade grid config is missing sections: {missing}")
    search = config["search"]
    if search.get("selection_metric") != "val_auprc":
        raise ValueError("Long-fade grid selection_metric must be val_auprc.")
    if search.get("selection_mode") != "max":
        raise ValueError("Long-fade grid selection_mode must be max.")
    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list.")
    seen = set()
    counts = {}
    max_trials = int(search["max_trials_per_model"])
    for model_spec in models:
        model_id = str(model_spec.get("model_id", ""))
        if model_id in seen:
            raise ValueError(f"Duplicate model_id in grid config: {model_id}")
        if model_id not in SUPPORTED_MODEL_IDS:
            raise ValueError(f"Unsupported model_id in grid config: {model_id}")
        if "parameter_grid" not in model_spec:
            raise ValueError(f"Missing parameter_grid for {model_id}.")
        count = len(expand_parameter_grid(model_spec["parameter_grid"]))
        if count > max_trials:
            raise ValueError(
                f"Configured {model_id} grid has {count} trials, exceeding "
                f"search.max_trials_per_model={max_trials}. Increase the safety "
                "limit explicitly; no combinations were skipped."
            )
        seen.add(model_id)
        counts[model_id] = count
    return counts


def load_split_bundle(
    dataset_path: Path,
    *,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame], dict[str, Any]]:
    """Load requested splits and dataset metadata."""

    arrays = {
        split: load_npz(dataset_path / f"{SPLIT_FILES[split]}.npz")
        for split in splits
    }
    metadata = {
        split: pd.read_parquet(dataset_path / f"{SPLIT_FILES[split]}_metadata.parquet")
        for split in splits
    }
    dataset_metadata = load_yaml_config(dataset_path / "dataset_metadata.yaml")
    return arrays, metadata, dataset_metadata


def resolve_trial_device_config(config: dict[str, Any], device_name: str) -> dict:
    """Return config with device-specific settings resolved for this trial."""

    result = copy.deepcopy(config)
    if result["model_id"] == MODEL_ID_XGBOOST:
        device = str(result.get("model", {}).get("device", "auto"))
        if device == "auto":
            result.setdefault("model", {})["device"] = (
                device_name if device_name.startswith("cuda") else "cpu"
            )
    return result


def fit_xgboost_arrays(config: dict[str, Any], x: np.ndarray, y: np.ndarray):
    """Fit XGBoost/fallback directly from tabular arrays."""

    scale_pos = positive_weight(y)
    model, backend = xgboost_classifier(config, scale_pos_weight=scale_pos)
    labels = y.astype(int)
    if backend == "xgboost":
        model.fit(x, labels)
    else:
        weights = np.where(labels == 1, scale_pos, 1.0)
        model.fit(x, labels, sample_weight=weights)
    return model, backend


def train_xgboost_trial(config: dict[str, Any], dataset_path: Path):
    """Fit one XGBoost trial on train and evaluate validation only."""

    arrays, _, _ = load_split_bundle(dataset_path, splits=("train", "validation"))
    model, backend = fit_xgboost_arrays(
        config,
        arrays["train"]["X_lag_scalar"],
        arrays["train"]["y_long_fade"],
    )
    logits, prob = predict_sklearn(model, arrays["validation"]["X_lag_scalar"])
    metrics = sample_metrics_from_prob(arrays["validation"]["y_long_fade"], prob)
    return model, backend, metrics, {"validation_logits": logits}


def sample_metrics_from_prob(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float | int]:
    """Compute sample metrics without importing private runner details."""

    from src.tasks.long_fade_detection.evaluation.metrics import (
        sample_classification_metrics,
    )

    return sample_classification_metrics(y_true, prob)


def train_neural_trial(
    config: dict[str, Any],
    dataset_path: Path,
    *,
    model_id: str,
    device_name: str,
):
    """Fit one neural trial on train and evaluate validation only."""

    arrays, metadata, _ = load_split_bundle(dataset_path, splits=("train", "validation"))
    from scripts.experiments.long_fade_detection.run_model import train_neural

    model, train_metadata = train_neural(
        config,
        arrays,
        model_id=model_id,
        device_name=device_name,
        max_epochs_override=None,
    )
    dataset = LongFadeDataset(
        arrays["validation"],
        sequence_key=str(config["dataset"]["sequence_input_key"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 256)),
        shuffle=False,
    )
    logits, prob = predict_neural(model, loader, device=torch.device(device_name))
    metrics = sample_metrics_from_prob(arrays["validation"]["y_long_fade"], prob)
    metrics.update(
        {
            "best_epoch": int(train_metadata["best_epoch"]),
            "pos_weight": float(train_metadata["pos_weight"]),
            "epochs_run": len(train_metadata["history"]),
        }
    )
    return model, "pytorch", metrics, {
        "history": train_metadata["history"],
        "validation_metadata_rows": len(metadata["validation"]),
        "validation_logits": logits,
    }


def trial_display_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact effective parameters for progress logging."""

    model_id = str(config["model_id"])
    return {
        "model": {
            "architecture": model_id,
            "context_length": config["dataset"]["context_length"],
            **dict(config.get("shapelets", {})),
            **dict(config.get("model", {})),
        },
        "optimizer": (
            {
                "name": "AdamW",
                "learning_rate": config["training"].get("learning_rate"),
                "weight_decay": config["training"].get("weight_decay"),
            }
            if model_id != MODEL_ID_XGBOOST
            else {"name": "XGBoost"}
        ),
        "training": {
            key: config["training"].get(key)
            for key in (
                "batch_size",
                "max_epochs",
                "early_stopping_patience",
                "gradient_clip_norm",
                "mixed_precision",
                "seed",
            )
            if key in config["training"]
        },
    }


def run_trial_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one validation-only grid-search trial."""

    device_name = str(job["device"])
    config = resolve_trial_device_config(copy.deepcopy(job["config"]), device_name)
    model_id = str(job["model_id"])
    trial_id = str(job["trial_id"])
    checkpoint_path = Path(job["checkpoint_path"])
    history_path = Path(job["history_path"])
    start = time.perf_counter()
    print(
        format_trial_start(
            index=int(job["index"]),
            total=int(job["total"]),
            run_id=str(job["target_run_id"]),
            trial_id=trial_id,
            device=device_name,
            parameters=trial_display_parameters(config),
        ),
        flush=True,
    )
    try:
        if model_id == MODEL_ID_XGBOOST:
            model, backend, metrics, extra = train_xgboost_trial(
                config,
                Path(job["dataset_path"]),
            )
            if bool(job["save_trial_checkpoints"]):
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, checkpoint_path)
        else:
            prepare_trial_device(device_name)
            model, backend, metrics, extra = train_neural_trial(
                config,
                Path(job["dataset_path"]),
                model_id=model_id,
                device_name=device_name,
            )
            if bool(job["save_trial_checkpoints"]):
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "trial_id": trial_id,
                        "model_id": model_id,
                        "config": config,
                        "state_dict": model.state_dict(),
                        "metrics": metrics,
                    },
                    checkpoint_path,
                )
            if bool(job["save_trial_history"]):
                history_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(extra["history"]).to_csv(history_path, index=False)
        row = {
            "trial_id": trial_id,
            "status": "complete",
            "model_id": model_id,
            **job["parameters"],
            **{f"val_{key}": value for key, value in metrics.items()},
            "backend": backend,
            "duration_seconds": time.perf_counter() - start,
            "checkpoint_path": (
                relative_project_path(checkpoint_path)
                if bool(job["save_trial_checkpoints"])
                else ""
            ),
            "device": device_name,
            "error": "",
        }
        print(
            f"  completed {trial_id} device={device_name} "
            f"{job['selection_metric']}={row[job['selection_metric']]:.6g}",
            flush=True,
        )
        return row
    except Exception as error:
        print(f"  failed {trial_id} device={device_name}: {error}", flush=True)
        return {
            "trial_id": trial_id,
            "status": "failed",
            "model_id": model_id,
            **job["parameters"],
            str(job["selection_metric"]): np.nan,
            "backend": "",
            "duration_seconds": time.perf_counter() - start,
            "checkpoint_path": "",
            "device": device_name,
            "error": repr(error),
        }


def concatenate_arrays(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    *,
    keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Concatenate selected arrays from two splits."""

    return {key: np.concatenate([first[key], second[key]], axis=0) for key in keys}


def train_neural_fixed_epochs(
    config: dict[str, Any],
    arrays: dict[str, dict[str, np.ndarray]],
    *,
    model_id: str,
    device_name: str,
    epochs: int,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train final neural model on train+validation for fixed epochs."""

    training = config["training"]
    set_seed(int(training.get("seed", 42)))
    device = torch.device(device_name)
    full_arrays = concatenate_arrays(
        arrays["train"],
        arrays["validation"],
        keys=("X_relative_to_threshold", "scalar_context_features", "y_long_fade"),
    )
    dataset = LongFadeDataset(
        full_arrays,
        sequence_key=str(config["dataset"]["sequence_input_key"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 256)),
        shuffle=True,
    )
    model_config = dict(config.get("model", {}))
    if model_id == MODEL_ID_SHAPELET_CONV:
        model_config["shapelets"] = config.get("shapelets", {})
    model = build_neural_classifier(
        model_id,
        context_length=int(config["dataset"]["context_length"]),
        scalar_context_dim=int(arrays["train"]["scalar_context_features"].shape[1]),
        config=model_config,
    ).to(device)
    pos_weight = torch.tensor(
        [positive_weight(full_arrays["y_long_fade"])],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0001)),
    )
    gradient_clip = float(training.get("gradient_clip_norm", 1.0))
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for x_sequence, x_scalar, y in loader:
            x_sequence = x_sequence.to(device)
            x_scalar = x_scalar.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_sequence, x_scalar)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite final-training loss: {loss.item()}")
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    return model, {"history": history, "pos_weight": float(pos_weight.item())}


def final_xgboost_predictions(config: dict[str, Any], arrays: dict[str, dict[str, np.ndarray]]):
    """Train selected XGBoost params on train+validation and predict all splits."""

    x_full = np.concatenate(
        [arrays["train"]["X_lag_scalar"], arrays["validation"]["X_lag_scalar"]],
        axis=0,
    )
    y_full = np.concatenate(
        [arrays["train"]["y_long_fade"], arrays["validation"]["y_long_fade"]],
        axis=0,
    )
    model, backend = fit_xgboost_arrays(config, x_full, y_full)
    outputs = {}
    for split in ("train", "validation", "test"):
        outputs[split] = predict_sklearn(model, arrays[split]["X_lag_scalar"])
    return model, backend, outputs


def make_prediction_frames(
    raw_outputs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    arrays: Mapping[str, dict[str, np.ndarray]],
    metadata: Mapping[str, pd.DataFrame],
    model_id: str,
    run_id: str,
) -> dict[str, pd.DataFrame]:
    """Convert split logits/probabilities into standard prediction tables."""

    return {
        split: prediction_frame(
            metadata[split],
            arrays=arrays[split],
            split=split,
            logits=logits,
            probabilities=prob,
            model_id=model_id,
            run_id=run_id,
        )
        for split, (logits, prob) in raw_outputs.items()
    }


def _deserialize_trial_parameter(name: str, value: Any) -> Any:
    """Recover one parameter value after a round-trip through CSV."""

    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "(", "{")):
            try:
                value = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                value = stripped
        else:
            value = stripped
    if name in INTEGER_PARAMETER_NAMES:
        return int(value)
    return value


def _best_parameters(
    best_trial: pd.Series,
    parameter_grid: Mapping[str, Any],
) -> dict[str, Any]:
    """Return best-trial parameters with CSV serialization artifacts removed."""

    return {
        name: _deserialize_trial_parameter(name, best_trial[name])
        for name in parameter_grid
    }


def _table_scalar(value: Any) -> Any:
    """Return a value that pandas can safely store in a one-row summary table."""

    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def save_final_best_run(
    *,
    config: dict[str, Any],
    model_spec: Mapping[str, Any],
    best_trial: pd.Series,
    dataset_path: Path,
    search_id: str,
    device_name: str,
) -> dict[str, Any]:
    """Retrain the selected model on train+validation and save test artifacts."""

    model_id = str(model_spec["model_id"])
    event_config = config["event_definition"]
    selection_id = str(config["selection_id"])
    parameters = _best_parameters(best_trial, model_spec["parameter_grid"])
    final_config = build_trial_config(config, model_spec, parameters)
    final_config = resolve_trial_device_config(final_config, device_name)
    run_id = make_run_id(
        threshold_db=float(event_config["threshold_db"]),
        min_fade_duration_seconds=int(event_config["min_fade_duration_seconds"]),
        model_id=model_id,
        selection_id=selection_id,
    )
    result_dir = run_dir(run_id)
    checkpoint_dir = model_dir(run_id)
    overwrite = bool(config.get("output", {}).get("overwrite", False))
    if result_dir.exists() and overwrite:
        shutil.rmtree(result_dir)
    if checkpoint_dir.exists() and overwrite:
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    arrays, metadata, dataset_metadata = load_split_bundle(dataset_path)

    if model_id == MODEL_ID_XGBOOST:
        model, backend, raw_outputs = final_xgboost_predictions(final_config, arrays)
        joblib.dump(model, checkpoint_dir / "model.joblib")
        extra_metadata = {
            "selected_device": final_config.get("model", {}).get("device", "cpu"),
            "best_epoch": None,
            "pos_weight": positive_weight(
                np.concatenate(
                    [
                        arrays["train"]["y_long_fade"],
                        arrays["validation"]["y_long_fade"],
                    ]
                )
            ),
        }
    else:
        selected_device = select_device() if device_name == "auto" else device_name
        best_epoch = int(best_trial.get("val_best_epoch", best_trial.get("best_epoch", 1)))
        model, train_metadata = train_neural_fixed_epochs(
            final_config,
            arrays,
            model_id=model_id,
            device_name=selected_device,
            epochs=max(best_epoch, 1),
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": final_config,
                "model_id": model_id,
                "run_id": run_id,
            },
            checkpoint_dir / "best_model.pt",
        )
        (result_dir / "tables").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(train_metadata["history"]).to_csv(
            result_dir / "tables" / "final_training_history.csv",
            index=False,
        )
        predictions = neural_predictions(
            model,
            arrays,
            metadata,
            config=final_config,
            model_id=model_id,
            run_id=run_id,
            device_name=selected_device,
        )
        metrics_summary, threshold_table, event_table = evaluate_predictions(predictions)
        backend = "pytorch"
        extra_metadata = {
            "selected_device": selected_device,
            "best_epoch": best_epoch,
            "pos_weight": train_metadata["pos_weight"],
        }
        raw_outputs = None

    if model_id == MODEL_ID_XGBOOST:
        predictions = make_prediction_frames(
            raw_outputs,
            arrays=arrays,
            metadata=metadata,
            model_id=model_id,
            run_id=run_id,
        )
        metrics_summary, threshold_table, event_table = evaluate_predictions(predictions)

    selection_columns = {
        f"selected_{key}": value
        for key, value in best_trial.to_dict().items()
        if str(key).startswith("val_")
    }
    metrics_summary.insert(0, "best_trial_id", best_trial["trial_id"])
    metrics_summary.insert(0, "run_id", run_id)
    metrics_summary.insert(0, "model_id", model_id)
    for key, value in reversed(selection_columns.items()):
        metrics_summary.insert(3, key, value)
    for key, value in parameters.items():
        metrics_summary[key] = _table_scalar(value)
    metrics_summary["final_retrained_on_full_development"] = True
    save_outputs(
        config=final_config,
        run_id=run_id,
        model_id=model_id,
        backend=backend,
        run_path=result_dir,
        checkpoint_path=checkpoint_dir,
        predictions=predictions,
        metrics_summary=metrics_summary,
        threshold_metrics=threshold_table,
        event_metrics=event_table,
        extra_metadata={
            **extra_metadata,
            "grid_search_id": search_id,
            "best_trial_id": best_trial["trial_id"],
            "selection_metric": config["search"]["selection_metric"],
            "selection_mode": config["search"]["selection_mode"],
            "dataset_metadata": dataset_metadata,
            "parameters": parameters,
        },
    )
    return {
        "model_id": model_id,
        "run_id": run_id,
        "best_trial_id": best_trial["trial_id"],
        "best_val_auprc": float(best_trial["val_auprc"]),
        "best_val_auroc": float(best_trial["val_auroc"]),
        "test_auprc": float(metrics_summary["test_auprc"].iloc[0]),
        "test_auroc": float(metrics_summary["test_auroc"].iloc[0]),
        "test_f1_at_0p5": float(metrics_summary["test_f1_at_0p5"].iloc[0]),
        "results_path": relative_project_path(result_dir),
        "model_path": relative_project_path(checkpoint_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_jobs(
    *,
    config: Mapping[str, Any],
    model_specs: Mapping[str, Mapping[str, Any]],
    expanded_trials: list[dict[str, Any]],
    dataset_path: Path,
    search_dir: Path,
) -> list[dict[str, Any]]:
    """Build serializable trial jobs."""

    jobs = []
    total = len(expanded_trials)
    for index, trial in enumerate(expanded_trials, start=1):
        trial_id = make_trial_id(trial["parameters"], index)
        model_id = str(trial["model_id"])
        jobs.append(
            {
                "index": index,
                "total": total,
                "trial_id": trial_id,
                "model_id": model_id,
                "parameters": trial["parameters"],
                "config": trial["config"],
                "dataset_path": str(dataset_path),
                "target_run_id": make_run_id(
                    threshold_db=float(config["event_definition"]["threshold_db"]),
                    min_fade_duration_seconds=int(
                        config["event_definition"]["min_fade_duration_seconds"]
                    ),
                    model_id=model_id,
                    selection_id=str(config["selection_id"]),
                ),
                "selection_metric": config["search"]["selection_metric"],
                "checkpoint_path": str(
                    search_dir
                    / sanitize_id(model_id)
                    / "checkpoints"
                    / f"{trial_id}.pt"
                ),
                "history_path": str(
                    search_dir
                    / sanitize_id(model_id)
                    / "validation_histories"
                    / f"{trial_id}.csv"
                ),
                "save_trial_checkpoints": bool(
                    config["search"].get("save_trial_checkpoints", False)
                ),
                "save_trial_history": bool(
                    config["search"].get("save_trial_history", False)
                ),
                "model_spec": dict(model_specs[model_id]),
            }
        )
    return jobs


def save_model_trial_tables(search_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-model trial tables plus one combined table."""

    combined = pd.DataFrame(rows)
    (search_dir / "tables").mkdir(parents=True, exist_ok=True)
    combined.to_csv(search_dir / "tables" / "trials.csv", index=False)
    for model_id, frame in combined.groupby("model_id", sort=False):
        table_dir = search_dir / sanitize_id(model_id) / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(table_dir / "trials.csv", index=False)


def main() -> None:
    """Run the configured long-fade detection grid search."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if args.max_trials_per_model is not None:
        config["search"]["max_trials_per_model"] = int(args.max_trials_per_model)
    counts = validate_config(config)
    dataset_path = project_path(config["dataset"]["path"])
    search_id = str(config["search_id"])
    selection_id = str(config["selection_id"])
    search_dir = grid_search_dir(selection_id=selection_id, search_id=search_id)
    model_specs = {str(spec["model_id"]): spec for spec in config["models"]}
    expanded = [
        trial
        for spec in config["models"]
        for trial in expand_model_trials(config, spec)
    ]
    total_trials = len(expanded)
    fallback_device = args.device or select_device()
    devices = (
        [fallback_device]
        if args.no_parallel
        else choose_trial_devices(
            config.get("parallel", {}),
            fallback_device=fallback_device,
        )
    )
    print(
        "=== Long-Fade Detection Grid Search ===\n"
        f"Config: {relative_project_path(config_path)}\n"
        f"Search ID: {search_id}\n"
        f"Dataset: {relative_project_path(dataset_path)}\n"
        f"Selection: {selection_id}\n"
        f"Selection metric: {config['search']['selection_metric']} "
        f"({config['search']['selection_mode']})\n"
        f"Trial counts: {counts}\n"
        f"Total trials: {total_trials}\n"
        f"Devices: {devices}\n"
        "Trial phase uses train+validation only; test is evaluated only for "
        "the selected model per family.",
        flush=True,
    )
    if args.dry_run:
        return
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    overwrite = bool(config.get("output", {}).get("overwrite", False)) or args.force
    if args.finalize_existing:
        trial_table = search_dir / "tables" / "trials.csv"
        if not trial_table.is_file():
            raise FileNotFoundError(
                "Cannot finalize existing grid because trials.csv was not found: "
                f"{trial_table}"
            )
        trials = pd.read_csv(trial_table)
        print(
            f"Finalizing from existing trial table: {relative_project_path(trial_table)}",
            flush=True,
        )
    else:
        if search_dir.exists() and overwrite:
            shutil.rmtree(search_dir)
        elif search_dir.exists():
            raise FileExistsError(f"Grid-search folder already exists: {search_dir}")
        ensure_results_subdirs(search_dir, ("tables",))
        save_yaml(search_dir / "config_resolved.yaml", config)
        save_yaml(
            search_dir / "metadata.yaml",
            {
                "search_id": search_id,
                "task_name": "long_fade_detection",
                "selection_id": selection_id,
                "dataset_path": relative_project_path(dataset_path),
                "config_path": relative_project_path(config_path),
                "config_fingerprint": config_fingerprint(config),
                "trial_counts": counts,
                "total_trials": total_trials,
                "selection_metric": config["search"]["selection_metric"],
                "selection_mode": config["search"]["selection_mode"],
                "devices": devices,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        jobs = build_jobs(
            config=config,
            model_specs=model_specs,
            expanded_trials=expanded,
            dataset_path=dataset_path,
            search_dir=search_dir,
        )
        rows = []
        for row in iter_parallel_trial_results(jobs, run_trial_job, devices):
            rows.append(row)
            save_model_trial_tables(search_dir, rows)
        trials = pd.DataFrame(rows)
        save_model_trial_tables(search_dir, rows)

    best_rows = []
    final_device = devices[0] if devices else fallback_device
    for model_id, frame in trials.groupby("model_id", sort=False):
        best_trial = select_best_trial(
            frame,
            metric=str(config["search"]["selection_metric"]),
            mode=str(config["search"]["selection_mode"]),
        )
        best_run = save_final_best_run(
            config=config,
            model_spec=model_specs[str(model_id)],
            best_trial=best_trial,
            dataset_path=dataset_path,
            search_id=search_id,
            device_name=final_device,
        )
        best_rows.append(best_run)
    best_runs = pd.DataFrame(best_rows)
    best_runs.to_csv(search_dir / "tables" / "best_runs.csv", index=False)
    selection_dir = model_selection_dir(selection_id=selection_id, search_id=search_id)
    (selection_dir / "tables").mkdir(parents=True, exist_ok=True)
    best_runs.to_csv(selection_dir / "tables" / "best_runs.csv", index=False)
    save_yaml(
        selection_dir / "metadata.yaml",
        {
            "search_id": search_id,
            "task_name": "long_fade_detection",
            "selection_id": selection_id,
            "selection_metric": config["search"]["selection_metric"],
            "selection_mode": config["search"]["selection_mode"],
            "source_grid_search_path": relative_project_path(search_dir),
            "summary_table": relative_project_path(
                selection_dir / "tables" / "best_runs.csv"
            ),
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    upsert_index_row(
        get_results_index_dir() / "grid_searches.csv",
        {
            "search_id": search_id,
            "model_family": "long_fade_detection",
            "target_run_id": ",".join(best_runs["run_id"].astype(str).tolist()),
            "selection_id": selection_id,
            "selection_metric": config["search"]["selection_metric"],
            "best_trial_id": ",".join(best_runs["best_trial_id"].astype(str).tolist()),
            "results_path": relative_project_path(search_dir),
            "best_model_path": ",".join(best_runs["model_path"].astype(str).tolist()),
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        id_column="search_id",
        columns=GRID_SEARCH_INDEX_COLUMNS,
    )
    print("=== Selected best long-fade runs ===")
    print(best_runs.to_string(index=False), flush=True)
    print(f"Grid results: {relative_project_path(search_dir)}", flush=True)


if __name__ == "__main__":
    main()
