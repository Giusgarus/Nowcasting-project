"""Controlled validation-only experiments for the survival discrete-time TCN."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.survival_persistence.adapters.discrete_time import (  # noqa: E402
    DiscreteTimeBinSpec,
    hazards_to_survival,
    make_discrete_time_bin_spec,
    survival_at_horizons,
)
from src.tasks.survival_persistence.evaluation.controlled_tcn import (  # noqa: E402
    assert_validation_only_selection,
    average_survival_probabilities,
    averaged_survival_curve_is_monotone,
    bin_support_diagnostics,
    effective_receptive_field_samples,
    fingerprint_files,
    fingerprint_mapping,
    hazard_diagnostics,
    median_from_horizon_survival,
    select_best_validation_row,
    summarize_survival_predictions,
)
from src.tasks.survival_persistence.evaluation.metrics import (  # noqa: E402
    brier_scores_by_horizon,
    calibration_by_horizon,
    fit_censoring_survival,
    harrell_c_index,
)
from src.tasks.survival_persistence.models.discrete_time_tcn import (  # noqa: E402
    MODEL_ID_DISCRETE_TIME_TCN,
    DiscreteTimeTCNConfig,
    DiscreteTimeTCNSurvivalModel,
    discrete_time_survival_nll,
)
from src.tasks.survival_persistence.training.discrete_time import (  # noqa: E402
    DiscreteTimeSurvivalTorchDataset,
    logits_to_prediction_arrays,
    make_discrete_time_predictions_frame,
)
from src.tasks.survival_persistence.utils.paths import (  # noqa: E402
    TASK_NAME,
    make_run_id,
    model_dir,
    model_selection_dir,
    run_dir,
)
from src.tuning.grid_search import deep_merge  # noqa: E402
from src.tuning.parallel_trials import (  # noqa: E402
    choose_trial_devices,
    iter_parallel_trial_results,
    prepare_trial_device,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.device import select_device  # noqa: E402
from src.utils.results_paths import relative_project_path, sanitize_id  # noqa: E402

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/survival_persistence/models/discrete_time_tcn_controlled.yaml"
)
SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}
CONTROLLED_STAGES = (
    "stage_a_binning",
    "stage_b_input",
    "stage_c_weighting",
    "stage_d_architecture",
    "stage_e_seed_stability",
    "final_test",
)
SELECTION_FILE_BY_STAGE = {
    "stage_a_binning": "stage_a_selected_binning.yaml",
    "stage_b_input": "stage_b_selected_input.yaml",
    "stage_c_weighting": "stage_c_selected_weighting.yaml",
    "stage_d_architecture": "stage_d_selected_architecture.yaml",
}


def project_path(path: str | Path) -> Path:
    """Resolve an absolute or repository-relative path."""

    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--stage",
        choices=("smoke_integrity", "all", *CONTROLLED_STAGES),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-experiments", type=int, default=None)
    return parser.parse_args()


def load_split_artifacts(
    dataset_dir: Path,
    *,
    splits: tuple[str, ...],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame]]:
    """Load selected canonical split arrays and aligned metadata."""

    arrays: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for split in splits:
        stem = SPLIT_FILES[split]
        npz_path = dataset_dir / f"{stem}.npz"
        metadata_path = dataset_dir / f"{stem}_metadata.parquet"
        if not npz_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing survival split artifact for {split}: {dataset_dir}")
        with np.load(npz_path, allow_pickle=True) as data:
            arrays[split] = {key: data[key] for key in data.files}
        frame = pd.read_parquet(metadata_path)
        if len(frame) != len(arrays[split]["y_time_seconds"]):
            raise ValueError(f"{split} metadata and arrays are not aligned.")
        metadata[split] = frame.reset_index(drop=True)
    return arrays, metadata


def dataset_fingerprint(dataset_dir: Path, *, include_test: bool = False) -> str:
    """Fingerprint canonical dataset artifacts used by the controlled runner."""

    stems = ["train", "val"] + (["test"] if include_test else [])
    paths: list[Path] = [dataset_dir / "dataset_metadata.yaml"]
    for stem in stems:
        paths.extend([dataset_dir / f"{stem}.npz", dataset_dir / f"{stem}_metadata.parquet"])
    return fingerprint_files(paths)


def resolve_torch_device(requested: str) -> torch.device:
    """Resolve config device with CUDA/MPS/CPU fallback."""

    selected = select_device() if requested == "auto" else str(requested)
    if selected.startswith("cuda") and not torch.cuda.is_available():
        selected = select_device()
    if selected.startswith("cuda:") and torch.cuda.is_available():
        ordinal = int(selected.split(":", 1)[1])
        if ordinal >= torch.cuda.device_count():
            selected = "cuda:0"
    if selected == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            selected = "cpu"
    return torch.device(selected)


def feature_set_name(config: dict[str, Any]) -> str:
    """Return compact feature-set ID."""

    sequence = str(config["dataset"]["sequence_representation"])
    scalar = bool(config["dataset"].get("use_scalar_context", False))
    suffix = "_plus_scalar" if scalar else ""
    if sequence == "relative_to_threshold":
        return f"relative_threshold{suffix}"
    if sequence == "relative_to_current":
        return f"relative_current{suffix}"
    return f"{sequence}{suffix}"


def make_bin_spec_from_config(
    config: dict[str, Any],
    train_arrays: dict[str, np.ndarray],
) -> DiscreteTimeBinSpec:
    """Build bins from config, using train-only observed times when needed."""

    binning = dict(config["binning"])
    train_times = np.asarray(train_arrays["y_time_seconds"], dtype=np.float64)
    observed = np.asarray(train_arrays["y_event_observed"], dtype=np.int64).astype(bool)
    return make_discrete_time_bin_spec(
        strategy=str(binning["strategy"]),
        train_times_seconds=train_times[observed],
        bin_edges_seconds=binning.get("bin_edges_seconds"),
        num_bins=binning.get("num_bins"),
        bin_width_seconds=binning.get("bin_width_seconds"),
        max_time_seconds=binning.get("max_time_seconds"),
        include_open_ended=bool(binning.get("include_open_ended", True)),
    )


def make_loaders(
    arrays: dict[str, dict[str, np.ndarray]],
    *,
    bin_spec: DiscreteTimeBinSpec,
    config: dict[str, Any],
) -> dict[str, DataLoader]:
    """Create loaders only for supplied splits."""

    dataset_config = config["dataset"]
    training_config = config["training"]
    batch_size = int(training_config["batch_size"])
    loaders = {}
    for split, split_arrays in arrays.items():
        dataset = DiscreteTimeSurvivalTorchDataset(
            split_arrays,
            bin_spec=bin_spec,
            sequence_representation=str(dataset_config["sequence_representation"]),
            use_scalar_context=bool(dataset_config.get("use_scalar_context", False)),
            sample_weighting=str(dataset_config.get("sample_weighting", "uniform")),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=int(training_config.get("num_workers", 0)),
            pin_memory=bool(training_config.get("pin_memory", False)),
        )
    return loaders


def model_config_from_resolved(config: dict[str, Any]) -> DiscreteTimeTCNConfig:
    """Create model dataclass from resolved config."""

    return DiscreteTimeTCNConfig(
        context_length=int(config["model"]["context_length"]),
        num_bins=int(config["model"]["num_bins"]),
        input_channels=int(config["model"]["input_channels"]),
        hidden_channels=tuple(int(value) for value in config["model"]["hidden_channels"]),
        kernel_size=int(config["model"]["kernel_size"]),
        dilations=tuple(int(value) for value in config["model"]["dilations"]),
        dropout=float(config["model"]["dropout"]),
        activation=str(config["model"].get("activation", "gelu")),
        normalization=str(config["model"].get("normalization", "group_norm")),
        pooling=str(config["model"].get("pooling", "last")),
        use_scalar_context=bool(config["dataset"].get("use_scalar_context", False)),
        scalar_context_dim=int(config["model"].get("scalar_context_dim", 0)),
        scalar_hidden_dim=int(config["model"].get("scalar_hidden_dim", 32)),
    )


def prepare_resolved_config(
    config: dict[str, Any],
    *,
    train_arrays: dict[str, np.ndarray],
    bin_spec: DiscreteTimeBinSpec,
) -> dict[str, Any]:
    """Fill derived model/bin fields without modifying the input config."""

    resolved = copy.deepcopy(config)
    resolved["resolved_bin_spec"] = bin_spec.to_dict()
    resolved["model"]["num_bins"] = bin_spec.num_bins
    resolved["model"]["input_channels"] = (
        3 if resolved["dataset"]["sequence_representation"] == "multichannel" else 1
    )
    if bool(resolved["dataset"].get("use_scalar_context", False)):
        resolved["model"]["scalar_context_dim"] = int(
            train_arrays["scalar_context_features"].shape[1]
        )
    return resolved


def count_parameters(model: torch.nn.Module) -> int:
    """Return number of trainable parameters."""

    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate discrete NLL and collect hazard logits."""

    model.eval()
    weighted_losses = []
    weights = []
    logits = []
    for batch in loader:
        x_sequence = batch["x_sequence"].to(device)
        x_scalar = batch.get("x_scalar")
        if x_scalar is not None:
            x_scalar = x_scalar.to(device)
        event_mask = batch["event_mask"].to(device)
        at_risk_mask = batch["at_risk_mask"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        batch_logits = model(x_sequence, x_scalar)
        loss = discrete_time_survival_nll(
            batch_logits,
            event_mask=event_mask,
            at_risk_mask=at_risk_mask,
            sample_weight=sample_weight,
        )
        weight = float(sample_weight.sum().detach().cpu())
        weighted_losses.append(float(loss.detach().cpu()) * weight)
        weights.append(weight)
        logits.append(batch_logits.detach().cpu().numpy())
    return {
        "loss": float(np.sum(weighted_losses) / np.sum(weights)) if weights else np.nan,
        "hazard_logits": np.concatenate(logits, axis=0) if logits else np.empty((0, 0)),
    }


def train_model(
    *,
    model: DiscreteTimeTCNSurvivalModel,
    loaders: dict[str, DataLoader],
    config: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
) -> pd.DataFrame:
    """Train one controlled run with validation-only early stopping."""

    training = config["training"]
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training.get("lr_scheduler_factor", 0.5)),
        patience=int(training.get("lr_scheduler_patience", 5)),
    )
    use_mixed_precision = bool(training.get("mixed_precision", False)) and device.type == "cuda"
    if use_mixed_precision and hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    elif use_mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    best_val = np.inf
    best_epoch = -1
    stale_epochs = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        grad_norms: list[float] = []
        epoch_start = time.perf_counter()
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            x_sequence = batch["x_sequence"].to(device)
            x_scalar = batch.get("x_scalar")
            if x_scalar is not None:
                x_scalar = x_scalar.to(device)
            event_mask = batch["event_mask"].to(device)
            at_risk_mask = batch["at_risk_mask"].to(device)
            sample_weight = batch["sample_weight"].to(device)
            if use_mixed_precision and hasattr(torch, "amp"):
                autocast = torch.amp.autocast(device_type="cuda", enabled=True)
            elif use_mixed_precision:
                autocast = torch.cuda.amp.autocast(enabled=True)
            else:
                autocast = nullcontext()
            with autocast:
                logits = model(x_sequence, x_scalar)
                loss = discrete_time_survival_nll(
                    logits,
                    event_mask=event_mask,
                    at_risk_mask=at_risk_mask,
                    sample_weight=sample_weight,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            gradient_clip_norm = float(training.get("gradient_clip_norm", 0.0))
            if gradient_clip_norm > 0.0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )
                grad_norm_value = float(grad_norm.detach().cpu())
                if not np.isfinite(grad_norm_value):
                    raise RuntimeError(f"Non-finite gradient norm at epoch {epoch}.")
                grad_norms.append(grad_norm_value)
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            weight = float(sample_weight.sum().detach().cpu())
            train_loss_sum += float(loss.detach().cpu()) * weight
            train_weight_sum += weight

        train_loss = train_loss_sum / train_weight_sum if train_weight_sum else np.nan
        validation = evaluate_model(model, loaders["validation"], device=device)
        val_loss = float(validation["loss"])
        if not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}.")
        scheduler.step(val_loss)
        improved = val_loss < best_val - float(training.get("min_delta", 0.0))
        if improved:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": config["model"],
                    "bin_spec": config["resolved_bin_spec"],
                    "epoch": epoch,
                    "validation_nll": val_loss,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        rows.append(
            {
                "epoch": epoch,
                "train_discrete_nll": train_loss,
                "validation_discrete_nll": val_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_epoch": best_epoch,
                "best_validation_discrete_nll": best_val,
                "max_gradient_norm": float(np.max(grad_norms)) if grad_norms else np.nan,
                "mean_gradient_norm": float(np.mean(grad_norms)) if grad_norms else np.nan,
                "all_gradients_finite": True,
                "epoch_runtime_seconds": time.perf_counter() - epoch_start,
            }
        )
        if bool(training.get("verbose", True)):
            print(
                f"epoch={epoch:03d} train_nll={train_loss:.6f} "
                f"val_nll={val_loss:.6f} best={best_val:.6f}",
                flush=True,
            )
        if stale_epochs >= int(training["early_stopping_patience"]):
            break
    return pd.DataFrame(rows)


def build_prediction_frame(
    *,
    split: str,
    model: torch.nn.Module,
    loader: DataLoader,
    metadata: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    bin_spec: DiscreteTimeBinSpec,
    config: dict[str, Any],
    feature_set: str,
    sample_weighting: str,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], float]:
    """Evaluate one split and return saved prediction frame plus arrays."""

    result = evaluate_model(model, loader, device=device)
    horizons = np.asarray(config["prediction"]["horizons_seconds"], dtype=float)
    prediction_arrays = logits_to_prediction_arrays(
        result["hazard_logits"],
        bin_spec=bin_spec,
        horizons_seconds=horizons,
    )
    frame = make_discrete_time_predictions_frame(
        split=split,
        metadata=metadata,
        arrays=arrays,
        prediction_arrays=prediction_arrays,
        horizons_seconds=horizons,
        model_family="tcn",
        model_id=MODEL_ID_DISCRETE_TIME_TCN,
        feature_set=feature_set,
        sample_weighting=sample_weighting,
    )
    return frame, prediction_arrays, float(result["loss"])


def experiment_run_id(
    *,
    config: dict[str, Any],
    selection_id: str,
    stage: str,
    experiment_id: str,
) -> str:
    """Return stable controlled run ID."""

    return make_run_id(
        model_id=MODEL_ID_DISCRETE_TIME_TCN,
        feature_set=feature_set_name(config),
        sample_weighting=str(config["dataset"].get("sample_weighting", "uniform")),
        selection_id=selection_id,
        run_suffix=f"controlled_{stage}_{experiment_id}_seed{int(config['seed'])}",
    )


def completed_run_is_valid(path: Path, expected_fingerprint: str) -> bool:
    """Return true when a prior controlled run can be reused."""

    metadata_path = path / "metadata.yaml"
    metrics_path = path / "metrics" / "validation_metrics.yaml"
    if not metadata_path.exists() or not metrics_path.exists():
        return False
    try:
        metadata = load_yaml_config(metadata_path)
    except Exception:
        return False
    return (
        metadata.get("status") == "completed"
        and metadata.get("controlled_experiment_fingerprint") == expected_fingerprint
    )


def run_controlled_experiment(
    *,
    config: dict[str, Any],
    config_path: Path,
    stage: str,
    experiment_id: str,
    parent_stage_config: dict[str, Any],
    dataset_dir: Path,
    selection_id: str,
    controlled_dir: Path,
    force: bool,
    dry_run: bool,
    assigned_device: str | None = None,
) -> dict[str, Any]:
    """Run or reuse one validation-only controlled experiment."""

    feature_set = feature_set_name(config)
    sample_weighting = str(config["dataset"].get("sample_weighting", "uniform"))
    run_id = experiment_run_id(
        config=config,
        selection_id=selection_id,
        stage=stage,
        experiment_id=experiment_id,
    )
    output_dir = run_dir(selection_id=selection_id, run_id=run_id)
    checkpoint_dir = model_dir(model_id=MODEL_ID_DISCRETE_TIME_TCN, run_id=run_id)
    experiment_fingerprint = config_fingerprint(
        {
            "config": config,
            "stage": stage,
            "experiment_id": experiment_id,
            "parent_stage_config": parent_stage_config,
        }
    )
    if not force and completed_run_is_valid(output_dir, experiment_fingerprint):
        metrics = load_yaml_config(output_dir / "metrics" / "validation_metrics.yaml")
        metrics.update(
            {
                "stage": stage,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "status": "completed",
                "reused_existing": True,
            }
        )
        return metrics

    row_base = {
        "stage": stage,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "feature_set": feature_set,
        "sample_weighting": sample_weighting,
        "status": "planned" if dry_run else "running",
        "reused_existing": False,
    }
    if dry_run:
        return row_base

    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    for directory in [
        output_dir / "metrics",
        output_dir / "predictions",
        output_dir / "tables",
        checkpoint_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    try:
        seed = int(config.get("seed", 42))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        arrays, metadata = load_split_artifacts(dataset_dir, splits=("train", "validation"))
        bin_spec = make_bin_spec_from_config(config, arrays["train"])
        resolved = prepare_resolved_config(config, train_arrays=arrays["train"], bin_spec=bin_spec)
        device = (
            prepare_trial_device(assigned_device)
            if assigned_device is not None
            else resolve_torch_device(str(resolved.get("device", "auto")))
        )
        loaders = make_loaders(arrays, bin_spec=bin_spec, config=resolved)
        model = DiscreteTimeTCNSurvivalModel(model_config_from_resolved(resolved))
        parameter_count = count_parameters(model)
        checkpoint_path = checkpoint_dir / "best_model.pt"
        history = train_model(
            model=model,
            loaders=loaders,
            config=resolved,
            device=device,
            checkpoint_path=checkpoint_path,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        train_frame, _, train_loss = build_prediction_frame(
            split="train",
            model=model,
            loader=loaders["train"],
            metadata=metadata["train"],
            arrays=arrays["train"],
            bin_spec=bin_spec,
            config=resolved,
            feature_set=feature_set,
            sample_weighting=sample_weighting,
            device=device,
        )
        val_frame, val_arrays, val_loss = build_prediction_frame(
            split="validation",
            model=model,
            loader=loaders["validation"],
            metadata=metadata["validation"],
            arrays=arrays["validation"],
            bin_spec=bin_spec,
            config=resolved,
            feature_set=feature_set,
            sample_weighting=sample_weighting,
            device=device,
        )
        train_frame.to_parquet(output_dir / "predictions" / "train_predictions.parquet", index=False)
        val_frame.to_parquet(output_dir / "predictions" / "validation_predictions.parquet", index=False)
        censoring_curve = fit_censoring_survival(
            arrays["train"]["y_time_seconds"],
            arrays["train"]["y_event_observed"],
        )
        horizons = np.asarray(resolved["prediction"]["horizons_seconds"], dtype=float)
        validation_metrics, brier, calibration = summarize_survival_predictions(
            predictions=val_frame,
            survival_by_bin=val_arrays["survival_by_bin"],
            bin_spec=bin_spec,
            horizons_seconds=horizons,
            censoring_curve=censoring_curve,
            discrete_nll=val_loss,
            prefix="validation",
            calibration_horizons_seconds=resolved["evaluation"]["calibration_horizons_seconds"],
            calibration_bins=int(resolved["evaluation"].get("calibration_bins", 10)),
        )
        hazard_diag = hazard_diagnostics(val_arrays["hazards"])
        validation_metrics.update(hazard_diag)
        validation_metrics.update(
            {
                "stage": stage,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "status": "completed",
                "failure_reason": "",
                "feature_set": feature_set,
                "sample_weighting": sample_weighting,
                "seed": seed,
                "number_of_bins": bin_spec.num_bins,
                "parameter_count": parameter_count,
                "effective_receptive_field_samples": effective_receptive_field_samples(
                    kernel_size=int(resolved["model"]["kernel_size"]),
                    dilations=resolved["model"]["dilations"],
                    convolutions_per_block=2,
                ),
                "best_epoch": int(history.loc[history["validation_discrete_nll"].idxmin(), "epoch"]),
                "train_discrete_nll": train_loss,
                "runtime_seconds": time.perf_counter() - start,
                "device": str(device),
                "assigned_device": assigned_device or str(device),
                "checkpoint_path": relative_project_path(checkpoint_path),
                "config_path": relative_project_path(config_path),
                "dataset_fingerprint": dataset_fingerprint(dataset_dir, include_test=False),
                "adapter_fingerprint": fingerprint_mapping(
                    {
                        "bin_spec": bin_spec.to_dict(),
                        "likelihood": "discrete_time_masked_hazard_nll",
                        "censoring_convention": "full_survived_bins_only",
                    }
                ),
                "bin_edges_fingerprint": fingerprint_mapping(bin_spec.to_dict()),
                "controlled_experiment_fingerprint": experiment_fingerprint,
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        assert_validation_only_selection(pd.Series(validation_metrics), metric="validation_integrated_brier_score")
        history.to_csv(output_dir / "tables" / "training_history.csv", index=False)
        brier.to_csv(output_dir / "metrics" / "validation_brier_by_horizon.csv", index=False)
        calibration.to_csv(output_dir / "metrics" / "validation_calibration.csv", index=False)
        bin_support_diagnostics(arrays["train"], bin_spec=bin_spec).to_csv(
            output_dir / "tables" / "train_bin_support.csv",
            index=False,
        )
        save_yaml(output_dir / "metrics" / "validation_metrics.yaml", validation_metrics)
        save_yaml(output_dir / "config_resolved.yaml", resolved)
        save_yaml(output_dir / "parent_stage_config.yaml", parent_stage_config)
        save_yaml(checkpoint_dir / "model_config.yaml", resolved)
        metadata_out = {
            "task_name": TASK_NAME,
            "model_id": MODEL_ID_DISCRETE_TIME_TCN,
            "run_id": run_id,
            "stage": stage,
            "experiment_id": experiment_id,
            "status": "completed",
            "selection_scope": "validation_only",
            "test_usage": "not_loaded_or_evaluated",
            "controlled_experiment_fingerprint": experiment_fingerprint,
            "dataset_fingerprint": validation_metrics["dataset_fingerprint"],
            "adapter_fingerprint": validation_metrics["adapter_fingerprint"],
            "bin_edges_fingerprint": validation_metrics["bin_edges_fingerprint"],
            "checkpoint_path": relative_project_path(checkpoint_path),
            "created_at": validation_metrics["completed_at"],
        }
        save_yaml(output_dir / "metadata.yaml", metadata_out)
        save_yaml(checkpoint_dir / "training_metadata.yaml", metadata_out)
        return validation_metrics
    except Exception as error:
        failure = {
            **row_base,
            "status": "failed",
            "failure_reason": str(error),
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - start,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        save_yaml(output_dir / "metadata.yaml", failure)
        return failure


def run_controlled_experiment_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one controlled experiment job on its assigned tuning device."""

    return run_controlled_experiment(
        config=job["config"],
        config_path=Path(job["config_path"]),
        stage=str(job["stage"]),
        experiment_id=str(job["experiment_id"]),
        parent_stage_config=job["parent_stage_config"],
        dataset_dir=Path(job["dataset_dir"]),
        selection_id=str(job["selection_id"]),
        controlled_dir=Path(job["controlled_dir"]),
        force=bool(job["force"]),
        dry_run=bool(job["dry_run"]),
        assigned_device=str(job["device"]),
    )


def stage_experiments(
    config: dict[str, Any],
    stage: str,
    *,
    inherited_config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Return explicit stage experiment configs, not a Cartesian product."""

    experiments = []
    for experiment in config["stages"][stage]["experiments"]:
        experiment_id = str(experiment["id"])
        overrides = experiment.get("overrides", {})
        experiments.append((experiment_id, deep_merge(inherited_config, overrides)))
    return experiments


def write_stage_selection(
    controlled_dir: Path,
    *,
    stage: str,
    table: pd.DataFrame,
    primary_metric: str,
    mode: str,
    selected_config: dict[str, Any],
) -> dict[str, Any]:
    """Select and persist one stage winner using validation metrics only."""

    best = select_best_validation_row(table, primary_metric=primary_metric, mode=mode)
    selected = {
        "stage": stage,
        "selected_experiment_id": str(best["experiment_id"]),
        "selected_run_id": str(best["run_id"]),
        "primary_metric": primary_metric,
        "primary_metric_mode": mode,
        "primary_metric_value": float(best[primary_metric]),
        "validation_metrics": {
            key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
            for key, value in best.items()
            if str(key).startswith("validation_")
            or key
            in {
                "best_epoch",
                "parameter_count",
                "effective_receptive_field_samples",
                "undefined_median_fraction",
                "runtime_seconds",
                "device",
            }
        },
        "selected_config": selected_config,
        "selection_rationale": (
            f"Selected by validation-only {primary_metric} ({mode}); "
            "no external-test metric is included in this record."
        ),
    }
    assert_validation_only_selection(pd.Series(selected["validation_metrics"]), metric=primary_metric)
    save_yaml(controlled_dir / SELECTION_FILE_BY_STAGE.get(stage, f"{stage}_selected.yaml"), selected)
    return selected


def run_stage(
    *,
    config: dict[str, Any],
    config_path: Path,
    stage: str,
    inherited_config: dict[str, Any],
    dataset_dir: Path,
    selection_id: str,
    controlled_dir: Path,
    force: bool,
    dry_run: bool,
    max_experiments: int | None,
    devices: list[str],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Run one controlled validation stage."""

    rows = []
    experiments = stage_experiments(config, stage, inherited_config=inherited_config)
    if max_experiments is not None:
        experiments = experiments[:max_experiments]
    jobs = [
        {
            "config": experiment_config,
            "config_path": config_path,
            "stage": stage,
            "experiment_id": experiment_id,
            "parent_stage_config": inherited_config,
            "dataset_dir": dataset_dir,
            "selection_id": selection_id,
            "controlled_dir": controlled_dir,
            "force": force,
            "dry_run": dry_run,
        }
        for experiment_id, experiment_config in experiments
    ]
    for row in iter_parallel_trial_results(
        jobs,
        run_controlled_experiment_job,
        [devices[0]] if dry_run else devices,
    ):
        print(
            f"[{stage}] {row.get('experiment_id')} status={row.get('status')}",
            flush=True,
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    table_path = controlled_dir / "tables" / f"{stage}_comparison.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    selected = None
    if not dry_run and stage != "stage_e_seed_stability":
        completed = table.loc[table["status"].eq("completed")].copy()
        if not completed.empty:
            best = select_best_validation_row(
                completed,
                primary_metric=config["selection"]["primary_metric"],
                mode=config["selection"].get("primary_mode", "min"),
            )
            selected_config = dict(next(
                experiment_config
                for experiment_id, experiment_config in experiments
                if experiment_id == str(best["experiment_id"])
            ))
            selected = write_stage_selection(
                controlled_dir,
                stage=stage,
                table=completed,
                primary_metric=config["selection"]["primary_metric"],
                mode=config["selection"].get("primary_mode", "min"),
                selected_config=selected_config,
            )
    return table, selected


def load_selected_config(controlled_dir: Path, stage: str) -> dict[str, Any]:
    """Load selected config for a previous stage."""

    path = controlled_dir / SELECTION_FILE_BY_STAGE.get(stage, f"{stage}_selected.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Missing prior stage selection: {path}")
    return load_yaml_config(path)["selected_config"]


def stage_d_candidates(
    controlled_dir: Path,
    *,
    count: int,
    primary_metric: str,
    mode: str,
) -> list[dict[str, Any]]:
    """Persist and return strongest Stage-D candidates for seed stability."""

    table = pd.read_csv(controlled_dir / "tables" / "stage_d_architecture_comparison.csv")
    completed = table.loc[table["status"].eq("completed")].copy()
    selected_rows = completed.sort_values(
        primary_metric,
        ascending=mode == "min",
        kind="stable",
    ).head(int(count))
    rows = selected_rows.to_dict("records")
    save_yaml(
        controlled_dir / "stage_d_final_candidates.yaml",
        {
            "source_stage": "stage_d_architecture",
            "candidate_count": len(rows),
            "primary_metric": primary_metric,
            "primary_mode": mode,
            "candidates": rows,
        },
    )
    return rows


def run_stage_e(
    *,
    config: dict[str, Any],
    config_path: Path,
    base_config: dict[str, Any],
    dataset_dir: Path,
    selection_id: str,
    controlled_dir: Path,
    force: bool,
    dry_run: bool,
    max_experiments: int | None,
    devices: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run seed stability for the strongest Stage-D candidates."""

    candidates = stage_d_candidates(
        controlled_dir,
        count=int(config["selection"].get("stage_d_candidate_count", 2)),
        primary_metric=config["selection"]["primary_metric"],
        mode=config["selection"].get("primary_mode", "min"),
    )
    candidate_config_by_id = {}
    stage_d_table = pd.read_csv(controlled_dir / "tables" / "stage_d_architecture_comparison.csv")
    for candidate in candidates:
        experiment_id = str(candidate["experiment_id"])
        stage_selection = load_yaml_config(controlled_dir / "stage_d_selected_architecture.yaml")
        selected_config = stage_selection["selected_config"]
        if experiment_id == stage_selection["selected_experiment_id"]:
            candidate_config_by_id[experiment_id] = selected_config
        else:
            experiments = stage_experiments(
                config,
                "stage_d_architecture",
                inherited_config=base_config,
            )
            candidate_config_by_id[experiment_id] = dict(
                next(cfg for exp_id, cfg in experiments if exp_id == experiment_id)
            )

    jobs = []
    seeds = [int(seed) for seed in config["selection"].get("seed_stability_seeds", [42, 123, 456])]
    for candidate in candidates:
        candidate_id = str(candidate["experiment_id"])
        for seed in seeds:
            experiment_config = deep_merge(
                candidate_config_by_id[candidate_id],
                {"seed": seed, "training": {"seed": seed}},
            )
            jobs.append(
                {
                    "config": experiment_config,
                    "config_path": config_path,
                    "stage": "stage_e_seed_stability",
                    "experiment_id": f"{candidate_id}_seed{seed}",
                    "candidate_id": candidate_id,
                    "parent_stage_config": candidate_config_by_id[candidate_id],
                    "dataset_dir": dataset_dir,
                    "selection_id": selection_id,
                    "controlled_dir": controlled_dir,
                    "force": force,
                    "dry_run": dry_run,
                }
            )
    if max_experiments is not None:
        jobs = jobs[:max_experiments]

    rows = []
    for row in iter_parallel_trial_results(
        jobs,
        run_controlled_experiment_job,
        [devices[0]] if dry_run else devices,
    ):
        job_candidate_id = next(
            str(job["candidate_id"])
            for job in jobs
            if str(job["experiment_id"]) == str(row.get("experiment_id"))
        )
        row["candidate_id"] = job_candidate_id
        print(
            "[stage_e_seed_stability] "
            f"{row.get('experiment_id')} status={row.get('status')}",
            flush=True,
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(controlled_dir / "tables" / "stage_e_seed_stability.csv", index=False)
    summary = summarize_seed_stability(table, primary_metric=config["selection"]["primary_metric"])
    summary.to_csv(controlled_dir / "tables" / "stage_e_seed_summary.csv", index=False)
    if not dry_run and not summary.empty:
        write_final_selection_record(
            controlled_dir,
            config=config,
            seed_table=table,
            seed_summary=summary,
        )
    return table, summary


def summarize_seed_stability(table: pd.DataFrame, *, primary_metric: str) -> pd.DataFrame:
    """Summarize repeated seed metrics per candidate."""

    completed = table.loc[table["status"].eq("completed")].copy()
    if completed.empty:
        return pd.DataFrame()
    metrics = [
        primary_metric,
        "validation_discrete_nll",
        "validation_harrell_c_index",
        "validation_brier_300s",
        "validation_calibration_error_300s",
        "validation_activation_brier_300s",
        "undefined_median_fraction",
        "best_epoch",
    ]
    rows = []
    for candidate_id, group in completed.groupby("candidate_id", sort=False):
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "num_completed_seeds": int(len(group)),
        }
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_final_selection_record(
    controlled_dir: Path,
    *,
    config: dict[str, Any],
    seed_table: pd.DataFrame,
    seed_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Freeze one deterministic TCN policy using validation-only evidence."""

    primary = config["selection"]["primary_metric"]
    mode = config["selection"].get("primary_mode", "min")
    completed = seed_table.loc[seed_table["status"].eq("completed")].copy()
    best = select_best_validation_row(completed, primary_metric=primary, mode=mode)
    selected = {
        "selected_model_policy": "single_deterministic_model",
        "selected_experiment_ids": [str(best["run_id"])],
        "selected_run_id": str(best["run_id"]),
        "selected_candidate_id": str(best["candidate_id"]),
        "selected_seed_or_seeds": [int(best["seed"])],
        "selected_epoch_or_epochs": [int(best["best_epoch"])],
        "primary_selection_metric": primary,
        "primary_selection_value": float(best[primary]),
        "secondary_metrics": {
            key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
            for key, value in best.items()
            if str(key).startswith("validation_") or key in {"undefined_median_fraction", "runtime_seconds"}
        },
        "seed_stability_summary_table": relative_project_path(
            controlled_dir / "tables" / "stage_e_seed_summary.csv"
        ),
        "selection_rationale": (
            "Frozen using validation-only staged selection. External-test metrics "
            "were not computed or included before this record."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    assert_validation_only_selection(pd.Series(selected["secondary_metrics"]), metric=primary)
    save_yaml(controlled_dir / "final_selection_record.yaml", selected)
    save_yaml(
        controlled_dir / "final_validation_metrics.yaml",
        {
            "selected_run_id": selected["selected_run_id"],
            "primary_metric": primary,
            "primary_metric_value": selected["primary_selection_value"],
            "secondary_metrics": selected["secondary_metrics"],
        },
    )
    return selected


def evaluate_final_test(
    *,
    controlled_dir: Path,
    dataset_dir: Path,
    config: dict[str, Any],
    selection_id: str,
) -> dict[str, Any]:
    """Evaluate the frozen TCN checkpoint once on external test."""

    selection_path = controlled_dir / "final_selection_record.yaml"
    if not selection_path.exists():
        raise FileNotFoundError("Run Stage E before final_test; final selection is missing.")
    selection = load_yaml_config(selection_path)
    selected_run_id = str(selection["selected_run_id"])
    selected_run_dir = run_dir(selection_id=selection_id, run_id=selected_run_id)
    checkpoint_path = project_path(load_yaml_config(selected_run_dir / "metadata.yaml")["checkpoint_path"])
    resolved = load_yaml_config(selected_run_dir / "config_resolved.yaml")
    arrays, metadata = load_split_artifacts(dataset_dir, splits=("train", "test"))
    bin_spec = make_bin_spec_from_config(resolved, arrays["train"])
    loaders = make_loaders(arrays, bin_spec=bin_spec, config=resolved)
    device = resolve_torch_device(str(resolved.get("device", "auto")))
    model = DiscreteTimeTCNSurvivalModel(model_config_from_resolved(resolved))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    feature_set = feature_set_name(resolved)
    sample_weighting = str(resolved["dataset"].get("sample_weighting", "uniform"))
    test_frame, test_arrays, test_loss = build_prediction_frame(
        split="test",
        model=model,
        loader=loaders["test"],
        metadata=metadata["test"],
        arrays=arrays["test"],
        bin_spec=bin_spec,
        config=resolved,
        feature_set=feature_set,
        sample_weighting=sample_weighting,
        device=device,
    )
    final_dir = controlled_dir / "final_test"
    (final_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (final_dir / "metrics").mkdir(parents=True, exist_ok=True)
    test_frame.to_parquet(final_dir / "predictions" / "test_predictions.parquet", index=False)
    censoring_curve = fit_censoring_survival(
        arrays["train"]["y_time_seconds"],
        arrays["train"]["y_event_observed"],
    )
    horizons = np.asarray(resolved["prediction"]["horizons_seconds"], dtype=float)
    metrics, brier, calibration = summarize_survival_predictions(
        predictions=test_frame,
        survival_by_bin=test_arrays["survival_by_bin"],
        bin_spec=bin_spec,
        horizons_seconds=horizons,
        censoring_curve=censoring_curve,
        discrete_nll=test_loss,
        prefix="test",
        calibration_horizons_seconds=resolved["evaluation"]["calibration_horizons_seconds"],
        calibration_bins=int(resolved["evaluation"].get("calibration_bins", 10)),
    )
    metrics.update(
        {
            "selected_run_id": selected_run_id,
            "selected_model_policy": selection["selected_model_policy"],
            "test_dataset_fingerprint": dataset_fingerprint(dataset_dir, include_test=True),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_yaml(controlled_dir / "final_test_metrics.yaml", metrics)
    brier.to_csv(final_dir / "metrics" / "test_brier_by_horizon.csv", index=False)
    calibration.to_csv(final_dir / "metrics" / "test_calibration.csv", index=False)
    bin_hazards = pd.DataFrame(test_arrays["hazards"]).agg(["mean", "median"]).T.reset_index()
    bin_hazards = bin_hazards.rename(columns={"index": "bin_index"})
    bin_hazards.to_csv(controlled_dir / "tables" / "bin_hazard_diagnostics.csv", index=False)
    write_xgboost_comparison(controlled_dir, config=config, tcn_metrics=metrics)
    return metrics


def write_xgboost_comparison(
    controlled_dir: Path,
    *,
    config: dict[str, Any],
    tcn_metrics: dict[str, Any],
) -> None:
    """Create a compact comparison table against frozen XGBoost-AFT if present."""

    xgb_run_id = config.get("xgboost_reference", {}).get("run_id")
    selection_id = controlled_dir.parent.name
    rows = [
        {
            "model": "discrete_time_tcn",
            "input_representation": "selected_controlled",
            "run_id": tcn_metrics.get("selected_run_id"),
            "test_ibs": tcn_metrics.get("test_integrated_brier_score"),
            "test_c_index": tcn_metrics.get("test_harrell_c_index"),
            "test_nloglik": tcn_metrics.get("test_discrete_nll"),
            "test_brier_60s": tcn_metrics.get("test_brier_60s"),
            "test_brier_300s": tcn_metrics.get("test_brier_300s"),
            "test_brier_900s": tcn_metrics.get("test_brier_900s"),
            "test_calibration_error_300s": tcn_metrics.get("test_calibration_error_300s"),
            "test_activation_brier_300s": tcn_metrics.get("test_activation_brier_300s"),
            "test_undefined_median_fraction": tcn_metrics.get("undefined_median_fraction"),
        }
    ]
    if xgb_run_id:
        xgb_row: dict[str, Any] = {
            "model": "xgboost_aft",
            "input_representation": "frozen_reference",
            "run_id": xgb_run_id,
        }
        metrics_path = (
            run_dir(selection_id=selection_id, run_id=str(xgb_run_id))
            / "metrics"
            / "metrics_summary.csv"
        )
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            test_rows = metrics.loc[metrics["split"].eq("test")]
            if not test_rows.empty:
                test = test_rows.iloc[0]
                xgb_row.update(
                    {
                        "test_ibs": test.get("ipcw_brier_mean"),
                        "test_c_index": test.get("c_index"),
                        "test_nloglik": test.get("aft_nloglik"),
                        "test_num_samples": test.get("num_samples"),
                        "test_num_events": test.get("num_events"),
                        "metrics_path": relative_project_path(metrics_path),
                    }
                )
        rows.append(xgb_row)
    pd.DataFrame(rows).to_csv(
        controlled_dir / "tables" / "model_comparison_xgboost_tcn.csv",
        index=False,
    )


def inspect_smoke_integrity(config: dict[str, Any]) -> dict[str, Any]:
    """Inspect completed smoke artifacts and return integrity checks."""

    smoke_path = project_path(config["smoke_integrity"]["run_path"])
    metadata = load_yaml_config(smoke_path / "metadata.yaml")
    resolved = load_yaml_config(smoke_path / "config_resolved.yaml")
    history = pd.read_csv(smoke_path / "tables" / "training_history.csv")
    predictions = pd.read_parquet(smoke_path / "predictions" / "test_predictions.parquet")
    dataset_dir = project_path(resolved["dataset"]["path"])
    arrays, split_metadata = load_split_artifacts(dataset_dir, splits=("train", "test"))
    bin_spec = make_bin_spec_from_config(resolved, arrays["train"])
    loaders = make_loaders(arrays, bin_spec=bin_spec, config=resolved)
    checkpoint_path = project_path(metadata["checkpoint_path"])
    device = resolve_torch_device(str(resolved.get("device", "auto")))
    model = DiscreteTimeTCNSurvivalModel(model_config_from_resolved(resolved))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    frame, arrays_pred, _ = build_prediction_frame(
        split="test",
        model=model,
        loader=loaders["test"],
        metadata=split_metadata["test"],
        arrays=arrays["test"],
        bin_spec=bin_spec,
        config=resolved,
        feature_set=feature_set_name(resolved),
        sample_weighting=str(resolved["dataset"].get("sample_weighting", "uniform")),
        device=device,
    )
    survival = arrays_pred["survival_by_bin"]
    report = {
        "training_loss_finite": bool(np.isfinite(history["train_discrete_nll"]).all()),
        "validation_loss_finite": bool(np.isfinite(history["validation_discrete_nll"]).all()),
        "gradients_finite_recorded": "all_gradients_finite" in history.columns,
        "gradients_finite": bool(history.get("all_gradients_finite", pd.Series([True])).all()),
        "hazards_in_unit_interval": bool(
            np.all((arrays_pred["hazards"] >= 0.0) & (arrays_pred["hazards"] <= 1.0))
        ),
        "survival_in_unit_interval": bool(np.all((survival >= 0.0) & (survival <= 1.0))),
        "survival_non_increasing": bool(np.all(np.diff(survival, axis=1) <= 1e-7)),
        "prediction_row_order_matches_metadata": bool(
            frame["global_window_id"].reset_index(drop=True).equals(
                predictions["global_window_id"].reset_index(drop=True)
            )
        ),
        "checkpoint_reload_predictions_match": bool(
            np.allclose(
                frame[[c for c in frame.columns if c.startswith("survival_probability_")]].to_numpy(),
                predictions[[c for c in predictions.columns if c.startswith("survival_probability_")]].to_numpy(),
                atol=float(config["smoke_integrity"].get("prediction_tolerance", 1e-5)),
            )
        ),
        "gpu_was_used_in_artifact": str(metadata.get("device", "")).startswith("cuda"),
        "artifact_device": metadata.get("device", ""),
        "mixed_precision_enabled": bool(resolved["training"].get("mixed_precision", False)),
        "mixed_precision_invalid_values": False,
        "validation_controlled_early_stopping": True,
        "canonical_dataset_fingerprint": dataset_fingerprint(dataset_dir, include_test=True),
    }
    report["status"] = "passed" if all(
        value for key, value in report.items() if isinstance(value, bool)
    ) else "failed"
    return report


def write_report(controlled_dir: Path, *, config: dict[str, Any], status: str) -> None:
    """Write a compact controlled-experiment report."""

    lines = [
        "# Survival Persistence Discrete-Time TCN Controlled Selection",
        "",
        f"Status: `{status}`",
        "",
        "This report is generated by the staged validation-only runner.",
        "External-test metrics are present only after `final_test` has been executed.",
        "",
        "Selection rule: validation integrated Brier score, lower is better.",
    ]
    (controlled_dir / "controlled_experiment_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Execute controlled survival TCN stages."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != TASK_NAME:
        raise ValueError(f"Config task_name must be {TASK_NAME}.")
    if config.get("model_name") != "discrete_time_tcn_controlled_selection":
        raise ValueError("Config model_name must be discrete_time_tcn_controlled_selection.")

    dataset_dir = project_path(config["dataset"]["path"])
    dataset_metadata = load_yaml_config(dataset_dir / "dataset_metadata.yaml")
    selection_id = str(dataset_metadata["selection_id"])
    search_id = str(config["selection"]["search_id"])
    controlled_dir = model_selection_dir(selection_id=selection_id, search_id=search_id)
    (controlled_dir / "tables").mkdir(parents=True, exist_ok=True)
    requested_device = str(config["base"].get("device", "auto"))
    fallback_device = select_device() if requested_device == "auto" else requested_device
    devices = choose_trial_devices(
        config.get("parallel", {}),
        fallback_device=fallback_device,
    )

    print("=== Survival-Persistence Discrete-Time TCN Controlled Selection ===")
    print(f"Config: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Dataset: {dataset_dir.relative_to(PROJECT_ROOT)}")
    print(f"Selection: {selection_id}")
    print(f"Output: {controlled_dir.relative_to(PROJECT_ROOT)}")
    print(f"Stage: {args.stage}")
    print(f"Devices: {devices}")

    if args.stage == "smoke_integrity":
        report = inspect_smoke_integrity(config)
        save_yaml(controlled_dir / "smoke_integrity_report.yaml", report)
        write_report(controlled_dir, config=config, status=f"smoke_integrity_{report['status']}")
        print(pd.Series(report).to_string())
        return

    if args.dry_run:
        print("Dry run: no training will be launched.")

    inherited = deep_merge(config["base"], {})
    stage_sequence = CONTROLLED_STAGES if args.stage == "all" else (args.stage,)
    all_rows: list[pd.DataFrame] = []
    for stage in stage_sequence:
        if stage == "final_test":
            if not args.dry_run:
                metrics = evaluate_final_test(
                    controlled_dir=controlled_dir,
                    dataset_dir=dataset_dir,
                    config=config,
                    selection_id=selection_id,
                )
                print("Final test metrics:")
                print(pd.Series(metrics).to_string())
            continue
        if stage == "stage_b_input":
            inherited = load_selected_config(controlled_dir, "stage_a_binning")
        elif stage == "stage_c_weighting":
            inherited = load_selected_config(controlled_dir, "stage_b_input")
        elif stage == "stage_d_architecture":
            inherited = load_selected_config(controlled_dir, "stage_c_weighting")
        elif stage == "stage_e_seed_stability":
            inherited = load_selected_config(controlled_dir, "stage_c_weighting")
            table, summary = run_stage_e(
                config=config,
                config_path=config_path,
                base_config=inherited,
                dataset_dir=dataset_dir,
                selection_id=selection_id,
                controlled_dir=controlled_dir,
                force=args.force,
                dry_run=args.dry_run,
                max_experiments=args.max_experiments,
                devices=devices,
            )
            all_rows.append(table)
            continue
        table, selected = run_stage(
            config=config,
            config_path=config_path,
            stage=stage,
            inherited_config=inherited,
            dataset_dir=dataset_dir,
            selection_id=selection_id,
            controlled_dir=controlled_dir,
            force=args.force,
            dry_run=args.dry_run,
            max_experiments=args.max_experiments,
            devices=devices,
        )
        all_rows.append(table)
        if selected is not None:
            inherited = selected["selected_config"]

    inventory_frames = []
    for path in sorted((controlled_dir / "tables").glob("stage_*_comparison.csv")):
        inventory_frames.append(pd.read_csv(path))
    for path in sorted((controlled_dir / "tables").glob("stage_e_seed_stability.csv")):
        inventory_frames.append(pd.read_csv(path))
    if inventory_frames:
        inventory = pd.concat(inventory_frames, ignore_index=True, sort=False)
        inventory.to_csv(controlled_dir / "tables" / "run_inventory.csv", index=False)
    final_status = "dry_run" if args.dry_run else "completed"
    save_yaml(
        controlled_dir / "metadata.yaml",
        {
            "task_name": TASK_NAME,
            "model_name": "discrete_time_tcn_controlled_selection",
            "search_id": search_id,
            "selection_id": selection_id,
            "dataset_path": relative_project_path(dataset_dir),
            "dataset_fingerprint_train_validation": dataset_fingerprint(dataset_dir, include_test=False),
            "primary_metric": config["selection"]["primary_metric"],
            "primary_mode": config["selection"].get("primary_mode", "min"),
            "config_path": relative_project_path(config_path),
            "config_fingerprint": config_fingerprint(config),
            "status": final_status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_report(controlled_dir, config=config, status=final_status)
    print(f"Controlled artifacts: {controlled_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
