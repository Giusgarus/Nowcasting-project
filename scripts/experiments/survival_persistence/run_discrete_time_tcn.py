"""Train and evaluate the survival-persistence discrete-time TCN baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    make_discrete_time_bin_spec,
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
    evaluate_discrete_time_model,
    logits_to_prediction_arrays,
    make_discrete_time_predictions_frame,
)
from src.tasks.survival_persistence.utils.paths import (  # noqa: E402
    RUN_INDEX_COLUMNS,
    TASK_NAME,
    make_run_id,
    model_dir,
    run_dir,
    run_index_path,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.device import select_device  # noqa: E402
from src.utils.results_paths import relative_project_path, upsert_index_row  # noqa: E402

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/survival_persistence/models/discrete_time_tcn_smoke.yaml"
)
SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}


def project_path(path: str | Path) -> Path:
    """Resolve an absolute or repository-relative path."""

    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_split_artifacts(dataset_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame]]:
    """Load canonical split arrays and aligned metadata."""

    arrays: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for split, stem in SPLIT_FILES.items():
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


def directory_has_files(path: Path) -> bool:
    """Return true when a directory already contains persisted files."""

    return path.exists() and any(child.is_file() for child in path.rglob("*"))


def resolve_torch_device(requested: str) -> torch.device:
    """Resolve config device with CUDA/MPS/CPU fallback."""

    requested = str(requested)
    selected = select_device() if requested == "auto" else requested
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
    """Return a compact feature-set ID for run naming."""

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
    """Build discrete-time bins using train-only labels when needed."""

    binning = dict(config["binning"])
    train_times = np.asarray(train_arrays["y_time_seconds"], dtype=np.float64)
    observed = np.asarray(train_arrays["y_event_observed"], dtype=np.int64).astype(bool)
    train_observed_times = train_times[observed]
    return make_discrete_time_bin_spec(
        strategy=str(binning["strategy"]),
        train_times_seconds=train_observed_times,
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
) -> tuple[dict[str, DiscreteTimeSurvivalTorchDataset], dict[str, DataLoader]]:
    """Create datasets and loaders for train/validation/test splits."""

    dataset_config = config["dataset"]
    training_config = config["training"]
    datasets = {
        split: DiscreteTimeSurvivalTorchDataset(
            split_arrays,
            bin_spec=bin_spec,
            sequence_representation=str(dataset_config["sequence_representation"]),
            use_scalar_context=bool(dataset_config.get("use_scalar_context", False)),
            sample_weighting=str(dataset_config.get("sample_weighting", "uniform")),
        )
        for split, split_arrays in arrays.items()
    }
    batch_size = int(training_config["batch_size"])
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(training_config.get("num_workers", 0)),
            pin_memory=bool(training_config.get("pin_memory", False)),
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(training_config.get("num_workers", 0)),
            pin_memory=bool(training_config.get("pin_memory", False)),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(training_config.get("num_workers", 0)),
            pin_memory=bool(training_config.get("pin_memory", False)),
        ),
    }
    return datasets, loaders


def train_model(
    *,
    model: DiscreteTimeTCNSurvivalModel,
    loaders: dict[str, DataLoader],
    config: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
) -> pd.DataFrame:
    """Train with validation-only early stopping and persist the best checkpoint."""

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
    patience = int(training["early_stopping_patience"])
    max_epochs = int(training["max_epochs"])
    gradient_clip_norm = float(training.get("gradient_clip_norm", 0.0))
    best_val = np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
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
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            if gradient_clip_norm > 0.0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            batch_weight = float(sample_weight.sum().detach().cpu())
            train_loss_sum += float(loss.detach().cpu()) * batch_weight
            train_weight_sum += batch_weight
        train_loss = train_loss_sum / train_weight_sum if train_weight_sum else np.nan
        val_result = evaluate_discrete_time_model(
            model,
            loaders["validation"],
            device=device,
            loss_fn=discrete_time_survival_nll,
        )
        val_loss = float(val_result["loss"])
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        improved = val_loss < best_val - float(training.get("min_delta", 0.0))
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
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
            epochs_without_improvement += 1
        history_rows.append(
            {
                "epoch": epoch,
                "train_discrete_nll": train_loss,
                "validation_discrete_nll": val_loss,
                "learning_rate": current_lr,
                "best_epoch": best_epoch,
                "best_validation_discrete_nll": best_val,
            }
        )
        print(
            f"epoch={epoch:03d} train_nll={train_loss:.6f} "
            f"val_nll={val_loss:.6f} best={best_val:.6f}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            break
    return pd.DataFrame(history_rows)


def metrics_for_predictions(
    *,
    method_name: str,
    split: str,
    predictions: pd.DataFrame,
    horizons_seconds: np.ndarray,
    censoring_curve,
    discrete_nll: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute survival metrics from saved discrete-time predictions."""

    times = predictions["y_time_seconds"].to_numpy(dtype=float)
    observed = predictions["y_event_observed"].to_numpy(dtype=int)
    median = predictions["predicted_median_remaining_seconds"].to_numpy(dtype=float)
    cindex = harrell_c_index(times, observed, median, higher_prediction_longer=True)
    survival = predictions[
        [f"survival_probability_{int(horizon)}s" for horizon in horizons_seconds]
    ].to_numpy(dtype=float)
    brier = brier_scores_by_horizon(
        times,
        observed,
        survival,
        horizons_seconds,
        censoring_curve=censoring_curve,
    )
    row: dict[str, Any] = {
        "method": method_name,
        "split": split,
        "num_samples": int(len(predictions)),
        "num_events": int(predictions["global_event_id"].nunique()),
        "discrete_time_nll": float(discrete_nll),
        "c_index": cindex["c_index"],
        "num_comparable_pairs": cindex["num_comparable_pairs"],
        "ipcw_brier_mean": float(brier["ipcw_brier"].mean()),
    }
    brier.insert(0, "method", method_name)
    brier.insert(1, "split", split)
    return row, brier


def calibration_tables(
    *,
    predictions: pd.DataFrame,
    horizons_seconds: list[float],
    n_bins: int,
    censoring_curve,
) -> pd.DataFrame:
    """Create calibration tables for configured horizons."""

    rows = []
    for horizon in horizons_seconds:
        column = f"survival_probability_{int(horizon)}s"
        if column not in predictions.columns:
            continue
        rows.append(
            calibration_by_horizon(
                predictions["y_time_seconds"].to_numpy(dtype=float),
                predictions["y_event_observed"].to_numpy(dtype=int),
                predictions[column].to_numpy(dtype=float),
                horizon_seconds=float(horizon),
                n_bins=n_bins,
                censoring_curve=censoring_curve,
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_plots(
    *,
    figures_dir: Path,
    training_history: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_brier: pd.DataFrame,
) -> None:
    """Save core matplotlib diagnostics."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    if not training_history.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(training_history["epoch"], training_history["train_discrete_nll"], label="train")
        ax.plot(
            training_history["epoch"],
            training_history["validation_discrete_nll"],
            label="validation",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Discrete-time NLL")
        ax.set_title("Discrete-time TCN training history")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "training_discrete_nll.png", dpi=150)
        plt.close(fig)

    if not test_predictions.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(test_predictions["predicted_median_remaining_seconds"], bins=40)
        ax.set_xlabel("Predicted median remaining seconds")
        ax.set_ylabel("Count")
        ax.set_title("External-test predicted median distribution")
        fig.tight_layout()
        fig.savefig(figures_dir / "test_predicted_median_distribution.png", dpi=150)
        plt.close(fig)

        p300_column = "survival_probability_300s"
        if p300_column in test_predictions.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(test_predictions[p300_column], bins=40)
            ax.set_xlabel("Predicted P(R > 300 s)")
            ax.set_ylabel("Count")
            ax.set_title("External-test P(R > 300 s) distribution")
            fig.tight_layout()
            fig.savefig(figures_dir / "test_survival_probability_300s_distribution.png", dpi=150)
            plt.close(fig)

    if not test_brier.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(test_brier["horizon_seconds"], test_brier["ipcw_brier"], marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("Horizon seconds")
        ax.set_ylabel("IPCW Brier score")
        ax.set_title("External-test Brier score by horizon")
        fig.tight_layout()
        fig.savefig(figures_dir / "test_brier_by_horizon.png", dpi=150)
        plt.close(fig)


def main() -> None:
    """Run the configured discrete-time TCN baseline."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != TASK_NAME:
        raise ValueError(f"Config task_name must be {TASK_NAME}.")
    if config.get("model_name") != MODEL_ID_DISCRETE_TIME_TCN:
        raise ValueError(f"Config model_name must be {MODEL_ID_DISCRETE_TIME_TCN}.")

    dataset_dir = project_path(config["dataset"]["path"])
    dataset_metadata = load_yaml_config(dataset_dir / "dataset_metadata.yaml")
    selection_id = str(dataset_metadata["selection_id"])
    model_id = MODEL_ID_DISCRETE_TIME_TCN
    features = feature_set_name(config)
    sample_weighting = str(config["dataset"].get("sample_weighting", "uniform"))
    run_id = make_run_id(
        model_id=model_id,
        feature_set=features,
        sample_weighting=sample_weighting,
        selection_id=selection_id,
        run_suffix=config.get("run_suffix"),
    )
    output_dir = run_dir(selection_id=selection_id, run_id=run_id)
    checkpoint_dir = model_dir(model_id=model_id, run_id=run_id)
    overwrite = bool(config.get("output", {}).get("overwrite", False))
    if directory_has_files(output_dir) and not overwrite:
        raise FileExistsError(f"Run output already exists: {output_dir}")
    if directory_has_files(checkpoint_dir) and not overwrite:
        raise FileExistsError(f"Model output already exists: {checkpoint_dir}")

    print("=== Survival-Persistence Discrete-Time TCN ===")
    print(f"Config: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Dataset: {dataset_dir.relative_to(PROJECT_ROOT)}")
    print(f"Run ID: {run_id}")
    print(f"Feature set: {features}")
    print(f"Sample weighting: {sample_weighting}")
    print("Uses validation only for early stopping; test is evaluated once.\n")
    if args.dry_run:
        return

    for directory in [
        output_dir / "metrics",
        output_dir / "predictions",
        output_dir / "figures",
        output_dir / "tables",
        checkpoint_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    arrays, split_metadata = load_split_artifacts(dataset_dir)
    bin_spec = make_bin_spec_from_config(config, arrays["train"])
    config["resolved_bin_spec"] = bin_spec.to_dict()
    config["model"]["num_bins"] = bin_spec.num_bins
    config["model"]["input_channels"] = (
        3 if config["dataset"]["sequence_representation"] == "multichannel" else 1
    )
    if bool(config["dataset"].get("use_scalar_context", False)):
        config["model"]["scalar_context_dim"] = int(
            arrays["train"]["scalar_context_features"].shape[1]
        )
    device = resolve_torch_device(str(config.get("device", "auto")))
    print(f"Device: {device}")
    print(f"Discrete bins: {bin_spec.num_bins}")

    _, loaders = make_loaders(arrays, bin_spec=bin_spec, config=config)
    model_config = DiscreteTimeTCNConfig(
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
    model = DiscreteTimeTCNSurvivalModel(model_config)
    checkpoint_path = checkpoint_dir / "best_model.pt"
    history = train_model(
        model=model,
        loaders=loaders,
        config=config,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    history.to_csv(output_dir / "tables" / "training_history.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    horizons = np.asarray(config["prediction"]["horizons_seconds"], dtype=float)
    prediction_frames: dict[str, pd.DataFrame] = {}
    eval_results: dict[str, dict[str, Any]] = {}
    for split, loader in loaders.items():
        result = evaluate_discrete_time_model(
            model,
            loader,
            device=device,
            loss_fn=discrete_time_survival_nll,
        )
        eval_results[split] = result
        prediction_arrays = logits_to_prediction_arrays(
            result["hazard_logits"],
            bin_spec=bin_spec,
            horizons_seconds=horizons,
        )
        frame = make_discrete_time_predictions_frame(
            split=split,
            metadata=split_metadata[split],
            arrays=arrays[split],
            prediction_arrays=prediction_arrays,
            horizons_seconds=horizons,
            model_family="tcn",
            model_id=model_id,
            feature_set=features,
            sample_weighting=sample_weighting,
        )
        prediction_frames[split] = frame
        frame.to_parquet(output_dir / "predictions" / f"{split}_predictions.parquet", index=False)

    train_times = arrays["train"]["y_time_seconds"]
    train_observed = arrays["train"]["y_event_observed"]
    censoring_curve = fit_censoring_survival(train_times, train_observed)
    metrics_rows: list[dict[str, Any]] = []
    brier_tables: list[pd.DataFrame] = []
    for split, frame in prediction_frames.items():
        metric_row, brier = metrics_for_predictions(
            method_name=model_id,
            split=split,
            predictions=frame,
            horizons_seconds=horizons,
            censoring_curve=censoring_curve,
            discrete_nll=float(eval_results[split]["loss"]),
        )
        metrics_rows.append(metric_row)
        brier_tables.append(brier)
    metrics_summary = pd.DataFrame(metrics_rows)
    brier_by_horizon = pd.concat(brier_tables, ignore_index=True)
    metrics_summary.to_csv(output_dir / "metrics" / "metrics_summary.csv", index=False)
    brier_by_horizon.to_csv(output_dir / "metrics" / "brier_by_horizon.csv", index=False)

    calibration = calibration_tables(
        predictions=prediction_frames["test"],
        horizons_seconds=list(config["evaluation"]["calibration_horizons_seconds"]),
        n_bins=int(config["evaluation"].get("calibration_bins", 10)),
        censoring_curve=censoring_curve,
    )
    calibration.to_csv(output_dir / "metrics" / "test_calibration.csv", index=False)

    (output_dir / "tables" / "bin_spec.json").write_text(
        json.dumps(bin_spec.to_dict(), indent=2),
        encoding="utf-8",
    )
    save_plots(
        figures_dir=output_dir / "figures",
        training_history=history,
        test_predictions=prediction_frames["test"],
        test_brier=brier_by_horizon.loc[brier_by_horizon["split"].eq("test")],
    )

    save_yaml(output_dir / "config_resolved.yaml", config)
    save_yaml(checkpoint_dir / "model_config.yaml", config)
    created_at = datetime.now(timezone.utc).isoformat()
    best_row = history.loc[history["validation_discrete_nll"].idxmin()]
    val_row = metrics_summary.loc[metrics_summary["split"].eq("validation")].iloc[0]
    test_row = metrics_summary.loc[metrics_summary["split"].eq("test")].iloc[0]
    metadata = {
        "task_name": TASK_NAME,
        "model_family": "tcn",
        "model_id": model_id,
        "run_id": run_id,
        "config_path": relative_project_path(config_path),
        "config_fingerprint": config_fingerprint(config),
        "dataset_path": relative_project_path(dataset_dir),
        "dataset_metadata": relative_project_path(dataset_dir / "dataset_metadata.yaml"),
        "selection_id": selection_id,
        "feature_set": features,
        "sample_weighting": sample_weighting,
        "device": str(device),
        "discrete_time_semantics": (
            "The model outputs one hazard logit per bin. Survival probabilities "
            "are computed as cumulative products of one minus hazard."
        ),
        "bin_spec": bin_spec.to_dict(),
        "best_epoch": int(best_row["epoch"]),
        "best_validation_discrete_nll": float(best_row["validation_discrete_nll"]),
        "checkpoint_path": relative_project_path(checkpoint_path),
        "output_files": {
            "metrics_summary": relative_project_path(output_dir / "metrics" / "metrics_summary.csv"),
            "brier_by_horizon": relative_project_path(output_dir / "metrics" / "brier_by_horizon.csv"),
            "test_predictions": relative_project_path(output_dir / "predictions" / "test_predictions.parquet"),
            "training_history": relative_project_path(output_dir / "tables" / "training_history.csv"),
        },
        "ipcw_note": "Censoring curve is fitted on the train split only.",
        "created_at": created_at,
    }
    save_yaml(output_dir / "metadata.yaml", metadata)
    save_yaml(checkpoint_dir / "training_metadata.yaml", metadata)

    upsert_index_row(
        run_index_path(),
        {
            "run_id": run_id,
            "task_name": TASK_NAME,
            "model_family": "tcn",
            "model_id": model_id,
            "feature_set": features,
            "sample_weighting": sample_weighting,
            "selection_id": selection_id,
            "dataset_path": relative_project_path(dataset_dir),
            "best_iteration": int(best_row["epoch"]),
            "best_val_aft_nloglik": np.nan,
            "best_val_discrete_nll": float(best_row["validation_discrete_nll"]),
            "val_c_index": val_row["c_index"],
            "val_ipcw_brier_mean": val_row["ipcw_brier_mean"],
            "test_c_index": test_row["c_index"],
            "test_ipcw_brier_mean": test_row["ipcw_brier_mean"],
            "created_at": created_at,
            "status": "completed",
        },
        id_column="run_id",
        columns=RUN_INDEX_COLUMNS,
    )

    print("=== Completed survival discrete-time TCN ===")
    print(f"Best epoch: {int(best_row['epoch'])}")
    print(f"Validation discrete NLL: {best_row['validation_discrete_nll']:.6f}")
    print(f"Test C-index: {test_row['c_index']:.6f}")
    print(f"Test IPCW Brier mean: {test_row['ipcw_brier_mean']:.6f}")
    print(f"Predictions: {(output_dir / 'predictions').relative_to(PROJECT_ROOT)}")
    print(f"Metrics: {(output_dir / 'metrics').relative_to(PROJECT_ROOT)}")
    print(f"Model: {checkpoint_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
