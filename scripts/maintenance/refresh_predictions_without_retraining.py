"""Refresh saved run predictions on rebuilt datasets without retraining models.

This utility is intentionally operational: it reloads existing checkpoints,
recomputes split predictions on the current canonical datasets, and overwrites
prediction/metric tables in the matching run folders. It does not modify model
weights, fit scalers, tune thresholds, or train any model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/nowcasting_matplotlib")))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp/nowcasting_matplotlib")))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.autoregressive.run_autoregressive_gru import (  # noqa: E402
    grouped_metrics,
    load_split_arrays as load_autoregressive_split_arrays,
    resolve_selection_folder,
)
from scripts.experiments.autoregressive.run_patchtst_grid_search import (  # noqa: E402
    build_patchtst_model,
    predictions_in_raw_scale as autoregressive_predictions_in_raw_scale,
)
from scripts.experiments.current_level_persistence.train_learnable_shapelets import (  # noqa: E402
    build_shapelet_config,
    evaluate_split as evaluate_current_level_split,
    load_npz as load_current_level_npz,
    make_loader as make_current_level_loader,
    make_loss as make_current_level_loss,
    prediction_frame as current_level_prediction_frame,
)
from scripts.experiments.long_fade_detection.run_model import (  # noqa: E402
    LongFadeDataset,
    evaluate_predictions as evaluate_long_fade_predictions,
    load_split_data as load_long_fade_split_data,
    neural_predictions as long_fade_neural_predictions,
    predict_sklearn as predict_long_fade_sklearn,
    prediction_frame as long_fade_prediction_frame,
    save_outputs as save_long_fade_outputs,
)
from scripts.experiments.survival_persistence.run_discrete_time_tcn import (  # noqa: E402
    load_split_artifacts as load_survival_split_artifacts,
    metrics_for_predictions as discrete_tcn_metrics_for_predictions,
    resolve_torch_device,
)
from scripts.experiments.survival_persistence.run_xgboost_aft import (  # noqa: E402
    calibration_tables as xgboost_aft_calibration_tables,
    load_split_artifacts as load_survival_aft_split_artifacts,
    make_predictions_frame as xgboost_aft_prediction_frame,
    metrics_for_predictions as xgboost_aft_metrics_for_predictions,
)
from src.tasks.autoregressive.evaluation.forecast_metrics import (  # noqa: E402
    compute_horizon_metrics,
    compute_trajectory_metrics,
    inverse_context_standardization,
)
from src.tasks.autoregressive.models.gru import build_gru_forecaster  # noqa: E402
from src.tasks.autoregressive.models.gru_training import (  # noqa: E402
    create_gru_data_loader,
    predict_gru_forecaster,
)
from src.tasks.autoregressive.models.patchtst_training import (  # noqa: E402
    create_patchtst_data_loader,
    predict_patchtst_forecaster,
)
from src.tasks.current_level_persistence.data.dataset import (  # noqa: E402
    ensure_scalar_context_features,
)
from src.tasks.current_level_persistence.evaluation.metrics import (  # noqa: E402
    compute_duration_metrics,
    compute_event_duration_metrics,
)
from src.tasks.current_level_persistence.models.learnable_shapelets import (  # noqa: E402
    build_learnable_shapelet_model,
)
from src.tasks.current_level_persistence.models.xgboost import (  # noqa: E402
    build_tabular_features,
    load_duration_regressor,
    predict_duration_regressor,
)
from src.tasks.current_level_persistence.utils.paths import (  # noqa: E402
    model_dir as current_level_model_dir,
    run_dir as current_level_run_dir,
)
from src.tasks.long_fade_detection.models.classifiers import (  # noqa: E402
    MODEL_ID_XGBOOST as LONG_FADE_XGBOOST_ID,
    build_neural_classifier,
)
from src.tasks.long_fade_detection.utils.paths import (  # noqa: E402
    model_dir as long_fade_model_dir,
    run_dir as long_fade_run_dir,
)
from src.tasks.survival_persistence.adapters.discrete_time import (  # noqa: E402
    DiscreteTimeBinSpec,
)
from src.tasks.survival_persistence.evaluation.metrics import (  # noqa: E402
    fit_censoring_survival,
)
from src.tasks.survival_persistence.models.discrete_time_tcn import (  # noqa: E402
    DiscreteTimeTCNConfig,
    DiscreteTimeTCNSurvivalModel,
    discrete_time_survival_nll,
)
from src.tasks.survival_persistence.models.xgboost_aft import (  # noqa: E402
    build_dmatrix,
    require_xgboost,
    validate_feature_compatibility,
)
from src.tasks.survival_persistence.training.discrete_time import (  # noqa: E402
    evaluate_discrete_time_model,
    logits_to_prediction_arrays,
    make_discrete_time_predictions_frame,
)
from src.tasks.survival_persistence.utils.paths import (  # noqa: E402
    model_dir as survival_model_dir,
    run_dir as survival_run_dir,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.device import select_device  # noqa: E402
from src.utils.paths import project_path  # noqa: E402
from src.utils.results_paths import (  # noqa: E402
    ensure_results_subdirs,
    get_model_dir,
    get_run_dir,
    relative_project_path,
    selection_id_from_metadata,
)


AUTOREGRESSIVE_L30_DATASET_ROOT = (
    PROJECT_ROOT / "data/processed/autoregressive/threshold_10p0/datasets_L30_h10"
)
CURRENT_LEVEL_DATASET_DIR = (
    PROJECT_ROOT
    / "data/processed/current_level_persistence/delta_0p5/L30/"
    / "externalHoldout_test_fc_uplink_fade"
)
LONG_FADE_DATASET_DIR = (
    PROJECT_ROOT
    / "data/processed/long_fade_detection/threshold_10p0/min_duration_300s/L30/"
    / "externalHoldout_test_fc_uplink_fade"
)
SURVIVAL_DATASET_DIR = (
    PROJECT_ROOT
    / "data/processed/survival_persistence/threshold_10p0/L30/"
    / "externalHoldout_test_fc_uplink_fade"
)
SURVIVAL_SELECTION_ID = "externalHoldout_test_fc_uplink_fade"
SURVIVAL_TCN_GRID_BEST_RUN_ID = (
    "survivalPersistence_discrete_time_tcn_relative_current_plus_scalar_uniform_"
    "externalHoldout_test_fc_uplink_fade_grid_extended_best"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="autoregressive,current_level_persistence,long_fade_detection,survival_persistence",
        help="Comma-separated tasks to refresh.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for neural checkpoints: auto, cuda, mps, cpu, cuda:0, ...",
    )
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Skip saved XGBoost models when xgboost is not available locally.",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as arrays:
        return {key: arrays[key] for key in arrays.files}


def load_checkpoint(path: Path, *, map_location: str | torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def update_metadata(path: Path, *, refresh_note: dict[str, Any]) -> None:
    metadata = load_yaml_config(path) if path.exists() else {}
    metadata["prediction_refresh"] = refresh_note
    save_yaml(path, metadata)


def torch_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(select_device())
    return torch.device(requested)


def build_autoregressive_prediction_table(
    metadata: pd.DataFrame,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    *,
    model_family: str,
    architecture: str,
    variant: str,
    y_pred_variant: np.ndarray | None = None,
) -> pd.DataFrame:
    """Create the standard long-form autoregressive prediction table."""

    metadata = metadata.reset_index(drop=True)
    copied_columns = [
        "window_id",
        "event_id",
        "dataset_id",
        "dataset_name",
        "quality_flag",
        "split",
        "input_start_time",
        "input_end_time",
        "target_start_time",
        "target_end_time",
        "scaling_mean",
        "scaling_std",
    ]
    rows: list[dict[str, Any]] = []
    for row_index, row in metadata.iterrows():
        for horizon_index in range(y_true_raw.shape[1]):
            observed = float(y_true_raw[row_index, horizon_index])
            predicted = float(y_pred_raw[row_index, horizon_index])
            output = {column: row[column] for column in copied_columns}
            output.update(
                {
                    "model_family": model_family,
                    "architecture": architecture,
                    "variant": variant,
                    "horizon_step": horizon_index + 1,
                    "y_true_raw": observed,
                    "y_pred_raw": predicted,
                    "error": predicted - observed,
                    "absolute_error": abs(predicted - observed),
                    "squared_error": (predicted - observed) ** 2,
                }
            )
            if y_pred_variant is not None and variant == "context_standard":
                output["y_pred_normalized"] = float(
                    y_pred_variant[row_index, horizon_index]
                )
            rows.append(output)
    return pd.DataFrame(rows)


def save_autoregressive_tables(
    *,
    run_id: str,
    model_family: str,
    architecture: str,
    variant: str,
    split_metadata: dict[str, pd.DataFrame],
    split_arrays: dict[str, dict[str, np.ndarray]],
    predictions_raw: dict[str, np.ndarray],
    predictions_variant: dict[str, np.ndarray] | None = None,
) -> None:
    run_path = get_run_dir(run_id)
    paths = ensure_results_subdirs(run_path, ("metrics", "predictions", "tables"))
    val_metrics = compute_trajectory_metrics(
        split_arrays["val"]["y_raw"],
        predictions_raw["val"],
    )
    test_metrics = compute_trajectory_metrics(
        split_arrays["test"]["y_raw"],
        predictions_raw["test"],
    )
    metrics_summary = pd.DataFrame(
        [
            {
                "architecture": architecture,
                "variant": variant,
                "validation_mae_raw": val_metrics["mae"],
                "validation_rmse_raw": val_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "num_train_windows": len(split_metadata["train"]),
                "num_val_windows": len(split_metadata["val"]),
                "num_test_windows": len(split_metadata["test"]),
                "refreshed_without_retraining": True,
            }
        ]
    )
    horizon_metrics = compute_horizon_metrics(
        split_arrays["test"]["y_raw"],
        predictions_raw["test"],
    )
    horizon_metrics.insert(0, "variant", variant)
    horizon_metrics.insert(0, "architecture", architecture)
    test_predictions = build_autoregressive_prediction_table(
        split_metadata["test"],
        split_arrays["test"]["y_raw"],
        predictions_raw["test"],
        model_family=model_family,
        architecture=architecture,
        variant=variant,
        y_pred_variant=(
            None if predictions_variant is None else predictions_variant["test"]
        ),
    )
    val_predictions = build_autoregressive_prediction_table(
        split_metadata["val"],
        split_arrays["val"]["y_raw"],
        predictions_raw["val"],
        model_family=model_family,
        architecture=architecture,
        variant=variant,
        y_pred_variant=(
            None if predictions_variant is None else predictions_variant["val"]
        ),
    )
    metrics_summary.to_csv(paths["metrics"] / "metrics_summary.csv", index=False)
    horizon_metrics.to_csv(paths["metrics"] / "metrics_by_horizon.csv", index=False)
    grouped_metrics(test_predictions, "dataset_name").to_csv(
        paths["metrics"] / "metrics_by_dataset.csv",
        index=False,
    )
    grouped_metrics(test_predictions, "quality_flag").to_csv(
        paths["metrics"] / "metrics_by_quality_flag.csv",
        index=False,
    )
    grouped_metrics(test_predictions, "event_id").to_csv(
        paths["metrics"] / "metrics_by_event.csv",
        index=False,
    )
    test_predictions.to_parquet(
        paths["predictions"] / "test_predictions.parquet",
        index=False,
    )
    val_predictions.to_parquet(
        paths["predictions"] / "val_predictions.parquet",
        index=False,
    )
    refresh_note = {
        "refreshed_at": now_utc(),
        "reason": "dataset rebuilt; predictions refreshed without retraining",
    }
    update_metadata(run_path / "metadata.yaml", refresh_note=refresh_note)


def load_autoregressive_dataset() -> tuple[
    Path,
    str,
    dict[str, Any],
    dict[str, dict[str, np.ndarray]],
    dict[str, pd.DataFrame],
]:
    dataset_folder = resolve_selection_folder(
        AUTOREGRESSIVE_L30_DATASET_ROOT,
        "externalHoldout_test_fc_uplink_fade",
    )
    dataset_metadata = load_yaml_config(dataset_folder / "dataset_metadata.yaml")
    selection_id = selection_id_from_metadata(dataset_metadata)
    split_arrays = {
        split: load_autoregressive_split_arrays(dataset_folder, split)
        for split in ("train", "val", "test")
    }
    split_metadata = {
        split: pd.read_parquet(dataset_folder / f"{split}_metadata.parquet")
        for split in ("train", "val", "test")
    }
    return dataset_folder, selection_id, dataset_metadata, split_arrays, split_metadata


def refresh_autoregressive_gru(device: torch.device) -> list[str]:
    dataset_folder, selection_id, _, split_arrays, split_metadata = (
        load_autoregressive_dataset()
    )
    refreshed: list[str] = []
    for checkpoint_path in sorted(
        (PROJECT_ROOT / "models/autoregressive/gru").glob(
            f"*{selection_id}/best_model.pt"
        )
    ):
        run_id = checkpoint_path.parent.name
        model_config = load_yaml_config(checkpoint_path.parent / "model_config.yaml")
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model = build_gru_forecaster(
            str(model_config["architecture"]),
            input_size=int(model_config["input_size"]),
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            prediction_length=int(model_config["prediction_length"]),
            dropout=float(model_config["dropout"]),
            bidirectional=bool(model_config["bidirectional"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        variant = str(model_config["variant"])
        pred_variant: dict[str, np.ndarray] = {}
        pred_raw: dict[str, np.ndarray] = {}
        for split in ("val", "test"):
            loader = create_gru_data_loader(
                dataset_folder / f"{split}.npz",
                variant=variant,
                batch_size=512,
                shuffle=False,
                seed=42,
            )
            predictions, indices = predict_gru_forecaster(model, loader, device=device)
            if not np.array_equal(indices, np.arange(len(indices))):
                raise RuntimeError(f"{run_id} {split} predictions are not aligned.")
            pred_variant[split] = predictions
            pred_raw[split] = (
                predictions
                if variant == "raw"
                else inverse_context_standardization(
                    predictions,
                    split_arrays[split]["scaling_mean"],
                    split_arrays[split]["scaling_std"],
                )
            )
        save_autoregressive_tables(
            run_id=run_id,
            model_family="gru",
            architecture=str(model_config["architecture"]),
            variant=variant,
            split_metadata=split_metadata,
            split_arrays=split_arrays,
            predictions_raw=pred_raw,
            predictions_variant=pred_variant,
        )
        refreshed.append(run_id)
    return refreshed


def refresh_autoregressive_patchtst(device: torch.device) -> list[str]:
    dataset_folder, selection_id, _, split_arrays, split_metadata = (
        load_autoregressive_dataset()
    )
    refreshed: list[str] = []
    for checkpoint_path in sorted(
        (PROJECT_ROOT / "models/autoregressive/patchtst").glob(
            f"*{selection_id}/best_model.pt"
        )
    ):
        run_id = checkpoint_path.parent.name
        model_config = load_yaml_config(checkpoint_path.parent / "model_config.yaml")
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model = build_patchtst_model(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        variant = str(model_config["variant"])
        pred_variant: dict[str, np.ndarray] = {}
        pred_raw: dict[str, np.ndarray] = {}
        for split in ("val", "test"):
            loader = create_patchtst_data_loader(
                dataset_folder / f"{split}.npz",
                variant=variant,
                batch_size=512,
                shuffle=False,
                seed=42,
            )
            predictions, indices = predict_patchtst_forecaster(
                model,
                loader,
                device=device,
            )
            if not np.array_equal(indices, np.arange(len(indices))):
                raise RuntimeError(f"{run_id} {split} predictions are not aligned.")
            pred_variant[split] = predictions
            pred_raw[split] = autoregressive_predictions_in_raw_scale(
                predictions,
                split_arrays[split],
                variant=variant,
            )
        save_autoregressive_tables(
            run_id=run_id,
            model_family="patchtst",
            architecture="patchtst",
            variant=variant,
            split_metadata=split_metadata,
            split_arrays=split_arrays,
            predictions_raw=pred_raw,
            predictions_variant=pred_variant,
        )
        refreshed.append(run_id)
    return refreshed


def refresh_autoregressive_xgboost() -> list[str]:
    from src.tasks.autoregressive.models.xgboost import (
        load_horizon_regressors,
        predict_horizon_regressors,
    )

    _, selection_id, _, split_arrays, split_metadata = load_autoregressive_dataset()
    refreshed: list[str] = []
    for model_path in sorted(
        (PROJECT_ROOT / "models/autoregressive/xgboost").glob(
            f"*{selection_id}/best_model.joblib"
        )
    ):
        run_id = model_path.parent.name
        checkpoint = load_horizon_regressors(model_path)
        models = checkpoint["models"]
        variant = str(checkpoint["metadata"]["variant"])
        pred_variant: dict[str, np.ndarray] = {}
        pred_raw: dict[str, np.ndarray] = {}
        for split in ("val", "test"):
            input_key = "X_raw" if variant == "raw" else "X_context_standard"
            predictions = predict_horizon_regressors(models, split_arrays[split][input_key])
            pred_variant[split] = predictions
            pred_raw[split] = (
                predictions
                if variant == "raw"
                else inverse_context_standardization(
                    predictions,
                    split_arrays[split]["scaling_mean"],
                    split_arrays[split]["scaling_std"],
                )
            )
        save_autoregressive_tables(
            run_id=run_id,
            model_family="xgboost",
            architecture="xgboost",
            variant=variant,
            split_metadata=split_metadata,
            split_arrays=split_arrays,
            predictions_raw=pred_raw,
            predictions_variant=pred_variant,
        )
        refreshed.append(run_id)
    return refreshed


def scaler_from_current_level_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    if metadata.get("scalar_context_scaler") is not None:
        return metadata["scalar_context_scaler"]
    features = metadata.get("features", {})
    return features.get("scalar_context_scaler")


def apply_saved_current_level_scalar_scaler(
    split_arrays: dict[str, dict[str, np.ndarray]],
    *,
    config: dict[str, Any],
    scaler: dict[str, Any] | None,
    delta: float,
) -> dict[str, dict[str, np.ndarray]]:
    features = config.get("features", {})
    if not bool(features.get("use_scalar_context", False)):
        return {split: dict(arrays) for split, arrays in split_arrays.items()}
    scalar_key = str(
        features.get(
            "scalar_context_key",
            config["dataset"].get("scalar_context_key", "scalar_context_features"),
        )
    )
    raw_key = str(config["dataset"].get("raw_input_key", "X_raw"))
    prepared = {
        split: ensure_scalar_context_features(
            arrays,
            scalar_context_key=scalar_key,
            raw_input_key=raw_key,
            delta=delta,
        )
        for split, arrays in split_arrays.items()
    }
    if bool(features.get("scalar_context_standardize", True)):
        if scaler is None:
            raise RuntimeError("Saved scalar scaler is required for predict-only refresh.")
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)
        for arrays in prepared.values():
            arrays[scalar_key] = ((arrays[scalar_key] - mean) / std).astype(np.float32)
    return prepared


def load_current_level_dataset() -> tuple[
    dict[str, Any],
    dict[str, dict[str, np.ndarray]],
    dict[str, pd.DataFrame],
]:
    dataset_metadata = load_yaml_config(CURRENT_LEVEL_DATASET_DIR / "dataset_metadata.yaml")
    split_arrays = {
        split: load_current_level_npz(
            CURRENT_LEVEL_DATASET_DIR / ("val.npz" if split == "val" else f"{split}.npz")
        )
        for split in ("train", "val", "test")
    }
    split_metadata = {
        split: pd.read_parquet(CURRENT_LEVEL_DATASET_DIR / f"{split}_metadata.parquet")
        for split in ("train", "val", "test")
    }
    return dataset_metadata, split_arrays, split_metadata


def current_level_metrics_row(
    *,
    model_id: str,
    run_id: str,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    split_metadata: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": model_id,
                "run_id": run_id,
                "val_mae_seconds": val_metrics["mae_seconds"],
                "val_rmse_seconds": val_metrics["rmse_seconds"],
                "val_median_ae_seconds": val_metrics["median_ae_seconds"],
                "test_mae_seconds": test_metrics["mae_seconds"],
                "test_rmse_seconds": test_metrics["rmse_seconds"],
                "test_median_ae_seconds": test_metrics["median_ae_seconds"],
                "num_train_windows": len(split_metadata["train"]),
                "num_val_windows": len(split_metadata["val"]),
                "num_test_windows": len(split_metadata["test"]),
                "refreshed_without_retraining": True,
            }
        ]
    )


def refresh_current_level_shapelets(device: torch.device) -> list[str]:
    dataset_metadata, raw_split_arrays, split_metadata = load_current_level_dataset()
    delta = float(dataset_metadata["delta"])
    refreshed: list[str] = []
    model_root = PROJECT_ROOT / "models/current_level_persistence"
    for checkpoint_path in sorted(model_root.glob("multiscale_shapelet_*/**/best_model.pt")):
        run_id = checkpoint_path.parent.name
        model_metadata = load_yaml_config(checkpoint_path.parent / "metadata.yaml")
        config = load_yaml_config(checkpoint_path.parent / "config_resolved.yaml")
        scaler = scaler_from_current_level_metadata(model_metadata)
        split_arrays = apply_saved_current_level_scalar_scaler(
            raw_split_arrays,
            config=config,
            scaler=scaler,
            delta=delta,
        )
        model_id = str(config["model_id"])
        model = build_learnable_shapelet_model(
            model_id,
            build_shapelet_config(config, int(dataset_metadata["context_length"])),
        )
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        input_key = str(config["dataset"]["input_key"])
        target_key = str(config["dataset"]["target_key"])
        scalar_key = (
            str(config["features"].get("scalar_context_key", "scalar_context_features"))
            if bool(config.get("features", {}).get("use_scalar_context", False))
            else None
        )
        loss_fn = make_current_level_loss(str(config["training"].get("loss", "huber")))
        predictions: dict[str, pd.DataFrame] = {}
        metrics: dict[str, dict[str, float]] = {}
        for split in ("val", "test"):
            loader = make_current_level_loader(
                split_arrays[split],
                input_key=input_key,
                target_key=target_key,
                scalar_context_key=scalar_key,
                batch_size=512,
                shuffle=False,
            )
            split_metrics, y_true, y_pred = evaluate_current_level_split(
                model,
                loader,
                split_metadata[split],
                loss_fn=loss_fn,
                device=device,
            )
            predictions[split] = current_level_prediction_frame(
                split_metadata[split],
                y_true,
                y_pred,
                model_id=model_id,
                run_id=run_id,
                split=split,
            )
            metrics[split] = split_metrics
        run_path = current_level_run_dir(run_id)
        paths = ensure_results_subdirs(run_path, ("metrics", "predictions", "tables"))
        current_level_metrics_row(
            model_id=model_id,
            run_id=run_id,
            val_metrics=metrics["val"],
            test_metrics=metrics["test"],
            split_metadata=split_metadata,
        ).to_csv(paths["metrics"] / "metrics_summary.csv", index=False)
        predictions["val"].to_parquet(
            paths["predictions"] / "val_predictions.parquet",
            index=False,
        )
        predictions["test"].to_parquet(
            paths["predictions"] / "test_predictions.parquet",
            index=False,
        )
        update_metadata(
            run_path / "metadata.yaml",
            refresh_note={
                "refreshed_at": now_utc(),
                "reason": "dataset rebuilt; predictions refreshed without retraining",
                "dataset_build_fingerprint": config_fingerprint(dataset_metadata),
            },
        )
        refreshed.append(run_id)
    return refreshed


def refresh_current_level_xgboost() -> list[str]:
    dataset_metadata, raw_split_arrays, split_metadata = load_current_level_dataset()
    delta = float(dataset_metadata["delta"])
    refreshed: list[str] = []
    for model_path in sorted(
        (PROJECT_ROOT / "models/current_level_persistence/xgboost").glob(
            "*/best_model.joblib"
        )
    ):
        run_id = model_path.parent.name
        model_metadata = load_yaml_config(model_path.parent / "metadata.yaml")
        config = load_yaml_config(model_path.parent / "config_resolved.yaml")
        scaler = scaler_from_current_level_metadata(model_metadata)
        split_arrays = apply_saved_current_level_scalar_scaler(
            raw_split_arrays,
            config=config,
            scaler=scaler,
            delta=delta,
        )
        checkpoint = load_duration_regressor(model_path)
        model = checkpoint["model"]
        input_key = str(config["dataset"]["input_key"])
        scalar_key = (
            str(config["features"].get("scalar_context_key", "scalar_context_features"))
            if bool(config.get("features", {}).get("use_scalar_context", False))
            else None
        )
        target_key = str(config["dataset"]["target_key"])
        predictions: dict[str, pd.DataFrame] = {}
        metrics: dict[str, dict[str, float]] = {}
        for split in ("val", "test"):
            x_values = build_tabular_features(
                split_arrays[split],
                input_key=input_key,
                scalar_context_key=scalar_key,
            )
            y_pred = predict_duration_regressor(model, x_values)
            y_true = np.asarray(split_arrays[split][target_key], dtype=np.float32)
            predictions[split] = current_level_prediction_frame(
                split_metadata[split],
                y_true,
                y_pred,
                model_id="xgboost",
                run_id=run_id,
                split=split,
            )
            metrics[split] = compute_duration_metrics(y_true, y_pred)
            metrics[split].update(
                compute_event_duration_metrics(
                    split_metadata[split]["global_event_id"].to_numpy(),
                    y_true,
                    y_pred,
                )
            )
        run_path = current_level_run_dir(run_id)
        paths = ensure_results_subdirs(run_path, ("metrics", "predictions", "tables"))
        current_level_metrics_row(
            model_id="xgboost",
            run_id=run_id,
            val_metrics=metrics["val"],
            test_metrics=metrics["test"],
            split_metadata=split_metadata,
        ).to_csv(paths["metrics"] / "metrics_summary.csv", index=False)
        predictions["val"].to_parquet(
            paths["predictions"] / "val_predictions.parquet",
            index=False,
        )
        predictions["test"].to_parquet(
            paths["predictions"] / "test_predictions.parquet",
            index=False,
        )
        update_metadata(
            run_path / "metadata.yaml",
            refresh_note={
                "refreshed_at": now_utc(),
                "reason": "dataset rebuilt; predictions refreshed without retraining",
                "dataset_build_fingerprint": config_fingerprint(dataset_metadata),
            },
        )
        refreshed.append(run_id)
    return refreshed


def refresh_long_fade(device: torch.device, *, skip_xgboost: bool) -> list[str]:
    arrays, metadata = load_long_fade_split_data(LONG_FADE_DATASET_DIR)
    dataset_metadata = load_yaml_config(LONG_FADE_DATASET_DIR / "dataset_metadata.yaml")
    dataset_config_fingerprint = dataset_metadata.get("config_fingerprint")
    refreshed: list[str] = []
    run_root = PROJECT_ROOT / "results/runs/long_fade_detection"
    for run_path in sorted(run_root.glob("*/*")):
        if not run_path.is_dir():
            continue
        run_id = run_path.name
        config_path = run_path / "config_resolved.yaml"
        if not config_path.exists():
            continue
        config = load_yaml_config(config_path)
        model_id = str(config["model_id"])
        checkpoint_path = long_fade_model_dir(run_id)
        if model_id == LONG_FADE_XGBOOST_ID:
            if skip_xgboost:
                continue
            import joblib

            model = joblib.load(checkpoint_path / "model.joblib")
            predictions = {}
            for split in ("train", "validation", "test"):
                logits, probs = predict_long_fade_sklearn(
                    model,
                    arrays[split]["X_lag_scalar"],
                )
                predictions[split] = long_fade_prediction_frame(
                    metadata[split],
                    arrays=arrays[split],
                    split=split,
                    logits=logits,
                    probabilities=probs,
                    model_id=model_id,
                    run_id=run_id,
                )
            backend = "xgboost"
            extra = {"refreshed_without_retraining": True}
        else:
            checkpoint = load_checkpoint(checkpoint_path / "best_model.pt", map_location=device)
            checkpoint_config = checkpoint.get("config", config)
            model_config = dict(checkpoint_config.get("model", {}))
            if "shapelet_convolution" in model_id:
                model_config["shapelets"] = checkpoint_config.get("shapelets", {})
            model = build_neural_classifier(
                model_id,
                context_length=int(checkpoint_config["dataset"]["context_length"]),
                scalar_context_dim=int(arrays["train"]["scalar_context_features"].shape[1]),
                config=model_config,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            predictions = long_fade_neural_predictions(
                model,
                arrays,
                metadata,
                config=checkpoint_config,
                model_id=model_id,
                run_id=run_id,
                device_name=str(device),
            )
            backend = "pytorch"
            extra = {
                "selected_device": str(device),
                "refreshed_without_retraining": True,
            }
        metrics_summary, threshold_table, event_table = evaluate_long_fade_predictions(
            predictions
        )
        metrics_summary.insert(0, "run_id", run_id)
        metrics_summary.insert(0, "model_id", model_id)
        save_long_fade_outputs(
            config=config,
            run_id=run_id,
            model_id=model_id,
            backend=backend,
            run_path=run_path,
            checkpoint_path=checkpoint_path,
            predictions=predictions,
            metrics_summary=metrics_summary,
            threshold_metrics=threshold_table,
            event_metrics=event_table,
            extra_metadata=extra,
        )
        update_metadata(
            run_path / "metadata.yaml",
            refresh_note={
                "refreshed_at": now_utc(),
                "reason": "dataset rebuilt; predictions refreshed without retraining",
                "dataset_config_fingerprint": dataset_config_fingerprint,
            },
        )
        refreshed.append(run_id)
    return refreshed


def refresh_survival_xgboost_aft() -> list[str]:
    xgb = require_xgboost()
    arrays, split_metadata = load_survival_aft_split_artifacts(SURVIVAL_DATASET_DIR)
    refreshed: list[str] = []
    for booster_path in sorted(
        (PROJECT_ROOT / "models/survival_persistence/xgboost_aft").glob(
            "*/booster.json"
        )
    ):
        run_id = booster_path.parent.name
        config = load_yaml_config(booster_path.parent / "model_config.yaml")
        metadata = load_yaml_config(booster_path.parent / "training_metadata.yaml")
        feature_set = str(config["dataset"]["feature_set"])
        sample_weighting = str(config["dataset"].get("sample_weighting", "uniform"))
        dmatrices = {}
        reference_features = None
        for split in ("train", "validation", "test"):
            dmatrix, features = build_dmatrix(
                arrays[split],
                feature_set=feature_set,
                sample_weighting=sample_weighting,
            )
            if reference_features is None:
                reference_features = features
            else:
                validate_feature_compatibility(reference_features, features)
            dmatrices[split] = dmatrix
        booster = xgb.Booster()
        booster.load_model(booster_path)
        best_iteration = int(metadata.get("best_iteration", booster.num_boosted_rounds() - 1))
        params = dict(config["model"])
        distribution = str(params["aft_loss_distribution"])
        scale = float(params["aft_loss_distribution_scale"])
        horizons = np.asarray(config["prediction"]["horizons_seconds"], dtype=float)
        output_dir = survival_run_dir(
            selection_id=SURVIVAL_SELECTION_ID,
            run_id=run_id,
        )
        paths = ensure_results_subdirs(
            output_dir,
            ("metrics", "predictions", "figures", "tables"),
        )
        prediction_frames = {}
        for split, dmatrix in dmatrices.items():
            location = booster.predict(
                dmatrix,
                output_margin=True,
                iteration_range=(0, best_iteration + 1),
            )
            frame, _ = xgboost_aft_prediction_frame(
                split=split,
                metadata=split_metadata[split],
                arrays=arrays[split],
                location=location,
                distribution=distribution,
                scale=scale,
                horizons_seconds=horizons,
                model_family="xgboost",
                model_id="xgboost_aft",
                feature_set=feature_set,
                sample_weighting=sample_weighting,
            )
            prediction_frames[split] = frame
            frame.to_parquet(
                paths["predictions"] / f"{split}_predictions.parquet",
                index=False,
            )
        censoring_curve = fit_censoring_survival(
            arrays["train"]["y_time_seconds"],
            arrays["train"]["y_event_observed"],
        )
        metric_rows = []
        brier_tables = []
        for split, frame in prediction_frames.items():
            row, brier = xgboost_aft_metrics_for_predictions(
                method_name="xgboost_aft",
                split=split,
                predictions=frame,
                horizons_seconds=horizons,
                censoring_curve=censoring_curve,
                distribution=distribution,
                scale=scale,
            )
            metric_rows.append(row)
            brier_tables.append(brier)
        pd.DataFrame(metric_rows).to_csv(
            paths["metrics"] / "metrics_summary.csv",
            index=False,
        )
        pd.concat(brier_tables, ignore_index=True).to_csv(
            paths["metrics"] / "brier_by_horizon.csv",
            index=False,
        )
        calibration = xgboost_aft_calibration_tables(
            predictions=prediction_frames["test"],
            horizons_seconds=list(config["evaluation"]["calibration_horizons_seconds"]),
            n_bins=int(config["evaluation"].get("calibration_bins", 10)),
            censoring_curve=censoring_curve,
        )
        calibration.to_csv(paths["metrics"] / "test_calibration.csv", index=False)
        update_metadata(
            output_dir / "metadata.yaml",
            refresh_note={
                "refreshed_at": now_utc(),
                "reason": "dataset rebuilt; predictions refreshed without retraining",
                "dataset_config_fingerprint": config_fingerprint(
                    load_yaml_config(SURVIVAL_DATASET_DIR / "dataset_metadata.yaml")
                ),
            },
        )
        refreshed.append(run_id)
    return refreshed


def bin_spec_from_saved_config(config: dict[str, Any]) -> DiscreteTimeBinSpec:
    spec = config.get("resolved_bin_spec")
    if spec is None:
        spec = config.get("bin_spec")
    if spec is None:
        raise RuntimeError("Saved discrete-time TCN config does not contain bin spec.")

    def parse(values: list[float | str]) -> np.ndarray:
        return np.asarray([np.inf if value == "inf" else float(value) for value in values])

    return DiscreteTimeBinSpec(
        left_edges_seconds=parse(spec["left_edges_seconds"]),
        right_edges_seconds=parse(spec["right_edges_seconds"]),
        strategy=str(spec["strategy"]),
        include_open_ended=bool(spec["include_open_ended"]),
    )


def refresh_survival_discrete_tcn(device: torch.device) -> list[str]:
    arrays, split_metadata = load_survival_split_artifacts(SURVIVAL_DATASET_DIR)
    run_id = SURVIVAL_TCN_GRID_BEST_RUN_ID
    model_path = survival_model_dir(model_id="discrete_time_tcn", run_id=run_id)
    config = load_yaml_config(model_path / "model_config.yaml")
    checkpoint = load_checkpoint(model_path / "best_model.pt", map_location=device)
    bin_spec = bin_spec_from_saved_config(config)
    dataset_config = dict(config["dataset"])
    prediction_config = dict(config["prediction"])
    horizons = np.asarray(prediction_config["horizons_seconds"], dtype=float)
    _, loaders = make_survival_discrete_loaders(
        arrays,
        bin_spec=bin_spec,
        config=config,
    )
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
        use_scalar_context=bool(dataset_config.get("use_scalar_context", False)),
        scalar_context_dim=int(config["model"].get("scalar_context_dim", 0)),
        scalar_hidden_dim=int(config["model"].get("scalar_hidden_dim", 32)),
    )
    model = DiscreteTimeTCNSurvivalModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    output_dir = survival_run_dir(selection_id=SURVIVAL_SELECTION_ID, run_id=run_id)
    paths = ensure_results_subdirs(
        output_dir,
        ("metrics", "predictions", "figures", "tables"),
    )
    feature_set = survival_discrete_feature_set_name(config)
    sample_weighting = str(dataset_config.get("sample_weighting", "uniform"))
    prediction_frames = {}
    eval_results = {}
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
            model_id="discrete_time_tcn",
            feature_set=feature_set,
            sample_weighting=sample_weighting,
        )
        prediction_frames[split] = frame
        frame.to_parquet(paths["predictions"] / f"{split}_predictions.parquet", index=False)
    censoring_curve = fit_censoring_survival(
        arrays["train"]["y_time_seconds"],
        arrays["train"]["y_event_observed"],
    )
    metric_rows = []
    brier_tables = []
    for split, frame in prediction_frames.items():
        row, brier = discrete_tcn_metrics_for_predictions(
            method_name="discrete_time_tcn",
            split=split,
            predictions=frame,
            horizons_seconds=horizons,
            censoring_curve=censoring_curve,
            discrete_nll=float(eval_results[split]["loss"]),
        )
        metric_rows.append(row)
        brier_tables.append(brier)
    pd.DataFrame(metric_rows).to_csv(paths["metrics"] / "metrics_summary.csv", index=False)
    pd.concat(brier_tables, ignore_index=True).to_csv(
        paths["metrics"] / "brier_by_horizon.csv",
        index=False,
    )
    update_metadata(
        output_dir / "metadata.yaml",
        refresh_note={
            "refreshed_at": now_utc(),
            "reason": "dataset rebuilt; predictions refreshed without retraining",
            "dataset_config_fingerprint": config_fingerprint(
                load_yaml_config(SURVIVAL_DATASET_DIR / "dataset_metadata.yaml")
            ),
        },
    )
    return [run_id]


def make_survival_discrete_loaders(
    arrays: dict[str, dict[str, np.ndarray]],
    *,
    bin_spec: DiscreteTimeBinSpec,
    config: dict[str, Any],
):
    from scripts.experiments.survival_persistence.run_discrete_time_tcn import (
        make_loaders,
    )

    return make_loaders(arrays, bin_spec=bin_spec, config=config)


def survival_discrete_feature_set_name(config: dict[str, Any]) -> str:
    sequence = str(config["dataset"]["sequence_representation"])
    scalar = bool(config["dataset"].get("use_scalar_context", False))
    suffix = "_plus_scalar" if scalar else ""
    if sequence == "relative_to_threshold":
        return f"relative_threshold{suffix}"
    if sequence == "relative_to_current":
        return f"relative_current{suffix}"
    return f"{sequence}{suffix}"


def print_refreshed(task: str, run_ids: list[str]) -> None:
    print(f"{task}: refreshed {len(run_ids)} run(s)")
    for run_id in run_ids:
        print(f"  - {run_id}")


def main() -> None:
    args = parse_args()
    selected_tasks = {
        item.strip()
        for item in str(args.tasks).split(",")
        if item.strip()
    }
    device = torch_device(str(args.device))
    print("=== Refresh Predictions Without Retraining ===")
    print(f"Tasks: {sorted(selected_tasks)}")
    print(f"Torch device: {device}")
    print(f"Skip XGBoost: {bool(args.skip_xgboost)}")

    if "autoregressive" in selected_tasks:
        run_ids = []
        run_ids.extend(refresh_autoregressive_gru(device))
        run_ids.extend(refresh_autoregressive_patchtst(device))
        if not args.skip_xgboost:
            run_ids.extend(refresh_autoregressive_xgboost())
        print_refreshed("autoregressive", run_ids)

    if "current_level_persistence" in selected_tasks:
        run_ids = refresh_current_level_shapelets(device)
        if not args.skip_xgboost:
            run_ids.extend(refresh_current_level_xgboost())
        print_refreshed("current_level_persistence", run_ids)

    if "long_fade_detection" in selected_tasks:
        run_ids = refresh_long_fade(device, skip_xgboost=bool(args.skip_xgboost))
        print_refreshed("long_fade_detection", run_ids)

    if "survival_persistence" in selected_tasks:
        run_ids = []
        if not args.skip_xgboost:
            run_ids.extend(refresh_survival_xgboost_aft())
        run_ids.extend(refresh_survival_discrete_tcn(device))
        print_refreshed("survival_persistence", run_ids)


if __name__ == "__main__":
    main()
