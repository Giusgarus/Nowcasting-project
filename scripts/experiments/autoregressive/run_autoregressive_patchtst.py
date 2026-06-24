"""Train configured PatchTST autoregressive forecasting baselines."""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.autoregressive.run_autoregressive_gru import (
    grouped_metrics,
    load_split_arrays,
    resolve_selection_folder,
)
from scripts.experiments.autoregressive.run_patchtst_grid_search import (
    build_loaders,
    build_patchtst_model,
    build_prediction_table,
    predictions_in_raw_scale,
    save_run_figures,
)
from src.evaluation.forecast_metrics import (
    compute_horizon_metrics,
    compute_trajectory_metrics,
)
from src.tasks.autoregressive.models.patchtst_training import (
    predict_patchtst_forecaster,
    train_patchtst_forecaster,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.device import select_device
from src.utils.paths import project_path
from src.utils.reproducibility import set_seed
from src.utils.results_paths import (
    COMPARISON_INDEX_COLUMNS,
    RUN_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_comparison_dir,
    get_model_dir,
    get_results_index_dir,
    get_run_dir,
    make_run_id,
    relative_project_path,
    selection_id_from_metadata,
    upsert_index_row,
)

CONFIG_PATH = PROJECT_ROOT / "configs/autoregressive/autoregressive_patchtst.yaml"


def validate_config(config: dict) -> None:
    """Validate that the configuration describes single PatchTST runs."""

    forbidden = {"search", "parallel", "parameter_grid", "fixed_model"}
    present = sorted(forbidden.intersection(config))
    if present:
        raise ValueError(
            "Single-run PatchTST config cannot contain grid-search sections: "
            + ", ".join(present)
        )
    if config["training"]["loss"] != "mse":
        raise ValueError("Only MSE PatchTST training loss is supported.")
    if not config["evaluation"]["evaluate_in_raw_scale"]:
        raise ValueError("PatchTST evaluation must run in raw scale.")
    variants = config["experiment"]["variants"]
    unsupported = sorted(set(variants).difference({"raw", "context_standard"}))
    if unsupported:
        raise ValueError(f"Unsupported PatchTST variants: {unsupported}")
    if not variants:
        raise ValueError("At least one PatchTST variant must be configured.")


def build_single_run_model_config(config: dict, variant: str) -> dict:
    """Return the complete serializable model configuration for one run."""

    return {
        "architecture": "patchtst",
        "variant": variant,
        "context_length": int(config["data"]["context_length"]),
        "prediction_length": int(config["data"]["prediction_length"]),
        **config["model"],
    }


def ensure_run_paths_are_available(
    model_dir: Path,
    result_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """Prevent silent replacement, or clear explicitly replaceable outputs."""

    if overwrite:
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(result_dir, ignore_errors=True)
        return
    for folder in (model_dir, result_dir):
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Run output already exists: {folder}")


def predict_raw_scale(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    arrays: dict[str, np.ndarray],
    *,
    variant: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict one split and return variant-scale and raw-scale trajectories."""

    prediction_variant, indices = predict_patchtst_forecaster(
        model,
        loader,
        device=device,
    )
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("PatchTST predictions are not aligned to metadata rows.")
    prediction_raw = predictions_in_raw_scale(
        prediction_variant,
        arrays,
        variant=variant,
    )
    return prediction_variant, prediction_raw


def main() -> None:
    """Train one configured PatchTST architecture for each data variant."""

    config = load_yaml_config(CONFIG_PATH)
    validate_config(config)
    dataset_folder = resolve_selection_folder(
        project_path(config["data"]["dataset_root"]),
        config["data"]["selection_folder"],
    )
    dataset_metadata = load_yaml_config(dataset_folder / "dataset_metadata.yaml")
    selection_id = selection_id_from_metadata(dataset_metadata)
    split_arrays = {
        split: load_split_arrays(dataset_folder, split)
        for split in ("train", "val", "test")
    }
    split_metadata = {
        split: pd.read_parquet(dataset_folder / f"{split}_metadata.parquet")
        for split in ("train", "val", "test")
    }
    for split, metadata in split_metadata.items():
        if metadata.empty:
            raise ValueError(f"The selected dataset has an empty {split} split.")

    comparison_id = f"patchtst_variant_{selection_id}"
    comparison_dir = get_comparison_dir(comparison_id)
    comparison_paths = ensure_results_subdirs(comparison_dir, ("tables",))
    comparison_path = (
        comparison_paths["tables"] / "patchtst_variant_comparison.csv"
    )
    if comparison_path.exists() and not config["output"]["overwrite"]:
        raise FileExistsError(f"Comparison output already exists: {comparison_path}")

    device = torch.device(select_device())
    print(
        "=== PatchTST single runs ===\n"
        f"Variants: {config['experiment']['variants']}\n"
        f"Device: {device}\n"
        f"Model: {json.dumps(config['model'], sort_keys=True)}\n"
        f"Training: {json.dumps(config['training'], sort_keys=True)}",
        flush=True,
    )
    comparison_rows = []
    for variant in config["experiment"]["variants"]:
        run_id = make_run_id("patchtst", "patchtst", variant, selection_id)
        model_dir = get_model_dir("patchtst", run_id)
        result_dir = get_run_dir(run_id)
        ensure_run_paths_are_available(
            model_dir,
            result_dir,
            overwrite=bool(config["output"]["overwrite"]),
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        result_paths = ensure_results_subdirs(
            result_dir,
            ("metrics", "predictions", "figures", "tables"),
        )

        print(f"\n=== Training {run_id} ===", flush=True)
        set_seed(int(config["training"]["seed"]))
        model_config = build_single_run_model_config(config, variant)
        model = build_patchtst_model(model_config)
        loaders = build_loaders(
            dataset_folder,
            variant=variant,
            training=config["training"],
            include_test=True,
        )
        result = train_patchtst_forecaster(
            model,
            loaders["train"],
            loaders["val"],
            device=device,
            learning_rate=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            max_epochs=int(config["training"]["max_epochs"]),
            early_stopping_patience=int(
                config["training"]["early_stopping_patience"]
            ),
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        )
        _, validation_raw = predict_raw_scale(
            model,
            loaders["val"],
            split_arrays["val"],
            variant=variant,
            device=device,
        )
        prediction_variant, test_raw = predict_raw_scale(
            model,
            loaders["test"],
            split_arrays["test"],
            variant=variant,
            device=device,
        )
        validation_metrics = compute_trajectory_metrics(
            split_arrays["val"]["y_raw"],
            validation_raw,
        )
        test_metrics = compute_trajectory_metrics(
            split_arrays["test"]["y_raw"],
            test_raw,
        )
        metrics_summary = pd.DataFrame(
            [
                {
                    "architecture": "patchtst",
                    "variant": variant,
                    "validation_mae_raw": validation_metrics["mae"],
                    "validation_rmse_raw": validation_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_rmse": test_metrics["rmse"],
                    "best_epoch": result.best_epoch,
                    "best_val_loss": result.best_val_loss,
                    **config["model"],
                    "learning_rate": config["training"]["learning_rate"],
                    "weight_decay": config["training"]["weight_decay"],
                    "num_train_windows": len(split_metadata["train"]),
                    "num_val_windows": len(split_metadata["val"]),
                    "num_test_windows": len(split_metadata["test"]),
                }
            ]
        )
        horizon_metrics = compute_horizon_metrics(
            split_arrays["test"]["y_raw"],
            test_raw,
        )
        horizon_metrics.insert(0, "variant", variant)
        horizon_metrics.insert(0, "architecture", "patchtst")
        predictions = build_prediction_table(
            split_metadata["test"],
            split_arrays["test"]["y_raw"],
            test_raw,
            variant=variant,
            y_pred_variant=prediction_variant,
        )
        history = pd.DataFrame(result.history)

        metrics_summary.to_csv(
            result_paths["metrics"] / "metrics_summary.csv",
            index=False,
        )
        horizon_metrics.to_csv(
            result_paths["metrics"] / "metrics_by_horizon.csv",
            index=False,
        )
        grouped_metrics(predictions, "dataset_name").to_csv(
            result_paths["metrics"] / "metrics_by_dataset.csv",
            index=False,
        )
        grouped_metrics(predictions, "quality_flag").to_csv(
            result_paths["metrics"] / "metrics_by_quality_flag.csv",
            index=False,
        )
        grouped_metrics(predictions, "event_id").to_csv(
            result_paths["metrics"] / "metrics_by_event.csv",
            index=False,
        )
        history.to_csv(
            result_paths["metrics"] / "training_history.csv",
            index=False,
        )
        metrics_summary.to_csv(
            result_paths["tables"] / "run_summary.csv",
            index=False,
        )
        predictions_path = result_paths["predictions"] / "test_predictions.parquet"
        if config["output"]["save_predictions"]:
            predictions.to_parquet(predictions_path, index=False)
        if config["output"]["save_plots"]:
            save_run_figures(
                history=history,
                horizon_metrics=horizon_metrics,
                metadata=split_metadata["test"],
                X_raw=split_arrays["test"]["X_raw"],
                y_true_raw=split_arrays["test"]["y_raw"],
                y_pred_raw=test_raw,
                figures_dir=result_paths["figures"],
            )

        torch.save(
            {
                "model_state_dict": result.best_state_dict,
                "model_config": model_config,
            },
            model_dir / "best_model.pt",
        )
        torch.save(
            {
                "model_state_dict": result.last_state_dict,
                "model_config": model_config,
            },
            model_dir / "last_model.pt",
        )
        save_yaml(model_dir / "model_config.yaml", model_config)
        created_at = datetime.now(timezone.utc).isoformat()
        training_metadata = {
            "run_type": "single_configuration",
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "learning_rate": float(config["training"]["learning_rate"]),
            "weight_decay": float(config["training"]["weight_decay"]),
            "batch_size": int(config["training"]["batch_size"]),
            "max_epochs": int(config["training"]["max_epochs"]),
            "early_stopping_patience": int(
                config["training"]["early_stopping_patience"]
            ),
            "gradient_clip_norm": float(config["training"]["gradient_clip_norm"]),
            "device": str(device),
            "seed": int(config["training"]["seed"]),
            "created_at": created_at,
        }
        save_yaml(model_dir / "training_metadata.yaml", training_metadata)
        save_yaml(
            result_dir / "metadata.yaml",
            {
                "run_id": run_id,
                "selection_id": selection_id,
                "architecture": "patchtst",
                "variant": variant,
                "metrics_in_raw_scale": True,
                "config_path": relative_project_path(CONFIG_PATH),
                "config_fingerprint": config_fingerprint(config),
                **training_metadata,
            },
        )
        upsert_index_row(
            get_results_index_dir() / "runs.csv",
            {
                "run_id": run_id,
                "model_family": "patchtst",
                "architecture": "patchtst",
                "variant": variant,
                "selection_id": selection_id,
                "dataset_selection": ";".join(dataset_metadata["selected_datasets"]),
                "threshold": dataset_metadata["threshold"],
                "context_length": dataset_metadata["context_length"],
                "prediction_length": dataset_metadata["prediction_length"],
                "results_path": relative_project_path(result_dir),
                "model_path": relative_project_path(model_dir),
                "predictions_path": (
                    relative_project_path(predictions_path)
                    if config["output"]["save_predictions"]
                    else ""
                ),
                "metrics_path": relative_project_path(result_paths["metrics"]),
                "status": "complete",
                "created_at": created_at,
            },
            id_column="run_id",
            columns=RUN_INDEX_COLUMNS,
        )
        comparison_rows.append(metrics_summary.iloc[0].to_dict())
        print(
            f"Completed {run_id}: validation RMSE raw="
            f"{validation_metrics['rmse']:.6g}, test RMSE raw="
            f"{test_metrics['rmse']:.6g}, best epoch={result.best_epoch}",
            flush=True,
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "validation_rmse_raw",
        kind="stable",
    )
    comparison.to_csv(comparison_path, index=False)
    created_at = datetime.now(timezone.utc).isoformat()
    save_yaml(
        comparison_dir / "metadata.yaml",
        {
            "comparison_id": comparison_id,
            "selection_id": selection_id,
            "comparison_type": "forecast_metrics_across_patchtst_single_runs",
            "table": relative_project_path(comparison_path),
            "created_at": created_at,
        },
    )
    upsert_index_row(
        get_results_index_dir() / "comparisons.csv",
        {
            "comparison_id": comparison_id,
            "method_id": "multiple_patchtst_single_runs",
            "reference_id": "",
            "comparison_type": "forecast_metrics",
            "selection_id": selection_id,
            "results_path": relative_project_path(comparison_dir),
            "metrics_path": relative_project_path(comparison_paths["tables"]),
            "figures_path": "",
            "status": "complete",
            "created_at": created_at,
        },
        id_column="comparison_id",
        columns=COMPARISON_INDEX_COLUMNS,
    )
    print("\n=== PatchTST single-run comparison ===", flush=True)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
