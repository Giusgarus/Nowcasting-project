"""Train and compare deterministic GRU autoregressive forecasting baselines."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.forecast_metrics import (
    compute_horizon_metrics,
    compute_trajectory_metrics,
    inverse_context_standardization,
)
from src.tasks.autoregressive.models.gru import build_gru_forecaster
from src.tasks.autoregressive.models.gru_training import (
    create_gru_data_loader,
    predict_gru_forecaster,
    train_gru_forecaster,
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

CONFIG_PATH = PROJECT_ROOT / "configs/autoregressive/autoregressive_gru.yaml"


def resolve_selection_folder(dataset_root: Path, configured: str) -> Path:
    """Resolve an explicit selection folder or the only available folder."""

    if configured != "auto":
        selected = dataset_root / configured
        if not selected.is_dir():
            raise FileNotFoundError(f"Dataset selection folder not found: {selected}")
        return selected
    folders = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if len(folders) != 1:
        available = ", ".join(path.name for path in folders) or "none"
        raise ValueError(
            "data.selection_folder='auto' requires exactly one folder; "
            f"available: {available}"
        )
    return folders[0]


def load_split_arrays(dataset_folder: Path, split: str) -> dict[str, np.ndarray]:
    """Load one immutable final-dataset NPZ split."""

    with np.load(dataset_folder / f"{split}.npz") as arrays:
        return {key: arrays[key] for key in arrays.files}


def build_prediction_table(
    metadata: pd.DataFrame,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    *,
    architecture: str,
    variant: str,
) -> pd.DataFrame:
    """Create a long-form raw-scale test prediction table."""

    rows = []
    for index, row in metadata.reset_index(drop=True).iterrows():
        for horizon_index in range(y_true_raw.shape[1]):
            observed = float(y_true_raw[index, horizon_index])
            predicted = float(y_pred_raw[index, horizon_index])
            rows.append(
                {
                    "window_id": row["window_id"],
                    "event_id": row["event_id"],
                    "dataset_id": row["dataset_id"],
                    "dataset_name": row["dataset_name"],
                    "quality_flag": row["quality_flag"],
                    "architecture": architecture,
                    "variant": variant,
                    "horizon_step": horizon_index + 1,
                    "y_true_raw": observed,
                    "y_pred_raw": predicted,
                    "absolute_error": abs(predicted - observed),
                    "squared_error": (predicted - observed) ** 2,
                }
            )
    return pd.DataFrame(rows)


def grouped_metrics(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    """Compute raw-scale metrics for each value of one metadata column."""

    rows = []
    for value, frame in predictions.groupby(column, sort=False):
        rows.append(
            {
                column: value,
                "mae": float(frame["absolute_error"].mean()),
                "rmse": float(np.sqrt(frame["squared_error"].mean())),
                "num_windows": frame["window_id"].nunique(),
                "num_events": frame["event_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def save_run_figures(
    history: pd.DataFrame,
    metadata: pd.DataFrame,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    figures_dir: Path,
) -> None:
    """Save compact optimization and raw-scale forecast diagnostics."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(history["epoch"], history["train_loss"], label="Train")
    axis.plot(history["epoch"], history["val_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_title("GRU training history")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "training_loss.png", dpi=150)
    plt.close(figure)

    sample_count = min(5, len(metadata))
    figure, axes = plt.subplots(sample_count, 1, figsize=(10, 3 * sample_count))
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(y_true_raw[index], label="Observed")
        axis.plot(y_pred_raw[index], label="Predicted")
        axis.set_title(str(metadata.iloc[index]["window_id"]))
        axis.set_xlabel("Horizon step")
        axis.set_ylabel("Signal")
        axis.legend()
    figure.tight_layout()
    examples_dir = figures_dir / "forecast_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(examples_dir / "test_forecast_examples.png", dpi=150)
    plt.close(figure)


def ensure_run_paths_are_available(
    model_dir: Path,
    result_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """Prevent silent replacement of an existing architecture/variant run."""

    if overwrite:
        return
    for folder in (model_dir, result_dir):
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Run output already exists: {folder}")


def main() -> None:
    """Run all configured GRU architecture and data-variant experiments."""

    config = load_yaml_config(CONFIG_PATH)
    if config["training"]["loss"] != "mse":
        raise ValueError("Only MSE training loss is currently supported.")
    if not config["evaluation"]["evaluate_in_raw_scale"]:
        raise ValueError("GRU baseline evaluation must run in raw scale.")
    if config["seq2seq"]["decoder_input_strategy"] != "last_observed_value":
        raise ValueError("Only last_observed_value decoder initialization is supported.")
    if not config["seq2seq"]["disable_teacher_forcing_eval"]:
        raise ValueError("Teacher forcing must be disabled during evaluation.")

    dataset_root = project_path(config["data"]["dataset_root"])
    dataset_folder = resolve_selection_folder(
        dataset_root,
        config["data"]["selection_folder"],
    )
    selection_folder = dataset_folder.name
    dataset_metadata = load_yaml_config(dataset_folder / "dataset_metadata.yaml")
    selection_id = selection_id_from_metadata(dataset_metadata)
    comparison_id = f"gru_architecture_variant_{selection_id}"
    comparison_dir = get_comparison_dir(comparison_id)
    comparison_paths = ensure_results_subdirs(
        comparison_dir,
        ("tables",),
    )
    comparison_path = (
        comparison_paths["tables"] / "gru_architecture_variant_comparison.csv"
    )
    if (
        comparison_path.exists()
        and not config["output"]["overwrite"]
    ):
        raise FileExistsError(f"Comparison output already exists: {comparison_path}")

    split_arrays = {
        split: load_split_arrays(dataset_folder, split)
        for split in ("train", "val", "test")
    }
    split_metadata = {
        split: pd.read_parquet(dataset_folder / f"{split}_metadata.parquet")
        for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        if len(split_metadata[split]) == 0:
            raise ValueError(f"The selected dataset has an empty {split} split.")

    device = torch.device(select_device())
    comparison_rows = []
    for architecture in config["experiment"]["architectures"]:
        for variant in config["experiment"]["variants"]:
            run_id = make_run_id("gru", architecture, variant, selection_id)
            model_dir = get_model_dir("gru", run_id)
            result_dir = get_run_dir(run_id)
            ensure_run_paths_are_available(
                model_dir,
                result_dir,
                overwrite=config["output"]["overwrite"],
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            result_paths = ensure_results_subdirs(
                result_dir,
                ("metrics", "predictions", "figures", "tables"),
            )
            set_seed(config["training"]["seed"])
            model = build_gru_forecaster(
                architecture,
                input_size=config["model"]["input_size"],
                hidden_size=config["model"]["hidden_size"],
                num_layers=config["model"]["num_layers"],
                prediction_length=config["data"]["prediction_length"],
                dropout=config["model"]["dropout"],
                bidirectional=config["model"]["bidirectional"],
            )
            loaders = {
                split: create_gru_data_loader(
                    dataset_folder / f"{split}.npz",
                    variant=variant,
                    batch_size=config["training"]["batch_size"],
                    shuffle=split == "train",
                    seed=config["training"]["seed"],
                )
                for split in ("train", "val", "test")
            }
            result = train_gru_forecaster(
                model,
                loaders["train"],
                loaders["val"],
                device=device,
                learning_rate=config["training"]["learning_rate"],
                max_epochs=config["training"]["max_epochs"],
                early_stopping_patience=config["training"][
                    "early_stopping_patience"
                ],
                gradient_clip_norm=config["training"]["gradient_clip_norm"],
                teacher_forcing_ratio=config["seq2seq"]["teacher_forcing_ratio"]
                if architecture == "gru_seq2seq"
                else 0.0,
            )
            prediction_variant, prediction_indices = predict_gru_forecaster(
                model,
                loaders["test"],
                device=device,
            )
            if not np.array_equal(prediction_indices, np.arange(len(prediction_indices))):
                raise ValueError("Test predictions are not aligned to metadata rows.")
            if variant == "context_standard":
                y_pred_raw = inverse_context_standardization(
                    prediction_variant,
                    split_arrays["test"]["scaling_mean"],
                    split_arrays["test"]["scaling_std"],
                )
            else:
                y_pred_raw = prediction_variant
            y_true_raw = split_arrays["test"]["y_raw"]
            metrics = compute_trajectory_metrics(y_true_raw, y_pred_raw)
            metrics_summary = pd.DataFrame(
                [
                    {
                        "architecture": architecture,
                        "variant": variant,
                        "test_mae": metrics["mae"],
                        "test_rmse": metrics["rmse"],
                        "best_epoch": result.best_epoch,
                        "best_val_loss": result.best_val_loss,
                        "num_train_windows": len(split_metadata["train"]),
                        "num_val_windows": len(split_metadata["val"]),
                        "num_test_windows": len(split_metadata["test"]),
                    }
                ]
            )
            horizon_metrics = compute_horizon_metrics(y_true_raw, y_pred_raw)
            horizon_metrics.insert(0, "variant", variant)
            horizon_metrics.insert(0, "architecture", architecture)
            predictions = build_prediction_table(
                split_metadata["test"],
                y_true_raw,
                y_pred_raw,
                architecture=architecture,
                variant=variant,
            )
            history = pd.DataFrame(result.history)

            metrics_summary.to_csv(
                result_paths["metrics"] / "metrics_summary.csv", index=False
            )
            horizon_metrics.to_csv(
                result_paths["metrics"] / "metrics_by_horizon.csv", index=False
            )
            grouped_metrics(predictions, "dataset_name").to_csv(
                result_paths["metrics"] / "metrics_by_dataset.csv",
                index=False,
            )
            grouped_metrics(predictions, "quality_flag").to_csv(
                result_paths["metrics"] / "metrics_by_quality_flag.csv",
                index=False,
            )
            history.to_csv(
                result_paths["metrics"] / "training_history.csv", index=False
            )
            if config["output"]["save_predictions"]:
                predictions.to_parquet(
                    result_paths["predictions"] / "test_predictions.parquet",
                    index=False,
                )
            if config["output"]["save_plots"]:
                save_run_figures(
                    history,
                    split_metadata["test"],
                    y_true_raw,
                    y_pred_raw,
                    result_paths["figures"],
                )

            model_config = {
                "architecture": architecture,
                "variant": variant,
                **config["model"],
                "prediction_length": config["data"]["prediction_length"],
            }
            torch.save(
                {
                    "model_state_dict": result.best_state_dict,
                    "model_config": model_config,
                },
                model_dir / "best_model.pt",
            )
            save_yaml(model_dir / "model_config.yaml", model_config)
            training_metadata = {
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "device": str(device),
                "seed": config["training"]["seed"],
                "teacher_forcing_ratio_training": config["seq2seq"][
                    "teacher_forcing_ratio"
                ]
                if architecture == "gru_seq2seq"
                else 0.0,
                "teacher_forcing_evaluation": 0.0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            save_yaml(model_dir / "training_metadata.yaml", training_metadata)
            save_yaml(
                result_dir / "metadata.yaml",
                {
                    "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
                    "config_fingerprint": config_fingerprint(config),
                    "selection_folder": selection_folder,
                    "selection_id": selection_id,
                    "run_id": run_id,
                    "architecture": architecture,
                    "variant": variant,
                    "device": str(device),
                    "metrics_in_raw_scale": True,
                    **training_metadata,
                },
            )
            upsert_index_row(
                get_results_index_dir() / "runs.csv",
                {
                    "run_id": run_id,
                    "model_family": "gru",
                    "architecture": architecture,
                    "variant": variant,
                    "selection_id": selection_id,
                    "dataset_selection": ";".join(
                        dataset_metadata["selected_datasets"]
                    ),
                    "threshold": dataset_metadata["threshold"],
                    "context_length": dataset_metadata["context_length"],
                    "prediction_length": dataset_metadata["prediction_length"],
                    "results_path": relative_project_path(result_dir),
                    "model_path": relative_project_path(model_dir),
                    "predictions_path": relative_project_path(
                        result_paths["predictions"] / "test_predictions.parquet"
                    ),
                    "metrics_path": relative_project_path(result_paths["metrics"]),
                    "status": "complete",
                    "created_at": training_metadata["created_at"],
                },
                id_column="run_id",
                columns=RUN_INDEX_COLUMNS,
            )
            comparison_rows.append(metrics_summary.iloc[0].to_dict())
            print(
                f"\nArchitecture: {architecture}\n"
                f"Variant: {variant}\n"
                f"Train windows: {len(split_metadata['train']):,}\n"
                f"Validation windows: {len(split_metadata['val']):,}\n"
                f"Test windows: {len(split_metadata['test']):,}\n"
                f"Best epoch: {result.best_epoch}\n"
                f"Best validation loss: {result.best_val_loss:.6g}\n"
                f"Test MAE raw scale: {metrics['mae']:.6g}\n"
                f"Test RMSE raw scale: {metrics['rmse']:.6g}\n"
                f"Output folder: {result_dir}"
            )

    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    save_yaml(
        comparison_dir / "metadata.yaml",
        {
            "comparison_id": comparison_id,
            "selection_id": selection_id,
            "comparison_type": "forecast_metrics_across_gru_runs",
            "table": relative_project_path(comparison_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    upsert_index_row(
        get_results_index_dir() / "comparisons.csv",
        {
            "comparison_id": comparison_id,
            "method_id": "multiple_gru_runs",
            "reference_id": "",
            "comparison_type": "forecast_metrics",
            "selection_id": selection_id,
            "results_path": relative_project_path(comparison_dir),
            "metrics_path": relative_project_path(comparison_paths["tables"]),
            "figures_path": "",
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        id_column="comparison_id",
        columns=COMPARISON_INDEX_COLUMNS,
    )
    print(f"\nSaved comparison table: {comparison_path}")


if __name__ == "__main__":
    main()
