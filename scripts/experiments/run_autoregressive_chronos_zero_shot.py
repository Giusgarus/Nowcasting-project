"""Evaluate pretrained Chronos zero-shot forecasts on the final test dataset."""

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.forecast_metrics import (
    compute_horizon_metrics,
    compute_trajectory_metrics,
)
from src.models.autoregressive.chronos_adapter import (
    SUPPORTED_VARIANTS,
    build_chronos_prediction_tables,
    load_chronos_pipeline,
    predict_chronos_batches,
    predictions_to_raw_scale,
    resolve_chronos_device,
    variant_contexts_and_targets,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.paths import project_path
from src.utils.reproducibility import set_seed
from src.utils.results_paths import (
    RUN_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_model_dir,
    get_results_index_dir,
    get_run_dir,
    make_run_id,
    relative_project_path,
    selection_id_from_metadata,
    upsert_index_row,
)

CONFIG_PATH = PROJECT_ROOT / "configs/autoregressive_chronos.yaml"


def resolve_selection_folder(dataset_root: Path, configured: str) -> Path:
    """Resolve an explicit final-dataset selection or the only available one."""

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


def load_test_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load only the immutable final test NPZ."""

    with np.load(path) as arrays:
        return {key: arrays[key] for key in arrays.files}


def validate_config(config: dict) -> None:
    """Validate zero-shot-only Chronos evaluation semantics."""

    if config["model"]["mode"] != "zero_shot":
        raise ValueError("Chronos evaluation mode must be zero_shot.")
    if config["data"]["evaluate_split"] != "test":
        raise ValueError("Chronos zero-shot evaluation currently requires test split.")
    if config["chronos"]["forecast_mode"] != "median":
        raise ValueError("Chronos deterministic metrics currently require median mode.")
    if not config["evaluation"]["evaluate_in_raw_scale"]:
        raise ValueError("Chronos metrics must be evaluated in raw signal scale.")
    unsupported = sorted(set(config["variants"]).difference(SUPPORTED_VARIANTS))
    if unsupported:
        raise ValueError(f"Unsupported Chronos variants: {unsupported}")
    if not config["variants"]:
        raise ValueError("At least one Chronos variant must be configured.")
    quantiles = [float(value) for value in config["chronos"]["quantiles"]]
    if 0.5 not in quantiles:
        raise ValueError("Chronos quantiles must include 0.5 for median forecasts.")


def grouped_metrics(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    """Compute raw-scale MAE and RMSE for each metadata group."""

    rows = []
    for value, frame in predictions.groupby(column, sort=False):
        rows.append(
            {
                column: value,
                "mae": float(frame["absolute_error"].mean()),
                "rmse": float(np.sqrt(frame["squared_error"].mean())),
                "num_windows": int(frame["window_id"].nunique()),
                "num_events": int(frame["event_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def save_horizon_plots(
    metrics: pd.DataFrame,
    figures_dir: Path,
    plots: dict,
) -> None:
    """Save configured raw-scale horizon metric plots."""

    for metric in ("mae", "rmse"):
        if not plots[f"plot_{metric}_by_horizon"]:
            continue
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(metrics["horizon_step"], metrics[metric], marker="o")
        axis.set(
            xlabel="Horizon step",
            ylabel=metric.upper(),
            title=f"Chronos zero-shot {metric.upper()} by horizon",
        )
        figure.tight_layout()
        figure.savefig(figures_dir / f"{metric}_by_horizon.png", dpi=150)
        plt.close(figure)


def save_forecast_plots(
    metadata: pd.DataFrame,
    X_raw: np.ndarray,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    quantiles_raw: dict[float, np.ndarray],
    figures_dir: Path,
    plots: dict,
    *,
    seed: int,
) -> None:
    """Save configured raw-scale random and worst Chronos forecasts."""

    errors = np.mean(np.abs(y_pred_raw - y_true_raw), axis=1)
    rng = np.random.default_rng(seed)
    random_count = min(int(plots["num_random_forecast_plots"]), len(metadata))
    random_indices = rng.choice(len(metadata), size=random_count, replace=False)
    worst_count = min(int(plots["num_worst_forecast_plots"]), len(metadata))
    worst_indices = np.argsort(errors)[-worst_count:][::-1]
    if plots["plot_forecast_examples"]:
        _save_forecast_set(
            random_indices,
            metadata,
            X_raw,
            y_true_raw,
            y_pred_raw,
            quantiles_raw,
            figures_dir / "forecast_examples",
        )
    if plots["plot_worst_forecasts"]:
        _save_forecast_set(
            worst_indices,
            metadata,
            X_raw,
            y_true_raw,
            y_pred_raw,
            quantiles_raw,
            figures_dir / "worst_forecasts",
        )


def _save_forecast_set(
    indices: np.ndarray,
    metadata: pd.DataFrame,
    X_raw: np.ndarray,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    quantiles_raw: dict[float, np.ndarray],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    context_steps = np.arange(-X_raw.shape[1] + 1, 1)
    future_steps = np.arange(1, y_true_raw.shape[1] + 1)
    lower = quantiles_raw.get(0.1)
    upper = quantiles_raw.get(0.9)
    for rank, index in enumerate(indices, start=1):
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.plot(context_steps, X_raw[index, :, 0], label="Past context")
        axis.plot(future_steps, y_true_raw[index], label="True future")
        axis.plot(future_steps, y_pred_raw[index], label="Chronos median forecast")
        if lower is not None and upper is not None:
            axis.fill_between(
                future_steps,
                lower[index],
                upper[index],
                alpha=0.2,
                label="Chronos q10-q90",
            )
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set(
            xlabel="Relative step",
            ylabel="Signal",
            title=str(metadata.iloc[index]["window_id"]),
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"forecast_{rank:02d}.png", dpi=150)
        plt.close(figure)


def main() -> None:
    """Run Chronos zero-shot inference without loading train or validation data."""

    config = load_yaml_config(CONFIG_PATH)
    validate_config(config)
    set_seed(int(config["seed"]))
    dataset_folder = resolve_selection_folder(
        project_path(config["data"]["dataset_root"]),
        config["data"]["selection_folder"],
    )
    dataset_metadata = load_yaml_config(dataset_folder / "dataset_metadata.yaml")
    selection_id = selection_id_from_metadata(dataset_metadata)
    if int(dataset_metadata["context_length"]) != int(config["data"]["context_length"]):
        raise ValueError("Chronos context_length differs from final dataset metadata.")
    if int(dataset_metadata["prediction_length"]) != int(
        config["data"]["prediction_length"]
    ):
        raise ValueError("Chronos prediction_length differs from final dataset metadata.")

    arrays = load_test_arrays(dataset_folder / "test.npz")
    metadata = pd.read_parquet(dataset_folder / "test_metadata.parquet")
    if metadata.empty:
        raise ValueError("The selected final dataset has an empty test split.")
    requested_device = resolve_chronos_device(config["model"]["device"])
    print(
        "=== Chronos zero-shot evaluation ===\n"
        f"Model: {config['model']['model_id']}\n"
        f"Selection: {selection_id}\n"
        f"Test windows: {len(metadata):,}\n"
        f"Variants: {config['variants']}\n"
        f"Requested device: {requested_device}\n"
        "No train or validation data will be loaded.",
        flush=True,
    )
    pipeline, actual_device = load_chronos_pipeline(
        config["model"]["model_id"],
        device=requested_device,
        torch_dtype=config["model"]["torch_dtype"],
    )
    if actual_device != requested_device:
        print(
            f"Chronos could not use {requested_device}; falling back to {actual_device}.",
            flush=True,
        )

    for variant in config["variants"]:
        set_seed(int(config["seed"]))
        run_id = make_run_id(
            "chronos",
            "chronos_zero_shot",
            variant,
            selection_id,
        )
        result_dir = get_run_dir(run_id)
        model_dir = get_model_dir("chronos", run_id)
        if config["output"]["overwrite"]:
            shutil.rmtree(result_dir, ignore_errors=True)
            shutil.rmtree(model_dir, ignore_errors=True)
        elif any(path.exists() and any(path.iterdir()) for path in (result_dir, model_dir)):
            raise FileExistsError(f"Chronos zero-shot run already exists: {run_id}")
        paths = ensure_results_subdirs(
            result_dir,
            ("metrics", "predictions", "figures", "tables"),
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        contexts, _ = variant_contexts_and_targets(arrays, variant)
        prediction_kwargs = {
            "prediction_length": int(config["data"]["prediction_length"]),
            "num_samples": int(config["chronos"]["num_samples"]),
            "batch_size": int(config["chronos"]["batch_size"]),
            "quantiles": config["chronos"]["quantiles"],
        }
        try:
            prediction_variant, quantiles_variant = predict_chronos_batches(
                pipeline,
                contexts,
                **prediction_kwargs,
            )
        except Exception as error:
            if actual_device != "mps":
                raise RuntimeError(
                    f"Chronos zero-shot inference failed for variant {variant}: {error}"
                ) from error
            print(
                f"Chronos inference failed on MPS ({error}); retrying on CPU.",
                flush=True,
            )
            pipeline, actual_device = load_chronos_pipeline(
                config["model"]["model_id"],
                device="cpu",
                torch_dtype=config["model"]["torch_dtype"],
            )
            prediction_variant, quantiles_variant = predict_chronos_batches(
                pipeline,
                contexts,
                **prediction_kwargs,
            )
        prediction_raw = predictions_to_raw_scale(
            prediction_variant,
            arrays,
            variant=variant,
        )
        quantiles_raw = {
            quantile: predictions_to_raw_scale(values, arrays, variant=variant)
            for quantile, values in quantiles_variant.items()
        }
        y_true_raw = arrays["y_raw"]
        metrics = compute_trajectory_metrics(y_true_raw, prediction_raw)
        horizon_metrics = compute_horizon_metrics(y_true_raw, prediction_raw)
        horizon_metrics.insert(0, "variant", variant)
        horizon_metrics.insert(0, "architecture", "chronos_zero_shot")
        summary = pd.DataFrame(
            [
                {
                    "model_family": "chronos",
                    "architecture": "chronos_zero_shot",
                    "variant": variant,
                    "mode": "zero_shot",
                    "split": "test",
                    "test_mae": metrics["mae"],
                    "test_rmse": metrics["rmse"],
                    "num_test_windows": len(metadata),
                    "num_samples": int(config["chronos"]["num_samples"]),
                    "forecast_mode": config["chronos"]["forecast_mode"],
                }
            ]
        )
        predictions, wide_predictions = build_chronos_prediction_tables(
            metadata,
            y_true_raw,
            prediction_raw,
            quantiles_raw,
            variant=variant,
        )
        summary.to_csv(paths["metrics"] / "metrics_summary.csv", index=False)
        horizon_metrics.to_csv(
            paths["metrics"] / "metrics_by_horizon.csv",
            index=False,
        )
        for group in config["evaluation"]["groupby_metrics"]:
            grouped_metrics(predictions, group).to_csv(
                paths["metrics"] / f"metrics_by_{group}.csv",
                index=False,
            )
        summary.to_csv(paths["tables"] / "run_summary.csv", index=False)
        wide_predictions.to_parquet(
            paths["tables"] / "test_predictions_wide.parquet",
            index=False,
        )
        predictions_path = paths["predictions"] / "test_predictions.parquet"
        if config["output"]["save_predictions"]:
            predictions.to_parquet(predictions_path, index=False)
        if config["output"]["save_plots"]:
            save_horizon_plots(horizon_metrics, paths["figures"], config["plots"])
            save_forecast_plots(
                metadata,
                arrays["X_raw"],
                y_true_raw,
                prediction_raw,
                quantiles_raw,
                paths["figures"],
                config["plots"],
                seed=int(config["seed"]),
            )

        created_at = datetime.now(timezone.utc).isoformat()
        model_metadata = {
            "model_family": "chronos",
            "architecture": "chronos_zero_shot",
            "mode": "zero_shot",
            "model_id": config["model"]["model_id"],
            "trained_or_fine_tuned": False,
            "requested_device": requested_device,
            "actual_device": actual_device,
            "torch_dtype": config["model"]["torch_dtype"],
            "created_at": created_at,
        }
        save_yaml(model_dir / "model_config.yaml", model_metadata)
        save_yaml(
            result_dir / "metadata.yaml",
            {
                "run_id": run_id,
                "selection_id": selection_id,
                "variant": variant,
                "split": "test",
                "metrics_in_raw_scale": True,
                "test_only_zero_shot_evaluation": True,
                "train_data_loaded": False,
                "validation_data_loaded": False,
                "config_path": relative_project_path(CONFIG_PATH),
                "config_fingerprint": config_fingerprint(config),
                "dataset_folder": relative_project_path(dataset_folder),
                "wide_predictions_path": relative_project_path(
                    paths["tables"] / "test_predictions_wide.parquet"
                ),
                **model_metadata,
            },
        )
        upsert_index_row(
            get_results_index_dir() / "runs.csv",
            {
                "run_id": run_id,
                "model_family": "chronos",
                "architecture": "chronos_zero_shot",
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
                "metrics_path": relative_project_path(paths["metrics"]),
                "status": "complete",
                "created_at": created_at,
            },
            id_column="run_id",
            columns=RUN_INDEX_COLUMNS,
        )
        print(
            f"Completed {run_id}: test MAE={metrics['mae']:.6g}, "
            f"test RMSE={metrics['rmse']:.6g}\n"
            f"Predictions: {predictions_path}\n"
            f"Metrics: {paths['metrics']}\n"
            f"Figures: {paths['figures']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
