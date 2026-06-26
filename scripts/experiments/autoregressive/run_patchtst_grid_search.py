"""Tune PatchTST variants and publish validation-selected canonical runs."""

import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
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
from src.tasks.autoregressive.evaluation.forecast_metrics import (
    compute_horizon_metrics,
    compute_trajectory_metrics,
    inverse_context_standardization,
)
from src.tasks.autoregressive.models.patchtst import PatchTSTForecaster
from src.tasks.autoregressive.models.patchtst_training import (
    PatchTSTForecastDataset,
    PatchTSTTrainingResult,
    create_patchtst_data_loader,
    predict_patchtst_forecaster,
    train_patchtst_forecaster,
    train_patchtst_forecaster_fixed_epochs,
)
from src.tuning.grid_search import (
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
from src.utils.paths import project_path
from src.utils.reproducibility import set_seed
from src.utils.results_paths import (
    COMPARISON_INDEX_COLUMNS,
    GRID_SEARCH_INDEX_COLUMNS,
    RUN_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_comparison_dir,
    get_grid_search_dir,
    get_model_dir,
    get_results_index_dir,
    get_run_dir,
    make_run_id,
    relative_project_path,
    selection_id_from_metadata,
    upsert_index_row,
)

CONFIG_PATH = PROJECT_ROOT / "configs/autoregressive/patchtst_grid_search.yaml"


def predictions_in_raw_scale(
    predictions: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    variant: str,
) -> np.ndarray:
    """Return PatchTST predictions on the original signal scale."""

    if variant == "raw":
        return predictions
    if variant == "context_standard":
        return inverse_context_standardization(
            predictions,
            arrays["scaling_mean"],
            arrays["scaling_std"],
        )
    raise ValueError(f"Unsupported PatchTST variant: {variant}")


def build_patchtst_model(model_config: dict) -> PatchTSTForecaster:
    """Build one PatchTST model from a complete model configuration."""

    return PatchTSTForecaster(
        context_length=int(model_config["context_length"]),
        prediction_length=int(model_config["prediction_length"]),
        input_channels=int(model_config["input_channels"]),
        patch_len=int(model_config["patch_len"]),
        stride=int(model_config["stride"]),
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
    )


def build_model_config(config: dict, parameters: dict, variant: str) -> dict:
    """Return the complete, serializable model configuration for one trial."""

    return {
        "architecture": "patchtst",
        "variant": variant,
        "context_length": int(config["data"]["context_length"]),
        "prediction_length": int(config["data"]["prediction_length"]),
        "input_channels": int(config["fixed_model"]["input_channels"]),
        **{
            name: parameters[name]
            for name in (
                "patch_len",
                "stride",
                "d_model",
                "n_heads",
                "num_layers",
                "dropout",
            )
        },
    }


def trial_display_parameters(config: dict, parameters: dict, variant: str) -> dict:
    """Return every effective parameter that defines one PatchTST trial."""

    return {
        "model": build_model_config(config, parameters, variant),
        "optimizer": {
            "name": "Adam",
            "learning_rate": parameters["learning_rate"],
            "weight_decay": parameters["weight_decay"],
        },
        "training": config["training"],
    }


def build_loaders(
    dataset_folder: Path,
    *,
    variant: str,
    training: dict,
    include_test: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    """Build deterministic PatchTST loaders."""

    splits = ("train", "val", "test") if include_test else ("train", "val")
    return {
        split: create_patchtst_data_loader(
            dataset_folder / f"{split}.npz",
            variant=variant,
            batch_size=int(training["batch_size"]),
            shuffle=split == "train",
            seed=int(training["seed"]),
        )
        for split in splits
    }


def build_full_development_loader(
    dataset_folder: Path,
    *,
    variant: str,
    training: dict,
) -> torch.utils.data.DataLoader:
    """Build a shuffled train+validation loader for final retraining."""

    generator = torch.Generator()
    generator.manual_seed(int(training["seed"]))
    dataset = torch.utils.data.ConcatDataset(
        [
            PatchTSTForecastDataset(dataset_folder / "train.npz", variant=variant),
            PatchTSTForecastDataset(dataset_folder / "val.npz", variant=variant),
        ]
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
    )


def train_trial(
    *,
    config: dict,
    parameters: dict,
    variant: str,
    dataset_folder: Path,
    validation_arrays: dict[str, np.ndarray],
    device: torch.device,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[PatchTSTTrainingResult, dict[str, float], dict]:
    """Train one trial and evaluate it only on raw-scale validation metrics."""

    set_seed(int(config["training"]["seed"]))
    model_config = build_model_config(config, parameters, variant)
    model = build_patchtst_model(model_config)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)
    loaders = build_loaders(
        dataset_folder,
        variant=variant,
        training=config["training"],
        include_test=False,
    )
    result = train_patchtst_forecaster(
        model,
        loaders["train"],
        loaders["val"],
        device=device,
        learning_rate=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
        max_epochs=int(config["training"]["max_epochs"]),
        early_stopping_patience=int(config["training"]["early_stopping_patience"]),
        gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
    )
    validation_variant, indices = predict_patchtst_forecaster(
        model,
        loaders["val"],
        device=device,
    )
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("Validation predictions are not aligned to metadata rows.")
    validation_raw = predictions_in_raw_scale(
        validation_variant,
        validation_arrays,
        variant=variant,
    )
    metrics = compute_trajectory_metrics(validation_arrays["y_raw"], validation_raw)
    return result, metrics, model_config


def evaluate_validation_checkpoint(
    *,
    result: PatchTSTTrainingResult,
    model_config: dict,
    variant: str,
    dataset_folder: Path,
    validation_arrays: dict[str, np.ndarray],
    training: dict,
    device: torch.device,
) -> dict[str, float]:
    """Recompute raw-scale validation metrics from a loaded best checkpoint."""

    model = build_patchtst_model(model_config)
    model.load_state_dict(result.best_state_dict)
    model.to(device)
    loader = build_loaders(
        dataset_folder,
        variant=variant,
        training=training,
        include_test=False,
    )["val"]
    prediction_variant, indices = predict_patchtst_forecaster(
        model,
        loader,
        device=device,
    )
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("Validation predictions are not aligned to metadata rows.")
    prediction_raw = predictions_in_raw_scale(
        prediction_variant,
        validation_arrays,
        variant=variant,
    )
    return compute_trajectory_metrics(validation_arrays["y_raw"], prediction_raw)


def trial_config_fingerprint(config: dict, parameters: dict, variant: str) -> str:
    """Fingerprint settings that affect one trial checkpoint."""

    training = {
        key: value
        for key, value in config["training"].items()
        if key
        not in {
            "skip_if_checkpoint_exists",
            "force_retrain",
            "resume_from_checkpoint",
        }
    }
    return config_fingerprint(
        {
            "variant": variant,
            "model_config": build_model_config(config, parameters, variant),
            "parameters": parameters,
            "training": training,
        }
    )


def save_trial_checkpoint(
    path: Path,
    *,
    trial_id: str,
    parameters: dict,
    model_config: dict,
    result: PatchTSTTrainingResult,
    validation_metrics: dict[str, float],
    trial_fingerprint: str,
) -> None:
    """Persist one validation-only trial for reuse without test information."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trial_id": trial_id,
            "parameters": parameters,
            "model_config": model_config,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "history": result.history,
            "best_state_dict": result.best_state_dict,
            "last_state_dict": result.last_state_dict,
            "validation_metrics": validation_metrics,
            "trial_fingerprint": trial_fingerprint,
        },
        path,
    )


def load_trial_checkpoint(
    path: Path,
) -> tuple[PatchTSTTrainingResult, dict, dict, dict, str]:
    """Load one trusted project-generated trial checkpoint."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    result = PatchTSTTrainingResult(
        best_epoch=int(checkpoint["best_epoch"]),
        best_val_loss=float(checkpoint["best_val_loss"]),
        history=list(checkpoint["history"]),
        best_state_dict=checkpoint["best_state_dict"],
        last_state_dict=checkpoint["last_state_dict"],
    )
    return (
        result,
        dict(checkpoint["validation_metrics"]),
        dict(checkpoint["model_config"]),
        dict(checkpoint["parameters"]),
        str(checkpoint["trial_fingerprint"]),
    )


def build_prediction_table(
    metadata: pd.DataFrame,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    *,
    variant: str,
    y_pred_variant: np.ndarray | None = None,
) -> pd.DataFrame:
    """Create the GRU-compatible long-form PatchTST prediction table."""

    metadata = metadata.reset_index(drop=True)
    rows = []
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
    for index, row in metadata.iterrows():
        for horizon_index in range(y_true_raw.shape[1]):
            observed = float(y_true_raw[index, horizon_index])
            predicted = float(y_pred_raw[index, horizon_index])
            output = {column: row[column] for column in copied_columns}
            output.update(
                {
                    "model_family": "patchtst",
                    "architecture": "patchtst",
                    "variant": variant,
                    "horizon_step": horizon_index + 1,
                    "y_true_raw": observed,
                    "y_pred_raw": predicted,
                    "error": predicted - observed,
                    "absolute_error": abs(predicted - observed),
                    "squared_error": (predicted - observed) ** 2,
                }
            )
            if variant == "context_standard" and y_pred_variant is not None:
                output["y_pred_normalized"] = float(
                    y_pred_variant[index, horizon_index]
                )
            rows.append(output)
    return pd.DataFrame(rows)


def save_run_figures(
    *,
    history: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    metadata: pd.DataFrame,
    X_raw: np.ndarray,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    figures_dir: Path,
) -> None:
    """Save optimization, horizon, example, and worst-forecast diagnostics."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(history["epoch"], history["train_loss"], label="Train")
    axis.plot(history["epoch"], history["val_loss"], label="Validation")
    axis.set(xlabel="Epoch", ylabel="MSE loss", title="PatchTST training history")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "training_loss.png", dpi=150)
    plt.close(figure)

    for metric in ("mae", "rmse"):
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(horizon_metrics["horizon_step"], horizon_metrics[metric], marker="o")
        axis.set(
            xlabel="Horizon step",
            ylabel=metric.upper(),
            title=f"PatchTST {metric.upper()} by horizon",
        )
        figure.tight_layout()
        figure.savefig(figures_dir / f"{metric}_by_horizon.png", dpi=150)
        plt.close(figure)

    errors = np.mean(np.abs(y_pred_raw - y_true_raw), axis=1)
    example_indices = np.arange(min(5, len(metadata)))
    worst_indices = np.argsort(errors)[-min(5, len(metadata)) :][::-1]
    _save_forecast_set(
        example_indices,
        metadata,
        X_raw,
        y_true_raw,
        y_pred_raw,
        figures_dir / "forecast_examples",
    )
    _save_forecast_set(
        worst_indices,
        metadata,
        X_raw,
        y_true_raw,
        y_pred_raw,
        figures_dir / "worst_forecasts",
    )


def _save_forecast_set(
    indices: np.ndarray,
    metadata: pd.DataFrame,
    X_raw: np.ndarray,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    context_length = X_raw.shape[1]
    past_steps = np.arange(-context_length + 1, 1)
    future_steps = np.arange(1, y_true_raw.shape[1] + 1)
    for rank, index in enumerate(indices, start=1):
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.plot(past_steps, X_raw[index, :, 0], label="Past context")
        axis.plot(future_steps, y_true_raw[index], label="True future")
        axis.plot(future_steps, y_pred_raw[index], label="Predicted future")
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


def save_canonical_run(
    *,
    config: dict,
    variant: str,
    run_id: str,
    search_id: str,
    best_trial: pd.Series,
    checkpoint_path: Path,
    dataset_folder: Path,
    dataset_metadata: dict,
    split_arrays: dict[str, dict[str, np.ndarray]],
    split_metadata: dict[str, pd.DataFrame],
    trials: pd.DataFrame,
    device: torch.device,
) -> dict:
    """Evaluate the validation-selected winner on test and publish artifacts."""

    result, _, model_config, _, _ = load_trial_checkpoint(checkpoint_path)
    model = build_patchtst_model(model_config)
    final_training = config.get("final_training", {})
    final_retraining_enabled = bool(
        final_training.get("retrain_on_full_development", False)
    )
    result_for_artifacts = result
    if final_retraining_enabled:
        if not final_training.get("use_best_epoch_from_grid", False):
            raise ValueError("Final retraining requires use_best_epoch_from_grid=true.")
        if final_training.get("early_stopping", True):
            raise ValueError("Final retraining must use early_stopping=false.")
        set_seed(int(config["training"]["seed"]))
        result_for_artifacts = train_patchtst_forecaster_fixed_epochs(
            model,
            build_full_development_loader(
                dataset_folder,
                variant=variant,
                training=config["training"],
            ),
            device=device,
            learning_rate=float(best_trial["learning_rate"]),
            weight_decay=float(best_trial["weight_decay"]),
            max_epochs=int(result.best_epoch),
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        )
    else:
        model.load_state_dict(result.best_state_dict)
    model.to(device)
    test_loader = build_loaders(
        dataset_folder,
        variant=variant,
        training=config["training"],
        include_test=True,
    )["test"]
    prediction_variant, indices = predict_patchtst_forecaster(
        model,
        test_loader,
        device=device,
    )
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("Test predictions are not aligned to metadata rows.")
    y_pred_raw = predictions_in_raw_scale(
        prediction_variant,
        split_arrays["test"],
        variant=variant,
    )
    y_true_raw = split_arrays["test"]["y_raw"]
    test_metrics = compute_trajectory_metrics(y_true_raw, y_pred_raw)

    model_dir = get_model_dir("patchtst", run_id)
    result_dir = get_run_dir(run_id)
    if config["output"]["overwrite"]:
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(result_dir, ignore_errors=True)
    elif model_dir.exists() or result_dir.exists():
        raise FileExistsError(f"Canonical PatchTST run already exists: {run_id}")
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = ensure_results_subdirs(
        result_dir,
        ("metrics", "predictions", "figures", "tables"),
    )

    metrics_summary = pd.DataFrame(
        [
            {
                "architecture": "patchtst",
                "variant": variant,
                "validation_mae_raw": float(best_trial["validation_mae_raw"]),
                "validation_rmse_raw": float(best_trial["validation_rmse_raw"]),
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "best_epoch": result.best_epoch,
                "final_training_epochs": result_for_artifacts.best_epoch,
                "final_retrained_on_full_development": final_retraining_enabled,
                "best_val_loss": result.best_val_loss,
                **{
                    name: best_trial[name]
                    for name in config["parameter_grid"]
                },
                "num_train_windows": len(split_metadata["train"]),
                "num_val_windows": len(split_metadata["val"]),
                "num_test_windows": len(split_metadata["test"]),
            }
        ]
    )
    horizon_metrics = compute_horizon_metrics(y_true_raw, y_pred_raw)
    horizon_metrics.insert(0, "variant", variant)
    horizon_metrics.insert(0, "architecture", "patchtst")
    predictions = build_prediction_table(
        split_metadata["test"],
        y_true_raw,
        y_pred_raw,
        variant=variant,
        y_pred_variant=prediction_variant,
    )
    history = pd.DataFrame(result_for_artifacts.history)
    metrics_summary.to_csv(paths["metrics"] / "metrics_summary.csv", index=False)
    horizon_metrics.to_csv(paths["metrics"] / "metrics_by_horizon.csv", index=False)
    grouped_metrics(predictions, "dataset_name").to_csv(
        paths["metrics"] / "metrics_by_dataset.csv", index=False
    )
    grouped_metrics(predictions, "quality_flag").to_csv(
        paths["metrics"] / "metrics_by_quality_flag.csv", index=False
    )
    grouped_metrics(predictions, "event_id").to_csv(
        paths["metrics"] / "metrics_by_event.csv", index=False
    )
    history.to_csv(paths["metrics"] / "training_history.csv", index=False)
    if config["output"]["save_best_predictions"]:
        predictions.to_parquet(
            paths["predictions"] / "test_predictions.parquet",
            index=False,
        )
    if config["output"]["save_best_plots"]:
        save_run_figures(
            history=history,
            horizon_metrics=horizon_metrics,
            metadata=split_metadata["test"],
            X_raw=split_arrays["test"]["X_raw"],
            y_true_raw=y_true_raw,
            y_pred_raw=y_pred_raw,
            figures_dir=paths["figures"],
        )

    ranking = trials.sort_values(
        ["validation_rmse_raw", "trial_id"],
        kind="stable",
    )
    trials.to_csv(paths["tables"] / "trial_results.csv", index=False)
    ranking.to_csv(paths["tables"] / "trial_ranking.csv", index=False)
    metrics_summary.to_csv(paths["tables"] / "best_trial_summary.csv", index=False)

    torch.save(
        {
            "model_state_dict": result_for_artifacts.best_state_dict,
            "model_config": model_config,
            "grid_search_id": search_id,
            "best_trial_id": best_trial["trial_id"],
            "final_retrained_on_full_development": final_retraining_enabled,
        },
        model_dir / "best_model.pt",
    )
    torch.save(
        {
            "model_state_dict": result_for_artifacts.last_state_dict,
            "model_config": model_config,
            "grid_search_id": search_id,
            "best_trial_id": best_trial["trial_id"],
            "final_retrained_on_full_development": final_retraining_enabled,
        },
        model_dir / "last_model.pt",
    )
    save_yaml(model_dir / "model_config.yaml", model_config)
    created_at = datetime.now(timezone.utc).isoformat()
    training_metadata = {
        "grid_search_id": search_id,
        "best_trial_id": best_trial["trial_id"],
        "selection_metric": config["search"]["selection_metric"],
        "selection_metric_value": float(best_trial["validation_rmse_raw"]),
        "best_epoch": result.best_epoch,
        "final_training": {
            "retrain_on_full_development": final_retraining_enabled,
            "epochs": int(result_for_artifacts.best_epoch),
            "use_best_epoch_from_grid": bool(
                final_training.get("use_best_epoch_from_grid", False)
            ),
            "early_stopping": bool(final_training.get("early_stopping", False)),
            "train_split": "train+val" if final_retraining_enabled else "train",
            "excluded_split": "test",
        },
        "best_val_loss": result.best_val_loss,
        "learning_rate": float(best_trial["learning_rate"]),
        "weight_decay": float(best_trial["weight_decay"]),
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
            "selection_id": selection_id_from_metadata(dataset_metadata),
            "architecture": "patchtst",
            "variant": variant,
            "metrics_in_raw_scale": True,
            "test_type": dataset_metadata.get("test_type"),
            "external_test_dataset": dataset_metadata.get("external_test_dataset"),
            "selection_mode": dataset_metadata.get("selection_mode"),
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
            "selection_id": selection_id_from_metadata(dataset_metadata),
            "dataset_selection": ";".join(dataset_metadata["selected_datasets"]),
            "threshold": dataset_metadata["threshold"],
            "context_length": dataset_metadata["context_length"],
            "prediction_length": dataset_metadata["prediction_length"],
            "results_path": relative_project_path(result_dir),
            "model_path": relative_project_path(model_dir),
            "predictions_path": relative_project_path(
                paths["predictions"] / "test_predictions.parquet"
            ),
            "metrics_path": relative_project_path(paths["metrics"]),
            "status": "complete",
            "created_at": created_at,
        },
        id_column="run_id",
        columns=RUN_INDEX_COLUMNS,
    )
    return metrics_summary.iloc[0].to_dict()


def validate_config(config: dict, num_candidates: int) -> None:
    """Validate search semantics before creating or replacing any artifact."""

    if config["search"]["selection_metric"] != "validation_rmse_raw":
        raise ValueError("PatchTST selection must use validation_rmse_raw.")
    if config["search"]["selection_mode"] != "min":
        raise ValueError("validation_rmse_raw must be minimized.")
    if config["training"]["loss"] != "mse":
        raise ValueError("Only MSE PatchTST training loss is supported.")
    if int(config["fixed_model"]["output_length"]) != int(
        config["data"]["prediction_length"]
    ):
        raise ValueError("fixed_model.output_length must equal prediction_length.")
    max_trials = config["search"].get("max_trials")
    if max_trials is not None and num_candidates > int(max_trials):
        raise ValueError(
            f"Configured PatchTST grid has {num_candidates} trials per variant, "
            f"exceeding search.max_trials={max_trials}. Reduce the grid or "
            "explicitly increase max_trials; no combinations were skipped."
        )


def run_patchtst_trial_job(job: dict) -> dict:
    """Train or reuse one PatchTST trial on its assigned worker device."""

    config = job["config"]
    parameters = job["parameters"]
    variant = job["variant"]
    trial_id = job["trial_id"]
    checkpoint_path = Path(job["checkpoint_path"])
    history_path = Path(job["history_path"])
    search_dir = Path(job["search_dir"])
    dataset_folder = Path(job["dataset_folder"])
    device = prepare_trial_device(job["device"])
    expected_fingerprint = trial_config_fingerprint(config, parameters, variant)
    start = time.perf_counter()
    displayed = trial_display_parameters(config, parameters, variant)
    print(
        format_trial_start(
            index=int(job["index"]),
            total=int(job["total"]),
            run_id=job["target_run_id"],
            trial_id=trial_id,
            device=device,
            parameters=displayed,
        ),
        flush=True,
    )
    try:
        validation_arrays = load_split_arrays(dataset_folder, "val")
        reused = False
        initial_state = None
        if checkpoint_path.exists():
            if config["training"]["force_retrain"]:
                pass
            elif config["training"]["resume_from_checkpoint"]:
                previous, _, _, saved_parameters, saved_fingerprint = (
                    load_trial_checkpoint(checkpoint_path)
                )
                if (
                    saved_parameters != parameters
                    or saved_fingerprint != expected_fingerprint
                ):
                    raise ValueError("Existing checkpoint configuration differs.")
                initial_state = previous.last_state_dict
            elif config["training"]["skip_if_checkpoint_exists"]:
                result, _, model_config, saved_parameters, saved_fingerprint = (
                    load_trial_checkpoint(checkpoint_path)
                )
                if (
                    saved_parameters != parameters
                    or saved_fingerprint != expected_fingerprint
                ):
                    raise ValueError("Existing checkpoint configuration differs.")
                metrics = evaluate_validation_checkpoint(
                    result=result,
                    model_config=model_config,
                    variant=variant,
                    dataset_folder=dataset_folder,
                    validation_arrays=validation_arrays,
                    training=config["training"],
                    device=device,
                )
                reused = True
            else:
                raise FileExistsError(
                    f"Trial checkpoint already exists: {checkpoint_path}"
                )
        if not reused:
            result, metrics, model_config = train_trial(
                config=config,
                parameters=parameters,
                variant=variant,
                dataset_folder=dataset_folder,
                validation_arrays=validation_arrays,
                device=device,
                initial_state_dict=initial_state,
            )
            save_trial_checkpoint(
                checkpoint_path,
                trial_id=trial_id,
                parameters=parameters,
                model_config=model_config,
                result=result,
                validation_metrics=metrics,
                trial_fingerprint=expected_fingerprint,
            )
        if config["output"]["save_trial_histories"]:
            pd.DataFrame(result.history).to_csv(history_path, index=False)
        row = {
            "trial_id": trial_id,
            "status": "complete",
            "variant": variant,
            **parameters,
            "best_epoch": result.best_epoch,
            "validation_mae_raw": metrics["mae"],
            "validation_rmse_raw": metrics["rmse"],
            "test_mae_raw": np.nan,
            "test_rmse_raw": np.nan,
            "checkpoint_path": relative_project_path(checkpoint_path),
            "model_path": relative_project_path(checkpoint_path),
            "results_path": relative_project_path(search_dir),
            "duration_seconds": time.perf_counter() - start,
            "checkpoint_reused": reused,
            "device": str(device),
            "error": "",
        }
        print(
            f"  completed {trial_id} device={device} "
            f"validation RMSE={metrics['rmse']:.6g} epoch={result.best_epoch} "
            f"reused={reused}",
            flush=True,
        )
        return row
    except Exception as error:
        print(f"  failed {trial_id} device={device}: {error}", flush=True)
        return {
            "trial_id": trial_id,
            "status": "failed",
            "variant": variant,
            **parameters,
            "best_epoch": np.nan,
            "validation_mae_raw": np.nan,
            "validation_rmse_raw": np.nan,
            "test_mae_raw": np.nan,
            "test_rmse_raw": np.nan,
            "checkpoint_path": relative_project_path(checkpoint_path),
            "model_path": relative_project_path(checkpoint_path),
            "results_path": relative_project_path(search_dir),
            "duration_seconds": time.perf_counter() - start,
            "checkpoint_reused": False,
            "device": str(device),
            "error": repr(error),
        }


def main() -> None:
    """Run the complete validation-only PatchTST search and publish winners."""

    config = load_yaml_config(CONFIG_PATH)
    candidates = expand_parameter_grid(config["parameter_grid"])
    total_trials = len(candidates) * len(config["search"]["variants"])
    print(
        "=== PatchTST grid search ===\n"
        f"Trials per variant: {len(candidates)}\n"
        f"Variants: {len(config['search']['variants'])}\n"
        f"Total training trials: {total_trials}\n"
        "Selection uses validation_rmse_raw only; test is evaluated after selection.",
        flush=True,
    )
    validate_config(config, len(candidates))

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
    devices = choose_trial_devices(config.get("parallel", {}))
    device = torch.device(devices[0])
    print(f"Trial worker devices: {devices}", flush=True)
    best_rows = []

    for variant in config["search"]["variants"]:
        run_id = make_run_id("patchtst", "patchtst", variant, selection_id)
        search_id = f"grid_{run_id}"
        search_dir = get_grid_search_dir(search_id)
        checkpoints_dir = search_dir / "checkpoints"
        histories_dir = search_dir / "validation_histories"
        search_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)
        histories_dir.mkdir(exist_ok=True)
        trial_rows = []
        print(f"=== {run_id}: {len(candidates)} trials ===", flush=True)

        jobs = [
            {
                "config": config,
                "parameters": parameters,
                "variant": variant,
                "target_run_id": run_id,
                "trial_id": make_trial_id(parameters, index),
                "index": index,
                "total": len(candidates),
                "checkpoint_path": str(
                    checkpoints_dir / f"{make_trial_id(parameters, index)}.pt"
                ),
                "history_path": str(
                    histories_dir / f"{make_trial_id(parameters, index)}.csv"
                ),
                "search_dir": str(search_dir),
                "dataset_folder": str(dataset_folder),
            }
            for index, parameters in enumerate(candidates, start=1)
        ]
        for row in iter_parallel_trial_results(
            jobs,
            run_patchtst_trial_job,
            devices,
        ):
            trial_rows.append(row)
            pd.DataFrame(trial_rows).to_csv(
                search_dir / "trial_results.csv",
                index=False,
            )

        trials = (
            pd.DataFrame(trial_rows)
            .sort_values("trial_id", kind="stable")
            .reset_index(drop=True)
        )
        best_trial = select_best_trial(
            trials,
            metric=config["search"]["selection_metric"],
            mode=config["search"]["selection_mode"],
        )
        best_checkpoint = PROJECT_ROOT / str(best_trial["checkpoint_path"])
        if config["output"]["update_canonical_runs"]:
            best_row = save_canonical_run(
                config=config,
                variant=variant,
                run_id=run_id,
                search_id=search_id,
                best_trial=best_trial,
                checkpoint_path=best_checkpoint,
                dataset_folder=dataset_folder,
                dataset_metadata=dataset_metadata,
                split_arrays=split_arrays,
                split_metadata=split_metadata,
                trials=trials,
                device=device,
            )
        else:
            best_row = {
                "architecture": "patchtst",
                "variant": variant,
                "validation_mae_raw": float(best_trial["validation_mae_raw"]),
                "validation_rmse_raw": float(best_trial["validation_rmse_raw"]),
                "test_mae": np.nan,
                "test_rmse": np.nan,
                **{name: best_trial[name] for name in config["parameter_grid"]},
            }
        best_rows.append(best_row)
        trials.loc[
            trials["trial_id"].eq(best_trial["trial_id"]),
            ["test_mae_raw", "test_rmse_raw"],
        ] = [best_row["test_mae"], best_row["test_rmse"]]
        trials.to_csv(search_dir / "trial_results.csv", index=False)
        trials.sort_values(
            ["validation_rmse_raw", "trial_id"],
            kind="stable",
        ).to_csv(search_dir / "trial_ranking.csv", index=False)
        pd.DataFrame([best_row]).to_csv(
            search_dir / "best_trial_summary.csv",
            index=False,
        )
        if config["output"]["update_canonical_runs"]:
            canonical_tables = get_run_dir(run_id) / "tables"
            trials.to_csv(canonical_tables / "trial_results.csv", index=False)
            trials.sort_values(
                ["validation_rmse_raw", "trial_id"],
                kind="stable",
            ).to_csv(canonical_tables / "trial_ranking.csv", index=False)
        save_yaml(
            search_dir / "best_params.yaml",
            {
                "search_id": search_id,
                "target_run_id": run_id,
                "best_trial_id": best_trial["trial_id"],
                "selection_metric": config["search"]["selection_metric"],
                "selection_metric_value": float(best_trial["validation_rmse_raw"]),
                "parameters": {
                    name: best_trial[name].item()
                    if isinstance(best_trial[name], np.generic)
                    else best_trial[name]
                    for name in config["parameter_grid"]
                },
            },
        )
        created_at = datetime.now(timezone.utc).isoformat()
        save_yaml(
            search_dir / "metadata.yaml",
            {
                "search_id": search_id,
                "model_family": "patchtst",
                "target_run_id": run_id,
                "selection_id": selection_id,
                "variant": variant,
                "num_trials": len(trials),
                "num_complete_trials": int(trials["status"].eq("complete").sum()),
                "selection_split": "validation",
                "test_used_for_selection": False,
                "trial_worker_devices": devices,
                "config_path": relative_project_path(CONFIG_PATH),
                "config_fingerprint": config_fingerprint(config),
                "created_at": created_at,
            },
        )
        upsert_index_row(
            get_results_index_dir() / "grid_searches.csv",
            {
                "search_id": search_id,
                "model_family": "patchtst",
                "target_run_id": run_id,
                "selection_id": selection_id,
                "selection_metric": config["search"]["selection_metric"],
                "best_trial_id": best_trial["trial_id"],
                "results_path": relative_project_path(search_dir),
                "best_model_path": relative_project_path(
                    get_model_dir("patchtst", run_id) / "best_model.pt"
                    if config["output"]["update_canonical_runs"]
                    else best_checkpoint
                ),
                "status": "complete",
                "created_at": created_at,
            },
            id_column="search_id",
            columns=GRID_SEARCH_INDEX_COLUMNS,
        )

    comparison_id = f"patchtst_search_{selection_id}"
    comparison_dir = get_comparison_dir(comparison_id)
    comparison_paths = ensure_results_subdirs(comparison_dir, ("tables",))
    comparison_path = (
        comparison_paths["tables"] / "patchtst_variant_comparison.csv"
    )
    pd.DataFrame(best_rows).to_csv(comparison_path, index=False)
    created_at = datetime.now(timezone.utc).isoformat()
    save_yaml(
        comparison_dir / "metadata.yaml",
        {
            "comparison_id": comparison_id,
            "selection_id": selection_id,
            "comparison_type": "forecast_metrics_across_tuned_patchtst_runs",
            "table": relative_project_path(comparison_path),
            "created_at": created_at,
        },
    )
    upsert_index_row(
        get_results_index_dir() / "comparisons.csv",
        {
            "comparison_id": comparison_id,
            "method_id": "multiple_tuned_patchtst_runs",
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
    print("=== Compact PatchTST ranking ===", flush=True)
    print(
        pd.DataFrame(best_rows)
        .sort_values("validation_rmse_raw")
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
