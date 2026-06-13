"""Tune all configured GRU runs using validation-only model selection."""

import json
import shutil
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_autoregressive_gru import (
    build_prediction_table,
    grouped_metrics,
    load_split_arrays,
    resolve_selection_folder,
    save_run_figures,
)
from src.evaluation.forecast_metrics import (
    compute_horizon_metrics,
    compute_trajectory_metrics,
    inverse_context_standardization,
)
from src.models.autoregressive.gru import build_gru_forecaster
from src.models.autoregressive.gru_training import (
    GRUTrainingResult,
    create_gru_data_loader,
    predict_gru_forecaster,
    train_gru_forecaster,
)
from src.tuning.grid_search import (
    expand_parameter_grid,
    make_trial_id,
    select_best_trial,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.device import select_device
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

CONFIG_PATH = PROJECT_ROOT / "configs/gru_grid_search.yaml"


def python_scalar(value):
    """Convert NumPy scalar values to YAML-safe Python scalars."""

    return value.item() if isinstance(value, np.generic) else value


def predictions_in_raw_scale(
    prediction_variant: np.ndarray,
    split_arrays: dict[str, np.ndarray],
    *,
    variant: str,
) -> np.ndarray:
    """Return variant predictions on the original signal scale."""

    if variant == "context_standard":
        return inverse_context_standardization(
            prediction_variant,
            split_arrays["scaling_mean"],
            split_arrays["scaling_std"],
        )
    if variant == "raw":
        return prediction_variant
    raise ValueError(f"Unsupported GRU data variant: {variant}")


def build_gru_trial_candidates(config: dict, architecture: str) -> list[dict]:
    """Expand shared parameters and conditional seq2seq teacher forcing."""

    if architecture not in {"gru_s2v", "gru_seq2seq"}:
        raise ValueError(f"Unsupported GRU architecture: {architecture}")
    candidates = expand_parameter_grid(config["parameter_grid"])
    if architecture != "gru_seq2seq":
        return [
            {**parameters, "teacher_forcing_ratio": 0.0}
            for parameters in candidates
        ]

    teacher_forcing = config["seq2seq"]["teacher_forcing_ratio"]
    if (
        isinstance(teacher_forcing, (str, bytes))
        or not isinstance(teacher_forcing, Sequence)
    ):
        teacher_forcing = [teacher_forcing]
    if not teacher_forcing:
        raise ValueError("seq2seq.teacher_forcing_ratio cannot be empty.")
    if any(not 0.0 <= float(ratio) <= 1.0 for ratio in teacher_forcing):
        raise ValueError("teacher_forcing_ratio values must be between zero and one.")
    return [
        {**parameters, "teacher_forcing_ratio": float(ratio)}
        for parameters in candidates
        for ratio in teacher_forcing
    ]


def trial_display_parameters(
    *,
    architecture: str,
    variant: str,
    parameters: dict,
    fixed_model: dict,
    training: dict,
    prediction_length: int,
) -> dict:
    """Return every effective parameter that defines one GRU trial."""

    return {
        "architecture": architecture,
        "variant": variant,
        "model": {
            **fixed_model,
            "hidden_size": parameters["hidden_size"],
            "num_layers": parameters["num_layers"],
            "prediction_length": prediction_length,
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": parameters["learning_rate"],
            "weight_decay": parameters["weight_decay"],
        },
        "training": {
            **training,
            "teacher_forcing_ratio": parameters["teacher_forcing_ratio"],
        },
    }


def build_loaders(
    dataset_folder: Path,
    *,
    variant: str,
    batch_size: int,
    seed: int,
    include_test: bool,
) -> dict[str, torch.utils.data.DataLoader]:
    """Build deterministic loaders for tuning or final evaluation."""

    splits = ("train", "val", "test") if include_test else ("train", "val")
    return {
        split: create_gru_data_loader(
            dataset_folder / f"{split}.npz",
            variant=variant,
            batch_size=batch_size,
            shuffle=split == "train",
            seed=seed,
        )
        for split in splits
    }


def train_trial(
    *,
    architecture: str,
    variant: str,
    parameters: dict,
    fixed_model: dict,
    training: dict,
    prediction_length: int,
    dataset_folder: Path,
    validation_arrays: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[GRUTrainingResult, dict[str, float], dict]:
    """Train one GRU trial and evaluate it only on validation data."""

    seed = int(training["seed"])
    set_seed(seed)
    model_config = {
        "architecture": architecture,
        "variant": variant,
        **fixed_model,
        "hidden_size": int(parameters["hidden_size"]),
        "num_layers": int(parameters["num_layers"]),
        "prediction_length": int(prediction_length),
    }
    model = build_gru_forecaster(
        architecture,
        input_size=int(model_config["input_size"]),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        prediction_length=int(model_config["prediction_length"]),
        dropout=float(model_config["dropout"]),
        bidirectional=bool(model_config["bidirectional"]),
    )
    loaders = build_loaders(
        dataset_folder,
        variant=variant,
        batch_size=int(training["batch_size"]),
        seed=seed,
        include_test=False,
    )
    result = train_gru_forecaster(
        model,
        loaders["train"],
        loaders["val"],
        device=device,
        learning_rate=float(parameters["learning_rate"]),
        max_epochs=int(training["max_epochs"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        teacher_forcing_ratio=float(parameters["teacher_forcing_ratio"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    validation_variant, _ = predict_gru_forecaster(
        model,
        loaders["val"],
        device=device,
    )
    validation_raw = predictions_in_raw_scale(
        validation_variant,
        validation_arrays,
        variant=variant,
    )
    validation_metrics = compute_trajectory_metrics(
        validation_arrays["y_raw"],
        validation_raw,
    )
    return result, validation_metrics, model_config


def save_canonical_best_run(
    *,
    run_id: str,
    search_id: str,
    architecture: str,
    variant: str,
    best_trial: pd.Series,
    best_result: GRUTrainingResult,
    model_config: dict,
    dataset_folder: Path,
    dataset_metadata: dict,
    split_arrays: dict[str, dict[str, np.ndarray]],
    split_metadata: dict[str, pd.DataFrame],
    training: dict,
    output: dict,
    device: torch.device,
) -> dict:
    """Save the selected checkpoint and its canonical raw-scale test outputs."""

    model_dir = get_model_dir("gru", run_id)
    result_dir = get_run_dir(run_id)
    if output["overwrite"]:
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(result_dir, ignore_errors=True)
    elif model_dir.exists() or result_dir.exists():
        raise FileExistsError(f"Canonical run already exists: {run_id}")
    model_dir.mkdir(parents=True, exist_ok=True)
    result_paths = ensure_results_subdirs(
        result_dir,
        ("metrics", "predictions", "figures", "tables"),
    )

    model = build_gru_forecaster(
        architecture,
        input_size=int(model_config["input_size"]),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        prediction_length=int(model_config["prediction_length"]),
        dropout=float(model_config["dropout"]),
        bidirectional=bool(model_config["bidirectional"]),
    )
    model.load_state_dict(best_result.best_state_dict)
    model.to(device)
    test_loader = build_loaders(
        dataset_folder,
        variant=variant,
        batch_size=int(training["batch_size"]),
        seed=int(training["seed"]),
        include_test=True,
    )["test"]
    test_variant, test_indices = predict_gru_forecaster(
        model,
        test_loader,
        device=device,
    )
    if not np.array_equal(test_indices, np.arange(len(test_indices))):
        raise ValueError("Test predictions are not aligned to metadata rows.")
    y_pred_raw = predictions_in_raw_scale(
        test_variant,
        split_arrays["test"],
        variant=variant,
    )
    y_true_raw = split_arrays["test"]["y_raw"]
    test_metrics = compute_trajectory_metrics(y_true_raw, y_pred_raw)
    metrics_summary = pd.DataFrame(
        [
            {
                "architecture": architecture,
                "variant": variant,
                "validation_mae_raw": float(best_trial["validation_mae_raw"]),
                "validation_rmse_raw": float(best_trial["validation_rmse_raw"]),
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "best_epoch": best_result.best_epoch,
                "best_val_loss": best_result.best_val_loss,
                "hidden_size": model_config["hidden_size"],
                "num_layers": model_config["num_layers"],
                "learning_rate": float(best_trial["learning_rate"]),
                "weight_decay": float(best_trial["weight_decay"]),
                "teacher_forcing_ratio": float(
                    best_trial["teacher_forcing_ratio"]
                ),
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
    history = pd.DataFrame(best_result.history)
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
    history.to_csv(result_paths["metrics"] / "training_history.csv", index=False)
    if output["save_best_predictions"]:
        predictions.to_parquet(
            result_paths["predictions"] / "test_predictions.parquet",
            index=False,
        )
    if output["save_best_plots"]:
        save_run_figures(
            history,
            split_metadata["test"],
            y_true_raw,
            y_pred_raw,
            result_paths["figures"],
        )

    created_at = datetime.now(timezone.utc).isoformat()
    checkpoint = {
        "model_state_dict": best_result.best_state_dict,
        "model_config": model_config,
        "grid_search_id": search_id,
        "best_trial_id": best_trial["trial_id"],
    }
    torch.save(checkpoint, model_dir / "best_model.pt")
    save_yaml(model_dir / "model_config.yaml", model_config)
    training_metadata = {
        "grid_search_id": search_id,
        "best_trial_id": best_trial["trial_id"],
        "selection_metric": "validation_rmse_raw",
        "selection_metric_value": float(best_trial["validation_rmse_raw"]),
        "best_epoch": best_result.best_epoch,
        "best_val_loss": best_result.best_val_loss,
        "learning_rate": float(best_trial["learning_rate"]),
        "weight_decay": float(best_trial["weight_decay"]),
        "batch_size": int(training["batch_size"]),
        "max_epochs": int(training["max_epochs"]),
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "device": str(device),
        "seed": int(training["seed"]),
        "teacher_forcing_ratio_training": float(
            best_trial["teacher_forcing_ratio"]
        ),
        "teacher_forcing_evaluation": 0.0,
        "created_at": created_at,
    }
    save_yaml(model_dir / "training_metadata.yaml", training_metadata)
    save_yaml(
        result_dir / "metadata.yaml",
        {
            "run_id": run_id,
            "selection_id": selection_id_from_metadata(dataset_metadata),
            "architecture": architecture,
            "variant": variant,
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
            "selection_id": selection_id_from_metadata(dataset_metadata),
            "dataset_selection": ";".join(dataset_metadata["selected_datasets"]),
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
            "created_at": created_at,
        },
        id_column="run_id",
        columns=RUN_INDEX_COLUMNS,
    )
    return metrics_summary.iloc[0].to_dict()


def main() -> None:
    """Tune each configured GRU architecture/variant and save only its winner."""

    config = load_yaml_config(CONFIG_PATH)
    search_config = config["search"]
    training = config["training"]
    output = config["output"]
    if search_config["selection_metric"] != "validation_rmse_raw":
        raise ValueError("This GRU adapter supports validation_rmse_raw selection.")
    if search_config["selection_mode"] != "min":
        raise ValueError("validation_rmse_raw must be minimized.")
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
    device = torch.device(select_device())
    candidate_counts = {
        architecture: len(build_gru_trial_candidates(config, architecture))
        for architecture in search_config["architectures"]
    }
    total_trials = sum(
        count * len(search_config["variants"])
        for count in candidate_counts.values()
    )
    print(
        "=== GRU grid search ===\n"
        f"Selection: {selection_id}\n"
        f"Device: {device}\n"
        f"Trials per architecture: {json.dumps(candidate_counts, sort_keys=True)}\n"
        f"Total training trials: {total_trials}\n"
        "Teacher forcing is tuned only for gru_seq2seq.\n"
        "Selection uses validation_rmse_raw only; test is evaluated after selection.\n",
        flush=True,
    )

    best_run_rows = []
    for architecture in search_config["architectures"]:
        candidates = build_gru_trial_candidates(config, architecture)
        for variant in search_config["variants"]:
            target_run_id = make_run_id("gru", architecture, variant, selection_id)
            search_id = f"grid_{target_run_id}"
            search_dir = get_grid_search_dir(search_id)
            if search_dir.exists() and output["overwrite"]:
                shutil.rmtree(search_dir)
            elif search_dir.exists() and any(search_dir.iterdir()):
                raise FileExistsError(f"Grid search already exists: {search_dir}")
            histories_dir = search_dir / "validation_histories"
            histories_dir.mkdir(parents=True, exist_ok=True)
            trial_rows = []
            best_artifact: tuple[str, GRUTrainingResult, dict] | None = None
            best_metric = float("inf")
            print(
                f"=== {target_run_id}: {len(candidates)} trials ===",
                flush=True,
            )
            for index, parameters in enumerate(candidates, start=1):
                trial_id = make_trial_id(parameters, index)
                start = time.perf_counter()
                display_parameters = trial_display_parameters(
                    architecture=architecture,
                    variant=variant,
                    parameters=parameters,
                    fixed_model=config["fixed_model"],
                    training=training,
                    prediction_length=int(config["data"]["prediction_length"]),
                )
                print(
                    f"[{index}/{len(candidates)}] {target_run_id} | {trial_id}\n"
                    f"  parameters={json.dumps(display_parameters, sort_keys=True)}",
                    flush=True,
                )
                try:
                    result, validation_metrics, model_config = train_trial(
                        architecture=architecture,
                        variant=variant,
                        parameters=parameters,
                        fixed_model=config["fixed_model"],
                        training=training,
                        prediction_length=int(config["data"]["prediction_length"]),
                        dataset_folder=dataset_folder,
                        validation_arrays=split_arrays["val"],
                        device=device,
                    )
                    duration = time.perf_counter() - start
                    row = {
                        "trial_id": trial_id,
                        "status": "complete",
                        "architecture": architecture,
                        "variant": variant,
                        **parameters,
                        "validation_mae_raw": validation_metrics["mae"],
                        "validation_rmse_raw": validation_metrics["rmse"],
                        "best_epoch": result.best_epoch,
                        "best_val_loss_training_scale": result.best_val_loss,
                        "epochs_run": len(result.history),
                        "duration_seconds": duration,
                        "error": "",
                    }
                    trial_rows.append(row)
                    if validation_metrics["rmse"] < best_metric:
                        best_metric = validation_metrics["rmse"]
                        best_artifact = (trial_id, result, model_config)
                    if output["save_trial_histories"]:
                        pd.DataFrame(result.history).to_csv(
                            histories_dir / f"{trial_id}.csv",
                            index=False,
                        )
                    print(
                        f"  validation MAE={validation_metrics['mae']:.6g} "
                        f"RMSE={validation_metrics['rmse']:.6g} "
                        f"epoch={result.best_epoch}",
                        flush=True,
                    )
                except Exception as error:
                    duration = time.perf_counter() - start
                    trial_rows.append(
                        {
                            "trial_id": trial_id,
                            "status": "failed",
                            "architecture": architecture,
                            "variant": variant,
                            **parameters,
                            "validation_mae_raw": np.nan,
                            "validation_rmse_raw": np.nan,
                            "best_epoch": np.nan,
                            "best_val_loss_training_scale": np.nan,
                            "epochs_run": np.nan,
                            "duration_seconds": duration,
                            "error": repr(error),
                        }
                    )
                    print(f"  failed: {error}", flush=True)

            trials = pd.DataFrame(trial_rows)
            trials_path = search_dir / "trials.csv"
            trials.to_csv(trials_path, index=False)
            best_trial = select_best_trial(
                trials,
                metric=search_config["selection_metric"],
                mode=search_config["selection_mode"],
            )
            if best_artifact is None or best_artifact[0] != best_trial["trial_id"]:
                raise RuntimeError("The retained best artifact does not match selection.")
            _, best_result, best_model_config = best_artifact
            best_params = {
                "search_id": search_id,
                "target_run_id": target_run_id,
                "best_trial_id": best_trial["trial_id"],
                "selection_metric": search_config["selection_metric"],
                "selection_metric_value": float(
                    best_trial[search_config["selection_metric"]]
                ),
                "parameters": {
                    name: python_scalar(best_trial[name])
                    for name in candidates[0]
                },
            }
            save_yaml(search_dir / "best_params.yaml", best_params)
            created_at = datetime.now(timezone.utc).isoformat()
            save_yaml(
                search_dir / "metadata.yaml",
                {
                    **best_params,
                    "model_family": search_config["model_family"],
                    "architecture": architecture,
                    "variant": variant,
                    "selection_id": selection_id,
                    "num_trials": len(trials),
                    "num_complete_trials": int(trials["status"].eq("complete").sum()),
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                    "config_path": relative_project_path(CONFIG_PATH),
                    "config_fingerprint": config_fingerprint(config),
                    "trials_path": relative_project_path(trials_path),
                    "created_at": created_at,
                },
            )
            if output["update_canonical_runs"]:
                best_run_rows.append(
                    save_canonical_best_run(
                        run_id=target_run_id,
                        search_id=search_id,
                        architecture=architecture,
                        variant=variant,
                        best_trial=best_trial,
                        best_result=best_result,
                        model_config=best_model_config,
                        dataset_folder=dataset_folder,
                        dataset_metadata=dataset_metadata,
                        split_arrays=split_arrays,
                        split_metadata=split_metadata,
                        training=training,
                        output=output,
                        device=device,
                    )
                )
            upsert_index_row(
                get_results_index_dir() / "grid_searches.csv",
                {
                    "search_id": search_id,
                    "model_family": search_config["model_family"],
                    "target_run_id": target_run_id,
                    "selection_id": selection_id,
                    "selection_metric": search_config["selection_metric"],
                    "best_trial_id": best_trial["trial_id"],
                    "results_path": relative_project_path(search_dir),
                    "best_model_path": relative_project_path(
                        get_model_dir("gru", target_run_id) / "best_model.pt"
                    ),
                    "status": "complete",
                    "created_at": created_at,
                },
                id_column="search_id",
                columns=GRID_SEARCH_INDEX_COLUMNS,
            )
            print(
                f"Selected {best_trial['trial_id']} for {target_run_id}: "
                f"validation RMSE={best_trial['validation_rmse_raw']:.6g}\n",
                flush=True,
            )

    comparison_id = f"gru_architecture_variant_{selection_id}"
    comparison_dir = get_comparison_dir(comparison_id)
    comparison_paths = ensure_results_subdirs(comparison_dir, ("tables",))
    comparison_path = (
        comparison_paths["tables"] / "gru_architecture_variant_comparison.csv"
    )
    pd.DataFrame(best_run_rows).to_csv(comparison_path, index=False)
    comparison_created_at = datetime.now(timezone.utc).isoformat()
    save_yaml(
        comparison_dir / "metadata.yaml",
        {
            "comparison_id": comparison_id,
            "selection_id": selection_id,
            "comparison_type": "forecast_metrics_across_tuned_gru_runs",
            "table": relative_project_path(comparison_path),
            "created_at": comparison_created_at,
        },
    )
    upsert_index_row(
        get_results_index_dir() / "comparisons.csv",
        {
            "comparison_id": comparison_id,
            "method_id": "multiple_tuned_gru_runs",
            "reference_id": "",
            "comparison_type": "forecast_metrics",
            "selection_id": selection_id,
            "results_path": relative_project_path(comparison_dir),
            "metrics_path": relative_project_path(comparison_paths["tables"]),
            "figures_path": "",
            "status": "complete",
            "created_at": comparison_created_at,
        },
        id_column="comparison_id",
        columns=COMPARISON_INDEX_COLUMNS,
    )
    print(f"Saved tuned GRU comparison: {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
