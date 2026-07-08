"""Tune XGBoost autoregressive baselines with one regressor per horizon."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.autoregressive.run_autoregressive_gru import (
    build_prediction_table,
    grouped_metrics,
    load_split_arrays,
    resolve_selection_folder,
)
from src.tasks.autoregressive.evaluation.forecast_metrics import (
    compute_horizon_metrics,
    compute_trajectory_metrics,
    inverse_context_standardization,
)
from src.tasks.autoregressive.models.xgboost import (
    fit_horizon_regressors,
    predict_horizon_regressors,
    save_horizon_regressors,
)
from src.tuning.grid_search import expand_parameter_grid, make_trial_id, select_best_trial
from src.tuning.trial_logging import format_trial_start
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.paths import project_path
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

CONFIG_PATH = PROJECT_ROOT / "configs/autoregressive/xgboost_grid_search.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-trials-per-variant", type=int, default=None)
    return parser.parse_args()


def validate_config(config: dict[str, Any], *, num_candidates: int) -> None:
    """Validate the XGBoost autoregressive grid before expensive work."""

    if config.get("task_name") != "autoregressive":
        raise ValueError("Config task_name must be autoregressive.")
    if config.get("model_family") != "xgboost":
        raise ValueError("Config model_family must be xgboost.")
    required = ["search_id", "data", "search", "parameter_grid", "xgboost", "output"]
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"XGBoost grid config is missing sections: {missing}")
    search = config["search"]
    if search.get("selection_metric") != "validation_rmse_raw":
        raise ValueError("Autoregressive XGBoost selection_metric must be validation_rmse_raw.")
    if search.get("selection_mode") != "min":
        raise ValueError("Autoregressive XGBoost selection_mode must be min.")
    variants = search.get("variants", [])
    if not variants or any(variant not in {"raw", "context_standard"} for variant in variants):
        raise ValueError("search.variants must contain raw and/or context_standard.")
    max_trials = int(search["max_trials_per_variant"])
    if num_candidates > max_trials:
        raise ValueError(
            f"Configured XGBoost grid has {num_candidates} trials per variant, "
            f"exceeding search.max_trials_per_variant={max_trials}. Increase the "
            "limit explicitly; no combinations were skipped."
        )


def variant_arrays(
    split_arrays: dict[str, np.ndarray],
    *,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X and y arrays for one configured variant."""

    if variant == "raw":
        return split_arrays["X_raw"], split_arrays["y_raw"]
    if variant == "context_standard":
        return split_arrays["X_context_standard"], split_arrays["y_context_standard"]
    raise ValueError(f"Unsupported XGBoost variant: {variant}")


def predictions_in_raw_scale(
    predictions: np.ndarray,
    split_arrays: dict[str, np.ndarray],
    *,
    variant: str,
) -> np.ndarray:
    """Return predictions on the raw signal scale."""

    if variant == "context_standard":
        return inverse_context_standardization(
            predictions,
            split_arrays["scaling_mean"],
            split_arrays["scaling_std"],
        )
    return predictions


def xgboost_parameters(config: dict[str, Any], trial_params: dict[str, Any]) -> dict[str, Any]:
    """Merge fixed XGBoost settings and trial parameters."""

    parameters = dict(config.get("xgboost", {}))
    parameters.update(trial_params)
    int_keys = {"n_estimators", "max_depth", "n_jobs"}
    for key in int_keys & set(parameters):
        parameters[key] = int(parameters[key])
    return parameters


def run_trial(
    *,
    config: dict[str, Any],
    variant: str,
    parameters: dict[str, Any],
    trial_id: str,
    split_arrays: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Train one trial on train and score validation only."""

    start = time.perf_counter()
    try:
        x_train, y_train = variant_arrays(split_arrays["train"], variant=variant)
        x_val, _ = variant_arrays(split_arrays["val"], variant=variant)
        models = fit_horizon_regressors(
            x_train,
            y_train,
            parameters=xgboost_parameters(config, parameters),
            seed=int(config["training"]["seed"]),
        )
        val_pred_variant = predict_horizon_regressors(models, x_val)
        val_pred_raw = predictions_in_raw_scale(
            val_pred_variant,
            split_arrays["val"],
            variant=variant,
        )
        val_metrics = compute_trajectory_metrics(split_arrays["val"]["y_raw"], val_pred_raw)
        return {
            "trial_id": trial_id,
            "status": "complete",
            "variant": variant,
            **parameters,
            "validation_mae_raw": val_metrics["mae"],
            "validation_rmse_raw": val_metrics["rmse"],
            "duration_seconds": time.perf_counter() - start,
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - protects long grid runs
        return {
            "trial_id": trial_id,
            "status": "failed",
            "variant": variant,
            **parameters,
            "validation_mae_raw": np.nan,
            "validation_rmse_raw": np.nan,
            "duration_seconds": time.perf_counter() - start,
            "error": str(exc),
        }


def save_best_run(
    *,
    config: dict[str, Any],
    dataset_folder: Path,
    dataset_metadata: dict[str, Any],
    split_arrays: dict[str, dict[str, np.ndarray]],
    split_metadata: dict[str, pd.DataFrame],
    selection_id: str,
    search_id: str,
    variant: str,
    best_trial: pd.Series,
) -> dict[str, Any]:
    """Retrain the selected trial on train+val and save test artifacts."""

    run_id = make_run_id("xgboost", "xgboost", variant, selection_id)
    result_dir = get_run_dir(run_id)
    model_dir = get_model_dir("xgboost", run_id)
    if result_dir.exists() and bool(config["output"]["overwrite"]):
        shutil.rmtree(result_dir)
    if model_dir.exists() and bool(config["output"]["overwrite"]):
        shutil.rmtree(model_dir)
    result_paths = ensure_results_subdirs(
        result_dir,
        ("metrics", "predictions", "figures", "tables"),
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    parameter_names = list(config["parameter_grid"])
    parameters = {name: best_trial[name].item() if hasattr(best_trial[name], "item") else best_trial[name] for name in parameter_names}
    x_train, y_train = variant_arrays(split_arrays["train"], variant=variant)
    x_val, y_val = variant_arrays(split_arrays["val"], variant=variant)
    x_full = np.concatenate([x_train, x_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    models = fit_horizon_regressors(
        x_full,
        y_full,
        parameters=xgboost_parameters(config, parameters),
        seed=int(config["training"]["seed"]),
    )
    x_test, _ = variant_arrays(split_arrays["test"], variant=variant)
    test_pred_variant = predict_horizon_regressors(models, x_test)
    test_pred_raw = predictions_in_raw_scale(
        test_pred_variant,
        split_arrays["test"],
        variant=variant,
    )
    y_true_raw = split_arrays["test"]["y_raw"]
    test_metrics = compute_trajectory_metrics(y_true_raw, test_pred_raw)
    metrics_summary = pd.DataFrame(
        [
            {
                "architecture": "xgboost",
                "variant": variant,
                "best_trial_id": best_trial["trial_id"],
                "validation_mae_raw": float(best_trial["validation_mae_raw"]),
                "validation_rmse_raw": float(best_trial["validation_rmse_raw"]),
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "final_retrained_on_full_development": True,
                **parameters,
                "num_train_windows": len(split_metadata["train"]),
                "num_val_windows": len(split_metadata["val"]),
                "num_test_windows": len(split_metadata["test"]),
            }
        ]
    )
    horizon_metrics = compute_horizon_metrics(y_true_raw, test_pred_raw)
    horizon_metrics.insert(0, "variant", variant)
    horizon_metrics.insert(0, "architecture", "xgboost")
    predictions = build_prediction_table(
        split_metadata["test"],
        y_true_raw,
        test_pred_raw,
        architecture="xgboost",
        variant=variant,
    )
    metrics_summary.to_csv(result_paths["metrics"] / "metrics_summary.csv", index=False)
    horizon_metrics.to_csv(result_paths["metrics"] / "metrics_by_horizon.csv", index=False)
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
    if bool(config["output"].get("save_best_predictions", True)):
        predictions.to_parquet(
            result_paths["predictions"] / "test_predictions.parquet",
            index=False,
        )

    created_at = datetime.now(timezone.utc).isoformat()
    model_metadata = {
        "run_id": run_id,
        "task_name": "autoregressive",
        "model_family": "xgboost",
        "architecture": "xgboost",
        "variant": variant,
        "selection_id": selection_id,
        "dataset_path": relative_project_path(dataset_folder),
        "grid_search_id": search_id,
        "best_trial_id": best_trial["trial_id"],
        "selection_metric": config["search"]["selection_metric"],
        "final_training": {
            "retrain_on_full_development": True,
            "train_split": "train+val",
            "excluded_split": "test",
        },
        "parameters": xgboost_parameters(config, parameters),
        "num_horizon_models": int(dataset_metadata["prediction_length"]),
        "created_at": created_at,
    }
    save_horizon_regressors(
        model_dir / "best_model.joblib",
        models=models,
        metadata=model_metadata,
    )
    save_yaml(model_dir / "metadata.yaml", model_metadata)
    save_yaml(model_dir / "config_resolved.yaml", config)
    save_yaml(result_dir / "metadata.yaml", model_metadata)
    save_yaml(result_dir / "config_resolved.yaml", config)
    upsert_index_row(
        get_results_index_dir() / "runs.csv",
        {
            "run_id": run_id,
            "model_family": "xgboost",
            "architecture": "xgboost",
            "variant": variant,
            "selection_id": selection_id,
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
    """Run validation-only XGBoost search and materialize best test runs."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if args.max_trials_per_variant is not None:
        config["search"]["max_trials_per_variant"] = int(args.max_trials_per_variant)
    candidates = expand_parameter_grid(config["parameter_grid"])
    validate_config(config, num_candidates=len(candidates))

    search_id = str(config["search_id"])
    total_trials = len(candidates) * len(config["search"]["variants"])
    if args.dry_run:
        print(
            "=== Autoregressive XGBoost Grid Search Dry Run ===\n"
            f"Config: {relative_project_path(config_path)}\n"
            f"Search ID: {search_id}\n"
            f"Dataset root: {config['data']['dataset_root']}\n"
            f"Selection folder: {config['data']['selection_folder']}\n"
            f"Variants: {config['search']['variants']}\n"
            f"Trials per variant: {len(candidates)}\n"
            f"Total trials: {total_trials}\n"
            "No dataset was loaded and no XGBoost models were trained.",
            flush=True,
        )
        return

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
    search_dir = get_grid_search_dir(search_id)
    print(
        "=== Autoregressive XGBoost Grid Search ===\n"
        f"Config: {relative_project_path(config_path)}\n"
        f"Search ID: {search_id}\n"
        f"Dataset: {relative_project_path(dataset_folder)}\n"
        f"Selection: {selection_id}\n"
        f"Variants: {config['search']['variants']}\n"
        f"Trials per variant: {len(candidates)}\n"
        f"Total trials: {total_trials}\n"
        "Each trial fits one independent regressor per forecast horizon.\n"
        "Selection uses validation_rmse_raw only; test is evaluated after selection.",
        flush=True,
    )

    overwrite = bool(config["output"].get("overwrite", False)) or args.force
    if search_dir.exists() and overwrite:
        shutil.rmtree(search_dir)
    elif search_dir.exists():
        raise FileExistsError(f"Grid-search folder already exists: {search_dir}")
    search_paths = ensure_results_subdirs(search_dir, ("tables",))
    save_yaml(
        search_dir / "metadata.yaml",
        {
            "search_id": search_id,
            "task_name": "autoregressive",
            "model_family": "xgboost",
            "selection_id": selection_id,
            "dataset_path": relative_project_path(dataset_folder),
            "config_path": relative_project_path(config_path),
            "config_fingerprint": config_fingerprint(config),
            "trials_per_variant": len(candidates),
            "total_trials": total_trials,
            "selection_metric": config["search"]["selection_metric"],
            "selection_mode": config["search"]["selection_mode"],
            "test_policy": "test split evaluated only after validation selection",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    save_yaml(search_dir / "config_resolved.yaml", config)

    best_rows = []
    for variant in config["search"]["variants"]:
        trial_rows = []
        variant_dir = search_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        for index, parameters in enumerate(candidates, start=1):
            trial_id = make_trial_id({"variant": variant, **parameters}, index)
            print(
                format_trial_start(
                    index=index,
                    total=len(candidates),
                    run_id=make_run_id("xgboost", "xgboost", variant, selection_id),
                    trial_id=trial_id,
                    device=str(config["xgboost"].get("device", "cpu")),
                    parameters={
                        "architecture": "xgboost",
                        "variant": variant,
                        "model": {
                            **config["xgboost"],
                            **parameters,
                            "num_horizon_models": int(dataset_metadata["prediction_length"]),
                        },
                    },
                ),
                flush=True,
            )
            row = run_trial(
                config=config,
                variant=variant,
                parameters=parameters,
                trial_id=trial_id,
                split_arrays=split_arrays,
            )
            trial_rows.append(row)
            pd.DataFrame(trial_rows).to_csv(variant_dir / "trials.csv", index=False)
            if row["status"] == "complete":
                print(
                    f"  completed {trial_id} "
                    f"validation_rmse_raw={row['validation_rmse_raw']:.6g}",
                    flush=True,
                )
            else:
                print(f"  failed {trial_id}: {row['error']}", flush=True)
        trials = pd.DataFrame(trial_rows)
        trials.to_csv(variant_dir / "trials.csv", index=False)
        best_trial = select_best_trial(
            trials,
            metric=config["search"]["selection_metric"],
            mode=config["search"]["selection_mode"],
        )
        save_yaml(
            variant_dir / "best_trial.yaml",
            {
                "variant": variant,
                **{
                    key: (value.item() if hasattr(value, "item") else value)
                    for key, value in best_trial.to_dict().items()
                },
            },
        )
        best_rows.append(
            save_best_run(
                config=config,
                dataset_folder=dataset_folder,
                dataset_metadata=dataset_metadata,
                split_arrays=split_arrays,
                split_metadata=split_metadata,
                selection_id=selection_id,
                search_id=search_id,
                variant=variant,
                best_trial=best_trial,
            )
        )
        upsert_index_row(
            get_results_index_dir() / "grid_searches.csv",
            {
                "search_id": f"{search_id}_{variant}",
                "model_family": "xgboost",
                "target_run_id": make_run_id("xgboost", "xgboost", variant, selection_id),
                "selection_id": selection_id,
                "selection_metric": config["search"]["selection_metric"],
                "best_trial_id": best_trial["trial_id"],
                "results_path": relative_project_path(variant_dir),
                "best_model_path": relative_project_path(
                    get_model_dir(
                        "xgboost",
                        make_run_id("xgboost", "xgboost", variant, selection_id),
                    )
                    / "best_model.joblib"
                ),
                "status": "complete",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            id_column="search_id",
            columns=GRID_SEARCH_INDEX_COLUMNS,
        )

    comparison_id = f"xgboost_search_{selection_id}"
    comparison_dir = get_comparison_dir(comparison_id)
    comparison_paths = ensure_results_subdirs(comparison_dir, ("tables",))
    comparison_path = comparison_paths["tables"] / "xgboost_variant_comparison.csv"
    pd.DataFrame(best_rows).to_csv(comparison_path, index=False)
    upsert_index_row(
        get_results_index_dir() / "comparisons.csv",
        {
            "comparison_id": comparison_id,
            "method_id": "multiple_xgboost_runs",
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
    print(f"Saved XGBoost comparison: {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
