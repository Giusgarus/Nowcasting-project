"""Tune XGBoost for current-level persistence duration regression."""

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

from scripts.experiments.current_level_persistence.train_learnable_shapelets import (
    prediction_frame,
)
from src.tasks.current_level_persistence.evaluation.metrics import (
    compute_duration_metrics,
    compute_event_duration_metrics,
)
from src.tasks.current_level_persistence.data.dataset import (
    prepare_scalar_context_splits,
)
from src.tasks.current_level_persistence.models.xgboost import (
    build_tabular_features,
    fit_duration_regressor,
    predict_duration_regressor,
    save_duration_regressor,
)
from src.tasks.current_level_persistence.utils.paths import (
    RUN_INDEX_COLUMNS,
    grid_search_dir,
    make_run_id,
    model_dir,
    run_dir,
    run_index_path,
)
from src.tuning.grid_search import expand_parameter_grid, make_trial_id, select_best_trial
from src.tuning.trial_logging import format_trial_start
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.paths import project_path
from src.utils.results_paths import (
    GRID_SEARCH_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_results_index_dir,
    relative_project_path,
    upsert_index_row,
)

CONFIG_PATH = (
    PROJECT_ROOT / "configs/current_level_persistence/xgboost_grid_search_delta_0p5.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-trials", type=int, default=None)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load one NPZ split into memory."""

    with np.load(path, allow_pickle=True) as arrays:
        return {key: arrays[key] for key in arrays.files}


def validate_config(config: dict[str, Any], *, num_candidates: int) -> None:
    """Validate current-level XGBoost grid-search config."""

    if config.get("task_name") != "current_level_persistence":
        raise ValueError("Config task_name must be current_level_persistence.")
    if config.get("model_family") != "xgboost":
        raise ValueError("Config model_family must be xgboost.")
    required = ["search_id", "selection_id", "dataset", "search", "parameter_grid", "xgboost", "output"]
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Current-level XGBoost config is missing sections: {missing}")
    if config["search"].get("selection_metric") != "val_mae_seconds":
        raise ValueError("Current-level XGBoost selection_metric must be val_mae_seconds.")
    if config["search"].get("selection_mode") != "min":
        raise ValueError("Current-level XGBoost selection_mode must be min.")
    max_trials = int(config["search"]["max_trials"])
    if num_candidates > max_trials:
        raise ValueError(
            f"Configured current-level XGBoost grid has {num_candidates} trials, "
            f"exceeding search.max_trials={max_trials}. Increase the limit "
            "explicitly; no combinations were skipped."
        )


def xgboost_parameters(config: dict[str, Any], trial_params: dict[str, Any]) -> dict[str, Any]:
    """Merge fixed XGBoost settings and trial parameters."""

    parameters = dict(config.get("xgboost", {}))
    parameters.update(trial_params)
    int_keys = {"n_estimators", "max_depth", "n_jobs"}
    for key in int_keys & set(parameters):
        parameters[key] = int(parameters[key])
    return parameters


def split_features(
    split_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
) -> np.ndarray:
    """Build configured tabular features for one split."""

    features = config.get("features", {})
    return build_tabular_features(
        split_arrays,
        input_key=str(config["dataset"]["input_key"]),
        scalar_context_key=(
            str(config["dataset"]["scalar_context_key"])
            if bool(features.get("use_scalar_context", True))
            else None
        ),
    )


def prepare_feature_arrays(
    split_arrays: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
    dataset_metadata: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any] | None]:
    """Prepare optional scalar-context features using train-only statistics."""

    features = config.get("features", {})
    if not bool(features.get("use_scalar_context", True)):
        return split_arrays, None
    prepared, scaler = prepare_scalar_context_splits(
        split_arrays,
        scalar_context_key=str(config["dataset"]["scalar_context_key"]),
        raw_input_key=str(config["dataset"].get("raw_input_key", "X_raw")),
        delta=float(dataset_metadata["delta"]),
        standardize=bool(features.get("scalar_context_standardize", True)),
    )
    return prepared, scaler


def evaluate_predictions(
    metadata: pd.DataFrame,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> dict[str, float]:
    """Compute duration and event-level metrics for one split."""

    metrics = compute_duration_metrics(y_true_log, y_pred_log)
    metrics.update(
        compute_event_duration_metrics(
            metadata["global_event_id"].to_numpy(),
            y_true_log,
            y_pred_log,
        )
    )
    return metrics


def run_trial(
    *,
    config: dict[str, Any],
    parameters: dict[str, Any],
    trial_id: str,
    split_arrays: dict[str, dict[str, np.ndarray]],
    split_metadata: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Fit one trial on train and score validation only."""

    start = time.perf_counter()
    try:
        x_train = split_features(split_arrays["train"], config)
        x_val = split_features(split_arrays["val"], config)
        target_key = str(config["dataset"]["target_key"])
        model = fit_duration_regressor(
            x_train,
            split_arrays["train"][target_key],
            parameters=xgboost_parameters(config, parameters),
            seed=int(config["training"]["seed"]),
        )
        val_pred = predict_duration_regressor(model, x_val)
        val_metrics = evaluate_predictions(
            split_metadata["val"],
            split_arrays["val"][target_key],
            val_pred,
        )
        return {
            "trial_id": trial_id,
            "status": "complete",
            **parameters,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "duration_seconds": time.perf_counter() - start,
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - protects long grid runs
        return {
            "trial_id": trial_id,
            "status": "failed",
            **parameters,
            "val_mae_seconds": np.nan,
            "val_rmse_seconds": np.nan,
            "duration_seconds": time.perf_counter() - start,
            "error": str(exc),
        }


def save_best_run(
    *,
    config: dict[str, Any],
    dataset_path: Path,
    dataset_metadata: dict[str, Any],
    split_arrays: dict[str, dict[str, np.ndarray]],
    split_metadata: dict[str, pd.DataFrame],
    best_trial: pd.Series,
    search_id: str,
    scalar_context_scaler: dict[str, Any] | None,
) -> dict[str, Any]:
    """Retrain selected XGBoost params on train+val and save test artifacts."""

    selection_id = str(config["selection_id"])
    delta = float(dataset_metadata["delta"])
    context_length = int(dataset_metadata["context_length"])
    model_id = "xgboost"
    run_id = make_run_id(
        delta=delta,
        context_length=context_length,
        model_id=model_id,
        selection_id=selection_id,
        scalar_context=bool(config.get("features", {}).get("use_scalar_context", True)),
    )
    result_dir = run_dir(run_id)
    checkpoint_dir = model_dir(run_id)
    if result_dir.exists() and bool(config["output"].get("overwrite", False)):
        shutil.rmtree(result_dir)
    if checkpoint_dir.exists() and bool(config["output"].get("overwrite", False)):
        shutil.rmtree(checkpoint_dir)
    result_paths = ensure_results_subdirs(
        result_dir,
        ("metrics", "predictions", "figures", "tables"),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    parameter_names = list(config["parameter_grid"])
    parameters = {
        name: best_trial[name].item() if hasattr(best_trial[name], "item") else best_trial[name]
        for name in parameter_names
    }
    x_train = split_features(split_arrays["train"], config)
    x_val = split_features(split_arrays["val"], config)
    x_full = np.concatenate([x_train, x_val], axis=0)
    target_key = str(config["dataset"]["target_key"])
    y_full = np.concatenate(
        [split_arrays["train"][target_key], split_arrays["val"][target_key]],
        axis=0,
    )
    model = fit_duration_regressor(
        x_full,
        y_full,
        parameters=xgboost_parameters(config, parameters),
        seed=int(config["training"]["seed"]),
    )
    x_test = split_features(split_arrays["test"], config)
    test_pred_log = predict_duration_regressor(model, x_test)
    val_metric_columns = {
        column: best_trial[column]
        for column in best_trial.index
        if str(column).startswith("val_")
    }
    test_metrics = evaluate_predictions(
        split_metadata["test"],
        split_arrays["test"][target_key],
        test_pred_log,
    )
    test_predictions = prediction_frame(
        split_metadata["test"],
        split_arrays["test"][target_key],
        test_pred_log,
        model_id=model_id,
        run_id=run_id,
        split="test",
    )
    if bool(config["output"].get("save_best_predictions", True)):
        test_predictions.to_parquet(
            result_paths["predictions"] / "test_predictions.parquet",
            index=False,
        )
    metrics_summary = {
        "run_id": run_id,
        "model_id": model_id,
        "best_trial_id": best_trial["trial_id"],
        **{key: (value.item() if hasattr(value, "item") else value) for key, value in val_metric_columns.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        **parameters,
        "final_retrained_on_full_development": True,
    }
    pd.DataFrame([metrics_summary]).to_csv(
        result_paths["metrics"] / "metrics_summary.csv",
        index=False,
    )
    pd.DataFrame([{f"test_{key}": value for key, value in test_metrics.items()}]).to_csv(
        result_paths["metrics"] / "test_metrics.csv",
        index=False,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "run_id": run_id,
        "task_name": "current_level_persistence",
        "model_family": "xgboost",
        "model_id": model_id,
        "selection_id": selection_id,
        "delta": delta,
        "context_length": context_length,
        "dataset_path": relative_project_path(dataset_path),
        "dataset_build_fingerprint": config_fingerprint(dataset_metadata),
        "target_search_scope": dataset_metadata.get("target_search_scope"),
        "censoring_policy": dataset_metadata.get("censoring_policy"),
        "features": {
            "input_key": config["dataset"]["input_key"],
            "use_scalar_context": bool(config.get("features", {}).get("use_scalar_context", True)),
            "scalar_context_key": config["dataset"].get("scalar_context_key"),
            "scalar_context_scaler": scalar_context_scaler,
        },
        "grid_search_id": search_id,
        "best_trial_id": best_trial["trial_id"],
        "selection_metric": config["search"]["selection_metric"],
        "final_training": {
            "retrain_on_full_development": True,
            "train_split": "train+val",
            "excluded_split": "test",
        },
        "parameters": xgboost_parameters(config, parameters),
        "created_at": created_at,
    }
    save_duration_regressor(
        checkpoint_dir / "best_model.joblib",
        model=model,
        metadata=metadata,
    )
    save_yaml(result_dir / "metadata.yaml", metadata)
    save_yaml(result_dir / "config_resolved.yaml", config)
    save_yaml(checkpoint_dir / "metadata.yaml", metadata)
    save_yaml(checkpoint_dir / "config_resolved.yaml", config)
    upsert_index_row(
        run_index_path(),
        {
            "run_id": run_id,
            "task_name": "current_level_persistence",
            "model_family": "xgboost",
            "model_id": model_id,
            "delta": delta,
            "context_length": context_length,
            "selection_id": selection_id,
            "dataset_path": relative_project_path(dataset_path),
            "best_epoch": 0,
            "best_val_mae_seconds": float(best_trial["val_mae_seconds"]),
            "best_val_rmse_seconds": float(best_trial["val_rmse_seconds"]),
            "test_mae_seconds": test_metrics["mae_seconds"],
            "test_rmse_seconds": test_metrics["rmse_seconds"],
            "test_median_ae_seconds": test_metrics["median_ae_seconds"],
            "created_at": created_at,
            "status": "complete",
        },
        id_column="run_id",
        columns=RUN_INDEX_COLUMNS,
    )
    return {
        "run_id": run_id,
        "model_id": model_id,
        "best_trial_id": best_trial["trial_id"],
        "best_val_mae_seconds": float(best_trial["val_mae_seconds"]),
        "best_val_rmse_seconds": float(best_trial["val_rmse_seconds"]),
        "test_mae_seconds": test_metrics["mae_seconds"],
        "test_rmse_seconds": test_metrics["rmse_seconds"],
        "results_path": relative_project_path(result_dir),
        "model_path": relative_project_path(checkpoint_dir),
        "created_at": created_at,
    }


def main() -> None:
    """Run validation-only XGBoost search and save the best test run."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if args.max_trials is not None:
        config["search"]["max_trials"] = int(args.max_trials)
    candidates = expand_parameter_grid(config["parameter_grid"])
    validate_config(config, num_candidates=len(candidates))

    dataset_path = project_path(config["dataset"]["path"])
    search_id = str(config["search_id"])
    if args.dry_run:
        print(
            "=== Current-Level Persistence XGBoost Grid Search Dry Run ===\n"
            f"Config: {relative_project_path(config_path)}\n"
            f"Search ID: {search_id}\n"
            f"Dataset: {relative_project_path(dataset_path)}\n"
            f"Selection: {config['selection_id']}\n"
            f"Trials: {len(candidates)}\n"
            "No dataset was loaded and no XGBoost models were trained.",
            flush=True,
        )
        return

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    split_arrays = {
        split: load_npz(dataset_path / f"{file_stem}.npz")
        for split, file_stem in {"train": "train", "val": "val", "test": "test"}.items()
    }
    split_metadata = {
        split: pd.read_parquet(dataset_path / f"{file_stem}_metadata.parquet")
        for split, file_stem in {"train": "train", "val": "val", "test": "test"}.items()
    }
    dataset_metadata = load_yaml_config(dataset_path / "dataset_metadata.yaml")
    split_arrays, scalar_context_scaler = prepare_feature_arrays(
        split_arrays,
        config,
        dataset_metadata,
    )
    search_dir = grid_search_dir(
        selection_id=str(config["selection_id"]),
        search_id=search_id,
    )
    print(
        "=== Current-Level Persistence XGBoost Grid Search ===\n"
        f"Config: {relative_project_path(config_path)}\n"
        f"Search ID: {search_id}\n"
        f"Dataset: {relative_project_path(dataset_path)}\n"
        f"Selection: {config['selection_id']}\n"
        f"Trials: {len(candidates)}\n"
        "Selection uses validation MAE seconds only; test is evaluated after selection.",
        flush=True,
    )

    overwrite = bool(config["output"].get("overwrite", False)) or args.force
    if search_dir.exists() and overwrite:
        shutil.rmtree(search_dir)
    elif search_dir.exists():
        raise FileExistsError(f"Grid-search folder already exists: {search_dir}")
    paths = ensure_results_subdirs(search_dir, ("tables",))
    save_yaml(
        search_dir / "metadata.yaml",
        {
            "search_id": search_id,
            "task_name": "current_level_persistence",
            "model_family": "xgboost",
            "selection_id": config["selection_id"],
            "dataset_path": relative_project_path(dataset_path),
            "dataset_build_fingerprint": config_fingerprint(dataset_metadata),
            "target_search_scope": dataset_metadata.get("target_search_scope"),
            "censoring_policy": dataset_metadata.get("censoring_policy"),
            "scalar_context_scaler": scalar_context_scaler,
            "config_path": relative_project_path(config_path),
            "config_fingerprint": config_fingerprint(config),
            "total_trials": len(candidates),
            "selection_metric": config["search"]["selection_metric"],
            "selection_mode": config["search"]["selection_mode"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    save_yaml(search_dir / "config_resolved.yaml", config)

    rows = []
    for index, parameters in enumerate(candidates, start=1):
        trial_id = make_trial_id(parameters, index)
        print(
            format_trial_start(
                index=index,
                total=len(candidates),
                run_id=make_run_id(
                    delta=float(dataset_metadata["delta"]),
                    context_length=int(dataset_metadata["context_length"]),
                    model_id="xgboost",
                    selection_id=str(config["selection_id"]),
                    scalar_context=bool(config.get("features", {}).get("use_scalar_context", True)),
                ),
                trial_id=trial_id,
                device=str(config["xgboost"].get("device", "cpu")),
                parameters={
                    "architecture": "xgboost",
                    "model": {**config["xgboost"], **parameters},
                },
            ),
            flush=True,
        )
        row = run_trial(
            config=config,
            parameters=parameters,
            trial_id=trial_id,
            split_arrays=split_arrays,
            split_metadata=split_metadata,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(paths["tables"] / "trials.csv", index=False)
        if row["status"] == "complete":
            print(
                f"  completed {trial_id} val_mae_seconds={row['val_mae_seconds']:.6g}",
                flush=True,
            )
        else:
            print(f"  failed {trial_id}: {row['error']}", flush=True)
    trials = pd.DataFrame(rows)
    trials.to_csv(paths["tables"] / "trials.csv", index=False)
    best_trial = select_best_trial(
        trials,
        metric=config["search"]["selection_metric"],
        mode=config["search"]["selection_mode"],
    )
    save_yaml(
        search_dir / "best_trial.yaml",
        {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in best_trial.to_dict().items()
        },
    )
    best_row = save_best_run(
        config=config,
        dataset_path=dataset_path,
        dataset_metadata=dataset_metadata,
        split_arrays=split_arrays,
        split_metadata=split_metadata,
        best_trial=best_trial,
        search_id=search_id,
        scalar_context_scaler=scalar_context_scaler,
    )
    pd.DataFrame([best_row]).to_csv(paths["tables"] / "best_runs.csv", index=False)
    upsert_index_row(
        get_results_index_dir() / "grid_searches.csv",
        {
            "search_id": search_id,
            "model_family": "xgboost",
            "target_run_id": best_row["run_id"],
            "selection_id": str(config["selection_id"]),
            "selection_metric": config["search"]["selection_metric"],
            "best_trial_id": best_trial["trial_id"],
            "results_path": relative_project_path(search_dir),
            "best_model_path": str(Path(best_row["model_path"]) / "best_model.joblib"),
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        id_column="search_id",
        columns=GRID_SEARCH_INDEX_COLUMNS,
    )
    print(
        "=== Selected best current-level XGBoost run ===\n"
        f"{best_row['run_id']} | val MAE={best_row['best_val_mae_seconds']:.6g} | "
        f"test MAE={best_row['test_mae_seconds']:.6g} | "
        f"test RMSE={best_row['test_rmse_seconds']:.6g}\n"
        f"Grid results: {relative_project_path(search_dir)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
