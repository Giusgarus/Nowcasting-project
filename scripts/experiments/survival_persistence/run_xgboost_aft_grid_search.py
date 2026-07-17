"""Validation-only grid search for survival-persistence XGBoost-AFT."""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.survival_persistence.run_xgboost_aft import (  # noqa: E402
    load_split_artifacts,
    make_predictions_frame,
    metrics_for_predictions,
    project_path,
    save_feature_importance,
    save_plots,
    training_history_frame,
)
from src.tasks.survival_persistence.evaluation.metrics import (  # noqa: E402
    brier_scores_by_horizon,
    calibration_by_horizon,
    event_level_bootstrap,
    fit_censoring_survival,
    harrell_c_index,
)
from src.tasks.survival_persistence.models.xgboost_aft import (  # noqa: E402
    aft_negative_log_likelihood,
    build_dmatrix,
    require_xgboost,
    validate_feature_compatibility,
)
from src.tasks.survival_persistence.utils.paths import (  # noqa: E402
    RUN_INDEX_COLUMNS,
    TASK_NAME,
    make_run_id,
    model_dir,
    run_dir,
    run_index_path,
)
from src.tuning.grid_search import (  # noqa: E402
    expand_parameter_grid,
    make_trial_id,
    select_best_trial,
)
from src.tuning.parallel_trials import (  # noqa: E402
    choose_trial_devices,
    iter_parallel_trial_results,
)
from src.tuning.trial_logging import format_trial_start  # noqa: E402
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.results_paths import (  # noqa: E402
    GRID_SEARCH_INDEX_COLUMNS,
    get_results_index_dir,
    relative_project_path,
    sanitize_id,
    upsert_index_row,
)

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/survival_persistence/models/xgboost_aft_grid_search.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Run only the first N expanded trials; useful for server smoke tests.",
    )
    parser.add_argument(
        "--skip-finalize",
        action="store_true",
        help="Run trials and write trials.csv without refitting/saving a best run.",
    )
    parser.add_argument(
        "--search-suffix",
        default=None,
        help="Append a suffix to the search_id, e.g. smoke, to avoid overwriting a full grid.",
    )
    return parser.parse_args()


def deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of base recursively updated by update."""

    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_flat_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict:
    """Apply dotted-key overrides to a copy of config."""

    result = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        target = result
        parts = str(dotted_key).split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = copy.deepcopy(value)
    return result


def single_run_base_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the single-run config template used by each trial."""

    return {
        "task_name": TASK_NAME,
        "model_name": "xgboost_aft",
        "schema_version": int(config.get("schema_version", 1)),
        "dataset": copy.deepcopy(config["dataset"]),
        "model": copy.deepcopy(config["model"]),
        "training": copy.deepcopy(config["training"]),
        "prediction": copy.deepcopy(config["prediction"]),
        "evaluation": copy.deepcopy(config["evaluation"]),
        "output": copy.deepcopy(config.get("output", {})),
        "seed": int(config.get("seed", 42)),
    }


def build_trial_config(config: Mapping[str, Any], parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Build one resolved trial config."""

    return apply_flat_overrides(single_run_base_config(config), parameters)


def validate_config(config: Mapping[str, Any]) -> int:
    """Validate survival XGBoost-AFT grid config and return number of trials."""

    if config.get("task_name") != TASK_NAME:
        raise ValueError(f"Config task_name must be {TASK_NAME}.")
    if config.get("model_name") != "xgboost_aft":
        raise ValueError("Config model_name must be xgboost_aft.")
    required = [
        "search_id",
        "dataset",
        "search",
        "parallel",
        "model",
        "training",
        "prediction",
        "evaluation",
        "parameter_grid",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Survival grid config missing sections: {missing}")
    search = config["search"]
    if search.get("selection_metric") != "val_aft_nloglik":
        raise ValueError("selection_metric must be val_aft_nloglik.")
    if search.get("selection_mode") != "min":
        raise ValueError("selection_mode must be min.")
    trials = expand_parameter_grid(config["parameter_grid"])
    max_trials = int(search["max_trials"])
    if len(trials) > max_trials:
        raise ValueError(
            f"Configured survival XGBoost-AFT grid has {len(trials)} trials, "
            f"exceeding search.max_trials={max_trials}. Increase the safety "
            "limit explicitly; no combinations were skipped."
        )
    return len(trials)


def grid_search_dir(*, selection_id: str, search_id: str) -> Path:
    """Return the survival grid-search artifact directory."""

    return (
        PROJECT_ROOT
        / "results"
        / "grid_searches"
        / TASK_NAME
        / sanitize_id(selection_id)
        / sanitize_id(search_id)
    )


def xgboost_params_for_device(params: Mapping[str, Any], device: str) -> dict[str, Any]:
    """Return XGBoost params adjusted for the assigned trial device."""

    output = dict(params)
    if str(device).startswith("cuda"):
        output["device"] = str(device)
    elif str(device) == "mps":
        output["device"] = "cpu"
        output["device_fallback_reason"] = "xgboost_does_not_support_mps"
    else:
        output["device"] = "cpu"
    return output


def train_trial(
    *,
    trial_config: Mapping[str, Any],
    arrays: dict[str, dict[str, np.ndarray]],
    device: str,
) -> tuple[dict[str, Any], Any, dict[str, dict[str, list[float]]]]:
    """Train one trial and return validation metrics plus booster."""

    xgb = require_xgboost()
    feature_set = str(trial_config["dataset"]["feature_set"])
    sample_weighting = str(trial_config["dataset"]["sample_weighting"])
    dtrain, train_features = build_dmatrix(
        arrays["train"],
        feature_set=feature_set,
        sample_weighting=sample_weighting,
    )
    dval, val_features = build_dmatrix(
        arrays["validation"],
        feature_set=feature_set,
        sample_weighting=sample_weighting,
    )
    validate_feature_compatibility(train_features, val_features)
    params = xgboost_params_for_device(trial_config["model"], device)
    params.pop("device_fallback_reason", None)
    params["seed"] = int(trial_config["training"].get("seed", 42))
    params["nthread"] = int(trial_config["training"].get("nthread", 4))
    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(trial_config["training"]["num_boost_round"]),
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=int(trial_config["training"]["early_stopping_rounds"]),
        evals_result=evals_result,
        verbose_eval=False,
    )
    best_iteration = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    location = booster.predict(
        dval,
        output_margin=True,
        iteration_range=(0, best_iteration + 1),
    )
    lower, upper = (
        arrays["validation"]["y_lower_bound_seconds"],
        arrays["validation"]["y_upper_bound_seconds"],
    )
    val_nloglik = aft_negative_log_likelihood(
        location,
        lower,
        upper,
        scale=float(trial_config["model"]["aft_loss_distribution_scale"]),
        distribution=str(trial_config["model"]["aft_loss_distribution"]),
    )
    cindex = harrell_c_index(
        arrays["validation"]["y_time_seconds"],
        arrays["validation"]["y_event_observed"],
        np.exp(location),
    )
    row = {
        "status": "complete",
        "val_aft_nloglik": float(val_nloglik),
        "val_c_index": float(cindex["c_index"]),
        "best_iteration": best_iteration,
        "best_score": float(getattr(booster, "best_score", np.nan)),
    }
    return row, booster, evals_result


def run_trial_job(job: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point for one grid-search trial."""

    try:
        arrays, _ = load_split_artifacts(project_path(job["dataset_path"]))
        row, _, _ = train_trial(
            trial_config=job["trial_config"],
            arrays=arrays,
            device=str(job["device"]),
        )
        return {
            **job["row_base"],
            **row,
            "device": str(job["device"]),
        }
    except Exception as error:  # pragma: no cover - exercised in failed trials
        return {
            **job["row_base"],
            "status": "failed",
            "device": str(job.get("device", "")),
            "error": f"{type(error).__name__}: {error}",
        }


def finalize_best_run(
    *,
    config: Mapping[str, Any],
    dataset_dir: Path,
    selection_id: str,
    best_trial: Mapping[str, Any],
    best_config: Mapping[str, Any],
    search_dir: Path,
    device: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Retrain/save the selected best run and return index metadata."""

    from scripts.experiments.survival_persistence.run_xgboost_aft import (
        calibration_tables,
        event_timepoint_table,
    )

    arrays, split_metadata = load_split_artifacts(dataset_dir)
    row, booster, evals_result = train_trial(
        trial_config=best_config,
        arrays=arrays,
        device=device,
    )
    feature_set = str(best_config["dataset"]["feature_set"])
    sample_weighting = str(best_config["dataset"]["sample_weighting"])
    run_id = make_run_id(
        model_id="xgboost_aft",
        feature_set=feature_set,
        sample_weighting=sample_weighting,
        selection_id=selection_id,
    )
    output_dir = run_dir(selection_id=selection_id, run_id=run_id)
    checkpoint_dir = model_dir(model_id="xgboost_aft", run_id=run_id)
    if not overwrite and output_dir.exists() and any(output_dir.rglob("*")):
        raise FileExistsError(f"Run output already exists: {output_dir}")
    for directory in [
        output_dir / "metrics",
        output_dir / "predictions",
        output_dir / "figures",
        output_dir / "tables",
        checkpoint_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    best_iteration = int(row["best_iteration"])
    distribution = str(best_config["model"]["aft_loss_distribution"])
    scale = float(best_config["model"]["aft_loss_distribution_scale"])
    horizons = np.asarray(best_config["prediction"]["horizons_seconds"], dtype=float)
    dmatrices = {}
    feature_matrix = None
    for split in ["train", "validation", "test"]:
        dmatrix, split_features = build_dmatrix(
            arrays[split],
            feature_set=feature_set,
            sample_weighting=sample_weighting,
        )
        if feature_matrix is None:
            feature_matrix = split_features
        else:
            validate_feature_compatibility(feature_matrix, split_features)
        dmatrices[split] = dmatrix
    prediction_frames = {}
    for split, dmatrix in dmatrices.items():
        location = booster.predict(
            dmatrix,
            output_margin=True,
            iteration_range=(0, best_iteration + 1),
        )
        frame, _ = make_predictions_frame(
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
        frame.to_parquet(output_dir / "predictions" / f"{split}_predictions.parquet", index=False)
    censoring_curve = fit_censoring_survival(
        arrays["train"]["y_time_seconds"],
        arrays["train"]["y_event_observed"],
    )
    metric_rows = []
    brier_rows = []
    for split, frame in prediction_frames.items():
        metric, _ = metrics_for_predictions(
            method_name="xgboost_aft",
            split=split,
            predictions=frame,
            horizons_seconds=horizons,
            censoring_curve=censoring_curve,
            distribution=distribution,
            scale=scale,
        )
        metric_rows.append(metric)
        brier_rows.append(
            brier_scores_by_horizon(
                frame["y_time_seconds"].to_numpy(dtype=float),
                frame["y_event_observed"].to_numpy(dtype=int),
                frame[[f"survival_probability_{int(h)}s" for h in horizons]].to_numpy(dtype=float),
                horizons,
                censoring_curve=censoring_curve,
            ).assign(method="xgboost_aft", split=split)
        )
    metrics_summary = pd.DataFrame(metric_rows)
    brier_by_horizon = pd.concat(brier_rows, ignore_index=True)
    metrics_summary.to_csv(output_dir / "metrics" / "metrics_summary.csv", index=False)
    brier_by_horizon.to_csv(output_dir / "metrics" / "brier_by_horizon.csv", index=False)
    calibration = calibration_tables(
        predictions=prediction_frames["test"],
        horizons_seconds=list(best_config["evaluation"]["calibration_horizons_seconds"]),
        n_bins=int(best_config["evaluation"].get("calibration_bins", 10)),
        censoring_curve=censoring_curve,
    )
    calibration.to_csv(output_dir / "metrics" / "test_calibration.csv", index=False)
    event_timepoint_table(prediction_frames["test"]).to_csv(
        output_dir / "tables" / "test_event_timepoint_predictions.csv",
        index=False,
    )
    bootstrap = event_level_bootstrap(
        prediction_frames["test"],
        event_column="global_event_id",
        metric_fn=lambda frame: harrell_c_index(
            frame["y_time_seconds"].to_numpy(dtype=float),
            frame["y_event_observed"].to_numpy(dtype=int),
            frame["predicted_median_remaining_seconds"].to_numpy(dtype=float),
        )["c_index"],
        n_replicates=int(best_config["evaluation"].get("bootstrap_replicates", 0)),
        seed=int(best_config.get("seed", 42)),
    )
    if not bootstrap.empty:
        bootstrap.to_csv(output_dir / "metrics" / "test_event_bootstrap_c_index.csv", index=False)
    booster_path = checkpoint_dir / "booster.json"
    booster.save_model(booster_path)
    history = training_history_frame(evals_result)
    if not history.empty:
        history.to_csv(output_dir / "tables" / "training_history.csv", index=False)
    assert feature_matrix is not None
    (output_dir / "tables" / "feature_names.txt").write_text(
        "\n".join(feature_matrix.names),
        encoding="utf-8",
    )
    save_feature_importance(
        booster,
        feature_matrix.names,
        output_dir / "tables",
        output_dir / "figures",
    )
    save_plots(
        figures_dir=output_dir / "figures",
        training_history=history,
        test_predictions=prediction_frames["test"],
        test_brier=brier_by_horizon.loc[brier_by_horizon["split"].eq("test")],
        calibration=calibration,
        horizons_seconds=horizons,
    )
    save_yaml(output_dir / "config_resolved.yaml", dict(best_config))
    save_yaml(checkpoint_dir / "model_config.yaml", dict(best_config))
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "task_name": TASK_NAME,
        "model_family": "xgboost",
        "model_id": "xgboost_aft",
        "run_id": run_id,
        "source_grid_search": relative_project_path(search_dir),
        "best_trial_id": best_trial["trial_id"],
        "dataset_path": relative_project_path(dataset_dir),
        "selection_id": selection_id,
        "feature_set": feature_set,
        "sample_weighting": sample_weighting,
        "best_iteration": best_iteration,
        "best_validation_aft_nloglik": row["val_aft_nloglik"],
        "booster_path": relative_project_path(booster_path),
        "created_at": created_at,
    }
    save_yaml(output_dir / "metadata.yaml", metadata)
    save_yaml(checkpoint_dir / "training_metadata.yaml", metadata)
    test_row = metrics_summary.loc[metrics_summary["split"].eq("test")].iloc[0]
    val_row = metrics_summary.loc[metrics_summary["split"].eq("validation")].iloc[0]
    upsert_index_row(
        run_index_path(),
        {
            "run_id": run_id,
            "task_name": TASK_NAME,
            "model_family": "xgboost",
            "model_id": "xgboost_aft",
            "feature_set": feature_set,
            "sample_weighting": sample_weighting,
            "selection_id": selection_id,
            "dataset_path": relative_project_path(dataset_dir),
            "best_iteration": best_iteration,
            "best_val_aft_nloglik": row["val_aft_nloglik"],
            "val_c_index": val_row["c_index"],
            "test_c_index": test_row["c_index"],
            "test_ipcw_brier_mean": test_row["ipcw_brier_mean"],
            "created_at": created_at,
            "status": "completed",
        },
        id_column="run_id",
        columns=RUN_INDEX_COLUMNS,
    )
    return {
        "run_id": run_id,
        "best_trial_id": best_trial["trial_id"],
        "best_val_aft_nloglik": row["val_aft_nloglik"],
        "test_c_index": test_row["c_index"],
        "test_ipcw_brier_mean": test_row["ipcw_brier_mean"],
        "results_path": relative_project_path(output_dir),
        "model_path": relative_project_path(checkpoint_dir),
        "created_at": created_at,
    }


def main() -> None:
    """Run survival XGBoost-AFT grid search."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    num_trials = validate_config(config)
    dataset_dir = project_path(config["dataset"]["path"])
    dataset_metadata = load_yaml_config(dataset_dir / "dataset_metadata.yaml")
    selection_id = str(dataset_metadata["selection_id"])
    search_id = str(config["search_id"])
    if args.search_suffix:
        search_id = f"{search_id}__{sanitize_id(args.search_suffix)}"
    search_dir = grid_search_dir(selection_id=selection_id, search_id=search_id)
    tables_dir = search_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    devices = (
        ["cpu"]
        if args.no_parallel
        else choose_trial_devices(config.get("parallel", {}))
    )
    print("=== Survival-Persistence XGBoost-AFT Grid Search ===")
    print(f"Config: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Search ID: {search_id}")
    print(f"Dataset: {dataset_dir.relative_to(PROJECT_ROOT)}")
    print(f"Selection: {selection_id}")
    effective_trials = num_trials
    if args.max_trials is not None:
        if args.max_trials < 1:
            raise ValueError("--max-trials must be positive.")
        effective_trials = min(num_trials, int(args.max_trials))
    print(f"Trials: {effective_trials} of {num_trials}")
    print(f"Devices: {devices}")
    print("Selection metric: val_aft_nloglik (min)")
    print("Trial phase uses train+validation; test is evaluated only for the selected model.\n")
    if args.dry_run:
        return
    require_xgboost()
    parameter_sets = expand_parameter_grid(config["parameter_grid"])[:effective_trials]
    jobs = []
    for index, parameters in enumerate(parameter_sets, start=1):
        trial_id = make_trial_id(parameters, index)
        trial_config = build_trial_config(config, parameters)
        row_base = {
            "trial_id": trial_id,
            "trial_index": index,
            "parameters": parameters,
            "feature_set": trial_config["dataset"]["feature_set"],
            "sample_weighting": trial_config["dataset"]["sample_weighting"],
            "aft_loss_distribution": trial_config["model"]["aft_loss_distribution"],
            "aft_loss_distribution_scale": trial_config["model"]["aft_loss_distribution_scale"],
        }
        jobs.append(
            {
                "dataset_path": str(dataset_dir),
                "trial_config": trial_config,
                "row_base": row_base,
            }
        )
    trial_rows = []
    if args.finalize_existing:
        trials_path = tables_dir / "trials.csv"
        if not trials_path.exists():
            raise FileNotFoundError(f"Cannot finalize existing grid; missing {trials_path}")
        trials = pd.read_csv(trials_path)
    else:
        for row in iter_parallel_trial_results(jobs, run_trial_job, devices):
            trial_rows.append(row)
            print(
                format_trial_start(
                    index=int(row["trial_index"]),
                    total=effective_trials,
                    run_id=f"survival_xgboost_aft_{row['feature_set']}_{row['sample_weighting']}",
                    trial_id=str(row["trial_id"]),
                    device=str(row["device"]),
                    parameters=dict(row["parameters"]) if isinstance(row["parameters"], dict) else row["parameters"],
                ),
                flush=True,
            )
            if row["status"] == "complete":
                print(
                    f"  completed {row['trial_id']} val_aft_nloglik={row['val_aft_nloglik']:.6f} "
                    f"val_c_index={row['val_c_index']:.6f}",
                    flush=True,
                )
            else:
                print(f"  failed {row['trial_id']}: {row.get('error', '')}", flush=True)
        trials = pd.DataFrame(trial_rows)
        trials.to_csv(tables_dir / "trials.csv", index=False)
        save_yaml(search_dir / "config_resolved.yaml", config)
        save_yaml(
            search_dir / "metadata.yaml",
            {
                "task_name": TASK_NAME,
                "model_name": "xgboost_aft",
                "search_id": search_id,
                "config_path": relative_project_path(config_path),
                "config_fingerprint": config_fingerprint(config),
                "selection_id": selection_id,
                "dataset_path": relative_project_path(dataset_dir),
                "num_trials": num_trials,
                "num_executed_trials": effective_trials,
                "devices": devices,
                "skip_finalize": bool(args.skip_finalize),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    if args.skip_finalize:
        print("Skipping best-run finalization as requested.")
        print(f"Smoke/grid trial results: {search_dir.relative_to(PROJECT_ROOT)}")
        return
    best = select_best_trial(
        trials,
        metric=str(config["search"]["selection_metric"]),
        mode=str(config["search"]["selection_mode"]),
    )
    best_parameters = best["parameters"]
    if isinstance(best_parameters, str):
        import ast
        best_parameters = ast.literal_eval(best_parameters)
    best_config = build_trial_config(config, best_parameters)
    best_run = finalize_best_run(
        config=config,
        dataset_dir=dataset_dir,
        selection_id=selection_id,
        best_trial=best,
        best_config=best_config,
        search_dir=search_dir,
        device=devices[0],
        overwrite=bool(config.get("output", {}).get("overwrite", False)) or args.force,
    )
    best_runs = pd.DataFrame([best_run])
    best_runs.to_csv(tables_dir / "best_runs.csv", index=False)
    upsert_index_row(
        get_results_index_dir() / "grid_searches.csv",
        {
            "search_id": search_id,
            "model_family": "xgboost",
            "target_run_id": best_run["run_id"],
            "selection_id": selection_id,
            "selection_metric": config["search"]["selection_metric"],
            "best_trial_id": best_run["best_trial_id"],
            "results_path": relative_project_path(search_dir),
            "best_model_path": best_run["model_path"],
            "status": "completed",
            "created_at": best_run["created_at"],
        },
        id_column="search_id",
        columns=GRID_SEARCH_INDEX_COLUMNS,
    )
    print("=== Selected best survival XGBoost-AFT run ===")
    print(best_runs.to_string(index=False))
    print(f"Grid results: {search_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
