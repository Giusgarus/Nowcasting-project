"""Validation-only grid search for the survival-persistence discrete-time TCN."""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.survival_persistence.run_discrete_time_tcn_controlled import (  # noqa: E402
    build_prediction_frame,
    count_parameters,
    dataset_fingerprint,
    feature_set_name,
    load_split_artifacts,
    make_bin_spec_from_config,
    make_loaders,
    model_config_from_resolved,
    prepare_resolved_config,
    project_path,
    train_model,
)
from src.tasks.survival_persistence.evaluation.controlled_tcn import (  # noqa: E402
    effective_receptive_field_samples,
    fingerprint_mapping,
    summarize_survival_predictions,
)
from src.tasks.survival_persistence.evaluation.metrics import fit_censoring_survival  # noqa: E402
from src.tasks.survival_persistence.models.discrete_time_tcn import (  # noqa: E402
    MODEL_ID_DISCRETE_TIME_TCN,
    DiscreteTimeTCNSurvivalModel,
)
from src.tasks.survival_persistence.utils.paths import (  # noqa: E402
    RUN_INDEX_COLUMNS,
    TASK_NAME,
    grid_search_dir,
    make_run_id,
    model_dir,
    model_selection_dir,
    run_dir,
    run_index_path,
)
from src.tuning.grid_search import (  # noqa: E402
    apply_flat_overrides,
    expand_parameter_grid,
    make_trial_id,
    select_best_trial,
)
from src.tuning.parallel_trials import (  # noqa: E402
    choose_trial_devices,
    iter_parallel_trial_results,
    prepare_trial_device,
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
    / "configs/survival_persistence/models/discrete_time_tcn_grid_search.yaml"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

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
        help="Run only the first N expanded trials; useful for smoke tests.",
    )
    parser.add_argument(
        "--skip-finalize",
        action="store_true",
        help="Run trials and write trials.csv without publishing/evaluating the best run.",
    )
    parser.add_argument(
        "--search-suffix",
        default=None,
        help="Append a suffix to the configured search_id, e.g. smoke.",
    )
    return parser.parse_args()


def build_trial_config(config: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """Build one single-run config from the fixed base plus flat overrides."""

    trial_config = apply_flat_overrides(config["base"], parameters)
    trial_config["seed"] = int(trial_config.get("seed", config.get("seed", 42)))
    trial_config.setdefault("training", {})["seed"] = int(
        trial_config["training"].get("seed", trial_config["seed"])
    )
    return trial_config


def validate_config(config: dict[str, Any]) -> int:
    """Validate grid config and return the full Cartesian trial count."""

    if config.get("task_name") != TASK_NAME:
        raise ValueError(f"Config task_name must be {TASK_NAME}.")
    if config.get("model_name") != "discrete_time_tcn_grid_search":
        raise ValueError("Config model_name must be discrete_time_tcn_grid_search.")
    required = ["search_id", "dataset", "search", "parallel", "base", "parameter_grid"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Discrete-time TCN grid config missing sections: {missing}")
    search = config["search"]
    if search.get("selection_metric") != "validation_integrated_brier_score":
        raise ValueError("selection_metric must be validation_integrated_brier_score.")
    if search.get("selection_mode") != "min":
        raise ValueError("selection_mode must be min.")
    trials = expand_parameter_grid(config["parameter_grid"])
    max_trials = int(search["max_trials"])
    if len(trials) > max_trials:
        raise ValueError(
            f"Configured survival TCN grid has {len(trials)} trials, exceeding "
            f"search.max_trials={max_trials}. Increase the safety limit explicitly; "
            "no combinations were skipped."
        )
    return len(trials)


def trial_dir(search_dir: Path, trial_id: str) -> Path:
    """Return the artifact directory for one validation trial."""

    return search_dir / "trials" / sanitize_id(trial_id)


def completed_trial_is_valid(path: Path, expected_fingerprint: str) -> bool:
    """Return true when a prior trial can be reused."""

    metadata_path = path / "metadata.yaml"
    metrics_path = path / "validation_metrics.yaml"
    checkpoint_path = path / "best_model.pt"
    if not metadata_path.exists() or not metrics_path.exists() or not checkpoint_path.exists():
        return False
    try:
        metadata = load_yaml_config(metadata_path)
    except Exception:
        return False
    return (
        metadata.get("status") == "complete"
        and metadata.get("trial_fingerprint") == expected_fingerprint
    )


def run_trial_job(job: dict[str, Any]) -> dict[str, Any]:
    """Train one validation-only TCN trial on its assigned worker device."""

    dataset_dir = Path(job["dataset_dir"])
    config = job["trial_config"]
    parameters = job["parameters"]
    trial_id = str(job["trial_id"])
    path = Path(job["trial_dir"])
    trial_fingerprint = str(job["trial_fingerprint"])
    save_trial_predictions = bool(job["save_trial_predictions"])
    device_name = str(job["device"])
    row_base = {
        "trial_id": trial_id,
        "trial_index": int(job["trial_index"]),
        "parameters": parameters,
        "feature_set": feature_set_name(config),
        "sample_weighting": str(config["dataset"].get("sample_weighting", "uniform")),
        "device": device_name,
        "trial_fingerprint": trial_fingerprint,
        "status": "running",
    }
    if completed_trial_is_valid(path, trial_fingerprint):
        metrics = load_yaml_config(path / "validation_metrics.yaml")
        metrics.update(row_base)
        metrics["status"] = "complete"
        metrics["reused_existing"] = True
        return metrics

    start = time.perf_counter()
    path.mkdir(parents=True, exist_ok=True)
    try:
        seed = int(config.get("seed", 42))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        arrays, metadata = load_split_artifacts(dataset_dir, splits=("train", "validation"))
        bin_spec = make_bin_spec_from_config(config, arrays["train"])
        resolved = prepare_resolved_config(config, train_arrays=arrays["train"], bin_spec=bin_spec)
        device = prepare_trial_device(device_name)
        loaders = make_loaders(arrays, bin_spec=bin_spec, config=resolved)
        model = DiscreteTimeTCNSurvivalModel(model_config_from_resolved(resolved))
        parameter_count = count_parameters(model)
        checkpoint_path = path / "best_model.pt"
        history = train_model(
            model=model,
            loaders=loaders,
            config=resolved,
            device=device,
            checkpoint_path=checkpoint_path,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        val_frame, val_arrays, val_loss = build_prediction_frame(
            split="validation",
            model=model,
            loader=loaders["validation"],
            metadata=metadata["validation"],
            arrays=arrays["validation"],
            bin_spec=bin_spec,
            config=resolved,
            feature_set=feature_set_name(resolved),
            sample_weighting=str(resolved["dataset"].get("sample_weighting", "uniform")),
            device=device,
        )
        censoring_curve = fit_censoring_survival(
            arrays["train"]["y_time_seconds"],
            arrays["train"]["y_event_observed"],
        )
        horizons = np.asarray(resolved["prediction"]["horizons_seconds"], dtype=float)
        metrics, brier, calibration = summarize_survival_predictions(
            predictions=val_frame,
            survival_by_bin=val_arrays["survival_by_bin"],
            bin_spec=bin_spec,
            horizons_seconds=horizons,
            censoring_curve=censoring_curve,
            discrete_nll=val_loss,
            prefix="validation",
            calibration_horizons_seconds=resolved["evaluation"]["calibration_horizons_seconds"],
            calibration_bins=int(resolved["evaluation"].get("calibration_bins", 10)),
        )
        best_epoch = int(history.loc[history["validation_discrete_nll"].idxmin(), "epoch"])
        metrics.update(
            {
                **row_base,
                "status": "complete",
                "reused_existing": False,
                "seed": seed,
                "number_of_bins": bin_spec.num_bins,
                "parameter_count": parameter_count,
                "effective_receptive_field_samples": effective_receptive_field_samples(
                    kernel_size=int(resolved["model"]["kernel_size"]),
                    dilations=resolved["model"]["dilations"],
                    convolutions_per_block=2,
                ),
                "best_epoch": best_epoch,
                "runtime_seconds": time.perf_counter() - start,
                "checkpoint_path": relative_project_path(checkpoint_path),
                "adapter_fingerprint": fingerprint_mapping(
                    {
                        "bin_spec": bin_spec.to_dict(),
                        "likelihood": "discrete_time_masked_hazard_nll",
                        "censoring_convention": "full_survived_bins_only",
                    }
                ),
                "bin_edges_fingerprint": fingerprint_mapping(bin_spec.to_dict()),
            }
        )
        history.to_csv(path / "training_history.csv", index=False)
        brier.to_csv(path / "validation_brier_by_horizon.csv", index=False)
        calibration.to_csv(path / "validation_calibration.csv", index=False)
        if save_trial_predictions:
            val_frame.to_parquet(path / "validation_predictions.parquet", index=False)
        save_yaml(path / "validation_metrics.yaml", metrics)
        save_yaml(path / "config_resolved.yaml", resolved)
        save_yaml(
            path / "metadata.yaml",
            {
                "task_name": TASK_NAME,
                "model_id": MODEL_ID_DISCRETE_TIME_TCN,
                "trial_id": trial_id,
                "status": "complete",
                "selection_scope": "validation_only",
                "test_usage": "not_loaded_or_evaluated",
                "trial_fingerprint": trial_fingerprint,
                "checkpoint_path": relative_project_path(checkpoint_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return metrics
    except Exception as error:
        failure = {
            **row_base,
            "status": "failed",
            "error": str(error),
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - start,
        }
        save_yaml(path / "metadata.yaml", failure)
        return failure


def build_jobs(
    *,
    config: dict[str, Any],
    dataset_dir: Path,
    search_dir: Path,
    parameter_sets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build deterministic trial jobs from expanded parameter sets."""

    jobs = []
    for index, parameters in enumerate(parameter_sets, start=1):
        trial_id = make_trial_id(parameters, index)
        trial_config = build_trial_config(config, parameters)
        trial_fingerprint = config_fingerprint(
            {
                "trial_config": trial_config,
                "parameters": parameters,
                "trial_id": trial_id,
                "dataset_path": relative_project_path(dataset_dir),
            }
        )
        jobs.append(
            {
                "dataset_dir": str(dataset_dir),
                "trial_config": trial_config,
                "parameters": parameters,
                "trial_id": trial_id,
                "trial_index": index,
                "trial_dir": str(trial_dir(search_dir, trial_id)),
                "trial_fingerprint": trial_fingerprint,
                "save_trial_predictions": bool(
                    config["search"].get("save_trial_predictions", False)
                ),
            }
        )
    return jobs


def parse_parameters(value: Any) -> dict[str, Any]:
    """Parse parameters from a live dict or a CSV string representation."""

    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    raise TypeError(f"Unsupported parameter record type: {type(value)!r}")


def metrics_summary_frame(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Return compact validation/test metric rows for a published run."""

    rows = []
    for split, metrics in [
        ("validation", validation_metrics),
        ("test", test_metrics),
    ]:
        rows.append(
            {
                "split": split,
                "discrete_nll": metrics.get(f"{split}_discrete_nll"),
                "integrated_brier_score": metrics.get(f"{split}_integrated_brier_score"),
                "harrell_c_index": metrics.get(f"{split}_harrell_c_index"),
                "brier_300s": metrics.get(f"{split}_brier_300s"),
                "calibration_error_300s": metrics.get(f"{split}_calibration_error_300s"),
                "activation_brier_300s": metrics.get(f"{split}_activation_brier_300s"),
                "event_balanced_brier_300s": metrics.get(f"{split}_event_balanced_brier_300s"),
                "undefined_median_fraction": metrics.get("undefined_median_fraction"),
            }
        )
    return pd.DataFrame(rows)


def build_tcn_run_index_row(
    *,
    run_id: str,
    feature_set: str,
    sample_weighting: str,
    selection_id: str,
    dataset_dir: Path,
    best_epoch: int,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    """Return a complete survival run-index row for a discrete-time TCN run."""

    return {
        "run_id": run_id,
        "task_name": TASK_NAME,
        "model_family": "tcn",
        "model_id": MODEL_ID_DISCRETE_TIME_TCN,
        "feature_set": feature_set,
        "sample_weighting": sample_weighting,
        "selection_id": selection_id,
        "dataset_path": relative_project_path(dataset_dir),
        "best_iteration": int(best_epoch),
        "best_val_aft_nloglik": np.nan,
        "best_val_discrete_nll": validation_metrics.get("validation_discrete_nll"),
        "val_c_index": validation_metrics.get("validation_harrell_c_index"),
        "val_ipcw_brier_mean": validation_metrics.get("validation_integrated_brier_score"),
        "test_c_index": test_metrics.get("test_harrell_c_index"),
        "test_ipcw_brier_mean": test_metrics.get("test_integrated_brier_score"),
        "created_at": created_at,
        "status": "complete",
    }


def finalize_best_run(
    *,
    config: dict[str, Any],
    config_path: Path,
    dataset_dir: Path,
    selection_id: str,
    search_id: str,
    search_dir: Path,
    best_trial: pd.Series,
    device_name: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Publish the selected grid checkpoint and evaluate external test once."""

    best_parameters = parse_parameters(best_trial["parameters"])
    best_config = build_trial_config(config, best_parameters)
    arrays, metadata = load_split_artifacts(dataset_dir, splits=("train", "validation", "test"))
    bin_spec = make_bin_spec_from_config(best_config, arrays["train"])
    resolved = prepare_resolved_config(best_config, train_arrays=arrays["train"], bin_spec=bin_spec)
    device = prepare_trial_device(device_name)
    loaders = make_loaders(arrays, bin_spec=bin_spec, config=resolved)
    source_checkpoint = project_path(best_trial["checkpoint_path"])
    model = DiscreteTimeTCNSurvivalModel(model_config_from_resolved(resolved))
    checkpoint = torch.load(source_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    feature_set = feature_set_name(resolved)
    sample_weighting = str(resolved["dataset"].get("sample_weighting", "uniform"))
    run_id = make_run_id(
        model_id=MODEL_ID_DISCRETE_TIME_TCN,
        feature_set=feature_set,
        sample_weighting=sample_weighting,
        selection_id=selection_id,
        run_suffix="grid_extended_best",
    )
    output_dir = run_dir(selection_id=selection_id, run_id=run_id)
    checkpoint_dir = model_dir(model_id=MODEL_ID_DISCRETE_TIME_TCN, run_id=run_id)
    if overwrite:
        for path in (output_dir, checkpoint_dir):
            if path.exists():
                shutil.rmtree(path)
    elif output_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"Best run already exists: {output_dir}")
    for directory in [
        output_dir / "metrics",
        output_dir / "predictions",
        output_dir / "tables",
        checkpoint_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    censoring_curve = fit_censoring_survival(
        arrays["train"]["y_time_seconds"],
        arrays["train"]["y_event_observed"],
    )
    horizons = np.asarray(resolved["prediction"]["horizons_seconds"], dtype=float)
    split_metrics: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        frame, prediction_arrays, split_loss = build_prediction_frame(
            split=split,
            model=model,
            loader=loaders[split],
            metadata=metadata[split],
            arrays=arrays[split],
            bin_spec=bin_spec,
            config=resolved,
            feature_set=feature_set,
            sample_weighting=sample_weighting,
            device=device,
        )
        frame.to_parquet(output_dir / "predictions" / f"{split}_predictions.parquet", index=False)
        metrics, brier, calibration = summarize_survival_predictions(
            predictions=frame,
            survival_by_bin=prediction_arrays["survival_by_bin"],
            bin_spec=bin_spec,
            horizons_seconds=horizons,
            censoring_curve=censoring_curve,
            discrete_nll=split_loss,
            prefix=split,
            calibration_horizons_seconds=resolved["evaluation"]["calibration_horizons_seconds"],
            calibration_bins=int(resolved["evaluation"].get("calibration_bins", 10)),
        )
        split_metrics[split] = metrics
        brier.to_csv(output_dir / "metrics" / f"{split}_brier_by_horizon.csv", index=False)
        calibration.to_csv(output_dir / "metrics" / f"{split}_calibration.csv", index=False)

    metrics_summary = metrics_summary_frame(
        split_metrics["validation"],
        split_metrics["test"],
    )
    metrics_summary.to_csv(output_dir / "metrics" / "metrics_summary.csv", index=False)
    source_history = trial_dir(search_dir, str(best_trial["trial_id"])) / "training_history.csv"
    if source_history.exists():
        shutil.copy2(source_history, output_dir / "tables" / "training_history.csv")
    target_checkpoint = checkpoint_dir / "best_model.pt"
    shutil.copy2(source_checkpoint, target_checkpoint)
    save_yaml(output_dir / "config_resolved.yaml", resolved)
    save_yaml(checkpoint_dir / "model_config.yaml", resolved)

    created_at = datetime.now(timezone.utc).isoformat()
    metadata_out = {
        "task_name": TASK_NAME,
        "model_family": "tcn",
        "model_id": MODEL_ID_DISCRETE_TIME_TCN,
        "run_id": run_id,
        "source_grid_search": relative_project_path(search_dir),
        "best_trial_id": str(best_trial["trial_id"]),
        "dataset_path": relative_project_path(dataset_dir),
        "selection_id": selection_id,
        "feature_set": feature_set,
        "sample_weighting": sample_weighting,
        "best_epoch": int(best_trial["best_epoch"]),
        "best_validation_integrated_brier_score": float(
            best_trial["validation_integrated_brier_score"]
        ),
        "checkpoint_path": relative_project_path(target_checkpoint),
        "created_at": created_at,
    }
    save_yaml(output_dir / "metadata.yaml", metadata_out)
    save_yaml(checkpoint_dir / "training_metadata.yaml", metadata_out)
    upsert_index_row(
        run_index_path(),
        build_tcn_run_index_row(
            run_id=run_id,
            feature_set=feature_set,
            sample_weighting=sample_weighting,
            selection_id=selection_id,
            dataset_dir=dataset_dir,
            best_epoch=int(best_trial["best_epoch"]),
            validation_metrics=split_metrics["validation"],
            test_metrics=split_metrics["test"],
            created_at=created_at,
        ),
        id_column="run_id",
        columns=RUN_INDEX_COLUMNS,
    )
    return {
        "run_id": run_id,
        "model_family": "tcn",
        "model_id": MODEL_ID_DISCRETE_TIME_TCN,
        "feature_set": feature_set,
        "sample_weighting": sample_weighting,
        "best_trial_id": str(best_trial["trial_id"]),
        "best_validation_integrated_brier_score": float(
            best_trial["validation_integrated_brier_score"]
        ),
        "test_integrated_brier_score": split_metrics["test"].get("test_integrated_brier_score"),
        "test_harrell_c_index": split_metrics["test"].get("test_harrell_c_index"),
        "results_path": relative_project_path(output_dir),
        "model_path": relative_project_path(checkpoint_dir),
        "predictions_path": relative_project_path(
            output_dir / "predictions" / "test_predictions.parquet"
        ),
        "metrics_path": relative_project_path(output_dir / "metrics"),
        "created_at": created_at,
    }


def main() -> None:
    """Run the survival TCN grid search."""

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
    devices = ["cpu"] if args.no_parallel else choose_trial_devices(config.get("parallel", {}))
    effective_trials = num_trials
    if args.max_trials is not None:
        if args.max_trials < 1:
            raise ValueError("--max-trials must be positive.")
        effective_trials = min(num_trials, int(args.max_trials))

    print("=== Survival-Persistence Discrete-Time TCN Grid Search ===")
    print(f"Config: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Search ID: {search_id}")
    print(f"Dataset: {dataset_dir.relative_to(PROJECT_ROOT)}")
    print(f"Selection: {selection_id}")
    print(f"Trials: {effective_trials} of {num_trials}")
    print(f"Devices: {devices}")
    print("Selection metric: validation_integrated_brier_score (min)")
    print("Trial phase uses train+validation; test is evaluated only for the selected model.\n")
    if args.dry_run:
        return

    if search_dir.exists() and args.force and not args.finalize_existing:
        shutil.rmtree(search_dir)
    search_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    parameter_sets = expand_parameter_grid(config["parameter_grid"])[:effective_trials]
    jobs = build_jobs(
        config=config,
        dataset_dir=dataset_dir,
        search_dir=search_dir,
        parameter_sets=parameter_sets,
    )
    if args.finalize_existing:
        trials_path = tables_dir / "trials.csv"
        if not trials_path.exists():
            raise FileNotFoundError(f"Cannot finalize existing grid; missing {trials_path}")
        trials = pd.read_csv(trials_path)
    else:
        rows = []
        for row in iter_parallel_trial_results(jobs, run_trial_job, devices):
            rows.append(row)
            print(
                format_trial_start(
                    index=int(row["trial_index"]),
                    total=effective_trials,
                    run_id=f"survival_tcn_{row['feature_set']}_{row['sample_weighting']}",
                    trial_id=str(row["trial_id"]),
                    device=str(row["device"]),
                    parameters=dict(row["parameters"]) if isinstance(row["parameters"], dict) else row["parameters"],
                ),
                flush=True,
            )
            if row["status"] == "complete":
                print(
                    f"  completed {row['trial_id']} "
                    f"val_ibs={row['validation_integrated_brier_score']:.6f} "
                    f"val_c_index={row['validation_harrell_c_index']:.6f}",
                    flush=True,
                )
            else:
                print(f"  failed {row['trial_id']}: {row.get('error', '')}", flush=True)
            pd.DataFrame(rows).to_csv(tables_dir / "trials.csv", index=False)
        trials = pd.DataFrame(rows)
        save_yaml(search_dir / "config_resolved.yaml", config)
        save_yaml(
            search_dir / "metadata.yaml",
            {
                "task_name": TASK_NAME,
                "model_name": "discrete_time_tcn_grid_search",
                "search_id": search_id,
                "config_path": relative_project_path(config_path),
                "config_fingerprint": config_fingerprint(config),
                "selection_id": selection_id,
                "dataset_path": relative_project_path(dataset_dir),
                "dataset_fingerprint_train_validation": dataset_fingerprint(
                    dataset_dir,
                    include_test=False,
                ),
                "num_trials": num_trials,
                "num_executed_trials": effective_trials,
                "devices": devices,
                "skip_finalize": bool(args.skip_finalize),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    if args.skip_finalize:
        print("Skipping best-run finalization as requested.")
        print(f"Grid trial results: {search_dir.relative_to(PROJECT_ROOT)}")
        return

    best = select_best_trial(
        trials,
        metric=str(config["search"]["selection_metric"]),
        mode=str(config["search"]["selection_mode"]),
    )
    best_run = finalize_best_run(
        config=config,
        config_path=config_path,
        dataset_dir=dataset_dir,
        selection_id=selection_id,
        search_id=search_id,
        search_dir=search_dir,
        best_trial=best,
        device_name=devices[0],
        overwrite=bool(config.get("output", {}).get("overwrite", False)) or args.force,
    )
    best_runs = pd.DataFrame([best_run])
    best_runs.to_csv(tables_dir / "best_runs.csv", index=False)
    selection_dir = model_selection_dir(selection_id=selection_id, search_id=search_id)
    (selection_dir / "tables").mkdir(parents=True, exist_ok=True)
    best_runs.to_csv(selection_dir / "tables" / "best_runs.csv", index=False)
    save_yaml(
        selection_dir / "metadata.yaml",
        {
            "search_id": search_id,
            "task_name": TASK_NAME,
            "model_family": "tcn",
            "selection_id": selection_id,
            "selection_metric": config["search"]["selection_metric"],
            "selection_mode": config["search"]["selection_mode"],
            "source_grid_search_path": relative_project_path(search_dir),
            "summary_table": relative_project_path(selection_dir / "tables" / "best_runs.csv"),
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    upsert_index_row(
        get_results_index_dir() / "grid_searches.csv",
        {
            "search_id": search_id,
            "model_family": "tcn",
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
    print("=== Selected best survival TCN grid run ===")
    print(best_runs.to_string(index=False))
    print(f"Grid results: {search_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
