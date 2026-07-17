"""Train and evaluate the survival-persistence XGBoost-AFT baseline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.survival_persistence.evaluation.baselines import (  # noqa: E402
    constant_median_baseline,
    kaplan_meier_marginal_baseline,
)
from src.tasks.survival_persistence.evaluation.metrics import (  # noqa: E402
    brier_scores_by_horizon,
    calibration_by_horizon,
    event_level_bootstrap,
    fit_censoring_survival,
    harrell_c_index,
)
from src.tasks.survival_persistence.models.xgboost_aft import (  # noqa: E402
    aft_label_bounds,
    aft_negative_log_likelihood,
    build_dmatrix,
    build_feature_matrix,
    predicted_mean_seconds,
    predicted_median_seconds,
    require_xgboost,
    survival_probability_at_horizon,
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
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.results_paths import relative_project_path, upsert_index_row  # noqa: E402

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/survival_persistence/models/xgboost_aft_scalar_context.yaml"
)
SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}


def project_path(path: str | Path) -> Path:
    """Resolve an absolute or repository-relative path."""

    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_split_artifacts(dataset_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame]]:
    """Load canonical split arrays and aligned metadata."""

    arrays: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for split, stem in SPLIT_FILES.items():
        npz_path = dataset_dir / f"{stem}.npz"
        metadata_path = dataset_dir / f"{stem}_metadata.parquet"
        if not npz_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing survival split artifact for {split}: {dataset_dir}")
        with np.load(npz_path, allow_pickle=True) as data:
            arrays[split] = {key: data[key] for key in data.files}
        frame = pd.read_parquet(metadata_path)
        if len(frame) != len(arrays[split]["y_time_seconds"]):
            raise ValueError(f"{split} metadata and arrays are not aligned.")
        metadata[split] = frame.reset_index(drop=True)
    return arrays, metadata


def directory_has_files(path: Path) -> bool:
    """Return true when a directory already contains persisted files."""

    return path.exists() and any(child.is_file() for child in path.rglob("*"))


def training_history_frame(evals_result: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    """Convert XGBoost eval history into a tidy table."""

    rows: list[dict[str, Any]] = []
    max_rounds = max(len(metrics) for split in evals_result.values() for metrics in split.values())
    for iteration in range(max_rounds):
        row: dict[str, Any] = {"boosting_round": iteration}
        for split, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                if iteration < len(values):
                    row[f"{split}_{metric_name}"] = values[iteration]
        rows.append(row)
    return pd.DataFrame(rows)


def make_predictions_frame(
    *,
    split: str,
    metadata: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    location: np.ndarray,
    distribution: str,
    scale: float,
    horizons_seconds: np.ndarray,
    model_family: str,
    model_id: str,
    feature_set: str,
    sample_weighting: str,
) -> tuple[pd.DataFrame, str]:
    """Build the saved prediction dataframe for one split."""

    median = predicted_median_seconds(location, scale=scale, distribution=distribution)
    mean, mean_reason = predicted_mean_seconds(
        location,
        scale=scale,
        distribution=distribution,
    )
    survival = survival_probability_at_horizon(
        location,
        horizons_seconds,
        scale=scale,
        distribution=distribution,
    )
    required_metadata = [
        "sample_time",
        "dataset_name",
        "dataset_id",
        "global_event_id",
        "global_window_id",
        "current_above_threshold",
        "event_balanced_weight",
        "elapsed_since_event_start_seconds",
        "elapsed_event_samples",
    ]
    missing = [column for column in required_metadata if column not in metadata.columns]
    if missing:
        raise ValueError(f"Prediction metadata missing columns: {missing}")
    frame = metadata.loc[:, required_metadata].copy()
    frame.insert(0, "split", split)
    frame["y_time_seconds"] = arrays["y_time_seconds"]
    frame["y_event_observed"] = arrays["y_event_observed"]
    frame["y_lower_bound_seconds"] = arrays["y_lower_bound_seconds"]
    frame["y_upper_bound_seconds"] = arrays["y_upper_bound_seconds"]
    frame["model_family"] = model_family
    frame["model_id"] = model_id
    frame["feature_set"] = feature_set
    frame["sample_weighting"] = sample_weighting
    frame["aft_location_prediction"] = location
    frame["predicted_median_remaining_seconds"] = median
    frame["predicted_mean_remaining_seconds"] = (
        mean if mean is not None else np.full(len(frame), np.nan)
    )
    for index, horizon in enumerate(horizons_seconds):
        frame[f"survival_probability_{int(horizon)}s"] = survival[:, index]
    return frame, mean_reason


def metrics_for_predictions(
    *,
    method_name: str,
    split: str,
    predictions: pd.DataFrame,
    horizons_seconds: np.ndarray,
    censoring_curve,
    distribution: str | None = None,
    scale: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute split-level metrics and horizon Brier scores."""

    times = predictions["y_time_seconds"].to_numpy(dtype=float)
    observed = predictions["y_event_observed"].to_numpy(dtype=int)
    median = predictions["predicted_median_remaining_seconds"].to_numpy(dtype=float)
    cindex = harrell_c_index(times, observed, median, higher_prediction_longer=True)
    survival = predictions[
        [f"survival_probability_{int(horizon)}s" for horizon in horizons_seconds]
    ].to_numpy(dtype=float)
    brier = brier_scores_by_horizon(
        times,
        observed,
        survival,
        horizons_seconds,
        censoring_curve=censoring_curve,
    )
    metric_row: dict[str, Any] = {
        "method": method_name,
        "split": split,
        "num_samples": int(len(predictions)),
        "num_events": int(predictions["global_event_id"].nunique()),
        "c_index": cindex["c_index"],
        "num_comparable_pairs": cindex["num_comparable_pairs"],
        "ipcw_brier_mean": float(brier["ipcw_brier"].mean()),
    }
    if distribution is not None and scale is not None:
        metric_row["aft_nloglik"] = aft_negative_log_likelihood(
            predictions["aft_location_prediction"].to_numpy(dtype=float),
            predictions["y_lower_bound_seconds"].to_numpy(dtype=float),
            predictions["y_upper_bound_seconds"].to_numpy(dtype=float),
            scale=scale,
            distribution=distribution,
        )
    brier.insert(0, "method", method_name)
    brier.insert(1, "split", split)
    return metric_row, brier


def calibration_tables(
    *,
    predictions: pd.DataFrame,
    horizons_seconds: list[float],
    n_bins: int,
    censoring_curve,
) -> pd.DataFrame:
    """Create calibration tables for configured horizons."""

    rows = []
    for horizon in horizons_seconds:
        column = f"survival_probability_{int(horizon)}s"
        if column not in predictions.columns:
            continue
        table = calibration_by_horizon(
            predictions["y_time_seconds"].to_numpy(dtype=float),
            predictions["y_event_observed"].to_numpy(dtype=int),
            predictions[column].to_numpy(dtype=float),
            horizon_seconds=float(horizon),
            n_bins=n_bins,
            censoring_curve=censoring_curve,
        )
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def event_timepoint_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select event-level diagnostic rows at operational elapsed-time slices."""

    rows = []
    targets = [
        ("activation_sample", 0.0),
        ("first_valid_context_sample", None),
        ("after_30s", 30.0),
        ("after_60s", 60.0),
        ("after_120s", 120.0),
        ("after_300s", 300.0),
    ]
    ordered = predictions.sort_values(
        ["global_event_id", "elapsed_since_event_start_seconds", "sample_time"],
        kind="stable",
    )
    for event_id, group in ordered.groupby("global_event_id", sort=False):
        for label, elapsed in targets:
            if elapsed is None:
                selected = group.iloc[[0]]
            else:
                candidates = group.loc[group["elapsed_since_event_start_seconds"] >= elapsed]
                if candidates.empty:
                    continue
                selected = candidates.iloc[[0]]
            row = selected.copy()
            row.insert(0, "event_timepoint", label)
            rows.append(row)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_plots(
    *,
    figures_dir: Path,
    training_history: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_brier: pd.DataFrame,
    calibration: pd.DataFrame,
    horizons_seconds: np.ndarray,
) -> None:
    """Save repository-consistent diagnostic plots with matplotlib only."""

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    if not training_history.empty:
        metric_columns = [column for column in training_history.columns if "aft-nloglik" in column]
        if metric_columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            for column in metric_columns:
                ax.plot(training_history["boosting_round"], training_history[column], label=column)
            ax.set_xlabel("Boosting round")
            ax.set_ylabel("AFT negative log-likelihood")
            ax.set_title("XGBoost-AFT training history")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures_dir / "training_aft_nloglik.png", dpi=150)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(test_predictions["predicted_median_remaining_seconds"], bins=40)
    ax.set_xlabel("Predicted median remaining seconds")
    ax.set_ylabel("Count")
    ax.set_title("External-test predicted median distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "test_predicted_median_distribution.png", dpi=150)
    plt.close(fig)

    p300_column = "survival_probability_300s"
    if p300_column in test_predictions.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(test_predictions[p300_column], bins=40)
        ax.set_xlabel("Predicted P(R > 300 s)")
        ax.set_ylabel("Count")
        ax.set_title("External-test P(R > 300 s) distribution")
        fig.tight_layout()
        fig.savefig(figures_dir / "test_survival_probability_300s_distribution.png", dpi=150)
        plt.close(fig)

    if not test_brier.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(test_brier["horizon_seconds"], test_brier["ipcw_brier"], marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("Horizon seconds")
        ax.set_ylabel("IPCW Brier score")
        ax.set_title("External-test Brier score by horizon")
        fig.tight_layout()
        fig.savefig(figures_dir / "test_brier_by_horizon.png", dpi=150)
        plt.close(fig)

    if not calibration.empty:
        for horizon, table in calibration.groupby("horizon_seconds"):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
            ax.scatter(table["mean_predicted_survival"], table["ipcw_observed_survival"])
            ax.set_xlabel("Mean predicted survival")
            ax.set_ylabel("IPCW observed survival")
            ax.set_title(f"Calibration at {int(horizon)} s")
            fig.tight_layout()
            fig.savefig(figures_dir / f"calibration_{int(horizon)}s.png", dpi=150)
            plt.close(fig)


def save_feature_importance(booster, feature_names: list[str], tables_dir: Path, figures_dir: Path) -> None:
    """Save XGBoost feature importance tables and a gain plot."""

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = []
    scores = {
        importance_type: booster.get_score(importance_type=importance_type)
        for importance_type in ["gain", "weight", "cover"]
    }
    for name in feature_names:
        rows.append(
            {
                "feature": name,
                "gain": float(scores["gain"].get(name, 0.0)),
                "weight": float(scores["weight"].get(name, 0.0)),
                "cover": float(scores["cover"].get(name, 0.0)),
            }
        )
    table = pd.DataFrame(rows).sort_values("gain", ascending=False, kind="stable")
    table.to_csv(tables_dir / "feature_importance.csv", index=False)
    top = table.head(25).iloc[::-1]
    if not top.empty:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(top))))
        ax.barh(top["feature"], top["gain"])
        ax.set_xlabel("Gain")
        ax.set_title("XGBoost-AFT feature importance")
        fig.tight_layout()
        fig.savefig(figures_dir / "feature_importance_gain.png", dpi=150)
        plt.close(fig)


def main() -> None:
    """Train and evaluate the configured XGBoost-AFT run."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != TASK_NAME:
        raise ValueError(f"Config task_name must be {TASK_NAME}.")
    if config.get("model_name") != "xgboost_aft":
        raise ValueError("Config model_name must be xgboost_aft.")

    dataset_dir = project_path(config["dataset"]["path"])
    dataset_metadata = load_yaml_config(dataset_dir / "dataset_metadata.yaml")
    selection_id = str(dataset_metadata["selection_id"])
    feature_set = str(config["dataset"]["feature_set"])
    sample_weighting = str(config["dataset"].get("sample_weighting", "uniform"))
    model_id = "xgboost_aft"
    run_id = make_run_id(
        model_id=model_id,
        feature_set=feature_set,
        sample_weighting=sample_weighting,
        selection_id=selection_id,
    )
    output_dir = run_dir(selection_id=selection_id, run_id=run_id)
    checkpoint_dir = model_dir(model_id=model_id, run_id=run_id)
    overwrite = bool(config.get("output", {}).get("overwrite", False))
    if directory_has_files(output_dir) and not overwrite:
        raise FileExistsError(f"Run output already exists: {output_dir}")
    if directory_has_files(checkpoint_dir) and not overwrite:
        raise FileExistsError(f"Model output already exists: {checkpoint_dir}")
    print("=== Survival-Persistence XGBoost-AFT ===")
    print(f"Config: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Dataset: {dataset_dir.relative_to(PROJECT_ROOT)}")
    print(f"Run ID: {run_id}")
    print(f"Feature set: {feature_set}")
    print(f"Sample weighting: {sample_weighting}")
    print("Uses validation only for early stopping; test is evaluated once.\n")
    if args.dry_run:
        return

    xgb = require_xgboost()

    for directory in [
        output_dir / "metrics",
        output_dir / "predictions",
        output_dir / "figures",
        output_dir / "tables",
        checkpoint_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    arrays, split_metadata = load_split_artifacts(dataset_dir)
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
    dtest, test_features = build_dmatrix(
        arrays["test"],
        feature_set=feature_set,
        sample_weighting=sample_weighting,
    )
    validate_feature_compatibility(train_features, val_features)
    validate_feature_compatibility(train_features, test_features)

    params = dict(config["model"])
    params["seed"] = int(config["training"].get("seed", config.get("seed", 42)))
    params["nthread"] = int(config["training"].get("nthread", -1))
    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(config["training"]["num_boost_round"]),
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=int(config["training"]["early_stopping_rounds"]),
        evals_result=evals_result,
        verbose_eval=False,
    )
    best_iteration = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    best_score = float(getattr(booster, "best_score", np.nan))
    booster_path = checkpoint_dir / "booster.json"
    booster.save_model(booster_path)

    distribution = str(params["aft_loss_distribution"])
    scale = float(params["aft_loss_distribution_scale"])
    horizons = np.asarray(config["prediction"]["horizons_seconds"], dtype=float)
    dmatrices = {"train": dtrain, "validation": dval, "test": dtest}
    prediction_frames: dict[str, pd.DataFrame] = {}
    mean_reason = ""
    for split, dmatrix in dmatrices.items():
        location = booster.predict(
            dmatrix,
            output_margin=True,
            iteration_range=(0, best_iteration + 1),
        )
        frame, split_mean_reason = make_predictions_frame(
            split=split,
            metadata=split_metadata[split],
            arrays=arrays[split],
            location=location,
            distribution=distribution,
            scale=scale,
            horizons_seconds=horizons,
            model_family="xgboost",
            model_id=model_id,
            feature_set=feature_set,
            sample_weighting=sample_weighting,
        )
        mean_reason = mean_reason or split_mean_reason
        prediction_frames[split] = frame
        frame.to_parquet(output_dir / "predictions" / f"{split}_predictions.parquet", index=False)

    training_history = training_history_frame(evals_result)
    training_history.to_csv(output_dir / "tables" / "training_history.csv", index=False)

    train_times = arrays["train"]["y_time_seconds"]
    train_observed = arrays["train"]["y_event_observed"]
    censoring_curve = fit_censoring_survival(train_times, train_observed)
    metrics_rows: list[dict[str, Any]] = []
    brier_tables: list[pd.DataFrame] = []
    for split, frame in prediction_frames.items():
        metric_row, brier = metrics_for_predictions(
            method_name="xgboost_aft",
            split=split,
            predictions=frame,
            horizons_seconds=horizons,
            censoring_curve=censoring_curve,
            distribution=distribution,
            scale=scale,
        )
        metrics_rows.append(metric_row)
        brier_tables.append(brier)

    baseline_metric_rows = []
    baseline_brier_tables = []
    for split, frame in prediction_frames.items():
        for baseline in [
            kaplan_meier_marginal_baseline(
                train_times=train_times,
                train_event_observed=train_observed,
                n_samples=len(frame),
                horizons_seconds=horizons,
            ),
            constant_median_baseline(
                train_times=train_times,
                train_event_observed=train_observed,
                n_samples=len(frame),
                horizons_seconds=horizons,
            ),
        ]:
            if np.isnan(baseline.survival_probabilities).any():
                continue
            baseline_frame = frame.copy()
            baseline_frame["predicted_median_remaining_seconds"] = baseline.median_seconds
            for index, horizon in enumerate(horizons):
                baseline_frame[f"survival_probability_{int(horizon)}s"] = (
                    baseline.survival_probabilities[:, index]
                )
            metric_row, brier = metrics_for_predictions(
                method_name=baseline.name,
                split=split,
                predictions=baseline_frame,
                horizons_seconds=horizons,
                censoring_curve=censoring_curve,
            )
            baseline_metric_rows.append(metric_row)
            baseline_brier_tables.append(brier)

    metrics_summary = pd.DataFrame(metrics_rows)
    baseline_metrics = pd.DataFrame(baseline_metric_rows)
    brier_by_horizon = pd.concat(brier_tables, ignore_index=True)
    baseline_brier = (
        pd.concat(baseline_brier_tables, ignore_index=True)
        if baseline_brier_tables
        else pd.DataFrame()
    )
    metrics_summary.to_csv(output_dir / "metrics" / "metrics_summary.csv", index=False)
    baseline_metrics.to_csv(output_dir / "metrics" / "baseline_metrics_summary.csv", index=False)
    brier_by_horizon.to_csv(output_dir / "metrics" / "brier_by_horizon.csv", index=False)
    baseline_brier.to_csv(output_dir / "metrics" / "baseline_brier_by_horizon.csv", index=False)

    calibration = calibration_tables(
        predictions=prediction_frames["test"],
        horizons_seconds=list(config["evaluation"]["calibration_horizons_seconds"]),
        n_bins=int(config["evaluation"].get("calibration_bins", 10)),
        censoring_curve=censoring_curve,
    )
    calibration.to_csv(output_dir / "metrics" / "test_calibration.csv", index=False)

    event_slices = event_timepoint_table(prediction_frames["test"])
    event_slices.to_csv(output_dir / "tables" / "test_event_timepoint_predictions.csv", index=False)

    bootstrap_replicates = int(config.get("evaluation", {}).get("bootstrap_replicates", 0))
    bootstrap = event_level_bootstrap(
        prediction_frames["test"],
        event_column="global_event_id",
        metric_fn=lambda frame: harrell_c_index(
            frame["y_time_seconds"].to_numpy(dtype=float),
            frame["y_event_observed"].to_numpy(dtype=int),
            frame["predicted_median_remaining_seconds"].to_numpy(dtype=float),
        )["c_index"],
        n_replicates=bootstrap_replicates,
        seed=int(config.get("seed", 42)),
    )
    if not bootstrap.empty:
        bootstrap.to_csv(output_dir / "metrics" / "test_event_bootstrap_c_index.csv", index=False)

    feature_names_path = output_dir / "tables" / "feature_names.json"
    feature_names_path.write_text(json.dumps(train_features.names, indent=2), encoding="utf-8")
    save_feature_importance(
        booster,
        train_features.names,
        output_dir / "tables",
        output_dir / "figures",
    )
    save_plots(
        figures_dir=output_dir / "figures",
        training_history=training_history,
        test_predictions=prediction_frames["test"],
        test_brier=brier_by_horizon.loc[brier_by_horizon["split"].eq("test")],
        calibration=calibration,
        horizons_seconds=horizons,
    )

    config_resolved_path = output_dir / "config_resolved.yaml"
    save_yaml(config_resolved_path, config)
    model_config_path = checkpoint_dir / "model_config.yaml"
    save_yaml(model_config_path, config)
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "task_name": TASK_NAME,
        "model_family": "xgboost",
        "model_id": model_id,
        "run_id": run_id,
        "config_path": relative_project_path(config_path),
        "config_fingerprint": config_fingerprint(config),
        "dataset_path": relative_project_path(dataset_dir),
        "dataset_metadata": relative_project_path(dataset_dir / "dataset_metadata.yaml"),
        "selection_id": selection_id,
        "feature_set": feature_set,
        "feature_names_path": relative_project_path(feature_names_path),
        "sample_weighting": sample_weighting,
        "aft_prediction_semantics": (
            "booster.predict(output_margin=True) is treated as the AFT log-time "
            "location mu; standard predict() is not used as a raw regression target."
        ),
        "aft_distribution": distribution,
        "aft_scale": scale,
        "predicted_mean_reason_if_missing": mean_reason,
        "best_iteration": best_iteration,
        "best_validation_aft_nloglik": best_score,
        "booster_path": relative_project_path(booster_path),
        "output_files": {
            "metrics_summary": relative_project_path(output_dir / "metrics" / "metrics_summary.csv"),
            "brier_by_horizon": relative_project_path(output_dir / "metrics" / "brier_by_horizon.csv"),
            "test_predictions": relative_project_path(output_dir / "predictions" / "test_predictions.parquet"),
            "training_history": relative_project_path(output_dir / "tables" / "training_history.csv"),
        },
        "ipcw_note": "Censoring curve is fitted on the train split only.",
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
            "model_id": model_id,
            "feature_set": feature_set,
            "sample_weighting": sample_weighting,
            "selection_id": selection_id,
            "dataset_path": relative_project_path(dataset_dir),
            "best_iteration": best_iteration,
            "best_val_aft_nloglik": best_score,
            "val_c_index": val_row["c_index"],
            "test_c_index": test_row["c_index"],
            "test_ipcw_brier_mean": test_row["ipcw_brier_mean"],
            "created_at": created_at,
            "status": "completed",
        },
        id_column="run_id",
        columns=RUN_INDEX_COLUMNS,
    )

    print("=== Completed survival XGBoost-AFT ===")
    print(f"Best iteration: {best_iteration}")
    print(f"Validation aft-nloglik: {best_score:.6f}")
    print(f"Test C-index: {test_row['c_index']:.6f}")
    print(f"Test IPCW Brier mean: {test_row['ipcw_brier_mean']:.6f}")
    print(f"Predictions: {(output_dir / 'predictions').relative_to(PROJECT_ROOT)}")
    print(f"Metrics: {(output_dir / 'metrics').relative_to(PROJECT_ROOT)}")
    print(f"Model: {booster_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
