"""Summarize completed survival-persistence model runs."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import save_yaml  # noqa: E402
from src.utils.results_paths import (  # noqa: E402
    COMPARISON_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_results_index_dir,
    relative_project_path,
    upsert_index_row,
)
from src.tasks.survival_persistence.evaluation.metrics import (  # noqa: E402
    fit_censoring_survival,
    ipcw_brier_score,
)

TASK_NAME = "survival_persistence"
SELECTION_ID = "externalHoldout_test_fc_uplink_fade"
COMPARISON_ID = f"survivalPersistence_model_comparison_{SELECTION_ID}"
RUNS_ROOT = PROJECT_ROOT / "results/runs" / TASK_NAME / SELECTION_ID
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/comparisons/model_selection"
    / TASK_NAME
    / SELECTION_ID
    / COMPARISON_ID
)

RUNS = [
    {
        "model": "xgboost_aft",
        "input_representation": "raw_flattened_uniform",
        "run_id": (
            "survivalPersistence_xgboost_aft_raw_flattened_uniform_"
            "externalHoldout_test_fc_uplink_fade"
        ),
        "nloglik_column": "aft_nloglik",
    },
    {
        "model": "discrete_time_tcn",
        "input_representation": "grid_extended_best_relative_current_plus_scalar",
        "run_id": (
            "survivalPersistence_discrete_time_tcn_relative_current_plus_scalar_"
            "uniform_externalHoldout_test_fc_uplink_fade_grid_extended_best"
        ),
        "nloglik_column": "discrete_time_nll",
    },
]

SUMMARY_COLUMNS = [
    "model",
    "input_representation",
    "run_id",
    "test_ibs",
    "test_c_index",
    "test_nloglik",
    "test_brier_60s",
    "test_brier_300s",
    "test_brier_900s",
    "test_calibration_error_300s",
    "test_activation_brier_300s",
    "test_undefined_median_fraction",
    "test_num_samples",
    "test_num_events",
    "metrics_path",
]


def _read_test_metrics(metrics_path: Path, nloglik_column: str) -> dict[str, Any]:
    metrics = pd.read_csv(metrics_path)
    test_rows = metrics.loc[metrics["split"].astype(str).eq("test")]
    if test_rows.empty:
        raise ValueError(f"No test row found in {metrics_path}")
    row = test_rows.iloc[0].to_dict()
    return {
        "test_ibs": row.get("ipcw_brier_mean"),
        "test_c_index": row.get("c_index"),
        "test_nloglik": row.get(nloglik_column),
        "test_num_samples": row.get("num_samples"),
        "test_num_events": row.get("num_events"),
    }


def _brier_at_horizon(brier_path: Path, horizon_seconds: int) -> float:
    if not brier_path.exists():
        return float("nan")
    brier = pd.read_csv(brier_path)
    rows = brier.loc[
        brier["split"].astype(str).eq("test")
        & np.isclose(brier["horizon_seconds"].astype(float), float(horizon_seconds))
    ]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0]["ipcw_brier"])


def _calibration_error_at_horizon(calibration_path: Path, horizon_seconds: int) -> float:
    if not calibration_path.exists():
        return float("nan")
    calibration = pd.read_csv(calibration_path)
    rows = calibration.loc[
        np.isclose(calibration["horizon_seconds"].astype(float), float(horizon_seconds))
    ].dropna(subset=["mean_predicted_survival", "ipcw_observed_survival"])
    if rows.empty:
        return float("nan")
    weights = rows["num_known_status"].to_numpy(dtype=float)
    errors = np.abs(
        rows["mean_predicted_survival"].to_numpy(dtype=float)
        - rows["ipcw_observed_survival"].to_numpy(dtype=float)
    )
    if weights.sum() <= 0:
        return float(errors.mean())
    return float(np.average(errors, weights=weights))


def _undefined_median_fraction(predictions_path: Path) -> float:
    if not predictions_path.exists():
        return float("nan")
    predictions = pd.read_parquet(predictions_path, columns=["predicted_median_remaining_seconds"])
    median = predictions["predicted_median_remaining_seconds"].to_numpy(dtype=float)
    if len(median) == 0:
        return float("nan")
    return float(np.mean(~np.isfinite(median)))


def _activation_brier_at_horizon(
    *,
    train_predictions_path: Path,
    test_predictions_path: Path,
    horizon_seconds: int,
) -> float:
    if not train_predictions_path.exists() or not test_predictions_path.exists():
        return float("nan")
    required_columns = [
        "global_event_id",
        "sample_time",
        "elapsed_since_event_start_seconds",
        "y_time_seconds",
        "y_event_observed",
        f"survival_probability_{horizon_seconds}s",
    ]
    train_predictions = pd.read_parquet(
        train_predictions_path,
        columns=["y_time_seconds", "y_event_observed"],
    )
    test_predictions = pd.read_parquet(test_predictions_path, columns=required_columns)
    if train_predictions.empty or test_predictions.empty:
        return float("nan")
    censoring_curve = fit_censoring_survival(
        train_predictions["y_time_seconds"].to_numpy(dtype=float),
        train_predictions["y_event_observed"].to_numpy(dtype=int),
    )
    activation = (
        test_predictions.sort_values(
            ["global_event_id", "elapsed_since_event_start_seconds", "sample_time"],
            kind="stable",
        )
        .groupby("global_event_id", sort=False)
        .head(1)
    )
    return float(
        ipcw_brier_score(
            activation["y_time_seconds"].to_numpy(dtype=float),
            activation["y_event_observed"].to_numpy(dtype=int),
            activation[f"survival_probability_{horizon_seconds}s"].to_numpy(dtype=float),
            horizon_seconds=float(horizon_seconds),
            censoring_curve=censoring_curve,
        )["ipcw_brier"]
    )


def _event_counts_from_predictions(predictions_path: Path) -> dict[str, int] | None:
    if not predictions_path.exists():
        return None
    predictions = pd.read_parquet(predictions_path, columns=["global_event_id"])
    return {
        "test_num_samples": int(len(predictions)),
        "test_num_events": int(predictions["global_event_id"].nunique()),
    }


def build_summary() -> pd.DataFrame:
    """Return one comparison row per selected survival baseline."""

    rows = []
    for spec in RUNS:
        run_dir = RUNS_ROOT / str(spec["run_id"])
        metrics_path = run_dir / "metrics" / "metrics_summary.csv"
        if not metrics_path.exists():
            continue
        row = {
            "model": spec["model"],
            "input_representation": spec["input_representation"],
            "run_id": spec["run_id"],
            **_read_test_metrics(metrics_path, str(spec["nloglik_column"])),
            "test_brier_60s": _brier_at_horizon(
                run_dir / "metrics" / "brier_by_horizon.csv",
                60,
            ),
            "test_brier_300s": _brier_at_horizon(
                run_dir / "metrics" / "brier_by_horizon.csv",
                300,
            ),
            "test_brier_900s": _brier_at_horizon(
                run_dir / "metrics" / "brier_by_horizon.csv",
                900,
            ),
            "test_calibration_error_300s": _calibration_error_at_horizon(
                run_dir / "metrics" / "test_calibration.csv",
                300,
            ),
            "test_activation_brier_300s": _activation_brier_at_horizon(
                train_predictions_path=run_dir / "predictions" / "train_predictions.parquet",
                test_predictions_path=run_dir / "predictions" / "test_predictions.parquet",
                horizon_seconds=300,
            ),
            "test_undefined_median_fraction": _undefined_median_fraction(
                run_dir / "predictions" / "test_predictions.parquet"
            ),
            "metrics_path": relative_project_path(metrics_path),
        }
        prediction_counts = _event_counts_from_predictions(
            run_dir / "predictions" / "test_predictions.parquet"
        )
        if prediction_counts is not None:
            row.update(prediction_counts)
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def main() -> None:
    """Write the survival model comparison table and metadata."""

    paths = ensure_results_subdirs(OUTPUT_DIR, ("tables",))
    summary = build_summary()
    table_path = paths["tables"] / "model_comparison_xgboost_tcn.csv"
    summary.to_csv(table_path, index=False)

    created_at = datetime.now(timezone.utc).isoformat()
    save_yaml(
        OUTPUT_DIR / "metadata.yaml",
        {
            "comparison_id": COMPARISON_ID,
            "comparison_type": "model_selection",
            "task_name": TASK_NAME,
            "selection_id": SELECTION_ID,
            "num_runs": int(len(summary)),
            "source": "refreshed_run_metrics",
            "output_files": {
                "model_comparison": relative_project_path(table_path),
            },
            "created_at": created_at,
        },
    )
    upsert_index_row(
        get_results_index_dir() / "comparisons.csv",
        {
            "comparison_id": COMPARISON_ID,
            "method_id": "xgboost_aft_vs_discrete_time_tcn",
            "reference_id": "",
            "comparison_type": "model_selection",
            "selection_id": SELECTION_ID,
            "results_path": relative_project_path(OUTPUT_DIR),
            "metrics_path": relative_project_path(paths["tables"]),
            "figures_path": "",
            "status": "complete",
            "created_at": created_at,
        },
        id_column="comparison_id",
        columns=COMPARISON_INDEX_COLUMNS,
    )
    print(f"Wrote {len(summary)} rows to {relative_project_path(table_path)}")


if __name__ == "__main__":
    main()
