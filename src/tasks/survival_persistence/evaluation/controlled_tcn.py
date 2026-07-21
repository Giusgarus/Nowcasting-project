"""Controlled-selection utilities for the survival discrete-time TCN."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tasks.survival_persistence.adapters.discrete_time import (
    DiscreteTimeBinSpec,
    continuous_to_discrete_survival_labels,
    median_survival_time_seconds,
    survival_at_horizons,
)
from src.tasks.survival_persistence.evaluation.metrics import (
    brier_scores_by_horizon,
    calibration_by_horizon,
    fit_censoring_survival,
    harrell_c_index,
    ipcw_brier_score,
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: Sequence[str | Path]) -> str:
    """Return a stable fingerprint over file paths and contents."""

    payload = []
    for path in sorted(Path(value) for value in paths):
        payload.append({"path": str(path), "sha256": sha256_file(path)})
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def fingerprint_mapping(mapping: Mapping[str, Any]) -> str:
    """Return a stable fingerprint for JSON-like metadata."""

    return hashlib.sha256(
        json.dumps(mapping, sort_keys=True, default=str, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def effective_receptive_field_samples(
    *,
    kernel_size: int,
    dilations: Sequence[int],
    convolutions_per_block: int = 2,
) -> int:
    """Return the causal TCN effective receptive field in samples."""

    if kernel_size < 1:
        raise ValueError("kernel_size must be positive.")
    if convolutions_per_block < 1:
        raise ValueError("convolutions_per_block must be positive.")
    dilation_values = [int(value) for value in dilations]
    if any(value < 1 for value in dilation_values):
        raise ValueError("dilations must be positive.")
    return int(1 + (int(kernel_size) - 1) * convolutions_per_block * sum(dilation_values))


def assert_validation_only_selection(
    row_or_table: pd.Series | pd.DataFrame,
    *,
    metric: str,
) -> None:
    """Fail when a selection metric or row contains test metric leakage."""

    if str(metric).startswith("test_"):
        raise ValueError("Controlled selection metric must not be a test metric.")
    columns = (
        list(row_or_table.index)
        if isinstance(row_or_table, pd.Series)
        else list(row_or_table.columns)
    )
    forbidden = [column for column in columns if str(column).startswith("test_")]
    if forbidden:
        raise ValueError(
            "Selection records must not contain test metrics before final_test: "
            f"{forbidden}"
        )


def select_best_validation_row(
    table: pd.DataFrame,
    *,
    primary_metric: str,
    mode: str = "min",
) -> pd.Series:
    """Select the best completed validation-only experiment row."""

    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'.")
    assert_validation_only_selection(
        table.drop(columns=[col for col in table.columns if col.startswith("test_")], errors="ignore"),
        metric=primary_metric,
    )
    if primary_metric not in table:
        raise ValueError(f"Missing primary metric: {primary_metric}")
    candidates = table.copy()
    if "status" in candidates:
        candidates = candidates.loc[candidates["status"].eq("completed")]
    values = pd.to_numeric(candidates[primary_metric], errors="coerce")
    candidates = candidates.loc[np.isfinite(values)].copy()
    candidates[primary_metric] = values.loc[candidates.index]
    if candidates.empty:
        raise ValueError("No completed finite validation rows are available.")
    tie_breakers = [primary_metric]
    ascending = [mode == "min"]
    for optional in [
        "validation_discrete_nll",
        "validation_calibration_error_300s",
        "undefined_median_fraction",
        "parameter_count",
        "runtime_seconds",
        "experiment_id",
    ]:
        if optional in candidates:
            tie_breakers.append(optional)
            ascending.append(True)
    return candidates.sort_values(tie_breakers, ascending=ascending, kind="stable").iloc[0]


def calibration_absolute_error(table: pd.DataFrame) -> float:
    """Return weighted mean absolute calibration error for one horizon table."""

    if table.empty:
        return float("nan")
    valid = table.dropna(subset=["mean_predicted_survival", "ipcw_observed_survival"])
    if valid.empty:
        return float("nan")
    weights = valid["num_known_status"].to_numpy(dtype=float)
    errors = np.abs(
        valid["mean_predicted_survival"].to_numpy(dtype=float)
        - valid["ipcw_observed_survival"].to_numpy(dtype=float)
    )
    if weights.sum() <= 0:
        return float(errors.mean())
    return float(np.average(errors, weights=weights))


def undefined_median_fraction(
    survival_by_bin: np.ndarray,
    *,
    bin_spec: DiscreteTimeBinSpec,
) -> float:
    """Return fraction of rows whose median is not reached in finite bins."""

    survival = np.asarray(survival_by_bin, dtype=np.float64)
    finite = np.isfinite(bin_spec.right_edges_seconds)
    if survival.ndim != 2 or survival.shape[1] != bin_spec.num_bins:
        raise ValueError("survival_by_bin must have shape (n_samples, n_bins).")
    reached = np.any((survival <= 0.5) & finite.reshape(1, -1), axis=1)
    return float(1.0 - reached.mean()) if len(reached) else float("nan")


def event_balanced_brier_at_horizon(
    predictions: pd.DataFrame,
    *,
    horizon_seconds: float,
    censoring_curve,
) -> float:
    """Return event-balanced IPCW Brier score at one horizon."""

    column = f"survival_probability_{int(horizon_seconds)}s"
    if column not in predictions:
        raise ValueError(f"Missing prediction column: {column}")
    rows = []
    for _, event_frame in predictions.groupby("global_event_id", sort=False):
        result = ipcw_brier_score(
            event_frame["y_time_seconds"].to_numpy(dtype=float),
            event_frame["y_event_observed"].to_numpy(dtype=int),
            event_frame[column].to_numpy(dtype=float),
            horizon_seconds=float(horizon_seconds),
            censoring_curve=censoring_curve,
        )
        rows.append(result["ipcw_brier"])
    return float(np.nanmean(rows)) if rows else float("nan")


def activation_brier_at_horizon(
    predictions: pd.DataFrame,
    *,
    horizon_seconds: float,
    censoring_curve,
) -> float:
    """Return IPCW Brier score using the first available sample per event."""

    ordered = predictions.sort_values(
        ["global_event_id", "elapsed_since_event_start_seconds", "sample_time"],
        kind="stable",
    )
    activation = ordered.groupby("global_event_id", sort=False).head(1)
    column = f"survival_probability_{int(horizon_seconds)}s"
    return float(
        ipcw_brier_score(
            activation["y_time_seconds"].to_numpy(dtype=float),
            activation["y_event_observed"].to_numpy(dtype=int),
            activation[column].to_numpy(dtype=float),
            horizon_seconds=float(horizon_seconds),
            censoring_curve=censoring_curve,
        )["ipcw_brier"]
    )


def summarize_survival_predictions(
    *,
    predictions: pd.DataFrame,
    survival_by_bin: np.ndarray,
    bin_spec: DiscreteTimeBinSpec,
    horizons_seconds: np.ndarray,
    censoring_curve,
    discrete_nll: float,
    prefix: str,
    calibration_horizons_seconds: Sequence[float],
    calibration_bins: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Compute controlled validation/test metrics for one predictions frame."""

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
    calibration_rows = []
    calibration_errors: dict[str, float] = {}
    for horizon in calibration_horizons_seconds:
        column = f"survival_probability_{int(horizon)}s"
        if column not in predictions:
            continue
        table = calibration_by_horizon(
            times,
            observed,
            predictions[column].to_numpy(dtype=float),
            horizon_seconds=float(horizon),
            n_bins=int(calibration_bins),
            censoring_curve=censoring_curve,
        )
        calibration_rows.append(table)
        calibration_errors[f"{prefix}_calibration_error_{int(horizon)}s"] = (
            calibration_absolute_error(table)
        )
    calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    row: dict[str, Any] = {
        f"{prefix}_discrete_nll": float(discrete_nll),
        f"{prefix}_integrated_brier_score": float(brier["ipcw_brier"].mean()),
        f"{prefix}_harrell_c_index": cindex["c_index"],
        f"{prefix}_ipcw_c_index": np.nan,
        f"{prefix}_num_comparable_pairs": cindex["num_comparable_pairs"],
        "undefined_median_fraction": undefined_median_fraction(
            survival_by_bin,
            bin_spec=bin_spec,
        ),
    }
    row.update(calibration_errors)
    for _, brier_row in brier.iterrows():
        horizon = int(brier_row["horizon_seconds"])
        row[f"{prefix}_brier_{horizon}s"] = float(brier_row["ipcw_brier"])
    if f"survival_probability_300s" in predictions:
        row[f"{prefix}_activation_brier_300s"] = activation_brier_at_horizon(
            predictions,
            horizon_seconds=300.0,
            censoring_curve=censoring_curve,
        )
        row[f"{prefix}_event_balanced_brier_300s"] = event_balanced_brier_at_horizon(
            predictions,
            horizon_seconds=300.0,
            censoring_curve=censoring_curve,
        )
    return row, brier, calibration


def bin_support_diagnostics(
    arrays: Mapping[str, np.ndarray],
    *,
    bin_spec: DiscreteTimeBinSpec,
) -> pd.DataFrame:
    """Summarize observed events, risk sets, and censoring support per bin."""

    labels = continuous_to_discrete_survival_labels(
        y_time_seconds=np.asarray(arrays["y_time_seconds"]),
        y_event_observed=np.asarray(arrays["y_event_observed"]),
        y_lower_bound_seconds=np.asarray(arrays["y_lower_bound_seconds"]),
        y_upper_bound_seconds=np.asarray(arrays["y_upper_bound_seconds"]),
        bin_spec=bin_spec,
    )
    rows = []
    for index in range(bin_spec.num_bins):
        censoring_endpoints = int(np.sum(labels.censoring_bin_index == index))
        observed_events = int(np.sum(labels.event_bin_index == index))
        at_risk = int(labels.at_risk_mask[:, index].sum())
        rows.append(
            {
                "bin_index": index,
                "left_edge_seconds": float(bin_spec.left_edges_seconds[index]),
                "right_edge_seconds": (
                    "inf"
                    if np.isinf(bin_spec.right_edges_seconds[index])
                    else float(bin_spec.right_edges_seconds[index])
                ),
                "observed_events_per_bin": observed_events,
                "at_risk_samples_per_bin": at_risk,
                "censoring_endpoints_per_bin": censoring_endpoints,
                "zero_event_bin": bool(observed_events == 0),
                "sparse_bin": bool(observed_events < 5 and at_risk > 0),
                "tail_support": bool(index == bin_spec.num_bins - 1),
            }
        )
    return pd.DataFrame(rows)


def hazard_diagnostics(
    hazards: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    """Return compact hazard stability diagnostics."""

    values = np.asarray(hazards, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("hazards must be two-dimensional.")
    return {
        "hazard_min": float(np.nanmin(values)),
        "hazard_max": float(np.nanmax(values)),
        "hazard_mean": float(np.nanmean(values)),
        "saturated_near_zero_fraction": float(np.mean(values < 1e-4)),
        "saturated_near_one_fraction": float(np.mean(values > 1.0 - 1e-4)),
    }


def average_survival_probabilities(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Average survival-probability columns from aligned prediction frames."""

    if not frames:
        raise ValueError("At least one predictions frame is required.")
    id_columns = ["global_window_id", "sample_time"]
    reference = frames[0].reset_index(drop=True)
    survival_columns = [
        column for column in reference.columns if column.startswith("survival_probability_")
    ]
    if not survival_columns:
        raise ValueError("Prediction frames contain no survival_probability_* columns.")
    stacked = []
    for frame in frames:
        candidate = frame.reset_index(drop=True)
        for column in id_columns:
            if column in reference and column in candidate:
                if not reference[column].equals(candidate[column]):
                    raise ValueError("Prediction frames are not row-aligned for ensembling.")
        stacked.append(candidate[survival_columns].to_numpy(dtype=np.float64))
    averaged = reference.copy()
    averaged[survival_columns] = np.mean(stacked, axis=0)
    return averaged


def averaged_survival_curve_is_monotone(frame: pd.DataFrame) -> bool:
    """Return true if all saved horizon survival curves are non-increasing."""

    columns = sorted(
        [column for column in frame.columns if column.startswith("survival_probability_")],
        key=lambda value: int(value.split("_")[-1][:-1]),
    )
    if not columns:
        raise ValueError("No survival probability columns found.")
    values = frame[columns].to_numpy(dtype=float)
    return bool(np.all(np.diff(values, axis=1) <= 1e-8))


def median_from_horizon_survival(
    frame: pd.DataFrame,
    *,
    max_horizon_seconds: float,
) -> np.ndarray:
    """Compute an ensemble median from averaged horizon survival probabilities."""

    columns = sorted(
        [column for column in frame.columns if column.startswith("survival_probability_")],
        key=lambda value: int(value.split("_")[-1][:-1]),
    )
    horizons = np.asarray([int(column.split("_")[-1][:-1]) for column in columns], dtype=float)
    values = frame[columns].to_numpy(dtype=float)
    median = np.full(len(frame), float(max_horizon_seconds), dtype=float)
    for index, row in enumerate(values):
        crossing = np.flatnonzero(row <= 0.5)
        if crossing.size:
            median[index] = float(horizons[crossing[0]])
    return median
