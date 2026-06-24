"""Metrics for current-level persistence duration predictions."""

from __future__ import annotations

import numpy as np


def seconds_from_log1p(values: np.ndarray, *, clip_negative: bool = True) -> np.ndarray:
    """Invert log1p seconds and optionally clip negative durations to zero.

    Models train in log-space but evaluation is reported in raw duration units.
    Numerical or model outputs may produce negative durations after ``expm1``;
    clipping keeps duration predictions physically meaningful.
    """

    seconds = np.expm1(np.asarray(values, dtype=float))
    if clip_negative:
        seconds = np.maximum(seconds, 0.0)
    return seconds


def compute_duration_metrics(
    y_true_log1p_seconds: np.ndarray,
    y_pred_log1p_seconds: np.ndarray,
    *,
    clip_negative_predictions: bool = True,
) -> dict[str, float]:
    """Compute duration metrics in seconds, minutes, and log1p seconds."""

    y_true_log = np.asarray(y_true_log1p_seconds, dtype=float)
    y_pred_log = np.asarray(y_pred_log1p_seconds, dtype=float)
    if y_true_log.shape != y_pred_log.shape:
        raise ValueError("True and predicted arrays must have the same shape.")
    if y_true_log.size == 0:
        raise ValueError("Metric inputs cannot be empty.")

    y_true_seconds = seconds_from_log1p(y_true_log, clip_negative=False)
    y_pred_seconds = seconds_from_log1p(
        y_pred_log,
        clip_negative=clip_negative_predictions,
    )
    error_seconds = y_pred_seconds - y_true_seconds
    abs_error_seconds = np.abs(error_seconds)
    error_minutes = error_seconds / 60.0
    abs_error_minutes = np.abs(error_minutes)
    log_error = y_pred_log - y_true_log
    return {
        "mae_seconds": float(np.mean(abs_error_seconds)),
        "rmse_seconds": float(np.sqrt(np.mean(np.square(error_seconds)))),
        "median_ae_seconds": float(np.median(abs_error_seconds)),
        "mae_minutes": float(np.mean(abs_error_minutes)),
        "rmse_minutes": float(np.sqrt(np.mean(np.square(error_minutes)))),
        "median_ae_minutes": float(np.median(abs_error_minutes)),
        "mae_log1p_seconds": float(np.mean(np.abs(log_error))),
        "rmse_log1p_seconds": float(np.sqrt(np.mean(np.square(log_error)))),
    }


def compute_event_duration_metrics(
    event_ids: np.ndarray,
    y_true_log1p_seconds: np.ndarray,
    y_pred_log1p_seconds: np.ndarray,
    *,
    clip_negative_predictions: bool = True,
) -> dict[str, float]:
    """Summarize per-event duration errors."""

    ids = np.asarray(event_ids)
    y_true_log = np.asarray(y_true_log1p_seconds, dtype=float)
    y_pred_log = np.asarray(y_pred_log1p_seconds, dtype=float)
    if not (len(ids) == len(y_true_log) == len(y_pred_log)):
        raise ValueError("event_ids, y_true, and y_pred must have matching lengths.")
    if len(ids) == 0:
        raise ValueError("Metric inputs cannot be empty.")

    y_true_seconds = seconds_from_log1p(y_true_log, clip_negative=False)
    y_pred_seconds = seconds_from_log1p(
        y_pred_log,
        clip_negative=clip_negative_predictions,
    )
    event_mae = []
    event_rmse = []
    for event_id in np.unique(ids):
        mask = ids == event_id
        error = y_pred_seconds[mask] - y_true_seconds[mask]
        event_mae.append(float(np.mean(np.abs(error))))
        event_rmse.append(float(np.sqrt(np.mean(np.square(error)))))
    return {
        "event_mae_seconds_mean": float(np.mean(event_mae)),
        "event_mae_seconds_median": float(np.median(event_mae)),
        "event_rmse_seconds_mean": float(np.mean(event_rmse)),
        "event_rmse_seconds_median": float(np.median(event_rmse)),
    }


def prediction_table_columns() -> list[str]:
    """Return the standard prediction table schema for this task."""

    return [
        "timestamp",
        "dataset_name",
        "event_id",
        "window_id",
        "global_event_id",
        "global_window_id",
        "split",
        "y_true_seconds",
        "y_true_log1p_seconds",
        "y_pred_seconds",
        "y_pred_log1p_seconds",
        "absolute_error_seconds",
        "absolute_error_minutes",
        "model_id",
        "run_id",
    ]
