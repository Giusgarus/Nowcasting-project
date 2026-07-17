"""Past/current feature construction for survival-persistence datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

SCALAR_CONTEXT_FEATURE_NAMES = (
    "current_signal",
    "signal_minus_threshold",
    "window_mean",
    "window_std",
    "window_min",
    "window_max",
    "window_range",
    "last_slope",
    "recent_slope_3",
    "recent_slope_5",
    "recent_mean_5",
    "recent_std_5",
    "elapsed_since_event_start_seconds",
    "elapsed_event_samples",
    "current_above_threshold",
    "time_since_last_above_threshold_seconds",
    "consecutive_below_threshold_samples",
    "fraction_above_threshold_recent_window",
)

FORBIDDEN_MODEL_INPUT_COLUMNS = {
    "event_end_time",
    "event_end_confirmation_time",
    "event_duration_seconds",
    "remaining_event_seconds",
    "event_observed",
    "censoring_flag",
    "future_threshold_crossings",
    "final_event_position_fraction",
    "y_time_seconds",
    "y_event_observed",
    "y_lower_bound_seconds",
    "y_upper_bound_seconds",
}


def relative_to_current(window: np.ndarray) -> np.ndarray:
    """Return ``window - S_t`` with the current value exactly zero."""

    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("window must be a non-empty one-dimensional array.")
    output = values - values[-1]
    output[-1] = np.float32(0.0)
    return output.astype(np.float32)


def relative_to_threshold(window: np.ndarray, *, threshold_on: float) -> np.ndarray:
    """Return ``window - threshold_on``."""

    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("window must be a non-empty one-dimensional array.")
    return (values - np.float32(threshold_on)).astype(np.float32)


def compute_scalar_context_features(
    x_raw: np.ndarray,
    *,
    threshold_on: float,
    elapsed_since_event_start_seconds: np.ndarray,
    elapsed_event_samples: np.ndarray,
    current_above_threshold: np.ndarray,
    time_since_last_above_threshold_seconds: np.ndarray,
    consecutive_below_threshold_samples: np.ndarray,
    fraction_above_threshold_recent_window: np.ndarray,
) -> np.ndarray:
    """Compute scalar features using only past/current information."""

    values = np.asarray(x_raw, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("X_raw must have shape (N, context_length).")
    if not np.isfinite(values).all():
        raise ValueError("X_raw contains non-finite values.")

    n_rows = len(values)

    def _aligned(array: np.ndarray, name: str) -> np.ndarray:
        output = np.asarray(array, dtype=np.float32).reshape(-1)
        if len(output) != n_rows:
            raise ValueError(f"{name} must contain one value per row.")
        return output

    elapsed_seconds = _aligned(
        elapsed_since_event_start_seconds,
        "elapsed_since_event_start_seconds",
    )
    elapsed_samples = _aligned(elapsed_event_samples, "elapsed_event_samples")
    above = _aligned(current_above_threshold, "current_above_threshold")
    since_above = _aligned(
        time_since_last_above_threshold_seconds,
        "time_since_last_above_threshold_seconds",
    )
    consecutive_below = _aligned(
        consecutive_below_threshold_samples,
        "consecutive_below_threshold_samples",
    )
    recent_fraction = _aligned(
        fraction_above_threshold_recent_window,
        "fraction_above_threshold_recent_window",
    )

    current = values[:, -1]
    window_min = values.min(axis=1)
    window_max = values.max(axis=1)

    def slope(span: int) -> np.ndarray:
        if values.shape[1] < 2:
            return np.zeros(n_rows, dtype=np.float32)
        reference_index = -min(span, values.shape[1])
        return current - values[:, reference_index]

    recent = values[:, -min(5, values.shape[1]) :]
    features = np.column_stack(
        [
            current,
            current - np.float32(threshold_on),
            values.mean(axis=1),
            values.std(axis=1, ddof=0),
            window_min,
            window_max,
            window_max - window_min,
            slope(2),
            slope(3),
            slope(5),
            recent.mean(axis=1),
            recent.std(axis=1, ddof=0),
            elapsed_seconds,
            elapsed_samples,
            above,
            since_above,
            consecutive_below,
            recent_fraction,
        ]
    )
    return features.astype(np.float32)


def standardize_scalar_context_splits(
    split_features: Mapping[str, np.ndarray],
    *,
    standardize: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Standardize scalar context using train-only statistics."""

    if "train" not in split_features or len(split_features["train"]) == 0:
        raise ValueError("A non-empty train split is required.")
    train = np.asarray(split_features["train"], dtype=np.float32)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    observed_std = train.std(axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    scale = np.where(observed_std > 0.0, observed_std, 1.0).astype(np.float32)
    if standardize:
        output = {
            split: ((np.asarray(values, dtype=np.float32) - mean) / scale).astype(
                np.float32
            )
            for split, values in split_features.items()
        }
    else:
        output = {
            split: np.asarray(values, dtype=np.float32)
            for split, values in split_features.items()
        }
    scaler = {
        "feature_names": list(SCALAR_CONTEXT_FEATURE_NAMES),
        "mean": mean.tolist(),
        "std": scale.tolist(),
        "observed_std": observed_std.tolist(),
        "fitted_on": "train",
        "standardized": bool(standardize),
    }
    return output, scaler
