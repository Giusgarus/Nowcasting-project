"""Shared conversion rules from continuous values to binary switch decisions."""

from collections.abc import Sequence

import numpy as np


def _binary_array(values: Sequence[int | bool] | np.ndarray) -> np.ndarray:
    """Return a validated binary NumPy array while preserving its shape."""

    array = np.asarray(values)
    if not np.all(np.isin(array, [0, 1])):
        raise ValueError("Switch values must contain only 0 and 1.")
    return array.astype(np.int8, copy=True)


def ensure_min_island_length(
    original_switch: Sequence[int | bool] | np.ndarray,
    switch_time: int,
) -> np.ndarray:
    """Extend every short island of ones to the right up to ``switch_time``."""

    if not isinstance(switch_time, (int, np.integer)) or switch_time < 1:
        raise ValueError("switch_time must be a positive integer.")

    original = _binary_array(original_switch)
    vector = original.ravel()
    adjusted = vector.copy()
    if vector.size == 0:
        return adjusted.reshape(original.shape)

    changes = np.diff(np.concatenate(([0], vector, [0])))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    for start, end in zip(starts, ends, strict=True):
        if end - start + 1 < switch_time:
            adjusted[start : min(start + switch_time, vector.size)] = 1
    return adjusted.reshape(original.shape)


def enforce_switch_time(
    original_switch: Sequence[int | bool] | np.ndarray,
    mask_signal_over_threshold: Sequence[int | bool] | np.ndarray,
    switch_time: int,
) -> np.ndarray:
    """Reproduce the legacy adjusted-switch post-processing rule."""

    if not isinstance(switch_time, (int, np.integer)) or switch_time < 1:
        raise ValueError("switch_time must be a positive integer.")

    original = _binary_array(original_switch)
    outage_mask = _binary_array(mask_signal_over_threshold)
    if original.shape != outage_mask.shape:
        raise ValueError("original_switch and mask_signal_over_threshold must match.")

    vector = original.ravel()
    mask = outage_mask.ravel()
    adjusted = np.zeros_like(vector, dtype=np.int8)
    if vector.size == 0:
        return adjusted.reshape(original.shape)

    changes = np.diff(np.concatenate(([0], vector, [0])))
    starts = np.flatnonzero(changes == 1)
    for start in starts:
        adjusted[start : min(start + switch_time, vector.size)] = 1

    mask_only = (mask == 1) & (vector == 0)
    adjusted[mask_only] = 1
    return adjusted.reshape(original.shape)


def detect_persistent_threshold_switch(
    signal_values: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    switch_time: int,
) -> np.ndarray:
    """Reproduce the legacy Perfect Switch persistent-threshold detection."""

    if not isinstance(switch_time, (int, np.integer)) or switch_time < 1:
        raise ValueError("switch_time must be a positive integer.")
    signal = np.asarray(signal_values, dtype=float).ravel()
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal values must be finite.")

    num_points_in_window = switch_time + 1
    switch = np.zeros(len(signal), dtype=np.int8)
    if len(signal) < num_points_in_window:
        return switch

    above_threshold = (signal > threshold).astype(np.int8)
    cumulative = np.cumsum(above_threshold)
    rolling_sum = np.empty(len(signal) - num_points_in_window + 1, dtype=int)
    rolling_sum[0] = cumulative[num_points_in_window - 1]
    rolling_sum[1:] = (
        cumulative[num_points_in_window:]
        - cumulative[: len(signal) - num_points_in_window]
    )
    for start in np.flatnonzero(rolling_sum == num_points_in_window):
        switch[start : start + num_points_in_window] = 1
    return switch


def compute_switch_from_signal_values(
    signal_values: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    condition: str = "greater_than",
    switch_time: int = 1,
    apply_min_island_length: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw-scale signal values to raw and postprocessed switch arrays."""

    signal = np.asarray(signal_values, dtype=float)
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal values must be finite.")

    conditions = {
        "greater_than": np.greater,
        "greater_than_or_equal": np.greater_equal,
        "less_than": np.less,
        "less_than_or_equal": np.less_equal,
    }
    if condition not in conditions:
        supported = ", ".join(conditions)
        raise ValueError(f"Unsupported switch condition '{condition}'. Supported: {supported}")

    raw_switch = conditions[condition](signal, threshold).astype(np.int8)
    if not apply_min_island_length:
        return raw_switch, raw_switch.copy()
    return raw_switch, ensure_min_island_length(raw_switch, switch_time)


def compute_model_switch_from_predictions(
    predicted_signal_values: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    condition: str = "greater_than",
    switch_time: int = 1,
    apply_min_island_length: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the shared signal-to-switch rule to any model's raw-scale forecasts."""

    return compute_switch_from_signal_values(
        predicted_signal_values,
        threshold=threshold,
        condition=condition,
        switch_time=switch_time,
        apply_min_island_length=apply_min_island_length,
    )
