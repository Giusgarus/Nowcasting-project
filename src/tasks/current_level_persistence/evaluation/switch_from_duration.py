"""Convert current-level persistence predictions into operational switches."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.switching.comparison import complete_explicit_switch_versions
from src.switching.conversion import compute_switch_from_signal_values


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def derive_switch_from_duration(
    predicted_remaining_persistence_seconds: np.ndarray,
    *,
    switch_time_seconds: float,
) -> np.ndarray:
    """Return the configured duration-to-switch decision rule."""

    if switch_time_seconds < 0:
        raise ValueError("switch_time_seconds must be non-negative.")
    return (
        np.asarray(predicted_remaining_persistence_seconds, dtype=float)
        >= float(switch_time_seconds)
    ).astype(np.int8)


def align_perfect_switch_to_duration_windows(
    perfect_switch_timeseries: pd.DataFrame,
    window_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Align Perfect Switch values to current-level decision timestamps."""

    _require_columns(
        perfect_switch_timeseries,
        [
            "event_id",
            "Time",
            "Signal_true",
            "outage_mask",
            "perfect_switch_raw",
            "perfect_switch_min_island_old",
            "perfect_switch_min_time",
            "perfect_switch_adjusted",
            "perfect_switch",
        ],
        "perfect_switch_timeseries",
    )
    _require_columns(
        window_metadata,
        [
            "window_id",
            "global_window_id",
            "event_id",
            "dataset_id",
            "dataset_name",
            "quality_flag",
            "split",
            "timestamp",
            "current_signal",
            "remaining_persistence_seconds",
        ],
        "window_metadata",
    )
    metadata = window_metadata.copy()
    metadata["Time"] = pd.to_datetime(metadata["timestamp"], errors="raise")
    reference_columns = [
        "event_id",
        "Time",
        "Signal_true",
        "outage_mask",
        "perfect_switch_raw",
        "perfect_switch_min_island_old",
        "perfect_switch_min_time",
        "perfect_switch_adjusted",
        "perfect_switch",
    ]
    aligned = metadata.merge(
        perfect_switch_timeseries[reference_columns],
        on=["event_id", "Time"],
        how="left",
        validate="many_to_one",
    )
    if aligned["perfect_switch"].isna().any():
        missing = aligned.loc[
            aligned["perfect_switch"].isna(),
            ["window_id", "event_id", "Time"],
        ].head(5)
        raise ValueError(
            "Some current-level windows could not be aligned to Perfect Switch. "
            f"Examples: {missing.to_dict(orient='records')}"
        )
    if not np.allclose(
        aligned["current_signal"].to_numpy(dtype=float),
        aligned["Signal_true"].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("Window current_signal values do not match the reference signal.")
    for column in (
        "outage_mask",
        "perfect_switch_raw",
        "perfect_switch_min_island_old",
        "perfect_switch_min_time",
        "perfect_switch_adjusted",
        "perfect_switch",
    ):
        aligned[column] = aligned[column].astype(np.int8)
    return aligned.sort_values(["event_id", "Time"], kind="stable").reset_index(
        drop=True
    )


def build_duration_switch_timeseries(
    predictions: pd.DataFrame,
    window_metadata: pd.DataFrame,
    perfect_switch_timeseries: pd.DataFrame,
    *,
    method_name: str,
    duration_threshold_seconds: float,
    signal_threshold: float,
    signal_condition: str,
    switch_time: int,
    apply_min_island_length: bool,
    require_current_signal_above_threshold: bool = True,
) -> pd.DataFrame:
    """Build a post-processed decision-time switch series from durations."""

    _require_columns(
        predictions,
        [
            "global_window_id",
            "event_id",
            "window_id",
            "y_true_seconds",
            "y_pred_seconds",
            "model_id",
            "run_id",
        ],
        "predictions",
    )
    aligned_reference = align_perfect_switch_to_duration_windows(
        perfect_switch_timeseries,
        window_metadata,
    )
    prediction_columns = [
        "global_window_id",
        "event_id",
        "window_id",
        "y_true_seconds",
        "y_pred_seconds",
        "model_id",
        "run_id",
    ]
    selected = aligned_reference.merge(
        predictions[prediction_columns],
        on=["global_window_id", "event_id", "window_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(selected) != len(window_metadata) or len(selected) != len(predictions):
        raise ValueError(
            "Predictions, metadata, and Perfect Switch windows are not one-to-one."
        )
    if not np.allclose(
        selected["remaining_persistence_seconds"].to_numpy(dtype=float),
        selected["y_true_seconds"].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-3,
    ):
        raise ValueError("Saved y_true_seconds values do not match dataset metadata.")
    if not np.isfinite(selected["y_pred_seconds"].to_numpy(dtype=float)).all():
        raise ValueError("Predicted persistence durations must be finite.")

    selected["duration_threshold_switch"] = derive_switch_from_duration(
        selected["y_pred_seconds"].to_numpy(dtype=float),
        switch_time_seconds=duration_threshold_seconds,
    )
    signal_gate, _ = compute_switch_from_signal_values(
        selected["current_signal"].to_numpy(dtype=float),
        threshold=signal_threshold,
        condition=signal_condition,
        switch_time=1,
        apply_min_island_length=False,
    )
    selected["current_signal_gate"] = signal_gate
    selected["model_switch_raw"] = selected["duration_threshold_switch"]
    if require_current_signal_above_threshold:
        selected["model_switch_raw"] = (
            selected["model_switch_raw"].astype(bool)
            & selected["current_signal_gate"].astype(bool)
        ).astype(np.int8)
    selected = complete_explicit_switch_versions(
        selected.sort_values(["event_id", "Time"], kind="stable").reset_index(
            drop=True
        ),
        threshold=signal_threshold,
        condition=signal_condition,
        switch_time=switch_time,
        apply_min_island_length=apply_min_island_length,
    )
    selected["method"] = method_name
    selected["decision_rule"] = "predicted_duration_threshold"
    selected["predicted_remaining_persistence_seconds"] = selected[
        "y_pred_seconds"
    ].astype(float)
    selected["true_remaining_persistence_seconds"] = selected[
        "y_true_seconds"
    ].astype(float)
    selected["duration_threshold_seconds"] = float(duration_threshold_seconds)
    selected["require_current_signal_above_threshold"] = bool(
        require_current_signal_above_threshold
    )
    selected["threshold"] = float(signal_threshold)
    selected["switch_time"] = int(switch_time)
    selected["split"] = "test"
    return selected[
        [
            "method",
            "run_id",
            "model_id",
            "event_id",
            "global_event_id",
            "dataset_id",
            "dataset_name",
            "quality_flag",
            "split",
            "Time",
            "Signal_true",
            "current_signal",
            "true_remaining_persistence_seconds",
            "predicted_remaining_persistence_seconds",
            "decision_rule",
            "duration_threshold_seconds",
            "duration_threshold_switch",
            "current_signal_gate",
            "require_current_signal_above_threshold",
            "outage_mask",
            "perfect_switch_raw",
            "perfect_switch_min_island_old",
            "perfect_switch_min_time",
            "perfect_switch_adjusted",
            "perfect_switch",
            "model_switch_raw",
            "model_switch_min_island_old",
            "model_switch_min_time",
            "model_switch_adjusted",
            "model_switch",
            "threshold",
            "switch_time",
            "window_id",
            "global_window_id",
        ]
    ]
