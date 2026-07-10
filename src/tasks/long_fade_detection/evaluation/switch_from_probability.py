"""Convert long-fade probabilities into operational switch decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.switching.comparison import complete_explicit_switch_versions
from src.switching.conversion import compute_switch_from_signal_values


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def derive_switch_from_probability(
    probabilities: np.ndarray,
    *,
    probability_threshold: float,
) -> np.ndarray:
    """Return 1 when predicted long-fade probability reaches the threshold."""

    values = np.asarray(probabilities, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Long-fade probabilities must be finite.")
    if probability_threshold < 0.0 or probability_threshold > 1.0:
        raise ValueError("probability_threshold must be in [0, 1].")
    return values >= float(probability_threshold)


def align_perfect_switch_to_long_fade_windows(
    perfect_switch_timeseries: pd.DataFrame,
    window_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Attach Perfect Switch values to long-fade decision timestamps."""

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
            "global_event_id",
            "dataset_id",
            "dataset_name",
            "split",
            "timestamp",
            "event_duration_seconds",
            "y_long_fade",
        ],
        "window_metadata",
    )
    metadata = window_metadata.copy()
    metadata["Time"] = pd.to_datetime(metadata["timestamp"], errors="raise")
    if "quality_flag" not in metadata:
        metadata["quality_flag"] = np.where(
            metadata["y_long_fade"].astype(int).eq(1),
            "long_fade",
            "short_fade",
        )

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
            "Some long-fade windows could not be aligned to Perfect Switch. "
            f"Examples: {missing.to_dict(orient='records')}"
        )
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


def build_probability_switch_timeseries(
    predictions: pd.DataFrame,
    window_metadata: pd.DataFrame,
    perfect_switch_timeseries: pd.DataFrame,
    *,
    method_name: str,
    probability_threshold: float,
    signal_threshold: float,
    signal_condition: str,
    switch_time: int,
    apply_min_island_length: bool,
    require_current_signal_above_threshold: bool = True,
) -> pd.DataFrame:
    """Build a post-processed switch series from long-fade probabilities."""

    _require_columns(
        predictions,
        [
            "event_id",
            "global_event_id",
            "window_id",
            "prob_long_fade",
            "y_true",
            "model_id",
            "run_id",
        ],
        "predictions",
    )
    aligned_reference = align_perfect_switch_to_long_fade_windows(
        perfect_switch_timeseries,
        window_metadata,
    )
    prediction_columns = [
        "event_id",
        "global_event_id",
        "window_id",
        "prob_long_fade",
        "y_true",
        "model_id",
        "run_id",
    ]
    selected = aligned_reference.merge(
        predictions[prediction_columns],
        on=["event_id", "global_event_id", "window_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(selected) != len(window_metadata) or len(selected) != len(predictions):
        raise ValueError(
            "Predictions, metadata, and Perfect Switch windows are not one-to-one."
        )
    if not np.array_equal(
        selected["y_long_fade"].to_numpy(dtype=int),
        selected["y_true"].to_numpy(dtype=int),
    ):
        raise ValueError("Saved y_true labels do not match dataset metadata.")

    selected["probability_threshold_switch"] = derive_switch_from_probability(
        selected["prob_long_fade"].to_numpy(dtype=float),
        probability_threshold=probability_threshold,
    ).astype(np.int8)
    signal_gate, _ = compute_switch_from_signal_values(
        selected["Signal_true"].to_numpy(dtype=float),
        threshold=signal_threshold,
        condition=signal_condition,
        switch_time=1,
        apply_min_island_length=False,
    )
    selected["current_signal_gate"] = signal_gate
    selected["model_switch_raw"] = selected["probability_threshold_switch"]
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
    selected["decision_rule"] = "predicted_long_fade_probability_threshold"
    selected["probability_threshold"] = float(probability_threshold)
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
            "event_duration_seconds",
            "y_long_fade",
            "prob_long_fade",
            "decision_rule",
            "probability_threshold",
            "probability_threshold_switch",
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
