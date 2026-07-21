"""Convert survival-curve predictions into operational switch decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.switching.comparison import complete_explicit_switch_versions
from src.switching.conversion import compute_switch_from_signal_values


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def survival_probability_column(horizon_seconds: float | int) -> str:
    """Return the saved prediction column for a survival horizon."""

    horizon = float(horizon_seconds)
    if horizon <= 0:
        raise ValueError("horizon_seconds must be positive.")
    if horizon.is_integer():
        token = str(int(horizon))
    else:
        token = f"{horizon:g}".replace(".", "p")
    return f"survival_probability_{token}s"


def derive_switch_from_survival_probability(
    survival_probabilities: np.ndarray,
    *,
    probability_threshold: float,
) -> np.ndarray:
    """Return 1 when survival probability reaches the configured threshold."""

    values = np.asarray(survival_probabilities, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Survival probabilities must be finite.")
    if probability_threshold < 0.0 or probability_threshold > 1.0:
        raise ValueError("probability_threshold must be in [0, 1].")
    return values >= float(probability_threshold)


def align_perfect_switch_to_survival_windows(
    perfect_switch_timeseries: pd.DataFrame,
    window_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Attach Perfect Switch values to survival decision timestamps."""

    _require_columns(
        perfect_switch_timeseries,
        [
            "global_event_id",
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
            "global_window_id",
            "global_event_id",
            "event_id",
            "dataset_id",
            "dataset_name",
            "split",
            "sample_time",
            "current_signal",
            "y_time_seconds",
            "y_event_observed",
        ],
        "window_metadata",
    )
    metadata = window_metadata.copy()
    metadata["Time"] = pd.to_datetime(metadata["sample_time"], errors="raise")
    if "quality_flag" not in metadata:
        metadata["quality_flag"] = np.where(
            metadata["y_event_observed"].astype(int).eq(1),
            "observed",
            "censored",
        )

    reference_columns = [
        "global_event_id",
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
        on=["global_event_id", "Time"],
        how="left",
        validate="many_to_one",
    )
    if aligned["perfect_switch"].isna().any():
        missing = aligned.loc[
            aligned["perfect_switch"].isna(),
            ["global_window_id", "global_event_id", "Time"],
        ].head(5)
        raise ValueError(
            "Some survival windows could not be aligned to Perfect Switch. "
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


def build_survival_switch_timeseries(
    predictions: pd.DataFrame,
    window_metadata: pd.DataFrame,
    perfect_switch_timeseries: pd.DataFrame,
    *,
    method_name: str,
    run_id: str,
    survival_horizon_seconds: float,
    probability_threshold: float,
    signal_threshold: float,
    signal_condition: str,
    switch_time: int,
    apply_min_island_length: bool,
    require_current_signal_above_threshold: bool = True,
) -> pd.DataFrame:
    """Build a post-processed switch series from survival probabilities."""

    probability_column = survival_probability_column(survival_horizon_seconds)
    _require_columns(
        predictions,
        [
            "global_event_id",
            "global_window_id",
            "y_time_seconds",
            "y_event_observed",
            "model_id",
            probability_column,
        ],
        "predictions",
    )
    aligned_reference = align_perfect_switch_to_survival_windows(
        perfect_switch_timeseries,
        window_metadata,
    )
    prediction_columns = [
        column
        for column in (
            "global_event_id",
            "global_window_id",
            "y_time_seconds",
            "y_event_observed",
            "model_family",
            "model_id",
            "feature_set",
            "sample_weighting",
            "predicted_median_remaining_seconds",
            "predicted_mean_remaining_seconds",
            probability_column,
        )
        if column in predictions.columns
    ]
    selected = aligned_reference.merge(
        predictions[prediction_columns],
        on=["global_event_id", "global_window_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_pred"),
    )
    if len(selected) != len(window_metadata) or len(selected) != len(predictions):
        raise ValueError(
            "Predictions, metadata, and Perfect Switch windows are not one-to-one."
        )
    if not np.allclose(
        selected["y_time_seconds"].to_numpy(dtype=float),
        selected["y_time_seconds_pred"].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-3,
    ):
        raise ValueError("Saved y_time_seconds values do not match dataset metadata.")
    if not np.array_equal(
        selected["y_event_observed"].to_numpy(dtype=int),
        selected["y_event_observed_pred"].to_numpy(dtype=int),
    ):
        raise ValueError("Saved y_event_observed values do not match dataset metadata.")

    selected["survival_probability"] = selected[probability_column].astype(float)
    selected["survival_threshold_switch"] = derive_switch_from_survival_probability(
        selected["survival_probability"].to_numpy(dtype=float),
        probability_threshold=probability_threshold,
    ).astype(np.int8)
    signal_gate, _ = compute_switch_from_signal_values(
        selected["current_signal"].to_numpy(dtype=float),
        threshold=signal_threshold,
        condition=signal_condition,
        switch_time=1,
        apply_min_island_length=False,
    )
    selected["current_signal_gate"] = signal_gate
    selected["model_switch_raw"] = selected["survival_threshold_switch"]
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
    selected["run_id"] = run_id
    selected["decision_rule"] = "survival_probability_threshold"
    selected["survival_horizon_seconds"] = float(survival_horizon_seconds)
    selected["survival_probability_threshold"] = float(probability_threshold)
    selected["require_current_signal_above_threshold"] = bool(
        require_current_signal_above_threshold
    )
    selected["threshold"] = float(signal_threshold)
    selected["switch_time"] = int(switch_time)
    selected["split"] = "test"

    optional_columns = [
        column
        for column in (
            "model_family",
            "feature_set",
            "sample_weighting",
            "predicted_median_remaining_seconds",
            "predicted_mean_remaining_seconds",
        )
        if column in selected.columns
    ]
    return selected[
        [
            "method",
            "run_id",
            "model_id",
            *optional_columns,
            "event_id",
            "global_event_id",
            "dataset_id",
            "dataset_name",
            "quality_flag",
            "split",
            "Time",
            "Signal_true",
            "current_signal",
            "y_time_seconds",
            "y_event_observed",
            "decision_rule",
            "survival_horizon_seconds",
            "survival_probability_threshold",
            "survival_probability",
            probability_column,
            "survival_threshold_switch",
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
