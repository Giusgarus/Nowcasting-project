"""Construction and alignment of the model-agnostic Perfect Switch oracle."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.switching.conversion import (
    compute_switch_from_signal_values,
    detect_persistent_threshold_switch,
    enforce_switch_time,
    hold_while_signal_above_threshold,
    ensure_min_island_length,
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def compute_perfect_switch_from_true_signal(
    event_windows: pd.DataFrame,
    *,
    signal_column: str = "Signal_prepared",
    threshold: float = 10.0,
    condition: str = "greater_than_or_equal",
    switch_time: int = 10,
    apply_min_island_length: bool = True,
    group_columns: Sequence[str] = ("event_id",),
) -> pd.DataFrame:
    """Compute the persistent-threshold Perfect Switch per event."""

    required = ["Time", signal_column, *group_columns]
    _require_columns(event_windows, required, "event_windows")
    if event_windows.empty:
        raise ValueError("event_windows cannot be empty.")

    output = event_windows.copy()
    output["Time"] = pd.to_datetime(output["Time"], errors="raise")
    sort_columns = [*group_columns, "Time"] if group_columns else ["Time"]
    output = output.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    output["Signal_true"] = output[signal_column].astype(float)
    output["outage_mask"] = 0
    output["perfect_switch_raw"] = 0
    output["perfect_switch_min_island_old"] = 0
    output["perfect_switch_min_time"] = 0
    output["perfect_switch_adjusted"] = 0
    output["perfect_switch"] = 0

    grouped_indices = (
        output.groupby(list(group_columns), sort=False, dropna=False).indices.values()
        if group_columns
        else [np.arange(len(output))]
    )
    for indices in grouped_indices:
        outage_mask, _ = compute_switch_from_signal_values(
            output.loc[indices, "Signal_true"].to_numpy(),
            threshold=threshold,
            condition=condition,
            switch_time=1,
            apply_min_island_length=False,
        )
        raw = detect_persistent_threshold_switch(
            output.loc[indices, "Signal_true"].to_numpy(),
            threshold=threshold,
            switch_time=switch_time,
            condition=condition,
        )
        min_island_old = (
            ensure_min_island_length(raw, switch_time)
            if apply_min_island_length
            else raw.copy()
        )
        min_time = hold_while_signal_above_threshold(min_island_old, outage_mask)
        adjusted = enforce_switch_time(min_time, outage_mask, switch_time)
        output.loc[indices, "outage_mask"] = outage_mask
        output.loc[indices, "perfect_switch_raw"] = raw
        output.loc[indices, "perfect_switch_min_island_old"] = min_island_old
        output.loc[indices, "perfect_switch_min_time"] = min_time
        output.loc[indices, "perfect_switch_adjusted"] = adjusted
        output.loc[indices, "perfect_switch"] = min_time

    output["outage_mask"] = output["outage_mask"].astype("int8")
    output["perfect_switch_raw"] = output["perfect_switch_raw"].astype("int8")
    output["perfect_switch_min_island_old"] = output[
        "perfect_switch_min_island_old"
    ].astype("int8")
    output["perfect_switch_min_time"] = output["perfect_switch_min_time"].astype("int8")
    output["perfect_switch_adjusted"] = output["perfect_switch_adjusted"].astype("int8")
    output["perfect_switch"] = output["perfect_switch"].astype("int8")
    output["threshold"] = float(threshold)
    output["condition"] = str(condition)
    output["switch_time"] = int(switch_time)
    return output


def align_perfect_switch_to_forecast_windows(
    perfect_switch_timeseries: pd.DataFrame,
    window_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Align true signals and Perfect Switch values to forecast target steps."""

    _require_columns(
        perfect_switch_timeseries,
        [
            "event_id",
            "event_point_idx",
            "Time",
            "Signal_true",
            "perfect_switch_raw",
            "perfect_switch",
            "threshold",
            "switch_time",
        ],
        "perfect_switch_timeseries",
    )
    _require_columns(
        window_metadata,
        [
            "window_id",
            "event_id",
            "dataset_id",
            "dataset_name",
            "split",
            "quality_flag",
            "target_start_idx",
            "target_end_idx",
            "prediction_length",
            "target_start_time",
            "target_end_time",
        ],
        "window_metadata",
    )

    timeseries = perfect_switch_timeseries.copy()
    if "perfect_switch_min_time" not in timeseries:
        timeseries["perfect_switch_min_time"] = timeseries["perfect_switch"]
    if "perfect_switch_min_island_old" not in timeseries:
        timeseries["perfect_switch_min_island_old"] = timeseries[
            "perfect_switch_min_time"
        ]
    if "outage_mask" not in timeseries:
        condition = (
            str(timeseries["condition"].dropna().iloc[0])
            if "condition" in timeseries and timeseries["condition"].notna().any()
            else "greater_than_or_equal"
        )
        outage_mask, _ = compute_switch_from_signal_values(
            timeseries["Signal_true"].to_numpy(dtype=float),
            threshold=float(timeseries["threshold"].astype(float).iloc[0]),
            condition=condition,
            switch_time=1,
            apply_min_island_length=False,
        )
        timeseries["outage_mask"] = outage_mask.astype("int8")
    if "perfect_switch_adjusted" not in timeseries:
        timeseries["perfect_switch_adjusted"] = timeseries["perfect_switch_min_time"]

    event_lookup: dict[str, pd.DataFrame] = {}
    for event_id, frame in timeseries.groupby("event_id", sort=False):
        ordered = frame.sort_values("event_point_idx", kind="stable")
        if ordered["event_point_idx"].duplicated().any():
            raise ValueError(f"Duplicate event_point_idx values for event {event_id}.")
        event_lookup[str(event_id)] = ordered.set_index("event_point_idx")

    rows: list[dict[str, object]] = []
    for window in window_metadata.itertuples(index=False):
        event_id = str(window.event_id)
        if event_id not in event_lookup:
            raise ValueError(f"Missing Perfect Switch timeseries for event {event_id}.")
        target_indices = np.arange(window.target_start_idx, window.target_end_idx + 1)
        if len(target_indices) != int(window.prediction_length):
            raise ValueError(f"Target index length mismatch for window {window.window_id}.")
        try:
            target = event_lookup[event_id].loc[target_indices]
        except KeyError as error:
            raise ValueError(
                f"Target indices are unavailable for window {window.window_id}."
            ) from error
        if target.iloc[0]["Time"] != pd.Timestamp(window.target_start_time):
            raise ValueError(f"Target start time mismatch for window {window.window_id}.")
        if target.iloc[-1]["Time"] != pd.Timestamp(window.target_end_time):
            raise ValueError(f"Target end time mismatch for window {window.window_id}.")

        for target_step, point in enumerate(target.itertuples(), start=1):
            rows.append(
                {
                    "window_id": window.window_id,
                    "event_id": window.event_id,
                    "dataset_id": window.dataset_id,
                    "dataset_name": window.dataset_name,
                    "split": window.split,
                    "target_step": target_step,
                    "target_time": point.Time,
                    "y_true_raw": float(point.Signal_true),
                    "perfect_switch_raw": int(point.perfect_switch_raw),
                    "perfect_switch_min_island_old": int(
                        point.perfect_switch_min_island_old
                    ),
                    "perfect_switch_min_time": int(point.perfect_switch_min_time),
                    "perfect_switch_adjusted": int(point.perfect_switch_adjusted),
                    "perfect_switch": int(point.perfect_switch),
                    "outage_mask": int(point.outage_mask),
                    "threshold": float(point.threshold),
                    "switch_time": int(point.switch_time),
                    "quality_flag": window.quality_flag,
                }
            )
    return pd.DataFrame(rows)
