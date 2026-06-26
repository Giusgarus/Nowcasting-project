"""No-leakage index construction for univariate Signal forecasting windows."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WindowIndices:
    """Half-open input and target intervals for one forecasting sample."""

    input_start: int
    input_stop: int
    target_start: int
    target_stop: int


def build_window_indices(
    n_observations: int,
    context_length: int,
    prediction_length: int,
    *,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> list[WindowIndices]:
    """Build windows whose inputs and targets both lie inside `[start, stop)`.

    The target begins exactly after the input, so target values can never enter
    the corresponding input window. The eventual input array contains only the
    univariate ``Signal`` series.
    """

    stop = n_observations if stop is None else stop
    if n_observations < 0:
        raise ValueError("n_observations cannot be negative.")
    if context_length < 1 or prediction_length < 1:
        raise ValueError("Window lengths must be positive.")
    if stride < 1:
        raise ValueError("stride must be positive.")
    if not 0 <= start <= stop <= n_observations:
        raise ValueError("start and stop must define a valid observation interval.")

    first_target_start = start + context_length
    last_target_start = stop - prediction_length
    if first_target_start > last_target_start:
        return []

    return [
        WindowIndices(
            input_start=target_start - context_length,
            input_stop=target_start,
            target_start=target_start,
            target_stop=target_start + prediction_length,
        )
        for target_start in range(first_target_start, last_target_start + 1, stride)
    ]


def build_autoregressive_window_index(
    event_windows: pd.DataFrame,
    event_quality: pd.DataFrame,
    events_with_splits: pd.DataFrame,
    *,
    context_length: int,
    prediction_length: int,
    gap_threshold_seconds: float,
    allow_warning_events: bool,
    signal_threshold: float,
) -> pd.DataFrame:
    """Build traceable forecasting-window indices inside prepared event windows."""

    columns = [
        "window_id",
        "event_id",
        "dataset_id",
        "dataset_name",
        "segment_id",
        "split",
        "quality_flag",
        "input_start_idx",
        "input_end_idx",
        "target_start_idx",
        "target_end_idx",
        "input_start_time",
        "input_end_time",
        "target_start_time",
        "target_end_time",
        "context_length",
        "prediction_length",
        "signal_threshold",
        "num_imputed_points_in_window",
        "region_type",
    ]
    if context_length < 1 or prediction_length < 1:
        raise ValueError("Window lengths must be positive.")
    allowed_quality = {"usable", "warning"} if allow_warning_events else {"usable"}
    quality_lookup = event_quality.set_index("event_id")["quality_flag"].to_dict()
    split_lookup = events_with_splits.set_index("event_id")["split"].to_dict()
    rows = []
    window_number = 0

    for event_id, event_frame in event_windows.groupby("event_id", sort=False):
        quality_flag = quality_lookup.get(event_id, "unusable")
        if quality_flag not in allowed_quality:
            continue
        ordered = event_frame.sort_values("Time", kind="stable").reset_index(drop=True)
        gap_groups = ordered["Time"].diff().dt.total_seconds().gt(
            gap_threshold_seconds
        ).cumsum()
        for (_, segment_id), segment_frame in ordered.groupby(
            [gap_groups, "segment_id"],
            sort=False,
        ):
            segment_frame = segment_frame.reset_index(drop=True)
            indices = build_window_indices(
                len(segment_frame),
                context_length,
                prediction_length,
            )
            for indices_item in indices:
                window_number += 1
                input_rows = segment_frame.iloc[
                    indices_item.input_start : indices_item.input_stop
                ]
                target_rows = segment_frame.iloc[
                    indices_item.target_start : indices_item.target_stop
                ]
                combined = pd.concat([input_rows, target_rows], ignore_index=True)
                rows.append(
                    {
                        "window_id": f"window_{window_number:09d}",
                        "event_id": event_id,
                        "dataset_id": segment_frame["dataset_id"].iloc[0],
                        "dataset_name": segment_frame["dataset_name"].iloc[0],
                        "segment_id": segment_id,
                        "split": split_lookup[event_id],
                        "quality_flag": quality_flag,
                        "input_start_idx": int(input_rows["event_point_idx"].iloc[0]),
                        "input_end_idx": int(input_rows["event_point_idx"].iloc[-1]),
                        "target_start_idx": int(target_rows["event_point_idx"].iloc[0]),
                        "target_end_idx": int(target_rows["event_point_idx"].iloc[-1]),
                        "input_start_time": input_rows["Time"].iloc[0],
                        "input_end_time": input_rows["Time"].iloc[-1],
                        "target_start_time": target_rows["Time"].iloc[0],
                        "target_end_time": target_rows["Time"].iloc[-1],
                        "context_length": context_length,
                        "prediction_length": prediction_length,
                        "signal_threshold": signal_threshold,
                        "num_imputed_points_in_window": int(
                            combined["is_imputed"].sum()
                        ),
                        "region_type": "event_window",
                    }
                )
    return pd.DataFrame(rows, columns=columns)
