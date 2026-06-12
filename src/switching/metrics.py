"""Descriptive summaries for binary switch references and decisions."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def find_positive_islands(values: Sequence[int | bool] | np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive start/end indices for contiguous islands of ones."""

    vector = np.asarray(values).astype(np.int8).ravel()
    if not np.all(np.isin(vector, [0, 1])):
        raise ValueError("Switch values must contain only 0 and 1.")
    changes = np.diff(np.concatenate(([0], vector, [0])))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _island_lengths(values: Sequence[int | bool] | np.ndarray) -> np.ndarray:
    return np.asarray(
        [end - start + 1 for start, end in find_positive_islands(values)],
        dtype=int,
    )


def compute_raw_vs_postprocessed_switch_summary(
    raw_switch: Sequence[int | bool] | np.ndarray,
    processed_switch: Sequence[int | bool] | np.ndarray,
) -> dict[str, float | int]:
    """Summarize the change introduced by minimum-island post-processing."""

    raw = np.asarray(raw_switch).astype(np.int8).ravel()
    processed = np.asarray(processed_switch).astype(np.int8).ravel()
    if raw.shape != processed.shape:
        raise ValueError("Raw and processed switches must have the same shape.")
    if not np.all(np.isin(raw, [0, 1])) or not np.all(np.isin(processed, [0, 1])):
        raise ValueError("Switch values must contain only 0 and 1.")
    num_points = len(raw)
    return {
        "num_points": num_points,
        "num_positive_raw": int(raw.sum()),
        "num_positive_after_min_island": int(processed.sum()),
        "num_points_changed_by_min_island": int(np.count_nonzero(raw != processed)),
        "positive_rate_raw": float(raw.mean()) if num_points else 0.0,
        "positive_rate_after_min_island": float(processed.mean()) if num_points else 0.0,
        "raw_vs_postprocessed_agreement": (
            float(np.mean(raw == processed)) if num_points else 1.0
        ),
        "num_islands_raw": len(find_positive_islands(raw)),
        "num_islands_after_min_island": len(find_positive_islands(processed)),
    }


def _positive_boundary_time(
    frame: pd.DataFrame,
    switch_column: str,
    *,
    first: bool,
) -> object:
    positive_times = frame.loc[frame[switch_column].eq(1), "Time"]
    if positive_times.empty:
        return pd.NaT
    return positive_times.iloc[0] if first else positive_times.iloc[-1]


def compute_event_level_switch_summary(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Create one descriptive Perfect Switch summary row per event."""

    rows = []
    for event_id, frame in timeseries.groupby("event_id", sort=False):
        frame = frame.sort_values("Time", kind="stable")
        summary = compute_raw_vs_postprocessed_switch_summary(
            frame["perfect_switch_raw"], frame["perfect_switch"]
        )
        rows.append(
            {
                "event_id": event_id,
                "dataset_id": frame["dataset_id"].iloc[0],
                "dataset_name": frame["dataset_name"].iloc[0],
                "split": frame["split"].iloc[0],
                **summary,
                "first_positive_time_raw": _positive_boundary_time(
                    frame, "perfect_switch_raw", first=True
                ),
                "first_positive_time_after_min_island": _positive_boundary_time(
                    frame, "perfect_switch", first=True
                ),
                "last_positive_time_raw": _positive_boundary_time(
                    frame, "perfect_switch_raw", first=False
                ),
                "last_positive_time_after_min_island": _positive_boundary_time(
                    frame, "perfect_switch", first=False
                ),
                "max_signal": float(frame["Signal_true"].max()),
                "quality_flag": frame["quality_flag"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def compute_dataset_level_switch_summary(
    timeseries: pd.DataFrame,
    event_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one descriptive Perfect Switch summary row per dataset."""

    events = (
        compute_event_level_switch_summary(timeseries)
        if event_summary is None
        else event_summary
    )
    rows = []
    for (dataset_id, dataset_name, split), frame in timeseries.groupby(
        ["dataset_id", "dataset_name", "split"], sort=False
    ):
        event_frame = events.loc[
            events["dataset_id"].eq(dataset_id) & events["split"].eq(split)
        ]
        summary = compute_raw_vs_postprocessed_switch_summary(
            frame["perfect_switch_raw"], frame["perfect_switch"]
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "split": split,
                "num_events": int(frame["event_id"].nunique()),
                **summary,
                "mean_event_positive_rate_raw": float(
                    event_frame["positive_rate_raw"].mean()
                ),
                "mean_event_positive_rate_after_min_island": float(
                    event_frame["positive_rate_after_min_island"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _observed_sample_durations_seconds(times: pd.Series) -> np.ndarray:
    timestamps = pd.to_datetime(times, errors="raise").to_numpy(dtype="datetime64[ns]")
    if len(timestamps) <= 1:
        return np.zeros(len(timestamps), dtype=float)
    differences = np.diff(timestamps).astype("timedelta64[ns]").astype(np.int64) / 1e9
    valid = differences[differences > 0]
    fallback = float(np.median(valid)) if valid.size else 0.0
    differences = np.where(differences > 0, differences, fallback)
    return np.concatenate((differences, [fallback])).astype(float)


def compute_duration_summary(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Summarize switch durations and islands for each event."""

    rows = []
    for event_id, frame in timeseries.groupby("event_id", sort=False):
        frame = frame.sort_values("Time", kind="stable")
        durations = _observed_sample_durations_seconds(frame["Time"])
        raw = frame["perfect_switch_raw"].to_numpy(dtype=np.int8)
        processed = frame["perfect_switch"].to_numpy(dtype=np.int8)
        raw_lengths = _island_lengths(raw)
        processed_lengths = _island_lengths(processed)
        rows.append(
            {
                "event_id": event_id,
                "dataset_id": frame["dataset_id"].iloc[0],
                "dataset_name": frame["dataset_name"].iloc[0],
                "split": frame["split"].iloc[0],
                "num_positive_samples_raw": int(raw.sum()),
                "num_positive_samples_after_min_island": int(processed.sum()),
                "duration_positive_raw_seconds": float(durations[raw == 1].sum()),
                "duration_positive_after_min_island_seconds": float(
                    durations[processed == 1].sum()
                ),
                "num_islands_raw": int(len(raw_lengths)),
                "num_islands_after_min_island": int(len(processed_lengths)),
                "max_island_length_raw": int(raw_lengths.max()) if raw_lengths.size else 0,
                "max_island_length_after_min_island": (
                    int(processed_lengths.max()) if processed_lengths.size else 0
                ),
                "mean_island_length_raw": (
                    float(raw_lengths.mean()) if raw_lengths.size else 0.0
                ),
                "mean_island_length_after_min_island": (
                    float(processed_lengths.mean()) if processed_lengths.size else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_global_switch_summary(
    timeseries: pd.DataFrame,
    *,
    selection_folder: str,
    num_windows: int,
    threshold: float,
    switch_time: int,
) -> pd.DataFrame:
    """Create the one-row Perfect Switch summary for the selected test data."""

    summary = compute_raw_vs_postprocessed_switch_summary(
        timeseries["perfect_switch_raw"], timeseries["perfect_switch"]
    )
    summary["num_islands_raw"] = sum(
        len(find_positive_islands(frame["perfect_switch_raw"]))
        for _, frame in timeseries.groupby("event_id", sort=False)
    )
    summary["num_islands_after_min_island"] = sum(
        len(find_positive_islands(frame["perfect_switch"]))
        for _, frame in timeseries.groupby("event_id", sort=False)
    )
    return pd.DataFrame(
        [
            {
                "selection_folder": selection_folder,
                "split": "test",
                "threshold": float(threshold),
                "switch_time": int(switch_time),
                "num_events": int(timeseries["event_id"].nunique()),
                "num_windows": int(num_windows),
                **summary,
            }
        ]
    )
