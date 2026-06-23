"""Descriptive summaries and behavior metrics for binary switch decisions."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.switching.conversion import compute_switch_from_signal_values


def _binary_vector(
    values: Sequence[int | bool] | np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """Return a validated one-dimensional binary vector."""

    series = pd.Series(np.asarray(values).ravel())
    if series.empty:
        return np.asarray([], dtype=np.int8)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{label} contains missing or non-numeric values.")
    vector = numeric.to_numpy()
    if not np.all(np.isin(vector, [0, 1])):
        raise ValueError(f"{label} must contain only binary 0/1 values.")
    return vector.astype(np.int8)


def count_active_islands(mask: Sequence[int | bool] | np.ndarray) -> int:
    """Count contiguous active islands in a binary mask."""

    vector = _binary_vector(mask, label="mask")
    padded = np.concatenate(([0], vector))
    return int(np.sum((padded[1:] == 1) & (padded[:-1] == 0)))


def compute_switch_behavior_metrics(
    model_switch_min_time: Sequence[int | bool] | np.ndarray,
    model_switch_adjusted: Sequence[int | bool] | np.ndarray,
    outage_mask: Sequence[int | bool] | np.ndarray,
    perfect_switch_min_time: Sequence[int | bool] | np.ndarray,
    *,
    method_id: str,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Compute switch-behavior metrics for one evaluated sequence."""

    min_time = _binary_vector(model_switch_min_time, label="model_switch_min_time")
    adjusted = _binary_vector(model_switch_adjusted, label="model_switch_adjusted")
    outage = _binary_vector(outage_mask, label="outage_mask")
    perfect = _binary_vector(perfect_switch_min_time, label="perfect_switch_min_time")
    lengths = {len(min_time), len(adjusted), len(outage), len(perfect)}
    if len(lengths) != 1:
        raise ValueError(
            "model_switch_min_time, model_switch_adjusted, outage_mask, and "
            "perfect_switch_min_time must have the same length."
        )

    num_samples = len(min_time)
    model_min_time_duration = int(min_time.sum())
    model_adjusted_duration = int(adjusted.sum())
    perfect_duration = int(perfect.sum())
    outage_duration = int(outage.sum())
    if perfect_duration == 0:
        duration_similarity = np.nan
    else:
        duration_similarity = (
            100.0
            * (perfect_duration - abs(perfect_duration - model_min_time_duration))
            / perfect_duration
        )

    row: dict[str, object] = {}
    if metadata:
        row.update(metadata)
    row.update(
        {
            "method_id": method_id,
            "num_samples": int(num_samples),
            "duration_samples": model_min_time_duration,
            "outage_time_covered_samples": model_adjusted_duration,
            "events": count_active_islands(min_time),
            "outage_mask_agreement_pct": (
                float(100.0 * np.mean(min_time == outage)) if num_samples else np.nan
            ),
            "outage_mask_agreement_pct_min_time": (
                float(100.0 * np.mean(min_time == outage)) if num_samples else np.nan
            ),
            "outage_mask_agreement_pct_adjusted": (
                float(100.0 * np.mean(adjusted == outage))
                if num_samples
                else np.nan
            ),
            "duration_similarity_vs_perfect_pct": float(duration_similarity),
            "perfect_min_time_duration_samples": perfect_duration,
            "outage_mask_duration_samples": outage_duration,
            "model_min_time_duration_samples": model_min_time_duration,
            "model_adjusted_duration_samples": model_adjusted_duration,
        }
    )
    return row


def _first_or_join(values: pd.Series) -> object:
    unique = values.dropna().astype(str).unique()
    if len(unique) == 0:
        return pd.NA
    if len(unique) == 1:
        return unique[0]
    return ";".join(sorted(unique))


def _metadata_for_frame(
    frame: pd.DataFrame,
    *,
    selection_id: str,
    test_type: str,
    threshold: float,
    switch_time: int,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "selection_id": selection_id,
        "test_type": test_type,
        "dataset_name": _first_or_join(frame["dataset_name"]),
        "threshold": float(threshold),
        "switch_time": int(switch_time),
    }
    if "dataset_id" in frame:
        metadata["dataset_id"] = _first_or_join(frame["dataset_id"])
    if "quality_flag" in frame:
        metadata["quality_flag"] = _first_or_join(frame["quality_flag"])
    return metadata


def _method_metadata(
    method_metadata: dict[str, object],
    *,
    method_id: str,
) -> dict[str, object]:
    return {
        "method_id": method_id,
        "method_name": method_metadata.get("method_name", method_id),
        "model_family": method_metadata.get("model_family", pd.NA),
        "architecture": method_metadata.get("architecture", pd.NA),
        "variant": method_metadata.get("variant", pd.NA),
        "mode": method_metadata.get("mode", pd.NA),
    }


def _perfect_method_metadata() -> dict[str, object]:
    return {
        "method_id": "perfect_switch",
        "method_name": "Perfect Switch",
        "model_family": "reference",
        "architecture": "perfect_switch",
        "variant": "reference",
        "mode": "oracle",
    }


def _complete_behavior_switch_columns(
    comparison_timeseries: pd.DataFrame,
    *,
    threshold: float,
    condition: str,
    perfect_timeseries: pd.DataFrame | None,
) -> pd.DataFrame:
    required_columns = {
        "event_id",
        "dataset_name",
        "Time",
        "Signal_true",
        "perfect_switch",
        "model_switch_raw",
        "model_switch",
    }
    missing = sorted(required_columns - set(comparison_timeseries.columns))
    if missing:
        raise ValueError(f"comparison_timeseries is missing required columns: {missing}")

    frame = comparison_timeseries.copy()
    frame["Time"] = pd.to_datetime(frame["Time"], errors="raise")
    if "outage_mask" not in frame:
        outage_mask, _ = compute_switch_from_signal_values(
            frame["Signal_true"].to_numpy(dtype=float),
            threshold=threshold,
            condition=condition,
            switch_time=1,
            apply_min_island_length=False,
        )
        frame["outage_mask"] = outage_mask

    if "perfect_switch_raw" not in frame.columns:
        if perfect_timeseries is not None and "perfect_switch_raw" in perfect_timeseries:
            reference_columns = [
                column
                for column in (
                    "perfect_switch_raw",
                    "perfect_switch_min_time",
                    "perfect_switch_adjusted",
                    "outage_mask",
                )
                if column in perfect_timeseries
            ]
            reference = perfect_timeseries[["event_id", "Time", *reference_columns]].copy()
            reference["Time"] = pd.to_datetime(reference["Time"], errors="raise")
            frame = frame.merge(
                reference,
                on=["event_id", "Time"],
                how="left",
                validate="many_to_one",
            )
            if frame["perfect_switch_raw"].isna().any():
                raise ValueError(
                    "Some comparison rows could not be aligned to perfect_switch_raw."
                )
        else:
            frame["perfect_switch_raw"] = frame["perfect_switch"]
    if "perfect_switch_min_time" not in frame:
        frame["perfect_switch_min_time"] = frame["perfect_switch"]
    if "perfect_switch_adjusted" not in frame:
        frame["perfect_switch_adjusted"] = frame["perfect_switch_min_time"]
    if "model_switch_min_time" not in frame:
        frame["model_switch_min_time"] = frame["model_switch"]
    if "model_switch_adjusted" not in frame:
        frame["model_switch_adjusted"] = frame["model_switch_min_time"]

    return frame.sort_values(["event_id", "Time"], kind="stable").reset_index(drop=True)


def build_switch_behavior_metrics_tables(
    comparison_timeseries: pd.DataFrame,
    *,
    method_id: str,
    method_metadata: dict[str, object],
    selection_id: str,
    test_type: str,
    threshold: float,
    condition: str,
    switch_time: int,
    perfect_timeseries: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build global and event-level switch-behavior tables for one comparison."""

    frame = _complete_behavior_switch_columns(
        comparison_timeseries,
        threshold=threshold,
        condition=condition,
        perfect_timeseries=perfect_timeseries,
    )
    model_metadata = _method_metadata(method_metadata, method_id=method_id)
    perfect_metadata = _perfect_method_metadata()

    global_rows = [
        compute_switch_behavior_metrics(
            frame["model_switch_min_time"],
            frame["model_switch_adjusted"],
            frame["outage_mask"],
            frame["perfect_switch_min_time"],
            method_id=method_id,
            metadata={
                **model_metadata,
                **_metadata_for_frame(
                    frame,
                    selection_id=selection_id,
                    test_type=test_type,
                    threshold=threshold,
                    switch_time=switch_time,
                ),
            },
        ),
        compute_switch_behavior_metrics(
            frame["perfect_switch_min_time"],
            frame["perfect_switch_adjusted"],
            frame["outage_mask"],
            frame["perfect_switch_min_time"],
            method_id="perfect_switch",
            metadata={
                **perfect_metadata,
                **_metadata_for_frame(
                    frame,
                    selection_id=selection_id,
                    test_type=test_type,
                    threshold=threshold,
                    switch_time=switch_time,
                ),
            },
        ),
    ]

    event_rows: list[dict[str, object]] = []
    for event_id, event_frame in frame.groupby("event_id", sort=False):
        event_metadata = _metadata_for_frame(
            event_frame,
            selection_id=selection_id,
            test_type=test_type,
            threshold=threshold,
            switch_time=switch_time,
        )
        event_metadata["event_id"] = event_id
        event_metadata["global_event_id"] = (
            event_frame["global_event_id"].iloc[0]
            if "global_event_id" in event_frame
            else event_id
        )
        event_rows.append(
            compute_switch_behavior_metrics(
                event_frame["model_switch_min_time"],
                event_frame["model_switch_adjusted"],
                event_frame["outage_mask"],
                event_frame["perfect_switch_min_time"],
                method_id=method_id,
                metadata={**model_metadata, **event_metadata},
            )
        )
        event_rows.append(
            compute_switch_behavior_metrics(
                event_frame["perfect_switch_min_time"],
                event_frame["perfect_switch_adjusted"],
                event_frame["outage_mask"],
                event_frame["perfect_switch_min_time"],
                method_id="perfect_switch",
                metadata={**perfect_metadata, **event_metadata},
            )
        )

    return pd.DataFrame(global_rows), pd.DataFrame(event_rows)


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
