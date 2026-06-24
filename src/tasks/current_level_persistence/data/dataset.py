"""Dataset construction for the current-level persistence task."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test")
SPLIT_FILE_NAMES = {"train": "train", "validation": "val", "test": "test"}


def remaining_persistence_samples(
    signal: np.ndarray,
    current_index: int,
    *,
    delta: float,
) -> int:
    """Count samples from ``current_index`` until signal falls below ``S_t - delta``.

    The current sample is included. If the signal never falls below the recovery
    level within the available sequence, the count reaches the sequence end.
    """

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if not 0 <= current_index < len(values):
        raise IndexError("current_index is outside the signal range.")
    if delta < 0:
        raise ValueError("delta must be non-negative.")
    current_signal = values[current_index]
    recovery_level = current_signal - float(delta)
    count = 0
    for value in values[current_index:]:
        if value < recovery_level:
            break
        count += 1
    return count


def relative_to_current(window: np.ndarray) -> np.ndarray:
    """Return ``window - S_t`` with the last context value exactly zero."""

    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("window must be one-dimensional.")
    if values.size == 0:
        raise ValueError("window cannot be empty.")
    relative = values - values[-1]
    relative[-1] = np.float32(0.0)
    return relative.astype(np.float32)


def build_current_level_persistence_index(
    event_windows: pd.DataFrame,
    *,
    context_length: int,
    delta: float,
    sampling_time_seconds: int,
    signal_column: str,
) -> pd.DataFrame:
    """Build traceable supervised windows and persistence-duration labels."""

    if context_length < 1:
        raise ValueError("context_length must be positive.")
    if sampling_time_seconds <= 0:
        raise ValueError("sampling_time_seconds must be positive.")
    required = {
        "dataset_id",
        "dataset_name",
        "segment_id",
        "event_id",
        "Time",
        "event_point_idx",
        signal_column,
    }
    missing = sorted(required - set(event_windows.columns))
    if missing:
        raise ValueError(f"Event windows are missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    window_number = 0
    group_columns = ["dataset_id", "dataset_name", "event_id", "segment_id"]
    for group_key, group in event_windows.groupby(group_columns, sort=False):
        dataset_id, dataset_name, event_id, segment_id = group_key
        ordered = group.sort_values("event_point_idx", kind="stable").reset_index(
            drop=True
        )
        signals = ordered[signal_column].to_numpy(dtype=np.float32)
        for position in range(context_length - 1, len(ordered)):
            window_number += 1
            current_signal = float(signals[position])
            persistence_samples = remaining_persistence_samples(
                signals,
                position,
                delta=delta,
            )
            persistence_seconds = float(persistence_samples * sampling_time_seconds)
            input_rows = ordered.iloc[position - context_length + 1 : position + 1]
            timestamp = pd.Timestamp(ordered["Time"].iloc[position])
            window_id = f"clp_window_{window_number:09d}"
            rows.append(
                {
                    "timestamp": timestamp,
                    "dataset_name": dataset_name,
                    "dataset_id": dataset_id,
                    "event_id": event_id,
                    "window_id": window_id,
                    "global_event_id": f"{dataset_name}::{event_id}",
                    "global_window_id": f"{dataset_name}::{window_id}",
                    "segment_id": segment_id,
                    "split": "",
                    "input_start_idx": int(input_rows["event_point_idx"].iloc[0]),
                    "input_end_idx": int(input_rows["event_point_idx"].iloc[-1]),
                    "input_start_time": pd.Timestamp(input_rows["Time"].iloc[0]),
                    "input_end_time": timestamp,
                    "current_point_idx": int(ordered["event_point_idx"].iloc[position]),
                    "current_signal": current_signal,
                    "recovery_level": current_signal - float(delta),
                    "remaining_persistence_samples": int(persistence_samples),
                    "remaining_persistence_seconds": persistence_seconds,
                    "log1p_remaining_persistence_seconds": float(
                        np.log1p(persistence_seconds)
                    ),
                    "delta": float(delta),
                    "context_length": int(context_length),
                    "sampling_time_seconds": int(sampling_time_seconds),
                    "quality_flag": "usable",
                }
            )
    return pd.DataFrame(rows)


def build_current_level_persistence_arrays(
    event_windows: pd.DataFrame,
    window_index: pd.DataFrame,
    *,
    context_length: int,
    signal_column: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame]]:
    """Create split NPZ arrays and aligned metadata from a persistence index."""

    required_index = {
        "global_event_id",
        "window_id",
        "event_id",
        "dataset_id",
        "split",
        "input_start_idx",
        "input_end_idx",
        "current_signal",
        "recovery_level",
        "remaining_persistence_samples",
        "remaining_persistence_seconds",
        "log1p_remaining_persistence_seconds",
        "delta",
        "sampling_time_seconds",
    }
    missing = sorted(required_index - set(window_index.columns))
    if missing:
        raise ValueError(f"Persistence window index is missing columns: {missing}")
    if not set(window_index["split"]).issubset(SPLITS):
        raise ValueError("Persistence window index contains unsupported split labels.")
    if window_index["global_window_id"].duplicated().any():
        raise ValueError("global_window_id values must be unique.")
    if window_index.groupby("global_event_id")["split"].nunique().max() > 1:
        raise ValueError("A global_event_id appears in more than one split.")

    event_windows = event_windows.copy()
    event_windows["global_event_id"] = (
        event_windows["dataset_name"].astype(str)
        + "::"
        + event_windows["event_id"].astype(str)
    )
    lookup = {
        event_id: frame.set_index("event_point_idx", drop=False).sort_index()
        for event_id, frame in event_windows.groupby("global_event_id", sort=False)
    }
    x_rows: list[np.ndarray] = []
    for row in window_index.itertuples(index=False):
        event_frame = lookup.get(row.global_event_id)
        if event_frame is None:
            raise ValueError(f"Event window not found for {row.global_event_id}.")
        indices = np.arange(row.input_start_idx, row.input_end_idx + 1)
        selected = event_frame.loc[indices]
        if len(selected) != context_length:
            raise ValueError(f"{row.window_id} has an invalid context length.")
        if pd.Timestamp(selected["Time"].iloc[-1]) != pd.Timestamp(row.timestamp):
            raise ValueError(f"{row.window_id} timestamp is not aligned with context.")
        x_rows.append(selected[signal_column].to_numpy(dtype=np.float32))

    x_raw = (
        np.stack(x_rows).astype(np.float32)
        if x_rows
        else np.empty((0, context_length), dtype=np.float32)
    )
    x_relative = (
        np.stack([relative_to_current(row) for row in x_raw]).astype(np.float32)
        if len(x_raw)
        else np.empty((0, context_length), dtype=np.float32)
    )
    all_arrays = {
        "X_raw": x_raw,
        "X_relative_to_current": x_relative,
        "y_remaining_persistence_samples": window_index[
            "remaining_persistence_samples"
        ].to_numpy(dtype=np.int64),
        "y_remaining_persistence_seconds": window_index[
            "remaining_persistence_seconds"
        ].to_numpy(dtype=np.float32),
        "y_log1p_remaining_persistence_seconds": window_index[
            "log1p_remaining_persistence_seconds"
        ].to_numpy(dtype=np.float32),
        "current_signal": window_index["current_signal"].to_numpy(dtype=np.float32),
        "recovery_level": window_index["recovery_level"].to_numpy(dtype=np.float32),
        "delta": window_index["delta"].to_numpy(dtype=np.float32),
        "sampling_time_seconds": window_index["sampling_time_seconds"].to_numpy(
            dtype=np.int64
        ),
        "window_id": window_index["window_id"].to_numpy(dtype=str),
        "event_id": window_index["event_id"].to_numpy(dtype=str),
        "dataset_id": window_index["dataset_id"].to_numpy(dtype=str),
        "global_window_id": window_index["global_window_id"].to_numpy(dtype=str),
        "global_event_id": window_index["global_event_id"].to_numpy(dtype=str),
    }

    split_arrays = {}
    split_metadata = {}
    for split in SPLITS:
        mask = window_index["split"].eq(split).to_numpy()
        split_arrays[split] = {key: value[mask] for key, value in all_arrays.items()}
        split_metadata[split] = window_index.loc[mask].reset_index(drop=True)
    return split_arrays, split_metadata


def save_split_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Save one split's arrays in compressed NPZ format."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **dict(arrays))


def summarize_split_metadata(split_metadata: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize split sizes and target durations."""

    rows = []
    for split, frame in split_metadata.items():
        rows.append(
            {
                "split": split,
                "num_windows": len(frame),
                "num_events": frame["global_event_id"].nunique(),
                "num_datasets": frame["dataset_id"].nunique(),
                "mean_persistence_seconds": float(
                    frame["remaining_persistence_seconds"].mean()
                )
                if len(frame)
                else np.nan,
                "median_persistence_seconds": float(
                    frame["remaining_persistence_seconds"].median()
                )
                if len(frame)
                else np.nan,
                "max_persistence_seconds": float(
                    frame["remaining_persistence_seconds"].max()
                )
                if len(frame)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
