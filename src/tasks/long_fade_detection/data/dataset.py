"""Dataset construction for binary long-fade detection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loading import load_signal_dataset
from src.data.splits import assign_external_holdout_splits

SPLITS = ("train", "validation", "test")
SPLIT_FILE_NAMES = {"train": "train", "validation": "val", "test": "test"}
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
    "elapsed_since_fade_start_seconds",
    "position_in_fade_samples",
)
FORBIDDEN_MODEL_INPUT_COLUMNS = {
    "above_threshold",
    "position_in_fade_fraction",
    "event_duration_seconds",
    "event_duration_samples",
    "inside_event",
    "y_long_fade",
}


@dataclass(frozen=True)
class GroupedFadeEvent:
    """One grouped fade event span."""

    dataset_name: str
    dataset_id: str
    event_id: str
    start_index: int
    end_index: int
    event_start_time: pd.Timestamp
    event_end_time: pd.Timestamp
    event_duration_samples: int
    event_duration_seconds: float
    y_long_fade: int


def threshold_mask(signal: np.ndarray, *, threshold_db: float, operator: str) -> np.ndarray:
    """Return the configured threshold mask."""

    values = np.asarray(signal, dtype=float)
    if operator == ">=":
        return values >= float(threshold_db)
    if operator == ">":
        return values > float(threshold_db)
    raise ValueError("threshold_operator must be '>=' or '>'.")


def group_fade_events(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    dataset_id: str,
    threshold_db: float,
    min_fade_duration_seconds: int,
    sampling_time_seconds: int,
    threshold_operator: str = ">=",
    event_grouping_hours: float = 3.0,
) -> list[GroupedFadeEvent]:
    """Group threshold crossings into fade events using the 3-hour convention.

    A grouped event starts at the first threshold crossing. Further crossings
    whose timestamp is within ``event_grouping_hours`` from the event start
    remain in the same event, even when intervening samples fall below the
    threshold. The event span is from the first to the last threshold-crossing
    sample in the group, inclusive.
    """

    required = {"Time", "Signal"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Signal frame is missing columns: {missing}")
    ordered = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    mask = threshold_mask(
        ordered["Signal"].to_numpy(dtype=float),
        threshold_db=threshold_db,
        operator=threshold_operator,
    )
    crossing_indices = np.flatnonzero(mask)
    if len(crossing_indices) == 0:
        return []

    max_delta = pd.Timedelta(hours=float(event_grouping_hours))
    events: list[GroupedFadeEvent] = []
    active_start = int(crossing_indices[0])
    active_end = int(crossing_indices[0])
    active_start_time = pd.Timestamp(ordered["Time"].iloc[active_start])

    def finalize(start_index: int, end_index: int) -> None:
        start_time = pd.Timestamp(ordered["Time"].iloc[start_index])
        end_time = pd.Timestamp(ordered["Time"].iloc[end_index])
        duration_samples = int(end_index - start_index + 1)
        duration_seconds = float(duration_samples * int(sampling_time_seconds))
        event_number = len(events) + 1
        events.append(
            GroupedFadeEvent(
                dataset_name=dataset_name,
                dataset_id=dataset_id,
                event_id=f"lfd_event_{event_number:05d}",
                start_index=int(start_index),
                end_index=int(end_index),
                event_start_time=start_time,
                event_end_time=end_time,
                event_duration_samples=duration_samples,
                event_duration_seconds=duration_seconds,
                y_long_fade=int(duration_seconds >= float(min_fade_duration_seconds)),
            )
        )

    for crossing_index in crossing_indices[1:]:
        crossing_time = pd.Timestamp(ordered["Time"].iloc[int(crossing_index)])
        if crossing_time - active_start_time <= max_delta:
            active_end = int(crossing_index)
            continue
        finalize(active_start, active_end)
        active_start = int(crossing_index)
        active_end = int(crossing_index)
        active_start_time = crossing_time
    finalize(active_start, active_end)
    return events


def relative_to_current(window: np.ndarray) -> np.ndarray:
    """Return ``window - S_t`` with the last value exactly zero."""

    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("window must be a non-empty one-dimensional array.")
    output = values - values[-1]
    output[-1] = np.float32(0.0)
    return output.astype(np.float32)


def relative_to_threshold(window: np.ndarray, *, threshold_db: float) -> np.ndarray:
    """Return ``window - threshold_db``."""

    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("window must be a non-empty one-dimensional array.")
    return (values - np.float32(threshold_db)).astype(np.float32)


def lag_feature_names(context_length: int) -> list[str]:
    """Return lag feature names from oldest to newest."""

    if context_length < 1:
        raise ValueError("context_length must be positive.")
    return [f"lag_{lag}" for lag in range(context_length - 1, -1, -1)]


def compute_scalar_context_features(
    x_raw: np.ndarray,
    *,
    threshold_db: float,
    position_in_fade_samples: np.ndarray,
    sampling_time_seconds: int,
) -> np.ndarray:
    """Compute scalar context features using only information available at t."""

    values = np.asarray(x_raw, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("X_raw must have shape (N, context_length).")
    if not np.isfinite(values).all():
        raise ValueError("X_raw contains non-finite values.")
    positions = np.asarray(position_in_fade_samples, dtype=np.float32).reshape(-1)
    if len(positions) != len(values):
        raise ValueError("position_in_fade_samples must match X_raw rows.")

    current = values[:, -1]
    window_min = values.min(axis=1)
    window_max = values.max(axis=1)

    def slope(span: int) -> np.ndarray:
        if values.shape[1] < 2:
            return np.zeros(len(values), dtype=np.float32)
        reference_index = -min(span, values.shape[1])
        return current - values[:, reference_index]

    recent = values[:, -min(5, values.shape[1]) :]
    features = np.column_stack(
        [
            current,
            current - np.float32(threshold_db),
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
            positions * np.float32(sampling_time_seconds),
            positions,
        ]
    )
    return features.astype(np.float32)


def standardize_scalar_context_splits(
    split_features: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Standardize scalar context with train-only statistics."""

    if "train" not in split_features or len(split_features["train"]) == 0:
        raise ValueError("A non-empty train split is required.")
    train = np.asarray(split_features["train"], dtype=np.float32)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    observed_std = train.std(axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    scale = np.where(observed_std > 0.0, observed_std, 1.0).astype(np.float32)
    standardized = {
        split: ((np.asarray(values, dtype=np.float32) - mean) / scale).astype(np.float32)
        for split, values in split_features.items()
    }
    scaler = {
        "feature_names": list(SCALAR_CONTEXT_FEATURE_NAMES),
        "mean": mean.tolist(),
        "std": scale.tolist(),
        "observed_std": observed_std.tolist(),
        "fitted_on": "train",
        "standardized": True,
    }
    return standardized, scaler


def build_lag_scalar_features(
    x_relative_to_threshold: np.ndarray,
    scalar_context_features: np.ndarray,
    *,
    context_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate lag-threshold features with standardized scalar context."""

    lags = np.asarray(x_relative_to_threshold, dtype=np.float32)
    scalars = np.asarray(scalar_context_features, dtype=np.float32)
    if lags.ndim != 2 or lags.shape[1] != int(context_length):
        raise ValueError("X_relative_to_threshold has invalid shape.")
    if scalars.ndim != 2 or scalars.shape[0] != lags.shape[0]:
        raise ValueError("scalar_context_features has invalid shape.")
    names = np.asarray(
        lag_feature_names(context_length) + list(SCALAR_CONTEXT_FEATURE_NAMES),
        dtype=str,
    )
    if any(name in FORBIDDEN_MODEL_INPUT_COLUMNS for name in names):
        raise ValueError("Forbidden target/check field included in model inputs.")
    return np.concatenate([lags, scalars], axis=1).astype(np.float32), names


def _context_is_contiguous(times: Sequence[pd.Timestamp], *, sampling_time_seconds: int) -> bool:
    if len(times) < 2:
        return True
    values = pd.to_datetime(pd.Series(times), errors="raise")
    deltas = values.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    return bool(np.all(deltas <= sampling_time_seconds * 1.5))


def build_long_fade_window_index(
    signal_frames: Mapping[str, pd.DataFrame],
    *,
    threshold_db: float,
    min_fade_duration_seconds: int,
    sampling_time_seconds: int,
    context_length: int,
    threshold_operator: str = ">=",
    event_grouping_hours: float = 3.0,
) -> pd.DataFrame:
    """Build one row per valid timestamp inside grouped fade events."""

    rows: list[dict[str, Any]] = []
    window_number = 0
    for dataset_number, (dataset_name, frame) in enumerate(signal_frames.items(), start=1):
        dataset_id = f"dataset_{dataset_number:03d}"
        ordered = (
            frame[["Time", "Signal"]]
            .dropna()
            .sort_values("Time", kind="stable")
            .drop_duplicates("Time", keep="first")
            .reset_index(drop=True)
        )
        events = group_fade_events(
            ordered,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            threshold_db=threshold_db,
            min_fade_duration_seconds=min_fade_duration_seconds,
            sampling_time_seconds=sampling_time_seconds,
            threshold_operator=threshold_operator,
            event_grouping_hours=event_grouping_hours,
        )
        for event in events:
            for position in range(event.start_index, event.end_index + 1):
                if position < context_length - 1:
                    continue
                input_start = position - context_length + 1
                input_end = position
                input_times = ordered["Time"].iloc[input_start : input_end + 1].tolist()
                if not _context_is_contiguous(
                    input_times,
                    sampling_time_seconds=sampling_time_seconds,
                ):
                    continue
                window_number += 1
                position_in_fade = int(position - event.start_index)
                timestamp = pd.Timestamp(ordered["Time"].iloc[position])
                window_id = f"lfd_window_{window_number:09d}"
                rows.append(
                    {
                        "timestamp": timestamp,
                        "dataset_name": dataset_name,
                        "dataset_id": dataset_id,
                        "event_id": event.event_id,
                        "window_id": window_id,
                        "split": "",
                        "input_start_index": int(input_start),
                        "input_end_index": int(input_end),
                        "input_start_time": pd.Timestamp(ordered["Time"].iloc[input_start]),
                        "input_end_time": timestamp,
                        "event_start_time": event.event_start_time,
                        "event_end_time": event.event_end_time,
                        "position_in_fade_samples": position_in_fade,
                        "elapsed_since_fade_start_seconds": float(
                            position_in_fade * sampling_time_seconds
                        ),
                        "inside_event": True,
                        "event_duration_samples": event.event_duration_samples,
                        "event_duration_seconds": event.event_duration_seconds,
                        "y_long_fade": event.y_long_fade,
                        "threshold_db": float(threshold_db),
                        "min_fade_duration_seconds": int(min_fade_duration_seconds),
                        "sampling_time_seconds": int(sampling_time_seconds),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_long_fade_arrays(
    signal_frames: Mapping[str, pd.DataFrame],
    selected_index: pd.DataFrame,
    *,
    context_length: int,
    threshold_db: float,
    sampling_time_seconds: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame], dict[str, Any]]:
    """Create split arrays, aligned metadata, and scalar scaler metadata."""

    required = {
        "dataset_name",
        "dataset_id",
        "event_id",
        "window_id",
        "global_event_id",
        "global_window_id",
        "split",
        "input_start_index",
        "input_end_index",
        "position_in_fade_samples",
        "event_duration_samples",
        "event_duration_seconds",
        "y_long_fade",
    }
    missing = sorted(required - set(selected_index.columns))
    if missing:
        raise ValueError(f"Long-fade index is missing columns: {missing}")
    if selected_index.groupby("global_event_id")["split"].nunique().max() > 1:
        raise ValueError("A grouped event appears in more than one split.")
    if selected_index["dataset_name"].nunique() < 1:
        raise ValueError("No datasets are available.")

    lookup = {
        dataset_name: frame.sort_values("Time", kind="stable")
        .drop_duplicates("Time", keep="first")
        .reset_index(drop=True)
        for dataset_name, frame in signal_frames.items()
    }
    x_raw_rows: list[np.ndarray] = []
    for row in selected_index.itertuples(index=False):
        frame = lookup[str(row.dataset_name)]
        selected = frame.iloc[int(row.input_start_index) : int(row.input_end_index) + 1]
        if len(selected) != context_length:
            raise ValueError(f"{row.window_id} has invalid context length.")
        if pd.Timestamp(selected["Time"].iloc[-1]) != pd.Timestamp(row.timestamp):
            raise ValueError(f"{row.window_id} timestamp is not aligned.")
        x_raw_rows.append(selected["Signal"].to_numpy(dtype=np.float32))
    x_raw = (
        np.stack(x_raw_rows).astype(np.float32)
        if x_raw_rows
        else np.empty((0, context_length), dtype=np.float32)
    )
    x_threshold = np.stack(
        [relative_to_threshold(row, threshold_db=threshold_db) for row in x_raw]
    ).astype(np.float32)
    x_current = np.stack([relative_to_current(row) for row in x_raw]).astype(np.float32)
    scalar_raw = compute_scalar_context_features(
        x_raw,
        threshold_db=threshold_db,
        position_in_fade_samples=selected_index[
            "position_in_fade_samples"
        ].to_numpy(dtype=np.float32),
        sampling_time_seconds=sampling_time_seconds,
    )

    split_scalar_raw = {
        split: scalar_raw[selected_index["split"].eq(split).to_numpy()]
        for split in SPLITS
    }
    split_scalar, scalar_scaler = standardize_scalar_context_splits(split_scalar_raw)

    base_arrays = {
        "X_raw": x_raw,
        "X_relative_to_threshold": x_threshold,
        "X_relative_to_current": x_current,
        "y_long_fade": selected_index["y_long_fade"].to_numpy(dtype=np.int64),
        "event_duration_samples": selected_index[
            "event_duration_samples"
        ].to_numpy(dtype=np.int64),
        "event_duration_seconds": selected_index[
            "event_duration_seconds"
        ].to_numpy(dtype=np.float32),
        "threshold_db": np.full(len(selected_index), threshold_db, dtype=np.float32),
        "min_fade_duration_seconds": selected_index[
            "min_fade_duration_seconds"
        ].to_numpy(dtype=np.float32),
        "sampling_time_seconds": selected_index[
            "sampling_time_seconds"
        ].to_numpy(dtype=np.int64),
        "inside_event": selected_index["inside_event"].astype(bool).to_numpy(),
        "window_id": selected_index["window_id"].to_numpy(dtype=str),
        "event_id": selected_index["event_id"].to_numpy(dtype=str),
        "dataset_id": selected_index["dataset_id"].to_numpy(dtype=str),
        "global_window_id": selected_index["global_window_id"].to_numpy(dtype=str),
        "global_event_id": selected_index["global_event_id"].to_numpy(dtype=str),
        "position_in_fade_samples": selected_index[
            "position_in_fade_samples"
        ].to_numpy(dtype=np.int64),
        "elapsed_since_fade_start_seconds": selected_index[
            "elapsed_since_fade_start_seconds"
        ].to_numpy(dtype=np.float32),
    }
    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    split_metadata: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        mask = selected_index["split"].eq(split).to_numpy()
        arrays = {key: value[mask] for key, value in base_arrays.items()}
        arrays["scalar_context_features"] = split_scalar[split]
        x_lag_scalar, lag_scalar_names = build_lag_scalar_features(
            arrays["X_relative_to_threshold"],
            arrays["scalar_context_features"],
            context_length=context_length,
        )
        arrays["X_lag_scalar"] = x_lag_scalar
        arrays["scalar_context_feature_names"] = np.asarray(
            SCALAR_CONTEXT_FEATURE_NAMES,
            dtype=str,
        )
        arrays["lag_feature_names"] = np.asarray(lag_feature_names(context_length), dtype=str)
        arrays["lag_scalar_feature_names"] = lag_scalar_names
        split_arrays[split] = arrays
        split_metadata[split] = selected_index.loc[mask].reset_index(drop=True)
    return split_arrays, split_metadata, scalar_scaler


def save_split_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Save one split NPZ."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **dict(arrays))


def split_file_stem(split: str) -> str:
    """Return the split file stem."""

    return SPLIT_FILE_NAMES[split]


def load_raw_signal_frames(
    raw_data_dir: str | Path,
    *,
    dataset_names: Sequence[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Load raw CSV files into strict Time/Signal frames."""

    root = Path(raw_data_dir)
    names = list(dataset_names) if dataset_names else sorted(path.name for path in root.glob("*.csv"))
    frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for name in names:
        path = root / name
        frame, meta = load_signal_dataset(path)
        frame = (
            frame[["Time", "Signal"]]
            .dropna()
            .sort_values("Time", kind="stable")
            .drop_duplicates("Time", keep="first")
            .reset_index(drop=True)
        )
        frames[name] = frame
        metadata[name] = meta
    return frames, metadata


def split_summary(split_metadata: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize sample/event counts and positive rates by split."""

    rows = []
    for split, frame in split_metadata.items():
        positives = int(frame["y_long_fade"].sum()) if len(frame) else 0
        rows.append(
            {
                "split": "val" if split == "validation" else split,
                "num_samples": int(len(frame)),
                "num_events": int(frame["global_event_id"].nunique()) if len(frame) else 0,
                "num_positive_samples": positives,
                "positive_rate": float(positives / len(frame)) if len(frame) else np.nan,
                "num_positive_events": int(
                    frame.groupby("global_event_id")["y_long_fade"].first().sum()
                )
                if len(frame)
                else 0,
            }
        )
    return pd.DataFrame(rows)
