"""Dataset construction for the current-level persistence task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test")
SPLIT_FILE_NAMES = {"train": "train", "validation": "val", "test": "test"}
SCALAR_CONTEXT_FEATURE_NAMES = (
    "current_signal",
    "recovery_level",
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
)


@dataclass(frozen=True)
class PersistenceTarget:
    """Observed duration or a lower bound when recovery is not observed."""

    observed: bool
    duration_seconds: float | None
    duration_samples: int | None
    recovery_time: pd.Timestamp | None
    censoring_lower_bound_seconds: float


class _SegmentRecoveryIndex:
    """Find the first future value below a query level in logarithmic time."""

    def __init__(self, frame: pd.DataFrame, *, signal_column: str) -> None:
        ordered = frame.sort_values("Time", kind="stable")
        if ordered["Time"].duplicated().any():
            raise ValueError("Full clean signal contains duplicate segment timestamps.")
        self.times = (
            pd.to_datetime(ordered["Time"], errors="raise")
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        self.signals = ordered[signal_column].to_numpy(dtype=np.float32)
        if not np.isfinite(self.signals).all():
            raise ValueError("Full clean signal contains non-finite values.")
        self.n_observations = len(self.signals)
        self.tree_size = 1 << max(0, self.n_observations - 1).bit_length()
        self.minimum_tree = np.full(
            self.tree_size * 2,
            np.inf,
            dtype=np.float32,
        )
        self.minimum_tree[
            self.tree_size : self.tree_size + self.n_observations
        ] = self.signals
        level_start = self.tree_size
        while level_start > 1:
            parent_start = level_start // 2
            self.minimum_tree[parent_start:level_start] = np.minimum(
                self.minimum_tree[level_start : 2 * level_start : 2],
                self.minimum_tree[level_start + 1 : 2 * level_start : 2],
            )
            level_start = parent_start

    def first_below_after(
        self,
        timestamp: pd.Timestamp,
        threshold: float,
    ) -> int | None:
        """Return the first index after ``timestamp`` whose signal is lower."""

        start = int(np.searchsorted(self.times, pd.Timestamp(timestamp).value, side="right"))
        result = self._search_first_below(
            node=1,
            left=0,
            right=self.tree_size,
            start=start,
            threshold=float(threshold),
        )
        return None if result < 0 or result >= self.n_observations else result

    def _search_first_below(
        self,
        *,
        node: int,
        left: int,
        right: int,
        start: int,
        threshold: float,
    ) -> int:
        if right <= start or self.minimum_tree[node] >= threshold:
            return -1
        if right - left == 1:
            return left
        middle = (left + right) // 2
        result = self._search_first_below(
            node=node * 2,
            left=left,
            right=middle,
            start=start,
            threshold=threshold,
        )
        if result >= 0:
            return result
        return self._search_first_below(
            node=node * 2 + 1,
            left=middle,
            right=right,
            start=start,
            threshold=threshold,
        )

    def persistence_target(
        self,
        *,
        timestamp: pd.Timestamp,
        current_signal: float,
        delta: float,
        sampling_time_seconds: int,
    ) -> PersistenceTarget:
        """Return an observed recovery duration or a right-censoring bound."""

        recovery_index = self.first_below_after(timestamp, current_signal - delta)
        current_time = pd.Timestamp(timestamp)
        if recovery_index is not None:
            recovery_time = pd.Timestamp(self.times[recovery_index])
            duration_seconds = float((recovery_time - current_time).total_seconds())
            if duration_seconds <= 0:
                raise ValueError("Recovery must occur after the decision timestamp.")
            return PersistenceTarget(
                observed=True,
                duration_seconds=duration_seconds,
                duration_samples=int(
                    np.ceil(duration_seconds / float(sampling_time_seconds))
                ),
                recovery_time=recovery_time,
                censoring_lower_bound_seconds=duration_seconds,
            )

        segment_end = pd.Timestamp(self.times[-1])
        lower_bound = max(0.0, float((segment_end - current_time).total_seconds()))
        return PersistenceTarget(
            observed=False,
            duration_seconds=None,
            duration_samples=None,
            recovery_time=None,
            censoring_lower_bound_seconds=lower_bound,
        )


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


def compute_scalar_context_features(
    x_raw: np.ndarray,
    *,
    delta: float | np.ndarray,
) -> np.ndarray:
    """Compute ordered absolute-level and summary features for raw windows.

    Short contexts use the earliest available value for slope spans longer than
    the window. Standard deviations use ``ddof=0``.
    """

    values = np.asarray(x_raw, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("X_raw must have shape (N, context_length) with L >= 1.")
    if not np.isfinite(values).all():
        raise ValueError("X_raw contains non-finite values.")

    delta_values = np.asarray(delta, dtype=np.float32)
    if delta_values.ndim == 0:
        delta_values = np.full(len(values), delta_values, dtype=np.float32)
    else:
        delta_values = delta_values.reshape(-1)
    if len(delta_values) != len(values):
        raise ValueError("delta must be scalar or contain one value per window.")

    current = values[:, -1]
    window_min = values.min(axis=1)
    window_max = values.max(axis=1)

    def slope(span: int) -> np.ndarray:
        reference_index = -min(span, values.shape[1])
        return current - values[:, reference_index]

    recent = values[:, -min(5, values.shape[1]) :]
    features = np.column_stack(
        [
            current,
            current - delta_values,
            values.mean(axis=1),
            values.std(axis=1, ddof=0),
            window_min,
            window_max,
            window_max - window_min,
            slope(2) if values.shape[1] >= 2 else np.zeros(len(values)),
            slope(3),
            slope(5),
            recent.mean(axis=1),
            recent.std(axis=1, ddof=0),
        ]
    )
    return features.astype(np.float32)


def ensure_scalar_context_features(
    arrays: Mapping[str, np.ndarray],
    *,
    scalar_context_key: str = "scalar_context_features",
    raw_input_key: str = "X_raw",
    delta: float | None = None,
) -> dict[str, np.ndarray]:
    """Return split arrays with canonical scalar context features available.

    Existing feature matrices are validated and reordered when their stored
    names contain the canonical feature set. Older datasets fall back to
    computing features from ``X_raw``.
    """

    output = dict(arrays)
    expected_names = list(SCALAR_CONTEXT_FEATURE_NAMES)
    if scalar_context_key in output:
        features = np.asarray(output[scalar_context_key], dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"{scalar_context_key} must have shape (N, features).")
        stored_names = output.get("scalar_context_feature_names")
        if stored_names is None:
            if features.shape[1] != len(expected_names):
                raise ValueError(
                    f"{scalar_context_key} has {features.shape[1]} columns; "
                    f"expected {len(expected_names)}."
                )
        else:
            names = [str(name) for name in np.asarray(stored_names).tolist()]
            if set(names) != set(expected_names) or len(names) != len(expected_names):
                raise ValueError(
                    "scalar_context_feature_names does not match the required features."
                )
            features = features[:, [names.index(name) for name in expected_names]]
        output[scalar_context_key] = features
    else:
        if raw_input_key not in output:
            raise ValueError(
                f"Scalar context was requested, but neither {scalar_context_key!r} "
                f"nor fallback raw input {raw_input_key!r} is available. Rebuild "
                "the current-level persistence dataset."
            )
        delta_values: float | np.ndarray
        if "delta" in output:
            delta_values = output["delta"]
        elif delta is not None:
            delta_values = float(delta)
        else:
            raise ValueError(
                "Cannot derive recovery_level for scalar context: delta is missing."
            )
        output[scalar_context_key] = compute_scalar_context_features(
            output[raw_input_key],
            delta=delta_values,
        )
    if len(output[scalar_context_key]) != len(output[raw_input_key]):
        raise ValueError("Scalar context and raw input arrays have different lengths.")
    if not np.isfinite(output[scalar_context_key]).all():
        raise ValueError("Scalar context features contain non-finite values.")
    output["scalar_context_feature_names"] = np.asarray(expected_names, dtype=str)
    return output


def prepare_scalar_context_splits(
    split_arrays: Mapping[str, Mapping[str, np.ndarray]],
    *,
    scalar_context_key: str = "scalar_context_features",
    raw_input_key: str = "X_raw",
    delta: float | None = None,
    standardize: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Prepare scalar context using statistics fitted on the train split only."""

    prepared = {
        split: ensure_scalar_context_features(
            arrays,
            scalar_context_key=scalar_context_key,
            raw_input_key=raw_input_key,
            delta=delta,
        )
        for split, arrays in split_arrays.items()
    }
    if "train" not in prepared or len(prepared["train"][scalar_context_key]) == 0:
        raise ValueError("A non-empty train split is required for scalar standardization.")
    train_features = prepared["train"][scalar_context_key]
    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    observed_std = train_features.std(axis=0, ddof=0, dtype=np.float64).astype(
        np.float32
    )
    scale = np.where(observed_std > 0.0, observed_std, 1.0).astype(np.float32)
    if standardize:
        for arrays in prepared.values():
            arrays[scalar_context_key] = (
                (arrays[scalar_context_key] - mean) / scale
            ).astype(np.float32)
    scaler = {
        "feature_names": list(SCALAR_CONTEXT_FEATURE_NAMES),
        "mean": mean.tolist(),
        "std": scale.tolist(),
        "observed_std": observed_std.tolist(),
        "fitted_on": "train",
        "standardized": bool(standardize),
    }
    return prepared, scaler


def build_current_level_persistence_index(
    event_windows: pd.DataFrame,
    event_quality: pd.DataFrame,
    full_signal: pd.DataFrame,
    *,
    context_length: int,
    delta: float,
    sampling_time_seconds: int,
    signal_column: str,
    full_signal_column: str,
    allow_warning_events: bool,
) -> pd.DataFrame:
    """Build windows whose targets search the full continuous signal segment."""

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
    required_quality = {
        "dataset_id",
        "dataset_name",
        "event_id",
        "quality_flag",
        "quality_reason",
    }
    missing_quality = sorted(required_quality - set(event_quality.columns))
    if missing_quality:
        raise ValueError(
            f"Event quality is missing required columns: {missing_quality}"
        )
    quality_keys = ["dataset_id", "dataset_name", "event_id"]
    if event_quality.duplicated(quality_keys).any():
        raise ValueError("Event quality must contain one row per event.")

    quality_lookup = event_quality.set_index(quality_keys)[
        ["quality_flag", "quality_reason"]
    ]
    event_keys = event_windows[quality_keys].drop_duplicates()
    missing_event_quality = event_keys.merge(
        event_quality[quality_keys],
        on=quality_keys,
        how="left",
        indicator=True,
    ).loc[lambda frame: frame["_merge"].eq("left_only"), quality_keys]
    if not missing_event_quality.empty:
        examples = missing_event_quality.head(5).to_dict(orient="records")
        raise ValueError(
            "Event quality does not cover all event windows. "
            f"Examples: {examples}"
        )
    allowed_quality = {"usable", "warning"} if allow_warning_events else {"usable"}
    required_full_signal = {
        "dataset_id",
        "segment_id",
        "Time",
        full_signal_column,
    }
    missing_full_signal = sorted(required_full_signal - set(full_signal.columns))
    if missing_full_signal:
        raise ValueError(
            f"Full clean signal is missing required columns: {missing_full_signal}"
        )

    allowed_event_ids = set(
        event_quality.loc[
            event_quality["quality_flag"].isin(allowed_quality),
            "event_id",
        ].astype(str)
    )
    needed_segment_keys = set(
        event_windows.loc[
            event_windows["event_id"].astype(str).isin(allowed_event_ids),
            ["dataset_id", "segment_id"],
        ].itertuples(index=False, name=None)
    )
    recovery_indices = {
        key: _SegmentRecoveryIndex(frame, signal_column=full_signal_column)
        for key, frame in full_signal.groupby(
            ["dataset_id", "segment_id"],
            sort=False,
        )
        if key in needed_segment_keys
    }
    missing_segments = sorted(needed_segment_keys - set(recovery_indices))
    if missing_segments:
        raise ValueError(
            "Full clean signal does not contain all required segments. "
            f"Examples: {missing_segments[:5]}"
        )

    rows: list[dict[str, Any]] = []
    window_number = 0
    group_columns = ["dataset_id", "dataset_name", "event_id", "segment_id"]
    for group_key, group in event_windows.groupby(group_columns, sort=False):
        dataset_id, dataset_name, event_id, segment_id = group_key
        ordered = group.sort_values("event_point_idx", kind="stable").reset_index(
            drop=True
        )
        quality = quality_lookup.loc[(dataset_id, dataset_name, event_id)]
        quality_flag = str(quality["quality_flag"])
        quality_reason = str(quality["quality_reason"])
        num_candidate_windows = max(0, len(ordered) - context_length + 1)
        if quality_flag not in allowed_quality:
            # Preserve IDs of all previously valid windows when quality rules tighten.
            window_number += num_candidate_windows
            continue
        recovery_index = recovery_indices[(dataset_id, segment_id)]
        signals = ordered[signal_column].to_numpy(dtype=np.float32)
        for position in range(context_length - 1, len(ordered)):
            window_number += 1
            current_signal = float(signals[position])
            timestamp = pd.Timestamp(ordered["Time"].iloc[position])
            target = recovery_index.persistence_target(
                timestamp=timestamp,
                current_signal=current_signal,
                delta=delta,
                sampling_time_seconds=sampling_time_seconds,
            )
            input_rows = ordered.iloc[position - context_length + 1 : position + 1]
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
                    "target_observed": bool(target.observed),
                    "recovery_time": target.recovery_time,
                    "remaining_persistence_samples": target.duration_samples,
                    "remaining_persistence_seconds": target.duration_seconds,
                    "log1p_remaining_persistence_seconds": (
                        float(np.log1p(target.duration_seconds))
                        if target.duration_seconds is not None
                        else np.nan
                    ),
                    "censoring_lower_bound_seconds": float(
                        target.censoring_lower_bound_seconds
                    ),
                    "target_source": "full_clean_signal_same_segment",
                    "delta": float(delta),
                    "context_length": int(context_length),
                    "sampling_time_seconds": int(sampling_time_seconds),
                    "quality_flag": quality_flag,
                    "quality_reason": quality_reason,
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
        "target_observed",
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
    if not window_index["target_observed"].astype(bool).all():
        raise ValueError("Censored targets must be excluded from supervised arrays.")
    target_columns = [
        "remaining_persistence_samples",
        "remaining_persistence_seconds",
        "log1p_remaining_persistence_seconds",
    ]
    if not np.isfinite(window_index[target_columns].to_numpy(dtype=float)).all():
        raise ValueError("Observed persistence targets must be finite.")

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
    scalar_context = compute_scalar_context_features(
        x_raw,
        delta=window_index["delta"].to_numpy(dtype=np.float32),
    )
    metadata_with_features = window_index.copy()
    for feature_index, feature_name in enumerate(SCALAR_CONTEXT_FEATURE_NAMES):
        metadata_with_features[feature_name] = scalar_context[:, feature_index]

    all_arrays = {
        "X_raw": x_raw,
        "X_relative_to_current": x_relative,
        "scalar_context_features": scalar_context,
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
        split_arrays[split]["scalar_context_feature_names"] = np.asarray(
            SCALAR_CONTEXT_FEATURE_NAMES,
            dtype=str,
        )
        split_metadata[split] = metadata_with_features.loc[mask].reset_index(drop=True)
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


def split_metadata_for_yaml(
    split_metadata: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Return compact split metadata for ``dataset_metadata.yaml``."""

    output: dict[str, dict[str, Any]] = {}
    for split, frame in split_metadata.items():
        key = "val" if split == "validation" else split
        output[key] = {
            "n_samples": int(len(frame)),
            "n_events": int(frame["global_event_id"].nunique()) if len(frame) else 0,
            "datasets": sorted(frame["dataset_name"].astype(str).unique().tolist())
            if len(frame)
            else [],
        }
    return output


def target_statistics_seconds_for_yaml(
    split_metadata: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, float | None]]:
    """Return target-duration statistics in seconds for each split."""

    output: dict[str, dict[str, float | None]] = {}
    for split, frame in split_metadata.items():
        key = "val" if split == "validation" else split
        if frame.empty:
            output[key] = {
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "p10": None,
                "p90": None,
            }
            continue
        values = frame["remaining_persistence_seconds"].to_numpy(dtype=float)
        output[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
        }
    return output
