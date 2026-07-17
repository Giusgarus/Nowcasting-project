"""Dataset construction for continuous-time survival persistence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tasks.survival_persistence.data.event_state_machine import (
    StableRecoveryConfig,
    SurvivalFadeEvent,
    detect_survival_fade_events,
    threshold_condition,
    validate_recovery_config,
)
from src.tasks.survival_persistence.data.features import (
    FORBIDDEN_MODEL_INPUT_COLUMNS,
    SCALAR_CONTEXT_FEATURE_NAMES,
    compute_scalar_context_features,
    relative_to_current,
    relative_to_threshold,
    standardize_scalar_context_splits,
)

SPLITS = ("train", "validation", "test")
SPLIT_FILE_NAMES = {"train": "train", "validation": "val", "test": "test"}


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _context_is_contiguous(times: pd.Series, *, sampling_time_seconds: int) -> bool:
    if len(times) < 2:
        return True
    deltas = pd.to_datetime(times, errors="raise").diff().dt.total_seconds().dropna()
    return bool(np.all(deltas.to_numpy(dtype=float) <= sampling_time_seconds * 1.5))


def load_full_signal(path: str | Path, *, signal_column: str = "Signal") -> pd.DataFrame:
    """Load the full cleaned signal used for survival target construction."""

    frame = pd.read_parquet(path)
    required = {"dataset_id", "dataset_name", "segment_id", "Time", signal_column}
    _require_columns(frame, required, "full_signal")
    output = (
        frame.loc[:, ["dataset_id", "dataset_name", "segment_id", "Time", signal_column]]
        .rename(columns={signal_column: "Signal"})
        .dropna(subset=["Time", "Signal"])
        .sort_values(["dataset_id", "segment_id", "Time"], kind="stable")
        .drop_duplicates(["dataset_id", "segment_id", "Time"], keep="first")
        .reset_index(drop=True)
    )
    if not np.isfinite(output["Signal"].to_numpy(dtype=float)).all():
        raise ValueError("full_signal contains non-finite Signal values.")
    return output


def build_survival_event_index(
    full_signal: pd.DataFrame,
    *,
    recovery_config: StableRecoveryConfig,
) -> pd.DataFrame:
    """Return one row per detected survival event."""

    validate_recovery_config(recovery_config)
    required = {"dataset_id", "dataset_name", "segment_id", "Time", "Signal"}
    _require_columns(full_signal, required, "full_signal")

    rows: list[dict[str, Any]] = []
    for (dataset_id, dataset_name, segment_id), segment in full_signal.groupby(
        ["dataset_id", "dataset_name", "segment_id"],
        sort=False,
    ):
        events = detect_survival_fade_events(
            segment[["Time", "Signal"]],
            dataset_name=str(dataset_name),
            dataset_id=str(dataset_id),
            segment_id=str(segment_id),
            event_id_prefix="sp_event",
            config=recovery_config,
        )
        for event in events:
            rows.append(_event_to_row(event))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _event_to_row(event: SurvivalFadeEvent) -> dict[str, Any]:
    return {
        "dataset_name": event.dataset_name,
        "dataset_id": event.dataset_id,
        "segment_id": event.segment_id,
        "event_id": event.event_id,
        "event_start_time": event.event_start_time,
        "event_end_time": event.event_end_time,
        "event_end_confirmation_time": event.event_end_confirmation_time,
        "event_observed": event.event_observed,
        "censoring_reason": event.censoring_reason,
        "segment_end_time": event.segment_end_time,
        "start_index": event.start_index,
        "sample_end_index": event.sample_end_index,
        "event_duration_seconds": event.event_duration_seconds,
        "event_duration_samples": event.event_duration_samples,
        "observed_followup_seconds": event.observed_followup_seconds,
    }


def build_survival_window_index(
    full_signal: pd.DataFrame,
    *,
    recovery_config: StableRecoveryConfig,
    context_length: int,
    sampling_time_seconds: int,
    sample_policy: str = "all_event_timestamps",
    require_contiguous_context: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build one survival sample row per eligible event timestamp."""

    if context_length < 1:
        raise ValueError("context_length must be positive.")
    if sampling_time_seconds <= 0:
        raise ValueError("sampling_time_seconds must be positive.")
    if sample_policy != "all_event_timestamps":
        raise ValueError("Only sample_policy=all_event_timestamps is supported.")

    event_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    diagnostics = {
        "num_detected_events": 0,
        "num_events_without_samples": 0,
        "num_context_exclusions": 0,
        "num_non_positive_target_exclusions": 0,
    }
    window_number = 0

    for dataset_key, dataset_frame in full_signal.groupby(
        ["dataset_id", "dataset_name"],
        sort=False,
    ):
        dataset_id, dataset_name = dataset_key
        event_counter = 0
        for segment_id, segment in dataset_frame.groupby("segment_id", sort=False):
            ordered = segment.sort_values("Time", kind="stable").reset_index(drop=True)
            events = detect_survival_fade_events(
                ordered[["Time", "Signal"]],
                dataset_name=str(dataset_name),
                dataset_id=str(dataset_id),
                segment_id=str(segment_id),
                event_id_prefix="sp_event",
                config=recovery_config,
            )
            for event in events:
                event_counter += 1
                event_id = f"sp_event_{event_counter:05d}"
                event = SurvivalFadeEvent(
                    **{**event.__dict__, "event_id": event_id}
                )
                diagnostics["num_detected_events"] += 1
                event_row = _event_to_row(event)
                event_rows.append(event_row)
                event_sample_count_before = len(sample_rows)
                _append_event_samples(
                    sample_rows,
                    ordered,
                    event,
                    recovery_config=recovery_config,
                    context_length=context_length,
                    sampling_time_seconds=sampling_time_seconds,
                    require_contiguous_context=require_contiguous_context,
                    next_window_number=lambda: _next_window_number(sample_rows),
                    diagnostics=diagnostics,
                )
                # Replace generated window IDs with a globally stable sequence.
                for row in sample_rows[event_sample_count_before:]:
                    window_number += 1
                    row["window_id"] = f"sp_window_{window_number:09d}"
                if len(sample_rows) == event_sample_count_before:
                    diagnostics["num_events_without_samples"] += 1

    window_index = pd.DataFrame(sample_rows)
    event_index = pd.DataFrame(event_rows)
    if not window_index.empty:
        counts = (
            window_index.groupby(["dataset_name", "event_id"], sort=False)
            .size()
            .rename("event_sample_count")
        )
        key = [window_index["dataset_name"], window_index["event_id"]]
        window_index["event_sample_count"] = (
            pd.MultiIndex.from_arrays(key).map(counts).astype(int)
        )
        window_index["event_balanced_weight"] = (
            1.0 / window_index["event_sample_count"].astype(float)
        )
    return window_index, event_index, diagnostics


def _next_window_number(rows: list[dict[str, Any]]) -> int:
    return len(rows) + 1


def _append_event_samples(
    rows: list[dict[str, Any]],
    segment: pd.DataFrame,
    event: SurvivalFadeEvent,
    *,
    recovery_config: StableRecoveryConfig,
    context_length: int,
    sampling_time_seconds: int,
    require_contiguous_context: bool,
    next_window_number: Any,
    diagnostics: dict[str, Any],
) -> None:
    signal = segment["Signal"].to_numpy(dtype=float)
    times = pd.to_datetime(segment["Time"], errors="raise")
    is_on = threshold_condition(
        signal,
        threshold=recovery_config.threshold_on,
        operator=recovery_config.threshold_on_operator,
    )
    sample_end = min(event.sample_end_index, len(segment) - 1)
    if sample_end < event.start_index:
        return

    last_above_time: pd.Timestamp | None = None
    consecutive_below = 0
    for position in range(event.start_index, sample_end + 1):
        timestamp = pd.Timestamp(times.iloc[position])
        if position < context_length - 1:
            diagnostics["num_context_exclusions"] += 1
            if is_on[position]:
                last_above_time = timestamp
                consecutive_below = 0
            else:
                consecutive_below += 1
            continue
        input_start = position - context_length + 1
        input_times = times.iloc[input_start : position + 1]
        if require_contiguous_context and not _context_is_contiguous(
            input_times,
            sampling_time_seconds=sampling_time_seconds,
        ):
            diagnostics["num_context_exclusions"] += 1
            continue
        if is_on[position]:
            last_above_time = timestamp
            consecutive_below = 0
        else:
            consecutive_below += 1

        if event.event_observed:
            assert event.event_end_time is not None
            y_time = float((event.event_end_time - timestamp).total_seconds())
            lower = y_time
            upper = y_time
        else:
            y_time = float((event.segment_end_time - timestamp).total_seconds())
            lower = y_time
            upper = np.inf
        if not np.isfinite(lower) or lower <= 0:
            diagnostics["num_non_positive_target_exclusions"] += 1
            continue
        elapsed_seconds = float((timestamp - event.event_start_time).total_seconds())
        elapsed_samples = int(position - event.start_index + 1)
        if last_above_time is None:
            time_since_last_above = elapsed_seconds
        else:
            time_since_last_above = float((timestamp - last_above_time).total_seconds())
        recent_start = max(event.start_index, position - 4)
        recent_fraction = float(is_on[recent_start : position + 1].mean())
        rows.append(
            {
                "timestamp": timestamp,
                "sample_time": timestamp,
                "dataset_name": event.dataset_name,
                "dataset_id": event.dataset_id,
                "segment_id": event.segment_id,
                "event_id": event.event_id,
                "window_id": f"sp_window_{next_window_number():09d}",
                "split": "",
                "input_start_index": int(input_start),
                "input_end_index": int(position),
                "input_start_time": pd.Timestamp(input_times.iloc[0]),
                "input_end_time": timestamp,
                "event_start_time": event.event_start_time,
                "event_end_time": event.event_end_time,
                "event_end_confirmation_time": event.event_end_confirmation_time,
                "event_observed": bool(event.event_observed),
                "censoring_reason": event.censoring_reason,
                "segment_end_time": event.segment_end_time,
                "elapsed_event_samples": elapsed_samples,
                "elapsed_since_event_start_seconds": elapsed_seconds,
                "current_signal": float(signal[position]),
                "current_above_threshold": bool(is_on[position]),
                "time_since_last_above_threshold_seconds": time_since_last_above,
                "consecutive_below_threshold_samples": int(consecutive_below),
                "fraction_above_threshold_recent_window": recent_fraction,
                "y_time_seconds": y_time,
                "y_event_observed": int(event.event_observed),
                "y_lower_bound_seconds": lower,
                "y_upper_bound_seconds": upper,
                "y_time_samples": int(np.ceil(y_time / float(sampling_time_seconds))),
                "target_source": "state_machine_stable_recovery_same_segment",
                "threshold_on": float(recovery_config.threshold_on),
                "threshold_off": float(recovery_config.threshold_off),
                "sampling_time_seconds": int(sampling_time_seconds),
            }
        )


def build_survival_arrays(
    full_signal: pd.DataFrame,
    selected_index: pd.DataFrame,
    *,
    context_length: int,
    threshold_on: float,
    standardize_scalar_context: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame], dict[str, Any]]:
    """Create split arrays, aligned metadata, and scalar scaler metadata."""

    required = {
        "dataset_name",
        "dataset_id",
        "segment_id",
        "event_id",
        "global_event_id",
        "global_window_id",
        "split",
        "input_start_index",
        "input_end_index",
        "y_time_seconds",
        "y_event_observed",
        "y_lower_bound_seconds",
        "y_upper_bound_seconds",
    }
    _require_columns(selected_index, required, "survival window index")
    if selected_index.groupby("global_event_id")["split"].nunique().max() > 1:
        raise ValueError("A survival event appears in more than one split.")
    lookup = {
        (str(dataset_id), str(segment_id)): frame.sort_values("Time", kind="stable")
        .reset_index(drop=True)
        for (dataset_id, segment_id), frame in full_signal.groupby(
            ["dataset_id", "segment_id"],
            sort=False,
        )
    }
    x_rows: list[np.ndarray] = []
    for row in selected_index.itertuples(index=False):
        frame = lookup[(str(row.dataset_id), str(row.segment_id))]
        selected = frame.iloc[int(row.input_start_index) : int(row.input_end_index) + 1]
        if len(selected) != context_length:
            raise ValueError(f"{row.window_id} has invalid context length.")
        if pd.Timestamp(selected["Time"].iloc[-1]) != pd.Timestamp(row.sample_time):
            raise ValueError(f"{row.window_id} context does not end at sample_time.")
        x_rows.append(selected["Signal"].to_numpy(dtype=np.float32))
    x_raw = np.stack(x_rows).astype(np.float32)
    x_threshold = np.stack(
        [relative_to_threshold(row, threshold_on=threshold_on) for row in x_raw]
    ).astype(np.float32)
    x_current = np.stack([relative_to_current(row) for row in x_raw]).astype(np.float32)
    scalar_raw = compute_scalar_context_features(
        x_raw,
        threshold_on=threshold_on,
        elapsed_since_event_start_seconds=selected_index[
            "elapsed_since_event_start_seconds"
        ].to_numpy(dtype=np.float32),
        elapsed_event_samples=selected_index[
            "elapsed_event_samples"
        ].to_numpy(dtype=np.float32),
        current_above_threshold=selected_index[
            "current_above_threshold"
        ].astype(float).to_numpy(dtype=np.float32),
        time_since_last_above_threshold_seconds=selected_index[
            "time_since_last_above_threshold_seconds"
        ].to_numpy(dtype=np.float32),
        consecutive_below_threshold_samples=selected_index[
            "consecutive_below_threshold_samples"
        ].to_numpy(dtype=np.float32),
        fraction_above_threshold_recent_window=selected_index[
            "fraction_above_threshold_recent_window"
        ].to_numpy(dtype=np.float32),
    )
    metadata_with_features = selected_index.copy()
    for feature_index, feature_name in enumerate(SCALAR_CONTEXT_FEATURE_NAMES):
        metadata_with_features[feature_name] = scalar_raw[:, feature_index]

    split_scalar_raw = {
        split: scalar_raw[selected_index["split"].eq(split).to_numpy()]
        for split in SPLITS
    }
    split_scalar, scalar_scaler = standardize_scalar_context_splits(
        split_scalar_raw,
        standardize=standardize_scalar_context,
    )

    base_arrays = {
        "X_raw": x_raw,
        "X_relative_to_threshold": x_threshold,
        "X_relative_to_current": x_current,
        "y_time_seconds": selected_index["y_time_seconds"].to_numpy(dtype=np.float32),
        "y_event_observed": selected_index["y_event_observed"].to_numpy(dtype=np.int64),
        "y_lower_bound_seconds": selected_index[
            "y_lower_bound_seconds"
        ].to_numpy(dtype=np.float32),
        "y_upper_bound_seconds": selected_index[
            "y_upper_bound_seconds"
        ].to_numpy(dtype=np.float32),
        "y_time_samples": selected_index["y_time_samples"].to_numpy(dtype=np.int64),
        "event_sample_count": selected_index[
            "event_sample_count"
        ].to_numpy(dtype=np.int64),
        "event_balanced_weight": selected_index[
            "event_balanced_weight"
        ].to_numpy(dtype=np.float32),
        "window_id": selected_index["window_id"].to_numpy(dtype=str),
        "event_id": selected_index["event_id"].to_numpy(dtype=str),
        "dataset_id": selected_index["dataset_id"].to_numpy(dtype=str),
        "global_window_id": selected_index["global_window_id"].to_numpy(dtype=str),
        "global_event_id": selected_index["global_event_id"].to_numpy(dtype=str),
    }
    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    split_metadata: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        mask = selected_index["split"].eq(split).to_numpy()
        arrays = {key: value[mask] for key, value in base_arrays.items()}
        arrays["scalar_context_features"] = split_scalar[split]
        arrays["scalar_context_feature_names"] = np.asarray(
            SCALAR_CONTEXT_FEATURE_NAMES,
            dtype=str,
        )
        split_arrays[split] = arrays
        split_metadata[split] = metadata_with_features.loc[mask].reset_index(drop=True)
    return split_arrays, split_metadata, scalar_scaler


def save_split_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Save one split NPZ."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **dict(arrays))


def summarize_split_metadata(split_metadata: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize split sizes, censoring, and survival times."""

    rows: list[dict[str, Any]] = []
    for split, frame in split_metadata.items():
        observed = frame.loc[frame["y_event_observed"].eq(1)]
        censored = frame.loc[frame["y_event_observed"].eq(0)]
        rows.append(
            {
                "split": split,
                "num_samples": int(len(frame)),
                "num_events": int(frame["global_event_id"].nunique()) if len(frame) else 0,
                "num_observed_samples": int(len(observed)),
                "num_censored_samples": int(len(censored)),
                "num_observed_events": int(observed["global_event_id"].nunique())
                if len(observed)
                else 0,
                "num_censored_events": int(censored["global_event_id"].nunique())
                if len(censored)
                else 0,
                "min_y_time_seconds": float(frame["y_time_seconds"].min())
                if len(frame)
                else np.nan,
                "median_y_time_seconds": float(frame["y_time_seconds"].median())
                if len(frame)
                else np.nan,
                "max_y_time_seconds": float(frame["y_time_seconds"].max())
                if len(frame)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_censoring(split_metadata: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize censoring reasons by split."""

    rows: list[dict[str, Any]] = []
    for split, frame in split_metadata.items():
        grouped = frame.groupby("censoring_reason", dropna=False, sort=False)
        for reason, reason_frame in grouped:
            rows.append(
                {
                    "split": split,
                    "censoring_reason": str(reason) if str(reason) else "observed",
                    "num_samples": int(len(reason_frame)),
                    "num_events": int(reason_frame["global_event_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def summarize_event_index(
    selected_index: pd.DataFrame,
    event_index: pd.DataFrame,
) -> pd.DataFrame:
    """Return one event-summary row for events represented in the dataset."""

    if selected_index.empty:
        return pd.DataFrame()
    event_stats = (
        selected_index.groupby("global_event_id", sort=False)
        .agg(
            split=("split", "first"),
            num_event_samples=("window_id", "count"),
            num_above_threshold_samples=("current_above_threshold", "sum"),
            num_below_threshold_inside_event=(
                "current_above_threshold",
                lambda values: int((~values.astype(bool)).sum()),
            ),
        )
        .reset_index()
    )
    events = event_index.copy()
    events["global_event_id"] = (
        events["dataset_name"].astype(str) + "::" + events["event_id"].astype(str)
    )
    return events.merge(event_stats, on="global_event_id", how="inner").sort_values(
        ["dataset_name", "event_start_time"],
        kind="stable",
    )


def split_metadata_for_yaml(
    split_metadata: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Return compact split metadata for YAML."""

    output: dict[str, dict[str, Any]] = {}
    for split, frame in split_metadata.items():
        key = "val" if split == "validation" else split
        output[key] = {
            "n_samples": int(len(frame)),
            "n_events": int(frame["global_event_id"].nunique()) if len(frame) else 0,
            "n_observed_samples": int(frame["y_event_observed"].eq(1).sum()),
            "n_censored_samples": int(frame["y_event_observed"].eq(0).sum()),
            "datasets": sorted(frame["dataset_name"].astype(str).unique().tolist())
            if len(frame)
            else [],
        }
    return output


def validate_model_input_fields() -> None:
    """Fail if scalar feature names accidentally include forbidden target fields."""

    overlap = set(SCALAR_CONTEXT_FEATURE_NAMES) & FORBIDDEN_MODEL_INPUT_COLUMNS
    if overlap:
        raise ValueError(f"Forbidden model input fields configured: {sorted(overlap)}")
