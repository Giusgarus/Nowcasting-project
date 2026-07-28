"""Candidate fade-event detection and event-window preparation."""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

EventCondition = Literal["greater_than", "greater_than_or_equal", "greater_or_equal"]
EVENT_QUALITY_COLUMNS = [
    "event_id",
    "dataset_id",
    "dataset_name",
    "event_timestamp",
    "window_start",
    "window_end",
    "actual_start_time",
    "actual_end_time",
    "num_samples",
    "expected_samples_approx",
    "coverage_ratio",
    "pre_event_coverage_minutes",
    "post_event_coverage_minutes",
    "median_dt_seconds",
    "mean_dt_seconds",
    "max_dt_seconds",
    "num_gaps_above_60s",
    "num_gaps_above_90s",
    "num_gaps_above_300s",
    "num_missing_signal",
    "num_duplicate_timestamps",
    "num_points_at_or_above_threshold",
    "num_points_below_threshold",
    "pct_points_at_or_above_threshold",
    "max_signal",
    "mean_signal",
    "std_signal",
    "num_imputed_points",
    "max_consecutive_imputed_points",
    "quality_flag",
    "quality_reason",
]


def threshold_folder_name(signal_threshold: float) -> str:
    """Return a filesystem-safe threshold folder name."""

    value = f"{signal_threshold:.10g}".replace("-", "neg").replace(".", "p")
    if "p" not in value:
        value += "p0"
    return f"threshold_{value}"


def detect_candidate_fade_events(
    clean_signal: pd.DataFrame,
    *,
    signal_threshold: float,
    condition: EventCondition,
    grouping_hours: float,
    window_pre_hours: float,
    window_post_hours: float,
) -> pd.DataFrame:
    """Detect threshold crossings grouped relative to each event's first crossing."""

    if condition not in {"greater_than", "greater_than_or_equal", "greater_or_equal"}:
        raise ValueError(
            "Only condition='greater_than' or 'greater_than_or_equal' is currently supported."
        )
    if grouping_hours <= 0 or window_pre_hours < 0 or window_post_hours < 0:
        raise ValueError("Event grouping and window durations must be valid.")

    columns = [
        "event_id",
        "dataset_id",
        "dataset_name",
        "segment_id",
        "event_timestamp",
        "window_start",
        "window_end",
        "signal_threshold",
        "grouping_hours",
        "window_pre_hours",
        "window_post_hours",
        "num_crossings",
        "first_crossing_signal",
        "max_signal_in_event_group",
        "actual_first_crossing_time",
        "actual_last_crossing_time",
    ]
    if clean_signal.empty:
        return pd.DataFrame(columns=columns)

    if condition == "greater_than":
        crossing_mask = clean_signal["Signal"].gt(signal_threshold)
    else:
        crossing_mask = clean_signal["Signal"].ge(signal_threshold)
    crossings = clean_signal.loc[crossing_mask].sort_values(
        "Time",
        kind="stable",
    )
    if crossings.empty:
        return pd.DataFrame(columns=columns)

    grouping_delta = pd.Timedelta(hours=grouping_hours)
    rows: list[dict[str, Any]] = []
    for dataset_id, dataset_crossings in crossings.groupby("dataset_id", sort=False):
        current_group: list[pd.Series] = []
        event_number = 0
        event_start: pd.Timestamp | None = None
        for _, crossing in dataset_crossings.iterrows():
            crossing_time = crossing["Time"]
            if event_start is None or crossing_time - event_start > grouping_delta:
                if current_group:
                    event_number += 1
                    rows.append(
                        _event_row(
                            current_group,
                            event_number,
                            signal_threshold,
                            grouping_hours,
                            window_pre_hours,
                            window_post_hours,
                        )
                    )
                current_group = [crossing]
                event_start = crossing_time
            else:
                current_group.append(crossing)
        if current_group:
            event_number += 1
            rows.append(
                _event_row(
                    current_group,
                    event_number,
                    signal_threshold,
                    grouping_hours,
                    window_pre_hours,
                    window_post_hours,
                )
            )
    return pd.DataFrame(rows, columns=columns)


def build_event_windows(
    clean_signal: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Extract observed rows inside each configured event-centered window."""

    columns = [
        "dataset_id",
        "dataset_name",
        "segment_id",
        "event_id",
        "event_timestamp",
        "window_start",
        "window_end",
        "Time",
        "Signal_original",
        "Signal_prepared",
        "relative_time_minutes",
        "is_imputed",
    ]
    frames = []
    for event in events.itertuples(index=False):
        selected = clean_signal.loc[
            clean_signal["dataset_id"].eq(event.dataset_id)
            & clean_signal["Time"].between(event.window_start, event.window_end)
        ].copy()
        if selected.empty:
            continue
        selected["event_id"] = event.event_id
        selected["event_timestamp"] = event.event_timestamp
        selected["window_start"] = event.window_start
        selected["window_end"] = event.window_end
        selected["Signal_original"] = selected["Signal"]
        selected["Signal_prepared"] = selected["Signal"]
        selected["relative_time_minutes"] = (
            selected["Time"] - event.event_timestamp
        ).dt.total_seconds() / 60
        selected["is_imputed"] = False
        frames.append(selected[columns])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def prepare_and_assess_event_windows(
    event_windows: pd.DataFrame,
    events: pd.DataFrame,
    *,
    expected_seconds: float,
    signal_threshold: float,
    min_coverage_ratio: float,
    max_gap_seconds_for_imputation: float,
    max_small_gap_samples: int,
    max_warning_gap_samples: int,
    context_length: int,
    prediction_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute recoverable event-window gaps and assess event-level quality."""

    if expected_seconds <= 0:
        raise ValueError("expected_seconds must be positive.")

    prepared_frames = []
    quality_rows = []
    grouped_windows = {
        event_id: group.copy()
        for event_id, group in event_windows.groupby("event_id", sort=False)
    }
    for event in events.itertuples(index=False):
        observed = grouped_windows.get(event.event_id, pd.DataFrame()).copy()
        prepared, inferred_missing, max_missing_run, original_max_gap = _impute_event_window(
            observed,
            expected_seconds=expected_seconds,
            max_gap_seconds_for_imputation=max_gap_seconds_for_imputation,
            max_warning_gap_samples=max_warning_gap_samples,
        )
        if not prepared.empty:
            prepared["event_point_idx"] = range(len(prepared))
            prepared_frames.append(prepared)
        quality_rows.append(
            _quality_row(
                event,
                prepared,
                inferred_missing=inferred_missing,
                max_missing_run=max_missing_run,
                original_max_gap=original_max_gap,
                expected_seconds=expected_seconds,
                signal_threshold=signal_threshold,
                min_coverage_ratio=min_coverage_ratio,
                max_gap_seconds_for_imputation=max_gap_seconds_for_imputation,
                max_small_gap_samples=max_small_gap_samples,
                max_warning_gap_samples=max_warning_gap_samples,
                minimum_window_samples=context_length + prediction_length,
            )
        )

    prepared_all = (
        pd.concat(prepared_frames, ignore_index=True)
        if prepared_frames
        else event_windows.assign(event_point_idx=pd.Series(dtype=int))
    )
    return prepared_all, pd.DataFrame(quality_rows, columns=EVENT_QUALITY_COLUMNS)


def _event_row(
    group: Sequence[pd.Series],
    event_number: int,
    signal_threshold: float,
    grouping_hours: float,
    window_pre_hours: float,
    window_post_hours: float,
) -> dict[str, Any]:
    first = group[0]
    event_timestamp = first["Time"]
    dataset_id = first["dataset_id"]
    return {
        "event_id": f"{dataset_id}_event_{event_number:05d}",
        "dataset_id": dataset_id,
        "dataset_name": first["dataset_name"],
        "segment_id": first["segment_id"],
        "event_timestamp": event_timestamp,
        "window_start": event_timestamp - pd.Timedelta(hours=window_pre_hours),
        "window_end": event_timestamp + pd.Timedelta(hours=window_post_hours),
        "signal_threshold": signal_threshold,
        "grouping_hours": grouping_hours,
        "window_pre_hours": window_pre_hours,
        "window_post_hours": window_post_hours,
        "num_crossings": len(group),
        "first_crossing_signal": float(first["Signal"]),
        "max_signal_in_event_group": float(max(row["Signal"] for row in group)),
        "actual_first_crossing_time": group[0]["Time"],
        "actual_last_crossing_time": group[-1]["Time"],
    }


def _impute_event_window(
    observed: pd.DataFrame,
    *,
    expected_seconds: float,
    max_gap_seconds_for_imputation: float,
    max_warning_gap_samples: int,
) -> tuple[pd.DataFrame, int, int, float]:
    if observed.empty:
        return observed, 0, 0, np.nan

    observed = observed.sort_values("Time", kind="stable").reset_index(drop=True)
    additions: list[dict[str, Any]] = []
    inferred_missing = 0
    max_missing_run = 0
    original_gaps = observed["Time"].diff().dt.total_seconds()
    for index in range(1, len(observed)):
        previous = observed.iloc[index - 1]
        current = observed.iloc[index]
        dt_seconds = (current["Time"] - previous["Time"]).total_seconds()
        if dt_seconds <= expected_seconds * 1.5:
            continue
        missing_count = max(0, int(round(dt_seconds / expected_seconds)) - 1)
        inferred_missing += missing_count
        max_missing_run = max(max_missing_run, missing_count)
        if (
            missing_count < 1
            or missing_count > max_warning_gap_samples
            or dt_seconds > max_gap_seconds_for_imputation
            or previous["segment_id"] != current["segment_id"]
        ):
            continue
        for offset in range(1, missing_count + 1):
            time = previous["Time"] + pd.Timedelta(seconds=expected_seconds * offset)
            ratio = (time - previous["Time"]).total_seconds() / dt_seconds
            signal = previous["Signal_prepared"] + ratio * (
                current["Signal_prepared"] - previous["Signal_prepared"]
            )
            addition = previous.to_dict()
            addition.update(
                {
                    "Time": time,
                    "Signal_original": np.nan,
                    "Signal_prepared": float(signal),
                    "relative_time_minutes": (
                        time - previous["event_timestamp"]
                    ).total_seconds()
                    / 60,
                    "is_imputed": True,
                }
            )
            additions.append(addition)
    if additions:
        observed = pd.concat([observed, pd.DataFrame(additions)], ignore_index=True)
        observed = observed.sort_values("Time", kind="stable").reset_index(drop=True)
    return (
        observed,
        inferred_missing,
        max_missing_run,
        float(original_gaps.max()) if len(original_gaps) else np.nan,
    )


def _quality_row(
    event: Any,
    prepared: pd.DataFrame,
    *,
    inferred_missing: int,
    max_missing_run: int,
    original_max_gap: float,
    expected_seconds: float,
    signal_threshold: float,
    min_coverage_ratio: float,
    max_gap_seconds_for_imputation: float,
    max_small_gap_samples: int,
    max_warning_gap_samples: int,
    minimum_window_samples: int,
) -> dict[str, Any]:
    expected_samples = int(
        round((event.window_end - event.window_start).total_seconds() / expected_seconds)
    ) + 1
    if prepared.empty:
        return _empty_quality_row(event, expected_samples)

    times = prepared["Time"]
    differences = times.diff().dt.total_seconds().dropna()
    num_imputed = int(prepared["is_imputed"].sum())
    max_imputed_run = _max_true_run(prepared["is_imputed"])
    observed_samples = int(prepared["Signal_original"].notna().sum())
    coverage_ratio = min(1.0, observed_samples / expected_samples)
    reasons = []
    quality_flag = "usable"
    if coverage_ratio < min_coverage_ratio:
        quality_flag = "unusable"
        reasons.append("coverage_below_minimum")
    if len(prepared) < minimum_window_samples:
        quality_flag = "unusable"
        reasons.append("too_few_samples_for_autoregressive_window")
    if np.isfinite(original_max_gap) and original_max_gap > max_gap_seconds_for_imputation:
        quality_flag = "unusable"
        reasons.append("large_temporal_gap")
    if max_missing_run > max_warning_gap_samples:
        quality_flag = "unusable"
        reasons.append("missing_run_above_warning_limit")
    elif max_missing_run > max_small_gap_samples and quality_flag != "unusable":
        quality_flag = "warning"
        reasons.append("moderate_gap_imputed")
    elif num_imputed and quality_flag == "usable":
        reasons.append("small_gap_imputed")

    above = prepared["Signal_prepared"].ge(signal_threshold)
    return {
        "event_id": event.event_id,
        "dataset_id": event.dataset_id,
        "dataset_name": event.dataset_name,
        "event_timestamp": event.event_timestamp,
        "window_start": event.window_start,
        "window_end": event.window_end,
        "actual_start_time": times.min(),
        "actual_end_time": times.max(),
        "num_samples": len(prepared),
        "expected_samples_approx": expected_samples,
        "coverage_ratio": coverage_ratio,
        "pre_event_coverage_minutes": (event.event_timestamp - times.min()).total_seconds()
        / 60,
        "post_event_coverage_minutes": (times.max() - event.event_timestamp).total_seconds()
        / 60,
        "median_dt_seconds": float(differences.median()) if len(differences) else np.nan,
        "mean_dt_seconds": float(differences.mean()) if len(differences) else np.nan,
        "max_dt_seconds": float(differences.max()) if len(differences) else np.nan,
        "num_gaps_above_60s": int((differences > 60).sum()),
        "num_gaps_above_90s": int((differences > 90).sum()),
        "num_gaps_above_300s": int((differences > 300).sum()),
        "num_missing_signal": inferred_missing,
        "num_duplicate_timestamps": int(times.duplicated().sum()),
        "num_points_at_or_above_threshold": int(above.sum()),
        "num_points_below_threshold": int((~above).sum()),
        "pct_points_at_or_above_threshold": float(above.mean() * 100),
        "max_signal": float(prepared["Signal_prepared"].max()),
        "mean_signal": float(prepared["Signal_prepared"].mean()),
        "std_signal": float(prepared["Signal_prepared"].std()),
        "num_imputed_points": num_imputed,
        "max_consecutive_imputed_points": max_imputed_run,
        "quality_flag": quality_flag,
        "quality_reason": ";".join(reasons) if reasons else "none",
    }


def _empty_quality_row(event: Any, expected_samples: int) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "dataset_id": event.dataset_id,
        "dataset_name": event.dataset_name,
        "event_timestamp": event.event_timestamp,
        "window_start": event.window_start,
        "window_end": event.window_end,
        "actual_start_time": pd.NaT,
        "actual_end_time": pd.NaT,
        "num_samples": 0,
        "expected_samples_approx": expected_samples,
        "coverage_ratio": 0.0,
        "pre_event_coverage_minutes": 0.0,
        "post_event_coverage_minutes": 0.0,
        "median_dt_seconds": np.nan,
        "mean_dt_seconds": np.nan,
        "max_dt_seconds": np.nan,
        "num_gaps_above_60s": 0,
        "num_gaps_above_90s": 0,
        "num_gaps_above_300s": 0,
        "num_missing_signal": expected_samples,
        "num_duplicate_timestamps": 0,
        "num_points_at_or_above_threshold": 0,
        "num_points_below_threshold": 0,
        "pct_points_at_or_above_threshold": 0.0,
        "max_signal": np.nan,
        "mean_signal": np.nan,
        "std_signal": np.nan,
        "num_imputed_points": 0,
        "max_consecutive_imputed_points": 0,
        "quality_flag": "unusable",
        "quality_reason": "empty_event_window",
    }


def _max_true_run(values: pd.Series) -> int:
    groups = values.ne(values.shift()).cumsum()
    true_runs = values.groupby(groups).sum()
    return int(true_runs.max()) if len(true_runs) else 0
