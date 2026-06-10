"""Descriptive signal-only data-quality analysis helpers."""

import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd


def safe_name(value: str) -> str:
    """Return a filesystem-friendly identifier."""

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def summarize_signal_dataframe(
    dataframe: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Combine loader metadata with basic temporal and Signal statistics."""

    summary = dict(metadata)
    if dataframe.empty:
        summary.update(
            {
                "start_time": pd.NaT,
                "end_time": pd.NaT,
                "duration_hours": np.nan,
                "min_signal": np.nan,
                "max_signal": np.nan,
                "mean_signal": np.nan,
                "std_signal": np.nan,
            }
        )
        return summary

    start_time = dataframe["Time"].min()
    end_time = dataframe["Time"].max()
    signal = dataframe["Signal"]
    summary.update(
        {
            "start_time": start_time,
            "end_time": end_time,
            "duration_hours": (end_time - start_time).total_seconds() / 3600,
            "min_signal": float(signal.min()),
            "max_signal": float(signal.max()),
            "mean_signal": float(signal.mean()),
            "std_signal": float(signal.std()),
        }
    )
    return summary


def compute_time_differences(dataframe: pd.DataFrame) -> pd.Series:
    """Return consecutive Time differences in seconds."""

    return dataframe["Time"].diff().dt.total_seconds().dropna()


def find_continuous_segments(
    dataframe: pd.DataFrame,
    gap_threshold_seconds: float,
) -> list[tuple[int, int]]:
    """Return half-open row intervals separated by gaps above the threshold."""

    if gap_threshold_seconds <= 0:
        raise ValueError("gap_threshold_seconds must be positive.")
    if dataframe.empty:
        return []

    differences = dataframe["Time"].diff().dt.total_seconds()
    segment_starts = [0, *np.flatnonzero(differences.to_numpy() > gap_threshold_seconds)]
    segment_stops = [*segment_starts[1:], len(dataframe)]
    return list(zip(segment_starts, segment_stops, strict=True))


def compute_sampling_summary(
    dataframe: pd.DataFrame,
    gap_threshold_seconds: float = 90,
) -> dict[str, Any]:
    """Compute sampling-interval and continuous-segment statistics."""

    differences = compute_time_differences(dataframe)
    segments = find_continuous_segments(dataframe, gap_threshold_seconds)
    segment_lengths = np.array([stop - start for start, stop in segments], dtype=int)
    segment_durations = [
        (
            dataframe["Time"].iloc[stop - 1] - dataframe["Time"].iloc[start]
        ).total_seconds()
        / 3600
        for start, stop in segments
    ]

    def statistic(name: str) -> float:
        return float(getattr(differences, name)()) if not differences.empty else np.nan

    return {
        "median_dt_seconds": statistic("median"),
        "mean_dt_seconds": statistic("mean"),
        "std_dt_seconds": statistic("std"),
        "min_dt_seconds": statistic("min"),
        "max_dt_seconds": statistic("max"),
        "pct_dt_between_20_40_seconds": float(differences.between(20, 40).mean() * 100)
        if not differences.empty
        else np.nan,
        "num_non_positive_dt": int((differences <= 0).sum()),
        "num_dt_above_60_seconds": int((differences > 60).sum()),
        "num_dt_above_90_seconds": int((differences > 90).sum()),
        "num_dt_above_300_seconds": int((differences > 300).sum()),
        "num_segments": len(segments),
        "longest_segment_samples": int(segment_lengths.max())
        if len(segment_lengths)
        else 0,
        "longest_segment_duration_hours": max(segment_durations, default=0.0),
        "median_segment_samples": float(np.median(segment_lengths))
        if len(segment_lengths)
        else 0.0,
    }


def count_valid_windows_by_segments(
    dataframe: pd.DataFrame,
    context_lengths: Sequence[int],
    prediction_lengths: Sequence[int],
    gap_threshold_seconds: float,
) -> dict[str, Any]:
    """Count autoregressive windows that remain inside continuous segments."""

    segments = find_continuous_segments(dataframe, gap_threshold_seconds)
    segment_lengths = [stop - start for start, stop in segments]
    result: dict[str, Any] = {
        "valid_rows": len(dataframe),
        "num_segments": len(segments),
        "longest_segment_samples": max(segment_lengths, default=0),
    }
    for context_length in context_lengths:
        for prediction_length in prediction_lengths:
            if context_length < 1 or prediction_length < 1:
                raise ValueError("Window lengths must be positive.")
            result[f"windows_L{context_length}_h{prediction_length}"] = sum(
                max(0, length - context_length - prediction_length + 1)
                for length in segment_lengths
            )
    return result


def compute_signal_distribution_summary(dataframe: pd.DataFrame) -> dict[str, float]:
    """Return requested descriptive Signal quantiles and statistics."""

    signal = dataframe["Signal"]
    quantiles = signal.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "min": float(signal.min()),
        "q01": float(quantiles.loc[0.01]),
        "q05": float(quantiles.loc[0.05]),
        "q10": float(quantiles.loc[0.10]),
        "q25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.50]),
        "mean": float(signal.mean()),
        "q75": float(quantiles.loc[0.75]),
        "q90": float(quantiles.loc[0.90]),
        "q95": float(quantiles.loc[0.95]),
        "q99": float(quantiles.loc[0.99]),
        "max": float(signal.max()),
        "std": float(signal.std()),
    }


def find_boolean_runs(mask: Iterable[bool]) -> list[tuple[int, int]]:
    """Return half-open intervals for contiguous True runs."""

    values = np.asarray(list(mask), dtype=bool)
    if not len(values):
        return []
    changes = np.diff(np.concatenate(([False], values, [False])).astype(int))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def compute_quantile_run_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarize exploratory high- and low-Signal quantile runs."""

    signal = dataframe["Signal"]
    conditions = [
        ("above_q90", "above", 0.90),
        ("above_q95", "above", 0.95),
        ("above_q99", "above", 0.99),
        ("below_q10", "below", 0.10),
        ("below_q05", "below", 0.05),
        ("below_q01", "below", 0.01),
    ]
    rows = []
    for condition, direction, quantile in conditions:
        threshold = float(signal.quantile(quantile))
        mask = signal > threshold if direction == "above" else signal < threshold
        runs = find_boolean_runs(mask)
        lengths = np.array([stop - start for start, stop in runs], dtype=int)
        rows.append(
            {
                "condition": condition,
                "threshold": threshold,
                "num_runs": len(runs),
                "max_run_samples": int(lengths.max()) if len(lengths) else 0,
                "median_run_samples": float(np.median(lengths)) if len(lengths) else 0.0,
                "num_runs_at_least_10_samples": int((lengths >= 10).sum()),
                "num_runs_at_least_20_samples": int((lengths >= 20).sum()),
            }
        )
    return pd.DataFrame(rows)


def longest_quantile_run(
    dataframe: pd.DataFrame,
    quantile: float,
    direction: Literal["above", "below"],
) -> tuple[int, int] | None:
    """Return the longest exploratory quantile run."""

    threshold = dataframe["Signal"].quantile(quantile)
    mask = (
        dataframe["Signal"] > threshold
        if direction == "above"
        else dataframe["Signal"] < threshold
    )
    runs = find_boolean_runs(mask)
    return max(runs, key=lambda interval: interval[1] - interval[0], default=None)
