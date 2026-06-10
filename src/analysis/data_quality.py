"""Descriptive signal-only data-quality analysis helpers."""

import re
from collections.abc import Iterable, Mapping, Sequence
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


def compute_rolling_baseline_summary(
    dataframe: pd.DataFrame,
    windows: Mapping[str, int],
) -> pd.DataFrame:
    """Summarize rolling baseline and variability diagnostics."""

    rows = []
    signal = dataframe["Signal"]
    for window_name, window_samples in windows.items():
        if window_samples < 1:
            raise ValueError("Rolling windows must contain at least one sample.")

        rolling = signal.rolling(window_samples, min_periods=window_samples)
        rolling_mean = rolling.mean()
        rolling_median = rolling.median()
        rolling_std = rolling.std()
        rolling_q05 = rolling.quantile(0.05)
        rolling_q95 = rolling.quantile(0.95)
        num_valid = int(rolling_mean.notna().sum())

        rows.append(
            {
                "window_name": window_name,
                "window_samples": window_samples,
                "status": "ok" if num_valid else "insufficient_points",
                "num_valid_points": num_valid,
                "rolling_mean_min": _series_statistic(rolling_mean, "min"),
                "rolling_mean_max": _series_statistic(rolling_mean, "max"),
                "rolling_mean_range": _series_range(rolling_mean),
                "rolling_median_min": _series_statistic(rolling_median, "min"),
                "rolling_median_max": _series_statistic(rolling_median, "max"),
                "rolling_median_range": _series_range(rolling_median),
                "rolling_std_median": _series_statistic(rolling_std, "median"),
                "rolling_std_q95": _series_quantile(rolling_std, 0.95),
                "rolling_q05_median": _series_statistic(rolling_q05, "median"),
                "rolling_q95_median": _series_statistic(rolling_q95, "median"),
            }
        )
    return pd.DataFrame(rows)


def compute_hourly_signal_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Group Signal statistics by hour of day."""

    grouped = dataframe.assign(hour=dataframe["Time"].dt.hour).groupby("hour")[
        "Signal"
    ]
    return grouped.agg(
        count="count",
        mean_signal="mean",
        median_signal="median",
        std_signal="std",
        q05_signal=lambda values: values.quantile(0.05),
        q95_signal=lambda values: values.quantile(0.95),
    ).reset_index()


def compute_daily_signal_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Group Signal statistics by calendar date."""

    grouped = dataframe.assign(date=dataframe["Time"].dt.date).groupby("date")[
        "Signal"
    ]
    return grouped.agg(
        count="count",
        mean_signal="mean",
        median_signal="median",
        std_signal="std",
        min_signal="min",
        max_signal="max",
        q05_signal=lambda values: values.quantile(0.05),
        q95_signal=lambda values: values.quantile(0.95),
    ).reset_index()


def compute_signal_changes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return consecutive Signal changes and their timing information."""

    signal = dataframe["Signal"]
    previous_signal = signal.shift(1)
    delta = signal - previous_signal
    safe_previous = previous_signal.abs().gt(np.finfo(float).eps)
    relative_delta = delta.div(previous_signal.abs().where(safe_previous))
    return pd.DataFrame(
        {
            "time_prev": dataframe["Time"].shift(1),
            "time_current": dataframe["Time"],
            "signal_prev": previous_signal,
            "signal_current": signal,
            "delta_signal": delta,
            "abs_delta_signal": delta.abs(),
            "relative_delta_signal": relative_delta,
            "dt_seconds": dataframe["Time"].diff().dt.total_seconds(),
        }
    )


def compute_signal_change_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Summarize local changes and robust spike-candidate thresholds."""

    changes = compute_signal_changes(dataframe)
    delta = _finite_values(changes["delta_signal"])
    abs_delta = _finite_values(changes["abs_delta_signal"])
    relative_delta = _finite_values(changes["relative_delta_signal"])
    q99_threshold = _series_quantile(abs_delta, 0.99)
    median_abs_delta = _series_statistic(abs_delta, "median")
    mad = _series_statistic((abs_delta - median_abs_delta).abs(), "median")
    mad_threshold = median_abs_delta + 5 * mad
    num_deltas = len(delta)
    num_spikes_q99 = int((abs_delta > q99_threshold).sum()) if num_deltas else 0
    num_spikes_mad = int((abs_delta > mad_threshold).sum()) if num_deltas else 0

    return {
        "num_points": len(dataframe),
        "num_deltas": num_deltas,
        "delta_mean": _series_statistic(delta, "mean"),
        "delta_std": _series_statistic(delta, "std"),
        "delta_q01": _series_quantile(delta, 0.01),
        "delta_q05": _series_quantile(delta, 0.05),
        "delta_median": _series_statistic(delta, "median"),
        "delta_q95": _series_quantile(delta, 0.95),
        "delta_q99": _series_quantile(delta, 0.99),
        "abs_delta_mean": _series_statistic(abs_delta, "mean"),
        "abs_delta_median": median_abs_delta,
        "abs_delta_q95": _series_quantile(abs_delta, 0.95),
        "abs_delta_q99": q99_threshold,
        "abs_delta_max": _series_statistic(abs_delta, "max"),
        "mad_abs_delta": mad,
        "spike_q99_threshold": q99_threshold,
        "spike_mad_threshold": mad_threshold,
        "num_spikes_q99": num_spikes_q99,
        "num_spikes_mad": num_spikes_mad,
        "pct_spikes_q99": _percentage(num_spikes_q99, num_deltas),
        "pct_spikes_mad": _percentage(num_spikes_mad, num_deltas),
        "num_valid_relative_deltas": len(relative_delta),
        "relative_delta_median": _series_statistic(relative_delta, "median"),
        "relative_delta_q95": _series_quantile(relative_delta, 0.95),
    }


def find_top_signal_spikes(dataframe: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Return the largest absolute consecutive Signal changes."""

    if top_k < 1:
        raise ValueError("top_k must be positive.")

    columns = [
        "rank",
        "time_prev",
        "time_current",
        "signal_prev",
        "signal_current",
        "delta_signal",
        "abs_delta_signal",
        "dt_seconds",
    ]
    changes = compute_signal_changes(dataframe).dropna(subset=["abs_delta_signal"])
    top = changes.nlargest(top_k, "abs_delta_signal", keep="first").copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    return top[columns].reset_index(drop=True)


def compute_normalization_diagnostics(
    dataframe: pd.DataFrame,
    rolling_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return temporary normalization variants and compact summaries."""

    if rolling_window < 1:
        raise ValueError("rolling_window must be positive.")

    signal = dataframe["Signal"].astype(float)
    median = float(signal.median())
    q25 = float(signal.quantile(0.25))
    q75 = float(signal.quantile(0.75))
    rolling = signal.rolling(rolling_window, min_periods=rolling_window)
    rolling_median = rolling.median()
    rolling_iqr = rolling.quantile(0.75) - rolling.quantile(0.25)
    rolling_centered = signal - rolling_median

    transformed = pd.DataFrame(
        {
            "Time": dataframe["Time"],
            "raw_signal": signal,
            "zscore_per_dataset": _safe_scaled(signal - signal.mean(), signal.std()),
            "robust_zscore_per_dataset": _safe_scaled(signal - median, q75 - q25),
            "minmax_per_dataset": _safe_scaled(signal - signal.min(), signal.max() - signal.min()),
            "rolling_centered_signal": rolling_centered,
            "rolling_robust_zscore": _safe_series_scaled(
                rolling_centered,
                rolling_iqr,
            ),
        }
    )

    rows = []
    for transformation in transformed.columns.drop("Time"):
        values = _finite_values(transformed[transformation])
        rows.append(
            {
                "transformation": transformation,
                "num_valid": len(values),
                "min": _series_statistic(values, "min"),
                "q01": _series_quantile(values, 0.01),
                "q05": _series_quantile(values, 0.05),
                "median": _series_statistic(values, "median"),
                "mean": _series_statistic(values, "mean"),
                "q95": _series_quantile(values, 0.95),
                "q99": _series_quantile(values, 0.99),
                "max": _series_statistic(values, "max"),
                "std": _series_statistic(values, "std"),
            }
        )
    return transformed, pd.DataFrame(rows)


def _finite_values(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def _series_statistic(values: pd.Series, statistic: str) -> float:
    finite = _finite_values(values)
    return float(getattr(finite, statistic)()) if not finite.empty else np.nan


def _series_quantile(values: pd.Series, quantile: float) -> float:
    finite = _finite_values(values)
    return float(finite.quantile(quantile)) if not finite.empty else np.nan


def _series_range(values: pd.Series) -> float:
    finite = _finite_values(values)
    return float(finite.max() - finite.min()) if not finite.empty else np.nan


def _safe_scaled(numerator: pd.Series, scale: float) -> pd.Series:
    if np.isfinite(scale) and abs(scale) > np.finfo(float).eps:
        return numerator / scale
    return pd.Series(0.0, index=numerator.index, dtype=float)


def _safe_series_scaled(numerator: pd.Series, scale: pd.Series) -> pd.Series:
    result = numerator / scale.mask(scale.abs().le(np.finfo(float).eps))
    return result.mask(scale.abs().le(np.finfo(float).eps) & numerator.eq(0), 0.0)


def _percentage(count: int, total: int) -> float:
    return count / total * 100 if total else 0.0
