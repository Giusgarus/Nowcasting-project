"""Tests for reusable exploratory data-quality helpers."""

import pandas as pd

from src.analysis.data_quality import (
    compute_hourly_signal_summary,
    compute_normalization_diagnostics,
    compute_quantile_run_summary,
    compute_rolling_baseline_summary,
    compute_sampling_summary,
    count_valid_windows_by_segments,
    find_continuous_segments,
    find_top_signal_spikes,
)


def _dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:30",
                    "2026-01-01 00:01:00",
                    "2026-01-01 00:05:00",
                    "2026-01-01 00:05:30",
                ]
            ),
            "Signal": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )


def test_find_continuous_segments_splits_only_at_large_gap() -> None:
    assert find_continuous_segments(_dataframe(), 90) == [(0, 3), (3, 5)]


def test_sampling_summary_reports_gaps_and_segments() -> None:
    summary = compute_sampling_summary(_dataframe(), 90)

    assert summary["median_dt_seconds"] == 30.0
    assert summary["num_dt_above_90_seconds"] == 1
    assert summary["num_segments"] == 2
    assert summary["longest_segment_samples"] == 3


def test_window_count_does_not_cross_segments() -> None:
    result = count_valid_windows_by_segments(
        _dataframe(),
        context_lengths=[2],
        prediction_lengths=[1],
        gap_threshold_seconds=90,
    )

    assert result["windows_L2_h1"] == 1


def test_quantile_run_summary_is_exploratory_and_complete() -> None:
    summary = compute_quantile_run_summary(_dataframe())

    assert set(summary["condition"]) == {
        "above_q90",
        "above_q95",
        "above_q99",
        "below_q10",
        "below_q05",
        "below_q01",
    }


def test_rolling_summary_reports_insufficient_points() -> None:
    summary = compute_rolling_baseline_summary(_dataframe(), {"too_long": 10})

    assert summary.loc[0, "status"] == "insufficient_points"
    assert summary.loc[0, "num_valid_points"] == 0
    assert pd.isna(summary.loc[0, "rolling_mean_min"])


def test_hourly_summary_groups_by_hour() -> None:
    summary = compute_hourly_signal_summary(_dataframe())

    assert summary["hour"].tolist() == [0]
    assert summary.loc[0, "count"] == 5
    assert summary.loc[0, "mean_signal"] == 2.0


def test_top_spikes_returns_largest_absolute_changes() -> None:
    dataframe = _dataframe().assign(Signal=[0.0, 1.0, 10.0, 11.0, 12.0])

    spikes = find_top_signal_spikes(dataframe, top_k=2)

    assert spikes["abs_delta_signal"].tolist() == [9.0, 1.0]
    assert spikes["rank"].tolist() == [1, 2]


def test_normalization_diagnostics_handles_constant_signal() -> None:
    dataframe = _dataframe().assign(Signal=5.0)

    transformed, summary = compute_normalization_diagnostics(dataframe, rolling_window=3)

    assert transformed["zscore_per_dataset"].eq(0).all()
    assert transformed["robust_zscore_per_dataset"].eq(0).all()
    assert transformed["minmax_per_dataset"].eq(0).all()
    assert transformed["rolling_robust_zscore"].dropna().eq(0).all()
    assert summary["num_valid"].gt(0).all()
