"""Tests for reusable exploratory data-quality helpers."""

import pandas as pd

from src.analysis.data_quality import (
    compute_quantile_run_summary,
    compute_sampling_summary,
    count_valid_windows_by_segments,
    find_continuous_segments,
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
