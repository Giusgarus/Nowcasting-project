"""Tests for event-window extraction, imputation, and quality flags."""

import pandas as pd
import pytest

from src.data.events import build_event_windows, prepare_and_assess_event_windows


def _event(window_end: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["dataset_001_event_00001"],
            "dataset_id": ["dataset_001"],
            "dataset_name": ["example.csv"],
            "segment_id": ["dataset_001_segment_0000"],
            "event_timestamp": [pd.Timestamp("2026-01-01 00:01:00")],
            "window_start": [pd.Timestamp("2026-01-01 00:00:00")],
            "window_end": [pd.Timestamp(window_end)],
            "signal_threshold": [10.0],
            "grouping_hours": [3.0],
            "window_pre_hours": [0.0],
            "window_post_hours": [0.0],
            "num_crossings": [1],
            "first_crossing_signal": [11.0],
            "max_signal_in_event_group": [11.0],
            "actual_first_crossing_time": [pd.Timestamp("2026-01-01 00:01:00")],
            "actual_last_crossing_time": [pd.Timestamp("2026-01-01 00:01:00")],
        }
    )


def _clean(times: list[str], signals: list[float]) -> pd.DataFrame:
    parsed = pd.to_datetime(times)
    return pd.DataFrame(
        {
            "dataset_id": "dataset_001",
            "dataset_name": "example.csv",
            "row_id": range(len(times)),
            "Time": parsed,
            "Signal": signals,
            "dt_seconds": pd.Series(parsed).diff().dt.total_seconds(),
            "segment_id": "dataset_001_segment_0000",
        }
    )


@pytest.mark.parametrize(
    ("times", "window_end", "max_imputation_gap", "expected_flag", "imputed_points"),
    [
        (
            [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:30",
                "2026-01-01 00:01:30",
                "2026-01-01 00:02:00",
            ],
            "2026-01-01 00:02:00",
            90,
            "usable",
            1,
        ),
        (
            [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:30",
                "2026-01-01 00:02:30",
                "2026-01-01 00:03:00",
            ],
            "2026-01-01 00:03:00",
            180,
            "warning",
            3,
        ),
        (
            [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:30",
                "2026-01-01 00:04:00",
            ],
            "2026-01-01 00:04:00",
            300,
            "unusable",
            0,
        ),
    ],
)
def test_imputation_and_quality_policy(
    times: list[str],
    window_end: str,
    max_imputation_gap: int,
    expected_flag: str,
    imputed_points: int,
) -> None:
    events = _event(window_end)
    observed = build_event_windows(_clean(times, [1.0] * len(times)), events)

    prepared, quality = prepare_and_assess_event_windows(
        observed,
        events,
        expected_seconds=30,
        signal_threshold=10.0,
        min_coverage_ratio=0.5,
        max_gap_seconds_for_imputation=max_imputation_gap,
        max_small_gap_samples=2,
        max_warning_gap_samples=5,
        context_length=1,
        prediction_length=1,
    )

    assert quality.loc[0, "quality_flag"] == expected_flag
    assert int(prepared["is_imputed"].sum()) == imputed_points


def test_empty_event_collection_preserves_quality_schema() -> None:
    events = _event("2026-01-01 00:02:00").iloc[0:0]
    observed = build_event_windows(_clean(["2026-01-01 00:00:00"], [1.0]), events)

    prepared, quality = prepare_and_assess_event_windows(
        observed,
        events,
        expected_seconds=30,
        signal_threshold=10.0,
        min_coverage_ratio=0.8,
        max_gap_seconds_for_imputation=90,
        max_small_gap_samples=2,
        max_warning_gap_samples=5,
        context_length=120,
        prediction_length=30,
    )

    assert prepared.empty
    assert quality.empty
    assert {"event_id", "quality_flag", "quality_reason"}.issubset(quality.columns)
