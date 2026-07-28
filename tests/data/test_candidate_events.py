"""Tests for candidate-event detection and chronological event splitting."""

import pandas as pd

from src.data.events import detect_candidate_fade_events
from src.data.splits import assign_event_splits


def _clean_signal(times: list[str], signals: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": "dataset_001",
            "dataset_name": "example.csv",
            "row_id": range(len(times)),
            "Time": pd.to_datetime(times),
            "Signal": signals,
            "dt_seconds": pd.Series(pd.to_datetime(times)).diff().dt.total_seconds(),
            "segment_id": "dataset_001_segment_0000",
        }
    )


def test_event_grouping_is_anchored_to_first_crossing() -> None:
    clean = _clean_signal(
        [
            "2026-01-01 00:00:00",
            "2026-01-01 02:00:00",
            "2026-01-01 03:01:00",
            "2026-01-01 05:00:00",
        ],
        [11.0, 12.0, 13.0, 14.0],
    )

    events = detect_candidate_fade_events(
        clean,
        signal_threshold=10.0,
        condition="greater_than",
        grouping_hours=3.0,
        window_pre_hours=1.5,
        window_post_hours=1.5,
    )

    assert len(events) == 2
    assert events["num_crossings"].tolist() == [2, 2]
    assert events["event_timestamp"].tolist() == pd.to_datetime(
        ["2026-01-01 00:00:00", "2026-01-01 03:01:00"]
    ).tolist()


def test_single_crossing_creates_centered_event_window() -> None:
    clean = _clean_signal(["2026-01-01 12:00:00"], [10.1])

    events = detect_candidate_fade_events(
        clean,
        signal_threshold=10.0,
        condition="greater_than",
        grouping_hours=3.0,
        window_pre_hours=1.5,
        window_post_hours=1.5,
    )

    assert len(events) == 1
    assert events.loc[0, "window_start"] == pd.Timestamp("2026-01-01 10:30:00")
    assert events.loc[0, "window_end"] == pd.Timestamp("2026-01-01 13:30:00")


def test_inclusive_threshold_condition_counts_equal_threshold() -> None:
    clean = _clean_signal(["2026-01-01 12:00:00"], [10.0])

    events = detect_candidate_fade_events(
        clean,
        signal_threshold=10.0,
        condition="greater_than_or_equal",
        grouping_hours=3.0,
        window_pre_hours=1.5,
        window_post_hours=1.5,
    )

    assert len(events) == 1
    assert events.loc[0, "first_crossing_signal"] == 10.0


def test_event_split_is_chronological_per_dataset() -> None:
    events = pd.DataFrame(
        {
            "event_id": [f"event_{index}" for index in range(10)],
            "dataset_id": "dataset_001",
            "event_timestamp": pd.date_range("2026-01-01", periods=10, freq="1D"),
        }
    )

    assigned = assign_event_splits(events, train_ratio=0.7, validation_ratio=0.15)

    assert assigned["split"].tolist() == [
        *(["train"] * 7),
        "validation",
        *(["test"] * 2),
    ]
