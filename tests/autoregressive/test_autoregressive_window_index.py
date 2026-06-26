"""Tests for event-window autoregressive index construction and summaries."""

import pandas as pd

from src.tasks.autoregressive.data.event_preparation import summarize_threshold_balance
from src.tasks.autoregressive.data.windowed_forecasting import build_autoregressive_window_index


def test_window_index_stays_inside_event_and_segment() -> None:
    times = pd.to_datetime(
        [
            "2026-01-01 00:00:00",
            "2026-01-01 00:00:30",
            "2026-01-01 00:01:00",
            "2026-01-01 00:05:00",
            "2026-01-01 00:05:30",
            "2026-01-01 00:06:00",
        ]
    )
    windows = pd.DataFrame(
        {
            "event_id": "event_1",
            "dataset_id": "dataset_001",
            "dataset_name": "example.csv",
            "segment_id": ["segment_1"] * 3 + ["segment_2"] * 3,
            "Time": times,
            "Signal_prepared": [1.0, 2.0, 3.0, 11.0, 12.0, 13.0],
            "is_imputed": False,
            "event_point_idx": range(6),
        }
    )
    quality = pd.DataFrame({"event_id": ["event_1"], "quality_flag": ["usable"]})
    events = pd.DataFrame({"event_id": ["event_1"], "split": ["train"]})

    index = build_autoregressive_window_index(
        windows,
        quality,
        events,
        context_length=2,
        prediction_length=1,
        gap_threshold_seconds=90,
        allow_warning_events=False,
        signal_threshold=10.0,
    )

    assert len(index) == 2
    assert index["segment_id"].nunique() == 2
    assert index["split"].eq("train").all()
    assert (index["input_end_idx"] - index["input_start_idx"] + 1).eq(2).all()
    assert (index["target_end_idx"] - index["target_start_idx"] + 1).eq(1).all()


def test_threshold_balance_counts_above_and_below_points() -> None:
    windows = pd.DataFrame(
        {
            "dataset_name": ["a.csv"] * 4,
            "Signal_prepared": [9.0, 10.0, 10.1, 12.0],
        }
    )

    summary = summarize_threshold_balance(
        windows,
        signal_threshold=10.0,
        dataset_names=["a.csv", "no_events.csv"],
    )
    dataset_row = summary.loc[summary["scope"].eq("dataset")].iloc[0]
    empty_row = summary.loc[summary["dataset_name"].eq("no_events.csv")].iloc[0]
    global_row = summary.loc[summary["scope"].eq("global")].iloc[0]

    assert dataset_row["points_above_threshold"] == 2
    assert dataset_row["points_below_or_equal_threshold"] == 2
    assert empty_row["total_points"] == 0
    assert global_row["total_points"] == 4
