"""Tests for model-agnostic Perfect Switch construction and summaries."""

import numpy as np
import pandas as pd

from src.switching.conversion import (
    compute_model_switch_from_predictions,
    compute_switch_from_signal_values,
    detect_persistent_threshold_switch,
    ensure_min_island_length,
)
from src.switching.metrics import (
    compute_event_level_switch_summary,
    compute_raw_vs_postprocessed_switch_summary,
    find_positive_islands,
)
from src.switching.reference_switch import (
    align_perfect_switch_to_forecast_windows,
    compute_perfect_switch_from_true_signal,
)


def test_ensure_min_island_length_examples_and_boundaries() -> None:
    examples = [
        ([0, 1, 0, 0, 0], [0, 1, 1, 1, 0]),
        ([0, 1, 1, 0, 0], [0, 1, 1, 1, 0]),
        ([0, 1, 1, 1, 0], [0, 1, 1, 1, 0]),
        ([0, 0, 0, 1, 0], [0, 0, 0, 1, 1]),
        ([0, 0, 0, 0], [0, 0, 0, 0]),
        ([1, 1, 1, 1], [1, 1, 1, 1]),
    ]
    for original, expected in examples:
        np.testing.assert_array_equal(
            ensure_min_island_length(original, 3),
            expected,
        )


def test_ensure_min_island_length_handles_multiple_original_islands() -> None:
    adjusted = ensure_min_island_length([0, 1, 0, 1, 0, 0], 3)

    np.testing.assert_array_equal(adjusted, [0, 1, 1, 1, 1, 1])
    assert adjusted[0] == 0


def test_legacy_persistent_threshold_requires_switch_time_plus_one_points() -> None:
    signal = [0.0, 11.0, 11.0, 11.0, 0.0, 12.0, 12.0]

    switch = detect_persistent_threshold_switch(
        signal,
        threshold=10.0,
        switch_time=2,
    )

    np.testing.assert_array_equal(switch, [0, 1, 1, 1, 0, 0, 0])


def test_perfect_switch_matches_legacy_persistent_threshold_rule() -> None:
    frame = pd.DataFrame(
        {
            "event_id": "event_a",
            "Time": pd.date_range("2026-01-01", periods=7, freq="30s"),
            "Signal_prepared": [10.0, 10.1, 10.2, 10.3, 9.0, 11.0, 11.1],
        }
    )

    processed = compute_perfect_switch_from_true_signal(
        frame, threshold=10.0, switch_time=2
    )
    raw_only = compute_perfect_switch_from_true_signal(
        frame,
        threshold=10.0,
        switch_time=2,
        apply_min_island_length=False,
    )

    np.testing.assert_array_equal(
        processed["perfect_switch_raw"], [0, 1, 1, 1, 0, 0, 0]
    )
    np.testing.assert_array_equal(processed["perfect_switch"], [0, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(
        raw_only["perfect_switch"], raw_only["perfect_switch_raw"]
    )


def test_perfect_switch_postprocessing_does_not_cross_event_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["a", "a", "b", "b"],
            "Time": pd.date_range("2026-01-01", periods=4, freq="30s"),
            "Signal_prepared": [0.0, 11.0, 0.0, 0.0],
        }
    )

    result = compute_perfect_switch_from_true_signal(
        frame, threshold=10.0, switch_time=3
    )

    np.testing.assert_array_equal(result["perfect_switch"], [0, 0, 0, 0])


def test_generic_signal_to_switch_rule_is_model_agnostic() -> None:
    raw, processed = compute_switch_from_signal_values(
        [10.0, 10.1, 9.0, 12.0],
        threshold=10.0,
        switch_time=2,
    )
    model_raw, model_processed = compute_model_switch_from_predictions(
        [10.0, 10.1, 9.0, 12.0],
        threshold=10.0,
        switch_time=2,
    )

    np.testing.assert_array_equal(raw, [0, 1, 0, 1])
    np.testing.assert_array_equal(processed, [0, 1, 1, 1])
    np.testing.assert_array_equal(model_raw, raw)
    np.testing.assert_array_equal(model_processed, processed)


def test_window_target_alignment_uses_correct_times_and_switch_values() -> None:
    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    timeseries = pd.DataFrame(
        {
            "event_id": "event_a",
            "event_point_idx": range(5),
            "Time": times,
            "Signal_true": [1.0, 2.0, 11.0, 12.0, 3.0],
            "perfect_switch_raw": [0, 0, 1, 1, 0],
            "perfect_switch": [0, 0, 1, 1, 0],
            "threshold": 10.0,
            "switch_time": 2,
        }
    )
    metadata = pd.DataFrame(
        [
            {
                "window_id": "window_1",
                "event_id": "event_a",
                "dataset_id": "dataset_a",
                "dataset_name": "a.csv",
                "split": "test",
                "quality_flag": "usable",
                "target_start_idx": 2,
                "target_end_idx": 4,
                "prediction_length": 3,
                "target_start_time": times[2],
                "target_end_time": times[4],
            }
        ]
    )

    aligned = align_perfect_switch_to_forecast_windows(timeseries, metadata)

    assert aligned["target_step"].tolist() == [1, 2, 3]
    assert aligned["target_time"].tolist() == list(times[2:5])
    assert aligned["y_true_raw"].tolist() == [11.0, 12.0, 3.0]
    assert aligned["perfect_switch"].tolist() == [1, 1, 0]


def test_summary_counts_islands_positives_and_changed_points() -> None:
    summary = compute_raw_vs_postprocessed_switch_summary(
        [0, 1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1, 1],
    )

    assert find_positive_islands([0, 1, 0, 0, 1, 1]) == [(1, 1), (4, 5)]
    assert summary["num_positive_raw"] == 3
    assert summary["num_positive_after_min_island"] == 4
    assert summary["num_points_changed_by_min_island"] == 1
    assert summary["num_islands_raw"] == 2
    assert summary["num_islands_after_min_island"] == 2


def test_event_summary_reports_expected_counts() -> None:
    timeseries = pd.DataFrame(
        {
            "event_id": "event_a",
            "dataset_id": "dataset_a",
            "dataset_name": "a.csv",
            "split": "test",
            "quality_flag": "usable",
            "Time": pd.date_range("2026-01-01", periods=4, freq="30s"),
            "Signal_true": [1.0, 11.0, 2.0, 3.0],
            "perfect_switch_raw": [0, 1, 0, 0],
            "perfect_switch": [0, 1, 1, 0],
        }
    )

    summary = compute_event_level_switch_summary(timeseries).iloc[0]

    assert summary["num_points"] == 4
    assert summary["num_positive_raw"] == 1
    assert summary["num_positive_after_min_island"] == 2
    assert summary["num_points_changed_by_min_island"] == 1
