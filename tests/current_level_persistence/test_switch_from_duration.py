import numpy as np
import pandas as pd

from src.switching.reference_switch import compute_perfect_switch_from_true_signal
from src.tasks.current_level_persistence.evaluation.switch_from_duration import (
    align_perfect_switch_to_duration_windows,
    build_duration_switch_timeseries,
    derive_switch_from_duration,
)


def _perfect_timeseries() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=6, freq="30s")
    events = pd.DataFrame(
        {
            "event_id": ["event_1"] * 6,
            "dataset_id": ["dataset_1"] * 6,
            "dataset_name": ["toy.csv"] * 6,
            "Time": times,
            "Signal_prepared": [0.0, 11.0, 11.0, 11.0, 11.0, 0.0],
            "quality_flag": ["usable"] * 6,
            "split": ["test"] * 6,
            "event_point_idx": np.arange(6),
        }
    )
    return compute_perfect_switch_from_true_signal(
        events,
        threshold=10.0,
        switch_time=2,
    )


def _metadata() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=6, freq="30s")
    return pd.DataFrame(
        {
            "window_id": [f"w{i}" for i in range(6)],
            "global_window_id": [f"toy.csv::w{i}" for i in range(6)],
            "global_event_id": ["toy.csv::event_1"] * 6,
            "event_id": ["event_1"] * 6,
            "dataset_id": ["dataset_1"] * 6,
            "dataset_name": ["toy.csv"] * 6,
            "quality_flag": ["usable"] * 6,
            "split": ["test"] * 6,
            "timestamp": times,
            "current_signal": [0.0, 11.0, 11.0, 11.0, 11.0, 0.0],
            "remaining_persistence_seconds": [30.0, 300.0, 30.0, 30.0, 30.0, 30.0],
        }
    )


def _predictions() -> pd.DataFrame:
    metadata = _metadata()
    return pd.DataFrame(
        {
            "global_window_id": metadata["global_window_id"],
            "event_id": metadata["event_id"],
            "window_id": metadata["window_id"],
            "y_true_seconds": metadata["remaining_persistence_seconds"],
            "y_pred_seconds": [0.0, 300.0, 0.0, 0.0, 0.0, 0.0],
            "model_id": ["model"] * 6,
            "run_id": ["run"] * 6,
        }
    )


def test_duration_threshold_is_inclusive() -> None:
    switch = derive_switch_from_duration(
        np.array([299.9, 300.0, 301.0]),
        switch_time_seconds=300.0,
    )

    np.testing.assert_array_equal(switch, [0, 1, 1])


def test_perfect_switch_aligns_to_decision_timestamps() -> None:
    aligned = align_perfect_switch_to_duration_windows(
        _perfect_timeseries(),
        _metadata(),
    )

    assert len(aligned) == 6
    np.testing.assert_allclose(aligned["current_signal"], aligned["Signal_true"])
    assert aligned["perfect_switch"].dtype == np.int8


def test_duration_switch_uses_shared_min_island_and_stateful_hold() -> None:
    comparison = build_duration_switch_timeseries(
        _predictions(),
        _metadata(),
        _perfect_timeseries(),
        method_name="Shapelet",
        duration_threshold_seconds=300.0,
        signal_threshold=10.0,
        signal_condition="greater_than",
        switch_time=2,
        apply_min_island_length=True,
    )

    np.testing.assert_array_equal(comparison["model_switch_raw"], [0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(
        comparison["model_switch_min_island_old"],
        [0, 1, 1, 0, 0, 0],
    )
    np.testing.assert_array_equal(
        comparison["model_switch_min_time"],
        [0, 1, 1, 1, 1, 0],
    )
    np.testing.assert_array_equal(
        comparison["model_switch"],
        comparison["model_switch_min_time"],
    )
    assert comparison["duration_threshold_seconds"].eq(300.0).all()


def test_duration_switch_is_gated_by_current_signal_level() -> None:
    predictions = _predictions()
    predictions.loc[0, "y_pred_seconds"] = 900.0

    comparison = build_duration_switch_timeseries(
        predictions,
        _metadata(),
        _perfect_timeseries(),
        method_name="Shapelet",
        duration_threshold_seconds=300.0,
        signal_threshold=10.0,
        signal_condition="greater_than",
        switch_time=2,
        apply_min_island_length=True,
    )

    assert comparison.loc[0, "duration_threshold_switch"] == 1
    assert comparison.loc[0, "current_signal_gate"] == 0
    assert comparison.loc[0, "model_switch_raw"] == 0
