import numpy as np
import pandas as pd

from src.switching.reference_switch import compute_perfect_switch_from_true_signal
from src.tasks.survival_persistence.evaluation.switch_from_survival import (
    align_perfect_switch_to_survival_windows,
    build_survival_switch_timeseries,
    derive_switch_from_survival_probability,
    survival_probability_column,
)


def _perfect_timeseries() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=6, freq="30s")
    events = pd.DataFrame(
        {
            "event_id": ["event_1"] * 6,
            "global_event_id": ["toy.csv::event_1"] * 6,
            "dataset_id": ["dataset_1"] * 6,
            "dataset_name": ["toy.csv"] * 6,
            "Time": times,
            "Signal": [0.0, 11.0, 11.0, 11.0, 11.0, 0.0],
            "quality_flag": ["observed"] * 6,
            "split": ["test"] * 6,
            "event_point_idx": np.arange(6),
        }
    )
    return compute_perfect_switch_from_true_signal(
        events,
        signal_column="Signal",
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
            "split": ["test"] * 6,
            "sample_time": times,
            "current_signal": [0.0, 11.0, 11.0, 11.0, 11.0, 0.0],
            "y_time_seconds": [30.0, 300.0, 270.0, 240.0, 210.0, 180.0],
            "y_event_observed": [1] * 6,
        }
    )


def _predictions() -> pd.DataFrame:
    metadata = _metadata()
    return pd.DataFrame(
        {
            "global_event_id": metadata["global_event_id"],
            "global_window_id": metadata["global_window_id"],
            "y_time_seconds": metadata["y_time_seconds"],
            "y_event_observed": metadata["y_event_observed"],
            "model_id": ["xgboost_aft"] * 6,
            "survival_probability_300s": [0.9, 0.5, 0.1, 0.1, 0.1, 0.9],
        }
    )


def test_survival_probability_column_uses_saved_schema() -> None:
    assert survival_probability_column(300) == "survival_probability_300s"


def test_survival_probability_threshold_is_inclusive() -> None:
    switch = derive_switch_from_survival_probability(
        np.array([0.49, 0.5, 0.51]),
        probability_threshold=0.5,
    )

    np.testing.assert_array_equal(switch, [0, 1, 1])


def test_perfect_switch_aligns_to_survival_decision_timestamps() -> None:
    aligned = align_perfect_switch_to_survival_windows(
        _perfect_timeseries(),
        _metadata(),
    )

    assert len(aligned) == 6
    np.testing.assert_allclose(aligned["current_signal"], aligned["Signal_true"])
    assert aligned["perfect_switch"].dtype == np.int8


def test_survival_switch_uses_shared_min_island_and_stateful_hold() -> None:
    comparison = build_survival_switch_timeseries(
        _predictions(),
        _metadata(),
        _perfect_timeseries(),
        method_name="XGBoost AFT",
        run_id="run",
        survival_horizon_seconds=300.0,
        probability_threshold=0.5,
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
    assert comparison["survival_horizon_seconds"].eq(300.0).all()


def test_survival_switch_is_gated_by_current_signal_level() -> None:
    comparison = build_survival_switch_timeseries(
        _predictions(),
        _metadata(),
        _perfect_timeseries(),
        method_name="XGBoost AFT",
        run_id="run",
        survival_horizon_seconds=300.0,
        probability_threshold=0.5,
        signal_threshold=10.0,
        signal_condition="greater_than",
        switch_time=2,
        apply_min_island_length=True,
    )

    assert comparison.loc[0, "survival_threshold_switch"] == 1
    assert comparison.loc[0, "current_signal_gate"] == 0
    assert comparison.loc[0, "model_switch_raw"] == 0
