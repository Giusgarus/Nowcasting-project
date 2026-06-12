"""Tests for model-derived switch alignment and Perfect Switch comparison."""

import pandas as pd

from src.switching.comparison import (
    align_forecast_predictions_to_targets,
    build_model_switch_timeseries,
    compute_model_vs_perfect_metrics,
    select_latest_available_forecasts,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_id": "w1",
                "event_id": "e1",
                "dataset_id": "d1",
                "dataset_name": "d.csv",
                "quality_flag": "usable",
                "horizon_step": 1,
                "y_true_raw": 11.0,
                "y_pred_raw": 12.0,
            },
            {
                "window_id": "w1",
                "event_id": "e1",
                "dataset_id": "d1",
                "dataset_name": "d.csv",
                "quality_flag": "usable",
                "horizon_step": 2,
                "y_true_raw": 9.0,
                "y_pred_raw": 8.0,
            },
            {
                "window_id": "w2",
                "event_id": "e1",
                "dataset_id": "d1",
                "dataset_name": "d.csv",
                "quality_flag": "usable",
                "horizon_step": 1,
                "y_true_raw": 9.0,
                "y_pred_raw": 11.0,
            },
        ]
    )


def _targets() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=2, freq="30s")
    return pd.DataFrame(
        [
            {
                "window_id": "w1",
                "event_id": "e1",
                "target_step": 1,
                "target_time": times[0],
                "y_true_raw": 11.0,
                "perfect_switch": 1,
            },
            {
                "window_id": "w1",
                "event_id": "e1",
                "target_step": 2,
                "target_time": times[1],
                "y_true_raw": 9.0,
                "perfect_switch": 0,
            },
            {
                "window_id": "w2",
                "event_id": "e1",
                "target_step": 1,
                "target_time": times[1],
                "y_true_raw": 9.0,
                "perfect_switch": 0,
            },
        ]
    )


def test_latest_available_forecast_selects_smallest_horizon() -> None:
    aligned = align_forecast_predictions_to_targets(_predictions(), _targets())

    selected = select_latest_available_forecasts(aligned)

    assert selected["horizon_step"].tolist() == [1, 1]
    assert selected["window_id"].tolist() == ["w1", "w2"]


def test_build_model_switch_timeseries_applies_shared_conversion() -> None:
    result = build_model_switch_timeseries(
        _predictions(),
        _targets(),
        method_name="Model",
        threshold=10.0,
        condition="greater_than",
        switch_time=2,
        apply_min_island_length=True,
    )

    assert result["Signal_predicted"].tolist() == [12.0, 11.0]
    assert result["model_switch_raw"].tolist() == [1, 1]
    assert result["model_switch"].tolist() == [1, 1]
    assert result["perfect_switch"].tolist() == [1, 0]


def test_model_vs_perfect_metrics_are_computed_on_covered_points() -> None:
    result = build_model_switch_timeseries(
        _predictions(),
        _targets(),
        method_name="Model",
        threshold=10.0,
        condition="greater_than",
        switch_time=2,
        apply_min_island_length=True,
    )

    metrics = compute_model_vs_perfect_metrics(result, by_event=False).iloc[0]

    assert metrics["num_points"] == 2
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
