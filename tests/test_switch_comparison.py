"""Tests for model-derived switch alignment and Perfect Switch comparison."""

import pandas as pd

from src.switching.comparison import (
    align_forecast_predictions_to_targets,
    build_model_switch_timeseries,
    compute_model_vs_perfect_metrics,
    select_horizon_threshold_count_decisions,
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


def _window_metadata() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=2, freq="30s")
    return pd.DataFrame(
        [
            {
                "window_id": "w1",
                "event_id": "e1",
                "dataset_id": "d1",
                "dataset_name": "d.csv",
                "quality_flag": "usable",
                "split": "test",
                "input_end_time": times[0],
            },
            {
                "window_id": "w2",
                "event_id": "e1",
                "dataset_id": "d1",
                "dataset_name": "d.csv",
                "quality_flag": "usable",
                "split": "test",
                "input_end_time": times[1],
            },
        ]
    )


def _perfect_timeseries() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=2, freq="30s")
    return pd.DataFrame(
        [
            {
                "event_id": "e1",
                "Time": times[0],
                "Signal_true": 10.5,
                "perfect_switch": 1,
            },
            {
                "event_id": "e1",
                "Time": times[1],
                "Signal_true": 9.0,
                "perfect_switch": 0,
            },
        ]
    )


def test_latest_available_forecast_selects_smallest_horizon() -> None:
    aligned = align_forecast_predictions_to_targets(_predictions(), _targets())

    selected = select_latest_available_forecasts(aligned)

    assert selected["horizon_step"].tolist() == [1, 1]
    assert selected["window_id"].tolist() == ["w1", "w2"]


def test_horizon_threshold_count_uses_full_forecast_at_input_end_time() -> None:
    predictions = pd.concat(
        [
            _predictions(),
            pd.DataFrame(
                [
                    {
                        "window_id": "w2",
                        "event_id": "e1",
                        "dataset_id": "d1",
                        "dataset_name": "d.csv",
                        "quality_flag": "usable",
                        "horizon_step": 2,
                        "y_true_raw": 7.0,
                        "y_pred_raw": 8.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    targets = pd.concat(
        [
            _targets(),
            pd.DataFrame(
                [
                    {
                        "window_id": "w2",
                        "event_id": "e1",
                        "target_step": 2,
                        "target_time": pd.Timestamp("2026-01-01 00:01:00"),
                        "y_true_raw": 7.0,
                        "perfect_switch": 0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    aligned = align_forecast_predictions_to_targets(predictions, targets)

    selected = select_horizon_threshold_count_decisions(
        aligned,
        _window_metadata(),
        _perfect_timeseries(),
        threshold=10.0,
        condition="greater_than",
        required_points_above_threshold=2,
    )

    assert selected["Time"].tolist() == _window_metadata()["input_end_time"].tolist()
    assert selected["num_horizon_points_above_threshold"].tolist() == [1, 1]
    assert selected["model_switch_raw"].tolist() == [0, 0]


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


def test_build_model_switch_timeseries_supports_horizon_count_rule() -> None:
    predictions = pd.DataFrame(
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
                "y_true_raw": 12.0,
                "y_pred_raw": 11.0,
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
            {
                "window_id": "w2",
                "event_id": "e1",
                "dataset_id": "d1",
                "dataset_name": "d.csv",
                "quality_flag": "usable",
                "horizon_step": 2,
                "y_true_raw": 7.0,
                "y_pred_raw": 8.0,
            },
        ]
    )
    targets = pd.DataFrame(
        [
            {
                "window_id": "w1",
                "event_id": "e1",
                "target_step": 1,
                "target_time": pd.Timestamp("2026-01-01 00:00:30"),
                "y_true_raw": 11.0,
                "perfect_switch": 1,
            },
            {
                "window_id": "w1",
                "event_id": "e1",
                "target_step": 2,
                "target_time": pd.Timestamp("2026-01-01 00:01:00"),
                "y_true_raw": 12.0,
                "perfect_switch": 1,
            },
            {
                "window_id": "w2",
                "event_id": "e1",
                "target_step": 1,
                "target_time": pd.Timestamp("2026-01-01 00:01:00"),
                "y_true_raw": 9.0,
                "perfect_switch": 0,
            },
            {
                "window_id": "w2",
                "event_id": "e1",
                "target_step": 2,
                "target_time": pd.Timestamp("2026-01-01 00:01:30"),
                "y_true_raw": 7.0,
                "perfect_switch": 0,
            },
        ]
    )

    result = build_model_switch_timeseries(
        predictions,
        targets,
        method_name="Model",
        threshold=10.0,
        condition="greater_than",
        switch_time=2,
        apply_min_island_length=False,
        prediction_aggregation="horizon_threshold_count",
        required_points_above_threshold=2,
        window_metadata=_window_metadata(),
        perfect_timeseries=_perfect_timeseries(),
    )

    assert result["decision_rule"].tolist() == [
        "horizon_threshold_count",
        "horizon_threshold_count",
    ]
    assert result["num_horizon_points_above_threshold"].tolist() == [2, 1]
    assert result["model_switch_raw"].tolist() == [1, 0]
    assert result["model_switch"].tolist() == [1, 0]
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
