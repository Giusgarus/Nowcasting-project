"""Tests for the stateful switch hold post-processing rule."""

import numpy as np
import pandas as pd
import pytest

from src.switching.comparison import build_model_switch_timeseries
from src.switching.conversion import (
    ensure_min_island_length,
    hold_while_signal_above_threshold,
)
from src.switching.metrics import compute_switch_behavior_metrics
from src.switching.reference_switch import compute_perfect_switch_from_true_signal


def test_hold_rule_keeps_switch_active_while_outage_continues() -> None:
    switch_after_min_island = np.array([0, 1, 1, 0, 0, 0, 0])
    outage_mask = np.array([0, 1, 1, 1, 1, 0, 0])

    result = hold_while_signal_above_threshold(
        switch_after_min_island, outage_mask
    )

    np.testing.assert_array_equal(result, [0, 1, 1, 1, 1, 0, 0])


def test_hold_rule_does_not_create_switch_on_from_outage_alone() -> None:
    result = hold_while_signal_above_threshold(
        [0, 0, 0, 0],
        [0, 1, 1, 0],
    )

    np.testing.assert_array_equal(result, [0, 0, 0, 0])


def test_hold_rule_allows_switch_off_when_outage_ends() -> None:
    result = hold_while_signal_above_threshold(
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1],
    )

    np.testing.assert_array_equal(result, [0, 1, 1, 0, 0])


def test_hold_rule_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="must match"):
        hold_while_signal_above_threshold([0, 1], [0, 1, 1])


def test_hold_rule_preserves_first_element() -> None:
    result = hold_while_signal_above_threshold([1, 0, 0], [1, 1, 0])

    np.testing.assert_array_equal(result, [1, 1, 0])


def test_pipeline_exposes_raw_old_min_island_and_new_min_time() -> None:
    raw = np.array([0, 1, 0, 0, 0])
    old_min_island = ensure_min_island_length(raw, 2)
    new_min_time = hold_while_signal_above_threshold(
        old_min_island,
        np.array([0, 1, 1, 1, 0]),
    )

    np.testing.assert_array_equal(old_min_island, [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(new_min_time, [0, 1, 1, 1, 0])


def test_metrics_use_new_min_time_not_old_min_island() -> None:
    old_min_island = np.array([0, 1, 1, 0, 0])
    new_min_time = np.array([0, 1, 1, 1, 0])
    adjusted = np.array([0, 1, 1, 1, 0])
    outage_mask = np.array([0, 1, 1, 1, 0])
    perfect_min_time = np.array([0, 1, 1, 1, 0])

    row = compute_switch_behavior_metrics(
        new_min_time,
        adjusted,
        outage_mask,
        perfect_min_time,
        method_id="model",
    )

    assert int(old_min_island.sum()) == 2
    assert row["duration_samples"] == 3
    assert row["events"] == 1
    assert row["outage_mask_agreement_pct"] == pytest.approx(100.0)


def test_perfect_switch_columns_are_saved_consistently() -> None:
    frame = pd.DataFrame(
        {
            "event_id": "event_a",
            "Time": pd.date_range("2026-01-01", periods=6, freq="30s"),
            "Signal_prepared": [0.0, 11.0, 11.0, 11.0, 11.0, 0.0],
        }
    )

    result = compute_perfect_switch_from_true_signal(
        frame,
        threshold=10.0,
        switch_time=2,
    )

    for column in (
        "outage_mask",
        "perfect_switch_raw",
        "perfect_switch_min_island_old",
        "perfect_switch_min_time",
        "perfect_switch_adjusted",
    ):
        assert column in result.columns
    np.testing.assert_array_equal(result["perfect_switch"], result["perfect_switch_min_time"])


def test_autoregressive_comparison_outputs_all_switch_versions() -> None:
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
                "window_id": "w2",
                "event_id": "e1",
                "dataset_id": "d1",
                "dataset_name": "d.csv",
                "quality_flag": "usable",
                "horizon_step": 1,
                "y_true_raw": 12.0,
                "y_pred_raw": 8.0,
            },
            {
                "window_id": "w3",
                "event_id": "e1",
                "dataset_id": "d1",
                "dataset_name": "d.csv",
                "quality_flag": "usable",
                "horizon_step": 1,
                "y_true_raw": 0.0,
                "y_pred_raw": 8.0,
            },
        ]
    )
    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    targets = pd.DataFrame(
        [
            {
                "window_id": f"w{idx + 1}",
                "event_id": "e1",
                "target_step": 1,
                "target_time": times[idx],
                "y_true_raw": y_true,
                "perfect_switch": perfect,
            }
            for idx, (y_true, perfect) in enumerate([(11.0, 1), (12.0, 1), (0.0, 0)])
        ]
    )

    result = build_model_switch_timeseries(
        predictions,
        targets,
        method_name="Model",
        threshold=10.0,
        condition="greater_than",
        switch_time=1,
        apply_min_island_length=True,
    )

    for column in (
        "outage_mask",
        "model_switch_raw",
        "model_switch_min_island_old",
        "model_switch_min_time",
        "model_switch_adjusted",
        "perfect_switch_raw",
        "perfect_switch_min_island_old",
        "perfect_switch_min_time",
        "perfect_switch_adjusted",
    ):
        assert column in result.columns
    np.testing.assert_array_equal(result["model_switch_raw"], [1, 0, 0])
    np.testing.assert_array_equal(result["model_switch_min_island_old"], [1, 0, 0])
    np.testing.assert_array_equal(result["model_switch_min_time"], [1, 1, 0])
