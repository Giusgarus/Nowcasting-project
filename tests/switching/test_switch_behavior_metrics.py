"""Tests for switch-behavior metric tables."""

import numpy as np
import pandas as pd
import pytest

from src.switching.conversion import enforce_switch_time
from src.switching.metrics import (
    build_switch_behavior_metrics_tables,
    compute_switch_behavior_metrics,
    count_active_islands,
)


def test_switch_behavior_metrics_match_legacy_formulas() -> None:
    model_switch_min_time = np.array([0, 1, 1, 0, 1, 1])
    model_switch_adjusted = np.array([0, 1, 1, 1, 1, 1])
    outage_mask = np.array([0, 1, 0, 1, 0, 0])
    perfect_switch_min_time = np.array([0, 1, 1, 1, 0, 0])

    row = compute_switch_behavior_metrics(
        model_switch_min_time,
        model_switch_adjusted,
        outage_mask,
        perfect_switch_min_time,
        method_id="model",
    )

    assert row["duration_samples"] == 4
    assert row["outage_time_covered_samples"] == 5
    assert row["events"] == 2
    assert row["outage_mask_agreement_pct"] == pytest.approx(2 / 6 * 100.0)
    assert row["duration_similarity_vs_perfect_pct"] == pytest.approx(
        100.0 * (3 - abs(3 - 4)) / 3
    )
    assert row["perfect_min_time_duration_samples"] == 3
    assert row["outage_mask_duration_samples"] == 2
    assert row["model_min_time_duration_samples"] == 4
    assert row["model_adjusted_duration_samples"] == 5


def test_enforce_switch_time_reproduces_adjusted_switch_behavior() -> None:
    original_switch = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    outage_mask = np.array([0, 0, 0, 1, 0, 0, 1, 0])

    adjusted = enforce_switch_time(original_switch, outage_mask, switch_time=2)

    np.testing.assert_array_equal(adjusted, [0, 1, 1, 1, 1, 1, 0, 0])


def test_count_active_islands_captures_event_level_fragmentation() -> None:
    model_raw = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1])

    assert count_active_islands(model_raw) == 3


def test_duration_similarity_is_nan_when_perfect_duration_is_zero() -> None:
    row = compute_switch_behavior_metrics(
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        method_id="model",
    )

    assert np.isnan(row["duration_similarity_vs_perfect_pct"])


def test_switch_behavior_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_switch_behavior_metrics(
            [0, 1],
            [0, 1],
            [0],
            [0, 1],
            method_id="model",
        )


def test_switch_behavior_metrics_accept_all_zero_and_all_one_switches() -> None:
    zero_row = compute_switch_behavior_metrics(
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        method_id="zero_model",
    )
    one_row = compute_switch_behavior_metrics(
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        method_id="one_model",
    )

    assert zero_row["duration_samples"] == 0
    assert zero_row["events"] == 0
    assert zero_row["outage_mask_agreement_pct"] == pytest.approx(100.0)
    assert np.isnan(zero_row["duration_similarity_vs_perfect_pct"])
    assert one_row["duration_samples"] == 3
    assert one_row["events"] == 1
    assert one_row["duration_similarity_vs_perfect_pct"] == pytest.approx(100.0)


def test_perfect_switch_row_can_be_computed() -> None:
    perfect_processed = np.array([0, 1, 1, 1])
    outage_mask = np.array([0, 1, 0, 0])

    row = compute_switch_behavior_metrics(
        perfect_processed,
        perfect_processed,
        outage_mask,
        perfect_processed,
        method_id="perfect_switch",
        metadata={"model_family": "reference"},
    )

    assert row["method_id"] == "perfect_switch"
    assert row["model_family"] == "reference"
    assert row["duration_samples"] == 3
    assert row["outage_time_covered_samples"] == 3


def _comparison_timeseries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["Model"] * 6,
            "event_id": ["e1", "e1", "e1", "e2", "e2", "e2"],
            "dataset_id": ["d1"] * 6,
            "dataset_name": ["d.csv"] * 6,
            "quality_flag": ["usable"] * 6,
            "split": ["test"] * 6,
            "Time": pd.date_range("2026-01-01", periods=6, freq="30s"),
            "Signal_true": [0.0, 11.0, 12.0, 0.0, 11.0, 0.0],
            "outage_mask": [0, 1, 1, 0, 1, 0],
            "perfect_switch_raw": [0, 1, 0, 0, 1, 0],
            "perfect_switch_min_time": [0, 1, 1, 0, 1, 0],
            "perfect_switch_adjusted": [0, 1, 1, 0, 1, 0],
            "perfect_switch": [0, 1, 1, 0, 1, 0],
            "model_switch_raw": [0, 1, 0, 0, 1, 1],
            "model_switch_min_time": [0, 1, 1, 0, 1, 1],
            "model_switch_adjusted": [0, 1, 1, 0, 1, 1],
            "model_switch": [0, 1, 1, 0, 1, 1],
            "threshold": [10.0] * 6,
            "switch_time": [2] * 6,
            "window_id": [f"w{i}" for i in range(6)],
        }
    )


def _perfect_timeseries() -> pd.DataFrame:
    return _comparison_timeseries()[
        [
            "event_id",
            "Time",
            "outage_mask",
            "perfect_switch_raw",
            "perfect_switch_min_time",
            "perfect_switch_adjusted",
            "perfect_switch",
        ]
    ].copy()


def test_global_and_event_switch_behavior_tables_are_built() -> None:
    global_table, event_table = build_switch_behavior_metrics_tables(
        _comparison_timeseries(),
        method_id="gru_s2v_raw_selection",
        method_metadata={
            "method_name": "GRU S2V Raw",
            "model_family": "gru",
            "architecture": "gru_s2v",
            "variant": "raw",
            "mode": "trained",
        },
        selection_id="selection",
        test_type="external_holdout",
        threshold=10.0,
        condition="greater_than",
        switch_time=2,
        perfect_timeseries=_perfect_timeseries(),
    )

    assert global_table["method_id"].tolist() == [
        "gru_s2v_raw_selection",
        "perfect_switch",
    ]
    assert set(event_table["method_id"]) == {
        "gru_s2v_raw_selection",
        "perfect_switch",
    }
    assert event_table.shape[0] == 4
    model_e2 = event_table.loc[
        event_table["method_id"].eq("gru_s2v_raw_selection")
        & event_table["event_id"].eq("e2")
    ].iloc[0]
    assert model_e2["events"] == 1
    assert model_e2["num_samples"] == 3
