"""Tests for cross-task switch comparison helpers."""

from pathlib import Path

import pandas as pd

from src.switching.cross_task import (
    SwitchComparisonSource,
    build_reference_grid_metrics,
    build_strict_intersection_metrics,
    load_reference_grid,
    parse_switch_eval_task,
)


def _source(
    predictions_path: Path,
    *,
    task_name: str,
    method_name: str,
    comparison_id: str,
) -> SwitchComparisonSource:
    return SwitchComparisonSource(
        task_name=task_name,
        selection_id="external_holdout",
        comparison_id=comparison_id,
        method_id=method_name.lower().replace(" ", "_"),
        results_path=predictions_path.parent.parent,
        predictions_path=predictions_path,
        method_name=method_name,
    )


def test_parse_switch_eval_task_from_results_path() -> None:
    task_name, selection_id = parse_switch_eval_task(
        "results/comparisons/switch_eval/survival_persistence/"
        "externalHoldout_test_fc_uplink_fade/switch_eval_model"
    )

    assert task_name == "survival_persistence"
    assert selection_id == "externalHoldout_test_fc_uplink_fade"


def test_load_reference_grid_accepts_signal_column(tmp_path: Path) -> None:
    path = tmp_path / "perfect.parquet"
    pd.DataFrame(
        {
            "Time": pd.date_range("2026-01-01", periods=2, freq="30s"),
            "Signal": [11.0, 9.0],
            "perfect_switch": [1, 0],
        }
    ).to_parquet(path)

    loaded = load_reference_grid(path)

    assert "Signal_true" in loaded.columns
    assert loaded["Signal_true"].tolist() == [11.0, 9.0]


def test_reference_grid_metrics_penalize_missing_native_decisions(
    tmp_path: Path,
) -> None:
    times = pd.date_range("2026-01-01", periods=4, freq="30s")
    reference_grid = pd.DataFrame(
        {
            "Time": times,
            "Signal_true": [11.0, 12.0, 9.0, 13.0],
            "perfect_switch": [1, 1, 0, 1],
        }
    )
    predictions_path = tmp_path / "run" / "predictions" / "model_vs_perfect_switch.parquet"
    predictions_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "Time": [times[1], times[2]],
            "model_switch_raw": [1, 1],
            "model_switch": [1, 1],
        }
    ).to_parquet(predictions_path)
    source = _source(
        predictions_path,
        task_name="survival_persistence",
        method_name="Survival",
        comparison_id="survival_switch",
    )

    metrics, coverage = build_reference_grid_metrics(
        reference_grid=reference_grid,
        sources=[source],
        missing_model_switch=0,
    )

    row = metrics.iloc[0]
    assert row["denominator"] == "reference_grid_missing_as_zero"
    assert row["num_points"] == 4
    assert row["num_aligned_native_points"] == 2
    assert row["coverage_pct"] == 50.0
    assert row["true_positives"] == 1
    assert row["false_positives"] == 1
    assert row["false_negatives"] == 2
    assert coverage.iloc[0]["perfect_positive_points_without_native_decision"] == 2


def test_strict_intersection_metrics_use_only_common_timestamps(
    tmp_path: Path,
) -> None:
    times = pd.date_range("2026-01-01", periods=4, freq="30s")
    reference_grid = pd.DataFrame(
        {
            "Time": times,
            "Signal_true": [11.0, 12.0, 9.0, 13.0],
            "perfect_switch": [1, 1, 0, 1],
        }
    )
    first_path = tmp_path / "first" / "predictions" / "model_vs_perfect_switch.parquet"
    second_path = tmp_path / "second" / "predictions" / "model_vs_perfect_switch.parquet"
    first_path.parent.mkdir(parents=True)
    second_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "Time": times,
            "model_switch_raw": [1, 1, 0, 1],
            "model_switch": [1, 0, 0, 1],
        }
    ).to_parquet(first_path)
    pd.DataFrame(
        {
            "Time": [times[1], times[2]],
            "model_switch_raw": [1, 1],
            "model_switch": [1, 1],
        }
    ).to_parquet(second_path)
    sources = [
        _source(
            first_path,
            task_name="autoregressive",
            method_name="Autoregressive",
            comparison_id="ar_switch",
        ),
        _source(
            second_path,
            task_name="survival_persistence",
            method_name="Survival",
            comparison_id="survival_switch",
        ),
    ]

    metrics = build_strict_intersection_metrics(
        reference_grid=reference_grid,
        sources=sources,
    ).set_index("task_name")

    assert metrics.loc["autoregressive", "num_points"] == 2
    assert metrics.loc["autoregressive", "true_negatives"] == 1
    assert metrics.loc["autoregressive", "false_negatives"] == 1
    assert metrics.loc["survival_persistence", "true_positives"] == 1
    assert metrics.loc["survival_persistence", "false_positives"] == 1
