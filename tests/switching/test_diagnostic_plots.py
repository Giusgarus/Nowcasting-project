"""Tests for switch diagnostic plotting helpers."""

from pathlib import Path

import pandas as pd

from src.switching.cross_task import SwitchComparisonSource
from src.switching.diagnostic_plots import (
    compute_reference_event_f1_matrix,
    plot_metric_bars,
    plot_switch_timeline_rows,
)


def _source(predictions_path: Path) -> SwitchComparisonSource:
    return SwitchComparisonSource(
        task_name="autoregressive",
        selection_id="selection",
        comparison_id="comparison",
        method_id="method",
        results_path=predictions_path.parent.parent,
        predictions_path=predictions_path,
        method_name="Method",
    )


def test_compute_reference_event_f1_matrix_uses_common_grid(tmp_path: Path) -> None:
    times = pd.date_range("2026-01-01", periods=4, freq="30s")
    reference = pd.DataFrame(
        {
            "Time": times,
            "event_id": ["e1", "e1", "e2", "e2"],
            "Signal_true": [11.0, 12.0, 9.0, 13.0],
            "perfect_switch": [1, 1, 0, 1],
        }
    )
    predictions_path = tmp_path / "run" / "predictions" / "model_vs_perfect_switch.parquet"
    predictions_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "Time": times,
            "model_switch_raw": [1, 1, 0, 0],
            "model_switch": [1, 1, 0, 0],
        }
    ).to_parquet(predictions_path)

    matrix = compute_reference_event_f1_matrix(
        reference,
        [_source(predictions_path)],
        event_ids=["e1", "e2"],
        missing_model_switch=0,
    )

    assert matrix.iloc[0]["e1"] == 1.0
    assert matrix.iloc[0]["e2"] == 0.0


def test_plot_helpers_write_png_files(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        {
            "task_name": ["autoregressive"],
            "method_name": ["Method"],
            "f1": [0.8],
            "precision": [1.0],
            "recall": [0.67],
            "balanced_accuracy": [0.83],
        }
    )
    metric_path = tmp_path / "metrics.png"
    plot_metric_bars(metrics, metric_path, title="Metrics")

    timeline = pd.DataFrame(
        {
            "Time": pd.date_range("2026-01-01", periods=4, freq="30s"),
            "Signal_true": [1.0, 12.0, 13.0, 1.0],
        }
    )
    timeline_path = tmp_path / "timeline.png"
    plot_switch_timeline_rows(
        timeline,
        {"Perfect Switch": [0, 1, 1, 0], "Method": [0, 1, 0, 0]},
        timeline_path,
        threshold=10.0,
        title="Timeline",
    )

    assert metric_path.exists()
    assert timeline_path.exists()
