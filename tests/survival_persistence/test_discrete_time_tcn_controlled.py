from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.experiments.survival_persistence.run_discrete_time_tcn_controlled import (
    completed_run_is_valid,
)
from src.tasks.survival_persistence.evaluation.controlled_tcn import (
    assert_validation_only_selection,
    average_survival_probabilities,
    averaged_survival_curve_is_monotone,
    effective_receptive_field_samples,
    median_from_horizon_survival,
    select_best_validation_row,
)


def test_selection_rejects_test_metric_primary() -> None:
    with pytest.raises(ValueError, match="test metric"):
        assert_validation_only_selection(pd.Series({"validation_ibs": 0.1}), metric="test_ibs")


def test_select_best_uses_validation_metric_and_tie_breakers() -> None:
    table = pd.DataFrame(
        {
            "experiment_id": ["larger", "smaller"],
            "status": ["completed", "completed"],
            "validation_integrated_brier_score": [0.10, 0.10],
            "validation_discrete_nll": [1.0, 1.0],
            "parameter_count": [200, 100],
        }
    )

    best = select_best_validation_row(
        table,
        primary_metric="validation_integrated_brier_score",
        mode="min",
    )

    assert best["experiment_id"] == "smaller"


def test_effective_receptive_field_counts_two_convs_per_block() -> None:
    assert effective_receptive_field_samples(kernel_size=3, dilations=[1, 2, 4]) == 29
    assert effective_receptive_field_samples(kernel_size=3, dilations=[1, 2, 4, 8]) == 61


def test_ensemble_average_requires_aligned_rows_and_stays_monotone() -> None:
    first = pd.DataFrame(
        {
            "global_window_id": ["w1", "w2"],
            "sample_time": pd.date_range("2026-01-01", periods=2, freq="30s"),
            "survival_probability_60s": [0.8, 0.7],
            "survival_probability_300s": [0.6, 0.4],
            "survival_probability_900s": [0.3, 0.2],
        }
    )
    second = first.copy()
    second["survival_probability_60s"] = [0.9, 0.8]

    averaged = average_survival_probabilities([first, second])

    assert averaged_survival_curve_is_monotone(averaged)
    np.testing.assert_allclose(averaged["survival_probability_60s"], [0.85, 0.75])


def test_ensemble_average_rejects_unaligned_rows() -> None:
    first = pd.DataFrame(
        {
            "global_window_id": ["w1"],
            "sample_time": [pd.Timestamp("2026-01-01")],
            "survival_probability_60s": [0.8],
        }
    )
    second = first.copy()
    second["global_window_id"] = ["w2"]

    with pytest.raises(ValueError, match="row-aligned"):
        average_survival_probabilities([first, second])


def test_ensemble_median_uses_first_horizon_below_half() -> None:
    frame = pd.DataFrame(
        {
            "survival_probability_60s": [0.8, 0.7],
            "survival_probability_300s": [0.4, 0.6],
            "survival_probability_900s": [0.2, 0.3],
        }
    )

    median = median_from_horizon_survival(frame, max_horizon_seconds=900)

    assert median.tolist() == [300.0, 900.0]


def test_completed_run_reuse_requires_matching_fingerprint(tmp_path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "metadata.yaml").write_text(
        "status: completed\ncontrolled_experiment_fingerprint: abc\n",
        encoding="utf-8",
    )
    (run_dir / "metrics" / "validation_metrics.yaml").write_text(
        "validation_integrated_brier_score: 0.1\n",
        encoding="utf-8",
    )

    assert completed_run_is_valid(run_dir, "abc")
    assert not completed_run_is_valid(run_dir, "other")
