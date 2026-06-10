"""Tests for shared forecast and switch metrics."""

import pytest

from src.evaluation.forecast_metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from src.evaluation.switch_metrics import compute_switch_metrics


def test_forecast_metrics() -> None:
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 4.0, 2.0]

    assert mean_absolute_error(actual, predicted) == pytest.approx(1.0)
    assert root_mean_squared_error(actual, predicted) == pytest.approx((5 / 3) ** 0.5)


def test_rejects_empty_forecast_metrics() -> None:
    with pytest.raises(ValueError):
        mean_absolute_error([], [])


def test_switch_metrics_and_duration() -> None:
    metrics = compute_switch_metrics(
        reference=[0, 1, 1, 0],
        predicted=[1, 1, 0, 0],
        sample_interval_seconds=30,
    )

    assert metrics.active_duration_seconds == 60
    assert metrics.true_positives == 1
    assert metrics.true_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.intersection_over_union == pytest.approx(1 / 3)


def test_all_zero_switches_use_defined_zero_division() -> None:
    metrics = compute_switch_metrics([0, 0], [0, 0])

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.false_positive_rate == 0.0


def test_rejects_non_binary_values() -> None:
    with pytest.raises(ValueError):
        compute_switch_metrics([0, 1], [0, 2])
