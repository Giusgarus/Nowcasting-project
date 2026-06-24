import numpy as np
import pytest

from src.tasks.current_level_persistence.evaluation.metrics import (
    compute_duration_metrics,
    seconds_from_log1p,
)


def test_log_predictions_are_inverted_to_seconds() -> None:
    values = np.log1p(np.array([0.0, 30.0, 90.0]))

    seconds = seconds_from_log1p(values)

    np.testing.assert_allclose(seconds, [0.0, 30.0, 90.0])


def test_negative_inverse_predictions_are_clipped_to_zero() -> None:
    seconds = seconds_from_log1p(np.array([-1.0]))

    np.testing.assert_allclose(seconds, [0.0])


def test_duration_metrics_are_reported_in_seconds_and_minutes() -> None:
    y_true = np.log1p(np.array([30.0, 90.0]))
    y_pred = np.log1p(np.array([60.0, 60.0]))

    metrics = compute_duration_metrics(y_true, y_pred)

    assert metrics["mae_seconds"] == pytest.approx(30.0)
    assert metrics["rmse_seconds"] == pytest.approx(30.0)
    assert metrics["median_ae_seconds"] == pytest.approx(30.0)
    assert metrics["mae_minutes"] == pytest.approx(0.5)
    assert metrics["rmse_minutes"] == pytest.approx(0.5)
    assert metrics["median_ae_minutes"] == pytest.approx(0.5)
    assert metrics["mae_log1p_seconds"] > 0.0
