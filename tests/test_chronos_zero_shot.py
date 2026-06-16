"""Tests for Chronos zero-shot post-processing without model downloads."""

import numpy as np
import pandas as pd
import torch

from src.evaluation.forecast_metrics import compute_trajectory_metrics
from src.models.autoregressive.chronos_adapter import (
    build_chronos_prediction_tables,
    load_chronos_pipeline,
    predict_chronos_batches,
    predictions_to_raw_scale,
    sample_forecast_statistics,
)
from src.switching.comparison import align_forecast_predictions_to_targets


class FakeChronosPipeline:
    """Return fixed sample offsets around the last context value."""

    def predict(self, inputs, *, prediction_length, num_samples):
        last = inputs[:, -1].numpy()
        offsets = np.arange(num_samples, dtype=np.float32) - (num_samples - 1) / 2
        samples = last[:, None, None] + offsets[None, :, None]
        return torch.from_numpy(np.repeat(samples, prediction_length, axis=2))


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_id": "w1",
                "event_id": "e1",
                "dataset_id": "d1",
                "dataset_name": "data.csv",
                "quality_flag": "usable",
                "split": "test",
                "input_start_time": "2026-01-01 00:00:00",
                "input_end_time": "2026-01-01 00:00:30",
                "target_start_time": "2026-01-01 00:01:00",
                "target_end_time": "2026-01-01 00:01:30",
                "scaling_mean": 10.0,
                "scaling_std": 2.0,
            }
        ]
    )


def test_sample_forecasts_are_reduced_to_median_and_quantiles() -> None:
    samples = np.array(
        [
            [
                [1.0, 10.0],
                [3.0, 30.0],
                [2.0, 20.0],
            ]
        ]
    )

    median, quantiles = sample_forecast_statistics(samples, [0.1, 0.5, 0.9])

    np.testing.assert_allclose(median, [[2.0, 20.0]])
    np.testing.assert_allclose(quantiles[0.5], median)
    assert quantiles[0.1].shape == (1, 2)
    assert quantiles[0.9].shape == (1, 2)


def test_batched_fake_chronos_preserves_input_order() -> None:
    contexts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    point, quantiles = predict_chronos_batches(
        FakeChronosPipeline(),
        contexts,
        prediction_length=2,
        num_samples=3,
        batch_size=2,
        quantiles=[0.1, 0.5, 0.9],
    )

    np.testing.assert_allclose(point, [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    np.testing.assert_allclose(quantiles[0.5], point)


def test_context_standard_predictions_are_inverse_transformed() -> None:
    predictions = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    arrays = {
        "scaling_mean": np.array([10.0, 20.0], dtype=np.float32),
        "scaling_std": np.array([2.0, 3.0], dtype=np.float32),
    }

    raw = predictions_to_raw_scale(
        predictions,
        arrays,
        variant="context_standard_optional",
    )

    np.testing.assert_allclose(raw, [[10.0, 12.0], [17.0, 26.0]])


def test_prediction_tables_include_switch_schema_and_wide_quantiles() -> None:
    actual = np.array([[11.0, 12.0]])
    predicted = np.array([[10.5, 12.5]])
    quantiles = {
        0.1: np.array([[9.0, 11.0]]),
        0.5: predicted,
        0.9: np.array([[12.0, 14.0]]),
    }

    long, wide = build_chronos_prediction_tables(
        _metadata(),
        actual,
        predicted,
        quantiles,
        variant="raw",
    )

    assert len(long) == 2
    assert {
        "window_id",
        "event_id",
        "horizon_step",
        "y_true_raw",
        "y_pred_raw",
        "y_pred_q10",
        "y_pred_q50",
        "y_pred_q90",
    }.issubset(long.columns)
    assert long["mode"].unique().tolist() == ["zero_shot"]
    assert {"y_true_raw_1", "y_pred_raw_2", "y_pred_q90_2"}.issubset(wide.columns)

    targets = pd.DataFrame(
        [
            {
                "window_id": "w1",
                "event_id": "e1",
                "target_step": step,
                "target_time": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=30 * step),
                "y_true_raw": actual[0, step - 1],
                "perfect_switch": 0,
            }
            for step in (1, 2)
        ]
    )
    aligned = align_forecast_predictions_to_targets(long, targets)
    assert len(aligned) == 2


def test_raw_scale_metrics_work_for_chronos_predictions() -> None:
    actual = np.array([[1.0, 2.0]])
    predicted = np.array([[2.0, 4.0]])

    metrics = compute_trajectory_metrics(actual, predicted)

    assert metrics["mae"] == 1.5
    assert metrics["rmse"] > metrics["mae"]


def test_mps_loading_falls_back_to_cpu() -> None:
    class FakeLoader:
        calls = []

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.calls.append((model_id, kwargs))
            if kwargs["device_map"] == "mps":
                raise RuntimeError("unsupported")
            return "pipeline"

    pipeline, device = load_chronos_pipeline(
        "amazon/chronos-t5-small",
        device="mps",
        torch_dtype="auto",
        pipeline_class=FakeLoader,
    )

    assert pipeline == "pipeline"
    assert device == "cpu"
    assert [call[1]["device_map"] for call in FakeLoader.calls] == ["mps", "cpu"]
