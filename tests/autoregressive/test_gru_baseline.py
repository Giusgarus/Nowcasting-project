"""Tests for deterministic GRU autoregressive forecasting baselines."""

import numpy as np
import pytest
import torch

from scripts.experiments.autoregressive.run_autoregressive_gru import (
    ensure_dataset_folder_complete,
)
from src.tasks.autoregressive.evaluation.forecast_metrics import (
    compute_horizon_metrics,
    compute_trajectory_metrics,
    inverse_context_standardization,
)
from src.tasks.autoregressive.models.gru import (
    GRUSeq2SeqForecaster,
    GRUSequenceToVectorForecaster,
)


def _s2v() -> GRUSequenceToVectorForecaster:
    return GRUSequenceToVectorForecaster(
        input_size=1,
        hidden_size=8,
        num_layers=1,
        prediction_length=10,
    )


def _seq2seq() -> GRUSeq2SeqForecaster:
    return GRUSeq2SeqForecaster(
        input_size=1,
        hidden_size=8,
        num_layers=1,
        prediction_length=10,
    )


def test_gru_sequence_to_vector_forward_shape() -> None:
    model = _s2v()
    output = model(torch.randn(4, 30, 1))

    assert output.shape == (4, 10)


def test_gru_seq2seq_forward_without_teacher_forcing() -> None:
    model = _seq2seq()
    output = model(torch.randn(4, 30, 1), teacher_forcing_ratio=0.0)

    assert output.shape == (4, 10)


def test_gru_seq2seq_forward_with_teacher_forcing() -> None:
    model = _seq2seq()
    model.train()
    output = model(
        torch.randn(4, 30, 1),
        y=torch.randn(4, 10),
        teacher_forcing_ratio=1.0,
    )

    assert output.shape == (4, 10)


def test_gru_seq2seq_evaluation_ignores_teacher_forcing_targets() -> None:
    model = _seq2seq()
    model.eval()
    X = torch.randn(3, 30, 1)
    first = model(X, y=torch.zeros(3, 10), teacher_forcing_ratio=1.0)
    second = model(X, y=torch.ones(3, 10) * 100, teacher_forcing_ratio=1.0)

    torch.testing.assert_close(first, second)


def test_context_standard_inverse_transform() -> None:
    predictions = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    means = np.array([10.0, 20.0], dtype=np.float32)
    stds = np.array([2.0, 3.0], dtype=np.float32)

    raw = inverse_context_standardization(predictions, means, stds)

    np.testing.assert_allclose(raw, [[10.0, 12.0], [17.0, 26.0]])


def test_dataset_folder_validation_reports_missing_metadata(tmp_path) -> None:
    (tmp_path / "train.npz").write_bytes(b"placeholder")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        ensure_dataset_folder_complete(tmp_path)


@pytest.mark.parametrize("model_factory", [_s2v, _seq2seq])
def test_raw_scale_metrics_work_for_both_architectures(model_factory) -> None:
    model = model_factory()
    model.eval()
    with torch.no_grad():
        prediction = model(torch.randn(5, 30, 1)).numpy()
    actual = np.zeros((5, 10), dtype=np.float32)

    metrics = compute_trajectory_metrics(actual, prediction)
    horizon_metrics = compute_horizon_metrics(actual, prediction)

    assert set(metrics) == {"mae", "rmse"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert horizon_metrics.shape == (10, 3)
