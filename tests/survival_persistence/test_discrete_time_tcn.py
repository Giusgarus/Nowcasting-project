from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.tasks.survival_persistence.models.discrete_time_tcn import (
    CausalConv1d,
    DiscreteTimeTCNConfig,
    DiscreteTimeTCNSurvivalModel,
    discrete_time_survival_nll,
)
from src.tasks.survival_persistence.training.discrete_time import (
    make_discrete_time_predictions_frame,
)


def test_discrete_time_nll_uses_only_at_risk_bins() -> None:
    logits = torch.zeros((2, 3), dtype=torch.float32)
    event_mask = torch.tensor(
        [[False, True, False], [False, False, False]],
        dtype=torch.bool,
    )
    at_risk_mask = torch.tensor(
        [[True, True, False], [True, False, False]],
        dtype=torch.bool,
    )

    loss = discrete_time_survival_nll(
        logits,
        event_mask=event_mask,
        at_risk_mask=at_risk_mask,
    )

    assert torch.isclose(loss, torch.tensor(1.5 * math.log(2.0), dtype=torch.float32))


def test_causal_conv_does_not_use_future_positions() -> None:
    conv = CausalConv1d(1, 1, kernel_size=3, dilation=1)
    with torch.no_grad():
        conv.conv.weight.fill_(1.0)
        conv.conv.bias.zero_()
    x_base = torch.zeros((1, 1, 5), dtype=torch.float32)
    x_future_changed = x_base.clone()
    x_future_changed[:, :, 4] = 10.0

    y_base = conv(x_base)
    y_changed = conv(x_future_changed)

    torch.testing.assert_close(y_base[:, :, :4], y_changed[:, :, :4])
    assert y_changed[:, :, 4].item() != y_base[:, :, 4].item()


def test_tcn_outputs_one_hazard_logit_per_bin_with_scalar_context() -> None:
    model = DiscreteTimeTCNSurvivalModel(
        DiscreteTimeTCNConfig(
            context_length=6,
            num_bins=4,
            input_channels=1,
            hidden_channels=(4, 4),
            dilations=(1, 2),
            normalization="none",
            dropout=0.0,
            use_scalar_context=True,
            scalar_context_dim=3,
            scalar_hidden_dim=5,
        )
    )

    logits = model(torch.randn(2, 1, 6), torch.randn(2, 3))

    assert logits.shape == (2, 4)


def test_tcn_requires_scalar_context_when_enabled() -> None:
    model = DiscreteTimeTCNSurvivalModel(
        DiscreteTimeTCNConfig(
            context_length=4,
            num_bins=2,
            hidden_channels=(4,),
            dilations=(1,),
            normalization="none",
            use_scalar_context=True,
            scalar_context_dim=2,
        )
    )

    with pytest.raises(ValueError, match="Scalar context"):
        model(torch.randn(1, 1, 4))


def test_prediction_frame_contains_survival_schema() -> None:
    metadata = pd.DataFrame(
        {
            "sample_time": pd.date_range("2026-01-01", periods=2, freq="30s"),
            "dataset_name": ["a.csv", "a.csv"],
            "dataset_id": ["dataset_001", "dataset_001"],
            "global_event_id": ["a::e1", "a::e1"],
            "global_window_id": ["a::w1", "a::w2"],
            "current_above_threshold": [True, True],
            "event_balanced_weight": [0.5, 0.5],
            "elapsed_since_event_start_seconds": [0.0, 30.0],
            "elapsed_event_samples": [1, 2],
            "current_signal": [12.0, 13.0],
        }
    )
    arrays = {
        "y_time_seconds": np.asarray([60.0, 90.0], dtype=np.float32),
        "y_event_observed": np.asarray([1, 0], dtype=np.int64),
        "y_lower_bound_seconds": np.asarray([60.0, 90.0], dtype=np.float32),
        "y_upper_bound_seconds": np.asarray([60.0, np.inf], dtype=np.float32),
    }
    frame = make_discrete_time_predictions_frame(
        split="test",
        metadata=metadata,
        arrays=arrays,
        prediction_arrays={
            "predicted_median_remaining_seconds": np.asarray([60.0, 120.0], dtype=np.float32),
            "survival_at_horizons": np.asarray([[0.8, 0.4], [0.9, 0.7]], dtype=np.float32),
        },
        horizons_seconds=np.asarray([30.0, 300.0]),
        model_family="tcn",
        model_id="discrete_time_tcn",
        feature_set="relative_threshold_plus_scalar",
        sample_weighting="uniform",
    )

    assert "survival_probability_30s" in frame.columns
    assert "survival_probability_300s" in frame.columns
    assert frame["model_id"].eq("discrete_time_tcn").all()
