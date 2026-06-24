"""Tests for the deterministic PatchTST autoregressive baseline."""

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.experiments.autoregressive.run_autoregressive_patchtst import (
    build_single_run_model_config,
    validate_config as validate_single_run_config,
)
from scripts.experiments.autoregressive.run_patchtst_grid_search import (
    build_prediction_table,
    load_trial_checkpoint,
    save_trial_checkpoint,
    trial_display_parameters,
    validate_config as validate_grid_config,
)
from src.evaluation.forecast_metrics import (
    compute_trajectory_metrics,
    inverse_context_standardization,
)
from src.tasks.autoregressive.models.patchtst import (
    PatchTSTForecaster,
    compute_num_patches,
)
from src.tasks.autoregressive.models.patchtst_training import PatchTSTTrainingResult


def _model(**overrides) -> PatchTSTForecaster:
    parameters = {
        "context_length": 30,
        "prediction_length": 10,
        "input_channels": 1,
        "patch_len": 5,
        "stride": 5,
        "d_model": 16,
        "n_heads": 2,
        "num_layers": 1,
        "dropout": 0.0,
    }
    parameters.update(overrides)
    return PatchTSTForecaster(**parameters)


def test_patchtst_forward_shape_and_patching() -> None:
    model = _model()
    X = torch.randn(4, 30, 1)

    output = model(X)
    patches = model.extract_patches(X)

    assert output.shape == (4, 10)
    assert patches.shape == (4, 1, 6, 5)
    assert model.num_patches == 6


def test_patch_count_supports_overlapping_patches() -> None:
    assert compute_num_patches(30, 5, 5) == 6
    assert compute_num_patches(30, 5, 2) == 13


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"patch_len": 31}, "patch_len"),
        ({"stride": 0}, "stride"),
        ({"d_model": 15, "n_heads": 2}, "divisible"),
    ],
)
def test_patchtst_rejects_invalid_configuration(parameters, message) -> None:
    with pytest.raises(ValueError, match=message):
        _model(**parameters)


def test_context_standard_predictions_return_to_raw_scale_and_score() -> None:
    predictions = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    raw = inverse_context_standardization(
        predictions,
        np.array([10.0, 20.0], dtype=np.float32),
        np.array([2.0, 3.0], dtype=np.float32),
    )
    metrics = compute_trajectory_metrics(raw, raw)

    np.testing.assert_allclose(raw, [[10.0, 12.0], [17.0, 26.0]])
    assert metrics == {"mae": 0.0, "rmse": 0.0}


def test_patchtst_prediction_table_matches_switch_compatible_schema() -> None:
    metadata = pd.DataFrame(
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
    table = build_prediction_table(
        metadata,
        np.array([[11.0, 12.0]]),
        np.array([[10.5, 12.5]]),
        variant="context_standard",
        y_pred_variant=np.array([[0.25, 1.25]]),
    )

    assert len(table) == 2
    assert {
        "window_id",
        "event_id",
        "dataset_id",
        "dataset_name",
        "quality_flag",
        "horizon_step",
        "y_true_raw",
        "y_pred_raw",
        "absolute_error",
    }.issubset(table.columns)
    assert table["model_family"].unique().tolist() == ["patchtst"]
    assert table["architecture"].unique().tolist() == ["patchtst"]
    assert table["y_pred_normalized"].tolist() == [0.25, 1.25]


def test_patchtst_grid_safety_limit_blocks_without_skipping() -> None:
    config = {
        "search": {
            "selection_metric": "validation_rmse_raw",
            "selection_mode": "min",
            "max_trials": 2,
        },
        "training": {"loss": "mse"},
        "fixed_model": {"output_length": 10},
        "data": {"prediction_length": 10},
    }

    with pytest.raises(ValueError, match="no combinations were skipped"):
        validate_grid_config(config, num_candidates=3)


def test_patchtst_single_run_config_rejects_grid_sections() -> None:
    config = {
        "experiment": {"variants": ["raw"]},
        "training": {"loss": "mse"},
        "evaluation": {"evaluate_in_raw_scale": True},
        "parameter_grid": {"d_model": [32, 64]},
    }

    with pytest.raises(ValueError, match="grid-search sections"):
        validate_single_run_config(config)


def test_patchtst_single_run_model_config_contains_one_parameter_set() -> None:
    config = {
        "data": {"context_length": 30, "prediction_length": 10},
        "model": {
            "input_channels": 1,
            "patch_len": 5,
            "stride": 5,
            "d_model": 64,
            "n_heads": 4,
            "num_layers": 2,
            "dropout": 0.1,
        },
    }

    model_config = build_single_run_model_config(config, "raw")

    assert model_config["architecture"] == "patchtst"
    assert model_config["variant"] == "raw"
    assert model_config["d_model"] == 64
    assert "parameter_grid" not in model_config


def test_patchtst_trial_display_contains_model_optimizer_and_training() -> None:
    config = {
        "data": {"context_length": 30, "prediction_length": 10},
        "fixed_model": {"input_channels": 1},
        "training": {"batch_size": 64, "seed": 42},
    }
    parameters = {
        "patch_len": 5,
        "stride": 2,
        "d_model": 32,
        "n_heads": 2,
        "num_layers": 1,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }

    display = trial_display_parameters(config, parameters, "raw")

    assert display["model"]["patch_len"] == 5
    assert display["optimizer"]["weight_decay"] == 0.0001
    assert display["training"]["batch_size"] == 64


def test_patchtst_trial_checkpoint_round_trip(tmp_path) -> None:
    model = _model()
    state = model.state_dict()
    result = PatchTSTTrainingResult(
        best_epoch=1,
        best_val_loss=0.5,
        history=[{"epoch": 1, "train_loss": 1.0, "val_loss": 0.5}],
        best_state_dict=state,
        last_state_dict=state,
    )
    path = tmp_path / "trial.pt"
    save_trial_checkpoint(
        path,
        trial_id="trial_001",
        parameters={"patch_len": 5},
        model_config={"patch_len": 5},
        result=result,
        validation_metrics={"mae": 1.0, "rmse": 2.0},
        trial_fingerprint="fingerprint",
    )

    loaded, metrics, model_config, parameters, fingerprint = load_trial_checkpoint(path)

    assert loaded.best_epoch == 1
    assert metrics == {"mae": 1.0, "rmse": 2.0}
    assert model_config == {"patch_len": 5}
    assert parameters == {"patch_len": 5}
    assert fingerprint == "fingerprint"
