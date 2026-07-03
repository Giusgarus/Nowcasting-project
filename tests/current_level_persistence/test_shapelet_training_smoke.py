from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.experiments.current_level_persistence.train_learnable_shapelets import (
    CurrentLevelPersistenceDataset,
    build_shapelet_config,
    make_loss,
    prediction_frame,
)
from src.tasks.current_level_persistence.models.learnable_shapelets import (
    build_learnable_shapelet_model,
)
from src.tasks.current_level_persistence.utils.paths import make_run_id, model_dir, run_dir
from src.utils.config import load_yaml_config


CONFIGS = [
    Path("configs/current_level_persistence/shapelet_mlp_delta_0p5.yaml"),
    Path("configs/current_level_persistence/shapelet_transformer_delta_0p5.yaml"),
    Path("configs/current_level_persistence/shapelet_convolution_delta_0p5.yaml"),
]


def test_shapelet_top_level_configs_parse_and_build_models() -> None:
    for path in CONFIGS:
        config = load_yaml_config(path)
        shapelet_config = build_shapelet_config(config, context_length=30)
        model = build_learnable_shapelet_model(config["model_id"], shapelet_config)

        assert config["task_name"] == "current_level_persistence"
        assert config["model_family"] == "learnable_shapelets"
        assert config["dataset"]["input_key"] == "X_relative_to_current"
        assert config["dataset"]["raw_input_key"] == "X_raw"
        assert config["features"]["use_scalar_context"] is True
        assert config["dataset"]["target_key"] == "y_log1p_remaining_persistence_seconds"
        assert model(torch.zeros(2, 30), torch.zeros(2, 12)).shape == (2,)


def test_tiny_synthetic_training_step_runs() -> None:
    config = load_yaml_config(CONFIGS[0])
    config["shapelets"]["n_shapelets_per_length"] = 2
    config["model"]["hidden_dim"] = 8
    config["model"]["num_hidden_layers"] = 1
    shapelet_config = build_shapelet_config(config, context_length=30)
    model = build_learnable_shapelet_model(config["model_id"], shapelet_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = make_loss("huber")
    arrays = {
        "X_relative_to_current": np.random.default_rng(42).normal(size=(8, 30)).astype(
            np.float32
        ),
        "scalar_context_features": np.random.default_rng(44)
        .normal(size=(8, 12))
        .astype(np.float32),
        "y_log1p_remaining_persistence_seconds": np.random.default_rng(43)
        .normal(size=8)
        .astype(np.float32),
    }
    dataset = CurrentLevelPersistenceDataset(
        arrays,
        "X_relative_to_current",
        "y_log1p_remaining_persistence_seconds",
        "scalar_context_features",
    )
    inputs, scalar_inputs, targets = next(
        iter(torch.utils.data.DataLoader(dataset, batch_size=4))
    )

    loss = loss_fn(model(inputs, scalar_inputs), targets)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_prediction_frame_schema_and_run_paths_are_task_separated() -> None:
    run_id = make_run_id(
        delta=0.5,
        context_length=30,
        model_id="multiscale_shapelet_mlp",
        selection_id="externalHoldout_test_fc_uplink_fade",
    )
    metadata = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=2, freq="30s"),
            "dataset_name": ["demo.csv", "demo.csv"],
            "event_id": [1, 1],
            "window_id": [10, 11],
            "global_event_id": ["demo_1", "demo_1"],
            "global_window_id": ["demo_10", "demo_11"],
        }
    )

    frame = prediction_frame(
        metadata,
        np.array([0.0, np.log1p(60.0)]),
        np.array([np.log1p(5.0), np.log1p(30.0)]),
        model_id="multiscale_shapelet_mlp",
        run_id=run_id,
        split="test",
    )

    assert {
        "timestamp",
        "dataset_name",
        "event_id",
        "window_id",
        "global_event_id",
        "global_window_id",
        "split",
        "y_true_log1p_seconds",
        "y_pred_log1p_seconds",
        "y_true_seconds",
        "y_pred_seconds",
        "absolute_error_seconds",
        "absolute_error_minutes",
        "model_id",
        "run_id",
    }.issubset(frame.columns)
    assert "currentLevelPersistence" in run_id
    assert "autoregressive" not in str(run_dir(run_id))
    assert "autoregressive" not in str(model_dir(run_id))
