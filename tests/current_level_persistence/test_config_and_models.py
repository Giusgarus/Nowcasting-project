from pathlib import Path

import torch

from src.tasks.current_level_persistence.models.learnable_shapelets import (
    SUPPORTED_MODEL_IDS,
    ShapeletConfig,
    build_learnable_shapelet_model,
)
from src.utils.config import load_yaml_config


CONFIG_DIR = Path("configs/current_level_persistence")


def test_current_level_persistence_configs_can_be_parsed() -> None:
    config_paths = sorted(CONFIG_DIR.glob("*.yaml"))

    assert config_paths
    for path in config_paths:
        config = load_yaml_config(path)
        assert config["task_name"] == "current_level_persistence"


def test_three_shapelet_model_ids_are_supported() -> None:
    assert set(SUPPORTED_MODEL_IDS) == {
        "multiscale_shapelet_mlp",
        "multiscale_shapelet_transformer",
        "multiscale_shapelet_convolution",
    }


def test_shapelet_models_return_one_scalar_per_window() -> None:
    config = ShapeletConfig(
        context_length=30,
        shapelet_lengths=(5, 10, 15),
        n_shapelets_per_length=2,
        hidden_size=16,
        dropout=0.0,
    )
    inputs = torch.zeros(4, 30)

    for model_id in SUPPORTED_MODEL_IDS:
        model = build_learnable_shapelet_model(model_id, config)
        output = model(inputs)
        assert output.shape == (4,)
