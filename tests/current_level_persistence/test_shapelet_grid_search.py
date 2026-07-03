import copy
from pathlib import Path

import pytest

from scripts.experiments.current_level_persistence.run_shapelet_grid_search import (
    build_trial_config,
    expand_model_trials,
    validate_config,
)
from src.tasks.current_level_persistence.utils.paths import grid_search_dir
from src.utils.config import load_yaml_config


CONFIG_PATH = Path(
    "configs/current_level_persistence/shapelet_grid_search_delta_0p5.yaml"
)


def test_shapelet_grid_config_expands_three_small_model_grids() -> None:
    config = load_yaml_config(CONFIG_PATH)

    validate_config(config)
    counts = {
        model_spec["model_id"]: len(expand_model_trials(config, model_spec))
        for model_spec in config["models"]
    }

    assert counts == {
        "multiscale_shapelet_mlp": 216,
        "multiscale_shapelet_transformer": 216,
        "multiscale_shapelet_convolution": 216,
    }
    assert sum(counts.values()) == 648

    model_specs = {spec["model_id"]: spec for spec in config["models"]}
    assert model_specs["multiscale_shapelet_mlp"]["parameter_grid"][
        "model.hidden_dim"
    ] == [64, 128, 256]
    assert model_specs["multiscale_shapelet_transformer"]["parameter_grid"][
        "model.d_model"
    ] == [32, 64, 128]
    assert model_specs["multiscale_shapelet_convolution"]["parameter_grid"][
        "model.conv_channels"
    ] == [32, 64, 128]
    for model_spec in model_specs.values():
        assert model_spec["parameter_grid"]["training.learning_rate"] == [
            0.0001,
            0.0003,
            0.001,
        ]
        assert model_spec["parameter_grid"][
            "shapelets.n_shapelets_per_length"
        ] == [16, 32, 64]
        assert model_spec["parameter_grid"]["scalar_encoder.hidden_dim"] == [16, 32]


def test_shapelet_grid_trial_config_applies_fixed_and_grid_overrides() -> None:
    config = load_yaml_config(CONFIG_PATH)
    convolution_spec = next(
        spec
        for spec in config["models"]
        if spec["model_id"] == "multiscale_shapelet_convolution"
    )

    trial_config = build_trial_config(
        config,
        convolution_spec,
        {
            "training.learning_rate": 0.0003,
            "model.conv_channels": 32,
            "model.num_conv_layers": 2,
            "model.dropout": 0.0,
            "shapelets.n_shapelets_per_length": 16,
            "scalar_encoder.hidden_dim": 32,
        },
    )

    assert trial_config["model_id"] == "multiscale_shapelet_convolution"
    assert trial_config["training"]["mixed_precision"] is False
    assert trial_config["features"]["use_scalar_context"] is True
    assert trial_config["scalar_encoder"]["hidden_dim"] == 32
    assert trial_config["training"]["learning_rate"] == 0.0003
    assert trial_config["model"]["conv_channels"] == 32
    assert trial_config["model"]["num_conv_layers"] == 2
    assert trial_config["shapelets"]["n_shapelets_per_length"] == 16


def test_shapelet_grid_disables_mixed_precision_for_all_models() -> None:
    config = load_yaml_config(CONFIG_PATH)

    for model_spec in config["models"]:
        for trial in expand_model_trials(config, model_spec):
            assert trial["config"]["training"]["mixed_precision"] is False


def test_shapelet_grid_rejects_test_metric_for_selection() -> None:
    config = load_yaml_config(CONFIG_PATH)
    invalid = copy.deepcopy(config)
    invalid["search"]["selection_metric"] = "test_mae_seconds"

    with pytest.raises(ValueError, match="validation metric"):
        validate_config(invalid)


def test_shapelet_grid_requires_trial_checkpoints_for_final_best_run() -> None:
    config = load_yaml_config(CONFIG_PATH)
    invalid = copy.deepcopy(config)
    invalid["search"]["save_trial_checkpoints"] = False

    with pytest.raises(ValueError, match="save_trial_checkpoints"):
        validate_config(invalid)


def test_shapelet_grid_path_is_task_scoped() -> None:
    path = grid_search_dir(
        selection_id="externalHoldout_test_fc_uplink_fade",
        search_id="shapelet_small_delta0p5_L30_externalHoldout_test_fc_uplink_fade",
    )

    assert "results/grid_searches/current_level_persistence" in str(path)
    assert "autoregressive" not in str(path)
