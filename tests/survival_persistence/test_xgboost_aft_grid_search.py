"""Tests for survival-persistence XGBoost-AFT grid search."""

from __future__ import annotations

from pathlib import Path

from scripts.experiments.survival_persistence.run_xgboost_aft_grid_search import (
    build_trial_config,
    validate_config,
    xgboost_params_for_device,
)
from src.tuning.grid_search import expand_parameter_grid
from src.utils.config import load_yaml_config
from src.utils.device import available_accelerator_devices


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/survival_persistence/models/xgboost_aft_grid_search.yaml"
)


def test_survival_xgboost_aft_grid_is_extended_but_bounded() -> None:
    config = load_yaml_config(CONFIG_PATH)

    count = validate_config(config)

    assert count == 1920
    assert count == len(expand_parameter_grid(config["parameter_grid"]))
    assert count <= config["search"]["max_trials"]


def test_survival_xgboost_aft_grid_trial_config_applies_overrides() -> None:
    config = load_yaml_config(CONFIG_PATH)
    parameters = {
        "dataset.feature_set": "relative_current_plus_scalar",
        "dataset.sample_weighting": "event_balanced",
        "model.aft_loss_distribution": "logistic",
        "model.aft_loss_distribution_scale": 2.0,
        "model.max_depth": 6,
        "model.learning_rate": 0.1,
        "model.min_child_weight": 1.0,
        "model.subsample": 0.8,
        "model.colsample_bytree": 1.0,
        "model.reg_lambda": 5.0,
        "model.reg_alpha": 0.0,
        "model.gamma": 0.0,
    }

    trial_config = build_trial_config(config, parameters)

    assert trial_config["dataset"]["feature_set"] == "relative_current_plus_scalar"
    assert trial_config["dataset"]["sample_weighting"] == "event_balanced"
    assert trial_config["model"]["aft_loss_distribution"] == "logistic"
    assert trial_config["model"]["max_depth"] == 6


def test_xgboost_device_params_prefer_cuda_and_fallback_for_mps() -> None:
    assert xgboost_params_for_device({}, "cuda:2")["device"] == "cuda:2"
    assert xgboost_params_for_device({}, "cpu")["device"] == "cpu"
    mps_params = xgboost_params_for_device({}, "mps")
    assert mps_params["device"] == "cpu"
    assert mps_params["device_fallback_reason"] == "xgboost_does_not_support_mps"


def test_available_accelerator_devices_always_has_fallback() -> None:
    devices = available_accelerator_devices()

    assert devices
    assert all(isinstance(device, str) for device in devices)
