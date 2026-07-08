"""Tests for the autoregressive XGBoost baseline utilities."""

from pathlib import Path

import numpy as np

from scripts.experiments.autoregressive.run_xgboost_grid_search import validate_config
from src.tasks.autoregressive.models.xgboost import flatten_forecast_context
from src.tuning.grid_search import expand_parameter_grid
from src.utils.config import load_yaml_config

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "configs").is_dir()
)


def test_autoregressive_xgboost_flattens_univariate_context_windows() -> None:
    context = np.arange(24, dtype=np.float32).reshape(2, 12, 1)

    flattened = flatten_forecast_context(context)

    assert flattened.shape == (2, 12)
    np.testing.assert_array_equal(flattened[0], np.arange(12, dtype=np.float32))


def test_autoregressive_xgboost_grid_uses_range_specs_and_safety_limit() -> None:
    config = load_yaml_config(
        PROJECT_ROOT / "configs/autoregressive/xgboost_grid_search.yaml"
    )
    candidates = expand_parameter_grid(config["parameter_grid"])

    assert len(candidates) == 1152
    assert config["search"]["max_trials_per_variant"] == 1152
    assert config["search"]["variants"] == ["raw", "context_standard"]
    validate_config(config, num_candidates=len(candidates))


def test_autoregressive_switch_comparison_includes_xgboost_methods() -> None:
    config = load_yaml_config(PROJECT_ROOT / "configs/switch_comparison.yaml")
    xgboost_methods = [
        (method["architecture"], method["variant"])
        for method in config["methods"]
        if method["model_family"] == "xgboost"
    ]

    assert sorted(xgboost_methods) == [
        ("xgboost", "context_standard"),
        ("xgboost", "raw"),
    ]
