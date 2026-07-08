"""Tests for the current-level persistence XGBoost baseline utilities."""

from pathlib import Path

import numpy as np

from scripts.experiments.current_level_persistence.run_xgboost_grid_search import (
    validate_config,
)
from src.tasks.current_level_persistence.models.xgboost import build_tabular_features
from src.tuning.grid_search import expand_parameter_grid
from src.utils.config import load_yaml_config

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "configs").is_dir()
)


def test_current_level_xgboost_concatenates_relative_and_scalar_features() -> None:
    arrays = {
        "X_relative_to_current": np.array(
            [[[0.1], [0.0]], [[-0.5], [0.0]]],
            dtype=np.float32,
        ),
        "scalar_context_features": np.array(
            [[10.0, 9.5, 0.2], [12.0, 11.5, -0.1]],
            dtype=np.float32,
        ),
    }

    features = build_tabular_features(
        arrays,
        input_key="X_relative_to_current",
        scalar_context_key="scalar_context_features",
    )

    np.testing.assert_allclose(
        features,
        [
            [0.1, 0.0, 10.0, 9.5, 0.2],
            [-0.5, 0.0, 12.0, 11.5, -0.1],
        ],
    )


def test_current_level_xgboost_grid_uses_range_specs_and_safety_limit() -> None:
    config = load_yaml_config(
        PROJECT_ROOT
        / "configs/current_level_persistence/xgboost_grid_search_delta_0p5.yaml"
    )
    candidates = expand_parameter_grid(config["parameter_grid"])

    assert len(candidates) == 1152
    assert config["search"]["max_trials"] == 1152
    assert config["features"]["use_scalar_context"] is True
    validate_config(config, num_candidates=len(candidates))


def test_current_level_switch_comparison_includes_xgboost_method() -> None:
    config = load_yaml_config(
        PROJECT_ROOT / "configs/current_level_persistence/switch_comparison.yaml"
    )
    xgboost_methods = [
        method for method in config["methods"]
        if method.get("model_family") == "xgboost"
    ]

    assert len(xgboost_methods) == 1
    assert xgboost_methods[0]["model_id"] == "xgboost"
    assert xgboost_methods[0]["run_id"].startswith(
        "currentLevelPersistence_delta0p5_L30_xgboost_scalarContext_"
    )
