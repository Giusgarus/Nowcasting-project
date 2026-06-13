"""Tests for model-agnostic grid-search utilities."""

from pathlib import Path

import pandas as pd
import pytest

from src.utils.config import load_yaml_config

from src.tuning.grid_search import (
    expand_parameter_grid,
    make_trial_id,
    select_best_trial,
)
from scripts.experiments.run_gru_grid_search import (
    build_gru_trial_candidates,
    trial_display_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_expand_parameter_grid_builds_cartesian_product() -> None:
    trials = expand_parameter_grid({"depth": [1, 2], "rate": [0.1, 0.01]})

    assert trials == [
        {"depth": 1, "rate": 0.1},
        {"depth": 1, "rate": 0.01},
        {"depth": 2, "rate": 0.1},
        {"depth": 2, "rate": 0.01},
    ]
    assert make_trial_id(trials[0], 1) == make_trial_id(trials[0], 1)


def test_select_best_trial_uses_completed_validation_metric_only() -> None:
    trials = pd.DataFrame(
        [
            {"trial_id": "a", "status": "complete", "validation_rmse": 2.0},
            {"trial_id": "b", "status": "failed", "validation_rmse": 1.0},
            {"trial_id": "c", "status": "complete", "validation_rmse": 1.5},
        ]
    )

    best = select_best_trial(trials, metric="validation_rmse", mode="min")

    assert best["trial_id"] == "c"


def test_grid_and_selection_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        expand_parameter_grid({})
    with pytest.raises(ValueError, match="Selection mode"):
        select_best_trial(
            pd.DataFrame([{"metric": 1.0}]),
            metric="metric",
            mode="smallest",
        )


def test_gru_grid_search_config_includes_num_layers_in_grid() -> None:
    config = load_yaml_config(PROJECT_ROOT / "configs/gru_grid_search.yaml")

    assert config["fixed_model"].get("num_layers") is None
    assert config["parameter_grid"]["num_layers"] == [1, 2, 3]
    assert len(build_gru_trial_candidates(config, "gru_s2v")) == 225
    assert len(build_gru_trial_candidates(config, "gru_seq2seq")) == 450


def test_gru_grid_search_expands_teacher_forcing_only_for_seq2seq() -> None:
    config = {
        "parameter_grid": {
            "hidden_size": [8],
            "num_layers": [1, 2],
            "learning_rate": [0.001],
            "weight_decay": [0.0],
        },
        "seq2seq": {"teacher_forcing_ratio": [0.0, 0.5]},
    }

    s2v = build_gru_trial_candidates(config, "gru_s2v")
    seq2seq = build_gru_trial_candidates(config, "gru_seq2seq")

    assert len(s2v) == 2
    assert {trial["teacher_forcing_ratio"] for trial in s2v} == {0.0}
    assert len(seq2seq) == 4
    assert {trial["teacher_forcing_ratio"] for trial in seq2seq} == {0.0, 0.5}

    with pytest.raises(ValueError, match="Unsupported GRU architecture"):
        build_gru_trial_candidates(config, "gru_typo")


def test_trial_display_parameters_includes_all_effective_settings() -> None:
    display = trial_display_parameters(
        architecture="gru_seq2seq",
        variant="raw",
        parameters={
            "hidden_size": 32,
            "num_layers": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "teacher_forcing_ratio": 0.5,
        },
        fixed_model={"input_size": 1, "dropout": 0.0, "bidirectional": False},
        training={"batch_size": 64, "max_epochs": 10, "seed": 42},
        prediction_length=10,
    )

    assert display["model"]["num_layers"] == 2
    assert display["optimizer"]["weight_decay"] == 0.0001
    assert display["training"]["teacher_forcing_ratio"] == 0.5
