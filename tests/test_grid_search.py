"""Tests for model-agnostic grid-search utilities."""

from pathlib import Path

import pandas as pd
import pytest

import src.tuning.parallel_trials as parallel_trials
from src.utils.config import load_yaml_config

from src.tuning.grid_search import (
    expand_parameter_grid,
    make_trial_id,
    select_best_trial,
)
from src.tuning.parallel_trials import (
    choose_trial_devices,
    iter_parallel_trial_results,
)
from scripts.experiments.run_gru_grid_search import (
    build_gru_trial_candidates,
    trial_display_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _serial_trial_worker(job: dict) -> tuple[int, str]:
    return job["value"] * 2, job["device"]


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


def test_parallel_trial_devices_use_available_cuda_dynamically() -> None:
    config = {"enabled": True, "max_workers": 3, "cuda_device_ids": "auto"}

    assert choose_trial_devices(config, cuda_count=3, fallback_device="cpu") == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
    ]
    assert choose_trial_devices(config, cuda_count=2, fallback_device="cpu") == [
        "cuda:0",
        "cuda:1",
    ]
    assert choose_trial_devices(config, cuda_count=0, fallback_device="cpu") == [
        "cpu"
    ]


def test_parallel_trial_devices_filter_unavailable_explicit_ids() -> None:
    config = {
        "enabled": True,
        "max_workers": "auto",
        "cuda_device_ids": [0, 2, 4],
    }

    assert choose_trial_devices(config, cuda_count=3, fallback_device="cpu") == [
        "cuda:0",
        "cuda:2",
    ]


def test_trial_scheduler_falls_back_to_serial_device_execution() -> None:
    results = list(
        iter_parallel_trial_results(
            [{"value": 1}, {"value": 2}],
            _serial_trial_worker,
            ["cpu"],
        )
    )

    assert results == [(2, "cpu"), (4, "cpu")]


def test_trial_scheduler_falls_back_when_processes_are_unavailable(
    monkeypatch,
) -> None:
    class UnavailableExecutor:
        def __init__(self, *args, **kwargs) -> None:
            raise PermissionError("not allowed")

    monkeypatch.setattr(parallel_trials, "ProcessPoolExecutor", UnavailableExecutor)
    with pytest.warns(RuntimeWarning, match="running trials sequentially"):
        results = list(
            iter_parallel_trial_results(
                [{"value": 1}, {"value": 2}, {"value": 3}],
                _serial_trial_worker,
                ["worker:0", "worker:1"],
            )
        )

    assert sorted(value for value, _ in results) == [2, 4, 6]
    assert {device for _, device in results} == {"worker:0", "worker:1"}
