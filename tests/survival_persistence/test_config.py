from __future__ import annotations

from pathlib import Path

from src.utils.config import load_yaml_config


def test_canonical_survival_config_is_task_scoped() -> None:
    config = load_yaml_config(
        Path("configs/survival_persistence/dataset_threshold10_L30_external_holdout.yaml")
    )

    assert config["task_name"] == "survival_persistence"
    assert config["event_definition"]["threshold_on"] == 10.0
    assert config["dataset"]["sample_policy"] == "all_event_timestamps"
    assert config["dataset"]["keep_censored_samples"] is True
