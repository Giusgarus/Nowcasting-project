from __future__ import annotations

from pathlib import Path

from scripts.experiments.survival_persistence.run_discrete_time_tcn_grid_search import (
    build_trial_config,
    completed_trial_is_valid,
    validate_config,
)
from src.utils.config import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/survival_persistence/models/discrete_time_tcn_grid_search.yaml"
)


def test_grid_config_expands_to_expected_trial_count() -> None:
    config = load_yaml_config(CONFIG_PATH)

    assert validate_config(config) == 1620


def test_build_trial_config_applies_flat_overrides() -> None:
    config = load_yaml_config(CONFIG_PATH)

    trial = build_trial_config(
        config,
        {
            "model.hidden_channels": [64, 64, 64],
            "model.kernel_size": 5,
            "training.learning_rate": 0.0015,
            "training.batch_size": 256,
        },
    )

    assert trial["model"]["hidden_channels"] == [64, 64, 64]
    assert trial["model"]["kernel_size"] == 5
    assert trial["training"]["learning_rate"] == 0.0015
    assert trial["training"]["batch_size"] == 256
    assert trial["dataset"]["sequence_representation"] == "relative_to_current"
    assert trial["dataset"]["sample_weighting"] == "uniform"


def test_completed_trial_reuse_requires_checkpoint_and_fingerprint(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    (trial_dir / "metadata.yaml").write_text(
        "status: complete\ntrial_fingerprint: abc\n",
        encoding="utf-8",
    )
    (trial_dir / "validation_metrics.yaml").write_text(
        "validation_integrated_brier_score: 0.1\n",
        encoding="utf-8",
    )

    assert not completed_trial_is_valid(trial_dir, "abc")

    (trial_dir / "best_model.pt").write_bytes(b"checkpoint")

    assert completed_trial_is_valid(trial_dir, "abc")
    assert not completed_trial_is_valid(trial_dir, "other")
