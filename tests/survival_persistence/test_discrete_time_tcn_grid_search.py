from __future__ import annotations

from pathlib import Path

from scripts.experiments.survival_persistence.run_discrete_time_tcn_grid_search import (
    build_trial_config,
    build_tcn_run_index_row,
    completed_trial_is_valid,
    validate_config,
)
from src.tasks.survival_persistence.utils.paths import RUN_INDEX_COLUMNS
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


def test_tcn_run_index_row_matches_survival_index_schema(tmp_path: Path) -> None:
    row = build_tcn_run_index_row(
        run_id="run",
        feature_set="relative_current_plus_scalar",
        sample_weighting="uniform",
        selection_id="selection",
        dataset_dir=tmp_path / "dataset",
        best_epoch=7,
        validation_metrics={
            "validation_discrete_nll": 1.2,
            "validation_harrell_c_index": 0.6,
            "validation_integrated_brier_score": 0.1,
        },
        test_metrics={
            "test_harrell_c_index": 0.55,
            "test_integrated_brier_score": 0.12,
        },
        created_at="2026-01-01T00:00:00+00:00",
    )

    assert set(RUN_INDEX_COLUMNS).issubset(row)
    assert row["best_val_aft_nloglik"] != row["best_val_aft_nloglik"]
    assert row["best_val_discrete_nll"] == 1.2
