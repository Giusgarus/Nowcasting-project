from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.experiments.long_fade_detection.run_grid_search import (
    build_trial_config,
    expand_model_trials,
    validate_config,
)
from src.data.splits import assign_external_holdout_splits
from src.utils.config import load_yaml_config
from src.tasks.long_fade_detection.data.dataset import (
    FORBIDDEN_MODEL_INPUT_COLUMNS,
    SCALAR_CONTEXT_FEATURE_NAMES,
    build_lag_scalar_features,
    build_long_fade_arrays,
    build_long_fade_window_index,
    compute_scalar_context_features,
    group_fade_events,
    lag_feature_names,
    load_prepared_event_window_signal_frames,
)
from src.tasks.long_fade_detection.evaluation.metrics import (
    event_level_metrics,
    sample_classification_metrics,
    threshold_metrics,
)
from src.tasks.long_fade_detection.models.classifiers import (
    ShapeletConvolutionClassifierConfig,
    TCNClassifierConfig,
    TCNLongFadeClassifier,
    MultiscaleShapeletConvolutionLongFadeClassifier,
)
from src.tasks.long_fade_detection.utils.paths import (
    make_run_id,
    make_selection_id,
    model_dir,
    processed_dataset_dir,
    run_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _frame(signals: list[float], *, start: str = "2021-01-01 00:00:00") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": pd.date_range(start, periods=len(signals), freq="30s"),
            "Signal": signals,
        }
    )


def _two_event_frame() -> pd.DataFrame:
    """Return a frame with two threshold groups separated by more than 3 hours."""

    return pd.DataFrame(
        {
            "Time": list(pd.date_range("2021-01-01 00:00:00", periods=5, freq="30s"))
            + list(pd.date_range("2021-01-01 04:00:00", periods=5, freq="30s")),
            "Signal": [0, 11, 12, 13, 0, 0, 11, 12, 13, 14],
        }
    )


def test_grouped_event_membership_keeps_temporary_below_threshold_gap() -> None:
    frame = _frame([0, 11, 9, 8, 12, 0])
    events = group_fade_events(
        frame,
        dataset_name="demo.csv",
        dataset_id="dataset_001",
        threshold_db=10.0,
        min_fade_duration_seconds=120,
        sampling_time_seconds=30,
        event_grouping_hours=3,
    )

    assert len(events) == 1
    assert events[0].start_index == 1
    assert events[0].end_index == 4
    assert events[0].event_duration_samples == 4
    assert events[0].event_duration_seconds == 120
    assert events[0].y_long_fade == 1


def test_threshold_and_min_duration_are_configurable() -> None:
    frame = _frame([0, 8, 8.5, 0, 11])
    low_threshold_events = group_fade_events(
        frame,
        dataset_name="demo.csv",
        dataset_id="dataset_001",
        threshold_db=8.0,
        min_fade_duration_seconds=90,
        sampling_time_seconds=30,
    )
    high_threshold_events = group_fade_events(
        frame,
        dataset_name="demo.csv",
        dataset_id="dataset_001",
        threshold_db=10.0,
        min_fade_duration_seconds=90,
        sampling_time_seconds=30,
    )

    assert len(low_threshold_events) == 1
    assert low_threshold_events[0].y_long_fade == 1
    assert len(high_threshold_events) == 1
    assert high_threshold_events[0].event_duration_seconds == 30
    assert high_threshold_events[0].y_long_fade == 0


def test_external_holdout_split_has_no_event_leakage() -> None:
    frames = {
        "dev.csv": _two_event_frame(),
        "test.csv": _frame([0, 11, 12, 13, 0], start="2021-02-01"),
    }
    index = build_long_fade_window_index(
        frames,
        threshold_db=10.0,
        min_fade_duration_seconds=60,
        sampling_time_seconds=30,
        context_length=2,
    )
    selected, _ = assign_external_holdout_splits(
        index,
        heldout_test_dataset="test.csv",
        development_datasets=["dev.csv"],
        train_ratio=0.5,
        min_events_for_validation_split=2,
    )

    assert set(selected.loc[selected["dataset_name"].eq("test.csv"), "split"]) == {"test"}
    assert selected.groupby("global_event_id")["split"].nunique().max() == 1


def test_scalar_features_and_lag_scalar_exclude_forbidden_fields() -> None:
    x_raw = np.array([[8, 9, 11, 12, 13]], dtype=np.float32)
    positions = np.array([3], dtype=np.float32)
    scalar = compute_scalar_context_features(
        x_raw,
        threshold_db=10.0,
        position_in_fade_samples=positions,
        sampling_time_seconds=30,
    )
    x_threshold = x_raw - 10.0
    x_lag_scalar, names = build_lag_scalar_features(
        x_threshold,
        scalar,
        context_length=5,
    )

    assert scalar.shape == (1, len(SCALAR_CONTEXT_FEATURE_NAMES))
    assert scalar[0, 0] == 13
    assert scalar[0, 1] == 3
    assert scalar[0, -2] == 90
    assert scalar[0, -1] == 3
    assert lag_feature_names(5) == ["lag_4", "lag_3", "lag_2", "lag_1", "lag_0"]
    assert x_lag_scalar.shape == (1, 5 + len(SCALAR_CONTEXT_FEATURE_NAMES))
    assert not (set(names.tolist()) & FORBIDDEN_MODEL_INPUT_COLUMNS)


def test_build_arrays_uses_train_only_scalar_standardization() -> None:
    frames = {
        "dev.csv": _two_event_frame(),
        "test.csv": _frame([0, 11, 12, 13, 0], start="2021-02-01"),
    }
    index = build_long_fade_window_index(
        frames,
        threshold_db=10.0,
        min_fade_duration_seconds=90,
        sampling_time_seconds=30,
        context_length=2,
    )
    selected, _ = assign_external_holdout_splits(
        index,
        heldout_test_dataset="test.csv",
        development_datasets=["dev.csv"],
        train_ratio=0.5,
        min_events_for_validation_split=2,
    )
    arrays, metadata, scaler = build_long_fade_arrays(
        frames,
        selected,
        context_length=2,
        threshold_db=10.0,
        sampling_time_seconds=30,
    )

    assert set(arrays) == {"train", "validation", "test"}
    assert np.allclose(
        arrays["train"]["scalar_context_features"].mean(axis=0),
        0.0,
        atol=1e-6,
    )
    assert scaler["fitted_on"] == "train"
    assert metadata["test"]["dataset_name"].eq("test.csv").all()


def test_load_prepared_event_windows_stitches_dataset_signal(tmp_path: Path) -> None:
    path = tmp_path / "event_windows.parquet"
    event_windows = pd.DataFrame(
        {
            "dataset_name": ["demo.csv", "demo.csv", "demo.csv", "other.csv"],
            "dataset_id": ["dataset_003", "dataset_003", "dataset_003", "dataset_004"],
            "Time": pd.to_datetime(
                [
                    "2021-01-01 00:00:00",
                    "2021-01-01 00:00:30",
                    "2021-01-01 00:00:30",
                    "2021-01-02 00:00:00",
                ]
            ),
            "Signal_prepared": [9.0, 10.0, 10.0, 11.0],
        }
    )
    event_windows.to_parquet(path, index=False)

    frames, metadata = load_prepared_event_window_signal_frames(
        path,
        dataset_names=["demo.csv"],
        signal_column="Signal_prepared",
    )

    assert set(frames) == {"demo.csv"}
    assert frames["demo.csv"]["Signal"].tolist() == [9.0, 10.0]
    assert frames["demo.csv"]["dataset_id"].unique().tolist() == ["dataset_003"]
    assert metadata["demo.csv"]["source_type"] == "prepared_event_windows"
    assert metadata["demo.csv"]["num_stitched_rows"] == 2


def test_xgboost_fallback_training_smoke() -> None:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=int)
    model = HistGradientBoostingClassifier(max_iter=2, random_state=42)
    model.fit(x, y)
    assert model.predict_proba(x).shape == (4, 2)


def test_tcn_forward_pass() -> None:
    model = TCNLongFadeClassifier(
        TCNClassifierConfig(
            context_length=30,
            hidden_channels=8,
            num_blocks=2,
            dilations=(1, 2),
            scalar_context_dim=len(SCALAR_CONTEXT_FEATURE_NAMES),
            scalar_hidden_dim=4,
        )
    )
    logits = model(
        torch.randn(3, 30),
        torch.randn(3, len(SCALAR_CONTEXT_FEATURE_NAMES)),
    )
    assert logits.shape == (3,)


def test_shapelet_convolution_classifier_forward_pass() -> None:
    model = MultiscaleShapeletConvolutionLongFadeClassifier(
        ShapeletConvolutionClassifierConfig(
            context_length=30,
            shapelet_lengths=(5, 10),
            n_shapelets_per_length=4,
            conv_channels=8,
            scalar_context_dim=len(SCALAR_CONTEXT_FEATURE_NAMES),
            scalar_hidden_dim=4,
        )
    )
    logits = model(
        torch.randn(2, 30),
        torch.randn(2, len(SCALAR_CONTEXT_FEATURE_NAMES)),
    )
    assert logits.shape == (2,)


def test_sample_and_event_metrics_at_configured_thresholds() -> None:
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.4, 0.6, 0.9])
    sample = sample_classification_metrics(y, prob)
    rows = threshold_metrics(y, prob)
    predictions = pd.DataFrame(
        {
            "global_event_id": ["a", "a", "b", "b"],
            "dataset_name": ["d", "d", "d", "d"],
            "event_id": ["a", "a", "b", "b"],
            "position_in_fade_samples": [0, 1, 0, 1],
            "event_duration_seconds": [60, 60, 300, 300],
            "y_true": y,
            "prob_long_fade": prob,
        }
    )
    event = event_level_metrics(predictions)

    assert sample["f1_at_0p5"] == 1.0
    assert {row["threshold"] for row in rows} == {0.3, 0.5, 0.7}
    assert set(event["aggregation"]) == {
        "mean_prob",
        "max_prob",
        "last_prob",
        "early_mean_prob",
    }


def test_long_fade_paths_do_not_touch_existing_task_folders() -> None:
    selection_id = make_selection_id("fc-uplink-fade.csv")
    run_id = make_run_id(
        threshold_db=10.0,
        min_fade_duration_seconds=300,
        model_id="tcn_classifier",
        selection_id=selection_id,
    )

    paths = [
        processed_dataset_dir(
            threshold_db=10.0,
            min_fade_duration_seconds=300,
            context_length=30,
            selection_id=selection_id,
        ),
        run_dir(run_id),
        model_dir(run_id),
    ]
    for path in paths:
        text = str(path)
        assert "long_fade_detection" in text
        assert "autoregressive" not in text
        assert "current_level_persistence" not in text
    assert selection_id in str(run_dir(run_id))
    assert selection_id in run_dir(run_id).name
    assert "tcn_classifier" in str(model_dir(run_id))


def test_long_fade_grid_config_expands_expected_trial_counts() -> None:
    config = load_yaml_config(
        PROJECT_ROOT
        / "configs/long_fade_detection/grid_search_threshold10_duration300.yaml"
    )

    counts = validate_config(config)

    assert counts == {
        "xgboost_lag_scalar_classifier": 2880,
        "tcn_classifier": 3456,
        "multiscale_shapelet_convolution_classifier": 2592,
    }


def test_long_fade_grid_flat_overrides_build_valid_trial_config() -> None:
    config = load_yaml_config(
        PROJECT_ROOT
        / "configs/long_fade_detection/grid_search_threshold10_duration300.yaml"
    )
    tcn_spec = next(
        spec for spec in config["models"] if spec["model_id"] == "tcn_classifier"
    )
    trial = expand_model_trials(config, tcn_spec)[0]
    resolved = build_trial_config(config, tcn_spec, trial["parameters"])

    assert resolved["model_id"] == "tcn_classifier"
    assert resolved["model"]["hidden_channels"] == 32
    assert resolved["model"]["num_blocks"] == 3
    assert resolved["training"]["batch_size"] == 128
    assert resolved["training"]["learning_rate"] == 0.0003
