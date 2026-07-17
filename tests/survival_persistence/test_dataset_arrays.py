from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.splits import assign_external_holdout_splits
from src.tasks.survival_persistence.data.dataset import (
    SCALAR_CONTEXT_FEATURE_NAMES,
    build_survival_arrays,
    build_survival_window_index,
)
from src.tasks.survival_persistence.data.event_state_machine import (
    StableRecoveryConfig,
)


def _signal_frame(
    signals: list[float],
    *,
    dataset_name: str = "dev.csv",
    dataset_id: str = "dataset_001",
    segment_id: str = "segment_001",
    start: str = "2026-01-01 00:00:00",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": [dataset_id] * len(signals),
            "dataset_name": [dataset_name] * len(signals),
            "segment_id": [segment_id] * len(signals),
            "Time": pd.date_range(start, periods=len(signals), freq="30s"),
            "Signal": signals,
        }
    )


def _config() -> StableRecoveryConfig:
    return StableRecoveryConfig(
        threshold_on=10.0,
        threshold_off=10.0,
        recovery_window_seconds=60,
        recovery_required_fraction=1.0,
        minimum_recovery_observations=2,
    )


def test_observed_targets_have_equal_bounds_and_exclude_event_end_sample() -> None:
    full_signal = _signal_frame([0, 11, 12, 8, 8, 8])
    index, events, _ = build_survival_window_index(
        full_signal,
        recovery_config=_config(),
        context_length=1,
        sampling_time_seconds=30,
    )

    assert len(events) == 1
    assert index["sample_time"].max() < events.loc[0, "event_end_time"]
    assert index["y_event_observed"].eq(1).all()
    np.testing.assert_allclose(
        index["y_lower_bound_seconds"],
        index["y_upper_bound_seconds"],
    )
    assert index["y_time_seconds"].gt(0).all()
    assert index["y_time_seconds"].tolist() == [60.0, 30.0]


def test_censored_targets_keep_infinite_upper_bounds() -> None:
    full_signal = _signal_frame([0, 11, 12, 11])
    index, events, _ = build_survival_window_index(
        full_signal,
        recovery_config=_config(),
        context_length=1,
        sampling_time_seconds=30,
    )

    assert len(events) == 1
    assert not bool(events.loc[0, "event_observed"])
    assert index["y_event_observed"].eq(0).all()
    assert np.isinf(index["y_upper_bound_seconds"].to_numpy(dtype=float)).all()
    assert index["y_lower_bound_seconds"].gt(0).all()


def test_contexts_end_at_sample_time_and_do_not_cross_gaps() -> None:
    full_signal = pd.DataFrame(
        {
            "dataset_id": ["dataset_001"] * 6,
            "dataset_name": ["dev.csv"] * 6,
            "segment_id": ["segment_001"] * 6,
            "Time": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:30",
                    "2026-01-01 00:05:00",
                    "2026-01-01 00:05:30",
                    "2026-01-01 00:06:00",
                    "2026-01-01 00:06:30",
                ]
            ),
            "Signal": [0, 11, 12, 8, 8, 8],
        }
    )
    index, _, diagnostics = build_survival_window_index(
        full_signal,
        recovery_config=_config(),
        context_length=2,
        sampling_time_seconds=30,
        require_contiguous_context=True,
    )

    assert diagnostics["num_context_exclusions"] > 0
    assert not index["sample_time"].isin([pd.Timestamp("2026-01-01 00:05:00")]).any()
    assert (index["input_end_time"] == index["sample_time"]).all()


def test_arrays_include_representations_and_train_only_scalar_standardization() -> None:
    full_signal = pd.concat(
        [
            _signal_frame([0, 11, 12, 8, 8, 8], dataset_name="dev_a.csv", dataset_id="dataset_001"),
            _signal_frame(
                [0, 11, 12, 13, 8, 8, 8],
                dataset_name="dev_a.csv",
                dataset_id="dataset_001",
                segment_id="segment_002",
                start="2026-01-02",
            ),
            _signal_frame(
                [0, 11, 12, 13, 14],
                dataset_name="test.csv",
                dataset_id="dataset_002",
                start="2026-02-01",
            ),
        ],
        ignore_index=True,
    )
    index, _, _ = build_survival_window_index(
        full_signal,
        recovery_config=_config(),
        context_length=2,
        sampling_time_seconds=30,
    )
    selected, _ = assign_external_holdout_splits(
        index,
        heldout_test_dataset="test.csv",
        development_datasets=["dev_a.csv"],
        train_ratio=0.5,
        min_events_for_validation_split=2,
    )
    assert selected.groupby("global_event_id")["split"].nunique().eq(1).all()
    assert selected.loc[selected["dataset_name"].eq("test.csv"), "split"].eq("test").all()
    arrays, metadata, scaler = build_survival_arrays(
        full_signal,
        selected,
        context_length=2,
        threshold_on=10.0,
        standardize_scalar_context=True,
    )

    assert set(arrays) == {"train", "validation", "test"}
    assert arrays["train"]["X_raw"].shape[1] == 2
    assert arrays["train"]["X_relative_to_threshold"].shape == arrays["train"]["X_raw"].shape
    assert arrays["train"]["X_relative_to_current"].shape == arrays["train"]["X_raw"].shape
    assert arrays["train"]["scalar_context_features"].shape[1] == len(
        SCALAR_CONTEXT_FEATURE_NAMES
    )
    assert scaler["fitted_on"] == "train"
    np.testing.assert_allclose(
        arrays["train"]["scalar_context_features"].mean(axis=0),
        0.0,
        atol=1e-6,
    )
    assert metadata["test"]["dataset_name"].eq("test.csv").all()
    for split_arrays in arrays.values():
        assert "y_time_bin_index" not in split_arrays
        assert "at_risk_mask" not in split_arrays
        assert "event_mask" not in split_arrays


def test_event_balanced_weights_sum_to_one_per_event() -> None:
    full_signal = _signal_frame([0, 11, 12, 8, 8, 8])
    index, _, _ = build_survival_window_index(
        full_signal,
        recovery_config=_config(),
        context_length=1,
        sampling_time_seconds=30,
    )

    sums = index.groupby(["dataset_name", "event_id"])["event_balanced_weight"].sum()
    np.testing.assert_allclose(sums.to_numpy(dtype=float), 1.0)
