import numpy as np
import pandas as pd

from src.tasks.current_level_persistence.data.dataset import (
    SCALAR_CONTEXT_FEATURE_NAMES,
    build_current_level_persistence_arrays,
    build_current_level_persistence_index,
    relative_to_current,
    remaining_persistence_samples,
    split_metadata_for_yaml,
    target_statistics_seconds_for_yaml,
)


def test_remaining_persistence_label_counts_until_recovery_level_break() -> None:
    signal = np.array([10.0, 10.5, 11.0, 10.8, 10.2, 9.9])

    samples = remaining_persistence_samples(signal, 2, delta=0.5)

    assert samples == 2


def test_relative_to_current_last_value_is_zero() -> None:
    window = np.array([10.0, 10.5, 11.0])

    relative = relative_to_current(window)

    np.testing.assert_allclose(relative, np.array([-1.0, -0.5, 0.0], dtype=np.float32))
    assert relative[-1] == 0.0


def test_dataset_arrays_include_current_level_targets_and_representations() -> None:
    event_windows = pd.DataFrame(
        {
            "dataset_id": ["dataset_001"] * 6,
            "dataset_name": ["toy.csv"] * 6,
            "segment_id": ["segment_001"] * 6,
            "event_id": ["event_001"] * 6,
            "Time": pd.date_range("2026-01-01", periods=6, freq="30s"),
            "event_point_idx": np.arange(6),
            "Signal_prepared": [10.0, 10.5, 11.0, 10.8, 10.2, 9.9],
        }
    )
    index = build_current_level_persistence_index(
        event_windows,
        context_length=3,
        delta=0.5,
        sampling_time_seconds=30,
        signal_column="Signal_prepared",
    )
    index["split"] = "train"

    arrays, metadata = build_current_level_persistence_arrays(
        event_windows,
        index,
        context_length=3,
        signal_column="Signal_prepared",
    )

    assert arrays["train"]["X_raw"].shape == (4, 3)
    np.testing.assert_allclose(arrays["train"]["X_raw"][0], [10.0, 10.5, 11.0])
    np.testing.assert_allclose(
        arrays["train"]["X_relative_to_current"][0],
        [-1.0, -0.5, 0.0],
    )
    assert arrays["train"]["y_remaining_persistence_samples"][0] == 2
    assert arrays["train"]["y_remaining_persistence_seconds"][0] == 60.0
    assert arrays["train"]["scalar_context_features"].shape == (4, 12)
    assert arrays["train"]["scalar_context_feature_names"].tolist() == list(
        SCALAR_CONTEXT_FEATURE_NAMES
    )
    assert set(SCALAR_CONTEXT_FEATURE_NAMES).issubset(metadata["train"].columns)
    assert metadata["train"].loc[0, "global_event_id"] == "toy.csv::event_001"
    assert metadata["train"].loc[0, "global_window_id"].startswith(
        "toy.csv::clp_window_"
    )


def test_dataset_metadata_helpers_use_required_split_names() -> None:
    metadata = {
        "train": pd.DataFrame(
            {
                "global_event_id": ["toy.csv::event_001", "toy.csv::event_001"],
                "dataset_name": ["toy.csv", "toy.csv"],
                "remaining_persistence_seconds": [30.0, 90.0],
            }
        ),
        "validation": pd.DataFrame(
            {
                "global_event_id": ["toy.csv::event_002"],
                "dataset_name": ["toy.csv"],
                "remaining_persistence_seconds": [60.0],
            }
        ),
        "test": pd.DataFrame(
            columns=[
                "global_event_id",
                "dataset_name",
                "remaining_persistence_seconds",
            ]
        ),
    }

    splits = split_metadata_for_yaml(metadata)
    stats = target_statistics_seconds_for_yaml(metadata)

    assert set(splits) == {"train", "val", "test"}
    assert splits["train"]["n_samples"] == 2
    assert splits["train"]["n_events"] == 1
    assert splits["val"]["n_samples"] == 1
    assert stats["train"]["mean"] == 60.0
    assert stats["train"]["p10"] == 36.0
    assert stats["test"]["mean"] is None
