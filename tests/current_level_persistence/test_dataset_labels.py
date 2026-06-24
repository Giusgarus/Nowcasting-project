import numpy as np
import pandas as pd

from src.tasks.current_level_persistence.data.dataset import (
    build_current_level_persistence_arrays,
    build_current_level_persistence_index,
    relative_to_current,
    remaining_persistence_samples,
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
    assert metadata["train"].loc[0, "global_event_id"] == "toy.csv::event_001"
    assert metadata["train"].loc[0, "global_window_id"].startswith(
        "toy.csv::clp_window_"
    )
