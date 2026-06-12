"""Tests for final supervised autoregressive dataset construction."""

import numpy as np
import pandas as pd
import pytest

from src.datasets.autoregressive_dataset import (
    build_autoregressive_arrays_from_window_index,
    compute_context_standardization,
    filter_window_index,
    rank_datasets_for_autoregressive_training,
    select_datasets,
    validate_event_split_integrity,
)


def _window_index() -> pd.DataFrame:
    rows = []
    definitions = [
        ("window_1", "event_a", "dataset_a", "a.csv", "usable", "train", 0),
        ("window_2", "event_a", "dataset_a", "a.csv", "warning", "train", 1),
        ("window_3", "event_b", "dataset_b", "b.csv", "usable", "test", 0),
        ("window_4", "event_c", "dataset_b", "b.csv", "usable", "validation", 0),
        ("window_5", "event_d", "dataset_c", "c.csv", "unusable", "test", 0),
        ("window_6", "event_b", "dataset_b", "b.csv", "usable", "test", 1),
    ]
    for window_id, event_id, dataset_id, dataset_name, quality, split, start in definitions:
        rows.append(
            {
                "window_id": window_id,
                "event_id": event_id,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "segment_id": f"{event_id}_segment",
                "split": split,
                "quality_flag": quality,
                "input_start_idx": start,
                "input_end_idx": start + 2,
                "target_start_idx": start + 3,
                "target_end_idx": start + 4,
                "input_start_time": pd.Timestamp("2026-01-01") + pd.Timedelta(
                    seconds=30 * start
                ),
                "input_end_time": pd.Timestamp("2026-01-01") + pd.Timedelta(
                    seconds=30 * (start + 2)
                ),
                "target_start_time": pd.Timestamp("2026-01-01") + pd.Timedelta(
                    seconds=30 * (start + 3)
                ),
                "target_end_time": pd.Timestamp("2026-01-01") + pd.Timedelta(
                    seconds=30 * (start + 4)
                ),
                "context_length": 3,
                "prediction_length": 2,
                "signal_threshold": 10.0,
                "num_imputed_points_in_window": 0,
                "region_type": "event_window",
            }
        )
    return pd.DataFrame(rows)


def _event_windows_for_event_a() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=6, freq="30s")
    return pd.DataFrame(
        {
            "event_id": "event_a",
            "dataset_id": "dataset_a",
            "dataset_name": "a.csv",
            "segment_id": "event_a_segment",
            "Time": times,
            "event_point_idx": range(6),
            "Signal_prepared": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
        }
    )


def test_dataset_selection_modes() -> None:
    ranking = rank_datasets_for_autoregressive_training(
        _window_index(),
        allow_warning_events=True,
        ranking_metric="num_valid_autoregressive_windows",
    )

    auto, auto_folder, _ = select_datasets(
        ranking, mode="auto_largest", selected_datasets=[]
    )
    single, _, _ = select_datasets(
        ranking, mode="single_dataset", selected_datasets=["a.csv"]
    )
    multiple, multi_folder, _ = select_datasets(
        ranking,
        mode="multiple_datasets",
        selected_datasets=["a.csv", "c.csv"],
    )
    all_datasets, _, _ = select_datasets(
        ranking, mode="all_datasets", selected_datasets=[]
    )

    assert auto == ["b.csv"]
    assert auto_folder == "auto_largest_b"
    assert single == ["a.csv"]
    assert multiple == ["a.csv", "c.csv"]
    assert multi_folder.startswith("multi_2datasets_")
    assert all_datasets == ["b.csv", "a.csv", "c.csv"]


def test_quality_filtering_controls_warning_and_excludes_unusable() -> None:
    with_warning = filter_window_index(_window_index(), allow_warning_events=True)
    without_warning = filter_window_index(_window_index(), allow_warning_events=False)

    assert set(with_warning["quality_flag"]) == {"usable", "warning"}
    assert set(without_warning["quality_flag"]) == {"usable"}


def test_event_split_integrity_rejects_leakage() -> None:
    index = _window_index()
    index.loc[index["window_id"].eq("window_2"), "split"] = "test"

    with pytest.raises(ValueError, match="more than one split"):
        validate_event_split_integrity(index)


def test_builder_shapes_standardization_and_metadata_alignment() -> None:
    index = _window_index().loc[lambda frame: frame["event_id"].eq("event_a")]
    arrays, metadata = build_autoregressive_arrays_from_window_index(
        _event_windows_for_event_a(),
        index,
        context_length=3,
        prediction_length=2,
        signal_column="Signal_prepared",
        epsilon=1.0e-6,
    )
    train = arrays["train"]

    assert train["X_raw"].shape == (2, 3, 1)
    assert train["y_raw"].shape == (2, 2)
    assert train["X_raw"].dtype == np.float32
    assert train["window_id"].tolist() == metadata["train"]["window_id"].tolist()
    assert train["scaling_mean"][0] == pytest.approx(2.0)
    assert train["scaling_mean"][0] < train["y_raw"][0].mean()
    reconstructed = (
        train["y_context_standard"] * train["scaling_std"][:, None]
        + train["scaling_mean"][:, None]
    )
    np.testing.assert_allclose(reconstructed, train["y_raw"], rtol=1e-5)


def test_context_standardization_uses_epsilon_for_near_zero_std() -> None:
    X_raw = np.ones((1, 3, 1), dtype=np.float32) * 5
    y_raw = np.array([[6.0, 7.0]], dtype=np.float32)

    normalized = compute_context_standardization(X_raw, y_raw, epsilon=1.0e-3)

    assert normalized["scaling_std_raw"][0] == 0
    assert normalized["scaling_std"][0] == pytest.approx(1.0e-3)
    np.testing.assert_allclose(normalized["X_context_standard"], 0)
