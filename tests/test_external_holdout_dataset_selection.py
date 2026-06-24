"""Tests for external-holdout autoregressive dataset selection."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.maintenance.clean_generated_datasets import discover_cleanup_candidates
from src.tasks.autoregressive.data.autoregressive_dataset import (
    assign_external_holdout_splits,
    build_autoregressive_arrays_from_window_index,
    rank_datasets_for_autoregressive_training,
    select_datasets,
)
from src.utils.results_paths import make_selection_id


def _toy_window_index() -> pd.DataFrame:
    rows = []
    definitions = [
        ("dev_a.csv", "dataset_a", 4),
        ("dev_b.csv", "dataset_b", 1),
        ("dev_c.csv", "dataset_c", 3),
        ("fc-uplink-fade.csv", "dataset_fc", 2),
    ]
    for dataset_name, dataset_id, num_events in definitions:
        for event_number in range(num_events):
            event_id = f"event_{event_number + 1:03d}"
            rows.append(
                {
                    "window_id": f"window_{event_number + 1:03d}",
                    "event_id": event_id,
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "segment_id": f"{dataset_id}_{event_id}_segment",
                    "split": "train",
                    "quality_flag": "usable",
                    "input_start_idx": 0,
                    "input_end_idx": 1,
                    "target_start_idx": 2,
                    "target_end_idx": 2,
                    "input_start_time": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(days=event_number),
                    "input_end_time": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(days=event_number, seconds=30),
                    "target_start_time": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(days=event_number, seconds=60),
                    "target_end_time": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(days=event_number, seconds=60),
                    "context_length": 2,
                    "prediction_length": 1,
                    "signal_threshold": 10.0,
                    "num_imputed_points_in_window": 0,
                    "region_type": "event_window",
                }
            )
    return pd.DataFrame(rows)


def _toy_event_windows(index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in index.itertuples(index=False):
        for point_idx in range(3):
            rows.append(
                {
                    "event_id": row.event_id,
                    "dataset_id": row.dataset_id,
                    "dataset_name": row.dataset_name,
                    "segment_id": row.segment_id,
                    "Time": row.input_start_time
                    + pd.Timedelta(seconds=30 * point_idx),
                    "event_point_idx": point_idx,
                    "Signal_prepared": float(point_idx + len(rows)),
                }
            )
    return pd.DataFrame(rows)


def test_external_holdout_split_assigns_heldout_only_to_test() -> None:
    selected, split_plan = assign_external_holdout_splits(
        _toy_window_index(),
        heldout_test_dataset="fc-uplink-fade.csv",
        development_datasets=["dev_a.csv", "dev_b.csv", "dev_c.csv"],
        train_ratio=0.85,
        min_events_for_validation_split=3,
    )

    heldout = selected.loc[selected["dataset_name"].eq("fc-uplink-fade.csv")]
    assert set(heldout["split"]) == {"test"}
    assert "fc-uplink-fade.csv" not in set(
        selected.loc[selected["split"].isin(["train", "validation"]), "dataset_name"]
    )
    assert selected.groupby("global_event_id")["split"].nunique().max() == 1
    assert selected["global_window_id"].is_unique
    assert set(selected.loc[selected["dataset_name"].eq("dev_b.csv"), "split"]) == {
        "train"
    }
    assert split_plan.loc[
        split_plan["dataset_name"].eq("dev_b.csv"), "policy"
    ].item() == "put_all_in_train"


def test_external_holdout_chronological_development_split() -> None:
    selected, _ = assign_external_holdout_splits(
        _toy_window_index(),
        heldout_test_dataset="fc-uplink-fade.csv",
        development_datasets=["dev_a.csv", "dev_b.csv", "dev_c.csv"],
        train_ratio=0.85,
        min_events_for_validation_split=3,
    )

    dev_a = selected.loc[selected["dataset_name"].eq("dev_a.csv")].sort_values(
        "input_start_time",
        kind="stable",
    )
    assert dev_a["split"].tolist() == ["train", "train", "train", "validation"]
    dev_c = selected.loc[selected["dataset_name"].eq("dev_c.csv")].sort_values(
        "input_start_time",
        kind="stable",
    )
    assert dev_c["split"].tolist() == ["train", "train", "validation"]


def test_external_holdout_arrays_are_concatenated_by_global_splits() -> None:
    index = _toy_window_index()
    selected, _ = assign_external_holdout_splits(
        index,
        heldout_test_dataset="fc-uplink-fade.csv",
        development_datasets=["dev_a.csv", "dev_b.csv", "dev_c.csv"],
        train_ratio=0.85,
        min_events_for_validation_split=3,
    )
    arrays, metadata = build_autoregressive_arrays_from_window_index(
        _toy_event_windows(index),
        selected,
        context_length=2,
        prediction_length=1,
        signal_column="Signal_prepared",
        epsilon=1.0e-6,
    )

    assert arrays["train"]["X_raw"].shape[0] == 6
    assert arrays["validation"]["X_raw"].shape[0] == 2
    assert arrays["test"]["X_raw"].shape[0] == 2
    assert set(metadata["test"]["dataset_name"]) == {"fc-uplink-fade.csv"}
    assert "global_event_id" in metadata["train"]
    assert "global_window_id" in metadata["train"]
    np.testing.assert_array_equal(
        arrays["train"]["global_event_id"],
        metadata["train"]["global_event_id"].to_numpy(dtype=str),
    )


def test_external_holdout_selection_and_selection_id() -> None:
    ranking = rank_datasets_for_autoregressive_training(
        _toy_window_index(),
        allow_warning_events=True,
        ranking_metric="num_valid_autoregressive_windows",
    )

    selected, folder, selected_ranking = select_datasets(
        ranking,
        mode="external_holdout",
        selected_datasets=[],
        heldout_test_dataset="fc-uplink-fade.csv",
        include_all_except_heldout=True,
    )
    selection_id = make_selection_id(
        mode="external_holdout",
        selected_datasets=selected,
        context_length=30,
        prediction_length=10,
        threshold=10.0,
    )

    assert selected[-1] == "fc-uplink-fade.csv"
    assert folder == "externalHoldout_test_fc_uplink_fade"
    assert selected_ranking["is_selected"].all()
    assert selection_id == "externalHoldout_test_fc_uplink_fade_L30_h10_thr10"


def test_cleanup_discovery_dry_run_does_not_delete(tmp_path: Path) -> None:
    index_path = tmp_path / "results" / "index" / "datasets.csv"
    obsolete = (
        tmp_path
        / "data"
        / "processed"
        / "autoregressive"
        / "threshold_10p0"
        / "datasets_L30_h10"
        / "auto_largest_old"
    )
    current = obsolete.parent / "externalHoldout_test_fc_uplink_fade"
    raw = tmp_path / "data" / "raw" / "fc-uplink-fade.csv"
    stale_threshold = (
        tmp_path
        / "data"
        / "processed"
        / "autoregressive"
        / "threshold_5p0"
    )
    stale_interim = tmp_path / "data" / "interim" / "event_windows" / "threshold_5p0"
    obsolete.mkdir(parents=True)
    current.mkdir(parents=True)
    stale_threshold.mkdir(parents=True)
    stale_interim.mkdir(parents=True)
    raw.parent.mkdir(parents=True)
    raw.write_text("raw", encoding="utf-8")
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        "\n".join(
            [
                "selection_id,dataset_selection_mode,selected_datasets,threshold,"
                "context_length,prediction_length,dataset_path,num_train_windows,"
                "num_val_windows,num_test_windows,created_at",
                "externalHoldout_test_fc_uplink_fade_L30_h10_thr10,"
                "external_holdout,fc-uplink-fade.csv,10.0,30,10,"
                "data/processed/autoregressive/threshold_10p0/"
                "datasets_L30_h10/externalHoldout_test_fc_uplink_fade,"
                "1,1,1,2026-01-01T00:00:00+00:00",
            ]
        ),
        encoding="utf-8",
    )

    candidates, uncertain = discover_cleanup_candidates(tmp_path)

    assert {candidate.path for candidate in candidates} == {
        obsolete,
        stale_threshold,
        stale_interim,
    }
    assert raw.exists()
    assert current.exists()
    assert uncertain == []
