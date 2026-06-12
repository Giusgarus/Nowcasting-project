"""Tests for stable output identifiers, paths, and indexes."""

from pathlib import Path

import pandas as pd

from src.utils.results_paths import (
    RUN_INDEX_COLUMNS,
    get_comparison_dir,
    get_data_preparation_dir,
    get_model_dir,
    get_perfect_switch_dir,
    get_run_dir,
    make_comparison_id,
    make_run_id,
    make_selection_id,
    resolve_processed_dataset_dir,
    upsert_index_row,
)


def test_stable_result_ids() -> None:
    selection_id = make_selection_id(
        mode="auto_largest",
        context_length=30,
        prediction_length=10,
        threshold=10.0,
        selected_datasets=["example.csv"],
    )
    run_id = make_run_id("gru", "gru_s2v", "context_standard", selection_id)

    assert selection_id == "autoLargest_L30_h10_thr10"
    assert run_id == "gru_s2v_contextStandard_autoLargest_L30_h10_thr10"
    assert (
        make_comparison_id(run_id)
        == "switch_eval_gru_s2v_contextStandard_autoLargest_L30_h10_thr10"
    )


def test_result_and_model_paths_are_shallow(tmp_path: Path) -> None:
    run_id = "gru_s2v_raw_autoLargest_L30_h10_thr10"

    assert get_run_dir(run_id, tmp_path) == tmp_path / "results/runs" / run_id
    assert get_model_dir("gru", run_id, tmp_path) == tmp_path / "models/gru" / run_id
    assert get_perfect_switch_dir("selection", tmp_path) == (
        tmp_path / "results/switching/perfect_switch/selection"
    )
    assert get_comparison_dir("comparison", tmp_path) == (
        tmp_path / "results/comparisons/comparison"
    )
    assert get_data_preparation_dir("selection", tmp_path) == (
        tmp_path / "results/data_preparation/selection"
    )


def test_processed_dataset_resolver_supports_new_and_legacy_paths(
    tmp_path: Path,
) -> None:
    legacy = (
        tmp_path
        / "data/processed/autoregressive/threshold_10p0"
        / "datasets_L30_h10/legacy_selection"
    )
    legacy.mkdir(parents=True)

    resolved_legacy = resolve_processed_dataset_dir(
        threshold_folder="threshold_10p0",
        context_length=30,
        prediction_length=10,
        selection_id="selection",
        legacy_selection_folder="legacy_selection",
        root=tmp_path,
    )
    assert resolved_legacy == legacy

    new = (
        tmp_path
        / "data/processed/autoregressive/threshold_10p0"
        / "L30_h10/selection"
    )
    new.mkdir(parents=True)
    resolved_new = resolve_processed_dataset_dir(
        threshold_folder="threshold_10p0",
        context_length=30,
        prediction_length=10,
        selection_id="selection",
        legacy_selection_folder="legacy_selection",
        root=tmp_path,
    )
    assert resolved_new == new


def test_index_upsert_replaces_same_id(tmp_path: Path) -> None:
    path = tmp_path / "results/index/runs.csv"
    row = {column: "" for column in RUN_INDEX_COLUMNS}
    row.update({"run_id": "run_a", "status": "first"})
    upsert_index_row(path, row, id_column="run_id", columns=RUN_INDEX_COLUMNS)
    row["status"] = "updated"
    upsert_index_row(path, row, id_column="run_id", columns=RUN_INDEX_COLUMNS)

    index = pd.read_csv(path)
    assert len(index) == 1
    assert index.iloc[0]["status"] == "updated"
