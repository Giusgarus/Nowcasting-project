"""Compute the Perfect Switch reference for survival-persistence test windows."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.switching.metrics import (  # noqa: E402
    compute_dataset_level_switch_summary,
    compute_duration_summary,
    compute_event_level_switch_summary,
    compute_global_switch_summary,
)
from src.switching.reference_switch import compute_perfect_switch_from_true_signal  # noqa: E402
from src.tasks.survival_persistence.evaluation.switch_from_survival import (  # noqa: E402
    align_perfect_switch_to_survival_windows,
)
from src.tasks.survival_persistence.utils.paths import TASK_NAME  # noqa: E402
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.results_paths import (  # noqa: E402
    SWITCH_REFERENCE_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_perfect_switch_dir,
    get_results_index_dir,
    relative_project_path,
    upsert_index_row,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/survival_persistence/perfect_switch.yaml"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _event_span_end(event: pd.Series) -> pd.Timestamp:
    if bool(event["event_observed"]):
        return pd.Timestamp(event["event_end_time"])
    return pd.Timestamp(event["segment_end_time"])


def build_test_event_timeseries(
    *,
    signal_source: pd.DataFrame,
    test_metadata: pd.DataFrame,
    signal_column: str,
) -> pd.DataFrame:
    """Return true signal samples over every survival test event span."""

    event_columns = [
        "dataset_name",
        "dataset_id",
        "segment_id",
        "event_id",
        "global_event_id",
        "event_start_time",
        "event_end_time",
        "event_observed",
        "censoring_reason",
        "segment_end_time",
    ]
    missing = sorted(set(event_columns) - set(test_metadata.columns))
    if missing:
        raise ValueError(f"test_metadata is missing columns: {missing}")

    signal_required = ["dataset_id", "dataset_name", "Time", signal_column]
    missing_signal = sorted(set(signal_required) - set(signal_source.columns))
    if missing_signal:
        raise ValueError(f"signal_source is missing columns: {missing_signal}")

    signal = signal_source.copy()
    signal["Time"] = pd.to_datetime(signal["Time"], errors="raise")
    events = test_metadata[event_columns].drop_duplicates(
        subset=["global_event_id"]
    )
    rows = []
    for _, event in events.iterrows():
        start = pd.Timestamp(event["event_start_time"])
        end = _event_span_end(event)
        selected = signal.loc[
            signal["dataset_id"].astype(str).eq(str(event["dataset_id"]))
            & signal["dataset_name"].astype(str).eq(str(event["dataset_name"]))
            & signal["Time"].ge(start)
            & signal["Time"].le(end),
            ["Time", signal_column],
        ].copy()
        if selected.empty:
            raise ValueError(
                f"No signal samples found for {event['global_event_id']}."
            )
        selected = selected.sort_values("Time", kind="stable").reset_index(drop=True)
        selected["dataset_name"] = str(event["dataset_name"])
        selected["dataset_id"] = str(event["dataset_id"])
        selected["segment_id"] = str(event["segment_id"])
        selected["event_id"] = str(event["event_id"])
        selected["global_event_id"] = str(event["global_event_id"])
        selected["split"] = "test"
        selected["event_timestamp"] = start
        selected["event_start_time"] = start
        selected["event_end_time"] = pd.Timestamp(event["event_end_time"])
        selected["event_observed"] = bool(event["event_observed"])
        selected["censoring_reason"] = str(event["censoring_reason"])
        selected["quality_flag"] = "observed" if bool(event["event_observed"]) else "censored"
        selected["event_point_idx"] = range(len(selected))
        rows.append(selected)

    if not rows:
        raise ValueError("No survival test event rows were constructed.")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != TASK_NAME:
        raise ValueError(f"Config task_name must be {TASK_NAME}.")

    data_config = config["data"]
    switch_config = config["switch"]
    dataset_path = project_path(data_config["dataset_path"])
    signal_source_path = project_path(data_config["signal_source_path"])
    signal_column = str(data_config["signal_column"])
    dataset_metadata = load_yaml_config(dataset_path / "dataset_metadata.yaml")
    selection_id = str(dataset_metadata["selection_id"])
    test_metadata_path = dataset_path / "test_metadata.parquet"
    test_metadata = pd.read_parquet(test_metadata_path)
    if test_metadata.empty or set(test_metadata["split"]) != {"test"}:
        raise ValueError("Survival test metadata must be a non-empty test split.")

    signal_source = pd.read_parquet(signal_source_path)
    event_timeseries = build_test_event_timeseries(
        signal_source=signal_source,
        test_metadata=test_metadata,
        signal_column=signal_column,
    )
    threshold = float(switch_config["threshold"])
    switch_time = int(switch_config["switch_time"])
    perfect_timeseries = compute_perfect_switch_from_true_signal(
        event_timeseries,
        signal_column=signal_column,
        threshold=threshold,
        condition=str(switch_config["condition"]),
        switch_time=switch_time,
        apply_min_island_length=bool(switch_config["apply_min_island_length"]),
    )
    decision_targets = align_perfect_switch_to_survival_windows(
        perfect_timeseries,
        test_metadata,
    )

    output_dir = get_perfect_switch_dir(selection_id, task_name=TASK_NAME)
    output_paths = ensure_results_subdirs(output_dir, ("tables",))
    tables_dir = output_paths["tables"]
    timeseries_path = tables_dir / "perfect_switch_timeseries.parquet"
    targets_path = tables_dir / "perfect_switch_decision_targets.parquet"
    perfect_timeseries.to_parquet(timeseries_path, index=False)
    decision_targets.to_parquet(targets_path, index=False)

    event_summary = compute_event_level_switch_summary(perfect_timeseries)
    dataset_summary = compute_dataset_level_switch_summary(
        perfect_timeseries,
        event_summary,
    )
    global_summary = compute_global_switch_summary(
        perfect_timeseries,
        selection_folder=selection_id,
        num_windows=len(test_metadata),
        threshold=threshold,
        switch_time=switch_time,
    )
    duration_summary = compute_duration_summary(perfect_timeseries)
    global_summary.to_csv(tables_dir / "perfect_switch_summary.csv", index=False)
    dataset_summary.to_csv(tables_dir / "perfect_switch_by_dataset.csv", index=False)
    event_summary.to_csv(tables_dir / "perfect_switch_by_event.csv", index=False)
    duration_summary.to_csv(
        tables_dir / "perfect_switch_duration_summary.csv",
        index=False,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    reference_id = f"perfect_switch_{TASK_NAME}_{selection_id}"
    metadata = {
        "reference_id": reference_id,
        "reference_type": "perfect_switch",
        "task_name": TASK_NAME,
        "selection_id": selection_id,
        "split": "test",
        "threshold": threshold,
        "condition": str(switch_config["condition"]),
        "switch_time": switch_time,
        "apply_min_island_length": bool(
            switch_config["apply_min_island_length"]
        ),
        "num_events": int(perfect_timeseries["event_id"].nunique()),
        "num_decision_windows": int(len(decision_targets)),
        "config_path": relative_project_path(config_path),
        "config_fingerprint": config_fingerprint(config),
        "input_files": {
            "signal_source": relative_project_path(signal_source_path),
            "test_metadata": relative_project_path(test_metadata_path),
        },
        "output_files": {
            "timeseries": relative_project_path(timeseries_path),
            "decision_targets": relative_project_path(targets_path),
        },
        "created_at": created_at,
    }
    save_yaml(output_dir / "metadata.yaml", metadata)
    upsert_index_row(
        get_results_index_dir() / "switch_references.csv",
        {
            "reference_id": reference_id,
            "reference_type": "perfect_switch",
            "selection_id": selection_id,
            "threshold": threshold,
            "switch_time": switch_time,
            "results_path": relative_project_path(output_dir),
            "timeseries_path": relative_project_path(timeseries_path),
            "window_targets_path": relative_project_path(targets_path),
            "status": "complete",
            "created_at": created_at,
        },
        id_column="reference_id",
        columns=SWITCH_REFERENCE_INDEX_COLUMNS,
    )

    print("=== Survival-Persistence Perfect Switch ===")
    print(f"Selection: {selection_id}")
    print(f"Events: {perfect_timeseries['event_id'].nunique()}")
    print(f"Decision windows: {len(decision_targets):,}")
    print(f"Output: {relative_project_path(output_dir)}")


if __name__ == "__main__":
    main()
