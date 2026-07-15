"""Compute the Perfect Switch reference for long-fade detection test windows."""

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
from src.tasks.long_fade_detection.data.dataset import (  # noqa: E402
    load_prepared_event_window_signal_frames,
    load_raw_signal_frames,
)
from src.tasks.long_fade_detection.evaluation.switch_from_probability import (  # noqa: E402
    align_perfect_switch_to_long_fade_windows,
)
from src.tasks.long_fade_detection.utils.paths import TASK_NAME  # noqa: E402
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.results_paths import (  # noqa: E402
    SWITCH_REFERENCE_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_perfect_switch_dir,
    get_results_index_dir,
    relative_project_path,
    upsert_index_row,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/long_fade_detection/perfect_switch.yaml"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def build_test_event_timeseries(
    *,
    signal_frames: dict[str, pd.DataFrame],
    test_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Return true signal samples for all long-fade test event spans."""

    event_columns = [
        "dataset_name",
        "dataset_id",
        "event_id",
        "global_event_id",
        "event_start_time",
        "event_end_time",
        "event_duration_samples",
        "event_duration_seconds",
        "y_long_fade",
    ]
    missing = sorted(set(event_columns) - set(test_metadata.columns))
    if missing:
        raise ValueError(f"test_metadata is missing columns: {missing}")

    events = test_metadata[event_columns].drop_duplicates(
        subset=["dataset_name", "event_id"]
    )
    rows = []
    for event in events.itertuples(index=False):
        frame = signal_frames[str(event.dataset_name)]
        start = pd.Timestamp(event.event_start_time)
        end = pd.Timestamp(event.event_end_time)
        selected = frame.loc[
            frame["Time"].ge(start) & frame["Time"].le(end),
            ["Time", "Signal"],
        ].copy()
        if selected.empty:
            raise ValueError(
                f"No signal samples found for {event.dataset_name} {event.event_id}."
            )
        selected = selected.sort_values("Time", kind="stable").reset_index(drop=True)
        selected["Signal_prepared"] = selected["Signal"].astype(float)
        selected["dataset_name"] = str(event.dataset_name)
        selected["dataset_id"] = str(event.dataset_id)
        selected["event_id"] = str(event.event_id)
        selected["global_event_id"] = str(event.global_event_id)
        selected["split"] = "test"
        selected["event_timestamp"] = start
        selected["event_start_time"] = start
        selected["event_end_time"] = end
        selected["event_duration_samples"] = int(event.event_duration_samples)
        selected["event_duration_seconds"] = float(event.event_duration_seconds)
        selected["y_long_fade"] = int(event.y_long_fade)
        selected["quality_flag"] = (
            "long_fade" if int(event.y_long_fade) == 1 else "short_fade"
        )
        selected["event_point_idx"] = range(len(selected))
        rows.append(selected)

    if not rows:
        raise ValueError("No long-fade test event rows were constructed.")
    return pd.concat(rows, ignore_index=True)


def load_signal_frames_from_config(
    data_config: dict,
    *,
    dataset_names: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict], str]:
    """Load the signal source used by the long-fade dataset."""

    if "event_windows_path" in data_config:
        frames, metadata = load_prepared_event_window_signal_frames(
            project_path(data_config["event_windows_path"]),
            dataset_names=dataset_names,
            signal_column=str(data_config.get("signal_column", "Signal_prepared")),
        )
        return frames, metadata, "prepared_event_windows"
    if "raw_data_dir" in data_config:
        frames, metadata = load_raw_signal_frames(
            project_path(data_config["raw_data_dir"]),
            dataset_names=dataset_names,
        )
        return frames, metadata, "raw_data"
    raise ValueError("data must define either event_windows_path or raw_data_dir.")


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != TASK_NAME:
        raise ValueError(f"Config task_name must be {TASK_NAME}.")

    data_config = config["data"]
    switch_config = config["switch"]
    dataset_path = project_path(data_config["dataset_path"])
    dataset_metadata = load_yaml_config(dataset_path / "dataset_metadata.yaml")
    selection_id = str(dataset_metadata["selection_id"])
    test_metadata_path = dataset_path / "test_metadata.parquet"
    test_metadata = pd.read_parquet(test_metadata_path)
    if test_metadata.empty or set(test_metadata["split"]) != {"test"}:
        raise ValueError("Long-fade test metadata must be a non-empty test split.")

    dataset_names = sorted(test_metadata["dataset_name"].astype(str).unique())
    signal_frames, load_metadata, source_type = load_signal_frames_from_config(
        data_config,
        dataset_names=dataset_names,
    )
    event_timeseries = build_test_event_timeseries(
        signal_frames=signal_frames,
        test_metadata=test_metadata,
    )
    threshold = float(switch_config["threshold"])
    switch_time = int(switch_config["switch_time"])
    perfect_timeseries = compute_perfect_switch_from_true_signal(
        event_timeseries,
        signal_column=str(data_config["signal_column"]),
        threshold=threshold,
        condition=str(switch_config["condition"]),
        switch_time=switch_time,
        apply_min_island_length=bool(switch_config["apply_min_island_length"]),
    )
    decision_targets = align_perfect_switch_to_long_fade_windows(
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
            "source_type": source_type,
            "source_metadata": load_metadata,
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

    print("=== Long-Fade Detection Perfect Switch ===")
    print(f"Selection: {selection_id}")
    print(f"Events: {perfect_timeseries['event_id'].nunique()}")
    print(f"Decision windows: {len(decision_targets):,}")
    print(f"Output: {relative_project_path(output_dir)}")


if __name__ == "__main__":
    main()
