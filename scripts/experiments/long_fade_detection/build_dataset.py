"""Build the binary long-fade detection dataset."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import assign_external_holdout_splits  # noqa: E402
from src.tasks.long_fade_detection.data.dataset import (  # noqa: E402
    SCALAR_CONTEXT_FEATURE_NAMES,
    SPLIT_FILE_NAMES,
    SPLITS,
    build_long_fade_arrays,
    build_long_fade_window_index,
    load_prepared_event_window_signal_frames,
    load_raw_signal_frames,
    save_split_npz,
    split_summary,
)
from src.tasks.long_fade_detection.utils.paths import make_selection_id  # noqa: E402
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml  # noqa: E402
from src.utils.results_paths import relative_project_path  # noqa: E402

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/long_fade_detection/xgboost_lag_scalar_threshold10_duration300.yaml"
)


def project_path(path: str | Path) -> Path:
    """Resolve a repository-relative or absolute path."""

    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def required_output_paths(output_dir: Path) -> list[Path]:
    """Return files that define a complete dataset build."""

    return [
        output_dir / "train.npz",
        output_dir / "val.npz",
        output_dir / "test.npz",
        output_dir / "train_metadata.parquet",
        output_dir / "val_metadata.parquet",
        output_dir / "test_metadata.parquet",
        output_dir / "dataset_metadata.yaml",
    ]


def dataset_is_complete(output_dir: Path) -> bool:
    """Return true when all required outputs exist."""

    return all(path.exists() for path in required_output_paths(output_dir))


def _split_counts(split_metadata: dict[str, pd.DataFrame]) -> dict[str, int | float]:
    output: dict[str, int | float] = {}
    for split, frame in split_metadata.items():
        prefix = "val" if split == "validation" else split
        output[f"n_{prefix}_samples"] = int(len(frame))
        output[f"n_{prefix}_events"] = int(frame["global_event_id"].nunique())
        output[f"positive_rate_{prefix}"] = (
            float(frame["y_long_fade"].mean()) if len(frame) else 0.0
        )
    return output


def load_signal_frames_from_config(
    source_config: dict,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict], str]:
    """Load the configured long-fade signal source."""

    dataset_names = source_config.get("datasets")
    if "event_windows_path" in source_config:
        frames, metadata = load_prepared_event_window_signal_frames(
            project_path(source_config["event_windows_path"]),
            dataset_names=dataset_names,
            signal_column=str(source_config.get("signal_column", "Signal_prepared")),
        )
        return frames, metadata, "prepared_event_windows"
    if "raw_data_dir" in source_config:
        frames, metadata = load_raw_signal_frames(
            project_path(source_config["raw_data_dir"]),
            dataset_names=dataset_names,
        )
        return frames, metadata, "raw_data"
    raise ValueError("source must define either event_windows_path or raw_data_dir.")


def main() -> None:
    """Build and save the configured dataset."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != "long_fade_detection":
        raise ValueError("Config task_name must be long_fade_detection.")

    event_config = config["event_definition"]
    dataset_config = config["dataset"]
    source_config = config["source"]
    split_config = config["split"]["development_event_split"]
    threshold_db = float(event_config["threshold_db"])
    min_duration = int(event_config["min_fade_duration_seconds"])
    sampling_time = int(event_config["sampling_time_seconds"])
    context_length = int(dataset_config["context_length"])
    external_test_dataset = str(source_config["external_test_dataset"])
    selection_id = make_selection_id(external_test_dataset)
    output_dir = project_path(dataset_config["path"])
    overwrite = bool(config.get("output", {}).get("overwrite", False)) or args.force

    print("=== Long-Fade Detection Dataset Builder ===")
    print(f"Config: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Threshold: {threshold_db} dB")
    print(f"Minimum fade duration: {min_duration} seconds")
    print(f"Context length: {context_length}")
    print(f"External test dataset: {external_test_dataset}")
    print(f"Output folder: {output_dir.relative_to(PROJECT_ROOT)}")
    print("Displays: no figures.")
    print("Saves: split NPZ files, split metadata parquet, and dataset metadata.\n")

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        if dataset_is_complete(output_dir):
            print(
                "Dataset already exists and appears complete. "
                "Use --force or output.overwrite=true to rebuild."
            )
            return
        raise FileExistsError(
            f"Output directory exists but is incomplete: {output_dir}. "
            "Use --force after reviewing it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_frames, load_metadata, source_type = load_signal_frames_from_config(
        source_config
    )
    window_index = build_long_fade_window_index(
        signal_frames,
        threshold_db=threshold_db,
        min_fade_duration_seconds=min_duration,
        sampling_time_seconds=sampling_time,
        context_length=context_length,
        threshold_operator=str(event_config.get("threshold_operator", ">=")),
        event_grouping_hours=float(event_config.get("event_grouping_hours", 3)),
    )
    if window_index.empty:
        raise ValueError("No long-fade candidate windows were built.")

    available_datasets = list(dict.fromkeys(window_index["dataset_name"].tolist()))
    if external_test_dataset not in available_datasets:
        raise ValueError(
            f"External test dataset has no long-fade windows: {external_test_dataset}"
        )
    development_datasets = [
        dataset for dataset in available_datasets if dataset != external_test_dataset
    ]
    selected_index, split_plan = assign_external_holdout_splits(
        window_index,
        heldout_test_dataset=external_test_dataset,
        development_datasets=development_datasets,
        train_ratio=float(split_config["train_fraction"]),
        min_events_for_validation_split=int(
            split_config["min_events_for_validation_split"]
        ),
    )
    split_arrays, split_metadata, scalar_scaler = build_long_fade_arrays(
        signal_frames,
        selected_index,
        context_length=context_length,
        threshold_db=threshold_db,
        sampling_time_seconds=sampling_time,
    )

    output_files: dict[str, str] = {}
    for split in SPLITS:
        stem = SPLIT_FILE_NAMES[split]
        npz_path = output_dir / f"{stem}.npz"
        metadata_path = output_dir / f"{stem}_metadata.parquet"
        save_split_npz(npz_path, split_arrays[split])
        split_metadata[split].to_parquet(metadata_path, index=False)
        output_files[f"{stem}_npz"] = relative_project_path(npz_path)
        output_files[f"{stem}_metadata"] = relative_project_path(metadata_path)

    split_summary_path = output_dir / "split_summary.csv"
    split_plan_path = output_dir / "external_holdout_split_plan.csv"
    event_summary_path = output_dir / "event_summary.csv"
    split_summary(split_metadata).to_csv(split_summary_path, index=False)
    split_plan.to_csv(split_plan_path, index=False)
    (
        selected_index.drop_duplicates("global_event_id")
        .loc[
            :,
            [
                "dataset_name",
                "dataset_id",
                "event_id",
                "global_event_id",
                "split",
                "event_start_time",
                "event_end_time",
                "event_duration_samples",
                "event_duration_seconds",
                "y_long_fade",
            ],
        ]
        .sort_values(["dataset_name", "event_start_time"], kind="stable")
        .to_csv(event_summary_path, index=False)
    )
    output_files["split_summary"] = relative_project_path(split_summary_path)
    output_files["external_holdout_split_plan"] = relative_project_path(split_plan_path)
    output_files["event_summary"] = relative_project_path(event_summary_path)

    counts = _split_counts(split_metadata)
    metadata_path = output_dir / "dataset_metadata.yaml"
    metadata = {
        "task_name": "long_fade_detection",
        "config_path": relative_project_path(config_path),
        "config_fingerprint": config_fingerprint(config),
        "selection_id": selection_id,
        "threshold_db": threshold_db,
        "threshold_operator": str(event_config.get("threshold_operator", ">=")),
        "min_fade_duration_seconds": min_duration,
        "sampling_time_seconds": sampling_time,
        "event_grouping_hours": float(event_config.get("event_grouping_hours", 3)),
        "event_membership_rule": (
            "samples between the first and last threshold-crossing point in each "
            "3-hour grouped event are inside the event, including temporary "
            "below-threshold samples"
        ),
        "context_length": context_length,
        "sequence_input_key": dataset_config["sequence_input_key"],
        "scalar_context_key": dataset_config["scalar_context_key"],
        "lag_scalar_input_key": dataset_config["lag_scalar_input_key"],
        "target_key": dataset_config["target_key"],
        "model_input_policy": {
            "sequence_input": "X_relative_to_threshold = X_raw - threshold_db",
            "xgboost_input": (
                "X_lag_scalar = flattened X_relative_to_threshold plus "
                "train-standardized scalar_context_features"
            ),
            "forbidden_inputs": [
                "above_threshold",
                "position_in_fade_fraction",
                "event_duration_seconds",
                "event_duration_samples",
                "inside_event",
                "y_long_fade",
            ],
        },
        "scalar_context_feature_names": list(SCALAR_CONTEXT_FEATURE_NAMES),
        "scalar_context_scaler": scalar_scaler,
        "external_test_dataset": external_test_dataset,
        "development_datasets": development_datasets,
        "source_datasets": list(signal_frames),
        "source_type": source_type,
        "source_signal_column": str(source_config.get("signal_column", "Signal")),
        "load_metadata": load_metadata,
        "split": config["split"],
        "output_files": {**output_files, "dataset_metadata": relative_project_path(metadata_path)},
        "created_at": datetime.now(timezone.utc).isoformat(),
        **counts,
    }
    save_yaml(metadata_path, metadata)

    print("=== Compact summary ===")
    print(f"Selection ID: {selection_id}")
    print(f"Train samples/events: {counts['n_train_samples']:,}/{counts['n_train_events']:,}")
    print(f"Val samples/events: {counts['n_val_samples']:,}/{counts['n_val_events']:,}")
    print(f"Test samples/events: {counts['n_test_samples']:,}/{counts['n_test_events']:,}")
    print(
        "Positive rates: "
        f"train={counts['positive_rate_train']:.3f}, "
        f"val={counts['positive_rate_val']:.3f}, "
        f"test={counts['positive_rate_test']:.3f}"
    )
    print(f"Metadata: {metadata_path.relative_to(PROJECT_ROOT)}")
    print("Long-fade detection dataset build completed.")


if __name__ == "__main__":
    main()
