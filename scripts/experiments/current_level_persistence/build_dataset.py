"""Build the current-level persistence supervised dataset."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import assign_external_holdout_splits
from src.data.imputation import (
    impute_small_time_gaps,
    small_gap_imputation_config_from_mapping,
)
from src.tasks.current_level_persistence.data.dataset import (
    SCALAR_CONTEXT_FEATURE_NAMES,
    SPLIT_FILE_NAMES,
    SPLITS,
    build_current_level_persistence_arrays,
    build_current_level_persistence_index,
    save_split_npz,
    split_metadata_for_yaml,
    summarize_split_metadata,
    target_statistics_seconds_for_yaml,
)
from src.tasks.current_level_persistence.utils.paths import make_selection_id
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.results_paths import relative_project_path

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/current_level_persistence/dataset_delta_0p5_L30_external_holdout.yaml"
)


def project_path(path: str | Path) -> Path:
    """Resolve an absolute or repository-relative path."""

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def parse_args() -> argparse.Namespace:
    """Parse dataset-builder options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Current-level persistence dataset YAML config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the configured output dataset already exists.",
    )
    return parser.parse_args()


def required_output_paths(output_dir: Path) -> list[Path]:
    """Return the files that define a complete dataset build."""

    return [
        output_dir / "train.npz",
        output_dir / "val.npz",
        output_dir / "test.npz",
        output_dir / "train_metadata.parquet",
        output_dir / "val_metadata.parquet",
        output_dir / "test_metadata.parquet",
        output_dir / "censored_window_metadata.parquet",
        output_dir / "dataset_metadata.yaml",
    ]


def dataset_is_complete(output_dir: Path) -> bool:
    """Return true when all required dataset files already exist."""

    return all(path.exists() for path in required_output_paths(output_dir))


def main() -> None:
    """Build and save the configured current-level persistence dataset."""

    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml_config(config_path)
    if config.get("task_name") != "current_level_persistence":
        raise ValueError("Config task_name must be current_level_persistence.")

    context_length = int(config["context_length"])
    delta = float(config["delta"])
    sampling_time_seconds = int(config["sampling_time_seconds"])
    external_test_dataset = str(config["external_test_dataset"])
    source = config["source"]
    quality_config = config["event_window_quality"]
    split_config = config["split"]["development_event_split"]
    output_dir = project_path(config["output_dir"])
    overwrite = bool(config.get("output", {}).get("overwrite", False)) or args.force
    input_columns = config.get("input_columns", {})
    signal_column = str(
        source.get("signal_column")
        or input_columns.get("signal")
        or "Signal_prepared"
    )
    full_signal_column = str(source.get("full_signal_column", "Signal"))
    selection_id = make_selection_id(external_test_dataset)

    print("=== Current-Level Persistence Dataset Builder ===")
    print(f"Config: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Task: {config['task_name']}")
    print(f"Context length: {context_length}")
    print(f"Delta: {delta}")
    print(f"Sampling time seconds: {sampling_time_seconds}")
    print(f"External test dataset: {external_test_dataset}")
    print(f"Output folder: {output_dir.relative_to(PROJECT_ROOT)}")
    print("Displays: no figures.")
    print("Saves: split NPZ files, split metadata parquet, and dataset metadata.\n")

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        if dataset_is_complete(output_dir):
            print(
                "Dataset already exists and appears complete. "
                "Skipping rebuild. Use --force or output.overwrite=true to rebuild.\n"
                f"Existing folder: {output_dir.relative_to(PROJECT_ROOT)}"
            )
            return
        raise FileExistsError(
            f"Output directory exists but is incomplete: {output_dir}. "
            "Use --force or output.overwrite=true after reviewing the folder."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    event_windows_path = project_path(source["event_windows_path"])
    event_quality_path = project_path(source["event_quality_path"])
    full_signal_path = project_path(source["full_signal_path"])
    event_windows = pd.read_parquet(event_windows_path)
    event_quality = pd.read_parquet(event_quality_path)
    full_signal = pd.read_parquet(full_signal_path)
    imputation_config = small_gap_imputation_config_from_mapping(
        config.get("small_gap_imputation"),
        enabled_default=False,
    )
    imputation_summary_path: Path | None = None
    if imputation_config.enabled:
        imputation_result = impute_small_time_gaps(
            full_signal,
            time_column="Time",
            signal_column=full_signal_column,
            group_columns=("dataset_id", "dataset_name", "segment_id"),
            config=imputation_config,
        )
        full_signal = imputation_result.dataframe
        imputation_summary_path = output_dir / "small_gap_imputation_summary.csv"
        imputation_result.summary.to_csv(imputation_summary_path, index=False)

    window_index = build_current_level_persistence_index(
        event_windows,
        event_quality,
        full_signal,
        context_length=context_length,
        delta=delta,
        sampling_time_seconds=sampling_time_seconds,
        signal_column=signal_column,
        full_signal_column=full_signal_column,
        allow_warning_events=bool(
            quality_config["allow_warning_events_for_window_index"]
        ),
    )
    if window_index.empty:
        raise ValueError("Current-level persistence window index is empty.")

    available_datasets = list(dict.fromkeys(window_index["dataset_name"].tolist()))
    if external_test_dataset not in available_datasets:
        raise ValueError(
            f"External test dataset not found in event windows: {external_test_dataset}"
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
    censored_index = selected_index.loc[
        ~selected_index["target_observed"].astype(bool)
    ].copy()
    selected_index = selected_index.loc[
        selected_index["target_observed"].astype(bool)
    ].copy()
    if selected_index.empty:
        raise ValueError("No observed persistence targets remain after censoring.")
    split_arrays, split_metadata = build_current_level_persistence_arrays(
        event_windows,
        selected_index,
        context_length=context_length,
        signal_column=signal_column,
    )

    output_files: dict[str, str] = {}
    if imputation_summary_path is not None:
        output_files["small_gap_imputation_summary"] = relative_project_path(
            imputation_summary_path
        )
    for split in SPLITS:
        file_stem = SPLIT_FILE_NAMES[split]
        npz_path = output_dir / f"{file_stem}.npz"
        metadata_path = output_dir / f"{file_stem}_metadata.parquet"
        save_split_npz(npz_path, split_arrays[split])
        split_metadata[split].to_parquet(metadata_path, index=False)
        output_files[f"{file_stem}_npz"] = relative_project_path(npz_path)
        output_files[f"{file_stem}_metadata"] = relative_project_path(metadata_path)

    split_summary = summarize_split_metadata(split_metadata)
    split_plan_path = output_dir / "external_holdout_split_plan.csv"
    split_summary_path = output_dir / "split_summary.csv"
    censored_metadata_path = output_dir / "censored_window_metadata.parquet"
    censoring_summary_path = output_dir / "censoring_summary.csv"
    split_plan.to_csv(split_plan_path, index=False)
    split_summary.to_csv(split_summary_path, index=False)
    censored_index.to_parquet(censored_metadata_path, index=False)
    censoring_summary = pd.DataFrame(
        [
            {
                "split": split,
                "num_observed_targets": int(selected_index["split"].eq(split).sum()),
                "num_censored_targets": int(censored_index["split"].eq(split).sum()),
            }
            for split in SPLITS
        ]
    )
    censoring_summary["num_candidate_windows"] = (
        censoring_summary["num_observed_targets"]
        + censoring_summary["num_censored_targets"]
    )
    censoring_summary["pct_censored"] = (
        100.0
        * censoring_summary["num_censored_targets"]
        / censoring_summary["num_candidate_windows"].clip(lower=1)
    )
    censoring_summary.to_csv(censoring_summary_path, index=False)
    output_files["external_holdout_split_plan"] = relative_project_path(split_plan_path)
    output_files["split_summary"] = relative_project_path(split_summary_path)
    output_files["censored_window_metadata"] = relative_project_path(
        censored_metadata_path
    )
    output_files["censoring_summary"] = relative_project_path(
        censoring_summary_path
    )

    dataset_metadata = {
        "task_name": "current_level_persistence",
        "config_path": relative_project_path(config_path),
        "config_fingerprint": config_fingerprint(config),
        "selection_id": selection_id,
        "selection_folder": output_dir.name,
        "context_length": context_length,
        "delta": delta,
        "sampling_time_seconds": sampling_time_seconds,
        "target_definition": (
            "elapsed seconds from t to the first observed future Signal below "
            "Signal_t - delta in the full clean continuous segment"
        ),
        "target_search_scope": "full_clean_signal_same_segment",
        "censoring_policy": (
            "exclude from supervised regression when recovery is not observed "
            "before the continuous segment ends"
        ),
        "input_representation": "X_relative_to_current = X_raw - Signal_t",
        "signal_column": signal_column,
        "input_representations": [
            "X_raw",
            "X_relative_to_current",
            "scalar_context_features",
        ],
        "scalar_context_feature_names": list(SCALAR_CONTEXT_FEATURE_NAMES),
        "targets": [
            "remaining_persistence_samples",
            "remaining_persistence_seconds",
            "log1p_remaining_persistence_seconds",
        ],
        "external_test_dataset": external_test_dataset,
        "development_datasets": development_datasets,
        "source_datasets": available_datasets,
        "event_window_quality": {
            "allow_warning_events_for_window_index": bool(
                quality_config["allow_warning_events_for_window_index"]
            ),
            "num_usable_events": int(event_quality["quality_flag"].eq("usable").sum()),
            "num_warning_events": int(event_quality["quality_flag"].eq("warning").sum()),
            "num_unusable_events_excluded": int(
                event_quality["quality_flag"].eq("unusable").sum()
            ),
        },
        "split": config["split"],
        "splits": split_metadata_for_yaml(split_metadata),
        "target_statistics_seconds": target_statistics_seconds_for_yaml(
            split_metadata
        ),
        "source_files": {
            "event_windows": relative_project_path(event_windows_path),
            "event_quality": relative_project_path(event_quality_path),
            "full_signal": relative_project_path(full_signal_path),
        },
        "small_gap_imputation": {
            "enabled": imputation_config.enabled,
            "scope": "full_signal_target_search",
            "expected_seconds": imputation_config.expected_seconds,
            "max_gap_seconds": imputation_config.max_gap_seconds,
            "max_missing_run": imputation_config.max_missing_run,
            "method": imputation_config.method,
            "summary": relative_project_path(imputation_summary_path)
            if imputation_summary_path is not None
            else "",
        },
        "output_files": output_files,
        "num_train_windows": len(split_metadata["train"]),
        "num_val_windows": len(split_metadata["validation"]),
        "num_test_windows": len(split_metadata["test"]),
        "num_censored_windows": len(censored_index),
        "num_events_total": int(selected_index["global_event_id"].nunique()),
        "num_datasets_total": int(selected_index["dataset_id"].nunique()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / "dataset_metadata.yaml"
    dataset_metadata["output_files"]["dataset_metadata"] = relative_project_path(
        metadata_path
    )
    save_yaml(metadata_path, dataset_metadata)

    print("=== Compact summary ===")
    print(f"Selection ID: {selection_id}")
    print(f"Development datasets: {', '.join(development_datasets)}")
    print(f"Train windows: {len(split_metadata['train']):,}")
    print(f"Validation windows: {len(split_metadata['validation']):,}")
    print(f"Test windows: {len(split_metadata['test']):,}")
    print(f"Censored windows excluded: {len(censored_index):,}")
    print(f"Metadata: {metadata_path.relative_to(PROJECT_ROOT)}")
    print("Current-level persistence dataset build completed.")


if __name__ == "__main__":
    main()
