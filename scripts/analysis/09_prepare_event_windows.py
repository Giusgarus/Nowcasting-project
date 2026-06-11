"""Prepare candidate-event windows and autoregressive window indices."""

# %%
# Path setup and constants
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.data_quality import safe_name
from src.analysis.event_preparation import (
    summarize_event_dataset,
    summarize_imputation,
    summarize_threshold_balance,
    summarize_window_index,
)
from src.data.events import (
    build_event_windows,
    detect_candidate_fade_events,
    prepare_and_assess_event_windows,
    threshold_folder_name,
)
from src.data.loading import load_signal_dataset
from src.data.preprocessing import clean_signal_dataframe
from src.data.splits import assign_event_splits
from src.datasets.windowed_forecasting import build_autoregressive_window_index
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.paths import project_path

CONFIG_PATH = PROJECT_ROOT / "configs/data_preparation.yaml"

# %%
# Load and validate configuration
print("=== Load configuration ===")
config = load_yaml_config(CONFIG_PATH)
if config["output"]["file_format"] != "parquet":
    raise ValueError("This pipeline currently requires output.file_format=parquet.")
if config["candidate_events"]["require_min_persistence"]:
    raise ValueError("Candidate-event preparation must not require minimum persistence.")
if config["split"]["method"] != "chronological_by_event_per_dataset":
    raise ValueError("Only chronological_by_event_per_dataset is supported.")
if config["event_window_quality"]["imputation_method"] != "linear_time":
    raise ValueError("Only linear_time event-window imputation is supported.")
if not (
    config["cleaning"]["drop_invalid_time"]
    and config["cleaning"]["drop_invalid_signal"]
):
    raise ValueError("The strict signal-only preparation currently requires dropping invalid rows.")

threshold = float(config["candidate_events"]["signal_threshold"])
threshold_folder = threshold_folder_name(threshold)
raw_dir = project_path(config["data"]["raw_dir"])
interim_dir = project_path(config["data"]["interim_dir"])
processed_dir = project_path(config["data"]["processed_dir"])
results_root = project_path(config["data"]["results_dir"])
clean_dir = interim_dir / "clean_signal"
candidate_dir = interim_dir / "candidate_events" / threshold_folder
event_window_dir = interim_dir / "event_windows" / threshold_folder
processed_output_dir = processed_dir / "autoregressive" / threshold_folder
results_dir = results_root / threshold_folder

# %%
# Run overview
print(
    "=== Analysis overview ===\n"
    "Prints: preparation progress and final event/window totals.\n"
    "Displays: no figures.\n"
    "Saves: clean Signal data, candidate events, prepared event windows, quality "
    "metadata, event-level splits, autoregressive window indices, and review tables.\n"
    "Note: candidate events are threshold-centered data selections, not Perfect "
    "Switch or a final operational event rule.\n"
)

# %%
# Protect threshold-specific outputs from incompatible silent overwrites
print("=== Check output compatibility ===")
fingerprint = config_fingerprint(config)
run_metadata_path = results_dir / "preparation_run_metadata.yaml"
if run_metadata_path.exists():
    previous_metadata = load_yaml_config(run_metadata_path)
    if previous_metadata.get("config_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Existing outputs under {results_dir} use a different configuration."
        )
cleaning_config = {
    "loading": config["loading"],
    "cleaning": config["cleaning"],
    "sampling": {"gap_threshold_seconds": config["sampling"]["gap_threshold_seconds"]},
}
cleaning_fingerprint = config_fingerprint(cleaning_config)
clean_metadata_path = clean_dir / "clean_signal_metadata.yaml"
if clean_metadata_path.exists():
    previous_clean_metadata = load_yaml_config(clean_metadata_path)
    if previous_clean_metadata.get("config_fingerprint") != cleaning_fingerprint:
        raise RuntimeError(
            f"Existing clean outputs under {clean_dir} use a different configuration."
        )

# %%
# Load, clean, segment, and save all raw datasets
print("=== Clean and segment raw datasets ===")
clean_frames = []
loading_metadata = []
raw_paths = sorted(raw_dir.glob("*.csv"))
for dataset_number, dataset_path in enumerate(raw_paths, start=1):
    dataset_id = f"dataset_{dataset_number:03d}"
    loaded, metadata = load_signal_dataset(
        dataset_path,
        has_header=config["loading"]["has_header"],
        time_col=config["loading"]["time_col"],
        signal_col=config["loading"]["signal_col"],
        time_col_index=config["loading"]["time_col_index"],
        signal_col_index=config["loading"]["signal_col_index"],
    )
    clean = clean_signal_dataframe(
        loaded,
        dataset_id=dataset_id,
        dataset_name=dataset_path.name,
        gap_threshold_seconds=config["sampling"]["gap_threshold_seconds"],
        duplicate_timestamp_strategy=config["cleaning"]["duplicate_timestamp_strategy"],
    )
    clean_frames.append(clean)
    loading_metadata.append(metadata)
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(clean_dir / f"{safe_name(dataset_path.stem)}_clean.parquet", index=False)
    print(f"Cleaned: {dataset_path.name} ({len(clean):,} rows)")

all_clean = pd.concat(clean_frames, ignore_index=True)
all_clean.to_parquet(clean_dir / "all_signal_clean.parquet", index=False)
dataset_names = all_clean["dataset_name"].drop_duplicates().tolist()
save_yaml(
    clean_metadata_path,
    {
        "config_fingerprint": cleaning_fingerprint,
        "datasets": loading_metadata,
    },
)

# %%
# Detect candidate events using first-crossing anchored grouping
print("=== Detect candidate fade events ===")
events = detect_candidate_fade_events(
    all_clean,
    signal_threshold=threshold,
    condition=config["candidate_events"]["condition"],
    grouping_hours=config["candidate_events"]["grouping_hours"],
    window_pre_hours=config["candidate_events"]["window_pre_hours"],
    window_post_hours=config["candidate_events"]["window_post_hours"],
)
events = assign_event_splits(
    events,
    train_ratio=config["split"]["train_ratio"],
    validation_ratio=config["split"]["val_ratio"],
)
candidate_dir.mkdir(parents=True, exist_ok=True)
events.to_parquet(candidate_dir / "candidate_fade_events.parquet", index=False)
event_counts = {
    dataset_name: int(events["dataset_name"].eq(dataset_name).sum())
    for dataset_name in all_clean["dataset_name"].drop_duplicates()
}
save_yaml(
    candidate_dir / "candidate_event_detection_metadata.yaml",
    {
        "threshold_folder": threshold_folder,
        "signal_threshold": threshold,
        "condition": config["candidate_events"]["condition"],
        "grouping_hours": config["candidate_events"]["grouping_hours"],
        "window_pre_hours": config["candidate_events"]["window_pre_hours"],
        "window_post_hours": config["candidate_events"]["window_post_hours"],
        "require_min_persistence": False,
        "events_per_dataset": event_counts,
        "datasets_without_events": [
            name for name, count in event_counts.items() if count == 0
        ],
    },
)
print(f"Candidate events detected: {len(events):,}")

# %%
# Extract event windows, impute recoverable gaps, and assess quality
print("=== Prepare and assess event windows ===")
observed_event_windows = build_event_windows(all_clean, events)
prepared_event_windows, event_quality = prepare_and_assess_event_windows(
    observed_event_windows,
    events,
    expected_seconds=config["sampling"]["expected_seconds"],
    signal_threshold=threshold,
    min_coverage_ratio=config["event_window_quality"]["min_coverage_ratio"],
    max_gap_seconds_for_imputation=config["event_window_quality"][
        "max_gap_seconds_for_imputation"
    ],
    max_small_gap_samples=config["event_window_quality"]["max_small_gap_samples"],
    max_warning_gap_samples=config["event_window_quality"]["max_warning_gap_samples"],
    context_length=config["autoregressive_windows"]["context_length"],
    prediction_length=config["autoregressive_windows"]["prediction_length"],
)
split_lookup = events.set_index("event_id")["split"]
prepared_event_windows["split"] = prepared_event_windows["event_id"].map(split_lookup)
event_window_dir.mkdir(parents=True, exist_ok=True)
if config["output"]["save_per_dataset_event_windows"]:
    for dataset_name in dataset_names:
        frame = prepared_event_windows.loc[
            prepared_event_windows["dataset_name"].eq(dataset_name)
        ]
        frame.to_parquet(
            event_window_dir / f"{safe_name(Path(dataset_name).stem)}_event_windows.parquet",
            index=False,
        )
if config["output"]["save_all_event_windows"]:
    prepared_event_windows.to_parquet(
        event_window_dir / "all_event_windows.parquet",
        index=False,
    )
event_quality.to_parquet(event_window_dir / "event_window_quality.parquet", index=False)

# %%
# Build the final traceable autoregressive window index
print("=== Build autoregressive window index ===")
window_config = config["autoregressive_windows"]
window_index = build_autoregressive_window_index(
    prepared_event_windows,
    event_quality,
    events,
    context_length=window_config["context_length"],
    prediction_length=window_config["prediction_length"],
    gap_threshold_seconds=config["sampling"]["gap_threshold_seconds"],
    allow_warning_events=config["event_window_quality"][
        "allow_warning_events_for_window_index"
    ],
    signal_threshold=threshold,
)
processed_output_dir.mkdir(parents=True, exist_ok=True)
events[
    ["event_id", "dataset_id", "dataset_name", "event_timestamp", "split"]
].to_parquet(processed_output_dir / "split_metadata.parquet", index=False)
window_index_path = processed_output_dir / (
    f"window_index_L{window_config['context_length']}_"
    f"h{window_config['prediction_length']}.parquet"
)
window_index.to_parquet(window_index_path, index=False)

# %%
# Create simplified review tables
print("=== Create event-dataset review summaries ===")
results_dir.mkdir(parents=True, exist_ok=True)
summary_tables = {
    "event_dataset_summary.csv": summarize_event_dataset(
        events,
        prepared_event_windows,
        event_quality,
        window_index,
        signal_threshold=threshold,
        dataset_names=dataset_names,
    ),
    "event_quality_summary.csv": event_quality,
    "threshold_balance_summary.csv": summarize_threshold_balance(
        prepared_event_windows,
        signal_threshold=threshold,
        dataset_names=dataset_names,
    ),
    "imputation_summary.csv": summarize_imputation(
        prepared_event_windows,
        event_quality,
        dataset_names=dataset_names,
    ),
    "window_index_summary.csv": summarize_window_index(
        events,
        event_quality,
        window_index,
        signal_threshold=threshold,
        context_length=window_config["context_length"],
        prediction_length=window_config["prediction_length"],
        dataset_names=dataset_names,
    ),
}
for filename, table in summary_tables.items():
    table.to_csv(results_dir / filename, index=False)
    print(f"Saved: {results_dir / filename}")

if config["output"]["save_metadata"]:
    save_yaml(
        run_metadata_path,
        {
            "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "config_fingerprint": fingerprint,
            "threshold_folder": threshold_folder,
            "output_folders": {
                "clean_signal": str(clean_dir.relative_to(PROJECT_ROOT)),
                "candidate_events": str(candidate_dir.relative_to(PROJECT_ROOT)),
                "event_windows": str(event_window_dir.relative_to(PROJECT_ROOT)),
                "processed_autoregressive": str(
                    processed_output_dir.relative_to(PROJECT_ROOT)
                ),
                "results": str(results_dir.relative_to(PROJECT_ROOT)),
            },
        },
    )

# %%
# Compact final summary
print("=== Compact final summary ===")
quality_counts = event_quality["quality_flag"].value_counts()
above_threshold = prepared_event_windows["Signal_prepared"].gt(threshold)
print(f"Datasets processed: {len(clean_frames)}")
print(f"Events detected: {len(events):,}")
print(f"Usable events: {int(quality_counts.get('usable', 0)):,}")
print(f"Warning events: {int(quality_counts.get('warning', 0)):,}")
print(f"Unusable events: {int(quality_counts.get('unusable', 0)):,}")
print(f"Total event-window points: {len(prepared_event_windows):,}")
print(f"Points above threshold: {int(above_threshold.sum()):,}")
print(f"Points below/equal threshold: {int((~above_threshold).sum()):,}")
print(f"Imputed points: {int(prepared_event_windows['is_imputed'].sum()):,}")
print(f"Autoregressive windows created: {len(window_index):,}")
print(f"Train windows: {int(window_index['split'].eq('train').sum()):,}")
print(f"Validation windows: {int(window_index['split'].eq('validation').sum()):,}")
print(f"Test windows: {int(window_index['split'].eq('test').sum()):,}")
print(f"Interim output: {interim_dir}")
print(f"Processed output: {processed_output_dir}")
print(f"Review summaries: {results_dir}")
