"""Build final supervised autoregressive datasets without training models."""

# %%
# Path setup and imports
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.autoregressive_dataset import (
    SPLIT_FILE_NAMES,
    SPLITS,
    build_autoregressive_arrays_from_window_index,
    filter_window_index,
    rank_datasets_for_autoregressive_training,
    save_split_npz,
    select_datasets,
    summarize_final_dataset,
    summarize_normalization,
    summarize_splits,
    validate_event_split_integrity,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.paths import project_path

CONFIG_PATH = PROJECT_ROOT / "configs/autoregressive_dataset.yaml"

# %%
# Load and validate configuration
print("=== Load configuration ===")
config = load_yaml_config(CONFIG_PATH)
forecasting = config["forecasting"]
selection = config["dataset_selection"]
normalization = config["normalization"]
output_config = config["output"]

required_versions = {"raw", "context_standard"}
if not required_versions.issubset(normalization["versions"]):
    raise ValueError("normalization.versions must include raw and context_standard.")
context_standard = normalization["context_standard"]
if (
    context_standard["center"] != "mean"
    or context_standard["scale"] != "std"
    or context_standard["fit_on"] != "input_context_only"
):
    raise ValueError("Only input-context mean/std standardization is supported.")

event_windows_path = project_path(config["data"]["event_windows_path"])
window_index_path = project_path(config["data"]["window_index_path"])
output_root = project_path(config["data"]["output_root"])
results_root = project_path(config["data"]["results_root"]) / "final_datasets"

# %%
# Run overview
print(
    "=== Analysis overview ===\n"
    "Prints: selected datasets, split sizes, normalization diagnostics, and paths.\n"
    "Displays: no figures.\n"
    "Saves: final raw/context-standard NPZ datasets, aligned metadata, and summaries.\n"
    "Note: this script preserves the official event-level split and trains no model.\n"
)

# %%
# Load official preparation outputs
print("=== Load event windows and official window index ===")
event_windows = pd.read_parquet(event_windows_path)
window_index = pd.read_parquet(window_index_path)
validate_event_split_integrity(window_index)

# %%
# Rank and select datasets
print("=== Rank and select datasets ===")
ranking = rank_datasets_for_autoregressive_training(
    window_index,
    allow_warning_events=selection["allow_warning_events"],
    ranking_metric=selection["ranking_metric"],
)
selected_datasets, selection_folder, ranking = select_datasets(
    ranking,
    mode=selection["mode"],
    selected_datasets=selection["selected_datasets"],
)
filtered_index = filter_window_index(
    window_index,
    allow_warning_events=selection["allow_warning_events"],
)
selected_index = filtered_index.loc[
    filtered_index["dataset_name"].isin(selected_datasets)
].copy()
if selected_index.empty:
    raise ValueError("Dataset selection produced no valid autoregressive windows.")
validate_event_split_integrity(selected_index)

output_dir = output_root / selection_folder
summary_dir = results_root / selection_folder
if not output_config["overwrite"]:
    existing = [
        path
        for folder in (output_dir, summary_dir)
        if folder.exists()
        for path in folder.iterdir()
    ]
    if existing:
        raise FileExistsError(
            f"Outputs already exist for this selection: {existing[0].parent}"
        )
output_dir.mkdir(parents=True, exist_ok=True)
summary_dir.mkdir(parents=True, exist_ok=True)

# %%
# Build final raw and context-standard datasets
print("=== Build supervised arrays and aligned metadata ===")
split_arrays, split_metadata = build_autoregressive_arrays_from_window_index(
    event_windows,
    selected_index,
    context_length=forecasting["context_length"],
    prediction_length=forecasting["prediction_length"],
    signal_column=forecasting["signal_column"],
    epsilon=float(context_standard["epsilon"]),
)

# %%
# Save split arrays and metadata
print("=== Save final split datasets ===")
output_files = {}
for split in SPLITS:
    file_stem = SPLIT_FILE_NAMES[split]
    if output_config["save_npz"]:
        npz_path = output_dir / f"{file_stem}.npz"
        save_split_npz(npz_path, split_arrays[split])
        output_files[f"{file_stem}_npz"] = str(npz_path.relative_to(PROJECT_ROOT))
    if output_config["save_metadata_parquet"]:
        metadata_path = output_dir / f"{file_stem}_metadata.parquet"
        split_metadata[split].to_parquet(metadata_path, index=False)
        output_files[f"{file_stem}_metadata"] = str(
            metadata_path.relative_to(PROJECT_ROOT)
        )

# %%
# Build and save review summaries
print("=== Save final dataset summaries ===")
thresholds = selected_index["signal_threshold"].drop_duplicates()
if len(thresholds) != 1:
    raise ValueError("Selected windows must use exactly one signal threshold.")
signal_threshold = float(thresholds.iloc[0])
summary_tables = {
    "selected_dataset_ranking.csv": ranking,
    "final_dataset_summary.csv": summarize_final_dataset(
        selected_index,
        selection_mode=selection["mode"],
        selection_folder=selection_folder,
        selected_datasets=selected_datasets,
        context_length=forecasting["context_length"],
        prediction_length=forecasting["prediction_length"],
        signal_threshold=signal_threshold,
        allow_warning_events=selection["allow_warning_events"],
        normalization_versions=normalization["versions"],
    ),
    "split_summary.csv": summarize_splits(split_metadata, ranking),
    "normalization_summary.csv": summarize_normalization(
        split_metadata,
        ranking,
        epsilon=float(context_standard["epsilon"]),
    ),
}
if output_config["save_summary_csv"]:
    for filename, table in summary_tables.items():
        path = summary_dir / filename
        table.to_csv(path, index=False)
        output_files[filename.removesuffix(".csv")] = str(path.relative_to(PROJECT_ROOT))

# %%
# Save reproducibility metadata
metadata = {
    "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
    "config_fingerprint": config_fingerprint(config),
    "threshold": signal_threshold,
    "threshold_folder": config["data"]["threshold_folder"],
    "context_length": forecasting["context_length"],
    "prediction_length": forecasting["prediction_length"],
    "signal_column": forecasting["signal_column"],
    "selected_datasets": selected_datasets,
    "dataset_selection_mode": selection["mode"],
    "selection_folder": selection_folder,
    "allow_warning_events": selection["allow_warning_events"],
    "normalization_versions": normalization["versions"],
    "normalization_epsilon": float(context_standard["epsilon"]),
    "input_files": {
        "event_windows": str(event_windows_path.relative_to(PROJECT_ROOT)),
        "window_index": str(window_index_path.relative_to(PROJECT_ROOT)),
    },
    "output_files": output_files,
    "num_windows_total": len(selected_index),
    "num_train_windows": len(split_metadata["train"]),
    "num_val_windows": len(split_metadata["validation"]),
    "num_test_windows": len(split_metadata["test"]),
    "num_events_total": int(selected_index["event_id"].nunique()),
    "num_train_events": int(split_metadata["train"]["event_id"].nunique()),
    "num_val_events": int(split_metadata["validation"]["event_id"].nunique()),
    "num_test_events": int(split_metadata["test"]["event_id"].nunique()),
    "num_datasets": len(selected_datasets),
    "ranking_metric": selection["ranking_metric"],
    "ranking_table": str(
        (summary_dir / "selected_dataset_ranking.csv").relative_to(PROJECT_ROOT)
    ),
    "chosen_dataset": selected_datasets[0]
    if selection["mode"] == "auto_largest"
    else None,
    "created_at": datetime.now(timezone.utc).isoformat(),
}
dataset_metadata_path = output_dir / "dataset_metadata.yaml"
metadata["output_files"]["dataset_metadata"] = str(
    dataset_metadata_path.relative_to(PROJECT_ROOT)
)
save_yaml(dataset_metadata_path, metadata)

# %%
# Compact final summary
normalization_summary = summary_tables["normalization_summary.csv"]
print("=== Compact final summary ===")
print(f"Selection mode: {selection['mode']}")
print(f"Selected datasets: {', '.join(selected_datasets)}")
print(f"Output folder: {output_dir}")
print(f"Train windows: {len(split_metadata['train']):,}")
print(f"Validation windows: {len(split_metadata['validation']):,}")
print(f"Test windows: {len(split_metadata['test']):,}")
print(f"Total events: {selected_index['event_id'].nunique():,}")
print(f"Normalization versions: {', '.join(normalization['versions'])}")
print(
    "Near-zero std windows: "
    f"{int(normalization_summary['num_near_zero_std_windows'].sum()):,}"
)
print(f"Summary folder: {summary_dir}")
