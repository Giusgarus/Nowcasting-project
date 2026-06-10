"""Analyze local Signal changes and exploratory spike candidates."""

# %%
# Path setup and constants
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.data_quality import (
    compute_signal_change_summary,
    compute_signal_changes,
    find_top_signal_spikes,
)
from src.analysis.plots import (
    plot_abs_delta_histogram,
    plot_delta_histogram,
    plot_signal_with_spike_markers,
    plot_spike_zoom,
)
from src.data.loading import load_signal_dataset

RAW_DIR = PROJECT_ROOT / "data/raw"
OUTPUT_DIR = PROJECT_ROOT / "results/data_analysis"
TOP_K_SPIKES = 20
N_SPIKE_ZOOMS = 3
ZOOM_RADIUS = 100

# %%
# Load datasets
print("=== Load datasets ===")
loaded_datasets = []
errors = []
for dataset_path in sorted(RAW_DIR.glob("*.csv")):
    try:
        dataframe, _ = load_signal_dataset(dataset_path)
        loaded_datasets.append((dataset_path.name, dataframe))
        print(f"Loaded: {dataset_path.name} ({len(dataframe):,} rows)")
    except Exception as error:
        errors.append({"dataset_name": dataset_path.name, "error": str(error)})
        print(f"Failed: {dataset_path.name}: {error}")

# %%
# Compute change summaries and display figures
print("=== Compute Signal-change and spike diagnostics ===")
summary_rows = []
spike_frames = []
figures_displayed = 0
for dataset_name, dataframe in loaded_datasets:
    summary_rows.append(
        {
            "dataset_name": dataset_name,
            **compute_signal_change_summary(dataframe),
        }
    )
    changes = compute_signal_changes(dataframe)
    spikes = find_top_signal_spikes(dataframe, TOP_K_SPIKES)
    spikes.insert(0, "dataset_name", dataset_name)
    spike_frames.append(spikes)

    plot_delta_histogram(
        changes["delta_signal"],
        f"Signed consecutive Signal changes | Dataset: {dataset_name}",
    )
    plot_abs_delta_histogram(
        changes["abs_delta_signal"],
        f"Absolute consecutive Signal changes | Dataset: {dataset_name}",
    )
    plot_signal_with_spike_markers(
        dataframe,
        spikes,
        f"Signal with largest change candidates | Dataset: {dataset_name}",
    )
    figures_displayed += 3

    for _, spike in spikes.head(N_SPIKE_ZOOMS).iterrows():
        plot_spike_zoom(
            dataframe,
            spike["time_current"],
            ZOOM_RADIUS,
            f"Spike candidate rank {int(spike['rank'])} | Dataset: {dataset_name}",
        )
        figures_displayed += 1
    print(f"Processed: {dataset_name}")

change_summary = pd.DataFrame(summary_rows)
top_spikes = pd.concat(spike_frames, ignore_index=True)

# %%
# Save compact and detailed tables
print("=== Save change tables ===")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
table_paths = {
    "summary": OUTPUT_DIR / "signal_change_summary.csv",
    "top_spikes": OUTPUT_DIR / "top_signal_spikes.csv",
}
change_summary.to_csv(table_paths["summary"], index=False)
top_spikes.to_csv(table_paths["top_spikes"], index=False)
for table_path in table_paths.values():
    print(f"Saved: {table_path}")

# %%
# Compact summary
print("=== Compact summary ===")
print(f"Files processed: {len(loaded_datasets)}")
print(f"Files failed: {len(errors)}")
print(f"Tables saved: {len(table_paths)}")
print(f"Figures displayed: {figures_displayed}")
print(f"Top spike rows saved: {len(top_spikes):,}")
for error in errors:
    print(f"Warning: {error['dataset_name']}: {error['error']}")
