"""Compare diagnostic Signal normalization variants."""

# %%
# Path setup and constants
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.data_quality import compute_normalization_diagnostics
from src.analysis.plots import (
    plot_normalization_histogram,
    plot_normalization_timeseries,
)
from src.data.loading import load_signal_dataset

RAW_DIR = PROJECT_ROOT / "data/raw"
OUTPUT_PATH = PROJECT_ROOT / "results/data_analysis/normalization_diagnostics_summary.csv"
ROLLING_WINDOW = 720
TIMESERIES_TRANSFORMATIONS = (
    "raw_signal",
    "zscore_per_dataset",
    "robust_zscore_per_dataset",
    "rolling_centered_signal",
)
HISTOGRAM_TRANSFORMATIONS = ("raw_signal", "robust_zscore_per_dataset")

# %%
# Run overview
print(
    "=== Analysis overview ===\n"
    "Prints: per-dataset outcomes and file, table, and displayed-figure totals.\n"
    "Displays: raw/normalized time series and comparison histograms by dataset.\n"
    f"Saves: summary statistics only in {OUTPUT_PATH}.\n"
    "Note: transformed series are not saved and no final normalization is selected.\n"
)

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
# Compute normalization diagnostics and display figures
print("=== Compute normalization diagnostics ===")
summary_frames = []
figures_displayed = 0
for dataset_name, dataframe in loaded_datasets:
    # Global diagnostics use the complete dataset and must not become model inputs.
    transformed, summary = compute_normalization_diagnostics(
        dataframe,
        ROLLING_WINDOW,
    )
    summary.insert(0, "dataset_name", dataset_name)
    summary_frames.append(summary)

    for transformation in TIMESERIES_TRANSFORMATIONS:
        plot_normalization_timeseries(
            transformed,
            transformation,
            f"{transformation} over time | Dataset: {dataset_name}",
        )
        figures_displayed += 1
    for transformation in HISTOGRAM_TRANSFORMATIONS:
        plot_normalization_histogram(
            transformed,
            transformation,
            f"{transformation} distribution | Dataset: {dataset_name}",
        )
        figures_displayed += 1
    print(f"Processed: {dataset_name}")

normalization_summary = pd.concat(summary_frames, ignore_index=True)

# %%
# Save compact table
print("=== Save normalization table ===")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
normalization_summary.to_csv(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH}")

# %%
# Compact summary
print("=== Compact summary ===")
print(f"Files processed: {len(loaded_datasets)}")
print(f"Files failed: {len(errors)}")
print("Tables saved: 1")
print(f"Figures displayed: {figures_displayed}")
print("Final normalization selected: no")
for error in errors:
    print(f"Warning: {error['dataset_name']}: {error['error']}")
