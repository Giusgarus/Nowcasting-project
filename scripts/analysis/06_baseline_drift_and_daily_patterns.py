"""Analyze Signal baseline drift and calendar-time patterns."""

# %%
# Path setup and constants
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.data_quality import (
    compute_daily_signal_summary,
    compute_hourly_signal_summary,
    compute_rolling_baseline_summary,
)
from src.analysis.plots import (
    plot_hourly_pattern,
    plot_rolling_std,
    plot_signal_with_rolling_means,
)
from src.data.loading import load_signal_dataset

RAW_DIR = PROJECT_ROOT / "data/raw"
OUTPUT_DIR = PROJECT_ROOT / "results/data_analysis"
ROLLING_WINDOWS = {"short": 120, "medium": 720, "long": 2880}
ROLLING_STD_WINDOWS = {"short": 120, "medium": 720}

# %%
# Run overview
print(
    "=== Analysis overview ===\n"
    "Prints: per-dataset outcomes and file, table, and insufficient-window totals.\n"
    "Displays: Signal with rolling means, rolling standard deviations, and hourly "
    "patterns by dataset.\n"
    f"Saves: three compact tables under {OUTPUT_DIR}.\n"
    "Note: rolling windows are measured in samples, not exact time durations.\n"
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
# Compute summaries and display figures
print("=== Compute baseline, drift, and calendar summaries ===")
baseline_frames = []
hourly_frames = []
daily_frames = []
figures_displayed = 0
for dataset_name, dataframe in loaded_datasets:
    # Rolling summaries describe drift without modifying or detrending Signal.
    baseline = compute_rolling_baseline_summary(dataframe, ROLLING_WINDOWS)
    baseline.insert(0, "dataset_name", dataset_name)
    baseline_frames.append(baseline)

    # Calendar summaries aggregate all observations sharing the same hour or date.
    hourly = compute_hourly_signal_summary(dataframe)
    hourly.insert(0, "dataset_name", dataset_name)
    hourly_frames.append(hourly)

    daily = compute_daily_signal_summary(dataframe)
    daily.insert(0, "dataset_name", dataset_name)
    daily_frames.append(daily)

    plot_signal_with_rolling_means(
        dataframe,
        ROLLING_WINDOWS,
        f"Signal with rolling baseline means | Dataset: {dataset_name}",
    )
    plot_rolling_std(
        dataframe,
        ROLLING_STD_WINDOWS,
        f"Rolling Signal variability | Dataset: {dataset_name}",
    )
    plot_hourly_pattern(
        hourly,
        f"Mean and median Signal by hour | Dataset: {dataset_name}",
    )
    figures_displayed += 3
    print(f"Processed: {dataset_name}")

baseline_summary = pd.concat(baseline_frames, ignore_index=True)
hourly_summary = pd.concat(hourly_frames, ignore_index=True)
daily_summary = pd.concat(daily_frames, ignore_index=True)

# %%
# Save compact tables
print("=== Save compact tables ===")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
table_paths = {
    "baseline": OUTPUT_DIR / "baseline_drift_summary.csv",
    "hourly": OUTPUT_DIR / "hourly_signal_patterns.csv",
    "daily": OUTPUT_DIR / "daily_signal_summary.csv",
}
baseline_summary.to_csv(table_paths["baseline"], index=False)
hourly_summary.to_csv(table_paths["hourly"], index=False)
daily_summary.to_csv(table_paths["daily"], index=False)
for table_path in table_paths.values():
    print(f"Saved: {table_path}")

# %%
# Compact summary
print("=== Compact summary ===")
print(f"Files processed: {len(loaded_datasets)}")
print(f"Files failed: {len(errors)}")
print(f"Tables saved: {len(table_paths)}")
print(f"Figures displayed: {figures_displayed}")
print(
    "Insufficient rolling windows: "
    f"{int(baseline_summary['status'].eq('insufficient_points').sum())}"
)
for error in errors:
    print(f"Warning: {error['dataset_name']}: {error['error']}")
