"""Explore quantile-based Signal runs without defining final events."""

# %%
# Path setup and constants
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.data_quality import (
    compute_quantile_run_summary,
    longest_quantile_run,
)
from src.analysis.plots import plot_signal_with_quantile_lines, plot_signal_zoom
from src.data.loading import load_signal_dataset

RAW_DIR = PROJECT_ROOT / "data/raw"
ZOOM_PADDING = 200
TABLE_PATH = PROJECT_ROOT / "results/data_analysis/candidate_event_quantile_summary.csv"

# %%
# Run overview
print(
    "=== Analysis overview ===\n"
    "Prints: processing progress and final table path.\n"
    "Displays: q05/q95 thresholds and zooms on the longest extreme runs.\n"
    f"Saves: {TABLE_PATH}.\n"
    "Note: this does not define final fade events.\n"
)

# %%
# Compute exploratory quantile-run summaries
print("=== Exploratory quantile-based candidate events ===")
summary_frames = []
for dataset_path in sorted(RAW_DIR.glob("*.csv")):
    dataframe, _ = load_signal_dataset(dataset_path)

    # Quantile thresholds are dataset-relative and are not operational event rules.
    summary = compute_quantile_run_summary(dataframe)
    summary.insert(0, "dataset_name", dataset_path.name)
    summary_frames.append(summary)

    plot_signal_with_quantile_lines(
        dataframe,
        f"Exploratory q05/q95 thresholds over time | Dataset: {dataset_path.name}",
    )

    for direction, quantile, description in (
        ("above", 0.95, "Longest exploratory run above q95"),
        ("below", 0.05, "Longest exploratory run below q05"),
    ):
        run = longest_quantile_run(dataframe, quantile, direction)
        if run is None:
            continue
        start, stop = run
        plot_signal_zoom(
            dataframe,
            start - ZOOM_PADDING,
            stop + ZOOM_PADDING,
            f"{description} | Dataset: {dataset_path.name}",
        )

    print(f"Processed: {dataset_path.name}")

candidate_summary = pd.concat(summary_frames, ignore_index=True)

# %%
# Save exploratory table
print("=== Save candidate-event exploration ===")
TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
candidate_summary.to_csv(TABLE_PATH, index=False)
print(f"Saved: {TABLE_PATH}")
