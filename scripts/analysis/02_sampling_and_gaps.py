"""Analyze sampling intervals and temporal gaps for all raw datasets."""

# %%
# Path setup and constants
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.data_quality import (
    compute_sampling_summary,
    compute_time_differences,
)
from src.analysis.plots import plot_sampling_histogram
from src.data.loading import load_signal_dataset

RAW_DIR = PROJECT_ROOT / "data/raw"
GAP_THRESHOLD_SECONDS = 90
TABLE_PATH = PROJECT_ROOT / "results/data_analysis/sampling_and_gaps_summary.csv"

# %%
# Load, compute, and plot
print("=== Sampling and gap analysis ===")
rows = []
for dataset_path in sorted(RAW_DIR.glob("*.csv")):
    dataframe, metadata = load_signal_dataset(dataset_path)
    summary = compute_sampling_summary(dataframe, GAP_THRESHOLD_SECONDS)
    rows.append({"dataset_name": dataset_path.name, "valid_rows": len(dataframe), **summary})
    plot_sampling_histogram(
        compute_time_differences(dataframe),
        f"Sampling interval distribution | Dataset: {dataset_path.name}",
    )
    print(
        f"{dataset_path.name}: median dt={summary['median_dt_seconds']:.2f}s, "
        f"segments={summary['num_segments']:,}"
    )

sampling_summary = pd.DataFrame(rows)

# %%
# Save summary table
print("=== Save sampling summary ===")
TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
sampling_summary.to_csv(TABLE_PATH, index=False)
print(f"Saved: {TABLE_PATH}")
