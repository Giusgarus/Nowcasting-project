"""Estimate autoregressive window availability without building datasets."""

# %%
# Path setup and constants
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.data_quality import count_valid_windows_by_segments
from src.data.loading import load_signal_dataset

RAW_DIR = PROJECT_ROOT / "data/raw"
CONTEXT_LENGTHS = [60, 120]
PREDICTION_LENGTHS = [20, 30]
GAP_THRESHOLD_SECONDS = 90
OUTPUT_PATH = PROJECT_ROOT / "results/data_analysis/window_availability_summary.csv"

# %%
# Count windows
print("=== Window availability ===")
rows = []
for dataset_path in sorted(RAW_DIR.glob("*.csv")):
    dataframe, _ = load_signal_dataset(dataset_path)
    summary = count_valid_windows_by_segments(
        dataframe,
        CONTEXT_LENGTHS,
        PREDICTION_LENGTHS,
        GAP_THRESHOLD_SECONDS,
    )
    rows.append({"dataset_name": dataset_path.name, **summary})
    print(
        f"{dataset_path.name}: rows={len(dataframe):,}, "
        f"segments={summary['num_segments']:,}"
    )

window_summary = pd.DataFrame(rows)

# %%
# Save table
print("=== Save window availability ===")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
window_summary.to_csv(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH}")
