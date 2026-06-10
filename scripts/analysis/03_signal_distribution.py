"""Compare Signal distributions across all raw datasets."""

# %%
# Path setup and constants
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.data_quality import compute_signal_distribution_summary
from src.analysis.plots import plot_combined_boxplot, plot_signal_histogram
from src.data.loading import load_signal_dataset

RAW_DIR = PROJECT_ROOT / "data/raw"
TABLE_PATH = PROJECT_ROOT / "results/data_analysis/signal_distribution_summary.csv"

# %%
# Load and compute distributions
print("=== Signal distribution analysis ===")
rows = []
signals = {}
for dataset_path in sorted(RAW_DIR.glob("*.csv")):
    dataframe, _ = load_signal_dataset(dataset_path)
    rows.append(
        {
            "dataset_name": dataset_path.name,
            **compute_signal_distribution_summary(dataframe),
        }
    )
    signals[dataset_path.name] = dataframe["Signal"]
    plot_signal_histogram(
        dataframe,
        f"Signal value distribution | Dataset: {dataset_path.name}",
    )
    print(f"Processed: {dataset_path.name}")

distribution_summary = pd.DataFrame(rows)

# %%
# Save tables and combined comparison
print("=== Save distribution outputs ===")
TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
distribution_summary.to_csv(TABLE_PATH, index=False)
plot_combined_boxplot(
    signals,
    "Signal distribution comparison across all datasets",
)
print(f"Saved: {TABLE_PATH}")
