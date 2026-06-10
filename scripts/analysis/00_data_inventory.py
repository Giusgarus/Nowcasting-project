"""Create an inventory of every raw signal-only dataset."""

# %%
# Path setup and constants
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.data_quality import summarize_signal_dataframe
from src.data.loading import load_signal_dataset

RAW_DIR = PROJECT_ROOT / "data/raw"
OUTPUT_PATH = PROJECT_ROOT / "results/tables/data_analysis/data_inventory_summary.csv"
SUSPECTED_SENTINEL_THRESHOLD = -10.0

# %%
# Discover raw datasets
print("=== Discover raw datasets ===")
dataset_paths = sorted(RAW_DIR.glob("*.csv"))
print(f"Files found: {len(dataset_paths)}")

# %%
# Load and summarize
print("=== Load and summarize ===")
rows = []
for dataset_path in dataset_paths:
    try:
        dataframe, metadata = load_signal_dataset(
            dataset_path,
            suspected_sentinel_threshold=SUSPECTED_SENTINEL_THRESHOLD,
        )
        summary = summarize_signal_dataframe(dataframe, metadata)
        summary.update({"load_status": "success", "error_message": ""})
        print(f"Loaded: {dataset_path.name} ({len(dataframe):,} valid rows)")
    except Exception as error:
        summary = {
            "dataset_name": dataset_path.name,
            "file_path": str(dataset_path),
            "load_status": "failed",
            "error_message": str(error),
        }
        print(f"Failed: {dataset_path.name}: {error}")
    rows.append(summary)

inventory = pd.DataFrame(rows)

# %%
# Save table
print("=== Save inventory ===")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
inventory.to_csv(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH}")

# %%
# Compact summary
print("=== Compact summary ===")
successful = inventory["load_status"].eq("success")


def count_datasets_with(column: str) -> int:
    """Count inventory rows whose requested quality count is positive."""

    values = inventory.get(column, pd.Series(index=inventory.index, dtype=float))
    return int(values.fillna(0).gt(0).sum())


print(f"Number of files found: {len(inventory)}")
print(f"Number successfully loaded: {int(successful.sum())}")
print(f"Number failed: {int((~successful).sum())}")
print(
    "Total parse-valid rows, including suspected sentinels: "
    f"{int(inventory.get('valid_rows', pd.Series(dtype=float)).sum()):,}"
)
print(
    "Datasets with duplicate timestamps: "
    f"{count_datasets_with('duplicate_timestamps')}"
)
print(
    "Datasets with invalid timestamps: "
    f"{count_datasets_with('invalid_timestamp_rows')}"
)
print(
    "Datasets with missing/non-numeric/non-finite signals: "
    f"{count_datasets_with('invalid_signal_rows')}"
)
print(
    f"Datasets with suspected Signal sentinels <= {SUSPECTED_SENTINEL_THRESHOLD:g}: "
    f"{count_datasets_with('suspected_sentinel_signal_rows')}"
)
print(
    "Datasets with invalid or suspected-sentinel rows: "
    f"{count_datasets_with('total_flagged_rows')}"
)

# %%
# Detailed quality issues by dataset
print("=== Detailed quality issues by dataset ===")
reason_columns = (
    ("missing_timestamp_rows", "missing timestamp"),
    ("unparseable_timestamp_rows", "unparseable timestamp"),
    ("missing_signal_rows", "missing Signal"),
    ("non_numeric_signal_rows", "non-numeric Signal"),
    ("non_finite_signal_rows", "non-finite Signal"),
    (
        "suspected_sentinel_signal_rows",
        f"suspected Signal sentinel <= {SUSPECTED_SENTINEL_THRESHOLD:g}",
    ),
    ("duplicate_timestamps", "duplicate timestamp"),
)
flagged_inventory = inventory[
    inventory.get("total_flagged_rows", pd.Series(index=inventory.index, dtype=float))
    .fillna(0)
    .gt(0)
]
if flagged_inventory.empty:
    print("No quality issues found.")
else:
    for _, row in flagged_inventory.iterrows():
        print(f"\n{row['dataset_name']}:")
        for column, label in reason_columns:
            count = int(row.get(column, 0) or 0)
            if count:
                print(f"  - {label}: {count:,}")
        print(f"  - total distinct flagged rows: {int(row['total_flagged_rows']):,}")
