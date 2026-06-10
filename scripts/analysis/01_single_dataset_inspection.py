"""Interactively inspect one raw signal-only dataset."""

# %%
# Path setup and editable constants
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plots import plot_signal_over_time, plot_signal_zoom
from src.data.loading import load_signal_dataset

DATASET_PATH = PROJECT_ROOT / "data/raw/fc-uplink-fade.csv"
ZOOM_OBSERVATIONS = 500

# %%
# Run overview
print(
    "=== Analysis overview ===\n"
    "Prints: metadata, initial/final rows, statistics, and largest consecutive "
    "change.\n"
    "Displays: full timeline, initial observations, and largest-change zoom.\n"
    "Saves: no files.\n"
)

# %%
# Load selected dataset
print("=== Load selected dataset ===")
dataframe, metadata = load_signal_dataset(DATASET_PATH)

# %%
# Inspect metadata and rows
print("=== Loader metadata ===")
for key, value in metadata.items():
    print(f"{key}: {value}")

print("\n=== Head ===")
print(dataframe.head())
print("\n=== Tail ===")
print(dataframe.tail())
print("\n=== Describe ===")
print(dataframe.describe(include="all"))

print("\n=== Compact temporal summary ===")
print(f"First timestamp: {dataframe['Time'].min()}")
print(f"Last timestamp: {dataframe['Time'].max()}")
print(f"Valid rows: {len(dataframe):,}")
print(f"Duplicate timestamps: {metadata['duplicate_timestamps']:,}")
print(f"Invalid timestamp rows: {metadata['invalid_timestamp_rows']:,}")
print(f"Invalid signal rows: {metadata['invalid_signal_rows']:,}")

# %%
# Locate largest absolute Signal change
print("=== Locate largest absolute Signal change ===")
# This diagnostic uses consecutive parse-valid rows; inspect sampling gaps separately.
largest_change_position = int(dataframe["Signal"].diff().abs().fillna(0).to_numpy().argmax())
zoom_start = max(0, largest_change_position - ZOOM_OBSERVATIONS // 2)
zoom_stop = min(len(dataframe), largest_change_position + ZOOM_OBSERVATIONS // 2)
print(f"Largest-change row position: {largest_change_position:,}")

# %%
# Display figures
print("=== Display figures ===")
plot_signal_over_time(
    dataframe,
    f"Full Signal timeline | Dataset: {DATASET_PATH.name}",
)
plot_signal_zoom(
    dataframe,
    0,
    ZOOM_OBSERVATIONS,
    f"Initial {ZOOM_OBSERVATIONS} observations | Dataset: {DATASET_PATH.name}",
)
plot_signal_zoom(
    dataframe,
    zoom_start,
    zoom_stop,
    f"Zoom around largest absolute Signal change | Dataset: {DATASET_PATH.name}",
)
