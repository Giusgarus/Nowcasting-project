"""Compute the model-agnostic Perfect Switch oracle on selected test events."""

# %%
# Path setup and imports
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.switching.metrics import (
    compute_dataset_level_switch_summary,
    compute_duration_summary,
    compute_event_level_switch_summary,
    compute_global_switch_summary,
)
from src.switching.plots import plot_perfect_switch_for_event
from src.switching.reference_switch import (
    align_perfect_switch_to_forecast_windows,
    compute_perfect_switch_from_true_signal,
)
from src.utils.config import config_fingerprint, load_yaml_config, save_yaml
from src.utils.paths import project_path
from src.utils.results_paths import (
    SWITCH_REFERENCE_INDEX_COLUMNS,
    ensure_results_subdirs,
    get_perfect_switch_dir,
    get_results_index_dir,
    relative_project_path,
    selection_id_from_metadata,
    upsert_index_row,
)

CONFIG_PATH = PROJECT_ROOT / "configs/perfect_switch.yaml"

OLD_REFERENCE_FUNCTIONS = [
    "threshold_detection_py",
    "ensure_min_island_length_py_matlab_port",
    "run_post_analysis",
]
PLOT_TYPES_FOUND = [
    "single-axis multi-switch overlay in code_output_validation.py",
    "vertical one-subplot-per-method comparison in event_001_switch_window.png",
]
PLOT_TYPES_REPRODUCED = [
    "one-subplot-per-method event comparison with repeated true signal and threshold"
]
PLOT_TYPES_SKIPPED = [
    "Smart baseline overlays: skipped because Smart baseline is not implemented yet.",
    "Model-derived switch comparisons: skipped because model-derived switches are not part of this step.",
]
PERFECT_SWITCH_RULE = (
    "Per event, perfect_switch_raw marks only runs where Signal_prepared > threshold "
    "for switch_time + 1 consecutive points. perfect_switch then applies the legacy "
    "right-only ensure_min_island_length function."
)


def resolve_selection_folder(dataset_root: Path, configured: str) -> Path:
    """Resolve an explicit selection folder or the only available folder."""

    if configured != "auto":
        selected = dataset_root / configured
        if not selected.is_dir():
            raise FileNotFoundError(f"Dataset selection folder not found: {selected}")
        return selected
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    folders = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if len(folders) != 1:
        available = ", ".join(path.name for path in folders) or "none"
        raise ValueError(
            "data.selection_folder='auto' requires exactly one folder; "
            f"available: {available}"
        )
    return folders[0]


# %%
# Load and validate configuration
print("=== Load Perfect Switch configuration ===")
config = load_yaml_config(CONFIG_PATH)
data_config = config["data"]
switch_config = config["switch"]
plot_config = config["plots"]
output_config = config["output"]

event_windows_path = project_path(data_config["event_windows_path"])
window_index_path = project_path(data_config["window_index_path"])
dataset_root = project_path(data_config["dataset_root"])
dataset_folder = resolve_selection_folder(dataset_root, data_config["selection_folder"])
selection_folder = dataset_folder.name
test_metadata_path = dataset_folder / "test_metadata.parquet"
test_npz_path = dataset_folder / "test.npz"
dataset_metadata_path = dataset_folder / "dataset_metadata.yaml"
dataset_metadata = load_yaml_config(dataset_metadata_path)
selection_id = selection_id_from_metadata(dataset_metadata)
output_dir = get_perfect_switch_dir(selection_id)
output_paths = ensure_results_subdirs(output_dir, ("tables", "figures"))
tables_dir = output_paths["tables"]
figures_dir = output_dir / "figures" / "event_plots"

threshold = float(switch_config["threshold"])
switch_time = int(switch_config["switch_time"])
condition = str(switch_config["condition"])
switch_rule = str(switch_config["rule"])
signal_column = str(switch_config["signal_column"])
apply_min_island_length = bool(switch_config["apply_min_island_length"])
if switch_rule != "legacy_persistent_threshold":
    raise ValueError("Only switch.rule=legacy_persistent_threshold is supported.")

# %%
# Run overview
print(
    "=== Analysis overview ===\n"
    "Prints: Perfect Switch rule, selected test data, descriptive totals, and legacy references.\n"
    "Displays: no figures.\n"
    "Saves: model-agnostic test-only Perfect Switch tables, metadata, and event plots.\n"
    "Note: this script does not train models and does not compute Smart baseline.\n"
)

# %%
# Protect outputs from silent overwrite
expected_output_paths = [
    tables_dir / "perfect_switch_timeseries.parquet",
    tables_dir / "perfect_switch_window_targets.parquet",
    tables_dir / "perfect_switch_summary.csv",
    tables_dir / "perfect_switch_by_dataset.csv",
    tables_dir / "perfect_switch_by_event.csv",
    tables_dir / "perfect_switch_duration_summary.csv",
    output_dir / "metadata.yaml",
]
if not output_config["overwrite"]:
    existing = [path for path in expected_output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Perfect Switch outputs already exist under {output_dir}; "
            "set output.overwrite=true to replace them."
        )

# %%
# Load official test metadata and select only its events
print("=== Load selected test windows and event data ===")
if dataset_metadata.get("selection_folder") != selection_folder:
    raise ValueError("dataset_metadata.yaml does not match the resolved selection folder.")
test_metadata = pd.read_parquet(test_metadata_path).reset_index(drop=True)
if test_metadata.empty:
    raise ValueError("The selected final dataset has no test windows.")
if set(test_metadata["split"].unique()) != {"test"}:
    raise ValueError("test_metadata.parquet must contain only the test split.")
test_dataset_names = set(test_metadata["dataset_name"].unique())
if dataset_metadata.get("selection_mode") == "external_holdout":
    heldout_dataset = dataset_metadata.get("heldout_test_dataset")
    if test_dataset_names != {heldout_dataset}:
        raise ValueError(
            "External-holdout test metadata must contain only the held-out dataset."
        )
else:
    if set(dataset_metadata.get("selected_datasets", [])) != test_dataset_names:
        raise ValueError(
            "Selected datasets in dataset_metadata.yaml do not match test metadata."
        )

event_windows = pd.read_parquet(event_windows_path)
test_event_ids = set(test_metadata["event_id"].astype(str))
test_dataset_ids = set(test_metadata["dataset_id"].astype(str))
test_events = event_windows.loc[
    event_windows["event_id"].astype(str).isin(test_event_ids)
    & event_windows["dataset_id"].astype(str).isin(test_dataset_ids)
].copy()
if test_events.empty:
    raise ValueError("No event-window rows match the selected test metadata.")
if set(test_events["event_id"].astype(str)) != test_event_ids:
    missing = sorted(test_event_ids - set(test_events["event_id"].astype(str)))
    raise ValueError(f"Missing event-window data for test events: {missing}")
test_events["split"] = "test"

quality_counts = test_metadata.groupby("event_id")["quality_flag"].nunique()
if quality_counts.gt(1).any():
    raise ValueError("A test event cannot have multiple quality flags.")
quality_lookup = test_metadata.groupby("event_id")["quality_flag"].first()
test_events["quality_flag"] = test_events["event_id"].map(quality_lookup)

# %%
# Compute Perfect Switch independently for every test event
print("=== Compute test-only Perfect Switch ===")
perfect_timeseries = compute_perfect_switch_from_true_signal(
    test_events,
    signal_column=signal_column,
    threshold=threshold,
    condition=condition,
    switch_time=switch_time,
    apply_min_island_length=apply_min_island_length,
)
timeseries_columns = [
    "dataset_id",
    "dataset_name",
    "event_id",
    "split",
    "Time",
    "Signal_true",
    "outage_mask",
    "perfect_switch_raw",
    "perfect_switch_min_time",
    "perfect_switch_adjusted",
    "perfect_switch",
    "threshold",
    "switch_time",
    "quality_flag",
    "relative_time_minutes",
    "window_start",
    "window_end",
    "event_timestamp",
    "event_point_idx",
]
perfect_timeseries = perfect_timeseries[
    [column for column in timeseries_columns if column in perfect_timeseries.columns]
]

# %%
# Align the oracle with every autoregressive test target step
print("=== Align Perfect Switch to test forecast targets ===")
window_targets = align_perfect_switch_to_forecast_windows(
    perfect_timeseries,
    test_metadata,
)
with np.load(test_npz_path) as test_arrays:
    expected_y_raw = test_arrays["y_raw"]
aligned_y_raw = window_targets["y_true_raw"].to_numpy().reshape(expected_y_raw.shape)
if not np.allclose(aligned_y_raw, expected_y_raw, rtol=1e-6, atol=1e-6):
    raise ValueError("Aligned target values do not match test.npz y_raw.")

# %%
# Build descriptive summaries
print("=== Build Perfect Switch summaries ===")
event_summary = compute_event_level_switch_summary(perfect_timeseries)
dataset_summary = compute_dataset_level_switch_summary(perfect_timeseries, event_summary)
duration_summary = compute_duration_summary(perfect_timeseries)
global_summary = compute_global_switch_summary(
    perfect_timeseries,
    selection_folder=selection_id,
    num_windows=len(test_metadata),
    threshold=threshold,
    switch_time=switch_time,
)

# %%
# Save model-independent outputs
print("=== Save Perfect Switch outputs ===")
output_dir.mkdir(parents=True, exist_ok=True)
output_files: dict[str, str] = {}
if output_config["save_tables"]:
    tables = {
        "perfect_switch_timeseries.parquet": perfect_timeseries,
        "perfect_switch_window_targets.parquet": window_targets,
        "perfect_switch_summary.csv": global_summary,
        "perfect_switch_by_dataset.csv": dataset_summary,
        "perfect_switch_by_event.csv": event_summary,
        "perfect_switch_duration_summary.csv": duration_summary,
    }
    for filename, table in tables.items():
        path = tables_dir / filename
        if path.suffix == ".parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False)
        output_files[path.stem] = relative_project_path(path)

# %%
# Reproduce applicable legacy Perfect Switch event plots
print("=== Reproduce applicable legacy Perfect Switch plots ===")
saved_figures: list[str] = []
if output_config["save_plots"] and plot_config["reproduce_old_perfect_switch_plots"]:
    max_events = max(0, int(plot_config["max_events"]))
    event_order = (
        perfect_timeseries.groupby("event_id", sort=False)["event_timestamp"]
        .first()
        .sort_values()
        .index[:max_events]
    )
    for event_id in event_order:
        event_frame = perfect_timeseries.loc[perfect_timeseries["event_id"].eq(event_id)]
        quality_flag = str(event_frame["quality_flag"].iloc[0])
        figure_path = figures_dir / f"{event_id}_perfect_switch_window.png"
        plot_perfect_switch_for_event(
            event_frame,
            figure_path,
            threshold=threshold,
            switch_time=switch_time,
            quality_flag=quality_flag,
        )
        saved_figures.append(relative_project_path(figure_path))

# %%
# Save reproducibility metadata
summary_row = global_summary.iloc[0]
metadata_path = output_dir / "metadata.yaml"
output_files["perfect_switch_metadata"] = relative_project_path(metadata_path)
fingerprint_config = {
    **config,
    "output": {**output_config, "overwrite": False},
}
metadata = {
    "selection_folder": selection_folder,
    "selection_id": selection_id,
    "reference_id": f"perfect_switch_{selection_id}",
    "threshold": threshold,
    "condition": condition,
    "rule": switch_rule,
    "switch_time": switch_time,
    "persistent_threshold_window_points": switch_time + 1,
    "apply_min_island_length": apply_min_island_length,
    "signal_column": signal_column,
    "split": "test",
    "config_path": relative_project_path(CONFIG_PATH),
    "config_fingerprint": config_fingerprint(fingerprint_config),
    "input_files": {
        "event_windows": relative_project_path(event_windows_path),
        "window_index": relative_project_path(window_index_path),
        "test_metadata": relative_project_path(test_metadata_path),
        "test_npz_validation": relative_project_path(test_npz_path),
        "dataset_metadata": relative_project_path(dataset_metadata_path),
    },
    "output_files": output_files,
    "figure_files": saved_figures,
    "old_project_reference_functions": OLD_REFERENCE_FUNCTIONS,
    "plot_types_found": PLOT_TYPES_FOUND,
    "plot_types_reproduced": PLOT_TYPES_REPRODUCED,
    "plot_types_skipped": PLOT_TYPES_SKIPPED,
    "duration_method": (
        "Observed forward timestamp intervals per event; the final sample uses the "
        "event median positive interval."
    ),
    "perfect_switch_rule": PERFECT_SWITCH_RULE,
    "legacy_match_note": (
        "Perfect Switch raw detection and minimum-island processing reproduce "
        "condivisione_files/code_output_validation.py. The later enforce_switch_time "
        "output is a separate adjusted switch and is intentionally not called Perfect Switch."
    ),
    "num_test_events": int(summary_row["num_events"]),
    "num_test_windows": int(summary_row["num_windows"]),
    "num_timeseries_points": int(summary_row["num_points"]),
    "num_positive_raw": int(summary_row["num_positive_raw"]),
    "num_positive_after_min_island": int(
        summary_row["num_positive_after_min_island"]
    ),
    "num_points_changed_by_min_island": int(
        summary_row["num_points_changed_by_min_island"]
    ),
    "num_figures_saved": len(saved_figures),
    "architecture_scope": (
        "Perfect Switch is a model-agnostic and task-agnostic oracle/reference. "
        "The switching utilities are not specific to GRU or only to autoregressive "
        "forecasting. They are intended to support future autoregressive, survival, "
        "shapelet, classification, duration, Smart baseline, and other baseline methods."
    ),
    "created_at": datetime.now(timezone.utc).isoformat(),
}
save_yaml(metadata_path, metadata)
upsert_index_row(
    get_results_index_dir() / "switch_references.csv",
    {
        "reference_id": f"perfect_switch_{selection_id}",
        "reference_type": "perfect_switch",
        "selection_id": selection_id,
        "threshold": threshold,
        "switch_time": switch_time,
        "results_path": relative_project_path(output_dir),
        "timeseries_path": relative_project_path(
            tables_dir / "perfect_switch_timeseries.parquet"
        ),
        "window_targets_path": relative_project_path(
            tables_dir / "perfect_switch_window_targets.parquet"
        ),
        "status": "complete",
        "created_at": metadata["created_at"],
    },
    id_column="reference_id",
    columns=SWITCH_REFERENCE_INDEX_COLUMNS,
)

# %%
# Compact final report
print("=== Compact final report ===")
print(f"Selection folder: {selection_folder}")
print(f"Selection ID: {selection_id}")
print(f"Test events: {int(summary_row['num_events']):,}")
print(f"Test windows: {int(summary_row['num_windows']):,}")
print(
    f"Perfect Switch raw: Signal_true > {threshold:g} for "
    f"{switch_time + 1} consecutive points"
)
print(f"Minimum island length: {switch_time} samples, extended only to the right")
print(f"Positive points raw: {int(summary_row['num_positive_raw']):,}")
print(
    "Positive points after min-island: "
    f"{int(summary_row['num_positive_after_min_island']):,}"
)
print(
    "Points changed by min-island: "
    f"{int(summary_row['num_points_changed_by_min_island']):,}"
)
print(f"Output folder: {output_dir}")
print(f"Figures saved: {len(saved_figures):,} under {figures_dir}")
print(f"Old plotting files/functions found: {', '.join(OLD_REFERENCE_FUNCTIONS)}")
print(f"Old plot types reproduced: {'; '.join(PLOT_TYPES_REPRODUCED)}")
print(f"Old plot types skipped: {'; '.join(PLOT_TYPES_SKIPPED)}")
print(f"Perfect Switch rule: {PERFECT_SWITCH_RULE}")
