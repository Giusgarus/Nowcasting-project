"""One-off sanity checks for final supervised autoregressive datasets."""

# %%
# Configuration and imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Set this when multiple selection folders are available.
SELECTION_FOLDER: str | None = "externalHoldout_test_fc_uplink_fade"
DATASET_ROOT = (
    PROJECT_ROOT
    / "data/processed/autoregressive/threshold_10p0/datasets_L30_h10"
)
SHOW_PLOTS = True
PLOTS_PER_SPLIT = 5
RANDOM_SEED = 42

SPLITS = ("train", "val", "test")
REQUIRED_ARRAY_KEYS = {
    "X_raw",
    "y_raw",
    "X_context_standard",
    "y_context_standard",
    "scaling_mean",
    "scaling_std",
    "window_id",
    "event_id",
    "dataset_id",
}
REQUIRED_METADATA_COLUMNS = {
    "window_id",
    "event_id",
    "dataset_id",
    "dataset_name",
    "split",
    "quality_flag",
    "input_start_time",
    "input_end_time",
    "target_start_time",
    "target_end_time",
    "context_length",
    "prediction_length",
    "num_imputed_points_in_window",
    "scaling_mean",
    "scaling_std",
    "normalization_method",
}
METADATA_SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}


def find_selection_folder(root: Path, configured: str | None) -> Path:
    """Resolve the requested selection folder or the only available folder."""

    if configured:
        selected = root / configured
        if not selected.is_dir():
            raise FileNotFoundError(f"Selection folder not found: {selected}")
        return selected

    folders = sorted(path for path in root.iterdir() if path.is_dir())
    if len(folders) == 1:
        return folders[0]
    if not folders:
        raise FileNotFoundError(f"No final dataset selection folders found under {root}")
    print("Multiple selection folders are available:")
    for folder in folders:
        print(f"  - {folder.name}")
    raise ValueError("Set SELECTION_FOLDER at the top of this script.")


def assert_shape(name: str, array: np.ndarray, expected: tuple[int, ...]) -> None:
    """Raise a clear error when an array shape differs from expectation."""

    if array.shape != expected:
        raise ValueError(f"{name} shape is {array.shape}; expected {expected}.")


def print_array_value_summary(name: str, array: np.ndarray) -> tuple[int, int]:
    """Print basic finite-value statistics and return NaN/inf counts."""

    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())
    if not array.size:
        print(f"  {name}: empty array, NaNs=0, infs=0")
        return nan_count, inf_count
    print(
        f"  {name}: min={array.min():.6g}, max={array.max():.6g}, "
        f"mean={array.mean():.6g}, std={array.std():.6g}, "
        f"NaNs={nan_count:,}, infs={inf_count:,}"
    )
    if nan_count or inf_count:
        raise ValueError(f"{name} contains NaN or infinite values.")
    return nan_count, inf_count


def check_split(
    dataset_folder: Path,
    split: str,
    *,
    context_length: int,
    prediction_length: int,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, float | int]]:
    """Validate one split's arrays, metadata, normalization, and values."""

    arrays = dict(np.load(dataset_folder / f"{split}.npz"))
    metadata = pd.read_parquet(dataset_folder / f"{split}_metadata.parquet")
    print(f"\n=== {split.upper()} ===")
    print(f"Available NPZ keys: {sorted(arrays)}")

    missing_keys = sorted(REQUIRED_ARRAY_KEYS - set(arrays))
    if missing_keys:
        raise ValueError(f"{split}.npz is missing required keys: {missing_keys}")
    count = len(arrays["X_raw"])
    expected_shapes = {
        "X_raw": (count, context_length, 1),
        "y_raw": (count, prediction_length),
        "X_context_standard": (count, context_length, 1),
        "y_context_standard": (count, prediction_length),
        "scaling_mean": (count,),
        "scaling_std": (count,),
        "window_id": (count,),
        "event_id": (count,),
        "dataset_id": (count,),
    }
    for key, shape in expected_shapes.items():
        assert_shape(f"{split}.{key}", arrays[key], shape)
    print(f"Array shapes: passed ({count:,} windows)")

    missing_columns = sorted(REQUIRED_METADATA_COLUMNS - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"{split} metadata is missing columns: {missing_columns}")
    if len(metadata) != count:
        raise ValueError(f"{split} metadata rows do not match NPZ windows.")
    for identifier in ("window_id", "event_id", "dataset_id"):
        if not np.array_equal(
            metadata[identifier].astype(str).to_numpy(),
            arrays[identifier].astype(str),
        ):
            raise ValueError(f"{split} metadata {identifier} is not aligned with NPZ.")
    for identifier in ("global_window_id", "global_event_id"):
        if identifier in metadata.columns and identifier in arrays:
            if not np.array_equal(
                metadata[identifier].astype(str).to_numpy(),
                arrays[identifier].astype(str),
            ):
                raise ValueError(
                    f"{split} metadata {identifier} is not aligned with NPZ."
                )
    expected_split = METADATA_SPLIT_NAMES[split]
    if not metadata["split"].eq(expected_split).all():
        raise ValueError(f"{split} metadata contains incorrect split labels.")
    print("Metadata alignment: passed")

    quality_counts = metadata["quality_flag"].value_counts().to_dict()
    print(f"Quality flags: {quality_counts}")
    if metadata["quality_flag"].eq("unusable").any():
        raise ValueError(f"{split} contains unusable windows.")

    context_means = arrays["X_context_standard"].mean(axis=(1, 2))
    context_stds = arrays["X_context_standard"].std(axis=(1, 2))
    raw_stds = arrays.get("scaling_std_raw", arrays["scaling_std"])
    near_zero_count = int((raw_stds <= epsilon).sum())
    if count:
        print(
            "Context-standard means: "
            f"mean={context_means.mean():.6g}, std={context_means.std():.6g}, "
            f"min={context_means.min():.6g}, max={context_means.max():.6g}"
        )
        print(
            "Context-standard stds: "
            f"mean={context_stds.mean():.6g}, std={context_stds.std():.6g}, "
            f"min={context_stds.min():.6g}, max={context_stds.max():.6g}"
        )
    else:
        print("Context-standard means/stds: split is empty")
    print(f"Near-zero raw context std windows: {near_zero_count:,}")

    sample_size = min(100, count)
    sample_indices = rng.choice(count, size=sample_size, replace=False)
    means = arrays["scaling_mean"][sample_indices]
    stds = arrays["scaling_std"][sample_indices]
    reconstructed_y = (
        arrays["y_context_standard"][sample_indices] * stds[:, None]
        + means[:, None]
    )
    reconstructed_X = (
        arrays["X_context_standard"][sample_indices] * stds[:, None, None]
        + means[:, None, None]
    )
    y_error = float(
        np.max(np.abs(reconstructed_y - arrays["y_raw"][sample_indices]))
    ) if sample_size else 0.0
    X_error = float(
        np.max(np.abs(reconstructed_X - arrays["X_raw"][sample_indices]))
    ) if sample_size else 0.0
    print(f"Maximum inverse-transform error: X={X_error:.6g}, y={y_error:.6g}")
    if not np.allclose(
        reconstructed_y,
        arrays["y_raw"][sample_indices],
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(f"{split} y inverse-transform check failed.")
    if not np.allclose(
        reconstructed_X,
        arrays["X_raw"][sample_indices],
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(f"{split} X inverse-transform check failed.")

    nan_total = 0
    inf_total = 0
    print("Value summaries:")
    for key in ("X_raw", "y_raw", "X_context_standard", "y_context_standard"):
        nan_count, inf_count = print_array_value_summary(key, arrays[key])
        nan_total += nan_count
        inf_total += inf_count

    print("Counts by dataset and quality:")
    print(
        metadata.groupby(["dataset_name", "quality_flag"])
        .size()
        .rename("num_windows")
        .to_string()
    )
    print(
        f"Unique events={metadata.get('global_event_id', metadata['event_id']).nunique():,}, "
        f"windows={metadata.get('global_window_id', metadata['window_id']).nunique():,}, "
        f"datasets={metadata['dataset_id'].nunique():,}"
    )
    return arrays, metadata, {
        "windows": count,
        "near_zero": near_zero_count,
        "nan_count": nan_total,
        "inf_count": inf_total,
        "warnings": int(metadata["quality_flag"].eq("warning").sum()),
        "unusable": int(metadata["quality_flag"].eq("unusable").sum()),
    }


def plot_sample_windows(
    arrays_by_split: dict[str, dict[str, np.ndarray]],
    metadata_by_split: dict[str, pd.DataFrame],
    *,
    plots_per_split: int,
    rng: np.random.Generator,
) -> None:
    """Display raw and context-standard views for a small random sample."""

    for split in SPLITS:
        arrays = arrays_by_split[split]
        metadata = metadata_by_split[split]
        sample_size = min(plots_per_split, len(metadata))
        for index in rng.choice(len(metadata), size=sample_size, replace=False):
            row = metadata.iloc[index]
            context_length = arrays["X_raw"].shape[1]
            raw = np.concatenate((arrays["X_raw"][index, :, 0], arrays["y_raw"][index]))
            standardized = np.concatenate(
                (
                    arrays["X_context_standard"][index, :, 0],
                    arrays["y_context_standard"][index],
                )
            )
            title = (
                f"{split} | {row.window_id} | {row.event_id}\n"
                f"{row.dataset_name} | {row.quality_flag}"
            )
            figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            axes[0].plot(raw)
            axes[0].axvline(context_length - 0.5, color="black", linestyle="--")
            axes[0].set_title(f"Raw | {title}")
            axes[0].set_ylabel("Signal")
            axes[1].plot(standardized)
            axes[1].axvline(context_length - 0.5, color="black", linestyle="--")
            axes[1].set_title("Context-standard")
            axes[1].set_ylabel("Standardized Signal")
            axes[1].set_xlabel("Sample index")
            figure.tight_layout()
    plt.show()
    plt.close("all")


# %%
# Resolve dataset folder and metadata
print("=== Final autoregressive dataset sanity check ===")
selection_path = find_selection_folder(DATASET_ROOT, SELECTION_FOLDER)
with (selection_path / "dataset_metadata.yaml").open(encoding="utf-8") as stream:
    dataset_metadata = yaml.safe_load(stream)
context_length = int(dataset_metadata.get("context_length", 30))
prediction_length = int(dataset_metadata.get("prediction_length", 10))
epsilon = float(dataset_metadata.get("normalization_epsilon", 1.0e-6))
print(f"Selection folder: {selection_path.name}")
print(f"Context length: {context_length}")
print(f"Prediction length: {prediction_length}")
print(f"Normalization epsilon: {epsilon}")

# %%
# Validate all splits
rng = np.random.default_rng(RANDOM_SEED)
arrays_by_split = {}
metadata_by_split = {}
reports = {}
for split_name in SPLITS:
    arrays, metadata, report = check_split(
        selection_path,
        split_name,
        context_length=context_length,
        prediction_length=prediction_length,
        epsilon=epsilon,
        rng=rng,
    )
    arrays_by_split[split_name] = arrays
    metadata_by_split[split_name] = metadata
    reports[split_name] = report

# %%
# Validate global split integrity and uniqueness
all_metadata = pd.concat(metadata_by_split.values(), ignore_index=True)
event_identity = "global_event_id" if "global_event_id" in all_metadata else "event_id"
window_identity = (
    "global_window_id" if "global_window_id" in all_metadata else "window_id"
)
leaking_events = all_metadata.groupby(event_identity)["split"].nunique()
leaking_events = leaking_events.loc[leaking_events.gt(1)]
if len(leaking_events):
    print(f"Events found in multiple splits:\n{leaking_events.to_string()}")
    raise ValueError("Event split leakage detected.")
if all_metadata[window_identity].duplicated().any():
    duplicates = all_metadata.loc[
        all_metadata[window_identity].duplicated(keep=False), window_identity
    ].unique()
    raise ValueError(f"Duplicate window IDs found: {duplicates.tolist()}")
print("\nGlobal event split integrity: passed")
print("Global window ID uniqueness: passed")

# %%
# Optional interactive plots
if SHOW_PLOTS:
    plot_sample_windows(
        arrays_by_split,
        metadata_by_split,
        plots_per_split=PLOTS_PER_SPLIT,
        rng=rng,
    )

# %%
# Compact final report
total_warnings = sum(report["warnings"] for report in reports.values())
total_unusable = sum(report["unusable"] for report in reports.values())
total_nans = sum(report["nan_count"] for report in reports.values())
total_infs = sum(report["inf_count"] for report in reports.values())
print("\nDataset sanity check completed.\n")
print(f"Selection folder: {selection_path.name}")
print(f"Train windows: {reports['train']['windows']:,}")
print(f"Val windows: {reports['val']['windows']:,}")
print(f"Test windows: {reports['test']['windows']:,}")
print(f"Unique events: {all_metadata[event_identity].nunique():,}")
print(f"Unique datasets: {all_metadata['dataset_id'].nunique():,}")
print(f"Warnings included: {'yes' if total_warnings else 'no'}")
print(f"Unusable windows present: {'yes' if total_unusable else 'no'}")
print(f"NaNs present: {'yes' if total_nans else 'no'}")
print(f"Inf values present: {'yes' if total_infs else 'no'}")
print("Normalization reconstruction: passed")
print("Event split leakage: passed")
print("Window ID uniqueness: passed")
print("\nAll sanity checks passed.")
