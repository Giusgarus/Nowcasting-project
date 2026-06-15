"""Stable identifiers, result paths, compatibility resolvers, and CSV indexes."""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import PROJECT_ROOT

RUN_INDEX_COLUMNS = [
    "run_id",
    "model_family",
    "architecture",
    "variant",
    "selection_id",
    "dataset_selection",
    "threshold",
    "context_length",
    "prediction_length",
    "results_path",
    "model_path",
    "predictions_path",
    "metrics_path",
    "status",
    "created_at",
]
SWITCH_REFERENCE_INDEX_COLUMNS = [
    "reference_id",
    "reference_type",
    "selection_id",
    "threshold",
    "switch_time",
    "results_path",
    "timeseries_path",
    "window_targets_path",
    "status",
    "created_at",
]
COMPARISON_INDEX_COLUMNS = [
    "comparison_id",
    "method_id",
    "reference_id",
    "comparison_type",
    "selection_id",
    "results_path",
    "metrics_path",
    "figures_path",
    "status",
    "created_at",
]
DATASET_INDEX_COLUMNS = [
    "selection_id",
    "dataset_selection_mode",
    "selected_datasets",
    "threshold",
    "context_length",
    "prediction_length",
    "dataset_path",
    "num_train_windows",
    "num_val_windows",
    "num_test_windows",
    "created_at",
]
GRID_SEARCH_INDEX_COLUMNS = [
    "search_id",
    "model_family",
    "target_run_id",
    "selection_id",
    "selection_metric",
    "best_trial_id",
    "results_path",
    "best_model_path",
    "status",
    "created_at",
]


def sanitize_id(value: str) -> str:
    """Return a compact filesystem-safe identifier component."""

    text = re.sub(r"\.[^.]+$", "", str(value).strip())
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text or "unnamed"


def _number_token(value: float | int) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}".replace("-", "m").replace(".", "p")


def variant_id(variant: str) -> str:
    """Convert configuration variant names to compact stable ID components."""

    parts = sanitize_id(variant).split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def make_selection_id(
    *,
    mode: str,
    context_length: int,
    prediction_length: int,
    threshold: float,
    selected_datasets: Sequence[str] = (),
) -> str:
    """Build a compact selection ID from dataset-selection metadata."""

    if mode == "auto_largest":
        prefix = "autoLargest"
    elif mode == "single_dataset":
        if len(selected_datasets) != 1:
            raise ValueError("single_dataset selection IDs require one dataset.")
        prefix = f"single_{sanitize_id(selected_datasets[0])}"
    elif mode == "multiple_datasets":
        if len(selected_datasets) < 2:
            raise ValueError("multiple_datasets selection IDs require at least two datasets.")
        prefix = f"multi{len(selected_datasets)}"
    elif mode == "all_datasets":
        prefix = "allDatasets"
    else:
        raise ValueError(f"Unsupported dataset selection mode: {mode}")
    return (
        f"{prefix}_L{int(context_length)}_h{int(prediction_length)}_"
        f"thr{_number_token(threshold)}"
    )


def selection_id_from_metadata(metadata: Mapping[str, Any]) -> str:
    """Build a selection ID from final-dataset metadata."""

    return make_selection_id(
        mode=str(metadata["dataset_selection_mode"]),
        selected_datasets=list(metadata.get("selected_datasets", [])),
        context_length=int(metadata["context_length"]),
        prediction_length=int(metadata["prediction_length"]),
        threshold=float(metadata.get("threshold", metadata.get("signal_threshold"))),
    )


def make_run_id(
    model_family: str,
    architecture: str,
    variant: str,
    selection_id: str,
) -> str:
    """Build a stable model run ID."""

    family = sanitize_id(model_family).lower()
    architecture_component = sanitize_id(architecture).lower()
    prefix = f"{family}_"
    if architecture_component == family:
        architecture_component = ""
    elif architecture_component.startswith(prefix):
        architecture_component = architecture_component[len(prefix) :]
    components = [family, architecture_component, variant_id(variant), selection_id]
    return "_".join(component for component in components if component)


def make_comparison_id(method_id: str, comparison_type: str = "switch_eval") -> str:
    """Build a stable comparison ID."""

    return f"{sanitize_id(comparison_type)}_{sanitize_id(method_id)}"


def get_run_dir(run_id: str, root: Path = PROJECT_ROOT) -> Path:
    return root / "results" / "runs" / run_id


def get_model_dir(model_family: str, run_id: str, root: Path = PROJECT_ROOT) -> Path:
    return root / "models" / sanitize_id(model_family).lower() / run_id


def get_perfect_switch_dir(selection_id: str, root: Path = PROJECT_ROOT) -> Path:
    return root / "results" / "switching" / "perfect_switch" / selection_id


def get_comparison_dir(comparison_id: str, root: Path = PROJECT_ROOT) -> Path:
    return root / "results" / "comparisons" / comparison_id


def get_data_preparation_dir(selection_id: str, root: Path = PROJECT_ROOT) -> Path:
    return root / "results" / "data_preparation" / selection_id


def get_results_index_dir(root: Path = PROJECT_ROOT) -> Path:
    return root / "results" / "index"


def get_grid_search_dir(search_id: str, root: Path = PROJECT_ROOT) -> Path:
    return root / "results" / "grid_searches" / sanitize_id(search_id)


def ensure_results_subdirs(base_dir: Path, subdirs: Sequence[str]) -> dict[str, Path]:
    """Create a result folder and its standard direct children."""

    base_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for subdir in subdirs:
        path = base_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = path
    return paths


def relative_project_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    """Return a repository-relative path when possible."""

    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def resolve_processed_dataset_dir(
    *,
    threshold_folder: str,
    context_length: int,
    prediction_length: int,
    selection_id: str | None = None,
    legacy_selection_folder: str | None = None,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve new or legacy final processed dataset folders during transition."""

    base = root / "data" / "processed" / "autoregressive" / threshold_folder
    candidates = []
    if selection_id:
        candidates.append(
            base / f"L{context_length}_h{prediction_length}" / selection_id
        )
    if legacy_selection_folder:
        candidates.append(
            base
            / f"datasets_L{context_length}_h{prediction_length}"
            / legacy_selection_folder
        )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Processed dataset folder not found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def upsert_index_row(
    path: Path,
    row: Mapping[str, Any],
    *,
    id_column: str,
    columns: Sequence[str],
) -> None:
    """Insert or replace one index row without duplicating its stable ID."""

    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"Index row is missing required columns: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
    existing = existing.reindex(columns=columns)
    existing = existing.loc[existing[id_column].astype(str).ne(str(row[id_column]))]
    updated = pd.concat(
        [existing, pd.DataFrame([{column: row[column] for column in columns}])],
        ignore_index=True,
    )
    updated.sort_values(id_column, kind="stable").to_csv(path, index=False)
