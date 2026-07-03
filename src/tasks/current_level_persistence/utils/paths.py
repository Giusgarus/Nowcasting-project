"""Path and identifier helpers for current-level persistence artifacts."""

from __future__ import annotations

from pathlib import Path
import re

from src.utils.paths import PROJECT_ROOT
from src.utils.results_paths import sanitize_id

TASK_NAME = "current_level_persistence"
RUN_INDEX_COLUMNS = [
    "run_id",
    "task_name",
    "model_family",
    "model_id",
    "delta",
    "context_length",
    "selection_id",
    "dataset_path",
    "best_epoch",
    "best_val_mae_seconds",
    "best_val_rmse_seconds",
    "test_mae_seconds",
    "test_rmse_seconds",
    "test_median_ae_seconds",
    "created_at",
    "status",
]


def number_token(value: float | int) -> str:
    """Return a filesystem-safe compact number token."""

    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}".replace("-", "m").replace(".", "p")


def make_selection_id(external_test_dataset: str) -> str:
    """Return the dataset-selection ID used by this task."""

    return f"externalHoldout_test_{sanitize_id(external_test_dataset)}"


def make_run_id(
    *,
    delta: float,
    context_length: int,
    model_id: str,
    selection_id: str,
    scalar_context: bool = False,
    run_suffix: str | None = None,
) -> str:
    """Build the stable current-level persistence run ID."""

    feature_suffix = "_scalarContext" if scalar_context else ""
    run_id = (
        f"currentLevelPersistence_delta{number_token(delta)}_"
        f"L{int(context_length)}_{sanitize_id(model_id)}{feature_suffix}_{selection_id}"
    )
    if run_suffix:
        run_id = f"{run_id}__{sanitize_id(run_suffix)}"
    return run_id


def base_run_id(run_id: str) -> str:
    """Return the run ID without an optional ``__suffix`` component."""

    return str(run_id).split("__", 1)[0]


def selection_id_from_run_id(run_id: str) -> str:
    """Extract the current-level persistence selection ID embedded in a run ID."""

    match = re.search(r"(externalHoldout_test_[A-Za-z0-9_]+)$", base_run_id(run_id))
    if not match:
        raise ValueError(
            f"Cannot infer current-level persistence selection from {run_id!r}."
        )
    return match.group(1)


def model_id_from_run_id(run_id: str) -> str:
    """Extract the model ID embedded in a current-level persistence run ID."""

    match = re.search(
        r"^currentLevelPersistence_delta[^_]+_L\d+_(.+)_externalHoldout_test_",
        base_run_id(run_id),
    )
    if not match:
        raise ValueError(f"Cannot infer current-level persistence model from {run_id!r}.")
    model_id = match.group(1)
    if model_id.endswith("_scalarContext"):
        model_id = model_id.removesuffix("_scalarContext")
    return sanitize_id(model_id)


def processed_dataset_dir(
    *,
    delta: float,
    context_length: int,
    selection_id: str,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the processed dataset directory for this task."""

    return (
        root
        / "data"
        / "processed"
        / TASK_NAME
        / f"delta_{number_token(delta)}"
        / f"L{int(context_length)}"
        / selection_id
    )


def run_dir(run_id: str, root: Path = PROJECT_ROOT) -> Path:
    """Return the task-separated run directory."""

    return (
        root
        / "results"
        / "runs"
        / TASK_NAME
        / selection_id_from_run_id(run_id)
        / run_id
    )


def model_dir(run_id: str, root: Path = PROJECT_ROOT) -> Path:
    """Return the task-separated checkpoint directory."""

    return root / "models" / TASK_NAME / model_id_from_run_id(run_id) / run_id


def grid_search_dir(
    *,
    selection_id: str,
    search_id: str,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the task-separated grid-search directory."""

    return (
        root
        / "results"
        / "grid_searches"
        / TASK_NAME
        / sanitize_id(selection_id)
        / sanitize_id(search_id)
    )


def run_index_path(root: Path = PROJECT_ROOT) -> Path:
    """Return the separate current-level persistence run index."""

    return root / "results" / "index" / f"{TASK_NAME}_runs.csv"
