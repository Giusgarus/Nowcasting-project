"""Path and identifier helpers for survival-persistence artifacts."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import PROJECT_ROOT
from src.utils.results_paths import sanitize_id

TASK_NAME = "survival_persistence"
RUN_INDEX_COLUMNS = [
    "run_id",
    "task_name",
    "model_family",
    "model_id",
    "feature_set",
    "sample_weighting",
    "selection_id",
    "dataset_path",
    "best_iteration",
    "best_val_aft_nloglik",
    "val_c_index",
    "test_c_index",
    "test_ipcw_brier_mean",
    "created_at",
    "status",
]


def number_token(value: float | int) -> str:
    """Return a filesystem-safe compact number token."""

    number = float(value)
    if number.is_integer():
        return f"{int(number)}p0"
    return f"{number:g}".replace("-", "m").replace(".", "p")


def threshold_folder(threshold_on: float) -> str:
    """Return the activation-threshold folder name."""

    return f"threshold_{number_token(threshold_on)}"


def make_selection_id(external_test_dataset: str) -> str:
    """Return the external-holdout selection ID."""

    return f"externalHoldout_test_{sanitize_id(external_test_dataset)}"


def make_run_id(
    *,
    model_id: str,
    feature_set: str,
    sample_weighting: str,
    selection_id: str,
    run_suffix: str | None = None,
) -> str:
    """Build a stable run ID for survival-persistence model baselines."""

    run_id = (
        f"survivalPersistence_{sanitize_id(model_id)}_"
        f"{sanitize_id(feature_set)}_{sanitize_id(sample_weighting)}_"
        f"{sanitize_id(selection_id)}"
    )
    if run_suffix:
        run_id = f"{run_id}__{sanitize_id(run_suffix)}"
    return run_id


def processed_dataset_dir(
    *,
    threshold_on: float,
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
        / threshold_folder(threshold_on)
        / f"L{int(context_length)}"
        / sanitize_id(selection_id)
    )


def data_preparation_dir(
    *,
    selection_id: str,
    threshold_on: float,
    context_length: int,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the human-readable data-preparation summary directory."""

    summary_id = f"{sanitize_id(selection_id)}_threshold{number_token(threshold_on)}_L{int(context_length)}"
    return root / "results" / "data_preparation" / TASK_NAME / summary_id


def run_dir(
    *,
    selection_id: str,
    run_id: str,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the task-scoped survival run directory."""

    return (
        root
        / "results"
        / "runs"
        / TASK_NAME
        / sanitize_id(selection_id)
        / sanitize_id(run_id)
    )


def model_dir(
    *,
    model_id: str,
    run_id: str,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the task-scoped survival model directory."""

    return root / "models" / TASK_NAME / sanitize_id(model_id) / sanitize_id(run_id)


def run_index_path(root: Path = PROJECT_ROOT) -> Path:
    """Return the task-scoped survival run index."""

    return root / "results" / "index" / f"{TASK_NAME}_runs.csv"


def grid_search_dir(
    *,
    selection_id: str,
    search_id: str,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the task-scoped survival grid-search artifact directory."""

    return (
        root
        / "results"
        / "grid_searches"
        / TASK_NAME
        / sanitize_id(selection_id)
        / sanitize_id(search_id)
    )


def model_selection_dir(
    *,
    selection_id: str,
    search_id: str,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the cross-run model-selection summary directory."""

    return (
        root
        / "results"
        / "comparisons"
        / "model_selection"
        / TASK_NAME
        / sanitize_id(selection_id)
        / sanitize_id(search_id)
    )


def switch_comparison_dir(
    *,
    selection_id: str,
    comparison_id: str,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return one task-scoped model-vs-Perfect switch-comparison directory."""

    return (
        root
        / "results"
        / "comparisons"
        / "switch_eval"
        / TASK_NAME
        / sanitize_id(selection_id)
        / sanitize_id(comparison_id)
    )


def switch_summary_dir(
    *,
    selection_id: str,
    summary_id: str,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the cross-model switch-summary directory for this task."""

    return (
        root
        / "results"
        / "comparisons"
        / "model_summary"
        / TASK_NAME
        / sanitize_id(selection_id)
        / sanitize_id(summary_id)
    )
