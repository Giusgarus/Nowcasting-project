"""Path and identifier helpers for long-fade detection artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from src.utils.paths import PROJECT_ROOT
from src.utils.results_paths import sanitize_id

TASK_NAME = "long_fade_detection"


def number_token(value: float | int) -> str:
    """Return a filesystem-safe compact number token."""

    number = float(value)
    if number.is_integer():
        return f"{int(number)}p0"
    return f"{number:g}".replace("-", "m").replace(".", "p")


def threshold_folder(threshold_db: float) -> str:
    """Return the threshold folder name."""

    return f"threshold_{number_token(threshold_db)}"


def duration_folder(min_fade_duration_seconds: int) -> str:
    """Return the minimum-duration folder name."""

    return f"min_duration_{int(min_fade_duration_seconds)}s"


def make_selection_id(external_test_dataset: str) -> str:
    """Return the external-holdout selection ID."""

    return f"externalHoldout_test_{sanitize_id(external_test_dataset)}"


def model_component(model_id: str) -> str:
    """Return the run-id model component expected for this task."""

    mapping = {
        "xgboost_lag_scalar_classifier": "xgboost_lag_scalar",
        "tcn_classifier": "tcn_classifier",
        "multiscale_shapelet_convolution_classifier": "shapelet_convolution_classifier",
    }
    return mapping.get(model_id, sanitize_id(model_id))


def base_run_id(run_id: str) -> str:
    """Return the run ID without an optional ``__suffix`` component."""

    return str(run_id).split("__", 1)[0]


def selection_id_from_run_id(run_id: str) -> str:
    """Extract the long-fade selection ID embedded in a run ID."""

    match = re.search(r"(externalHoldout_test_[A-Za-z0-9_]+)$", base_run_id(run_id))
    if not match:
        raise ValueError(f"Cannot infer long-fade selection from {run_id!r}.")
    return match.group(1)


def model_id_from_run_id(run_id: str) -> str:
    """Extract the canonical long-fade model ID embedded in a run ID."""

    match = re.search(
        r"^longFadeDetection_thr[^_]+_dur\d+_(.+)_externalHoldout_test_",
        base_run_id(run_id),
    )
    if not match:
        raise ValueError(f"Cannot infer long-fade model ID from {run_id!r}.")
    reverse_mapping = {
        "xgboost_lag_scalar": "xgboost_lag_scalar_classifier",
        "tcn_classifier": "tcn_classifier",
        "shapelet_convolution_classifier": "multiscale_shapelet_convolution_classifier",
    }
    return reverse_mapping.get(match.group(1), sanitize_id(match.group(1)))


def make_run_id(
    *,
    threshold_db: float,
    min_fade_duration_seconds: int,
    model_id: str,
    selection_id: str,
    run_suffix: str | None = None,
) -> str:
    """Return the stable run ID."""

    run_id = (
        f"longFadeDetection_thr{number_token(threshold_db)}_"
        f"dur{int(min_fade_duration_seconds)}_{model_component(model_id)}_"
        f"{selection_id}"
    )
    if run_suffix:
        run_id = f"{run_id}__{sanitize_id(run_suffix)}"
    return run_id


def processed_dataset_dir(
    *,
    threshold_db: float,
    min_fade_duration_seconds: int,
    context_length: int,
    selection_id: str,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the processed dataset directory."""

    return (
        root
        / "data"
        / "processed"
        / TASK_NAME
        / threshold_folder(threshold_db)
        / duration_folder(min_fade_duration_seconds)
        / f"L{int(context_length)}"
        / sanitize_id(selection_id)
    )


def run_dir(run_id: str, root: Path = PROJECT_ROOT) -> Path:
    """Return the task- and selection-scoped run directory."""

    return (
        root
        / "results"
        / "runs"
        / TASK_NAME
        / selection_id_from_run_id(run_id)
        / sanitize_id(run_id)
    )


def model_dir(run_id: str, root: Path = PROJECT_ROOT) -> Path:
    """Return the task- and model-scoped checkpoint directory."""

    return (
        root
        / "models"
        / TASK_NAME
        / model_id_from_run_id(run_id)
        / sanitize_id(run_id)
    )


def grid_search_dir(
    *,
    selection_id: str,
    search_id: str,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the grid-search artifact directory."""

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
    """Return the compact model-selection summary directory."""

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


def model_summary_dir(
    *,
    selection_id: str,
    summary_id: str,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Return the long-fade model-summary directory."""

    return switch_summary_dir(
        selection_id=selection_id,
        summary_id=summary_id,
        root=root,
    )


def summary_tables_dir(root: Path = PROJECT_ROOT) -> Path:
    """Return the long-fade summary table directory."""

    return root / "results" / "tables" / TASK_NAME


def summary_reports_dir(root: Path = PROJECT_ROOT) -> Path:
    """Return the long-fade summary report directory."""

    return root / "results" / "reports" / TASK_NAME
