"""Path and identifier helpers for survival-persistence artifacts."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import PROJECT_ROOT
from src.utils.results_paths import sanitize_id

TASK_NAME = "survival_persistence"


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
