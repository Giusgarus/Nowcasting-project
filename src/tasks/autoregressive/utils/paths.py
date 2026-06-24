"""Task-scoped path helpers for autoregressive forecasting artifacts."""

from pathlib import Path

from src.utils.paths import PROJECT_ROOT
from src.utils.results_paths import (
    TASK_AUTOREGRESSIVE,
    get_comparison_dir,
    get_data_preparation_dir,
    get_grid_search_dir,
    get_model_dir,
    get_perfect_switch_dir,
    get_run_dir,
    resolve_comparison_dir,
    resolve_data_preparation_dir,
    resolve_grid_search_dir,
    resolve_model_dir,
    resolve_perfect_switch_dir,
    resolve_run_dir,
)

TASK_NAME = TASK_AUTOREGRESSIVE


def processed_dataset_root(root: Path = PROJECT_ROOT) -> Path:
    """Return the root for processed autoregressive datasets."""

    return root / "data" / "processed" / TASK_NAME

