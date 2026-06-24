"""Tests for the task-scoped autoregressive namespace."""

from src.tasks.autoregressive.data.autoregressive_dataset import (
    build_autoregressive_arrays_from_window_index,
)
from src.tasks.autoregressive.data.windowed_forecasting import build_window_indices
from src.tasks.autoregressive.models.gru import build_gru_forecaster
from src.tasks.autoregressive.models.patchtst import PatchTSTForecaster


def test_autoregressive_task_namespace_reexports_existing_components() -> None:
    assert callable(build_window_indices)
    assert callable(build_autoregressive_arrays_from_window_index)
    assert callable(build_gru_forecaster)
    assert PatchTSTForecaster.__name__ == "PatchTSTForecaster"

