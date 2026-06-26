"""Tests for project-wide utility functions."""

import random

from src.utils.device import select_device
from src.utils.paths import PROJECT_ROOT, project_path
from src.utils.reproducibility import set_seed


def test_select_device_returns_supported_name() -> None:
    assert select_device() in {"cuda", "mps", "cpu"}


def test_project_path_is_relative_to_repository_root() -> None:
    assert project_path("configs", "data.yaml") == PROJECT_ROOT / "configs" / "data.yaml"


def test_set_seed_makes_python_random_repeatable() -> None:
    set_seed(42)
    first_value = random.random()
    set_seed(42)

    assert random.random() == first_value
