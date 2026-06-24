"""Compatibility wrapper for the task-scoped autoregressive sanity checks."""

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parent
    / "autoregressive"
    / "11_sanity_check_autoregressive_dataset.py"
)


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")

