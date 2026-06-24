"""Compatibility wrapper for the task-scoped autoregressive dataset builder."""

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parent
    / "autoregressive"
    / "10_build_autoregressive_datasets.py"
)


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")

