"""Compatibility wrapper for the task-scoped model-summary script."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parent / "autoregressive" / "14_compare_model_results.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")

