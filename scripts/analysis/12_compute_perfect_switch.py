"""Compatibility wrapper for the task-scoped Perfect Switch script."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parent / "autoregressive" / "12_compute_perfect_switch.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")

