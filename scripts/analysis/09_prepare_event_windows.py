"""Compatibility wrapper for the task-scoped autoregressive preparation script."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parent / "autoregressive" / "09_prepare_event_windows.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")

