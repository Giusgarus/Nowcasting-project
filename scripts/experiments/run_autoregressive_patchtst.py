"""Compatibility wrapper for the task-scoped PatchTST autoregressive entrypoint."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.autoregressive.run_autoregressive_patchtst import *  # noqa: F401,F403
from scripts.experiments.autoregressive.run_autoregressive_patchtst import main


if __name__ == "__main__":
    main()
