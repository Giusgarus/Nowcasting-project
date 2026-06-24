"""Compatibility entrypoint for the current-level persistence dataset builder."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.current_level_persistence.build_dataset import *  # noqa: F401,F403
from scripts.experiments.current_level_persistence.build_dataset import main


if __name__ == "__main__":
    main()
