"""Compatibility wrapper for current-level persistence shapelet training."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.current_level_persistence.train_learnable_shapelets import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    main()
