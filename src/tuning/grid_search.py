"""Model-agnostic utilities for deterministic validation-only grid searches."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np
import pandas as pd


def expand_parameter_grid(
    parameter_grid: Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    """Return the Cartesian product of a non-empty parameter grid."""

    if not parameter_grid:
        raise ValueError("parameter_grid cannot be empty.")
    names = list(parameter_grid)
    values = []
    for name in names:
        choices = parameter_grid[name]
        if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
            raise ValueError(f"Grid parameter '{name}' must contain a sequence.")
        if not choices:
            raise ValueError(f"Grid parameter '{name}' cannot be empty.")
        values.append(list(choices))
    return [
        dict(zip(names, combination, strict=True))
        for combination in product(*values)
    ]


def make_trial_id(parameters: Mapping[str, Any], index: int) -> str:
    """Build a compact stable trial ID from parameters and a display index."""

    serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:10]
    return f"trial_{index:03d}_{digest}"


def select_best_trial(
    trials: pd.DataFrame,
    *,
    metric: str,
    mode: str,
) -> pd.Series:
    """Select the best completed finite trial using one validation metric."""

    if metric not in trials:
        raise ValueError(f"Selection metric not found in trials: {metric}")
    if mode not in {"min", "max"}:
        raise ValueError("Selection mode must be 'min' or 'max'.")
    candidates = trials.copy()
    if "status" in candidates:
        candidates = candidates.loc[candidates["status"].eq("complete")]
    numeric_metric = pd.to_numeric(candidates[metric], errors="coerce")
    candidates = candidates.loc[np.isfinite(numeric_metric)].copy()
    candidates[metric] = numeric_metric.loc[candidates.index]
    if candidates.empty:
        raise ValueError("No completed finite trials are available for selection.")
    ascending = mode == "min"
    tie_breakers = [metric]
    directions = [ascending]
    if "trial_id" in candidates:
        tie_breakers.append("trial_id")
        directions.append(True)
    return candidates.sort_values(
        tie_breakers,
        ascending=directions,
        kind="stable",
    ).iloc[0]
