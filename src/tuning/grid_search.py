"""Model-agnostic utilities for deterministic validation-only grid searches."""

import hashlib
import json
from decimal import Decimal
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np
import pandas as pd


def _expand_range_spec(name: str, spec: Mapping[str, Any]) -> list[Any]:
    """Expand an inclusive ``start``/``stop``/``step`` range specification."""

    required = {"start", "stop", "step"}
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Grid range parameter '{name}' is missing keys: {missing}")
    start = Decimal(str(spec["start"]))
    stop = Decimal(str(spec["stop"]))
    step = Decimal(str(spec["step"]))
    if step <= 0:
        raise ValueError(f"Grid range parameter '{name}' must use a positive step.")
    if start > stop:
        raise ValueError(f"Grid range parameter '{name}' must have start <= stop.")

    values = []
    current = start
    while current <= stop:
        values.append(current)
        current += step
    if not values:
        raise ValueError(f"Grid range parameter '{name}' expanded to no values.")

    integer_like = all(
        isinstance(spec[key], (int, np.integer)) and not isinstance(spec[key], bool)
        for key in required
    )
    if integer_like:
        return [int(value) for value in values]
    return [float(value) for value in values]


def expand_parameter_values(name: str, choices: Any) -> list[Any]:
    """Expand either an explicit sequence or an inclusive range spec."""

    if isinstance(choices, Mapping):
        return _expand_range_spec(name, choices)
    if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
        raise ValueError(
            f"Grid parameter '{name}' must contain a sequence or range spec."
        )
    if not choices:
        raise ValueError(f"Grid parameter '{name}' cannot be empty.")
    return list(choices)


def expand_parameter_grid(
    parameter_grid: Mapping[str, Sequence[Any] | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the Cartesian product of a non-empty parameter grid.

    Values may be explicit sequences or inclusive range specs such as:

    ``{"start": 0.1, "stop": 0.5, "step": 0.2}``
    """

    if not parameter_grid:
        raise ValueError("parameter_grid cannot be empty.")
    names = list(parameter_grid)
    values = []
    for name in names:
        values.append(expand_parameter_values(name, parameter_grid[name]))
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
