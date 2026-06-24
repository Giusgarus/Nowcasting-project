"""Placeholder for deriving switch decisions from predicted persistence duration.

Future rule:

``predicted_switch = 1 if predicted_remaining_persistence_seconds >= switch_time_seconds else 0``

Switch metrics are intentionally not implemented in this step; this task first
needs duration-model validation before the operational conversion is finalized.
"""

from __future__ import annotations

import numpy as np


def derive_switch_from_duration(
    predicted_remaining_persistence_seconds: np.ndarray,
    *,
    switch_time_seconds: float,
) -> np.ndarray:
    """Return the future duration-to-switch placeholder rule."""

    if switch_time_seconds < 0:
        raise ValueError("switch_time_seconds must be non-negative.")
    return (
        np.asarray(predicted_remaining_persistence_seconds, dtype=float)
        >= float(switch_time_seconds)
    ).astype(np.int8)
