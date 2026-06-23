"""Model-agnostic switching references, conversion rules, summaries, and plots."""

from src.switching.conversion import (
    compute_model_switch_from_predictions,
    compute_switch_from_signal_values,
    detect_persistent_threshold_switch,
    enforce_switch_time,
    hold_while_signal_above_threshold,
    ensure_min_island_length,
)
from src.switching.reference_switch import (
    align_perfect_switch_to_forecast_windows,
    compute_perfect_switch_from_true_signal,
)

__all__ = [
    "align_perfect_switch_to_forecast_windows",
    "compute_model_switch_from_predictions",
    "compute_perfect_switch_from_true_signal",
    "compute_switch_from_signal_values",
    "detect_persistent_threshold_switch",
    "enforce_switch_time",
    "hold_while_signal_above_threshold",
    "ensure_min_island_length",
]
