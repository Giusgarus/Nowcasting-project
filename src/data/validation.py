"""Validation for the strict two-column Time and Signal schema."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SignalColumnSelection:
    """Required signal-only columns and explicitly ignored extra columns."""

    time_col: str = "Time"
    signal_col: str = "Signal"
    ignored_columns: tuple[str, ...] = ()

    @property
    def input_size(self) -> int:
        """Return the fixed univariate model input dimension."""

        return 1


def validate_signal_columns(
    available_columns: Sequence[str],
    *,
    time_col: str = "Time",
    signal_col: str = "Signal",
) -> SignalColumnSelection:
    """Require Time and Signal and identify every extra column to ignore."""

    if len(available_columns) < 2:
        raise ValueError("At least two columns are required: Time and Signal.")
    if time_col == signal_col:
        raise ValueError("Time and Signal columns must be different.")
    if len(set(available_columns)) != len(available_columns):
        raise ValueError("Column names must be unique.")

    missing = [
        column for column in (time_col, signal_col) if column not in available_columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    ignored = tuple(
        column for column in available_columns if column not in (time_col, signal_col)
    )
    return SignalColumnSelection(
        time_col=time_col,
        signal_col=signal_col,
        ignored_columns=ignored,
    )
