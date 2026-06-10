"""Tests for strict Time and Signal column validation."""

import pytest

from src.data.validation import validate_signal_columns


def test_exact_signal_schema_has_no_ignored_columns() -> None:
    selection = validate_signal_columns(["Time", "Signal"])

    assert selection.time_col == "Time"
    assert selection.signal_col == "Signal"
    assert selection.ignored_columns == ()
    assert selection.input_size == 1


def test_extra_columns_are_explicitly_ignored() -> None:
    selection = validate_signal_columns(["Time", "Signal", "Unused"])

    assert selection.ignored_columns == ("Unused",)
    assert selection.input_size == 1


def test_fewer_than_two_columns_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="At least two columns"):
        validate_signal_columns(["Signal"])


def test_missing_required_named_column_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Missing required columns: Time"):
        validate_signal_columns(["Timestamp", "Signal"])


def test_duplicate_columns_raise() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        validate_signal_columns(["Time", "Signal", "Signal"])
