"""Tests for chronological splitting."""

import pytest

from src.data.splits import chronological_split


def test_returns_contiguous_non_overlapping_ranges() -> None:
    split = chronological_split(10, train_fraction=0.6, validation_fraction=0.2)

    assert (split.train.start, split.train.stop) == (0, 6)
    assert (split.validation.start, split.validation.stop) == (6, 8)
    assert (split.test.start, split.test.stop) == (8, 10)


def test_rejects_fractions_without_test_remainder() -> None:
    with pytest.raises(ValueError):
        chronological_split(10, train_fraction=0.8, validation_fraction=0.2)


def test_rejects_empty_split() -> None:
    with pytest.raises(ValueError):
        chronological_split(3, train_fraction=0.1, validation_fraction=0.1)
