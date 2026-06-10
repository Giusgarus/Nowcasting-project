"""Diagnostic metrics for autoregressive trajectory forecasts."""

from collections.abc import Sequence
from math import sqrt


def _paired_values(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> zip:
    if len(actual) == 0:
        raise ValueError("Metric inputs cannot be empty.")
    if len(actual) != len(predicted):
        raise ValueError("Metric inputs must have the same length.")
    return zip(actual, predicted, strict=True)


def mean_absolute_error(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Return the mean absolute error for equally sized flat sequences."""

    errors = [abs(observed - forecast) for observed, forecast in _paired_values(actual, predicted)]
    return sum(errors) / len(errors)


def root_mean_squared_error(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Return the root mean squared error for equally sized flat sequences."""

    squared_errors = [
        (observed - forecast) ** 2
        for observed, forecast in _paired_values(actual, predicted)
    ]
    return sqrt(sum(squared_errors) / len(squared_errors))
