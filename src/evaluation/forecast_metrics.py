"""Diagnostic metrics for autoregressive trajectory forecasts."""

from collections.abc import Sequence
from math import sqrt

import numpy as np
import pandas as pd


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


def inverse_context_standardization(
    predictions: np.ndarray,
    scaling_mean: np.ndarray,
    scaling_std: np.ndarray,
) -> np.ndarray:
    """Return trajectory predictions to raw scale using per-context statistics."""

    predictions = np.asarray(predictions)
    scaling_mean = np.asarray(scaling_mean)
    scaling_std = np.asarray(scaling_std)
    if predictions.ndim != 2:
        raise ValueError("predictions must have shape (N, prediction_length).")
    if scaling_mean.shape != (len(predictions),) or scaling_std.shape != (
        len(predictions),
    ):
        raise ValueError("Scaling arrays must have shape (N,).")
    return predictions * scaling_std[:, None] + scaling_mean[:, None]


def compute_trajectory_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Compute global MAE and RMSE for equally shaped trajectory arrays."""

    actual, predicted = _validated_trajectory_arrays(actual, predicted)
    errors = predicted - actual
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def compute_horizon_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    """Compute MAE and RMSE separately for every forecast horizon step."""

    actual, predicted = _validated_trajectory_arrays(actual, predicted)
    errors = predicted - actual
    return pd.DataFrame(
        {
            "horizon_step": np.arange(1, actual.shape[1] + 1),
            "mae": np.mean(np.abs(errors), axis=0),
            "rmse": np.sqrt(np.mean(np.square(errors), axis=0)),
        }
    )


def _validated_trajectory_arrays(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 2 or predicted.ndim != 2:
        raise ValueError("Trajectory arrays must have shape (N, prediction_length).")
    if actual.shape != predicted.shape:
        raise ValueError("Trajectory arrays must have identical shapes.")
    if actual.size == 0:
        raise ValueError("Trajectory arrays cannot be empty.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Trajectory arrays must contain only finite values.")
    return actual, predicted
