"""Non-learned survival baselines for evaluation context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.tasks.survival_persistence.evaluation.metrics import fit_kaplan_meier


@dataclass(frozen=True)
class BaselinePrediction:
    """Predicted survival curve and median for a baseline method."""

    name: str
    survival_probabilities: np.ndarray
    median_seconds: np.ndarray


def kaplan_meier_marginal_baseline(
    *,
    train_times: np.ndarray,
    train_event_observed: np.ndarray,
    n_samples: int,
    horizons_seconds: np.ndarray,
) -> BaselinePrediction:
    """Predict the train-fitted marginal KM curve for every sample."""

    curve = fit_kaplan_meier(train_times, train_event_observed)
    horizon_probabilities = curve.predict(horizons_seconds)
    survival = np.tile(horizon_probabilities.reshape(1, -1), (n_samples, 1))
    median = curve.median()
    median_values = np.full(n_samples, np.nan if median is None else median)
    return BaselinePrediction(
        name="kaplan_meier_marginal",
        survival_probabilities=survival,
        median_seconds=median_values,
    )


def constant_median_baseline(
    *,
    train_times: np.ndarray,
    train_event_observed: np.ndarray,
    n_samples: int,
    horizons_seconds: np.ndarray,
) -> BaselinePrediction:
    """Use the train-fitted KM median and its implied step survival curve."""

    curve = fit_kaplan_meier(train_times, train_event_observed)
    median = curve.median()
    median_value = np.nan if median is None else float(median)
    median_values = np.full(n_samples, median_value)
    if np.isnan(median_value):
        survival = np.full((n_samples, len(horizons_seconds)), np.nan)
    else:
        survival = np.tile(
            (np.asarray(horizons_seconds, dtype=float) < median_value).astype(float),
            (n_samples, 1),
        )
    return BaselinePrediction(
        name="constant_train_km_median",
        survival_probabilities=survival,
        median_seconds=median_values,
    )
