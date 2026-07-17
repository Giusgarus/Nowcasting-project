"""Survival-aware metrics and non-learned baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KaplanMeierCurve:
    """Simple right-censored Kaplan-Meier step curve."""

    event_times: np.ndarray
    survival: np.ndarray

    def predict(self, times: np.ndarray | list[float]) -> np.ndarray:
        """Return S(t) using the last known step at or before each time."""

        query = np.asarray(times, dtype=np.float64)
        if self.event_times.size == 0:
            return np.ones_like(query, dtype=np.float64)
        indices = np.searchsorted(self.event_times, query, side="right") - 1
        output = np.ones_like(query, dtype=np.float64)
        valid = indices >= 0
        output[valid] = self.survival[indices[valid]]
        return np.clip(output, 0.0, 1.0)

    def median(self) -> float | None:
        """Return the first time where S(t) <= 0.5, if identifiable."""

        if self.event_times.size == 0:
            return None
        mask = self.survival <= 0.5
        if not mask.any():
            return None
        return float(self.event_times[np.argmax(mask)])


def fit_kaplan_meier(times: np.ndarray, event_observed: np.ndarray) -> KaplanMeierCurve:
    """Fit a Kaplan-Meier curve from durations and event indicators."""

    durations = np.asarray(times, dtype=np.float64)
    events = np.asarray(event_observed, dtype=bool)
    if durations.ndim != 1 or events.ndim != 1 or len(durations) != len(events):
        raise ValueError("times and event_observed must be aligned one-dimensional arrays.")
    if np.any(~np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("Kaplan-Meier durations must be positive and finite.")
    unique_times = np.sort(np.unique(durations))
    survival_values: list[float] = []
    event_times: list[float] = []
    survival = 1.0
    for time in unique_times:
        at_risk = int(np.sum(durations >= time))
        observed_events = int(np.sum((durations == time) & events))
        if at_risk <= 0:
            continue
        if observed_events > 0:
            survival *= 1.0 - observed_events / float(at_risk)
            event_times.append(float(time))
            survival_values.append(float(survival))
    return KaplanMeierCurve(
        event_times=np.asarray(event_times, dtype=np.float64),
        survival=np.asarray(survival_values, dtype=np.float64),
    )


def fit_censoring_survival(times: np.ndarray, event_observed: np.ndarray) -> KaplanMeierCurve:
    """Fit G(t)=P(C>t) by treating censoring as the event of interest."""

    return fit_kaplan_meier(times, 1 - np.asarray(event_observed, dtype=np.int64))


def harrell_c_index(
    times: np.ndarray,
    event_observed: np.ndarray,
    prediction: np.ndarray,
    *,
    higher_prediction_longer: bool = True,
) -> dict[str, float | int]:
    """Compute Harrell's C-index for right-censored survival data.

    A pair is comparable when the shorter observed time belongs to an observed
    event. Larger predictions are interpreted as longer survival by default.
    Prediction ties receive half credit.
    """

    durations = np.asarray(times, dtype=np.float64)
    observed = np.asarray(event_observed, dtype=bool)
    scores = np.asarray(prediction, dtype=np.float64)
    if not (durations.ndim == observed.ndim == scores.ndim == 1):
        raise ValueError("C-index inputs must be one-dimensional.")
    if not (len(durations) == len(observed) == len(scores)):
        raise ValueError("C-index inputs must have the same length.")
    comparable = 0
    concordant = 0.0
    tied_predictions = 0
    n = len(durations)
    for i in range(n):
        for j in range(i + 1, n):
            if durations[i] == durations[j]:
                continue
            if durations[i] < durations[j]:
                shorter, longer = i, j
            else:
                shorter, longer = j, i
            if not observed[shorter]:
                continue
            comparable += 1
            shorter_score = scores[shorter]
            longer_score = scores[longer]
            if np.isclose(shorter_score, longer_score):
                tied_predictions += 1
                concordant += 0.5
            elif higher_prediction_longer:
                concordant += float(shorter_score < longer_score)
            else:
                concordant += float(shorter_score > longer_score)
    c_index = concordant / comparable if comparable else np.nan
    return {
        "c_index": float(c_index),
        "num_comparable_pairs": int(comparable),
        "num_tied_prediction_pairs": int(tied_predictions),
    }


def horizon_known_status(
    times: np.ndarray,
    event_observed: np.ndarray,
    *,
    horizon_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return known survival-beyond-horizon labels and a known-status mask."""

    durations = np.asarray(times, dtype=np.float64)
    observed = np.asarray(event_observed, dtype=bool)
    horizon = float(horizon_seconds)
    survived = durations > horizon
    known = survived | observed
    return survived.astype(np.float64), known


def ipcw_brier_score(
    times: np.ndarray,
    event_observed: np.ndarray,
    predicted_survival: np.ndarray,
    *,
    horizon_seconds: float,
    censoring_curve: KaplanMeierCurve,
    epsilon: float = 1e-6,
) -> dict[str, float | int]:
    """Compute IPCW Brier score for S(horizon | X)."""

    durations = np.asarray(times, dtype=np.float64)
    observed = np.asarray(event_observed, dtype=bool)
    predicted = np.asarray(predicted_survival, dtype=np.float64)
    horizon = float(horizon_seconds)
    if not (len(durations) == len(observed) == len(predicted)):
        raise ValueError("Brier inputs must have the same length.")
    if np.any(predicted < 0.0) or np.any(predicted > 1.0):
        raise ValueError("Predicted survival probabilities must be in [0, 1].")

    survived, known = horizon_known_status(
        durations,
        observed.astype(int),
        horizon_seconds=horizon,
    )
    weights = np.zeros_like(durations, dtype=np.float64)
    event_before_horizon = observed & (durations <= horizon)
    alive_at_horizon = durations > horizon
    if event_before_horizon.any():
        weights[event_before_horizon] = 1.0 / np.clip(
            censoring_curve.predict(durations[event_before_horizon]),
            epsilon,
            None,
        )
    if alive_at_horizon.any():
        weights[alive_at_horizon] = 1.0 / np.clip(
            censoring_curve.predict(np.asarray([horizon]))[0],
            epsilon,
            None,
        )
    contribution = weights * np.square(survived - predicted)
    return {
        "horizon_seconds": horizon,
        "ipcw_brier": float(contribution.mean()),
        "num_samples": int(len(durations)),
        "num_known_status": int(known.sum()),
        "num_unknown_censored_before_horizon": int((~known).sum()),
        "sum_ipcw_weights": float(weights.sum()),
    }


def brier_scores_by_horizon(
    times: np.ndarray,
    event_observed: np.ndarray,
    survival_probabilities: np.ndarray,
    horizons_seconds: np.ndarray,
    *,
    censoring_curve: KaplanMeierCurve,
) -> pd.DataFrame:
    """Return IPCW Brier scores for a horizon grid."""

    rows = []
    probabilities = np.asarray(survival_probabilities, dtype=np.float64)
    horizons = np.asarray(horizons_seconds, dtype=np.float64)
    if probabilities.shape != (len(times), len(horizons)):
        raise ValueError("survival_probabilities has incompatible shape.")
    for index, horizon in enumerate(horizons):
        rows.append(
            ipcw_brier_score(
                times,
                event_observed,
                probabilities[:, index],
                horizon_seconds=float(horizon),
                censoring_curve=censoring_curve,
            )
        )
    return pd.DataFrame(rows)


def calibration_by_horizon(
    times: np.ndarray,
    event_observed: np.ndarray,
    predicted_survival: np.ndarray,
    *,
    horizon_seconds: float,
    n_bins: int,
    censoring_curve: KaplanMeierCurve,
) -> pd.DataFrame:
    """Build IPCW calibration bins for one survival horizon."""

    predicted = np.asarray(predicted_survival, dtype=np.float64)
    survived, known = horizon_known_status(
        times,
        event_observed,
        horizon_seconds=horizon_seconds,
    )
    if len(predicted) < n_bins:
        n_bins = max(1, len(predicted))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(predicted, quantiles))
    if len(edges) <= 1:
        edges = np.asarray([predicted.min() - 1e-12, predicted.max() + 1e-12])
    bin_ids = np.digitize(predicted, edges[1:-1], right=True)
    durations = np.asarray(times, dtype=np.float64)
    observed = np.asarray(event_observed, dtype=bool)
    weights_all = np.zeros_like(durations, dtype=np.float64)
    event_before_horizon = observed & (durations <= float(horizon_seconds))
    alive_at_horizon = durations > float(horizon_seconds)
    if event_before_horizon.any():
        weights_all[event_before_horizon] = 1.0 / np.clip(
            censoring_curve.predict(durations[event_before_horizon]),
            1e-6,
            None,
        )
    if alive_at_horizon.any():
        weights_all[alive_at_horizon] = 1.0 / np.clip(
            censoring_curve.predict([float(horizon_seconds)])[0],
            1e-6,
            None,
        )
    rows = []
    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        known_mask = mask & known
        weights = weights_all[known_mask]
        observed_survival = (
            float(np.average(survived[known_mask], weights=weights))
            if weights.size and weights.sum() > 0.0
            else np.nan
        )
        rows.append(
            {
                "horizon_seconds": float(horizon_seconds),
                "bin_id": int(bin_id),
                "bin_left": float(edges[bin_id]),
                "bin_right": float(edges[bin_id + 1]),
                "num_samples": int(mask.sum()),
                "num_known_status": int(known_mask.sum()),
                "mean_predicted_survival": float(predicted[mask].mean()),
                "ipcw_observed_survival": observed_survival,
            }
        )
    return pd.DataFrame(rows)


def event_level_bootstrap(
    frame: pd.DataFrame,
    *,
    event_column: str,
    metric_fn: Callable[[pd.DataFrame], float],
    n_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap a metric by resampling complete events."""

    if n_replicates <= 0:
        return pd.DataFrame()
    event_ids = np.asarray(sorted(frame[event_column].dropna().unique()), dtype=object)
    if len(event_ids) == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    grouped = {event_id: group for event_id, group in frame.groupby(event_column, sort=False)}
    for replicate in range(n_replicates):
        sampled = rng.choice(event_ids, size=len(event_ids), replace=True)
        sample_frame = pd.concat([grouped[event_id] for event_id in sampled], ignore_index=True)
        rows.append({"bootstrap_replicate": replicate, "metric": metric_fn(sample_frame)})
    return pd.DataFrame(rows)
