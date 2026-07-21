"""Discrete-time label adapter for survival-persistence models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiscreteTimeBinSpec:
    """Survival time grid used by a discrete-time hazard model.

    Intervals are left-open and right-closed: ``(left_edges[k], right_edges[k]]``.
    When ``include_open_ended`` is true, the last right edge is ``inf`` and the
    final bin represents all residual durations beyond the last finite edge.
    """

    left_edges_seconds: np.ndarray
    right_edges_seconds: np.ndarray
    strategy: str
    include_open_ended: bool

    def __post_init__(self) -> None:
        left = np.asarray(self.left_edges_seconds, dtype=np.float64)
        right = np.asarray(self.right_edges_seconds, dtype=np.float64)
        if left.ndim != 1 or right.ndim != 1 or len(left) != len(right):
            raise ValueError("Discrete-time bin edges must be aligned one-dimensional arrays.")
        if len(left) == 0:
            raise ValueError("At least one discrete-time bin is required.")
        if np.any(~np.isfinite(left)) or np.any(left < 0.0):
            raise ValueError("Left bin edges must be finite and non-negative.")
        finite_right = right[np.isfinite(right)]
        if np.any(finite_right <= 0.0):
            raise ValueError("Finite right bin edges must be positive.")
        if np.any(left >= right):
            raise ValueError("Every bin must have left_edge < right_edge.")
        if not np.all(np.diff(left) >= 0.0):
            raise ValueError("Left bin edges must be monotone.")
        if not np.all(np.diff(right) > 0.0):
            raise ValueError("Right bin edges must be strictly increasing.")
        object.__setattr__(self, "left_edges_seconds", left)
        object.__setattr__(self, "right_edges_seconds", right)

    @property
    def num_bins(self) -> int:
        """Return the number of discrete hazard bins."""

        return int(len(self.right_edges_seconds))

    @property
    def finite_right_edges_seconds(self) -> np.ndarray:
        """Return finite right edges only."""

        return self.right_edges_seconds[np.isfinite(self.right_edges_seconds)]

    @property
    def max_finite_time_seconds(self) -> float:
        """Return the largest finite horizon represented by the grid."""

        finite = self.finite_right_edges_seconds
        if len(finite) == 0:
            raise ValueError("The discrete-time grid has no finite right edge.")
        return float(finite[-1])

    def to_dict(self) -> dict[str, Any]:
        """Return YAML/JSON-friendly bin metadata."""

        def serialize(values: np.ndarray) -> list[float | str]:
            output: list[float | str] = []
            for value in values:
                output.append("inf" if np.isinf(value) else float(value))
            return output

        return {
            "strategy": self.strategy,
            "include_open_ended": bool(self.include_open_ended),
            "num_bins": self.num_bins,
            "left_edges_seconds": serialize(self.left_edges_seconds),
            "right_edges_seconds": serialize(self.right_edges_seconds),
            "finite_right_edges_seconds": serialize(self.finite_right_edges_seconds),
        }


@dataclass(frozen=True)
class DiscreteTimeSurvivalLabels:
    """Discrete labels and likelihood masks derived from continuous survival labels."""

    event_bin_index: np.ndarray
    censoring_bin_index: np.ndarray
    at_risk_mask: np.ndarray
    event_mask: np.ndarray
    valid_likelihood_mask: np.ndarray


def _validated_edges(edges: np.ndarray | list[float], *, name: str) -> np.ndarray:
    values = np.asarray(edges, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"{name} must contain at least two edges.")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values.")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    if not np.isclose(values[0], 0.0):
        raise ValueError(f"{name} must start at 0 seconds.")
    return values.astype(np.float64)


def make_discrete_time_bin_spec(
    *,
    strategy: str,
    train_times_seconds: np.ndarray | None = None,
    bin_edges_seconds: list[float] | np.ndarray | None = None,
    num_bins: int | None = None,
    bin_width_seconds: float | None = None,
    max_time_seconds: float | None = None,
    include_open_ended: bool = True,
) -> DiscreteTimeBinSpec:
    """Create a discrete-time survival bin specification.

    ``explicit`` and ``hybrid`` both consume ``bin_edges_seconds``. ``uniform``
    builds an equally spaced finite grid. ``train_quantile`` uses observed train
    labels only and therefore must receive ``train_times_seconds`` from the
    training split.
    """

    strategy = str(strategy)
    if strategy in {"explicit", "hybrid"}:
        if bin_edges_seconds is None:
            raise ValueError(f"{strategy} binning requires bin_edges_seconds.")
        finite_edges = _validated_edges(bin_edges_seconds, name="bin_edges_seconds")
    elif strategy == "uniform":
        if bin_width_seconds is not None:
            width = float(bin_width_seconds)
            if width <= 0.0:
                raise ValueError("bin_width_seconds must be positive.")
            if max_time_seconds is None:
                if train_times_seconds is None:
                    raise ValueError("uniform binning needs max_time_seconds or train_times_seconds.")
                max_time_seconds = float(np.nanmax(train_times_seconds))
            max_time = float(max_time_seconds)
            finite_edges = np.arange(0.0, max_time + width, width, dtype=np.float64)
            if finite_edges[-1] < max_time:
                finite_edges = np.append(finite_edges, max_time)
        else:
            if num_bins is None or num_bins < 1:
                raise ValueError("uniform binning needs positive num_bins or bin_width_seconds.")
            if max_time_seconds is None:
                if train_times_seconds is None:
                    raise ValueError("uniform binning needs max_time_seconds or train_times_seconds.")
                max_time_seconds = float(np.nanmax(train_times_seconds))
            finite_edges = np.linspace(0.0, float(max_time_seconds), int(num_bins) + 1)
        finite_edges = _validated_edges(finite_edges, name="uniform_edges")
    elif strategy == "train_quantile":
        if train_times_seconds is None:
            raise ValueError("train_quantile binning requires train_times_seconds.")
        if num_bins is None or num_bins < 1:
            raise ValueError("train_quantile binning requires positive num_bins.")
        train_times = np.asarray(train_times_seconds, dtype=np.float64)
        train_times = train_times[np.isfinite(train_times) & (train_times > 0.0)]
        if len(train_times) == 0:
            raise ValueError("train_times_seconds contains no positive finite values.")
        probabilities = np.linspace(0.0, 1.0, int(num_bins) + 1)
        finite_edges = np.quantile(train_times, probabilities)
        finite_edges[0] = 0.0
        finite_edges = np.unique(finite_edges)
        if len(finite_edges) < 2:
            raise ValueError("train_quantile binning collapsed to fewer than two edges.")
        finite_edges = _validated_edges(finite_edges, name="train_quantile_edges")
    else:
        raise ValueError(
            "strategy must be one of: explicit, hybrid, uniform, train_quantile."
        )

    left_edges = finite_edges[:-1]
    right_edges = finite_edges[1:]
    if include_open_ended:
        left_edges = np.append(left_edges, finite_edges[-1])
        right_edges = np.append(right_edges, np.inf)
    return DiscreteTimeBinSpec(
        left_edges_seconds=left_edges,
        right_edges_seconds=right_edges,
        strategy=strategy,
        include_open_ended=bool(include_open_ended),
    )


def continuous_to_discrete_survival_labels(
    *,
    y_time_seconds: np.ndarray,
    y_event_observed: np.ndarray,
    y_lower_bound_seconds: np.ndarray,
    y_upper_bound_seconds: np.ndarray,
    bin_spec: DiscreteTimeBinSpec,
) -> DiscreteTimeSurvivalLabels:
    """Convert continuous right-censored labels into discrete hazard masks."""

    times = np.asarray(y_time_seconds, dtype=np.float64).reshape(-1)
    observed = np.asarray(y_event_observed, dtype=np.int64).reshape(-1)
    lower = np.asarray(y_lower_bound_seconds, dtype=np.float64).reshape(-1)
    upper = np.asarray(y_upper_bound_seconds, dtype=np.float64).reshape(-1)
    if not (len(times) == len(observed) == len(lower) == len(upper)):
        raise ValueError("Continuous survival label arrays must have the same length.")
    if np.any(~np.isfinite(times)) or np.any(times <= 0.0):
        raise ValueError("y_time_seconds must be positive and finite.")
    if np.any(~np.isfinite(lower)) or np.any(lower <= 0.0):
        raise ValueError("y_lower_bound_seconds must be positive and finite.")
    if np.any((observed != 0) & (observed != 1)):
        raise ValueError("y_event_observed must contain only 0/1.")

    n_rows = len(times)
    n_bins = bin_spec.num_bins
    event_bin_index = np.full(n_rows, -1, dtype=np.int64)
    censoring_bin_index = np.full(n_rows, -1, dtype=np.int64)
    at_risk_mask = np.zeros((n_rows, n_bins), dtype=bool)
    event_mask = np.zeros((n_rows, n_bins), dtype=bool)
    right_edges = bin_spec.right_edges_seconds

    for row in range(n_rows):
        if observed[row] == 1:
            if not np.isfinite(upper[row]) or not np.isclose(lower[row], upper[row]):
                raise ValueError("Observed survival samples must have equal finite lower/upper bounds.")
            event_bin = int(np.searchsorted(right_edges, times[row], side="left"))
            if event_bin >= n_bins:
                raise ValueError(
                    "Observed event time is outside the discrete grid and no open-ended bin is available."
                )
            event_bin_index[row] = event_bin
            at_risk_mask[row, : event_bin + 1] = True
            event_mask[row, event_bin] = True
        else:
            if not np.isinf(upper[row]):
                raise ValueError("Censored survival samples must use infinite upper bounds.")
            fully_survived = right_edges <= lower[row]
            n_full = int(np.sum(fully_survived))
            if n_full > 0:
                at_risk_mask[row, :n_full] = True
                censoring_bin_index[row] = n_full - 1

    return DiscreteTimeSurvivalLabels(
        event_bin_index=event_bin_index,
        censoring_bin_index=censoring_bin_index,
        at_risk_mask=at_risk_mask,
        event_mask=event_mask,
        valid_likelihood_mask=at_risk_mask.copy(),
    )


def hazards_to_survival(hazards: np.ndarray, *, epsilon: float = 1e-7) -> np.ndarray:
    """Convert discrete hazards to survival probabilities after each bin."""

    values = np.asarray(hazards, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("hazards must have shape (n_samples, n_bins).")
    if np.any(~np.isfinite(values)):
        raise ValueError("hazards contains non-finite values.")
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    log_survival = np.cumsum(np.log1p(-clipped), axis=1)
    return np.exp(log_survival)


def survival_at_horizons(
    survival_by_bin: np.ndarray,
    *,
    bin_spec: DiscreteTimeBinSpec,
    horizons_seconds: np.ndarray | list[float],
) -> np.ndarray:
    """Return stepwise survival probabilities for requested finite horizons."""

    survival = np.asarray(survival_by_bin, dtype=np.float64)
    horizons = np.asarray(horizons_seconds, dtype=np.float64).reshape(-1)
    if survival.ndim != 2 or survival.shape[1] != bin_spec.num_bins:
        raise ValueError("survival_by_bin must have shape (n_samples, n_bins).")
    if np.any(~np.isfinite(horizons)) or np.any(horizons <= 0.0):
        raise ValueError("horizons_seconds must contain positive finite values.")
    columns = []
    for horizon in horizons:
        index = int(np.searchsorted(bin_spec.right_edges_seconds, horizon, side="left"))
        index = min(index, bin_spec.num_bins - 1)
        columns.append(survival[:, index])
    return np.stack(columns, axis=1)


def median_survival_time_seconds(
    survival_by_bin: np.ndarray,
    *,
    bin_spec: DiscreteTimeBinSpec,
) -> np.ndarray:
    """Return the first finite right edge where survival drops below 0.5.

    When the median is not identifiable within finite bins, the returned value is
    capped at the largest finite grid edge so downstream ranking metrics remain
    finite and deterministic.
    """

    survival = np.asarray(survival_by_bin, dtype=np.float64)
    if survival.ndim != 2 or survival.shape[1] != bin_spec.num_bins:
        raise ValueError("survival_by_bin must have shape (n_samples, n_bins).")
    medians = np.full(len(survival), bin_spec.max_finite_time_seconds, dtype=np.float64)
    finite = np.isfinite(bin_spec.right_edges_seconds)
    for row_index, row in enumerate(survival):
        finite_crossings = np.flatnonzero((row <= 0.5) & finite)
        if finite_crossings.size:
            medians[row_index] = float(bin_spec.right_edges_seconds[finite_crossings[0]])
    return medians
