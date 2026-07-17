"""XGBoost-AFT feature, label, and prediction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, gamma, log, pi, sin
from typing import Any

import numpy as np


FEATURE_SET_SCALAR = "scalar_context"
FEATURE_SET_RAW = "raw_flattened"
FEATURE_SET_RAW_PLUS_SCALAR = "raw_plus_scalar"
FEATURE_SET_THRESHOLD_PLUS_SCALAR = "relative_threshold_plus_scalar"
FEATURE_SET_CURRENT_PLUS_SCALAR = "relative_current_plus_scalar"
SUPPORTED_FEATURE_SETS = {
    FEATURE_SET_SCALAR,
    FEATURE_SET_RAW,
    FEATURE_SET_RAW_PLUS_SCALAR,
    FEATURE_SET_THRESHOLD_PLUS_SCALAR,
    FEATURE_SET_CURRENT_PLUS_SCALAR,
}
FORBIDDEN_FEATURE_KEYS = {
    "y_time_seconds",
    "y_event_observed",
    "y_lower_bound_seconds",
    "y_upper_bound_seconds",
    "y_time_samples",
    "event_id",
    "window_id",
    "global_event_id",
    "global_window_id",
    "event_sample_count",
    "event_balanced_weight",
}


@dataclass(frozen=True)
class FeatureMatrix:
    """Feature matrix plus deterministic names."""

    values: np.ndarray
    names: list[str]


def require_xgboost() -> Any:
    """Return the imported XGBoost module or raise an actionable error."""

    try:
        import xgboost as xgb
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "XGBoost is required for the survival_persistence xgboost_aft "
            "baseline. Install/sync the project environment so that the "
            "'xgboost' Python package is available."
        ) from error
    return xgb


def build_feature_matrix(
    arrays: dict[str, np.ndarray],
    *,
    feature_set: str,
) -> FeatureMatrix:
    """Build a deterministic tabular feature matrix from canonical arrays."""

    if feature_set not in SUPPORTED_FEATURE_SETS:
        raise ValueError(
            f"Unsupported feature_set={feature_set!r}. "
            f"Supported: {sorted(SUPPORTED_FEATURE_SETS)}"
        )
    parts: list[np.ndarray] = []
    names: list[str] = []

    def add_flat_context(key: str, prefix: str) -> None:
        if key not in arrays:
            raise ValueError(f"Missing required array {key!r} for {feature_set}.")
        values = np.asarray(arrays[key], dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"{key} must have shape (n_samples, context_length).")
        parts.append(values)
        context_length = values.shape[1]
        names.extend(
            f"{prefix}_t_minus_{context_length - index - 1}"
            for index in range(context_length)
        )

    def add_scalar_context() -> None:
        if "scalar_context_features" not in arrays:
            raise ValueError("Missing scalar_context_features.")
        values = np.asarray(arrays["scalar_context_features"], dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("scalar_context_features must be two-dimensional.")
        stored_names = arrays.get("scalar_context_feature_names")
        if stored_names is None:
            scalar_names = [f"scalar_context_{index}" for index in range(values.shape[1])]
        else:
            scalar_names = [str(name) for name in np.asarray(stored_names).tolist()]
            if len(scalar_names) != values.shape[1]:
                raise ValueError("scalar_context_feature_names length mismatch.")
        parts.append(values)
        names.extend(scalar_names)

    if feature_set == FEATURE_SET_SCALAR:
        add_scalar_context()
    elif feature_set == FEATURE_SET_RAW:
        add_flat_context("X_raw", "raw_signal")
    elif feature_set == FEATURE_SET_RAW_PLUS_SCALAR:
        add_flat_context("X_raw", "raw_signal")
        add_scalar_context()
    elif feature_set == FEATURE_SET_THRESHOLD_PLUS_SCALAR:
        add_flat_context("X_relative_to_threshold", "relative_to_threshold")
        add_scalar_context()
    elif feature_set == FEATURE_SET_CURRENT_PLUS_SCALAR:
        add_flat_context("X_relative_to_current", "relative_to_current")
        add_scalar_context()

    matrix = np.concatenate(parts, axis=1).astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Feature matrix must be non-empty and two-dimensional.")
    if len(names) != matrix.shape[1]:
        raise ValueError("Feature name count does not match feature columns.")
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique.")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains NaN or infinity.")
    forbidden_used = sorted(set(names) & FORBIDDEN_FEATURE_KEYS)
    if forbidden_used:
        raise ValueError(f"Forbidden fields entered feature matrix: {forbidden_used}")
    return FeatureMatrix(values=matrix, names=names)


def validate_feature_compatibility(reference: FeatureMatrix, candidate: FeatureMatrix) -> None:
    """Fail if two split feature matrices are not schema-compatible."""

    if reference.values.shape[1] != candidate.values.shape[1]:
        raise ValueError("Feature dimensions differ between splits.")
    if reference.names != candidate.names:
        raise ValueError("Feature ordering differs between splits.")


def aft_label_bounds(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return validated XGBoost-AFT lower and upper label bounds."""

    lower = np.asarray(arrays["y_lower_bound_seconds"], dtype=np.float64)
    upper = np.asarray(arrays["y_upper_bound_seconds"], dtype=np.float64)
    observed = np.asarray(arrays["y_event_observed"], dtype=np.int64)
    y_time = np.asarray(arrays["y_time_seconds"], dtype=np.float64)
    if lower.ndim != 1 or upper.ndim != 1 or observed.ndim != 1:
        raise ValueError("AFT labels must be one-dimensional.")
    if not (len(lower) == len(upper) == len(observed) == len(y_time)):
        raise ValueError("AFT label arrays have inconsistent lengths.")
    if np.any(~np.isfinite(lower)) or np.any(lower <= 0.0):
        raise ValueError("AFT lower bounds must be finite and strictly positive.")
    observed_mask = observed.astype(bool)
    censored_mask = ~observed_mask
    if not np.allclose(lower[observed_mask], upper[observed_mask]):
        raise ValueError("Observed AFT samples must have equal lower/upper bounds.")
    if not np.isfinite(upper[observed_mask]).all():
        raise ValueError("Observed AFT upper bounds must be finite.")
    if censored_mask.any() and not np.isinf(upper[censored_mask]).all():
        raise ValueError("Right-censored AFT upper bounds must remain infinite.")
    if not np.allclose(lower, y_time):
        raise ValueError("AFT lower bounds must match y_time_seconds.")
    return lower.astype(np.float32), upper.astype(np.float32)


def sample_weights(
    arrays: dict[str, np.ndarray],
    *,
    mode: str,
) -> np.ndarray | None:
    """Return optional training sample weights from canonical arrays."""

    if mode == "uniform":
        return None
    if mode == "event_balanced":
        weights = np.asarray(arrays["event_balanced_weight"], dtype=np.float32)
        if weights.ndim != 1 or len(weights) != len(arrays["y_time_seconds"]):
            raise ValueError("event_balanced_weight has invalid shape.")
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("event_balanced_weight must be positive and finite.")
        return weights
    raise ValueError("sample_weighting must be 'uniform' or 'event_balanced'.")


def build_dmatrix(
    arrays: dict[str, np.ndarray],
    *,
    feature_set: str,
    sample_weighting: str = "uniform",
):
    """Build an XGBoost DMatrix with AFT lower/upper labels."""

    xgb = require_xgboost()
    feature_matrix = build_feature_matrix(arrays, feature_set=feature_set)
    lower, upper = aft_label_bounds(arrays)
    dmatrix = xgb.DMatrix(feature_matrix.values, feature_names=feature_matrix.names)
    dmatrix.set_float_info("label_lower_bound", lower)
    dmatrix.set_float_info("label_upper_bound", upper)
    weights = sample_weights(arrays, mode=sample_weighting)
    if weights is not None:
        dmatrix.set_weight(weights)
    return dmatrix, feature_matrix


def _standard_normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))


def _standard_normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def aft_cdf(z: np.ndarray, *, distribution: str) -> np.ndarray:
    """Return the standard AFT error CDF used by XGBoost distributions."""

    values = np.asarray(z, dtype=np.float64)
    if distribution == "normal":
        return _standard_normal_cdf(values)
    if distribution == "logistic":
        return 1.0 / (1.0 + np.exp(-values))
    if distribution == "extreme":
        clipped = np.clip(values, -60.0, 60.0)
        return 1.0 - np.exp(-np.exp(clipped))
    raise ValueError(f"Unsupported AFT distribution: {distribution}")


def aft_pdf(z: np.ndarray, *, distribution: str) -> np.ndarray:
    """Return the standard AFT error PDF used for nloglik diagnostics."""

    values = np.asarray(z, dtype=np.float64)
    if distribution == "normal":
        return _standard_normal_pdf(values)
    if distribution == "logistic":
        clipped = np.clip(values, -60.0, 60.0)
        exp_neg = np.exp(-clipped)
        return exp_neg / np.square(1.0 + exp_neg)
    if distribution == "extreme":
        clipped = np.clip(values, -60.0, 60.0)
        return np.exp(clipped - np.exp(clipped))
    raise ValueError(f"Unsupported AFT distribution: {distribution}")


def aft_quantile(probability: float, *, distribution: str) -> float:
    """Return the standard-error quantile for supported AFT distributions."""

    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be in (0, 1).")
    if distribution == "normal":
        if np.isclose(p, 0.5):
            return 0.0
        try:
            from scipy.stats import norm
        except ImportError as error:  # pragma: no cover - scipy is project dependency
            raise RuntimeError("scipy is required for non-median normal quantiles.") from error
        return float(norm.ppf(p))
    if distribution == "logistic":
        return float(np.log(p / (1.0 - p)))
    if distribution == "extreme":
        return float(np.log(-np.log1p(-p)))
    raise ValueError(f"Unsupported AFT distribution: {distribution}")


def predicted_median_seconds(
    location: np.ndarray,
    *,
    scale: float,
    distribution: str,
) -> np.ndarray:
    """Return the conditional median remaining duration in seconds."""

    q50 = aft_quantile(0.5, distribution=distribution)
    return np.exp(np.asarray(location, dtype=np.float64) + float(scale) * q50)


def predicted_mean_seconds(
    location: np.ndarray,
    *,
    scale: float,
    distribution: str,
) -> tuple[np.ndarray | None, str]:
    """Return the conditional mean duration where the distribution defines it."""

    sigma = float(scale)
    mu = np.asarray(location, dtype=np.float64)
    if distribution == "normal":
        return np.exp(mu + 0.5 * sigma * sigma), ""
    if distribution == "extreme":
        return np.exp(mu) * gamma(1.0 + sigma), ""
    if distribution == "logistic":
        if sigma >= 1.0:
            return None, "undefined for logistic AFT scale >= 1"
        multiplier = pi * sigma / sin(pi * sigma)
        return np.exp(mu) * multiplier, ""
    raise ValueError(f"Unsupported AFT distribution: {distribution}")


def survival_probability_at_horizon(
    location: np.ndarray,
    horizons_seconds: np.ndarray,
    *,
    scale: float,
    distribution: str,
) -> np.ndarray:
    """Return S(u | X) for each row and horizon."""

    mu = np.asarray(location, dtype=np.float64).reshape(-1, 1)
    horizons = np.asarray(horizons_seconds, dtype=np.float64).reshape(1, -1)
    if np.any(~np.isfinite(horizons)) or np.any(horizons <= 0.0):
        raise ValueError("horizons_seconds must contain positive finite values.")
    z = (np.log(horizons) - mu) / float(scale)
    probabilities = 1.0 - aft_cdf(z, distribution=distribution)
    return np.clip(probabilities, 0.0, 1.0)


def aft_negative_log_likelihood(
    location: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    *,
    scale: float,
    distribution: str,
) -> float:
    """Compute mean AFT negative log-likelihood with right-censoring support."""

    mu = np.asarray(location, dtype=np.float64)
    lower = np.asarray(lower_bound, dtype=np.float64)
    upper = np.asarray(upper_bound, dtype=np.float64)
    sigma = float(scale)
    if sigma <= 0.0:
        raise ValueError("scale must be positive.")
    if np.any(lower <= 0.0) or np.any(~np.isfinite(lower)):
        raise ValueError("lower bounds must be positive and finite.")
    z_lower = (np.log(lower) - mu) / sigma
    observed = np.isfinite(upper) & np.isclose(lower, upper)
    right_censored = np.isinf(upper)
    interval = ~(observed | right_censored)
    likelihood = np.empty_like(lower, dtype=np.float64)
    likelihood[observed] = aft_pdf(z_lower[observed], distribution=distribution) / sigma
    likelihood[right_censored] = 1.0 - aft_cdf(
        z_lower[right_censored],
        distribution=distribution,
    )
    if interval.any():
        z_upper = (np.log(upper[interval]) - mu[interval]) / sigma
        likelihood[interval] = aft_cdf(
            z_upper,
            distribution=distribution,
        ) - aft_cdf(z_lower[interval], distribution=distribution)
    return float(-np.log(np.clip(likelihood, 1e-12, 1.0)).mean())
