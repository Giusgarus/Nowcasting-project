"""XGBoost helpers for current-level persistence regression."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def require_xgboost_regressor():
    """Return ``xgboost.XGBRegressor`` or raise an actionable import error."""

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "XGBoost is required for this runner. Install/update the project "
            "environment so that the 'xgboost' Python package is available."
        ) from exc
    return XGBRegressor


def build_tabular_features(
    arrays: Mapping[str, np.ndarray],
    *,
    input_key: str,
    scalar_context_key: str | None = None,
) -> np.ndarray:
    """Build finite tabular features from signal-window arrays."""

    if input_key not in arrays:
        raise KeyError(f"Missing XGBoost input array: {input_key}")
    features = np.asarray(arrays[input_key], dtype=np.float32)
    if features.ndim == 3 and features.shape[-1] == 1:
        features = features[:, :, 0]
    if features.ndim != 2:
        raise ValueError("Current-level XGBoost input must have shape (N, L[, 1]).")
    blocks = [features]
    if scalar_context_key:
        if scalar_context_key not in arrays:
            raise KeyError(f"Missing scalar-context array: {scalar_context_key}")
        scalar_features = np.asarray(arrays[scalar_context_key], dtype=np.float32)
        if scalar_features.ndim != 2 or len(scalar_features) != len(features):
            raise ValueError("Scalar-context features must have shape (N, K).")
        blocks.append(scalar_features)
    output = np.concatenate(blocks, axis=1)
    if not np.isfinite(output).all():
        raise ValueError("Current-level XGBoost features must be finite.")
    return output


def fit_duration_regressor(
    x_train: np.ndarray,
    y_train_log1p_seconds: np.ndarray,
    *,
    parameters: Mapping[str, Any],
    seed: int,
) -> Any:
    """Fit one XGBoost regressor for log1p persistence duration."""

    XGBRegressor = require_xgboost_regressor()
    target = np.asarray(y_train_log1p_seconds, dtype=np.float32)
    if target.ndim != 1 or len(target) != len(x_train):
        raise ValueError("Target must have shape (N,).")
    model_parameters = dict(parameters)
    model_parameters.setdefault("objective", "reg:squarederror")
    model = XGBRegressor(
        random_state=int(seed),
        **model_parameters,
    )
    model.fit(np.asarray(x_train, dtype=np.float32), target)
    return model


def predict_duration_regressor(model: Any, x_values: np.ndarray) -> np.ndarray:
    """Predict log1p duration values with a fitted regressor."""

    predictions = np.asarray(model.predict(np.asarray(x_values, dtype=np.float32)))
    if predictions.ndim != 1 or len(predictions) != len(x_values):
        raise ValueError("XGBoost duration predictions must have shape (N,).")
    if not np.isfinite(predictions).all():
        raise ValueError("XGBoost duration predictions must be finite.")
    return predictions.astype(np.float32)


def save_duration_regressor(
    path: Path,
    *,
    model: Any,
    metadata: Mapping[str, Any],
) -> None:
    """Persist a fitted duration regressor and reproducibility metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": dict(metadata)}, path)


def load_duration_regressor(path: Path) -> dict[str, Any]:
    """Load a saved current-level XGBoost checkpoint."""

    return joblib.load(path)
