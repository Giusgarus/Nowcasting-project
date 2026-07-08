"""XGBoost helpers for autoregressive multi-horizon forecasting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def flatten_forecast_context(values: np.ndarray) -> np.ndarray:
    """Return a 2D tabular feature matrix from univariate context windows."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError("Autoregressive XGBoost inputs must have shape (N, L[, 1]).")
    if not np.isfinite(array).all():
        raise ValueError("Autoregressive XGBoost inputs must be finite.")
    return array


def fit_horizon_regressors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    parameters: Mapping[str, Any],
    seed: int,
) -> list[Any]:
    """Fit one independent XGBoost regressor for each forecast horizon."""

    XGBRegressor = require_xgboost_regressor()
    features = flatten_forecast_context(x_train)
    targets = np.asarray(y_train, dtype=np.float32)
    if targets.ndim != 2 or len(targets) != len(features):
        raise ValueError("Targets must have shape (N, prediction_length).")
    model_parameters = dict(parameters)
    model_parameters.setdefault("objective", "reg:squarederror")
    models = []
    for horizon_index in range(targets.shape[1]):
        model = XGBRegressor(
            random_state=int(seed) + horizon_index,
            **model_parameters,
        )
        model.fit(features, targets[:, horizon_index])
        models.append(model)
    return models


def predict_horizon_regressors(
    models: Sequence[Any],
    x_values: np.ndarray,
) -> np.ndarray:
    """Predict a full trajectory using one regressor per horizon."""

    if not models:
        raise ValueError("At least one horizon regressor is required.")
    features = flatten_forecast_context(x_values)
    predictions = [model.predict(features) for model in models]
    return np.stack(predictions, axis=1).astype(np.float32)


def save_horizon_regressors(
    path: Path,
    *,
    models: Sequence[Any],
    metadata: Mapping[str, Any],
) -> None:
    """Persist fitted horizon regressors and reproducibility metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": list(models), "metadata": dict(metadata)}, path)


def load_horizon_regressors(path: Path) -> dict[str, Any]:
    """Load a saved autoregressive XGBoost checkpoint."""

    return joblib.load(path)
