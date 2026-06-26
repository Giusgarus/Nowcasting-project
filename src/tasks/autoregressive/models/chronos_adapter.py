"""Adapter utilities for pretrained Chronos zero-shot forecasting."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.tasks.autoregressive.evaluation.forecast_metrics import inverse_context_standardization
from src.utils.device import select_device

SUPPORTED_VARIANTS = {"raw", "context_standard_optional"}


def resolve_chronos_device(configured: str) -> str:
    """Resolve an explicit or automatic Chronos inference device."""

    if configured == "auto":
        return select_device()
    if configured.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Chronos requested CUDA, but PyTorch cannot access CUDA.")
        return configured
    if configured == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("Chronos requested MPS, but PyTorch cannot access MPS.")
        return configured
    if configured == "cpu":
        return configured
    raise ValueError(f"Unsupported Chronos device: {configured}")


def load_chronos_pipeline(
    model_id: str,
    *,
    device: str,
    torch_dtype: str,
    pipeline_class: type | None = None,
) -> tuple[Any, str]:
    """Load an original Chronos pipeline, falling back from MPS to CPU."""

    if pipeline_class is None:
        try:
            from chronos import ChronosPipeline
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Chronos zero-shot evaluation requires `chronos-forecasting`. "
                "Run `conda env update -n Nowcasting -f environment.yml --prune`."
            ) from error
        pipeline_class = ChronosPipeline

    kwargs = {"device_map": device, "dtype": torch_dtype}
    try:
        return _load_pipeline_with_dtype_fallback(
            pipeline_class,
            model_id,
            kwargs,
        ), device
    except Exception as error:
        if device != "mps":
            raise RuntimeError(
                f"Could not load Chronos model {model_id!r} on {device}: {error}"
            ) from error
        try:
            return (
                _load_pipeline_with_dtype_fallback(
                    pipeline_class,
                    model_id,
                    {"device_map": "cpu", "dtype": torch_dtype},
                ),
                "cpu",
            )
        except Exception as cpu_error:
            raise RuntimeError(
                f"Chronos failed on MPS and CPU for model {model_id!r}: {cpu_error}"
            ) from cpu_error


def _load_pipeline_with_dtype_fallback(
    pipeline_class: type,
    model_id: str,
    kwargs: dict[str, str],
) -> Any:
    """Load Chronos using new ``dtype`` API, falling back for old packages."""

    try:
        return pipeline_class.from_pretrained(model_id, **kwargs)
    except TypeError as error:
        if "dtype" not in str(error):
            raise
        legacy_kwargs = dict(kwargs)
        legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
        return pipeline_class.from_pretrained(model_id, **legacy_kwargs)


def variant_contexts_and_targets(
    arrays: Mapping[str, np.ndarray],
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Chronos contexts and variant-scale targets."""

    if variant == "raw":
        context_key, target_key = "X_raw", "y_raw"
    elif variant == "context_standard_optional":
        context_key, target_key = "X_context_standard", "y_context_standard"
    else:
        raise ValueError(f"Unsupported Chronos variant: {variant}")
    contexts = np.asarray(arrays[context_key], dtype=np.float32)
    targets = np.asarray(arrays[target_key], dtype=np.float32)
    if contexts.ndim != 3 or contexts.shape[-1] != 1:
        raise ValueError("Chronos contexts must have shape (N, context_length, 1).")
    if targets.ndim != 2 or len(targets) != len(contexts):
        raise ValueError("Chronos targets must have shape (N, prediction_length).")
    return contexts[:, :, 0], targets


def sample_forecast_statistics(
    sample_forecasts: np.ndarray | torch.Tensor,
    quantiles: Sequence[float],
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Convert Chronos sample paths into median and quantile trajectories."""

    if isinstance(sample_forecasts, torch.Tensor):
        samples = sample_forecasts.detach().cpu().numpy()
    else:
        samples = np.asarray(sample_forecasts)
    if samples.ndim != 3:
        raise ValueError(
            "Chronos sample forecasts must have shape "
            "(batch, num_samples, prediction_length)."
        )
    if samples.shape[1] < 1 or not np.isfinite(samples).all():
        raise ValueError("Chronos sample forecasts must be finite and non-empty.")
    requested = [float(value) for value in quantiles]
    if any(value < 0 or value > 1 for value in requested):
        raise ValueError("Chronos quantiles must lie in [0, 1].")
    point = np.median(samples, axis=1).astype(np.float32)
    quantile_values = np.quantile(samples, requested, axis=1).astype(np.float32)
    return point, {
        quantile: quantile_values[index]
        for index, quantile in enumerate(requested)
    }


def predict_chronos_batches(
    pipeline: Any,
    contexts: np.ndarray,
    *,
    prediction_length: int,
    num_samples: int,
    batch_size: int,
    quantiles: Sequence[float],
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Run zero-shot Chronos inference in deterministic input-order batches."""

    contexts = np.asarray(contexts, dtype=np.float32)
    if contexts.ndim != 2 or len(contexts) == 0:
        raise ValueError("Chronos contexts must have shape (N, context_length).")
    if prediction_length < 1 or num_samples < 1 or batch_size < 1:
        raise ValueError("Prediction length, samples, and batch size must be positive.")
    point_batches: list[np.ndarray] = []
    quantile_batches: dict[float, list[np.ndarray]] = {
        float(quantile): [] for quantile in quantiles
    }
    for start in range(0, len(contexts), batch_size):
        batch = torch.from_numpy(contexts[start : start + batch_size])
        samples = pipeline.predict(
            batch,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )
        point, batch_quantiles = sample_forecast_statistics(samples, quantiles)
        point_batches.append(point)
        for quantile, values in batch_quantiles.items():
            quantile_batches[quantile].append(values)
    return np.concatenate(point_batches), {
        quantile: np.concatenate(parts)
        for quantile, parts in quantile_batches.items()
    }


def predictions_to_raw_scale(
    predictions: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    variant: str,
) -> np.ndarray:
    """Return raw-scale trajectories for a supported Chronos variant."""

    if variant == "raw":
        return np.asarray(predictions)
    if variant == "context_standard_optional":
        return inverse_context_standardization(
            predictions,
            arrays["scaling_mean"],
            arrays["scaling_std"],
        )
    raise ValueError(f"Unsupported Chronos variant: {variant}")


def build_chronos_prediction_tables(
    metadata: pd.DataFrame,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    quantiles_raw: Mapping[float, np.ndarray],
    *,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build switch-compatible long predictions and one-row-per-window output."""

    metadata = metadata.reset_index(drop=True)
    if len(metadata) != len(y_true_raw) or y_true_raw.shape != y_pred_raw.shape:
        raise ValueError("Chronos predictions and metadata are not aligned.")
    copied_columns = [
        "window_id",
        "event_id",
        "dataset_id",
        "dataset_name",
        "quality_flag",
        "split",
        "input_start_time",
        "input_end_time",
        "target_start_time",
        "target_end_time",
        "scaling_mean",
        "scaling_std",
    ]
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for index, row in metadata.iterrows():
        common = {column: row[column] for column in copied_columns}
        common.update(
            {
                "model_family": "chronos",
                "architecture": "chronos_zero_shot",
                "variant": variant,
                "mode": "zero_shot",
            }
        )
        wide = dict(common)
        for horizon_index in range(y_true_raw.shape[1]):
            step = horizon_index + 1
            observed = float(y_true_raw[index, horizon_index])
            predicted = float(y_pred_raw[index, horizon_index])
            error = predicted - observed
            long_row = {
                **common,
                "horizon_step": step,
                "y_true_raw": observed,
                "y_pred_raw": predicted,
                "error": error,
                "absolute_error": abs(error),
                "squared_error": error**2,
            }
            wide[f"y_true_raw_{step}"] = observed
            wide[f"y_pred_raw_{step}"] = predicted
            wide[f"error_{step}"] = error
            wide[f"abs_error_{step}"] = abs(error)
            for quantile, values in quantiles_raw.items():
                label = f"q{int(round(float(quantile) * 100)):02d}"
                value = float(values[index, horizon_index])
                long_row[f"y_pred_{label}"] = value
                wide[f"y_pred_{label}_{step}"] = value
            long_rows.append(long_row)
        wide_rows.append(wide)
    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows)
