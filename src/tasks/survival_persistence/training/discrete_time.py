"""Training/evaluation helpers for discrete-time survival models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.tasks.survival_persistence.adapters.discrete_time import (
    DiscreteTimeBinSpec,
    DiscreteTimeSurvivalLabels,
    continuous_to_discrete_survival_labels,
    hazards_to_survival,
    median_survival_time_seconds,
    survival_at_horizons,
)


SUPPORTED_SEQUENCE_REPRESENTATIONS = {
    "raw": ("X_raw",),
    "relative_to_threshold": ("X_relative_to_threshold",),
    "relative_to_current": ("X_relative_to_current",),
    "multichannel": (
        "X_relative_to_threshold",
        "X_relative_to_current",
        "X_raw",
    ),
}


def build_sequence_features(
    arrays: Mapping[str, np.ndarray],
    *,
    representation: str,
) -> np.ndarray:
    """Return sequence features with shape ``(n_samples, channels, context_length)``."""

    if representation not in SUPPORTED_SEQUENCE_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported sequence representation {representation!r}. "
            f"Supported: {sorted(SUPPORTED_SEQUENCE_REPRESENTATIONS)}"
        )
    channels = []
    for key in SUPPORTED_SEQUENCE_REPRESENTATIONS[representation]:
        if key not in arrays:
            raise ValueError(f"Missing required sequence array: {key}")
        values = np.asarray(arrays[key], dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"{key} must have shape (n_samples, context_length).")
        channels.append(values)
    stacked = np.stack(channels, axis=1).astype(np.float32)
    if not np.isfinite(stacked).all():
        raise ValueError("Sequence features contain NaN or infinity.")
    return stacked


def sample_weights(
    arrays: Mapping[str, np.ndarray],
    *,
    mode: str,
) -> np.ndarray:
    """Return finite positive sample weights."""

    if mode == "uniform":
        return np.ones(len(arrays["y_time_seconds"]), dtype=np.float32)
    if mode == "event_balanced":
        values = np.asarray(arrays["event_balanced_weight"], dtype=np.float32)
        if values.ndim != 1 or len(values) != len(arrays["y_time_seconds"]):
            raise ValueError("event_balanced_weight has invalid shape.")
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("event_balanced_weight must be positive and finite.")
        return values
    raise ValueError("sample_weighting must be 'uniform' or 'event_balanced'.")


class DiscreteTimeSurvivalTorchDataset(Dataset):
    """Torch dataset wrapping canonical survival arrays plus discrete labels."""

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        bin_spec: DiscreteTimeBinSpec,
        sequence_representation: str,
        use_scalar_context: bool,
        sample_weighting: str,
    ) -> None:
        self.x_sequence = build_sequence_features(
            arrays,
            representation=sequence_representation,
        )
        if use_scalar_context:
            if "scalar_context_features" not in arrays:
                raise ValueError("scalar_context_features are required by this config.")
            scalar = np.asarray(arrays["scalar_context_features"], dtype=np.float32)
            if scalar.ndim != 2 or len(scalar) != len(self.x_sequence):
                raise ValueError("scalar_context_features have invalid shape.")
            if not np.isfinite(scalar).all():
                raise ValueError("scalar_context_features contain NaN or infinity.")
            self.x_scalar: np.ndarray | None = scalar
        else:
            self.x_scalar = None
        self.labels = continuous_to_discrete_survival_labels(
            y_time_seconds=np.asarray(arrays["y_time_seconds"]),
            y_event_observed=np.asarray(arrays["y_event_observed"]),
            y_lower_bound_seconds=np.asarray(arrays["y_lower_bound_seconds"]),
            y_upper_bound_seconds=np.asarray(arrays["y_upper_bound_seconds"]),
            bin_spec=bin_spec,
        )
        self.weights = sample_weights(arrays, mode=sample_weighting)
        self.y_time_seconds = np.asarray(arrays["y_time_seconds"], dtype=np.float32)
        self.y_event_observed = np.asarray(arrays["y_event_observed"], dtype=np.int64)
        self.y_lower_bound_seconds = np.asarray(arrays["y_lower_bound_seconds"], dtype=np.float32)
        self.y_upper_bound_seconds = np.asarray(arrays["y_upper_bound_seconds"], dtype=np.float32)
        if len(self.weights) != len(self.x_sequence):
            raise ValueError("weights and sequence features are not aligned.")

    def __len__(self) -> int:
        """Return dataset size."""

        return int(len(self.x_sequence))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one sample as tensors."""

        item = {
            "x_sequence": torch.as_tensor(self.x_sequence[index], dtype=torch.float32),
            "event_mask": torch.as_tensor(self.labels.event_mask[index], dtype=torch.bool),
            "at_risk_mask": torch.as_tensor(self.labels.at_risk_mask[index], dtype=torch.bool),
            "sample_weight": torch.as_tensor(self.weights[index], dtype=torch.float32),
            "y_time_seconds": torch.as_tensor(self.y_time_seconds[index], dtype=torch.float32),
            "y_event_observed": torch.as_tensor(self.y_event_observed[index], dtype=torch.int64),
        }
        if self.x_scalar is not None:
            item["x_scalar"] = torch.as_tensor(self.x_scalar[index], dtype=torch.float32)
        return item


def logits_to_prediction_arrays(
    hazard_logits: np.ndarray,
    *,
    bin_spec: DiscreteTimeBinSpec,
    horizons_seconds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert hazard logits to hazards, survival, horizon survival, and median."""

    logits = np.asarray(hazard_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != bin_spec.num_bins:
        raise ValueError("hazard_logits must have shape (n_samples, n_bins).")
    hazards = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    survival_by_bin = hazards_to_survival(hazards)
    horizon_survival = survival_at_horizons(
        survival_by_bin,
        bin_spec=bin_spec,
        horizons_seconds=horizons_seconds,
    )
    median = median_survival_time_seconds(survival_by_bin, bin_spec=bin_spec)
    return {
        "hazards": hazards.astype(np.float32),
        "survival_by_bin": survival_by_bin.astype(np.float32),
        "survival_at_horizons": horizon_survival.astype(np.float32),
        "predicted_median_remaining_seconds": median.astype(np.float32),
    }


@torch.no_grad()
def evaluate_discrete_time_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    loss_fn: Any,
) -> dict[str, Any]:
    """Evaluate loss and collect hazard logits for one split."""

    model.eval()
    losses = []
    weights = []
    logits_chunks = []
    time_chunks = []
    observed_chunks = []
    for batch in loader:
        x_sequence = batch["x_sequence"].to(device)
        x_scalar = batch.get("x_scalar")
        if x_scalar is not None:
            x_scalar = x_scalar.to(device)
        event_mask = batch["event_mask"].to(device)
        at_risk_mask = batch["at_risk_mask"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        logits = model(x_sequence, x_scalar)
        loss = loss_fn(
            logits,
            event_mask=event_mask,
            at_risk_mask=at_risk_mask,
            sample_weight=sample_weight,
        )
        batch_weight = float(sample_weight.sum().detach().cpu())
        losses.append(float(loss.detach().cpu()) * batch_weight)
        weights.append(batch_weight)
        logits_chunks.append(logits.detach().cpu().numpy())
        time_chunks.append(batch["y_time_seconds"].detach().cpu().numpy())
        observed_chunks.append(batch["y_event_observed"].detach().cpu().numpy())
    total_weight = float(np.sum(weights))
    return {
        "loss": float(np.sum(losses) / total_weight) if total_weight > 0.0 else np.nan,
        "hazard_logits": np.concatenate(logits_chunks, axis=0) if logits_chunks else np.empty((0, 0)),
        "y_time_seconds": np.concatenate(time_chunks, axis=0) if time_chunks else np.empty(0),
        "y_event_observed": np.concatenate(observed_chunks, axis=0) if observed_chunks else np.empty(0),
    }


def make_discrete_time_predictions_frame(
    *,
    split: str,
    metadata: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    prediction_arrays: Mapping[str, np.ndarray],
    horizons_seconds: np.ndarray,
    model_family: str,
    model_id: str,
    feature_set: str,
    sample_weighting: str,
) -> pd.DataFrame:
    """Build the persisted prediction dataframe for one split."""

    required_metadata = [
        "sample_time",
        "dataset_name",
        "dataset_id",
        "global_event_id",
        "global_window_id",
        "current_above_threshold",
        "event_balanced_weight",
        "elapsed_since_event_start_seconds",
        "elapsed_event_samples",
        "current_signal",
    ]
    missing = [column for column in required_metadata if column not in metadata.columns]
    if missing:
        raise ValueError(f"Prediction metadata missing columns: {missing}")
    frame = metadata.loc[:, required_metadata].copy()
    frame.insert(0, "split", split)
    frame["y_time_seconds"] = np.asarray(arrays["y_time_seconds"], dtype=np.float32)
    frame["y_event_observed"] = np.asarray(arrays["y_event_observed"], dtype=np.int64)
    frame["y_lower_bound_seconds"] = np.asarray(arrays["y_lower_bound_seconds"], dtype=np.float32)
    frame["y_upper_bound_seconds"] = np.asarray(arrays["y_upper_bound_seconds"], dtype=np.float32)
    frame["model_family"] = model_family
    frame["model_id"] = model_id
    frame["feature_set"] = feature_set
    frame["sample_weighting"] = sample_weighting
    frame["predicted_median_remaining_seconds"] = prediction_arrays[
        "predicted_median_remaining_seconds"
    ]
    survival = np.asarray(prediction_arrays["survival_at_horizons"], dtype=np.float32)
    if survival.shape != (len(frame), len(horizons_seconds)):
        raise ValueError("survival_at_horizons shape does not match predictions frame.")
    for index, horizon in enumerate(horizons_seconds):
        frame[f"survival_probability_{int(horizon)}s"] = survival[:, index]
    return frame
