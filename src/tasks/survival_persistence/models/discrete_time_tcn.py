"""Causal TCN hazard model for discrete-time survival persistence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


MODEL_ID_DISCRETE_TIME_TCN = "discrete_time_tcn"


@dataclass(frozen=True)
class DiscreteTimeTCNConfig:
    """Configuration for the discrete-time survival TCN."""

    context_length: int
    num_bins: int
    input_channels: int = 1
    hidden_channels: tuple[int, ...] = (16, 16, 32)
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1
    activation: str = "gelu"
    normalization: str = "group_norm"
    pooling: str = "last"
    use_scalar_context: bool = True
    scalar_context_dim: int = 18
    scalar_hidden_dim: int = 32


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError("activation must be one of: gelu, relu, silu.")


def _normalization(name: str, channels: int) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name == "group_norm":
        return nn.GroupNorm(num_groups=1, num_channels=channels)
    raise ValueError("normalization must be one of: none, group_norm.")


class CausalConv1d(nn.Module):
    """One-dimensional convolution with left-only temporal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive.")
        if dilation < 1:
            raise ValueError("dilation must be positive.")
        self.left_padding = int((kernel_size - 1) * dilation)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a length-preserving causal convolution."""

        padded = F.pad(inputs, (self.left_padding, 0))
        return self.conv(padded)


class CausalResidualBlock(nn.Module):
    """Residual block made of two causal dilated convolutions."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: str,
        normalization: str,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            _normalization(normalization, out_channels),
            _activation(activation),
            nn.Dropout(dropout),
            CausalConv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            _normalization(normalization, out_channels),
            _activation(activation),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return residual causal features."""

        return self.net(inputs) + self.residual(inputs)


class ScalarContextEncoder(nn.Module):
    """Small MLP for train-standardized scalar context features."""

    def __init__(self, input_dim: int, hidden_dim: int, *, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive when scalar context is enabled.")
        self.output_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode scalar context."""

        return self.net(inputs)


def sequence_to_channels(
    inputs: torch.Tensor,
    *,
    context_length: int,
    input_channels: int,
) -> torch.Tensor:
    """Normalize sequence input to ``(batch, channels, context_length)``."""

    if inputs.ndim == 2:
        values = inputs.unsqueeze(1)
    elif inputs.ndim == 3 and inputs.shape[1] == input_channels:
        values = inputs
    elif inputs.ndim == 3 and inputs.shape[-1] == input_channels:
        values = inputs.transpose(1, 2)
    else:
        raise ValueError(
            "sequence input must be (B, L), (B, C, L), or (B, L, C)."
        )
    if values.shape[-1] != context_length:
        raise ValueError("sequence context length does not match model config.")
    if values.shape[1] != input_channels:
        raise ValueError("sequence input channel count does not match model config.")
    return values


def discrete_time_survival_nll(
    hazard_logits: torch.Tensor,
    *,
    event_mask: torch.Tensor,
    at_risk_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return masked discrete-time negative log-likelihood.

    At-risk bins before an observed event contribute ``log(1 - h_k)``. The
    observed event bin contributes ``log(h_k)``. Right-censored samples
    contribute only fully observed survived intervals.
    """

    if hazard_logits.shape != event_mask.shape or hazard_logits.shape != at_risk_mask.shape:
        raise ValueError("hazard_logits, event_mask, and at_risk_mask must have the same shape.")
    targets = event_mask.to(dtype=hazard_logits.dtype)
    mask = at_risk_mask.to(dtype=hazard_logits.dtype)
    elementwise = F.binary_cross_entropy_with_logits(
        hazard_logits,
        targets,
        reduction="none",
    )
    per_sample = (elementwise * mask).sum(dim=1)
    valid_sample = mask.sum(dim=1) > 0
    if sample_weight is None:
        weights = valid_sample.to(dtype=hazard_logits.dtype)
    else:
        weights = sample_weight.to(dtype=hazard_logits.dtype).reshape(-1) * valid_sample.to(
            dtype=hazard_logits.dtype
        )
    denominator = torch.clamp(weights.sum(), min=torch.finfo(hazard_logits.dtype).eps)
    return (per_sample * weights).sum() / denominator


class DiscreteTimeTCNSurvivalModel(nn.Module):
    """Causal TCN that outputs one hazard logit per discrete survival bin."""

    model_id = MODEL_ID_DISCRETE_TIME_TCN

    def __init__(self, config: DiscreteTimeTCNConfig) -> None:
        super().__init__()
        if config.pooling not in {"last", "last_mean_max"}:
            raise ValueError("pooling must be 'last' or 'last_mean_max'.")
        if len(config.hidden_channels) == 0:
            raise ValueError("hidden_channels must contain at least one layer.")
        if len(config.dilations) < len(config.hidden_channels):
            raise ValueError("dilations must contain one value per hidden layer.")
        self.config = config
        self.context_length = int(config.context_length)
        self.input_channels = int(config.input_channels)
        channels = [self.input_channels, *[int(value) for value in config.hidden_channels]]
        blocks = []
        for layer_index, out_channels in enumerate(config.hidden_channels):
            blocks.append(
                CausalResidualBlock(
                    in_channels=channels[layer_index],
                    out_channels=int(out_channels),
                    kernel_size=int(config.kernel_size),
                    dilation=int(config.dilations[layer_index]),
                    dropout=float(config.dropout),
                    activation=config.activation,
                    normalization=config.normalization,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        encoded_channels = int(config.hidden_channels[-1])
        sequence_feature_dim = (
            encoded_channels if config.pooling == "last" else encoded_channels * 3
        )
        self.scalar_encoder = (
            ScalarContextEncoder(
                config.scalar_context_dim,
                config.scalar_hidden_dim,
                dropout=config.dropout,
            )
            if config.use_scalar_context
            else None
        )
        scalar_dim = self.scalar_encoder.output_dim if self.scalar_encoder else 0
        self.head = nn.Sequential(
            nn.LayerNorm(sequence_feature_dim + scalar_dim),
            nn.Linear(sequence_feature_dim + scalar_dim, encoded_channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(encoded_channels, int(config.num_bins)),
        )

    def encode_sequence(self, x_sequence: torch.Tensor) -> torch.Tensor:
        """Return causal sequence features with shape ``(B, C, L)``."""

        sequence = sequence_to_channels(
            x_sequence,
            context_length=self.context_length,
            input_channels=self.input_channels,
        )
        return self.blocks(sequence)

    def pool_sequence(self, encoded: torch.Tensor) -> torch.Tensor:
        """Pool encoded sequence features for the current-time prediction."""

        if self.config.pooling == "last":
            return encoded[:, :, -1]
        return torch.cat(
            [encoded[:, :, -1], encoded.mean(dim=-1), encoded.max(dim=-1).values],
            dim=1,
        )

    def forward(
        self,
        x_sequence: torch.Tensor,
        x_scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return hazard logits with shape ``(batch, num_bins)``."""

        pooled = self.pool_sequence(self.encode_sequence(x_sequence))
        if self.scalar_encoder is not None:
            if x_scalar is None:
                raise ValueError("Scalar context is enabled but x_scalar was not provided.")
            pooled = torch.cat([pooled, self.scalar_encoder(x_scalar)], dim=1)
        return self.head(pooled)
