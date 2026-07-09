"""Neural classifiers for long-fade detection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.tasks.current_level_persistence.models.learnable_shapelets import (
    MultiscaleLearnableShapeletLayer,
    ScalarContextEncoder,
)

MODEL_ID_TCN = "tcn_classifier"
MODEL_ID_SHAPELET_CONV = "multiscale_shapelet_convolution_classifier"
MODEL_ID_XGBOOST = "xgboost_lag_scalar_classifier"
SUPPORTED_MODEL_IDS = (MODEL_ID_XGBOOST, MODEL_ID_TCN, MODEL_ID_SHAPELET_CONV)


@dataclass(frozen=True)
class TCNClassifierConfig:
    """Configuration for the TCN long-fade classifier."""

    context_length: int
    input_channels: int = 1
    hidden_channels: int = 64
    num_blocks: int = 4
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1
    pooling: str = "last_mean_max"
    use_scalar_context: bool = True
    scalar_context_dim: int = 14
    scalar_hidden_dim: int = 32


@dataclass(frozen=True)
class ShapeletConvolutionClassifierConfig:
    """Configuration for the shapelet-convolution classifier."""

    context_length: int
    shapelet_lengths: tuple[int, ...] = (5, 10, 15)
    n_shapelets_per_length: int = 32
    conv_channels: int = 64
    num_conv_layers: int = 2
    kernel_size: int = 3
    dropout: float = 0.1
    pooling: str = "last_mean_max"
    use_scalar_context: bool = True
    scalar_context_dim: int = 14
    scalar_hidden_dim: int = 32


class TemporalBlock(nn.Module):
    """One residual dilated temporal convolution block."""

    def __init__(
        self,
        *,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return residual block output."""

        return inputs + self.net(inputs)


def _sequence_to_channels(inputs: torch.Tensor, *, context_length: int) -> torch.Tensor:
    """Normalize sequence input to ``(batch, 1, context_length)``."""

    if inputs.ndim == 2:
        values = inputs.unsqueeze(1)
    elif inputs.ndim == 3 and inputs.shape[-1] == 1:
        values = inputs.transpose(1, 2)
    elif inputs.ndim == 3 and inputs.shape[1] == 1:
        values = inputs
    else:
        raise ValueError("sequence input must be (B, L), (B, L, 1), or (B, 1, L).")
    if values.shape[-1] != context_length:
        raise ValueError("sequence context length does not match model config.")
    return values


def _last_mean_max_pool(encoded: torch.Tensor) -> torch.Tensor:
    """Concatenate last, mean, and max temporal pooling."""

    return torch.cat(
        [
            encoded[:, :, -1],
            encoded.mean(dim=-1),
            encoded.max(dim=-1).values,
        ],
        dim=1,
    )


def _make_scalar_encoder(
    *,
    enabled: bool,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
) -> ScalarContextEncoder | None:
    if not enabled:
        return None
    return ScalarContextEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def _append_scalar(
    features: torch.Tensor,
    x_scalar: torch.Tensor | None,
    encoder: ScalarContextEncoder | None,
) -> torch.Tensor:
    if encoder is None:
        return features
    if x_scalar is None:
        raise ValueError("Scalar context is enabled but x_scalar was not provided.")
    return torch.cat([features, encoder(x_scalar)], dim=1)


class TCNLongFadeClassifier(nn.Module):
    """Residual TCN classifier for long-fade detection."""

    model_id = MODEL_ID_TCN

    def __init__(self, config: TCNClassifierConfig) -> None:
        super().__init__()
        if config.pooling != "last_mean_max":
            raise ValueError("TCN classifier currently supports last_mean_max pooling.")
        dilations = config.dilations[: config.num_blocks]
        if len(dilations) < config.num_blocks:
            raise ValueError("dilations must contain at least num_blocks values.")
        self.context_length = int(config.context_length)
        self.input_projection = nn.Conv1d(
            config.input_channels,
            config.hidden_channels,
            kernel_size=1,
        )
        self.blocks = nn.Sequential(
            *[
                TemporalBlock(
                    channels=config.hidden_channels,
                    kernel_size=config.kernel_size,
                    dilation=int(dilation),
                    dropout=config.dropout,
                )
                for dilation in dilations
            ]
        )
        self.scalar_encoder = _make_scalar_encoder(
            enabled=config.use_scalar_context,
            input_dim=config.scalar_context_dim,
            hidden_dim=config.scalar_hidden_dim,
            dropout=config.dropout,
        )
        scalar_dim = self.scalar_encoder.output_dim if self.scalar_encoder else 0
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_channels * 3 + scalar_dim),
            nn.Linear(config.hidden_channels * 3 + scalar_dim, config.hidden_channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, 1),
        )

    def forward(
        self,
        x_sequence: torch.Tensor,
        x_scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits for long-fade probability."""

        sequence = _sequence_to_channels(x_sequence, context_length=self.context_length)
        encoded = self.blocks(self.input_projection(sequence))
        pooled = _last_mean_max_pool(encoded)
        combined = _append_scalar(pooled, x_scalar, self.scalar_encoder)
        return self.head(combined).squeeze(-1)


class MultiscaleShapeletConvolutionLongFadeClassifier(nn.Module):
    """Multiscale shapelet convolution classifier for long-fade detection."""

    model_id = MODEL_ID_SHAPELET_CONV

    def __init__(self, config: ShapeletConvolutionClassifierConfig) -> None:
        super().__init__()
        if config.pooling != "last_mean_max":
            raise ValueError(
                "Shapelet convolution classifier supports last_mean_max pooling."
            )
        self.context_length = int(config.context_length)
        self.shapelets = MultiscaleLearnableShapeletLayer(
            context_length=config.context_length,
            shapelet_lengths=tuple(config.shapelet_lengths),
            n_shapelets_per_length=config.n_shapelets_per_length,
        )
        self.scale_normalizers = nn.ModuleDict()
        self.scale_convolutions = nn.ModuleDict()
        padding = config.kernel_size // 2
        for length in self.shapelets.shapelet_lengths:
            self.scale_normalizers[str(length)] = nn.GroupNorm(
                num_groups=1,
                num_channels=self.shapelets.n_shapelets_per_length,
            )
            layers: list[nn.Module] = []
            input_channels = self.shapelets.n_shapelets_per_length
            for _ in range(config.num_conv_layers):
                layers.extend(
                    [
                        nn.Conv1d(
                            input_channels,
                            config.conv_channels,
                            kernel_size=config.kernel_size,
                            padding=padding,
                        ),
                        nn.GroupNorm(num_groups=1, num_channels=config.conv_channels),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                    ]
                )
                input_channels = config.conv_channels
            self.scale_convolutions[str(length)] = nn.Sequential(*layers)
        per_scale_dim = config.conv_channels * 3
        pooled_dim = per_scale_dim * len(self.shapelets.shapelet_lengths)
        self.scalar_encoder = _make_scalar_encoder(
            enabled=config.use_scalar_context,
            input_dim=config.scalar_context_dim,
            hidden_dim=config.scalar_hidden_dim,
            dropout=config.dropout,
        )
        scalar_dim = self.scalar_encoder.output_dim if self.scalar_encoder else 0
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim + scalar_dim),
            nn.Linear(pooled_dim + scalar_dim, config.conv_channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.conv_channels, 1),
        )

    def forward(
        self,
        x_sequence: torch.Tensor,
        x_scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits for long-fade probability."""

        pooled_by_scale = []
        for length, maps in self.shapelets.response_maps_by_length(x_sequence).items():
            maps = torch.log1p(torch.clamp(maps, min=0.0))
            maps = self.scale_normalizers[str(length)](maps)
            encoded = self.scale_convolutions[str(length)](maps)
            pooled_by_scale.append(_last_mean_max_pool(encoded))
        features = torch.cat(pooled_by_scale, dim=1)
        combined = _append_scalar(features, x_scalar, self.scalar_encoder)
        return self.head(combined).squeeze(-1)


def build_neural_classifier(
    model_id: str,
    *,
    context_length: int,
    scalar_context_dim: int,
    config: dict,
) -> nn.Module:
    """Build one supported neural long-fade classifier."""

    if model_id == MODEL_ID_TCN:
        model_config = dict(config)
        model_config.setdefault("context_length", context_length)
        model_config.setdefault("scalar_context_dim", scalar_context_dim)
        model_config["dilations"] = tuple(model_config.get("dilations", (1, 2, 4, 8)))
        return TCNLongFadeClassifier(TCNClassifierConfig(**model_config))
    if model_id == MODEL_ID_SHAPELET_CONV:
        model_config = dict(config)
        shapelet_config = model_config.pop("shapelets", {})
        if "shapelet_lengths" in shapelet_config:
            shapelet_config["shapelet_lengths"] = tuple(
                shapelet_config["shapelet_lengths"]
            )
        merged = {**shapelet_config, **model_config}
        merged.setdefault("context_length", context_length)
        merged.setdefault("scalar_context_dim", scalar_context_dim)
        return MultiscaleShapeletConvolutionLongFadeClassifier(
            ShapeletConvolutionClassifierConfig(**merged)
        )
    raise ValueError(f"Unsupported neural long-fade model_id: {model_id}")
