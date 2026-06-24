"""Learnable-shapelet models for current-level persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

MODEL_ID_MLP = "multiscale_shapelet_mlp"
MODEL_ID_TRANSFORMER = "multiscale_shapelet_transformer"
MODEL_ID_CONVOLUTION = "multiscale_shapelet_convolution"
SUPPORTED_MODEL_IDS = (
    MODEL_ID_MLP,
    MODEL_ID_TRANSFORMER,
    MODEL_ID_CONVOLUTION,
)


@dataclass(frozen=True)
class ShapeletConfig:
    """Shared learnable-shapelet configuration."""

    context_length: int
    shapelet_lengths: tuple[int, ...] = (5, 10, 15)
    n_shapelets_per_length: int = 32
    hidden_size: int = 128
    dropout: float = 0.1


class MultiscaleLearnableShapeletLayer(nn.Module):
    """Compute minimum squared distance to learnable shapelets at each scale."""

    def __init__(
        self,
        *,
        context_length: int,
        shapelet_lengths: tuple[int, ...],
        n_shapelets_per_length: int,
    ) -> None:
        super().__init__()
        if context_length < 1:
            raise ValueError("context_length must be positive.")
        if n_shapelets_per_length < 1:
            raise ValueError("n_shapelets_per_length must be positive.")
        if any(length < 1 or length > context_length for length in shapelet_lengths):
            raise ValueError("shapelet lengths must lie in [1, context_length].")
        self.context_length = int(context_length)
        self.shapelet_lengths = tuple(int(length) for length in shapelet_lengths)
        self.n_shapelets_per_length = int(n_shapelets_per_length)
        self.shapelets = nn.ParameterDict(
            {
                str(length): nn.Parameter(
                    torch.randn(n_shapelets_per_length, length) * 0.02
                )
                for length in self.shapelet_lengths
            }
        )

    @property
    def output_size(self) -> int:
        """Return the number of shapelet-distance features."""

        return len(self.shapelet_lengths) * self.n_shapelets_per_length

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return multiscale minimum-distance features.

        ``inputs`` may have shape ``(batch, context_length)`` or
        ``(batch, context_length, 1)``.
        """

        if inputs.ndim == 3 and inputs.shape[-1] == 1:
            values = inputs.squeeze(-1)
        elif inputs.ndim == 2:
            values = inputs
        else:
            raise ValueError("inputs must have shape (B, L) or (B, L, 1).")
        if values.shape[1] != self.context_length:
            raise ValueError("inputs context length does not match model config.")

        features = []
        for length in self.shapelet_lengths:
            windows = values.unfold(dimension=1, size=length, step=1)
            shapelets = self.shapelets[str(length)]
            distances = torch.square(windows[:, :, None, :] - shapelets[None, None])
            min_distances = distances.mean(dim=-1).min(dim=1).values
            features.append(min_distances)
        return torch.cat(features, dim=1)


class MultiscaleLearnableShapeletMLP(nn.Module):
    """Shapelet feature extractor with an MLP regression head."""

    model_id = MODEL_ID_MLP

    def __init__(self, config: ShapeletConfig) -> None:
        super().__init__()
        self.shapelets = MultiscaleLearnableShapeletLayer(
            context_length=config.context_length,
            shapelet_lengths=config.shapelet_lengths,
            n_shapelets_per_length=config.n_shapelets_per_length,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.shapelets.output_size),
            nn.Linear(self.shapelets.output_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict log1p remaining persistence seconds."""

        return self.head(self.shapelets(inputs)).squeeze(-1)


class MultiscaleLearnableShapeletTransformer(nn.Module):
    """Shapelet feature extractor with a small Transformer encoder head."""

    model_id = MODEL_ID_TRANSFORMER

    def __init__(self, config: ShapeletConfig, *, n_heads: int = 4) -> None:
        super().__init__()
        self.shapelets = MultiscaleLearnableShapeletLayer(
            context_length=config.context_length,
            shapelet_lengths=config.shapelet_lengths,
            n_shapelets_per_length=config.n_shapelets_per_length,
        )
        self.projection = nn.Linear(1, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=n_heads,
            dim_feedforward=config.hidden_size * 2,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict log1p remaining persistence seconds."""

        features = self.shapelets(inputs).unsqueeze(-1)
        encoded = self.encoder(self.projection(features))
        return self.head(encoded.mean(dim=1)).squeeze(-1)


class MultiscaleLearnableShapeletConvolution(nn.Module):
    """Shapelet feature extractor with a convolutional regression head."""

    model_id = MODEL_ID_CONVOLUTION

    def __init__(self, config: ShapeletConfig) -> None:
        super().__init__()
        self.shapelets = MultiscaleLearnableShapeletLayer(
            context_length=config.context_length,
            shapelet_lengths=config.shapelet_lengths,
            n_shapelets_per_length=config.n_shapelets_per_length,
        )
        self.head = nn.Sequential(
            nn.Conv1d(1, config.hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict log1p remaining persistence seconds."""

        features = self.shapelets(inputs).unsqueeze(1)
        return self.head(features).squeeze(-1)


def build_learnable_shapelet_model(
    model_id: Literal[
        "multiscale_shapelet_mlp",
        "multiscale_shapelet_transformer",
        "multiscale_shapelet_convolution",
    ]
    | str,
    config: ShapeletConfig,
) -> nn.Module:
    """Instantiate one supported learnable-shapelet model."""

    if model_id == MODEL_ID_MLP:
        return MultiscaleLearnableShapeletMLP(config)
    if model_id == MODEL_ID_TRANSFORMER:
        return MultiscaleLearnableShapeletTransformer(config)
    if model_id == MODEL_ID_CONVOLUTION:
        return MultiscaleLearnableShapeletConvolution(config)
    raise ValueError(f"Unsupported current-level persistence model_id: {model_id}")
