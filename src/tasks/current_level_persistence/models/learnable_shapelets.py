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
    hidden_dim: int | None = None
    num_hidden_layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    transformer_layers: int = 1
    pooling: str = "mean"
    conv_channels: int = 64
    num_conv_layers: int = 2
    kernel_size: int = 3
    dropout: float = 0.1
    use_scalar_context: bool = False
    scalar_context_dim: int = 0
    scalar_encoder_hidden_dim: int = 32
    scalar_encoder_dropout: float = 0.1

    @property
    def mlp_hidden_dim(self) -> int:
        """Return the MLP hidden width, accepting both config key conventions."""

        return int(self.hidden_dim if self.hidden_dim is not None else self.hidden_size)


class ScalarContextEncoder(nn.Module):
    """Encode standardized scalar context into a compact auxiliary embedding."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("scalar_context_dim must be positive when enabled.")
        if hidden_dim < 1:
            raise ValueError("scalar encoder hidden_dim must be positive.")
        self.input_dim = int(input_dim)
        self.output_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one scalar-context embedding per window."""

        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                "x_scalar must have shape "
                f"(batch, {self.input_dim}); received {tuple(inputs.shape)}."
            )
        return self.encoder(inputs)


def build_scalar_encoder(config: ShapeletConfig) -> ScalarContextEncoder | None:
    """Build the optional scalar-context branch."""

    if not config.use_scalar_context:
        return None
    return ScalarContextEncoder(
        input_dim=config.scalar_context_dim,
        hidden_dim=config.scalar_encoder_hidden_dim,
        dropout=config.scalar_encoder_dropout,
    )


def append_scalar_context(
    shapelet_features: torch.Tensor,
    x_scalar: torch.Tensor | None,
    scalar_encoder: ScalarContextEncoder | None,
) -> torch.Tensor:
    """Concatenate shapelet features with the optional scalar embedding."""

    if scalar_encoder is None:
        return shapelet_features
    if x_scalar is None:
        raise ValueError(
            "Scalar context is enabled for this model, but x_scalar was not provided."
        )
    scalar_embedding = scalar_encoder(x_scalar)
    if scalar_embedding.shape[0] != shapelet_features.shape[0]:
        raise ValueError("x_shapelet and x_scalar batch sizes do not match.")
    return torch.cat([shapelet_features, scalar_embedding], dim=1)


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

        ``inputs`` may have shape ``(batch, context_length)``,
        ``(batch, context_length, 1)``, or ``(batch, 1, context_length)``.
        """

        values = self._normalize_inputs(inputs)
        features = []
        for length in self.shapelet_lengths:
            distances = self._distance_map_for_length(values, length)
            min_distances = distances.min(dim=1).values
            features.append(min_distances)
        return torch.cat(features, dim=1)

    def _normalize_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return inputs as ``(batch, context_length)``."""

        if inputs.ndim == 3 and inputs.shape[-1] == 1:
            values = inputs.squeeze(-1)
        elif inputs.ndim == 3 and inputs.shape[1] == 1:
            values = inputs.squeeze(1)
        elif inputs.ndim == 2:
            values = inputs
        else:
            raise ValueError("inputs must have shape (B, L), (B, L, 1), or (B, 1, L).")
        if values.shape[1] != self.context_length:
            raise ValueError("inputs context length does not match model config.")
        return values

    def _distance_map_for_length(
        self,
        values: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """Return mean-squared distance maps with shape ``(B, positions, S)``."""

        windows = values.unfold(dimension=1, size=length, step=1)
        shapelets = self.shapelets[str(length)]
        distances = torch.square(windows[:, :, None, :] - shapelets[None, None])
        return distances.mean(dim=-1)

    def response_maps(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return padded multiscale shapelet distance maps for convolution heads.

        Response-map lengths differ by shapelet length. This method pads shorter
        maps on the right to the longest map length, then concatenates all
        shapelets along the channel dimension, returning ``(B, total_shapelets, T)``.
        """

        values = self._normalize_inputs(inputs)

        max_positions = max(
            self.context_length - length + 1 for length in self.shapelet_lengths
        )
        maps = []
        for length in self.shapelet_lengths:
            distances = self._distance_map_for_length(values, length).transpose(1, 2)
            pad_width = max_positions - distances.shape[-1]
            if pad_width > 0:
                distances = torch.nn.functional.pad(distances, (0, pad_width))
            maps.append(distances)
        return torch.cat(maps, dim=1)

    def response_maps_by_length(self, inputs: torch.Tensor) -> dict[int, torch.Tensor]:
        """Return unpadded response maps grouped by shapelet length.

        Each value has shape ``(B, n_shapelets_per_length, positions_for_length)``.
        This is the preferred representation for convolutional heads because it
        avoids padding response maps with artificial distances.
        """

        values = self._normalize_inputs(inputs)
        return {
            int(length): self._distance_map_for_length(values, length).transpose(1, 2)
            for length in self.shapelet_lengths
        }

    def shapelets_by_length(self) -> dict[int, torch.Tensor]:
        """Return learned shapelet tensors grouped by length."""

        return {
            int(length): self.shapelets[str(length)].detach().cpu()
            for length in self.shapelet_lengths
        }


def compressed_shapelet_features(
    shapelets: MultiscaleLearnableShapeletLayer,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Return numerically stable shapelet-distance features.

    Minimum squared distances can become large for relative signal windows.
    ``log1p`` compression preserves ordering while keeping gradients finite for
    the downstream heads.
    """

    return torch.log1p(torch.clamp(shapelets(inputs), min=0.0))


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
        self.scalar_encoder = build_scalar_encoder(config)
        scalar_dim = self.scalar_encoder.output_dim if self.scalar_encoder else 0
        input_dim = self.shapelets.output_size + scalar_dim
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        for _ in range(config.num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(input_dim, config.mlp_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            input_dim = config.mlp_hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        x_shapelet: torch.Tensor,
        x_scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict log1p remaining persistence seconds."""

        features = compressed_shapelet_features(self.shapelets, x_shapelet)
        combined = append_scalar_context(features, x_scalar, self.scalar_encoder)
        return self.head(combined).squeeze(-1)


class MultiscaleLearnableShapeletTransformer(nn.Module):
    """Shapelet feature extractor with a small Transformer encoder head."""

    model_id = MODEL_ID_TRANSFORMER

    def __init__(self, config: ShapeletConfig) -> None:
        super().__init__()
        self.shapelets = MultiscaleLearnableShapeletLayer(
            context_length=config.context_length,
            shapelet_lengths=config.shapelet_lengths,
            n_shapelets_per_length=config.n_shapelets_per_length,
        )
        self.pooling = config.pooling
        self.feature_norm = nn.LayerNorm(self.shapelets.output_size)
        self.projection = nn.Linear(1, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 2,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
        )
        self.scalar_encoder = build_scalar_encoder(config)
        scalar_dim = self.scalar_encoder.output_dim if self.scalar_encoder else 0
        self.head = nn.Linear(config.d_model + scalar_dim, 1)

    def forward(
        self,
        x_shapelet: torch.Tensor,
        x_scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict log1p remaining persistence seconds."""

        features = self.feature_norm(
            compressed_shapelet_features(self.shapelets, x_shapelet)
        ).unsqueeze(-1)
        encoded = self.encoder(self.projection(features))
        if self.pooling == "mean":
            pooled = encoded.mean(dim=1)
        elif self.pooling == "max":
            pooled = encoded.max(dim=1).values
        else:
            raise ValueError(f"Unsupported transformer pooling: {self.pooling}")
        combined = append_scalar_context(pooled, x_scalar, self.scalar_encoder)
        return self.head(combined).squeeze(-1)


class MultiscaleLearnableShapeletConvolution(nn.Module):
    """Shapelet extractor with per-scale convolutional regression heads.

    Response maps have different temporal lengths for different shapelet
    lengths. This head processes each scale separately, pools each scale, then
    concatenates the pooled scale features before the final regression layer.
    """

    model_id = MODEL_ID_CONVOLUTION

    def __init__(self, config: ShapeletConfig) -> None:
        super().__init__()
        self.shapelets = MultiscaleLearnableShapeletLayer(
            context_length=config.context_length,
            shapelet_lengths=config.shapelet_lengths,
            n_shapelets_per_length=config.n_shapelets_per_length,
        )
        self.pooling = config.pooling
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
        pooled_size = config.conv_channels * len(self.shapelets.shapelet_lengths)
        self.scalar_encoder = build_scalar_encoder(config)
        scalar_dim = self.scalar_encoder.output_dim if self.scalar_encoder else 0
        self.head_norm = nn.LayerNorm(pooled_size + scalar_dim)
        self.head = nn.Linear(
            pooled_size + scalar_dim,
            1,
        )

    def forward(
        self,
        x_shapelet: torch.Tensor,
        x_scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict log1p remaining persistence seconds."""

        pooled_by_scale = []
        for length, maps in self.shapelets.response_maps_by_length(x_shapelet).items():
            # Distances are non-negative and can have a long right tail; log1p
            # compression keeps the convolutional head numerically stable.
            maps = torch.log1p(torch.clamp(maps, min=0.0))
            maps = self.scale_normalizers[str(length)](maps)
            encoded = self.scale_convolutions[str(length)](maps)
            if self.pooling == "mean":
                pooled = encoded.mean(dim=-1)
            elif self.pooling == "max":
                pooled = encoded.max(dim=-1).values
            else:
                raise ValueError(f"Unsupported convolution pooling: {self.pooling}")
            pooled_by_scale.append(pooled)
        shapelet_features = torch.cat(pooled_by_scale, dim=1)
        combined = append_scalar_context(
            shapelet_features,
            x_scalar,
            self.scalar_encoder,
        )
        return self.head(self.head_norm(combined)).squeeze(-1)


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


MultiscaleShapeletMLP = MultiscaleLearnableShapeletMLP
MultiscaleShapeletTransformer = MultiscaleLearnableShapeletTransformer
MultiscaleShapeletConvolution = MultiscaleLearnableShapeletConvolution
