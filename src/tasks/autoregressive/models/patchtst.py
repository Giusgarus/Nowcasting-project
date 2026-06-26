"""Deterministic PatchTST-style model for univariate trajectory forecasting."""

import torch
from torch import nn


def compute_num_patches(context_length: int, patch_len: int, stride: int) -> int:
    """Return the number of complete temporal patches extracted from a context."""

    if context_length < 1:
        raise ValueError("context_length must be positive.")
    if patch_len < 1 or patch_len > context_length:
        raise ValueError("patch_len must be between one and context_length.")
    if stride < 1:
        raise ValueError("stride must be positive.")
    return 1 + (context_length - patch_len) // stride


class PatchTSTForecaster(nn.Module):
    """Encode channel-independent temporal patches and forecast one trajectory."""

    def __init__(
        self,
        *,
        context_length: int,
        prediction_length: int,
        input_channels: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if prediction_length < 1 or input_channels < 1:
            raise ValueError("prediction_length and input_channels must be positive.")
        if d_model < 1 or n_heads < 1 or num_layers < 1:
            raise ValueError("Transformer dimensions and layer count must be positive.")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.input_channels = input_channels
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_patches = compute_num_patches(context_length, patch_len, stride)

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_head = nn.Linear(
            input_channels * self.num_patches * d_model,
            prediction_length,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the learned positional embedding."""

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Return patches with shape ``(batch, channels, patches, patch_len)``."""

        self._validate_input(x)
        return x.transpose(1, 2).unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a direct forecast with shape ``(batch, prediction_length)``."""

        patches = self.extract_patches(x)
        batch_size, channels, num_patches, patch_len = patches.shape
        tokens = patches.reshape(batch_size * channels, num_patches, patch_len)
        tokens = self.patch_embedding(tokens) + self.position_embedding
        encoded = self.encoder(tokens)
        encoded = encoded.reshape(batch_size, channels * num_patches * self.d_model)
        return self.output_head(encoded)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, context_length, channels).")
        if x.shape[1] != self.context_length:
            raise ValueError(
                f"Expected context_length={self.context_length}, got {x.shape[1]}."
            )
        if x.shape[2] != self.input_channels:
            raise ValueError(
                f"Expected input_channels={self.input_channels}, got {x.shape[2]}."
            )
