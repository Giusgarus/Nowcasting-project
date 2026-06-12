"""Deterministic GRU architectures for univariate trajectory forecasting."""

import torch
from torch import nn


class GRUSequenceToVectorForecaster(nn.Module):
    """Encode a context and predict the full future trajectory directly."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        prediction_length: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        _validate_model_parameters(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            prediction_length=prediction_length,
            dropout=dropout,
        )
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.prediction_length = prediction_length
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.output_head = nn.Linear(
            hidden_size * self.num_directions,
            prediction_length,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a direct multi-step forecast with shape ``(batch, horizon)``."""

        _validate_input_tensor(x)
        _, hidden = self.encoder(x)
        final_hidden = _last_layer_hidden(
            hidden,
            num_layers=self.num_layers,
            num_directions=self.num_directions,
        )
        return self.output_head(final_hidden)


class GRUSeq2SeqForecaster(nn.Module):
    """Encode a context and autoregressively decode the future trajectory."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        prediction_length: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        _validate_model_parameters(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            prediction_length=prediction_length,
            dropout=dropout,
        )
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.prediction_length = prediction_length
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=False,
            batch_first=True,
        )
        self.encoder_to_decoder = nn.Linear(
            hidden_size * self.num_directions,
            hidden_size,
        )
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Decode one future value at a time.

        Teacher forcing is available only while the module is in training mode.
        Evaluation always feeds back the previous prediction.
        """

        _validate_input_tensor(x)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {x.shape[-1]}.")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between zero and one.")
        if y is not None and y.shape != (x.shape[0], self.prediction_length):
            raise ValueError(
                "y must have shape (batch, prediction_length) when provided."
            )

        _, encoder_hidden = self.encoder(x)
        decoder_hidden = self._prepare_decoder_hidden(encoder_hidden)
        decoder_input = x[:, -1:, :1]
        predictions = []
        can_teacher_force = self.training and y is not None and teacher_forcing_ratio > 0

        for step in range(self.prediction_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input,
                decoder_hidden,
            )
            prediction = self.output_head(decoder_output[:, -1, :])
            predictions.append(prediction)
            if can_teacher_force:
                force_mask = torch.rand(
                    x.shape[0],
                    1,
                    device=x.device,
                ).lt(teacher_forcing_ratio)
                true_previous = y[:, step : step + 1]
                decoder_input = torch.where(force_mask, true_previous, prediction)
            else:
                decoder_input = prediction
            decoder_input = decoder_input.unsqueeze(1)
        return torch.cat(predictions, dim=1)

    def _prepare_decoder_hidden(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        layers = encoder_hidden.view(
            self.num_layers,
            self.num_directions,
            encoder_hidden.shape[1],
            self.hidden_size,
        )
        layers = layers.permute(0, 2, 1, 3).reshape(
            self.num_layers,
            encoder_hidden.shape[1],
            self.hidden_size * self.num_directions,
        )
        return torch.tanh(self.encoder_to_decoder(layers))


def build_gru_forecaster(
    architecture: str,
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    prediction_length: int,
    dropout: float,
    bidirectional: bool,
) -> nn.Module:
    """Build one configured GRU forecasting architecture."""

    common = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "prediction_length": prediction_length,
        "dropout": dropout,
        "bidirectional": bidirectional,
    }
    if architecture == "gru_s2v":
        return GRUSequenceToVectorForecaster(**common)
    if architecture == "gru_seq2seq":
        return GRUSeq2SeqForecaster(**common)
    raise ValueError(f"Unsupported GRU architecture: {architecture}")


def _last_layer_hidden(
    hidden: torch.Tensor,
    *,
    num_layers: int,
    num_directions: int,
) -> torch.Tensor:
    layers = hidden.view(
        num_layers,
        num_directions,
        hidden.shape[1],
        hidden.shape[2],
    )
    final_layer = layers[-1].permute(1, 0, 2)
    return final_layer.reshape(final_layer.shape[0], -1)


def _validate_input_tensor(x: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, context_length, input_size).")


def _validate_model_parameters(
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    prediction_length: int,
    dropout: float,
) -> None:
    if min(input_size, hidden_size, num_layers, prediction_length) < 1:
        raise ValueError("Model dimensions and layer count must be positive.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
