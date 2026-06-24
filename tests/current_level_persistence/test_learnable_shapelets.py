import torch

from src.tasks.current_level_persistence.models.learnable_shapelets import (
    SUPPORTED_MODEL_IDS,
    MultiscaleLearnableShapeletLayer,
    ShapeletConfig,
    build_learnable_shapelet_model,
)


def test_shapelet_layer_returns_expected_feature_count() -> None:
    layer = MultiscaleLearnableShapeletLayer(
        context_length=30,
        shapelet_lengths=(5, 10, 15),
        n_shapelets_per_length=4,
    )
    inputs = torch.randn(3, 30)

    features = layer(inputs)

    assert features.shape == (3, 12)
    assert layer.output_size == 12


def test_shapelet_layer_accepts_common_input_conventions() -> None:
    layer = MultiscaleLearnableShapeletLayer(
        context_length=30,
        shapelet_lengths=(5, 10),
        n_shapelets_per_length=2,
    )
    flat = torch.randn(2, 30)

    assert layer(flat).shape == (2, 4)
    assert layer(flat.unsqueeze(-1)).shape == (2, 4)
    assert layer(flat.unsqueeze(1)).shape == (2, 4)


def test_convolution_response_maps_are_padded_to_common_length() -> None:
    layer = MultiscaleLearnableShapeletLayer(
        context_length=30,
        shapelet_lengths=(5, 10, 15),
        n_shapelets_per_length=3,
    )

    maps = layer.response_maps(torch.randn(2, 30))

    assert maps.shape == (2, 9, 26)


def test_each_shapelet_model_outputs_one_scalar_per_window_and_backpropagates() -> None:
    config = ShapeletConfig(
        context_length=30,
        shapelet_lengths=(5, 10, 15),
        n_shapelets_per_length=2,
        hidden_dim=16,
        num_hidden_layers=1,
        d_model=8,
        n_heads=2,
        conv_channels=8,
        dropout=0.0,
    )
    inputs = torch.randn(4, 30)
    targets = torch.randn(4)

    for model_id in SUPPORTED_MODEL_IDS:
        model = build_learnable_shapelet_model(model_id, config)
        output = model(inputs)
        loss = torch.nn.functional.smooth_l1_loss(output, targets)
        loss.backward()

        assert output.shape == (4,)
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
