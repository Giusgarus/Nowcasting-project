import numpy as np
import pytest
import torch

from scripts.experiments.current_level_persistence.train_learnable_shapelets import (
    scalar_context_metadata,
)
from src.tasks.current_level_persistence.data.dataset import (
    SCALAR_CONTEXT_FEATURE_NAMES,
    compute_scalar_context_features,
    ensure_scalar_context_features,
    prepare_scalar_context_splits,
)
from src.tasks.current_level_persistence.models.learnable_shapelets import (
    SUPPORTED_MODEL_IDS,
    ShapeletConfig,
    build_learnable_shapelet_model,
)
from src.tasks.current_level_persistence.utils.paths import make_run_id, run_dir


def test_scalar_context_values_and_feature_order() -> None:
    x_raw = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)

    features = compute_scalar_context_features(x_raw, delta=0.5)

    assert SCALAR_CONTEXT_FEATURE_NAMES == (
        "current_signal",
        "recovery_level",
        "window_mean",
        "window_std",
        "window_min",
        "window_max",
        "window_range",
        "last_slope",
        "recent_slope_3",
        "recent_slope_5",
        "recent_mean_5",
        "recent_std_5",
    )
    np.testing.assert_allclose(
        features[0],
        [5.0, 4.5, 3.0, np.sqrt(2.0), 1.0, 5.0, 4.0, 1.0, 2.0, 4.0, 3.0, np.sqrt(2.0)],
    )


def test_old_split_falls_back_to_x_raw_for_scalar_context() -> None:
    arrays = {
        "X_raw": np.array([[9.0, 9.5, 10.0]], dtype=np.float32),
        "delta": np.array([0.5], dtype=np.float32),
    }

    prepared = ensure_scalar_context_features(arrays)

    assert prepared["scalar_context_features"].shape == (1, 12)
    assert prepared["scalar_context_feature_names"].tolist() == list(
        SCALAR_CONTEXT_FEATURE_NAMES
    )


def test_scalar_context_standardization_uses_train_statistics_only() -> None:
    split_arrays = {
        "train": {
            "X_raw": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
            "delta": np.array([0.5, 0.5], dtype=np.float32),
        },
        "val": {
            "X_raw": np.array([[100.0, 101.0]], dtype=np.float32),
            "delta": np.array([0.5], dtype=np.float32),
        },
        "test": {
            "X_raw": np.array([[200.0, 201.0]], dtype=np.float32),
            "delta": np.array([0.5], dtype=np.float32),
        },
    }
    raw_train = compute_scalar_context_features(
        split_arrays["train"]["X_raw"],
        delta=split_arrays["train"]["delta"],
    )

    prepared, scaler = prepare_scalar_context_splits(split_arrays)

    np.testing.assert_allclose(scaler["mean"], raw_train.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(
        prepared["train"]["scalar_context_features"].mean(axis=0),
        np.zeros(12),
        atol=1e-6,
    )
    assert prepared["val"]["scalar_context_features"][0, 0] > 10.0
    assert scaler["fitted_on"] == "train"


def test_all_models_accept_shapelet_and_scalar_inputs() -> None:
    config = ShapeletConfig(
        context_length=30,
        shapelet_lengths=(5, 10, 15),
        n_shapelets_per_length=2,
        hidden_dim=8,
        num_hidden_layers=1,
        d_model=8,
        n_heads=2,
        conv_channels=8,
        num_conv_layers=1,
        dropout=0.0,
        use_scalar_context=True,
        scalar_context_dim=12,
        scalar_encoder_hidden_dim=4,
        scalar_encoder_dropout=0.0,
    )
    x_shapelet = torch.randn(3, 30)
    x_scalar = torch.randn(3, 12)

    for model_id in SUPPORTED_MODEL_IDS:
        model = build_learnable_shapelet_model(model_id, config)
        assert model(x_shapelet, x_scalar).shape == (3,)
        with pytest.raises(ValueError, match="x_scalar was not provided"):
            model(x_shapelet)


def test_scalar_context_run_id_and_metadata_are_separate() -> None:
    run_id = make_run_id(
        delta=0.5,
        context_length=30,
        model_id="multiscale_shapelet_mlp",
        selection_id="externalHoldout_test_fc_uplink_fade",
        scalar_context=True,
    )
    config = {
        "dataset": {
            "input_key": "X_relative_to_current",
            "scalar_context_key": "scalar_context_features",
        },
        "features": {
            "use_scalar_context": True,
            "scalar_context_key": "scalar_context_features",
            "scalar_context_standardize": True,
        },
    }
    scaler = {"mean": [0.0] * 12, "std": [1.0] * 12}

    metadata = scalar_context_metadata(config, scaler)

    assert "scalarContext" in run_id
    assert "autoregressive" not in str(run_dir(run_id))
    assert metadata["features"]["scalar_context_feature_names"] == list(
        SCALAR_CONTEXT_FEATURE_NAMES
    )
    assert metadata["features"]["scalar_scaler_fitted_on"] == "train"
    assert metadata["scalar_context_scaler"] == scaler
