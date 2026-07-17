"""Tests for survival-persistence XGBoost-AFT baseline utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tasks.survival_persistence.evaluation.baselines import (
    kaplan_meier_marginal_baseline,
)
from src.tasks.survival_persistence.evaluation.metrics import (
    event_level_bootstrap,
    harrell_c_index,
    horizon_known_status,
)
from src.tasks.survival_persistence.models.xgboost_aft import (
    aft_label_bounds,
    build_feature_matrix,
    predicted_median_seconds,
    sample_weights,
    survival_probability_at_horizon,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "X_raw": np.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.float32),
        "X_relative_to_threshold": np.asarray(
            [[-9, -8, -7], [-8, -7, -6], [-7, -6, -5]],
            dtype=np.float32,
        ),
        "X_relative_to_current": np.asarray(
            [[-2, -1, 0], [-2, -1, 0], [-2, -1, 0]],
            dtype=np.float32,
        ),
        "scalar_context_features": np.asarray(
            [[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]],
            dtype=np.float32,
        ),
        "scalar_context_feature_names": np.asarray(["current_signal", "window_mean"]),
        "y_time_seconds": np.asarray([60.0, 120.0, 180.0], dtype=np.float32),
        "y_event_observed": np.asarray([1, 0, 1], dtype=np.int64),
        "y_lower_bound_seconds": np.asarray([60.0, 120.0, 180.0], dtype=np.float32),
        "y_upper_bound_seconds": np.asarray([60.0, np.inf, 180.0], dtype=np.float32),
        "event_balanced_weight": np.asarray([0.5, 0.5, 1.0], dtype=np.float32),
    }


def test_aft_label_bounds_preserve_censored_infinity() -> None:
    lower, upper = aft_label_bounds(_arrays())

    assert lower.tolist() == [60.0, 120.0, 180.0]
    assert np.isinf(upper[1])


def test_aft_label_bounds_reject_observed_interval_mismatch() -> None:
    arrays = _arrays()
    arrays["y_upper_bound_seconds"] = arrays["y_upper_bound_seconds"].copy()
    arrays["y_upper_bound_seconds"][0] = 90.0

    with pytest.raises(ValueError, match="Observed AFT samples"):
        aft_label_bounds(arrays)


def test_feature_matrices_have_deterministic_ordering() -> None:
    first = build_feature_matrix(_arrays(), feature_set="raw_plus_scalar")
    second = build_feature_matrix(_arrays(), feature_set="raw_plus_scalar")

    assert first.names == second.names
    assert first.names == [
        "raw_signal_t_minus_2",
        "raw_signal_t_minus_1",
        "raw_signal_t_minus_0",
        "current_signal",
        "window_mean",
    ]
    assert np.array_equal(first.values, second.values)


def test_feature_matrix_rejects_non_finite_values() -> None:
    arrays = _arrays()
    arrays["scalar_context_features"] = arrays["scalar_context_features"].copy()
    arrays["scalar_context_features"][0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or infinity"):
        build_feature_matrix(arrays, feature_set="scalar_context")


def test_event_balanced_weights_are_loaded_from_arrays() -> None:
    weights = sample_weights(_arrays(), mode="event_balanced")

    assert weights is not None
    assert weights.tolist() == [0.5, 0.5, 1.0]
    assert sample_weights(_arrays(), mode="uniform") is None


def test_aft_survival_probability_is_monotone_and_bounded() -> None:
    location = np.asarray([np.log(300.0)])
    horizons = np.asarray([60.0, 300.0, 900.0])
    probabilities = survival_probability_at_horizon(
        location,
        horizons,
        scale=1.0,
        distribution="normal",
    )[0]

    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert probabilities[0] >= probabilities[1] >= probabilities[2]


def test_aft_logistic_and_extreme_medians() -> None:
    location = np.asarray([np.log(100.0)])

    logistic = predicted_median_seconds(location, scale=1.0, distribution="logistic")
    extreme = predicted_median_seconds(location, scale=1.0, distribution="extreme")

    assert np.isclose(logistic[0], 100.0)
    assert np.isclose(extreme[0], 100.0 * np.log(2.0))


def test_harrell_c_index_comparable_pair_logic() -> None:
    result = harrell_c_index(
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([1, 0, 1]),
        np.asarray([10.0, 20.0, 30.0]),
    )

    assert result["num_comparable_pairs"] == 2
    assert result["c_index"] == 1.0


def test_horizon_status_keeps_censored_before_horizon_unknown() -> None:
    survived, known = horizon_known_status(
        np.asarray([100.0, 200.0, 400.0]),
        np.asarray([0, 1, 0]),
        horizon_seconds=300.0,
    )

    assert survived.tolist() == [0.0, 0.0, 1.0]
    assert known.tolist() == [False, True, True]


def test_kaplan_meier_baseline_uses_train_curve_for_all_samples() -> None:
    baseline = kaplan_meier_marginal_baseline(
        train_times=np.asarray([100.0, 200.0, 300.0]),
        train_event_observed=np.asarray([1, 1, 0]),
        n_samples=2,
        horizons_seconds=np.asarray([50.0, 150.0]),
    )

    assert baseline.survival_probabilities.shape == (2, 2)
    assert np.array_equal(
        baseline.survival_probabilities[0],
        baseline.survival_probabilities[1],
    )


def test_event_bootstrap_resamples_complete_events() -> None:
    frame = pd.DataFrame(
        {
            "event": ["a", "a", "b", "b"],
            "value": [1.0, 2.0, 10.0, 20.0],
        }
    )
    boot = event_level_bootstrap(
        frame,
        event_column="event",
        metric_fn=lambda sample: float(sample["value"].sum()),
        n_replicates=5,
        seed=42,
    )

    assert len(boot) == 5
    assert set(boot.columns) == {"bootstrap_replicate", "metric"}


def test_xgboost_model_save_load_predictions_match_when_available(tmp_path) -> None:
    xgb = pytest.importorskip("xgboost")
    arrays = _arrays()
    dtrain, _ = __import__(
        "src.tasks.survival_persistence.models.xgboost_aft",
        fromlist=["build_dmatrix"],
    ).build_dmatrix(arrays, feature_set="scalar_context")
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        "tree_method": "hist",
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=2)
    before = booster.predict(dtrain, output_margin=True)
    model_path = tmp_path / "booster.json"
    booster.save_model(model_path)
    loaded = xgb.Booster()
    loaded.load_model(model_path)
    after = loaded.predict(dtrain, output_margin=True)

    assert np.allclose(before, after)
