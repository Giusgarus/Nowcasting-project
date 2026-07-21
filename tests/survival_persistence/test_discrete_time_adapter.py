from __future__ import annotations

import numpy as np

from src.tasks.survival_persistence.adapters.discrete_time import (
    continuous_to_discrete_survival_labels,
    hazards_to_survival,
    make_discrete_time_bin_spec,
    median_survival_time_seconds,
    survival_at_horizons,
)


def test_discrete_labels_respect_observed_and_censoring_boundaries() -> None:
    spec = make_discrete_time_bin_spec(
        strategy="hybrid",
        bin_edges_seconds=[0, 30, 60, 120],
        include_open_ended=True,
    )

    labels = continuous_to_discrete_survival_labels(
        y_time_seconds=np.asarray([30.0, 60.0, 45.0, 60.0]),
        y_event_observed=np.asarray([1, 1, 0, 0]),
        y_lower_bound_seconds=np.asarray([30.0, 60.0, 45.0, 60.0]),
        y_upper_bound_seconds=np.asarray([30.0, 60.0, np.inf, np.inf]),
        bin_spec=spec,
    )

    assert labels.event_bin_index.tolist() == [0, 1, -1, -1]
    assert labels.censoring_bin_index.tolist() == [-1, -1, 0, 1]
    assert labels.at_risk_mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, False, False, False],
        [True, True, False, False],
    ]
    assert labels.event_mask.tolist() == [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]


def test_hazard_survival_horizon_and_capped_median_are_stable() -> None:
    spec = make_discrete_time_bin_spec(
        strategy="hybrid",
        bin_edges_seconds=[0, 30, 60, 120],
        include_open_ended=True,
    )
    hazards = np.asarray([[0.1, 0.2, 0.5, 0.9], [0.01, 0.01, 0.01, 0.01]])

    survival = hazards_to_survival(hazards)
    horizon_survival = survival_at_horizons(
        survival,
        bin_spec=spec,
        horizons_seconds=[30, 90, 120],
    )
    median = median_survival_time_seconds(survival, bin_spec=spec)

    assert survival.shape == (2, 4)
    assert np.all(np.diff(survival, axis=1) <= 0.0)
    np.testing.assert_allclose(horizon_survival[0], [0.9, 0.36, 0.36])
    assert median.tolist() == [120.0, 120.0]
