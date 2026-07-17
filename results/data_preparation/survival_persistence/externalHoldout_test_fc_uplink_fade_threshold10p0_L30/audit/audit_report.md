# Survival Persistence Dataset Audit

Config: `configs/survival_persistence/dataset_threshold10_L30_external_holdout.yaml`
Dataset: `data/processed/survival_persistence/threshold_10p0/L30/externalHoldout_test_fc_uplink_fade`
Config fingerprint: `0b3d213ef5a1661b3f4829af3bae53ac034ab5d0bd04fb48479091bf2fcd450a`

## Event Counts

| split | number_of_events | number_observed | number_censored | censoring_rate |
| --- | --- | --- | --- | --- |
| train | 117 | 117 | 0 | 0.0 |
| validation | 24 | 23 | 1 | 0.041666666666666664 |
| test | 56 | 54 | 2 | 0.03571428571428571 |

## Sample Counts

| split | number_of_samples | observed_target_samples | censored_target_samples | above_threshold_samples | below_threshold_internal_samples | pct_above_threshold_samples | pct_below_threshold_internal_samples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1655 | 1655 | 0 | 1360 | 295 | 82.17522658610272 | 17.82477341389728 |
| validation | 774 | 768 | 6 | 693 | 81 | 89.53488372093024 | 10.465116279069768 |
| test | 900 | 830 | 70 | 761 | 139 | 84.55555555555556 | 15.444444444444445 |

## Compatibility

| adapter | lower_bounds_strictly_positive | observed_upper_equals_lower | censored_upper_is_inf | tabular_features_all_finite | passed |
| --- | --- | --- | --- | --- | --- |
| xgboost_aft | True | True | True | True | True |
| discrete_time_tcn_and_deephit | True |  |  |  | True |

## Bugs Found

- No dataset bugs found by this audit.

Survival models were not implemented or trained during this audit.
