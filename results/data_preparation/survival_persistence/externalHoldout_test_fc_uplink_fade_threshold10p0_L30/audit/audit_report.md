# Survival Persistence Dataset Audit

Config: `configs/survival_persistence/dataset_threshold10_L30_external_holdout.yaml`
Dataset: `data/processed/survival_persistence/threshold_10p0/L30/externalHoldout_test_fc_uplink_fade`
Config fingerprint: `638fd82dd0f210108292c2de01869c313fdbe8ac015aa81a1ae8aa67c2ab366a`

## Event Counts

| split | number_of_events | number_observed | number_censored | censoring_rate |
| --- | --- | --- | --- | --- |
| train | 117 | 117 | 0 | 0.0 |
| validation | 23 | 22 | 1 | 0.043478260869565216 |
| test | 55 | 53 | 2 | 0.03636363636363636 |

## Sample Counts

| split | number_of_samples | observed_target_samples | censored_target_samples | above_threshold_samples | below_threshold_internal_samples | pct_above_threshold_samples | pct_below_threshold_internal_samples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1729 | 1729 | 0 | 1421 | 308 | 82.18623481781377 | 17.813765182186234 |
| validation | 729 | 723 | 6 | 666 | 63 | 91.35802469135803 | 8.641975308641975 |
| test | 921 | 851 | 70 | 775 | 146 | 84.14766558089035 | 15.852334419109662 |

## Compatibility

| adapter | lower_bounds_strictly_positive | observed_upper_equals_lower | censored_upper_is_inf | tabular_features_all_finite | passed |
| --- | --- | --- | --- | --- | --- |
| xgboost_aft | True | True | True | True | True |
| discrete_time_tcn_and_deephit | True |  |  |  | True |

## Bugs Found

- No dataset bugs found by this audit.

Survival models were not implemented or trained during this audit.
