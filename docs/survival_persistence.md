# Survival Persistence

The `survival_persistence` task builds a model-independent continuous-time
survival dataset for grouped fade episodes.

At a decision timestamp `t`, the future target is:

```text
S(u | X_t) = P(R_t > u | X_t)
R_t = event_end_time - t
```

`event_end_time` is the start of a stable recovery interval for the current
fade episode. It is not the timestamp at which recovery can finally be
confirmed.

## Event State Machine

Events are constructed from the full cleaned signal inside each continuous
acquisition segment.

The state machine has four conceptual states:

```text
NORMAL
ACTIVE_FADE
CANDIDATE_RECOVERY
CONFIRMED_RECOVERY
```

An event starts when the activation condition is satisfied, for example:

```text
Signal > threshold_on
```

A candidate recovery starts when the recovery condition is satisfied, for
example:

```text
Signal <= threshold_off
```

The candidate recovery becomes confirmed only when the configured future
recovery window satisfies the required recovery fraction. Temporary movements
below the activation threshold do not end an event unless this stable-recovery
rule is satisfied.

The implemented transition rules are:

```text
NORMAL -> ACTIVE_FADE
    when threshold_on_condition is true.

ACTIVE_FADE -> CANDIDATE_RECOVERY
    when threshold_off_condition is true.

CANDIDATE_RECOVERY -> ACTIVE_FADE
    when a later threshold_on crossing occurs and
    abort_candidate_recovery_on_threshold_on_crossing is true.

CANDIDATE_RECOVERY -> NORMAL
    when stable recovery is confirmed.
```

The default configuration uses:

```text
threshold_on = 10.0
threshold_off = 10.0
recovery_window_seconds = 300
recovery_required_fraction = 0.8
minimum_recovery_observations = 2
```

Different `threshold_on` and `threshold_off` values support hysteresis.

## Recovery-Window Convention

The recovery window is timestamp-based, not sample-count based.

For a candidate recovery starting at `c`, the audited convention is:

```text
[c, c + recovery_window_seconds]
```

Both endpoints are inclusive. Observations exactly at the candidate timestamp
and exactly at the right boundary are included in the denominator.

For the initial `sample_fraction` mode:

```text
recovery_fraction =
    count(observations inside the inclusive window satisfying threshold_off)
    /
    count(observations inside the inclusive window)
```

Stable recovery is confirmed only when:

```text
num_observations >= minimum_recovery_observations
recovery_fraction >= recovery_required_fraction
```

With the default configuration, a new `threshold_on` crossing after the
candidate start aborts the candidate immediately. A later recovery candidate can
start from a subsequent timestamp.

## End Time And Confirmation Time

Observed events store two timestamps:

```text
event_end_time
event_end_confirmation_time
```

`event_end_time` is the first timestamp of the stable recovery interval.
`event_end_confirmation_time` is the later timestamp where enough future
observations are available to confirm that the recovery was stable.

The survival target uses only:

```text
event_end_time - sample_time
```

The confirmation delay is retained for interpretation, but it is not added to
the physical remaining duration.

The configured dataset builder can first apply conservative small-gap
interpolation to the full clean signal before event detection and target
construction. The default policy matches the prepared event-window branch:
expected 30-second sampling, maximum 90-second gap, maximum two missing samples,
linear interpolation in time, and no interpolation across `segment_id`
boundaries.

## Censoring

If stable recovery is not observed before the continuous acquisition segment
ends, the event is right-censored.

Observed samples store:

```text
y_event_observed = 1
y_lower_bound_seconds = y_upper_bound_seconds = y_time_seconds
```

Censored samples store:

```text
y_event_observed = 0
y_lower_bound_seconds = segment_end_time - sample_time
y_upper_bound_seconds = inf
```

Censored samples are kept in the dataset. They are not dropped.

## Sample Generation

The default policy is:

```text
sample_policy = all_event_timestamps
```

Observed events produce samples for:

```text
event_start_time <= sample_time < event_end_time
```

The sample at `event_end_time` is excluded so all survival times remain
strictly positive.

Temporary below-threshold samples inside an active fade remain valid samples.
The dataset stores `current_above_threshold` so later training or operational
evaluation can restrict decision timestamps if needed.

## Saved Inputs

The canonical dataset stores:

```text
X_raw
X_relative_to_threshold
X_relative_to_current
scalar_context_features
```

Scalar features are computed only from past/current data and are standardized
with train-only statistics.

The slope-style scalar features use sample-index differences inside the context
window, not timestamp-normalized rates. They are therefore local signal-change
features, not physical dB-per-second derivatives.

The canonical dataset deliberately excludes target or future fields from model
inputs, including:

```text
event_end_time
event_end_confirmation_time
event_duration_seconds
y_event_observed
y_time_seconds
future threshold crossings
final event position fraction
```

## Model Adapters

The canonical dataset is continuous-time. It does not save fixed DeepHit or
discrete-time masks.

Model adapters derive their own labels:

- XGBoost-AFT: use lower and upper survival bounds;
- discrete-time TCN: derive bins and masks from model-configured bin edges in memory;
- future DeepHit: derive event/censoring bins from model-configured bin edges.

Changing time bins must not require rebuilding this dataset.

The current audit verified that the same canonical labels can support:

- XGBoost-AFT lower/upper bounds;
- discrete-time masks generated in memory from at least two bin sets;
- DeepHit-style event/censoring bin assignment with an explicit tail bin.

The recommended future tail strategy is an explicit tail bin after the last
finite edge, rather than truncating long or censored observations silently.

## Current Canonical Dataset

Default canonical build:

```text
threshold_on = 10.0
threshold_off = 10.0
context_length = 30
external_holdout = fc-uplink-fade.csv
```

Processed dataset:

```text
data/processed/survival_persistence/threshold_10p0/L30/externalHoldout_test_fc_uplink_fade/
```

Audit tables:

```text
results/data_preparation/survival_persistence/externalHoldout_test_fc_uplink_fade_threshold10p0_L30/audit/
```

Audited dataset counts:

```text
train:      117 events, 1655 samples, 0 censored events
validation:  24 events,  774 samples, 1 censored event
test:        56 events,  900 samples, 2 censored events
```

Additional diagnostics from the audit:

```text
represented events: 197
observed events: 194
censored events: 3
context exclusions: 0
```

The 300-second value is reported only as a diagnostic threshold. It is not used
to filter the survival dataset.

## Build Command

```bash
conda run -n Nowcasting python scripts/experiments/survival_persistence/build_dataset.py \
  --config configs/survival_persistence/dataset_threshold10_L30_external_holdout.yaml \
  --force
```

## Audit Command

```bash
conda run -n Nowcasting python scripts/experiments/survival_persistence/audit_dataset.py \
  --config configs/survival_persistence/dataset_threshold10_L30_external_holdout.yaml
```

## XGBoost-AFT Baseline

The first model baseline uses XGBoost with `objective: survival:aft`.
It consumes configurable tabular feature sets from the canonical dataset and
uses the saved AFT labels directly:

```text
label_lower_bound = y_lower_bound_seconds
label_upper_bound = y_upper_bound_seconds
```

Observed samples have equal lower/upper bounds. Right-censored samples keep
`upper = inf`; they are not dropped or converted to finite pseudo-targets.

Default config:

```text
configs/survival_persistence/models/xgboost_aft_scalar_context.yaml
```

Run:

```bash
conda run -n Nowcasting env PYTHONPATH=. python scripts/experiments/survival_persistence/run_xgboost_aft.py \
  --config configs/survival_persistence/models/xgboost_aft_scalar_context.yaml
```

Outputs:

```text
results/runs/survival_persistence/<selection_id>/<run_id>/
models/survival_persistence/xgboost_aft/<run_id>/
results/index/survival_persistence_runs.csv
```

The runner saves split predictions, AFT loss, Harrell C-index, IPCW Brier
scores, calibration tables, train-fitted Kaplan-Meier baseline metrics,
feature names, feature importance, and diagnostic plots. Early stopping uses
validation only; test is evaluated once after model selection.

## XGBoost-AFT Grid Search

The grid-search runner evaluates a validation-only grid over feature set,
sample weighting, AFT distribution, AFT scale, and XGBoost tree parameters.
It uses all available CUDA GPUs for independent trials when possible; if CUDA
is unavailable it falls back to MPS when usable by the model backend, otherwise
CPU. XGBoost does not support MPS directly, so MPS is recorded as a CPU
fallback for this model family.

Default config:

```text
configs/survival_persistence/models/xgboost_aft_grid_search.yaml
```

The default grid currently contains `1920` trials.

Run:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/experiments/survival_persistence/run_xgboost_aft_grid_search.py \
  --config configs/survival_persistence/models/xgboost_aft_grid_search.yaml
```

Outputs:

```text
results/grid_searches/survival_persistence/<selection_id>/<search_id>/
results/comparisons/model_selection/survival_persistence/<selection_id>/<search_id>/
results/runs/survival_persistence/<selection_id>/<best_run_id>/
models/survival_persistence/xgboost_aft/<best_run_id>/
```

## Discrete-Time TCN Baseline

The second survival baseline is a causal PyTorch TCN that predicts one discrete
hazard logit per survival-time bin:

```text
h_k = P(R in bin k | R survived all previous bins, X_t)
S_k = product_j<=k (1 - h_j)
```

The canonical continuous-time dataset is not rebuilt. The runner converts
`y_time_seconds`, `y_event_observed`, `y_lower_bound_seconds`, and
`y_upper_bound_seconds` to discrete event/censoring masks in memory.
Observed samples contribute survived bins before the event plus the event bin.
Right-censored samples contribute only fully observed survived bins; the
partially observed censoring bin is excluded.

Smoke config:

```text
configs/survival_persistence/models/discrete_time_tcn_smoke.yaml
```

Run:

```bash
conda run -n Nowcasting env PYTHONPATH=. python scripts/experiments/survival_persistence/run_discrete_time_tcn.py \
  --config configs/survival_persistence/models/discrete_time_tcn_smoke.yaml
```

Outputs:

```text
results/runs/survival_persistence/<selection_id>/<run_id>/
models/survival_persistence/discrete_time_tcn/<run_id>/
results/index/survival_persistence_runs.csv
```

The smoke baseline uses `relative_to_threshold` sequence input plus train-
standardized scalar context. It uses validation NLL for early stopping and
evaluates the external test split once after selecting the best epoch.

## Controlled Discrete-Time TCN Selection

The controlled-selection runner is separate from the smoke runner. It runs the
staged validation-only protocol:

```text
stage_a_binning
stage_b_input
stage_c_weighting
stage_d_architecture
stage_e_seed_stability
final_test
```

Stages A-E load only train and validation artifacts. The external test split is
loaded only by `final_test`, after `final_selection_record.yaml` has been
written from validation evidence.

Config:

```text
configs/survival_persistence/models/discrete_time_tcn_controlled.yaml
```

Smoke-integrity check:

```bash
conda run -n Nowcasting env PYTHONPATH=. python scripts/experiments/survival_persistence/run_discrete_time_tcn_controlled.py \
  --config configs/survival_persistence/models/discrete_time_tcn_controlled.yaml \
  --stage smoke_integrity
```

Run the full controlled protocol:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/experiments/survival_persistence/run_discrete_time_tcn_controlled.py \
  --config configs/survival_persistence/models/discrete_time_tcn_controlled.yaml \
  --stage all
```

Outputs:

```text
results/comparisons/model_selection/survival_persistence/<selection_id>/<search_id>/
results/runs/survival_persistence/<selection_id>/<controlled_run_id>/
models/survival_persistence/discrete_time_tcn/<controlled_run_id>/
```

Selection uses validation Integrated Brier Score as the primary metric. The
runner saves stage comparison tables, selected-stage YAML files, seed-stability
tables, `final_selection_record.yaml`, and final test metrics only after the
selection has been frozen.

## Switch Evaluation

The survival model does not directly output a binary switch. The current
operational conversion uses the saved survival probability at the 300-second
horizon:

```text
model_switch_raw(t) = 1 if S(300s | X_t) >= 0.5 and Signal(t) > 10.0
```

The threshold `0.5` is a fixed decision rule, not fitted on the test set. After
the raw decision is formed, the shared switch post-processing is applied:

```text
raw switch
  -> min-island rule
  -> stateful hold while the true signal remains above threshold
  -> model_switch_min_time
```

The Perfect Switch reference is task-scoped and uses the same shared reference
implementation as the other tasks.

Commands:

```bash
conda run -n Nowcasting env PYTHONPATH=. python scripts/analysis/survival_persistence/12_compute_perfect_switch.py \
  --config configs/survival_persistence/perfect_switch.yaml

conda run -n Nowcasting env PYTHONPATH=. python scripts/analysis/survival_persistence/13_compare_switch_methods.py \
  --config configs/survival_persistence/switch_comparison.yaml
```

Outputs:

```text
results/switching/perfect_switch/survival_persistence/<selection_id>/
results/comparisons/switch_eval/survival_persistence/<selection_id>/<comparison_id>/
results/comparisons/model_summary/survival_persistence/<selection_id>/<summary_id>/
```

Event plots are displayed over a wider 90-minute-before/90-minute-after window
for visual comparability with other tasks. Switch values outside the native
survival decision timestamps are displayed as zero only in the plot; switch
metrics remain computed only on native survival decision timestamps.
