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

Future model adapters should derive their own labels:

- XGBoost-AFT: use lower and upper survival bounds;
- discrete-time TCN: derive bins and masks from model-configured bin edges;
- DeepHit: derive event/censoring bins from model-configured bin edges.

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
test:        55 events,  875 samples, 2 censored events
```

Additional diagnostics from the audit:

```text
represented events: 196
observed events: 193
censored events: 3
events with internal below-threshold samples: 86
observed events shorter than 300 seconds: 93
context exclusions: 25
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
