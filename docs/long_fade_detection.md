# Long-Fade Detection

`long_fade_detection` is the binary follow-up to the older
`current_level_persistence` duration-regression task.

The duration-regression formulation predicted a remaining persistence time in
seconds. It performed poorly for the operational goal, even after learnable
shapelets, wider grids, FP32 training, and scalar context features. The new
task specializes the question into binary grouped-event detection.

## Formulation

At each timestamp `t` inside a grouped fade event, predict whether the whole
grouped event duration is at least a configurable minimum duration:

```text
y_long_fade = 1 if event_duration_seconds >= min_fade_duration_seconds
```

Initial default:

```text
threshold_db = 10.0
min_fade_duration_seconds = 300
sampling_time_seconds = 30
context_length = 30
```

This is not a remaining-duration task. Every timestamp inside the same grouped
event has the same event label.

## Event Membership

Events are grouped with the 3-hour convention:

```text
first crossing at or above threshold starts an event
further crossings within 3 hours from the first crossing stay in that event
new event starts only when a crossing occurs more than 3 hours after that start
```

The event span runs from the first to the last threshold-crossing sample in the
group. Samples below threshold inside that span remain event members. Those
temporary drops do not split the event.

## Inputs

Main sequence input:

```text
X_relative_to_threshold = X_raw - threshold_db
```

Scalar context features:

```text
current_signal
signal_minus_threshold
window_mean
window_std
window_min
window_max
window_range
last_slope
recent_slope_3
recent_slope_5
recent_mean_5
recent_std_5
elapsed_since_fade_start_seconds
position_in_fade_samples
```

Scalar features are standardized using train-only statistics.

Forbidden model inputs:

```text
above_threshold
position_in_fade_fraction
event_duration_seconds
event_duration_samples
inside_event
y_long_fade
```

`position_in_fade_fraction` is forbidden because it requires knowing the final
event duration.

## Dataset

The long-fade dataset is built from the shared prepared event-window signal,
not directly from raw CSV rows. The builder stitches
`data/interim/event_windows/threshold_10p0/all_event_windows.parquet` into one
time-ordered `Time`/`Signal` series per dataset using `Signal_prepared`. This
keeps the long-fade inputs aligned with the common cleaned/imputed signal used
by the autoregressive and current-level persistence branches.

The builder also exposes the shared small-gap imputation switch. With the
current configs this is mostly a safety pass, because `Signal_prepared` already
contains recoverable event-window interpolation; it still never bridges long
acquisition gaps or different loaded series.

Build the dataset:

```bash
conda run -n Nowcasting python scripts/experiments/long_fade_detection/build_dataset.py \
  --config configs/long_fade_detection/xgboost_lag_scalar_threshold10_duration300.yaml
```

Output:

```text
data/processed/long_fade_detection/threshold_10p0/min_duration_300s/L30/externalHoldout_test_fc_uplink_fade/
```

Split policy:

```text
fc-uplink-fade.csv = external test
all other datasets = development
development events = chronological 85% train / 15% validation
```

No event is split across train, validation, and test.

## Models

Implemented first-stage models:

- `xgboost_lag_scalar_classifier`
- `tcn_classifier`
- `multiscale_shapelet_convolution_classifier`

XGBoost uses:

```text
X_lag_scalar = flattened X_relative_to_threshold + scalar_context_features
```

Lag order is oldest to newest:

```text
lag_29, lag_28, ..., lag_0
```

where `lag_0 = S_t - threshold_db`.

TCN and shapelet-convolution models use `X_relative_to_threshold` plus scalar
context.

## Grid Search

The main first-stage experiment is now an extended validation grid across the
three model families:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/long_fade_detection/run_grid_search.py \
  --config configs/long_fade_detection/grid_search_threshold10_duration300.yaml
```

Current grid size:

```text
xgboost_lag_scalar_classifier                  2,880 trials
tcn_classifier                                 3,456 trials
multiscale_shapelet_convolution_classifier     2,592 trials
total                                          8,928 trials
```

The grid selects by validation AUPRC only. For each model family, the selected
configuration is retrained on train+validation and then evaluated once on the
external test split.

Device behavior:

```text
CUDA available -> one independent trial worker per visible CUDA GPU
MPS/CPU only   -> one worker on the selected fallback device
```

## Single-Run Commands

These are useful for debugging one configured model without running the full
grid.

XGBoost:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/long_fade_detection/run_model.py \
  --config configs/long_fade_detection/xgboost_lag_scalar_threshold10_duration300.yaml
```

TCN:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/long_fade_detection/run_model.py \
  --config configs/long_fade_detection/tcn_threshold10_duration300.yaml
```

Shapelet convolution:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/long_fade_detection/run_model.py \
  --config configs/long_fade_detection/shapelet_conv_threshold10_duration300.yaml
```

Smoke server launcher:

```bash
bash scripts/experiments/long_fade_detection/run_server_smoke.sh
```

Full server launcher:

```bash
bash scripts/experiments/long_fade_detection/run_server.sh
```

Use these launchers inside `tmux` for remote runs. The full server launcher
runs the extended grid. The smoke launcher runs a grid dry-run and does not
train models.

## Metrics

Sample-level metrics include AUROC, AUPRC, accuracy, balanced accuracy,
precision, recall, F1, specificity, confusion counts, and Brier score.

Thresholds evaluated:

```text
0.3, 0.5, 0.7
```

Event-level probabilities are aggregated with:

```text
mean_prob
max_prob
last_prob
early_mean_prob
```

where `early_mean_prob` uses the first `min(10, event_length)` samples.

## Summary

After runs complete:

```bash
conda run -n Nowcasting python scripts/experiments/long_fade_detection/summarize_runs.py
```

Outputs:

```text
results/comparisons/model_summary/long_fade_detection/<selection_id>/long_fade_model_summary_thr10p0_dur300/tables/long_fade_model_comparison.csv
results/comparisons/model_summary/long_fade_detection/<selection_id>/long_fade_model_summary_thr10p0_dur300/reports/long_fade_model_comparison.md
```

## Switch Evaluation

The long-fade classifiers output `P(long_fade)`. The current switch conversion
uses:

```text
model_switch_raw(t) = 1 if P(long_fade | X_t) >= 0.5 and Signal(t) >= 10.0
```

Then the shared min-island and stateful signal-above-threshold hold are applied.

Commands:

```bash
conda run -n Nowcasting python scripts/analysis/long_fade_detection/12_compute_perfect_switch.py \
  --config configs/long_fade_detection/perfect_switch.yaml

conda run -n Nowcasting python scripts/analysis/long_fade_detection/13_compare_switch_methods.py \
  --config configs/long_fade_detection/switch_comparison.yaml
```

If only plot layout changed and existing model-vs-Perfect parquet files are
still valid, refresh plots without recomputing metrics:

```bash
conda run -n Nowcasting env PYTHONPATH=. python scripts/analysis/long_fade_detection/13_compare_switch_methods.py \
  --config configs/long_fade_detection/switch_comparison.yaml \
  --refresh-plots-only
```

Event plots are displayed over a wider 90-minute-before/90-minute-after window.
Switch values outside native long-fade decision timestamps are displayed as zero
only in the plot; metrics remain computed only on native long-fade timestamps.

## Diagnostic Plots

After switch comparisons exist, run:

```bash
conda run -n Nowcasting python scripts/analysis/16_plot_switch_diagnostics.py \
  --config configs/switch_diagnostics.yaml
```

Long-fade diagnostic figures are written under:

```text
results/comparisons/switch_diagnostics/long_fade_detection/<selection_id>/<method_id>/figures/
```

The main long-fade plots are:

- `event_timelines/`: true signal, Perfect Switch, raw probability-derived
  switch, and final post-processed switch;
- `task_specific/probability_calibration.png`: predicted `P(long_fade)` versus
  observed long-fade frequency by probability bin;
- `task_specific/probability_distribution.png`: distribution of predicted
  long-fade probabilities, useful to see whether a classifier is saturated near
  0, near 1, or around the decision threshold.

Cross-task figures are explained in
[Switch Diagnostics And Cross-Task Plots](switch_diagnostics.md).
