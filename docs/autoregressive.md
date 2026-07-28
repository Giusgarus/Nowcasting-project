# Autoregressive Forecasting

## Task Definition

The autoregressive task predicts the future `Signal` trajectory from a past
signal context.

At prediction time `t`, the model input is:

```text
X_t = [S_{t-L+1}, ..., S_t]
```

The target is:

```text
Y_t = [S_{t+1}, ..., S_{t+h}]
```

The current default setup is:

```text
signal_threshold = 10.0
context_length = 30
prediction_length = 10
sampling_time_seconds = 30
```

The longer-context Chronos diagnostic setup uses:

```text
context_length = 120
prediction_length = 10
```

Forecast metrics are reported as diagnostics. The operational evaluation is
based on switch decisions derived from the predicted trajectory and compared
against the model-independent Perfect Switch reference.

## Data Policy

The task is strictly signal-only.

Every loaded dataset is standardized to:

```text
Time
Signal
```

Models receive only `Signal`. Extra source columns are ignored. The input
window never includes future target values.

## Dataset Pipeline

Autoregressive data preparation has three stages.

### 1. Event Windows

Configuration:

```text
configs/data_preparation.yaml
```

Command:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/09_prepare_event_windows.py
```

This stage loads raw signal files, builds clean signal tables, detects
candidate threshold crossings, creates event-centered windows, records quality
flags, and writes the no-leakage window index.

Main outputs:

```text
data/interim/clean_signal/
data/interim/candidate_events/
data/interim/event_windows/
data/processed/autoregressive/threshold_10p0/window_index_L30_h10.parquet
results/data_preparation/autoregressive/
```

Candidate events are used to select useful windows. They are not the final
Perfect Switch definition.

### 2. Final Supervised Dataset

Configuration:

```text
configs/autoregressive/autoregressive_dataset.yaml
```

Command:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/10_build_autoregressive_datasets.py
```

To intentionally rebuild an existing or partial dataset folder, add `--force`:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/10_build_autoregressive_datasets.py \
  --config configs/autoregressive/autoregressive_dataset.yaml \
  --force
```

The default dataset protocol is external holdout:

```yaml
dataset_selection:
  mode: external_holdout
  heldout_test_dataset: fc-uplink-fade.csv
```

This means:

- `fc-uplink-fade.csv` is used only for final test evaluation;
- the remaining datasets form the development pool;
- development events are split chronologically into train and validation;
- datasets with too few usable events remain entirely in train;
- no test data is used for hyperparameter selection, early stopping, or
  threshold calibration.

Main output:

```text
data/processed/autoregressive/threshold_10p0/datasets_L30_h10/externalHoldout_test_fc_uplink_fade/
```

The dataset folder contains:

```text
train.npz
val.npz
test.npz
train_metadata.parquet
val_metadata.parquet
test_metadata.parquet
dataset_metadata.yaml
```

Each NPZ stores raw and context-standard variants:

```text
X_raw
y_raw
X_context_standard
y_context_standard
scaling_mean
scaling_std
window_id
event_id
dataset_id
```

For `context_standard`, scaling is fitted only from the input context of that
window. Targets and predictions are converted back to raw signal scale for
metrics.

### 3. Sanity Check

Command:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/11_sanity_check_autoregressive_dataset.py \
  --no-plots
```

This checks shapes, metadata alignment, split integrity, normalization
reconstruction, missing values, infinities, and event leakage.

Do not start a model campaign unless the script finishes with:

```text
All sanity checks passed.
```

## Perfect Switch Reference

Configuration:

```text
configs/perfect_switch.yaml
```

Command:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/12_compute_perfect_switch.py
```

The Perfect Switch is model-independent and must exist before model-vs-switch
comparisons are generated.

Main outputs:

```text
results/switching/perfect_switch/autoregressive/<selection_id>/
results/index/switch_references.csv
```

Changing the Perfect Switch rule is a methodological change because every
switch metric depends on it.

## Implemented Models

### GRU

Implemented architectures:

```text
gru_s2v
gru_seq2seq
```

Single-run config:

```text
configs/autoregressive/autoregressive_gru.yaml
```

Command:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_autoregressive_gru.py
```

Grid-search config:

```text
configs/autoregressive/gru_grid_search.yaml
```

Command:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_gru_grid_search.py
```

The grid search selects by raw-scale validation RMSE. When final retraining is
enabled, the selected hyperparameters are retrained on train+validation for the
validation-selected `best_epoch`, then evaluated once on the external test
split.

### PatchTST

Single-run config:

```text
configs/autoregressive/autoregressive_patchtst.yaml
```

Command:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_autoregressive_patchtst.py
```

Grid-search config:

```text
configs/autoregressive/patchtst_grid_search.yaml
```

Command:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_patchtst_grid_search.py
```

The PatchTST grid also selects by raw-scale validation RMSE and evaluates test
only after the validation winner is fixed.

### XGBoost

XGBoost is implemented as a tabular autoregressive baseline. The input context
is flattened from `(N, context_length, 1)` to `(N, context_length)`.
Because the task target is a 10-step trajectory, the runner trains 10
independent regressors per trial, one for each horizon step.

Grid-search config:

```text
configs/autoregressive/xgboost_grid_search.yaml
```

Command:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_xgboost_grid_search.py \
  --config configs/autoregressive/xgboost_grid_search.yaml
```

The grid is expressed with inclusive `start`/`stop`/`step` ranges rather than
manual value lists. It evaluates `raw` and `context_standard` variants,
selects by raw-scale validation RMSE, then retrains the selected parameters on
train+validation and evaluates the external test split once.

### Chronos Zero-Shot

Default config:

```text
configs/autoregressive/autoregressive_chronos.yaml
```

Command:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_autoregressive_chronos_zero_shot.py
```

Longer-context config:

```text
configs/autoregressive/autoregressive_chronos_L120_h10.yaml
```

Command:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_autoregressive_chronos_zero_shot.py \
  --config configs/autoregressive/autoregressive_chronos_L120_h10.yaml
```

Chronos uses pretrained `amazon/chronos-t5-small` in zero-shot mode. It does
not train, fine-tune, load validation data, or save a trained checkpoint. Point
metrics use the median forecast; q10/q50/q90 forecasts are also saved when
available.

## Switch Conversion

Model forecasts are converted to one decision at each `input_end_time`.

Configuration:

```text
configs/switch_comparison.yaml
```

Current rule:

```yaml
conversion:
  prediction_aggregation: horizon_threshold_count
  required_points_above_threshold: 10
  threshold: 10.0
  condition: greater_than_or_equal
```

With `prediction_length = 10`, this means:

```text
switch(t) = 1 if all 10 predicted future points are >= 10.0
```

The configured switch post-processing is then applied. Switch comparisons use
the post-processed model switch against the post-processed Perfect Switch.

## Switch Comparisons

Command:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/13_compare_switch_methods.py
```

For Chronos `L120_h10`:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/13_compare_switch_methods.py \
  --config configs/switch_comparison_chronos_L120_h10.yaml
```

Main outputs:

```text
results/comparisons/switch_eval/autoregressive/<selection_id>/<comparison_id>/
results/index/comparisons.csv
```

Each comparison folder contains exactly one model/run compared with one
reference. Batch configs may list multiple methods, but outputs stay separate.

## Cross-Model Summary Tables

Command:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/14_compare_model_results.py
```

Main outputs:

```text
results/comparisons/model_summary/autoregressive/<selection_id>/<comparison_id>/tables/validation_forecast_comparison.csv
results/comparisons/model_summary/autoregressive/<selection_id>/<comparison_id>/tables/test_forecast_switch_comparison.csv
results/comparisons/model_summary/autoregressive/<selection_id>/<comparison_id>/tables/global_switch_behavior_comparison.csv
```

The validation table is used for model selection diagnostics. The test table
contains final forecast and switch metrics. Chronos zero-shot has no validation
metrics because it never loads validation data.

## Artifact Layout

Model run outputs:

```text
results/runs/autoregressive/<selection_id>/<run_id>/
```

Model checkpoints:

```text
models/autoregressive/<model_family>/<run_id>/
```

Grid-search outputs:

```text
results/grid_searches/autoregressive/<selection_id>/<search_id>/
```

Perfect Switch outputs:

```text
results/switching/perfect_switch/autoregressive/<selection_id>/
```

Comparison outputs:

```text
results/comparisons/<comparison_type>/autoregressive/<selection_id>/<comparison_id>/
```

Central indexes:

```text
results/index/datasets.csv
results/index/runs.csv
results/index/grid_searches.csv
results/index/switch_references.csv
results/index/comparisons.csv
```

## Leakage Rules

- Inputs may use only `Signal` values available up to prediction time `t`.
- Targets may use future `Signal` values, but target values never enter the
  input window.
- Splits are chronological and event-aware.
- Grid searches select using validation metrics only.
- The external test dataset is used only after model selection is finished.
- Switch thresholds and conversion settings must not be tuned on test data.
