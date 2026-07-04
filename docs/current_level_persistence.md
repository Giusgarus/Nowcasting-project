# Current-Level Persistence

## Task Definition

`current_level_persistence` predicts how long the signal will remain above a
level defined relative to the current observation.

At prediction time `t`, the model receives only the recent signal history:

```text
X_raw = [S_{t-L+1}, ..., S_t]
```

The current default setup is:

```text
delta = 0.5
context_length = 30
sampling_time_seconds = 30
```

The recovery level is:

```text
recovery_level = S_t - delta
```

The label searches the complete cleaned signal after `t` for the first
observation for which:

```text
Signal >= recovery_level
```

The duration is the actual timestamp difference between `t` and that first
recovery observation. The event-centered window limits which decision
timestamps are modelled, but it does not truncate the future target search.

The 30-point context must remain inside one continuous `segment_id`. If no
recovery is observed before that acquisition segment ends, the target is
right-censored and excluded from supervised duration regression. A later value
after a data gap is not treated as proof that the signal remained above the
level during the unobserved interval.

## Difference From Autoregressive Forecasting

Autoregressive forecasting predicts the next signal trajectory:

```text
[S_{t+1}, ..., S_{t+h}]
```

Current-level persistence predicts one scalar duration target instead:

```text
remaining_persistence_seconds
```

It is therefore a separate task with separate datasets, results, models, and
configuration files.

## Input Representation

The raw context is saved as:

```text
X_raw
```

The main representation for shapelet models is:

```text
X_relative_to_current = X_raw - S_t
```

The last value of `X_relative_to_current` is exactly zero.

To preserve absolute-level information, shapelet models also receive
`scalar_context_features` in this fixed order:

```text
current_signal, recovery_level, window_mean, window_std,
window_min, window_max, window_range, last_slope,
recent_slope_3, recent_slope_5, recent_mean_5, recent_std_5
```

These values are derived from `X_raw`. Their scaler is fitted on train only and
reused unchanged for validation and test. `X_relative_to_current` remains the
input to the shapelet encoder.

## Targets

The dataset stores:

```text
remaining_persistence_samples
remaining_persistence_seconds
log1p_remaining_persistence_seconds
target_observed
recovery_time
censoring_lower_bound_seconds
```

`remaining_persistence_seconds` is the primary target.
`remaining_persistence_samples` is derived as
`ceil(remaining_persistence_seconds / sampling_time_seconds)` for diagnostics.
Censored candidates are written separately to
`censored_window_metadata.parquet`; they are not included in train,
validation, or test NPZ arrays.

Models train on `log1p_remaining_persistence_seconds` by default and evaluation
reports metrics in raw seconds and minutes.

## Artifact Conventions

Dataset configuration:

```text
configs/current_level_persistence/dataset_delta_0p5_L30_external_holdout.yaml
```

Build command:

```bash
python scripts/experiments/current_level_persistence/build_dataset.py \
  --config configs/current_level_persistence/dataset_delta_0p5_L30_external_holdout.yaml
```

If the dataset folder already contains a complete build, the script skips
without overwriting. Pass `--force` only when intentionally rebuilding the same
dataset.

The builder reads the shared `event_window_quality.parquet` table. Events
marked `unusable` are excluded, while warning events are included only when
`event_window_quality.allow_warning_events_for_window_index` is enabled. The
stored `quality_flag` and `quality_reason` therefore come from common data
preparation rather than being assigned by this task.

Targets are resolved from `data/interim/clean_signal/all_signal_clean.parquet`,
not from the clipped event window. `censoring_summary.csv` reports observed and
censored candidate counts per split.

Datasets are stored under:

```text
data/processed/current_level_persistence/delta_0p5/L30/externalHoldout_test_fc_uplink_fade/
```

Run outputs use:

```text
results/runs/current_level_persistence/<selection_id>/<run_id>/
```

Checkpoints use:

```text
models/current_level_persistence/<model_id>/<run_id>/
```

Scalar-context run IDs use:

```text
currentLevelPersistence_delta0p5_L30_<model_id>_scalarContext_<selection_id>
```

This keeps them separate from earlier shapelet-only runs.

## Prepared Model Families

The initial learnable-shapelet model IDs are:

```text
multiscale_shapelet_mlp
multiscale_shapelet_transformer
multiscale_shapelet_convolution
```

All three use:

```yaml
shapelet_lengths: [5, 10, 15]
n_shapelets_per_length: 32
```

The common pipeline is:

```text
X_relative_to_current -> multiscale learnable shapelet layer -> shapelet representation
scalar_context_features -> Linear -> ReLU -> Dropout -> scalar embedding
shapelet representation + scalar embedding -> regression head
  -> predicted log1p remaining persistence seconds
```

For the convolutional variant, response maps have different lengths because
the shapelet lengths differ. The implementation processes each shapelet length
as a separate scale, pools each scale, then concatenates the pooled scale
features before the final regression layer. The convolutional response maps are
log-compressed and normalized per scale for numerical stability.

All three configs enable the auxiliary scalar branch by default with a 32-unit
embedding. Set `features.use_scalar_context: false` only to reproduce the old
shapelet-only architecture.

## Train Learnable-Shapelet Models

Run the task-scoped training entrypoint from the repository root:

```bash
python scripts/experiments/current_level_persistence/train_learnable_shapelets.py \
  --config configs/current_level_persistence/shapelet_mlp_delta_0p5.yaml
```

The other implemented variants use:

```bash
python scripts/experiments/current_level_persistence/train_learnable_shapelets.py \
  --config configs/current_level_persistence/shapelet_transformer_delta_0p5.yaml

python scripts/experiments/current_level_persistence/train_learnable_shapelets.py \
  --config configs/current_level_persistence/shapelet_convolution_delta_0p5.yaml
```

Each run saves validation/test predictions, metrics, diagnostic figures,
learned-shapelet plots, `metadata.yaml`, and `config_resolved.yaml` under:

```text
results/runs/current_level_persistence/<selection_id>/<run_id>/
```

Best and last checkpoints are saved under:

```text
models/current_level_persistence/<model_id>/<run_id>/
```

The scalar-context grid has a distinct `search_id` containing `scalarContext`,
so it does not overwrite the previous search.

## Server Launchers

Dataset rebuild, smoke-run, and three-GPU commands are documented in:

```text
docs/current_level_persistence_server_runs.md
```

## Grid Search

Use the grid runner when selecting among small architecture and optimization
variants:

```bash
python scripts/experiments/current_level_persistence/run_shapelet_grid_search.py \
  --config configs/current_level_persistence/shapelet_grid_search_delta_0p5.yaml
```

The default scalar-context grid is intentionally extensive:

```text
216 trials for multiscale_shapelet_mlp
216 trials for multiscale_shapelet_transformer
216 trials for multiscale_shapelet_convolution
648 trials total
```

The configurable parameters live in:

```text
configs/current_level_persistence/shapelet_grid_search_delta_0p5.yaml
```

The grid varies learning rate (`0.0001/0.0003/0.001`), shapelets per length
(`16/32/64`), dropout (`0.0/0.1`), scalar-encoder width (`16/32`), model depth
(`1/2`), and model-specific capacity. Capacity values are `64/128/256` for the
MLP and `32/64/128` for Transformer and convolution. It uses
the shared device-aware scheduler, so independent trials can run across
multiple visible CUDA GPUs.

Mixed precision is disabled for these shapelet models. The shapelet extractor
uses squared distances, which are fragile in half precision when relative
signal windows contain large deviations. CUDA is still used, but training runs
in fp32.

Selection is validation-only:

```text
selection_metric = val_mae_seconds
selection_mode   = min
```

The test split is evaluated only after the winner for each model has been
selected. By default, the selected winner is retrained on train+validation for
the best epoch count found during the grid, then saved as the canonical run for
that model ID.

Grid artifacts are stored under:

```text
results/grid_searches/current_level_persistence/<selection_id>/<search_id>/
```

Final selected runs are stored under the normal run and checkpoint folders:

```text
results/runs/current_level_persistence/<selection_id>/<run_id>/
models/current_level_persistence/<model_id>/<run_id>/
```

## Compare Initial Shapelet Models

After training the MLP, Transformer, and convolutional variants, build the
compact comparison table:

```bash
python scripts/experiments/current_level_persistence/summarize_runs.py
```

Main output:

```text
results/comparisons/model_summary/current_level_persistence/externalHoldout_test_fc_uplink_fade/initial_shapelet_model_comparison_delta0p5/tables/initial_shapelet_model_comparison_delta0p5.csv
```

## Switch Derivation

Switch decisions are derived at each current-level prediction timestamp:

```text
model_switch_raw(t) = 1
  if Signal_t > 10
  and predicted_remaining_persistence_seconds(t) >= 300 seconds
```

The duration threshold is configured in
`configs/current_level_persistence/switch_comparison.yaml`; it is an explicit
operational rule and is not calibrated on the test set.

After thresholding, the task reuses the same shared post-processing as the
autoregressive branch, independently within each event:

```text
raw decision
  -> ensure_min_island_length(..., switch_time=10)
  -> hold while the observed Signal remains above 10
  -> model_switch_min_time
```

Compute the task-scoped Perfect Switch and model comparisons with:

```bash
conda run -n Nowcasting python \
  scripts/analysis/current_level_persistence/12_compute_perfect_switch.py

conda run -n Nowcasting python \
  scripts/analysis/current_level_persistence/13_compare_switch_methods.py
```

Each model is stored in a separate comparison folder under
`results/comparisons/switch_eval/current_level_persistence/`. Cross-model
tables are stored under
`results/comparisons/model_summary/current_level_persistence/`.

The conversion and operational switch metrics will be added after duration
models have been validated.
