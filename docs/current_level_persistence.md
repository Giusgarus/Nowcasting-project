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

The label counts consecutive samples starting at `t` for which:

```text
Signal >= recovery_level
```

until the first future sample below `recovery_level`.

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

## Targets

The dataset stores:

```text
remaining_persistence_samples
remaining_persistence_seconds
log1p_remaining_persistence_seconds
```

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

The prepared run ID format is:

```text
currentLevelPersistence_delta0p5_L30_<model_id>_<selection_id>
```

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
X_relative_to_current
  -> multiscale learnable shapelet layer
  -> head
  -> predicted log1p remaining persistence seconds
```

For the convolutional variant, response maps have different lengths because
the shapelet lengths differ. The implementation processes each shapelet length
as a separate scale, pools each scale, then concatenates the pooled scale
features before the final regression layer. The convolutional response maps are
log-compressed and normalized per scale for numerical stability.

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

## Grid Search

Use the grid runner when selecting among small architecture and optimization
variants:

```bash
python scripts/experiments/current_level_persistence/run_shapelet_grid_search.py \
  --config configs/current_level_persistence/shapelet_grid_search_delta_0p5.yaml
```

The default grid is intentionally small:

```text
16 trials for multiscale_shapelet_mlp
16 trials for multiscale_shapelet_transformer
16 trials for multiscale_shapelet_convolution
```

The configurable parameters live in:

```text
configs/current_level_persistence/shapelet_grid_search_delta_0p5.yaml
```

The grid currently varies learning rate, number of shapelets per length,
dropout, and one model-specific capacity parameter per architecture. It uses
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

## Future Switch Derivation

Switch metrics are not implemented yet for this task. The planned conversion is:

```text
predicted_switch = 1 if predicted_remaining_persistence_seconds >= switch_time_seconds else 0
```

The conversion and operational switch metrics will be added after duration
models have been validated.
