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
python scripts/experiments/build_current_level_persistence_dataset.py \
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
features before the final regression layer.

## Train Learnable-Shapelet Models

Use the top-level wrapper when running from the repository root:

```bash
python scripts/experiments/run_shapelet_current_level_persistence.py \
  --config configs/current_level_persistence/shapelet_mlp_delta_0p5.yaml
```

The other implemented variants use:

```bash
python scripts/experiments/run_shapelet_current_level_persistence.py \
  --config configs/current_level_persistence/shapelet_transformer_delta_0p5.yaml

python scripts/experiments/run_shapelet_current_level_persistence.py \
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

For remote multi-GPU execution commands, see
[`current_level_persistence_server_runs.md`](current_level_persistence_server_runs.md).

## Future Switch Derivation

Switch metrics are not implemented yet for this task. The planned conversion is:

```text
predicted_switch = 1 if predicted_remaining_persistence_seconds >= switch_time_seconds else 0
```

The conversion and operational switch metrics will be added after duration
models have been validated.
