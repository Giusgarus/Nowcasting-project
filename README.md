# Satellite Attenuation Nowcasting

This repository contains the advanced modelling phase of a satellite
attenuation nowcasting project.

The practical objective is operational: use only information available up to
the current timestamp to decide whether a satellite link is likely to require a
backup switch. Forecast and duration losses are useful diagnostics, but the
main evaluation is the quality of the resulting switch behaviour against a
model-independent reference.

## Implemented Scope

### 1. Autoregressive Forecasting

The autoregressive task predicts a future `Signal` trajectory:

$$
\left[S_{t+1}, \ldots, S_{t+h}\right]
$$

from a past context:

$$
\left[S_{t-L+1}, \ldots, S_t\right]
$$

Implemented models:

- GRU sequence-to-vector;
- GRU encoder-decoder;
- PatchTST-style Transformer;
- XGBoost tabular baseline, with one regressor per forecast horizon;
- Chronos zero-shot.

Implemented evaluation:

- raw-scale forecast MAE/RMSE;
- horizon-wise forecast metrics;
- Perfect Switch reference construction;
- model-derived switch conversion;
- model-vs-Perfect-Switch metrics;
- cross-model validation/test summary tables.

Main documentation:

- [Autoregressive Forecasting](docs/autoregressive.md)
- [Experiment Execution Guide](EXPERIMENT_GUIDE.md)

### 2. Current-Level Persistence

The `current_level_persistence` task estimates the residual persistence of the
current signal level. At timestamp `t`, the model observes only the past and
present context:

$$
X_t = \left[S_{t-L+1}, \ldots, S_t\right]
$$

For a configured positive margin `delta`, expressed in the same units as
`Signal`, the persistence reference level is:

$$
r_t(\delta) = S_t - \delta
$$

The first recovery timestamp is the first future timestamp in the same
continuous acquisition segment whose signal falls below this reference level:

$$
u_t(\delta) = \min \{u > t : S_u < r_t(\delta)\}
$$

The uncensored regression target is the corresponding elapsed duration:

$$
T_t(\delta) = elapseds\_seconds(t, u_t(\delta))
$$

Thus, `delta` defines the required decrease below the current value before the
current-level persistence interval is considered ended. With the current
default, `delta = 0.5`, the signal is treated as persistent while it remains at
or above `S_t - 0.5`.

The shapelet input is expressed relative to the current value, with the
subtraction applied elementwise:

$$
X_t^{rel} = X_t - S_t
$$

Current setup:

```text
delta = 0.5
context_length = 30
target = log1p_remaining_persistence_seconds
shapelet input = X_relative_to_current
auxiliary input = scalar_context_features derived from X_raw
```

Duration targets search the full cleaned signal beyond event-window edges and
use actual elapsed timestamps. Contexts never cross acquisition gaps; samples
whose recovery is not observed before a continuous segment ends are marked as
right-censored and excluded from duration regression.

Recoverable small gaps can be interpolated before duration target search using
the shared conservative policy: expected 30-second sampling, maximum 90-second
gap, maximum two inserted samples, and no interpolation across acquisition
segments.

The 12 scalar features preserve absolute level and recent summary information.
Their standardization is fitted on train only. Scalar-context runs use distinct
IDs, so the earlier shapelet-only results are not overwritten.

Implemented models:

- multiscale learnable-shapelet MLP;
- multiscale learnable-shapelet Transformer;
- multiscale learnable-shapelet convolutional head;
- XGBoost tabular duration baseline.

Main documentation:

- [Current-Level Persistence](docs/current_level_persistence.md)

Current-level duration predictions are converted into switch decisions with a
configured 300-second persistence threshold and the shared autoregressive
post-processing pipeline.

### 3. Long-Fade Detection

The `long_fade_detection` task is a binary grouped-event classification
problem. For a configured fade threshold $\theta$, define threshold-crossing
timestamps as:

$$
C_\theta = \{t : S_t \geq \theta\}
$$

A grouped fade event starts at the first threshold crossing. Further crossings
within the grouping horizon $G$ from that first crossing are assigned to the
same event; the event span includes all timestamps from the first to the last
crossing in the group, including temporary below-threshold samples.

For grouped event $E_k$, let its duration be:

$$
D_k = |E_k| \cdot \Delta t
$$

where $\Delta t$ is the sampling time. At each timestamp $t$ inside $E_k$, the
binary target is:

$$
y_t = \mathbb{1}\{D_k \geq d_{\min}\}
$$

All timestamps inside the same grouped event therefore share the same label.
The model observes only information available up to timestamp $t$, typically
the threshold-relative context:

$$
X_t^{\theta} = \left[S_{t-L+1} - \theta, \ldots, S_t - \theta\right]
$$

Current defaults are $\theta = 10.0$, $d_{\min} = 300$ seconds, $G = 3$ hours,
$\Delta t = 30$ seconds, and $L = 30$.

The implemented dataset builder uses the shared prepared event-window signal
(`Signal_prepared`) stitched into one time-ordered series per dataset, rather
than reading the long-fade inputs directly from raw CSV files.

Implemented first-stage models:

- XGBoost lag+scalar classifier;
- TCN classifier;
- multiscale shapelet-convolution classifier.

Main documentation:

- [Long-Fade Detection](docs/long_fade_detection.md)

### 4. Survival Persistence

The `survival_persistence` task builds a model-independent continuous-time
survival dataset for the remaining duration of the current grouped fade
episode:

$$
S(u | X_t) = P(R_t > u | X_t)
$$

where `R_t` is the remaining time until the beginning of stable recovery. The
event definition uses a state machine with configurable activation and recovery
thresholds, stable-recovery confirmation, and right-censoring at continuous
segment ends.

Implemented scope:

- dataset builder, XGBoost-AFT baseline, and discrete-time TCN smoke baseline;
- XGBoost-AFT validation grid search;
- observed and right-censored continuous-time labels;
- train-only scalar-context standardization;
- event-balanced sample weights;
- reproducible dataset-audit diagnostics;
- survival-probability-to-switch evaluation against Perfect Switch.

DeepHit survival models are not implemented yet.

Main documentation:

- [Survival Persistence](docs/survival_persistence.md)

## Switch Decision Rules

All final switch metrics compare a model-derived binary switch against the
task-scoped Perfect Switch reference. Each task first builds a raw decision at
its own decision timestamp, then applies the shared post-processing:

```text
model_switch_raw
  -> min-island extension to switch_time samples
  -> hold switch active while the observed Signal remains above threshold
  -> model_switch
```

The legacy adjusted switch columns are still saved for diagnostics, but the
reported model-vs-Perfect metrics use the post-processed `model_switch` and
`perfect_switch` columns.

Current configured rules:

- **Autoregressive forecasting**:
  `model_switch_raw(t) = 1` when at least
  `required_points_above_threshold` predicted horizon values exceed the signal
  threshold. In the canonical `L30_h10_thr10` setup this means all 10 predicted
  future points must be greater than `10.0`.
- **Current-level persistence**:
  `model_switch_raw(t) = 1` when the predicted remaining persistence duration
  is at least `300` seconds and, when configured, the current observed signal is
  greater than `10.0`.
- **Long-fade detection**:
  `model_switch_raw(t) = 1` when `P(long_fade | X_t) >= 0.5` and, when
  configured, the current observed signal is greater than `10.0`.
- **Survival persistence**:
  `model_switch_raw(t) = 1` when the predicted survival probability at the
  300-second horizon satisfies `S(300s | X_t) >= 0.5` and, when configured, the
  current observed signal is greater than `10.0`.

Switch metrics are computed on each task's native decision timestamps.
Display-only event plots for long-fade and survival may be widened to a
90-minute-before/90-minute-after view, with missing task decisions shown as
zero outside the native decision segment. That plotting extension does not
change the metric denominator.

### Deferred Work

The following branches are intentionally not finalized yet:

- Mamba autoregressive experiments;
- additional survival model families beyond XGBoost-AFT and discrete-time TCN;
- the broader shapelet-pattern operational branch beyond the implemented
  current-level persistence task;
- Smart/baseline switch integration.

Do not add detailed assumptions for these branches unless they are explicitly
specified.

## Data Policy

The repository is signal-only.

All loaded datasets are standardized to exactly:

```text
Time
Signal
```

Rules:

- first source column is `Time` and second source column is `Signal` for
  no-header files;
- extra source columns are ignored explicitly;
- models use only `Signal`;
- raw files under `data/raw/` must never be modified;
- attenuation construction is disabled unless explicitly configured.

## Environment

The conda environment is named:

```text
Nowcasting
```

Create it:

```bash
conda env create -f environment.yml
```

Synchronize it:

```bash
conda env update -n Nowcasting -f environment.yml --prune
```

Run the test suite:

```bash
conda run -n Nowcasting python -m pytest -q
```

Task or area-specific tests can be run from the corresponding folders:

```bash
conda run -n Nowcasting python -m pytest tests/autoregressive -q
conda run -n Nowcasting python -m pytest tests/current_level_persistence -q
conda run -n Nowcasting python -m pytest tests/long_fade_detection -q
conda run -n Nowcasting python -m pytest tests/data -q
conda run -n Nowcasting python -m pytest tests/switching -q
conda run -n Nowcasting python -m pytest tests/tuning -q
```

For long commands where live progress matters:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting python SCRIPT_PATH
```

Device selection is centralized. PyTorch code uses CUDA when available, then
Apple MPS when available, otherwise CPU. GRU, PatchTST, and current-level
shapelet grid searches can schedule independent trials across multiple visible
CUDA GPUs.

## Core Workflows

### Raw-Data Analysis

Exploratory scripts:

```text
scripts/analysis/00_data_inventory.py
scripts/analysis/01_single_dataset_inspection.py
scripts/analysis/02_sampling_and_gaps.py
scripts/analysis/03_signal_distribution.py
scripts/analysis/04_window_availability.py
scripts/analysis/05_candidate_event_exploration.py
scripts/analysis/06_baseline_drift_and_daily_patterns.py
scripts/analysis/07_signal_change_and_outlier_analysis.py
scripts/analysis/08_normalization_diagnostics.py
```

Shared raw-data analysis tables are saved under:

```text
results/data_analysis/
```

Exploratory figures are displayed interactively and are not saved
automatically.

### Autoregressive Pipeline

Canonical order:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/09_prepare_event_windows.py
conda run -n Nowcasting python scripts/analysis/autoregressive/10_build_autoregressive_datasets.py
conda run -n Nowcasting python scripts/analysis/autoregressive/11_sanity_check_autoregressive_dataset.py --no-plots
conda run -n Nowcasting python scripts/analysis/autoregressive/12_compute_perfect_switch.py
```

Use `--force` with `10_build_autoregressive_datasets.py` only when you want to
replace an existing or partial dataset build.

Then run model experiments or grid searches:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_gru_grid_search.py

PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_patchtst_grid_search.py

PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_xgboost_grid_search.py \
  --config configs/autoregressive/xgboost_grid_search.yaml

PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_autoregressive_chronos_zero_shot.py
```

Build switch comparisons and model summaries:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/13_compare_switch_methods.py
conda run -n Nowcasting python scripts/analysis/autoregressive/14_compare_model_results.py
```

Details are in [docs/autoregressive.md](docs/autoregressive.md).

### Current-Level Persistence Pipeline

Build the dataset:

```bash
conda run -n Nowcasting python scripts/experiments/current_level_persistence/build_dataset.py \
  --config configs/current_level_persistence/dataset_delta_0p5_L30_external_holdout.yaml
```

Current-level persistence reuses the shared event-quality assessment. It does
not create windows from events marked `unusable`; warning-event inclusion is
explicitly configured.

Train the three learnable-shapelet variants:

```bash
conda run -n Nowcasting python scripts/experiments/current_level_persistence/train_learnable_shapelets.py \
  --config configs/current_level_persistence/shapelet_mlp_delta_0p5.yaml

conda run -n Nowcasting python scripts/experiments/current_level_persistence/train_learnable_shapelets.py \
  --config configs/current_level_persistence/shapelet_transformer_delta_0p5.yaml

conda run -n Nowcasting python scripts/experiments/current_level_persistence/train_learnable_shapelets.py \
  --config configs/current_level_persistence/shapelet_convolution_delta_0p5.yaml
```

Or run the small validation-only grid search for all three variants:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/current_level_persistence/run_shapelet_grid_search.py \
  --config configs/current_level_persistence/shapelet_grid_search_delta_0p5.yaml
```

The grid currently evaluates 216 trials per model (648 total), selects winners using
validation MAE in seconds, then evaluates the selected winner for each model on
the test split.

The current-level XGBoost grid uses scalar-context tabular features and selects
one duration regressor with validation MAE in seconds:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/current_level_persistence/run_xgboost_grid_search.py \
  --config configs/current_level_persistence/xgboost_grid_search_delta_0p5.yaml
```

Details are in
[docs/current_level_persistence.md](docs/current_level_persistence.md).

### Long-Fade Detection Pipeline

Build the first-stage binary long-fade dataset:

```bash
conda run -n Nowcasting python scripts/experiments/long_fade_detection/build_dataset.py \
  --config configs/long_fade_detection/xgboost_lag_scalar_threshold10_duration300.yaml
```

Run the extended first-stage grid:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/long_fade_detection/run_grid_search.py \
  --config configs/long_fade_detection/grid_search_threshold10_duration300.yaml
```

Summarize completed long-fade runs:

```bash
conda run -n Nowcasting python scripts/experiments/long_fade_detection/summarize_runs.py
```

Build task-scoped Perfect Switch and switch comparisons:

```bash
conda run -n Nowcasting python scripts/analysis/long_fade_detection/12_compute_perfect_switch.py
conda run -n Nowcasting python scripts/analysis/long_fade_detection/13_compare_switch_methods.py
```

Details are in [docs/long_fade_detection.md](docs/long_fade_detection.md).

### Survival Persistence Pipeline

Build and audit the survival dataset:

```bash
conda run -n Nowcasting env PYTHONPATH=. python scripts/experiments/survival_persistence/build_dataset.py \
  --config configs/survival_persistence/dataset_threshold10_L30_external_holdout.yaml \
  --force

conda run -n Nowcasting env PYTHONPATH=. python scripts/experiments/survival_persistence/audit_dataset.py \
  --config configs/survival_persistence/dataset_threshold10_L30_external_holdout.yaml
```

Run the XGBoost-AFT grid search:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/experiments/survival_persistence/run_xgboost_aft_grid_search.py \
  --config configs/survival_persistence/models/xgboost_aft_grid_search.yaml
```

Run the discrete-time TCN smoke baseline:

```bash
conda run -n Nowcasting env PYTHONPATH=. python scripts/experiments/survival_persistence/run_discrete_time_tcn.py \
  --config configs/survival_persistence/models/discrete_time_tcn_smoke.yaml
```

Run the controlled validation-only TCN selection:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/experiments/survival_persistence/run_discrete_time_tcn_controlled.py \
  --config configs/survival_persistence/models/discrete_time_tcn_controlled.yaml \
  --stage all
```

Run the extended discrete-time TCN grid:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/experiments/survival_persistence/run_discrete_time_tcn_grid_search.py \
  --config configs/survival_persistence/models/discrete_time_tcn_grid_search.yaml
```

The grid currently expands to 1620 validation trials using range specs for the
numeric hyperparameters. It keeps the controlled winner's input representation,
weighting, and binning fixed, then evaluates the external test split only for
the selected checkpoint.

The clean XGBoost-AFT versus TCN grid-best survival comparison is saved under:

```text
results/comparisons/model_selection/survival_persistence/<selection_id>/survivalPersistence_model_comparison_<selection_id>/tables/model_comparison_xgboost_tcn.csv
```

Build Perfect Switch and switch metrics:

```bash
conda run -n Nowcasting env PYTHONPATH=. python scripts/analysis/survival_persistence/12_compute_perfect_switch.py \
  --config configs/survival_persistence/perfect_switch.yaml

conda run -n Nowcasting env PYTHONPATH=. python scripts/analysis/survival_persistence/13_compare_switch_methods.py \
  --config configs/survival_persistence/switch_comparison.yaml
```

Details are in [docs/survival_persistence.md](docs/survival_persistence.md).

## Configuration Layout

```text
configs/
├── autoregressive/
│   ├── autoregressive_dataset*.yaml
│   ├── autoregressive_gru.yaml
│   ├── autoregressive_patchtst.yaml
│   ├── autoregressive_chronos*.yaml
│   ├── gru_grid_search.yaml
│   ├── patchtst_grid_search.yaml
│   └── xgboost_grid_search.yaml
├── current_level_persistence/
│   ├── dataset_delta_0p5_L30_external_holdout.yaml
│   ├── perfect_switch.yaml
│   ├── shapelet_mlp_delta_0p5.yaml
│   ├── shapelet_transformer_delta_0p5.yaml
│   ├── shapelet_convolution_delta_0p5.yaml
│   ├── shapelet_grid_search_delta_0p5.yaml
│   ├── switch_comparison.yaml
│   └── xgboost_grid_search_delta_0p5.yaml
├── long_fade_detection/
│   ├── perfect_switch.yaml
│   ├── xgboost_lag_scalar_threshold10_duration300.yaml
│   ├── tcn_threshold10_duration300.yaml
│   ├── shapelet_conv_threshold10_duration300.yaml
│   ├── switch_comparison.yaml
│   └── grid_search_threshold10_duration300.yaml
├── survival_persistence/
│   ├── dataset_threshold10_L30_external_holdout.yaml
│   ├── perfect_switch.yaml
│   ├── switch_comparison.yaml
│   └── models/
│       ├── xgboost_aft_grid_search.yaml
│       └── xgboost_aft_scalar_context.yaml
├── data.yaml
├── data_preparation.yaml
├── perfect_switch*.yaml
└── switch_comparison*.yaml
```

Task-specific model and dataset configs live under their task folder. Shared
autoregressive preparation, Perfect Switch, and switch-comparison configs remain
top-level for the autoregressive branch.

## Artifact Layout

Datasets:

```text
data/processed/autoregressive/
data/processed/current_level_persistence/
data/processed/long_fade_detection/
data/processed/survival_persistence/
```

Model checkpoints:

```text
models/<task>/<model_family_or_model_id>/<run_id>/
```

Model outputs:

```text
results/runs/<task>/<selection_id>/<run_id>/
```

Perfect Switch references:

```text
results/switching/perfect_switch/<task>/<selection_id>/
```

Comparisons:

```text
results/comparisons/<comparison_type>/<task>/<selection_id>/<comparison_id>/
```

Grid searches:

```text
results/grid_searches/<task>/<selection_id>/<search_id>/
```

Central indexes:

```text
results/index/datasets.csv
results/index/runs.csv
results/index/grid_searches.csv
results/index/switch_references.csv
results/index/comparisons.csv
results/index/current_level_persistence_runs.csv
results/index/survival_persistence_runs.csv
```

The repository tracks lightweight CSV/YAML summaries and selected prediction
files needed to regenerate comparisons. Heavy generated artifacts such as
figures, trial checkpoints, validation histories, and most intermediate
Parquet files remain ignored.

## Repository Layout

```text
configs/        Configuration files
data/           Raw, interim, and processed datasets
docs/           Task-specific methodological notes
scripts/        Analysis, preparation, training, and maintenance entrypoints
src/            Reusable package code
models/         Trained checkpoints and model metadata
results/        Generated tables, predictions, comparisons, and indexes
reports/        Research notes and experiment logs
tests/          Pytest suite
```

Canonical task code lives under:

```text
src/tasks/autoregressive/
src/tasks/current_level_persistence/
src/tasks/long_fade_detection/
```

New code and documentation should use the task-scoped modules and directories.
Compatibility namespaces under `src/datasets/` and
`src/evaluation/forecast_metrics.py` exist only to keep older imports working.

The test suite is organized by area:

```text
tests/autoregressive/
tests/current_level_persistence/
tests/long_fade_detection/
tests/data/
tests/evaluation/
tests/switching/
tests/tuning/
tests/utils/
```

## Reproducibility Rules

- Use chronological, event-aware splits.
- Fit model-selection decisions on validation data only.
- Keep external test data for final evaluation only.
- Never tune switch thresholds or conversion rules on test data.
- Do not modify files under `data/raw/`.
- Do not use non-signal covariates.
- Record run metadata, config fingerprints, selected device, metrics, and
  artifact paths.
