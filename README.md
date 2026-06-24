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

```text
[S_{t+1}, ..., S_{t+h}]
```

from a past context:

```text
[S_{t-L+1}, ..., S_t]
```

Implemented models:

- GRU sequence-to-vector;
- GRU encoder-decoder;
- PatchTST-style Transformer;
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

The `current_level_persistence` task predicts a scalar duration: how long the
signal remains above a recovery level defined relative to the current signal.

Current setup:

```text
delta = 0.5
context_length = 30
target = log1p_remaining_persistence_seconds
input = X_relative_to_current
```

Implemented models:

- multiscale learnable-shapelet MLP;
- multiscale learnable-shapelet Transformer;
- multiscale learnable-shapelet convolutional head.

Main documentation:

- [Current-Level Persistence](docs/current_level_persistence.md)

### Deferred Work

The following branches are intentionally not finalized yet:

- Mamba autoregressive experiments;
- probabilistic survival modelling;
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

For long commands where live progress matters:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting python SCRIPT_PATH
```

Device selection is centralized. PyTorch code uses CUDA when available, then
Apple MPS when available, otherwise CPU. GRU and PatchTST grid searches can
schedule independent trials across multiple visible CUDA GPUs.

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

Then run model experiments or grid searches:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_gru_grid_search.py

PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_patchtst_grid_search.py

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
conda run -n Nowcasting python scripts/experiments/build_current_level_persistence_dataset.py \
  --config configs/current_level_persistence/dataset_delta_0p5_L30_external_holdout.yaml
```

Train the three learnable-shapelet variants:

```bash
conda run -n Nowcasting python scripts/experiments/run_shapelet_current_level_persistence.py \
  --config configs/current_level_persistence/shapelet_mlp_delta_0p5.yaml

conda run -n Nowcasting python scripts/experiments/run_shapelet_current_level_persistence.py \
  --config configs/current_level_persistence/shapelet_transformer_delta_0p5.yaml

conda run -n Nowcasting python scripts/experiments/run_shapelet_current_level_persistence.py \
  --config configs/current_level_persistence/shapelet_convolution_delta_0p5.yaml
```

Details are in
[docs/current_level_persistence.md](docs/current_level_persistence.md).

## Configuration Layout

```text
configs/
├── autoregressive/
│   ├── autoregressive_dataset*.yaml
│   ├── autoregressive_gru.yaml
│   ├── autoregressive_patchtst.yaml
│   ├── autoregressive_chronos*.yaml
│   ├── gru_grid_search.yaml
│   └── patchtst_grid_search.yaml
├── current_level_persistence/
│   ├── dataset_delta_0p5_L30_external_holdout.yaml
│   ├── shapelet_mlp_delta_0p5.yaml
│   ├── shapelet_transformer_delta_0p5.yaml
│   └── shapelet_convolution_delta_0p5.yaml
├── data.yaml
├── data_preparation.yaml
├── perfect_switch*.yaml
└── switch_comparison*.yaml
```

Task-specific model and dataset configs live under their task folder. Shared
preparation, Perfect Switch, and switch-comparison configs remain top-level.

## Artifact Layout

Datasets:

```text
data/processed/autoregressive/
data/processed/current_level_persistence/
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
```

Compatibility wrappers may exist at older script paths, but new code should
use the task-scoped modules and directories.

## Reproducibility Rules

- Use chronological, event-aware splits.
- Fit model-selection decisions on validation data only.
- Keep external test data for final evaluation only.
- Never tune switch thresholds or conversion rules on test data.
- Do not modify files under `data/raw/`.
- Do not use non-signal covariates.
- Record run metadata, config fingerprints, selected device, metrics, and
  artifact paths.

See [AGENTS.md](AGENTS.md) for the complete engineering and methodological
policy.
