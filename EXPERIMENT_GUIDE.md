# Experiment Execution Guide

This document is the operational runbook for the currently implemented
pipeline. Run every command from the repository root.

The complete implemented workflow is:

```text
raw data review
    -> event-window preparation
    -> final autoregressive datasets
    -> dataset sanity checks
    -> Perfect Switch reference
    -> model training or grid search
    -> model-versus-Perfect-Switch comparisons
```

Chronos, Mamba, shapelet, and survival experiments are not implemented yet.

## 1. Environment And Verification

The conda environment is named `Nowcasting`.

Create it on a new machine:

```bash
conda env create -f environment.yml
```

Synchronize an existing environment:

```bash
conda env update -n Nowcasting -f environment.yml --prune
```

Run the complete test suite before a new experimental campaign:

```bash
conda run -n Nowcasting python -m pytest -q
```

For commands where live progress is useful, use:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting python SCRIPT_PATH
```

Device behavior:

- single model runs use the best device selected by PyTorch;
- grid searches distribute independent trials across up to three available
  CUDA GPUs;
- grid searches fall back to fewer GPUs or one CPU/MPS worker automatically.

## 2. Raw-Data Analysis

Scripts `00` through `08` are exploratory. They do not modify `data/raw/` and
are not required every time models are trained. Run them after adding or
changing raw datasets, or when reviewing methodological assumptions.

Tables are saved under `results/data_analysis/`. Exploratory figures are
displayed interactively and are not saved automatically.

| Script | Purpose | Main saved output |
|---|---|---|
| `00_data_inventory.py` | Loadability, schema, invalid values, and dataset quality | `data_inventory_summary.csv` |
| `01_single_dataset_inspection.py` | Detailed interactive inspection of one selected dataset | None |
| `02_sampling_and_gaps.py` | Sampling intervals, temporal gaps, and continuous segments | `sampling_and_gaps_summary.csv` |
| `03_signal_distribution.py` | Signal distributions and cross-dataset comparison | `signal_distribution_summary.csv` |
| `04_window_availability.py` | Approximate valid-window counts for candidate lengths | `window_availability_summary.csv` |
| `05_candidate_event_exploration.py` | Exploratory quantile-based extreme runs | `candidate_event_quantile_summary.csv` |
| `06_baseline_drift_and_daily_patterns.py` | Rolling baseline, drift, hourly, and daily patterns | Three baseline/calendar tables |
| `07_signal_change_and_outlier_analysis.py` | Signal changes and exploratory spike candidates | Signal-change and spike tables |
| `08_normalization_diagnostics.py` | Diagnostic comparison of normalization methods | `normalization_diagnostics_summary.csv` |

Run the analyses individually:

```bash
conda run -n Nowcasting python scripts/analysis/00_data_inventory.py
conda run -n Nowcasting python scripts/analysis/01_single_dataset_inspection.py
conda run -n Nowcasting python scripts/analysis/02_sampling_and_gaps.py
conda run -n Nowcasting python scripts/analysis/03_signal_distribution.py
conda run -n Nowcasting python scripts/analysis/04_window_availability.py
conda run -n Nowcasting python scripts/analysis/05_candidate_event_exploration.py
conda run -n Nowcasting python scripts/analysis/06_baseline_drift_and_daily_patterns.py
conda run -n Nowcasting python scripts/analysis/07_signal_change_and_outlier_analysis.py
conda run -n Nowcasting python scripts/analysis/08_normalization_diagnostics.py
```

Before running `01`, set `DATASET_PATH` inside the script to the dataset to
inspect. Scripts `04`, `06`, `07`, and `08` also expose editable exploratory
constants near the top of each file.

## 3. Event Windows And Window Index

Configuration:

```text
configs/data_preparation.yaml
```

Important settings include:

- candidate-event threshold and grouping;
- event-window duration;
- quality and imputation rules;
- `context_length` and `prediction_length`;
- chronological event-level split ratios.

Run:

```bash
conda run -n Nowcasting python scripts/analysis/09_prepare_event_windows.py
```

Main outputs:

```text
data/interim/clean_signal/
data/interim/candidate_events/
data/interim/event_windows/<threshold_folder>/
data/processed/autoregressive/<threshold_folder>/window_index_L<context>_h<horizon>.parquet
results/data_preparation/<threshold_folder>/
```

This stage creates event-centered data and the official no-leakage window
index. It does not create the final model-ready NPZ datasets.

The script protects existing threshold-specific outputs from incompatible
silent overwrites. When changing threshold, context length, prediction length,
or preparation assumptions, keep all downstream configuration paths
consistent.

## 4. Final Autoregressive Datasets

Configuration:

```text
configs/autoregressive_dataset.yaml
```

This stage selects datasets and builds aligned `train`, `val`, and `test`
arrays for:

- `raw`;
- `context_standard`, fitted independently using each input context only.

Run:

```bash
conda run -n Nowcasting python scripts/analysis/10_build_autoregressive_datasets.py
```

Main outputs:

```text
data/processed/autoregressive/<threshold_folder>/datasets_L<context>_h<horizon>/<selection_folder>/
results/data_preparation/<selection_id>/
results/index/datasets.csv
```

The final dataset folder contains:

```text
train.npz
val.npz
test.npz
train_metadata.parquet
val_metadata.parquet
test_metadata.parquet
dataset_metadata.yaml
```

`output.overwrite` defaults to `false`. Set it to `true` only when
intentionally rebuilding the same dataset selection.

## 5. Final Dataset Sanity Check

The sanity-check script validates shapes, metadata alignment, normalization
reconstruction, split integrity, NaNs, infinities, and event leakage.

Run:

```bash
conda run -n Nowcasting python scripts/analysis/11_sanity_check_autoregressive_dataset.py
```

If multiple selection folders exist, set `SELECTION_FOLDER` near the top of
the script. Set `SHOW_PLOTS = false` for a non-interactive machine.

Do not start a model campaign unless this script finishes with:

```text
All sanity checks passed.
```

## 6. Perfect Switch Reference

Configuration:

```text
configs/perfect_switch.yaml
```

The Perfect Switch is model-independent and uses only the selected test events.
It may be computed before or after model training, but it must exist before
switch comparisons.

Run:

```bash
conda run -n Nowcasting python scripts/analysis/12_compute_perfect_switch.py
```

Main outputs:

```text
results/switching/perfect_switch/<selection_id>/
results/index/switch_references.csv
```

`output.overwrite` defaults to `false`. Changing the Perfect Switch rule
changes the reference for every model comparison and must be treated as a
methodological change.

## 7. GRU Experiments

### 7.1 Single Configured Runs

Configuration:

```text
configs/autoregressive_gru.yaml
```

The configured architectures and variants are trained once each. Edit
`experiment.architectures` and `experiment.variants` to run a subset.

Run:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/run_autoregressive_gru.py
```

### 7.2 GRU Grid Search

Configuration:

```text
configs/gru_grid_search.yaml
```

Run:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/run_gru_grid_search.py
```

The grid search:

- tunes every configured architecture and variant independently;
- tunes `teacher_forcing_ratio` only for `gru_seq2seq`;
- selects the best trial using raw-scale validation RMSE;
- evaluates only the selected trial on test;
- saves the selected canonical model when `update_canonical_runs: true`.

## 8. PatchTST Experiments

### 8.1 Single Configured Runs

Configuration:

```text
configs/autoregressive_patchtst.yaml
```

Run:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/run_autoregressive_patchtst.py
```

To execute exactly one run, leave one value under `experiment.variants`.

### 8.2 PatchTST Grid Search

Configuration:

```text
configs/patchtst_grid_search.yaml
```

Run:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/run_patchtst_grid_search.py
```

The current complete PatchTST grid contains 2,592 trials per variant.
`search.max_trials: 200` intentionally blocks that full search. Reduce the
grid or explicitly increase the limit before running it.

The grid selects by raw-scale validation RMSE and evaluates test only after
selection.

## 9. Model And Grid-Search Outputs

Single runs and grid-search winners publish canonical artifacts under:

```text
models/<model_family>/<run_id>/
results/runs/<run_id>/
```

Typical canonical model artifacts:

```text
best_model.pt
model_config.yaml
training_metadata.yaml
```

PatchTST runs also save `last_model.pt`. The canonical `best_model.pt` is the
checkpoint to use for final predictions and comparisons.

Grid-search details are saved separately under:

```text
results/grid_searches/<search_id>/
```

They include trial checkpoints, rankings, validation histories, metadata, and
`best_params.yaml`.

Current grid configurations use:

```yaml
output:
  overwrite: true
  update_canonical_runs: true
```

Therefore, rerunning a grid search can replace its previous search outputs and
canonical winner. Review these settings before every long campaign.

Single runs and grid-search winners intentionally target the same canonical
run IDs for an equivalent model, variant, and dataset selection. A single-run
configuration uses `overwrite: false` by default and will stop instead of
silently replacing an existing grid-search winner.

## 10. Switch Comparisons

Configuration:

```text
configs/switch_comparison.yaml
```

Each configured method is compared independently against the Perfect Switch.
The configuration currently lists the four GRU architecture/variant runs.
Add PatchTST or future model runs explicitly when their canonical predictions
exist.

Run:

```bash
conda run -n Nowcasting python scripts/analysis/13_compare_switch_methods.py
```

Main outputs:

```text
results/comparisons/switch_eval_<run_id>/
results/index/comparisons.csv
```

Each comparison folder contains metrics, aligned model-versus-reference
predictions, metadata, and event plots showing that model against Perfect
Switch. Methods are never merged into one switch-evaluation folder.

## 11. Central Result Indexes

Use these files to locate generated artifacts without scanning every folder:

```text
results/index/datasets.csv
results/index/runs.csv
results/index/grid_searches.csv
results/index/switch_references.csv
results/index/comparisons.csv
```

## 12. Recommended Complete Campaign

For a new raw-data or methodological configuration:

1. Run tests.
2. Run relevant exploratory analyses `00-08`.
3. Review and run `09_prepare_event_windows.py`.
4. Review and run `10_build_autoregressive_datasets.py`.
5. Run `11_sanity_check_autoregressive_dataset.py`.
6. Run `12_compute_perfect_switch.py`.
7. Run single model experiments for quick validation.
8. Review grid size, GPU availability, and overwrite settings.
9. Run GRU and/or PatchTST grid searches.
10. Add completed canonical runs to `configs/switch_comparison.yaml`.
11. Run `13_compare_switch_methods.py`.
12. Inspect forecast metrics, switch metrics, event plots, and central indexes.

## 13. Configuration Consistency Checklist

Before running downstream stages, verify that these values refer to the same
prepared dataset:

- signal threshold;
- threshold folder;
- context length;
- prediction length;
- dataset root;
- selection folder or selection ID.

Never use the test split for hyperparameter selection, early stopping,
threshold calibration, or model choice. Grid searches select using validation
RMSE only; test metrics are final diagnostics for the selected winner.
