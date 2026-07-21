# Experiment Execution Guide

This document is the operational runbook for the currently implemented
pipeline. Run every command from the repository root.

For the task-level methodological summary of the autoregressive branch, see
`docs/autoregressive.md`.

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

Mamba experiments are not implemented yet. The separate
`current_level_persistence` task has a task-scoped dataset builder and
learnable-shapelet/XGBoost runners; see `docs/current_level_persistence.md`
for its dataset and model commands.
The first-stage `long_fade_detection` binary task is documented in
`docs/long_fade_detection.md`.
The `survival_persistence` task currently includes the model-independent
dataset pipeline and the first XGBoost-AFT baseline/grid; see
`docs/survival_persistence.md`.

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

Run focused test groups when changing one area:

```bash
conda run -n Nowcasting python -m pytest tests/autoregressive -q
conda run -n Nowcasting python -m pytest tests/current_level_persistence -q
conda run -n Nowcasting python -m pytest tests/long_fade_detection -q
conda run -n Nowcasting python -m pytest tests/data -q
conda run -n Nowcasting python -m pytest tests/switching -q
conda run -n Nowcasting python -m pytest tests/tuning -q
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
conda run -n Nowcasting python scripts/analysis/autoregressive/09_prepare_event_windows.py
```

Main outputs:

```text
data/interim/clean_signal/
data/interim/candidate_events/
data/interim/event_windows/<threshold_folder>/
data/processed/autoregressive/<threshold_folder>/window_index_L<context>_h<horizon>.parquet
results/data_preparation/autoregressive/<threshold_folder>/
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
configs/autoregressive/autoregressive_dataset.yaml
```

This stage selects datasets and builds aligned `train`, `val`, and `test`
arrays for:

- `raw`;
- `context_standard`, fitted independently using each input context only.

The default selection mode is now:

```yaml
dataset_selection:
  mode: external_holdout
  heldout_test_dataset: fc-uplink-fade.csv
```

This means:

- `fc-uplink-fade.csv` is excluded from training, validation, early stopping,
  hyperparameter tuning, and final retraining;
- all other available datasets are development datasets;
- development events are split chronologically per dataset into train/val;
- datasets with fewer than three usable events are kept entirely in train;
- final `train.npz`, `val.npz`, and `test.npz` are global merged split files.

Run:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/10_build_autoregressive_datasets.py
```

Alternate dataset configurations can be passed explicitly:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/10_build_autoregressive_datasets.py \
  --config configs/autoregressive/autoregressive_dataset_L120_h10.yaml
```

If a previous or partial build already exists, rebuild intentionally with:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/10_build_autoregressive_datasets.py \
  --config configs/autoregressive/autoregressive_dataset.yaml \
  --force
```

The `L120_h10` dataset config expects the matching
`window_index_L120_h10.parquet` to exist under
`data/processed/autoregressive/threshold_10p0/`. It reuses the already
prepared event windows and creates a separate selection ID:
`externalHoldout_test_fc_uplink_fade_L120_h10_thr10`.

Main outputs:

```text
data/processed/autoregressive/<threshold_folder>/datasets_L<context>_h<horizon>/<selection_folder>/
results/data_preparation/autoregressive/<selection_id>/
results/index/datasets.csv
```

Current external-holdout output path:

```text
data/processed/autoregressive/threshold_10p0/datasets_L30_h10/externalHoldout_test_fc_uplink_fade/
results/data_preparation/autoregressive/externalHoldout_test_fc_uplink_fade_L30_h10_thr10/
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

Generated pilot datasets can be reviewed with a dry-run cleanup plan:

```bash
conda run -n Nowcasting python scripts/maintenance/clean_generated_datasets.py --dry-run
```

Only use `--execute` after checking the printed removal list. The script never
targets `data/raw`, source code, configs, tests, AGENTS.md, or reports.

## 5. Final Dataset Sanity Check

The sanity-check script validates shapes, metadata alignment, normalization
reconstruction, split integrity, NaNs, infinities, and event leakage.

Run:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/11_sanity_check_autoregressive_dataset.py \
  --no-plots
```

Pass a different dataset configuration explicitly when checking an alternate
context length:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/11_sanity_check_autoregressive_dataset.py \
  --config configs/autoregressive/autoregressive_dataset_L120_h10.yaml \
  --no-plots
```

If multiple selection folders exist under the configured dataset root, pass
`--selection-folder`. Use `--plots` when running interactively and you want
random raw/context-standard window plots.

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
conda run -n Nowcasting python scripts/analysis/autoregressive/12_compute_perfect_switch.py
```

For a non-default selection, use its dedicated config:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/12_compute_perfect_switch.py \
  --config configs/perfect_switch_L120_h10.yaml
```

Main outputs:

```text
results/switching/perfect_switch/autoregressive/<selection_id>/
results/index/switch_references.csv
```

`output.overwrite` defaults to `false`. Changing the Perfect Switch rule
changes the reference for every model comparison and must be treated as a
methodological change.

## 7. GRU Experiments

### 7.1 Single Configured Runs

Configuration:

```text
configs/autoregressive/autoregressive_gru.yaml
```

The configured architectures and variants are trained once each. Edit
`experiment.architectures` and `experiment.variants` to run a subset.

Run:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_autoregressive_gru.py
```

### 7.2 GRU Grid Search

Configuration:

```text
configs/autoregressive/gru_grid_search.yaml
```

Run:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_gru_grid_search.py
```

The extended default GRU grid has 72 trials for `gru_s2v` per variant and 216
trials for `gru_seq2seq` per variant because teacher forcing is expanded only
for the encoder-decoder architecture. Across the two variants, this is 576
training trials. Selection uses `validation_rmse_raw`.
With `final_training.retrain_on_full_development: true`, the selected
hyperparameters are retrained on `train+val` for the selected `best_epoch`
before evaluating the external `test.npz`.

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
configs/autoregressive/autoregressive_patchtst.yaml
```

Run:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_autoregressive_patchtst.py
```

To execute exactly one run, leave one value under `experiment.variants`.

### 8.2 PatchTST Grid Search

Configuration:

```text
configs/autoregressive/patchtst_grid_search.yaml
```

Run:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_patchtst_grid_search.py
```

The current extended PatchTST grid contains 864 trials per variant and is
capped by `search.max_trials: 864`. Across the two variants, this is 1,728
training trials.

The grid selects by raw-scale validation RMSE. With
`final_training.retrain_on_full_development: true`, the selected
hyperparameters are retrained on `train+val` for the validation-selected
`best_epoch`, then evaluated once on the external `test.npz`.

## 9. Chronos Zero-Shot Evaluation

Configuration:

```text
configs/autoregressive/autoregressive_chronos.yaml
```

Chronos evaluates only the final external-holdout test contexts. It does not
load train or validation data and does not train or fine-tune the pretrained
model.

Run:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_autoregressive_chronos_zero_shot.py
```

Longer-context Chronos zero-shot evaluation uses:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_autoregressive_chronos_zero_shot.py \
  --config configs/autoregressive/autoregressive_chronos_L120_h10.yaml
```

The default model is `amazon/chronos-t5-small`, evaluated on the raw variant.
The optional `context_standard_optional` variant can be added explicitly.
Deterministic metrics use the median of sampled Chronos trajectories; q10,
q50, and q90 are also saved.

The first run downloads the pretrained model from Hugging Face. No trained
checkpoint is written locally by this project.

## 10. XGBoost Autoregressive Baseline

Configuration:

```text
configs/autoregressive/xgboost_grid_search.yaml
```

Run:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/autoregressive/run_xgboost_grid_search.py \
  --config configs/autoregressive/xgboost_grid_search.yaml
```

This baseline flattens the univariate context window and trains one XGBoost
regressor for each forecast horizon. With `prediction_length: 10`, each trial
therefore fits 10 independent regressors.

The grid is configured with inclusive range specs:

```yaml
parameter_grid:
  max_depth:
    start: 2
    stop: 6
    step: 2
```

The current config evaluates 1,152 trials per variant for `raw` and
`context_standard`. Selection uses raw-scale validation RMSE. The selected
parameters are retrained on train+validation and evaluated once on the
external test split.

XGBoost can use the GPU only when the installed XGBoost build supports CUDA.
The default config uses `device: cpu` for portability; set `xgboost.device:
cuda` on the server only after verifying that the server package supports it.

## 11. Model And Grid-Search Outputs

Single runs and grid-search winners publish canonical artifacts under:

```text
models/autoregressive/<model_family>/<run_id>/
results/runs/autoregressive/<selection_id>/<run_id>/
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
results/grid_searches/autoregressive/<selection_id>/<search_id>/
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

Git tracks lightweight CSV/YAML summaries and
`results/runs/autoregressive/<selection_id>/<run_id>/predictions/test_predictions.parquet`, so
switch comparisons can be regenerated on another machine after a push. Figures,
checkpoints, validation histories, and most intermediate Parquet outputs remain
ignored.

## 12. Switch Comparisons

Configuration:

```text
configs/switch_comparison.yaml
```

Each configured method is compared independently against the Perfect Switch.
The configuration currently lists the four GRU architecture/variant runs, the
two PatchTST variants, and Chronos zero-shot raw. Add future model runs
explicitly when their canonical predictions exist.

The current operational conversion rule is configured as:

```yaml
conversion:
  prediction_aggregation: horizon_threshold_count
  required_points_above_threshold: 10
```

At each `input_end_time`, the model switch is activated only when at least the
configured number of future horizon predictions satisfies the threshold
condition. With `prediction_length: 10` and
`required_points_above_threshold: 10`, this means all 10 predicted future
points must be above the 10.0 signal threshold before minimum-island
post-processing is applied.

Run:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/13_compare_switch_methods.py
```

For the Chronos `L120_h10` variant:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/13_compare_switch_methods.py \
  --config configs/switch_comparison_chronos_L120_h10.yaml
```

Main outputs:

```text
results/comparisons/switch_eval/autoregressive/<selection_id>/switch_eval_<run_id>/
results/index/comparisons.csv
```

Each comparison folder contains metrics, aligned model-versus-reference
predictions, metadata, and event plots showing that model against Perfect
Switch. Methods are never merged into one switch-evaluation folder.

## 13. Cross-Model Result Tables

After model runs and switch comparisons exist, build the compact summary
tables:

```bash
conda run -n Nowcasting python scripts/analysis/autoregressive/14_compare_model_results.py
```

Main outputs:

```text
results/comparisons/model_summary/autoregressive/<selection_id>/model_result_summary_<selection_id>/tables/validation_forecast_comparison.csv
results/comparisons/model_summary/autoregressive/<selection_id>/model_result_summary_<selection_id>/tables/test_forecast_switch_comparison.csv
```

The validation table contains raw-scale validation MAE and RMSE when available.
Chronos zero-shot has no validation row metrics because it does not train or
load validation data. The test table contains raw-scale test MAE/RMSE plus the
model-vs-Perfect Switch metrics when the switch comparison has been generated.

## 14. Current-Level Persistence

Build the current-level persistence dataset:

```bash
conda run -n Nowcasting python scripts/experiments/current_level_persistence/build_dataset.py \
  --config configs/current_level_persistence/dataset_delta_0p5_L30_external_holdout.yaml
```

This builder applies the shared event-quality table: it excludes `unusable`
events and follows the configuration for warning events. Rebuild and retrain
the affected models whenever this quality policy changes.

The target search uses the complete cleaned signal, so event-window boundaries
do not truncate persistence durations. Context windows stay within continuous
acquisition segments. Candidates without an observed recovery before a true
segment boundary are saved to `censored_window_metadata.parquet` and excluded
from supervised regression; review `censoring_summary.csv` after each build.

Rebuild with `--force` when existing NPZ files predate
`scalar_context_features`. The runner can reconstruct these features from
`X_raw`, but a rebuild also persists feature values and names in NPZ/parquet
artifacts.

Train the three initial learnable-shapelet variants:

```bash
conda run -n Nowcasting python scripts/experiments/current_level_persistence/train_learnable_shapelets.py \
  --config configs/current_level_persistence/shapelet_mlp_delta_0p5.yaml

conda run -n Nowcasting python scripts/experiments/current_level_persistence/train_learnable_shapelets.py \
  --config configs/current_level_persistence/shapelet_transformer_delta_0p5.yaml

conda run -n Nowcasting python scripts/experiments/current_level_persistence/train_learnable_shapelets.py \
  --config configs/current_level_persistence/shapelet_convolution_delta_0p5.yaml
```

Run the small validation-only grid search for all three variants:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/current_level_persistence/run_shapelet_grid_search.py \
  --config configs/current_level_persistence/shapelet_grid_search_delta_0p5.yaml
```

The current grid has 216 trials for each of:

- `multiscale_shapelet_mlp`;
- `multiscale_shapelet_transformer`;
- `multiscale_shapelet_convolution`.

This gives 648 trials total. It varies learning rate, model width, model depth,
dropout, shapelets per length, and scalar-encoder width. The safety limit is
explicitly set to 256 trials per model. This campaign is intended for a
multi-GPU server and may take a long time.

The trial phase uses only train and validation data. The winner for each model
is selected with `val_mae_seconds`; only then is the selected model materialized
as a canonical run and evaluated on the test split.

Each run saves metrics, predictions, diagnostic figures, learned shapelets,
metadata, and checkpoints under the task-specific `results/runs/` and
`models/` folders.
The compact model-selection summary is mirrored under:

```text
results/comparisons/model_selection/current_level_persistence/<selection_id>/<search_id>/
```

The three configs enable scalar context. Run IDs contain `scalarContext`, and
the train-fitted scalar means/stds are stored in metadata and checkpoints.

Optional convenience launcher for running the three single-model shapelet
configs concurrently, assigning one visible CUDA device to each process:

```bash
PYTHONUNBUFFERED=1 bash \
  scripts/experiments/current_level_persistence/run_shapelet_scalar_context_parallel.sh
```

Override the visible device IDs when needed:

```bash
GPU_IDS="0 1 2" PYTHONUNBUFFERED=1 bash \
  scripts/experiments/current_level_persistence/run_shapelet_scalar_context_parallel.sh
```

Use this launcher inside `tmux` or another session manager for long runs. It is
only a shell convenience wrapper around `train_learnable_shapelets.py`; the
grid-search runner above already has its own device-aware scheduler.

Run the XGBoost duration baseline:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/current_level_persistence/run_xgboost_grid_search.py \
  --config configs/current_level_persistence/xgboost_grid_search_delta_0p5.yaml
```

This baseline uses flattened `X_relative_to_current` plus the configured
scalar-context features, predicts `log1p_remaining_persistence_seconds`, and
reports metrics in seconds/minutes. The grid uses range specs instead of fixed
manual lists and currently evaluates 1,152 validation trials.
Its compact model-selection summary uses the same
`results/comparisons/model_selection/current_level_persistence/` hierarchy.

After all current-level runs finish, build the comparison table:

```bash
conda run -n Nowcasting python scripts/experiments/current_level_persistence/summarize_runs.py
```

Main output:

```text
results/comparisons/model_summary/current_level_persistence/externalHoldout_test_fc_uplink_fade/current_level_model_comparison_delta0p5/tables/current_level_model_comparison_delta0p5.csv
```

Build the current-level Perfect Switch and evaluate duration-derived switch
decisions:

```bash
conda run -n Nowcasting python \
  scripts/analysis/current_level_persistence/12_compute_perfect_switch.py

conda run -n Nowcasting python \
  scripts/analysis/current_level_persistence/13_compare_switch_methods.py
```

The configured decision is positive when the current signal is above 10 and
predicted remaining persistence is at least 300 seconds. It then uses the same
10-sample minimum-island and stateful signal-above-threshold hold used by
autoregressive switch evaluation.

## 15. Long-Fade Detection

`long_fade_detection` is a binary grouped-event task. At each timestamp inside
a grouped fade event, it predicts whether that whole grouped event lasts at
least the configured minimum duration.

Default setup:

```text
threshold_db = 10.0
min_fade_duration_seconds = 300
sampling_time_seconds = 30
context_length = 30
external test dataset = fc-uplink-fade.csv
```

Events are grouped from threshold crossings using the 3-hour convention.
Temporary below-threshold samples between the first and last grouped crossing
remain inside the event. Every timestamp inside the same grouped event receives
the same `y_long_fade` label.

Build the dataset:

```bash
conda run -n Nowcasting python scripts/experiments/long_fade_detection/build_dataset.py \
  --config configs/long_fade_detection/xgboost_lag_scalar_threshold10_duration300.yaml
```

The dataset is saved under:

```text
data/processed/long_fade_detection/threshold_10p0/min_duration_300s/L30/externalHoldout_test_fc_uplink_fade/
```

Run the extended first-stage grid:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/long_fade_detection/run_grid_search.py \
  --config configs/long_fade_detection/grid_search_threshold10_duration300.yaml
```

The current extended grid has:

- `xgboost_lag_scalar_classifier`: 2,880 trials;
- `tcn_classifier`: 3,456 trials;
- `multiscale_shapelet_convolution_classifier`: 2,592 trials.

This is 8,928 validation trials total. The grid selects one winner per model
family using validation AUPRC, retrains each selected configuration on
train+validation, then evaluates the external test split once.

Parallel scheduling uses all visible CUDA GPUs when available. If CUDA is not
available, it falls back to the selected MPS/CPU device with one worker.

For server execution, run these inside `tmux`:

```bash
bash scripts/experiments/long_fade_detection/run_server_smoke.sh
bash scripts/experiments/long_fade_detection/run_server.sh
```

Implemented first-stage models:

- `xgboost_lag_scalar_classifier`: flattened threshold-relative lags plus
  train-standardized scalar context;
- `tcn_classifier`: temporal convolution over `X_relative_to_threshold` plus
  scalar context;
- `multiscale_shapelet_convolution_classifier`: learnable shapelet response
  maps plus scalar context.

Main outputs:

```text
results/runs/long_fade_detection/<run_id>/
models/long_fade_detection/<run_id>/
```

Build the initial comparison summary after runs finish:

```bash
conda run -n Nowcasting python scripts/experiments/long_fade_detection/summarize_runs.py
```

Summary outputs:

```text
results/comparisons/model_summary/long_fade_detection/<selection_id>/long_fade_model_summary_thr10p0_dur300/tables/long_fade_model_comparison.csv
results/comparisons/model_summary/long_fade_detection/<selection_id>/long_fade_model_summary_thr10p0_dur300/reports/long_fade_model_comparison.md
```

Build long-fade switch references and comparisons:

```bash
conda run -n Nowcasting python scripts/analysis/long_fade_detection/12_compute_perfect_switch.py \
  --config configs/long_fade_detection/perfect_switch.yaml

conda run -n Nowcasting python scripts/analysis/long_fade_detection/13_compare_switch_methods.py \
  --config configs/long_fade_detection/switch_comparison.yaml
```

Details are in [docs/long_fade_detection.md](docs/long_fade_detection.md).

## 16. Survival Persistence

Build and audit the model-independent survival dataset:

```bash
conda run -n Nowcasting env PYTHONPATH=. python scripts/experiments/survival_persistence/build_dataset.py \
  --config configs/survival_persistence/dataset_threshold10_L30_external_holdout.yaml \
  --force

conda run -n Nowcasting env PYTHONPATH=. python scripts/experiments/survival_persistence/audit_dataset.py \
  --config configs/survival_persistence/dataset_threshold10_L30_external_holdout.yaml
```

Run the first XGBoost-AFT baseline:

```bash
conda run -n Nowcasting env PYTHONPATH=. python scripts/experiments/survival_persistence/run_xgboost_aft.py \
  --config configs/survival_persistence/models/xgboost_aft_scalar_context.yaml
```

The baseline uses validation early stopping and evaluates the external test
split once. Outputs are under:

```text
results/runs/survival_persistence/<selection_id>/<run_id>/
models/survival_persistence/xgboost_aft/<run_id>/
```

Run the extended XGBoost-AFT grid search:

```bash
mkdir -p logs/survival_persistence

PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/experiments/survival_persistence/run_xgboost_aft_grid_search.py \
  --config configs/survival_persistence/models/xgboost_aft_grid_search.yaml \
  2>&1 | tee logs/survival_persistence/xgboost_aft_grid_search.log
```

Grid outputs:

```text
results/grid_searches/survival_persistence/<selection_id>/<search_id>/
results/comparisons/model_selection/survival_persistence/<selection_id>/<search_id>/
results/runs/survival_persistence/<selection_id>/<best_run_id>/
models/survival_persistence/xgboost_aft/<best_run_id>/
```

In `tmux`, start a session and detach after launching:

```bash
tmux new -s survival_aft_grid
mkdir -p logs/survival_persistence
PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/experiments/survival_persistence/run_xgboost_aft_grid_search.py \
  --config configs/survival_persistence/models/xgboost_aft_grid_search.yaml \
  2>&1 | tee logs/survival_persistence/xgboost_aft_grid_search.log
```

Detach with `Ctrl-b`, then `d`.

Before launching the full grid on a new server, run a small smoke test. This
runs only six trials, writes them under a separate smoke search folder, and
does not finalize or overwrite the canonical best run:

```bash
mkdir -p logs/survival_persistence

PYTHONUNBUFFERED=1 PYTHONPATH=. python scripts/experiments/survival_persistence/run_xgboost_aft_grid_search.py \
  --config configs/survival_persistence/models/xgboost_aft_grid_search.yaml \
  --max-trials 6 \
  --skip-finalize \
  --search-suffix smoke \
  2>&1 | tee logs/survival_persistence/xgboost_aft_grid_search_smoke.log
```

Build survival Perfect Switch and switch metrics from saved predictions:

```bash
conda run -n Nowcasting env PYTHONPATH=. python scripts/analysis/survival_persistence/12_compute_perfect_switch.py \
  --config configs/survival_persistence/perfect_switch.yaml

conda run -n Nowcasting env PYTHONPATH=. python scripts/analysis/survival_persistence/13_compare_switch_methods.py \
  --config configs/survival_persistence/switch_comparison.yaml
```

The current conversion rule is:

```text
model_switch_raw(t) = 1 if S(300s | X_t) >= 0.5 and Signal(t) > 10.0
```

The shared switch post-processing is then applied. Event plots use a display-only
90-minute-before/90-minute-after window, while metrics remain computed on native
survival decision timestamps.

## 17. Central Result Indexes

Use these files to locate generated artifacts without scanning every folder:

```text
results/index/datasets.csv
results/index/runs.csv
results/index/grid_searches.csv
results/index/switch_references.csv
results/index/comparisons.csv
results/index/survival_persistence_runs.csv
```

## 18. Recommended Complete Campaign

For a new raw-data or methodological configuration:

1. Run tests.
2. Run relevant exploratory analyses `00-08`.
3. Review and run `09_prepare_event_windows.py`.
4. Review and run `10_build_autoregressive_datasets.py`.
5. Run `11_sanity_check_autoregressive_dataset.py`.
6. Run `12_compute_perfect_switch.py`.
7. Run single model experiments for quick validation.
8. Review grid size, GPU availability, and overwrite settings.
9. Run GRU, PatchTST, XGBoost, and/or Chronos zero-shot evaluation.
10. Add completed canonical runs to `configs/switch_comparison.yaml`.
11. Run `13_compare_switch_methods.py`.
12. Run `14_compare_model_results.py`.
13. Inspect forecast metrics, switch metrics, event plots, summary tables, and
    central indexes.
14. For `long_fade_detection`, build its dataset separately, then run the
    three first-stage classifiers and the summary script.
15. For `survival_persistence`, build and audit the dataset, run XGBoost-AFT,
    then build Perfect Switch and survival-probability switch comparisons.

## 19. Configuration Consistency Checklist

Before running downstream stages, verify that these values refer to the same
prepared dataset:

- signal threshold;
- threshold folder;
- context length;
- prediction length;
- dataset root;
- selection folder or selection ID.

Never use the test split for hyperparameter selection, early stopping,
threshold calibration, or model choice. Autoregressive grids select using
validation RMSE; current-level persistence grids select using the configured
validation duration metric. Long-fade detection neural runs use validation
AUPRC for early stopping and model selection. Test metrics are final
diagnostics for selected winners only.
