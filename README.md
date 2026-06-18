# Satellite Attenuation Nowcasting

This repository contains the advanced modelling phase of a satellite
attenuation nowcasting project. The current implementation priority is the
autoregressive forecasting branch and the shared preprocessing and evaluation
infrastructure.

The final objective is to derive reliable operational switch decisions from
information available up to the prediction timestamp. Forecast losses are
diagnostic; switch-level and event-level metrics are the primary evaluation.

## Current Status

The repository currently provides an implemented autoregressive workflow:

- strict signal-only defaults using `Signal` as both target and model input;
- CSV loading with header, no-header, and automatic header-detection modes;
- normalization of every loaded dataset to exactly `Time` and `Signal`;
- reusable raw-data quality, sampling, distribution, window-availability, and
  candidate-event analysis helpers;
- interactive raw-data analyses for inventory, sampling, distributions,
  baseline drift, calendar patterns, spikes, and normalization diagnostics;
- configurable candidate-event preparation with clean Parquet data,
  event-centered windows, quality/imputation tracking, event-level splits, and
  traceable autoregressive window indices;
- configurable final supervised autoregressive datasets with modular dataset
  selection, external holdout evaluation, raw arrays, and context-only
  standardization;
- deterministic GRU sequence-to-vector and encoder-decoder forecasting
  baselines for raw and context-standard dataset variants;
- a deterministic PatchTST-style Transformer with validation-only
  hyperparameter search for raw and context-standard variants;
- pretrained Chronos T5 zero-shot evaluation on final autoregressive test
  contexts, with median and quantile forecasts;
- independent model-versus-Perfect-Switch comparisons for completed GRU,
  PatchTST, and Chronos runs;
- cross-model summary tables for validation forecast metrics and test
  forecast-plus-switch metrics;
- configuration templates with unresolved empirical assumptions left explicit;
- chronological split helpers;
- no-leakage autoregressive window-index construction;
- strict `Time` and `Signal` schema validation;
- forecast and sample-level switch metrics;
- centralized device selection and reproducibility utilities;
- `pytest` unit tests.

Shapelet/survival architectures and final operational event definitions remain
deferred until their assumptions are specified. Candidate events currently
serve only to select data-preparation windows.

All models are univariate and use only `Signal`. Input files may contain
headers or omit them; no-header files use the first column as `Time` and the
second as `Signal`. Extra columns are explicitly ignored. Attenuation
construction is disabled by default and requires an explicitly configured
reference method.

The default loader accepts ISO-8601 timestamps and the source formats currently
observed in the raw datasets: `d/m/Y H:M:S` and `d-Mon-Y H:M:S`. Other formats
require an explicit parser.

## Environment

The existing conda environment is named `Nowcasting`. Recreate or update it
from the only environment specification file:

```bash
conda env create -f environment.yml
conda env update -n Nowcasting -f environment.yml --prune
```

Run project commands in that environment:

```bash
conda run -n Nowcasting python -m pytest
```

Run the GRU grid search with live, unbuffered trial output:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/run_gru_grid_search.py
```

Before every trial, the script prints its full effective model, optimizer, and
training parameters. Shared grid parameters apply to both architectures;
`teacher_forcing_ratio` is expanded only for `gru_seq2seq`.

Run one PatchTST configuration for each variant using
`configs/autoregressive_patchtst.yaml`:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/run_autoregressive_patchtst.py
```

To execute exactly one run, leave only one value under
`experiment.variants`, for example `raw`.

Run the PatchTST grid search using `configs/patchtst_grid_search.yaml` after
building the external-holdout dataset:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/run_patchtst_grid_search.py
```

The full PatchTST grid is printed before the safety check. Trial checkpoints
are reusable, selection uses raw-scale validation RMSE, and only the selected
trial is evaluated on the external test dataset. The current extended
configuration expands to 864 trials per variant.

Run Chronos zero-shot evaluation on the external-holdout test split:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n Nowcasting \
  python scripts/experiments/run_autoregressive_chronos_zero_shot.py
```

After model predictions exist, build model-versus-Perfect-Switch comparisons
and the compact cross-model summary tables:

```bash
conda run -n Nowcasting python scripts/analysis/13_compare_switch_methods.py
conda run -n Nowcasting python scripts/analysis/14_compare_model_results.py
```

The current switch-conversion rule is decision-time based: at each
`input_end_time`, the model switch turns on only if all 10 saved future
predictions are above the configured 10.0 threshold, then the minimum-island
post-processing is applied. The required number of above-threshold horizon
points is configurable in `configs/switch_comparison.yaml`.

The default supervised dataset setup is an external-holdout protocol:

- `fc-uplink-fade.csv` is used only for final external test evaluation;
- every other available dataset is development data;
- development events are split chronologically per dataset into train and
  validation;
- development datasets with fewer than three usable events are kept entirely in
  train;
- grid search selects only on validation RMSE;
- enabled GRU/PatchTST grid runs retrain the selected hyperparameters on
  train+validation for the validation-selected `best_epoch` before external
  test evaluation.

GRU and PatchTST grid searches schedule independent trials across the CUDA GPUs
available at launch time. Their `parallel.max_workers: 3` setting uses up to
three GPUs, but automatically falls back to fewer GPU workers or one CPU/MPS
worker when necessary. `cuda_device_ids: auto` means all GPUs visible to
PyTorch; set an explicit list such as `[0, 2]`, or restrict
`CUDA_VISIBLE_DEVICES`, when a visible GPU is reserved or occupied by another
job.

When dependencies change, export the active environment without its local
prefix:

```bash
conda env export --no-builds | grep -v "^prefix:" > environment.yml
```

## Repository Layout

- `configs/`: data preparation and autoregressive experiment configurations.
- `data/`: raw, interim, and processed data. Raw data must never be modified.
- `scripts/analysis/`: interactive Python analysis scripts using `# %%` cells.
- `scripts/experiments/`: executable autoregressive experiments.
- `src/`: reusable analysis, data, dataset, model, evaluation, and utility code.
- `models/<model_family>/<run_id>/`: trained checkpoints and model metadata.
- `results/data_analysis/`: shared raw-data analysis tables; exploratory
  figures are displayed interactively and are not saved automatically.
- `results/runs/<run_id>/`: one shallow folder for each model run.
- `results/switching/perfect_switch/<selection_id>/`: Perfect Switch reference.
- `results/comparisons/<comparison_id>/`: model and baseline comparisons.
  Each switch-evaluation folder compares exactly one run with one reference.
- `results/grid_searches/<search_id>/`: validation-only hyperparameter-search
  trials and selected parameters.
- `results/data_preparation/<selection_id>/`: human-readable dataset summaries.
- `results/index/`: central CSV indexes for datasets, runs, references,
  comparisons, and grid searches.
- `reports/`: methodological notes and experiment logs.
- `tests/`: unit tests.

See `AGENTS.md` for the full methodological and engineering policy.

See [`EXPERIMENT_GUIDE.md`](EXPERIMENT_GUIDE.md) for the ordered commands and
configuration workflow from raw-data analysis through model and switch
evaluation.
