# Satellite Attenuation Nowcasting

This repository contains the advanced modelling phase of a satellite
attenuation nowcasting project. The current implementation priority is the
autoregressive forecasting branch and the shared preprocessing and evaluation
infrastructure.

The final objective is to derive reliable operational switch decisions from
information available up to the prediction timestamp. Forecast losses are
diagnostic; switch-level and event-level metrics are the primary evaluation.

## Current Status

The repository currently provides an initial, dependency-light foundation:

- strict signal-only defaults using `Signal` as both target and model input;
- CSV loading with header, no-header, and automatic header-detection modes;
- normalization of every loaded dataset to exactly `Time` and `Signal`;
- reusable raw-data quality, sampling, distribution, window-availability, and
  candidate-event analysis helpers;
- six interactive raw-data analysis scripts with generated tables and figures;
- configuration templates with unresolved empirical assumptions left explicit;
- chronological split helpers;
- no-leakage autoregressive window-index construction;
- strict `Time` and `Signal` schema validation;
- forecast and sample-level switch metrics;
- centralized device selection and reproducibility utilities;
- `pytest` unit tests.

Model architectures, reference-switch post-processing, event definitions, and
dataset-specific loading are intentionally deferred until their assumptions are
specified.

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

When dependencies change, export the active environment without its local
prefix:

```bash
conda env export --no-builds | grep -v "^prefix:" > environment.yml
```

## Repository Layout

- `configs/`: data and autoregressive experiment configuration templates.
- `data/`: raw, interim, and processed data. Raw data must never be modified.
- `scripts/analysis/`: interactive Python analysis scripts using `# %%` cells.
- `scripts/experiments/`: executable autoregressive experiments.
- `src/`: reusable analysis, data, dataset, model, evaluation, and utility code.
- `models/<task>/<model>/`: generated model artifacts without run-ID folders.
- `results/tables/data_analysis/`: shared raw-data analysis tables; exploratory
  figures are displayed interactively and are not saved automatically.
- `results/<task>/<model>/`: model-specific figures, tables, predictions, and
  reports.
- `results/<task>/comparisons/`: cross-model outputs for one task.
- `reports/`: methodological notes and experiment logs.
- `tests/`: unit tests.

See `AGENTS.md` for the full methodological and engineering policy.
