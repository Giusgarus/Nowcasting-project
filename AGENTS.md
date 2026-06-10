# AGENTS.md

## Project overview

This repository implements the **advanced modelling phase** of a satellite attenuation nowcasting project.

The previous repository was used for an exploratory task-selection phase. In that phase, several alternative task formulations were tested using simple diagnostic models such as Ridge/logistic regression and XGBoost. The goal of the previous phase was not to select a final production model, but to identify which task formulations were empirically promising.

This new repository focuses on the selected task formulations:

1. **Autoregressive forecasting**
2. **Shapelet-pattern modelling**
3. **Probabilistic survival modelling**

At the current stage, the autoregressive branch is the most explicitly defined advanced-modelling branch. Candidate architectures for the shapelet and survival branches will be specified later. Until then, do not create detailed model assumptions, scripts, or architecture-specific files for those branches unless explicitly requested.

The practical problem remains an early operational decision problem. Satellite terminal operators may need to switch to a backup communication link when a fade event is expected to persist long enough to justify the switch. The system should therefore use the information available up to the current time to estimate the future evolution, persistence, or residual duration distribution of the ongoing attenuation event.

The project should emphasize:

- reproducibility;
- clean Python implementation;
- transparent modelling assumptions;
- no future leakage;
- careful empirical validation;
- comparison against the existing engineering baseline and reference switch rules;
- operationally meaningful switch metrics;
- interpretability of model outputs when possible.

---

## Main research objective

Develop an advanced Python-based nowcasting framework for satellite downlink fade events.

The objective is not to test many unrelated model families. The objective is to implement stronger task-specific pipelines for the selected formulations and evaluate whether they lead to reliable operational switch decisions.

The final evaluation should focus primarily on the quality of the derived switch decisions, not only on intermediate forecast, duration, or survival losses.

---

## Dataset policy

This repository uses a strict signal-only policy.

All loaded datasets must be normalized to exactly two columns:

```text
Time
Signal
```

Most available datasets contain only timestamp and signal values. Many files
do not contain headers. For no-header files, the loader must assume:

```text
first column  = Time
second column = Signal
```

If a dataset contains additional columns, they must be explicitly ignored.
The project must not use any additional input variables.

All models must be treated as univariate time-series models using only the
`Signal` column as input. The target is also `Signal`.

The common loading function must:

- read delimited files with or without headers;
- standardize loaded data to `Time` and `Signal`;
- parse `Time`;
- sort by `Time`;
- keep only `Time` and `Signal`;
- raise a clear error if fewer than two columns are available;
- explicitly report ignored extra columns;
- never use extra columns as model inputs.

If attenuation is constructed from `Signal` for downstream switch logic,
construction must be explicitly enabled and its reference method must be
configured and documented. Do not silently construct attenuation using an
undocumented baseline.

---

## Selected task formulations

### 1. Autoregressive forecasting

The autoregressive task predicts the future `Signal` trajectory.

At prediction time `t`, the input is a past signal window:

```text
X_t = [S_{t-L+1}, ..., S_t]
```

The target is a future signal trajectory:

```text
Y_t = [S_{t+1}, S_{t+2}, ..., S_{t+h}]
```

The predicted future signal trajectory is converted into a binary switch
decision using a common switch-conversion rule only after that rule has been
explicitly configured.

Candidate model families for this task:

- LSTM encoder-decoder;
- Transformer-based forecasting model, preferably PatchTST or an equivalent time-series Transformer;
- Chronos zero-shot;
- Chronos fine-tuning, if technically feasible;
- Mamba-based forecasting model in zero-shot mode, if a stable implementation is available;
- Mamba-based fine-tuning, only as a second-stage experiment if zero-shot results and implementation stability justify it.

The autoregressive branch must report both:

- forecast metrics, such as MAE and RMSE;
- switch metrics derived from the predicted future trajectory.

Forecast metrics are diagnostic. Switch metrics remain the main operational evaluation.

---

### 2. Shapelet-pattern modelling

The shapelet task models local temporal patterns in the recent `Signal`
history.

This task has been selected for the advanced phase, but the advanced architecture and final modelling strategy have not yet been fixed.

Do not introduce detailed shapelet-specific model assumptions, architectures, or scripts unless explicitly requested.

For now, only preserve the high-level task idea:

- input: recent `Signal` window;
- objective: exploit local temporal shape and pattern information;
- final evaluation: common switch-evaluation pipeline.

---

### 3. Probabilistic survival modelling

The survival task estimates a distribution over the residual duration of the ongoing attenuation event.

At time `t`, let `R_t` be the remaining duration of the event. The model should eventually estimate a survival curve:

```text
S(u | X_t) = P(R_t > u | X_t)
```

for a discrete or continuous set of future horizons.

This task has been selected for the advanced phase, but the final architecture and implementation strategy have not yet been fixed.

Do not introduce detailed survival-specific model assumptions, architectures, or scripts unless explicitly requested.

For now, only preserve the high-level task idea:

- input: past `Signal` window;
- objective: estimate residual-duration uncertainty or persistence probabilities;
- final evaluation: common switch-evaluation pipeline.

---

## Expected project structure

Use the following folder structure.

```text
Nowcasting-advanced/
│
├── AGENTS.md
├── README.md
├── environment.yml
├── .gitignore
│
├── configs/
│   ├── data.yaml
│   ├── autoregressive_lstm.yaml
│   ├── autoregressive_patchtst.yaml
│   ├── autoregressive_chronos.yaml
│   └── autoregressive_mamba.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── scripts/
│   ├── analysis/
│   │   ├── 00_data_inventory.py
│   │   ├── 01_single_dataset_inspection.py
│   │   ├── 02_sampling_and_gaps.py
│   │   ├── 03_signal_distribution.py
│   │   ├── 04_window_availability.py
│   │   └── 05_candidate_event_exploration.py
│   │
│   └── experiments/
│       ├── run_autoregressive_lstm.py
│       ├── run_autoregressive_patchtst.py
│       ├── run_autoregressive_chronos.py
│       └── run_autoregressive_mamba.py
│
├── src/
│   ├── analysis/
│   │   ├── data_quality.py
│   │   └── plots.py
│   │
│   ├── data/
│   │   ├── loading.py
│   │   ├── preprocessing.py
│   │   ├── reference_switch.py
│   │   ├── events.py
│   │   ├── splits.py
│   │   └── validation.py
│   │
│   ├── datasets/
│   │   └── windowed_forecasting.py
│   │
│   ├── models/
│   │   └── autoregressive/
│   │       ├── lstm.py
│   │       ├── patchtst.py
│   │       ├── chronos_adapter.py
│   │       └── mamba_adapter.py
│   │
│   ├── evaluation/
│   │   ├── forecast_metrics.py
│   │   ├── switch_metrics.py
│   │   ├── event_window_metrics.py
│   │   └── plots.py
│   │
│   └── utils/
│       ├── config.py
│       ├── device.py
│       ├── paths.py
│       ├── logging.py
│       └── reproducibility.py
│
├── models/
│   ├── autoregressive/
│   │   ├── lstm/
│   │   ├── patchtst/
│   │   ├── chronos/
│   │   └── mamba/
│   ├── shapelet/
│   └── survival/
│
├── results/
│   ├── tables/
│   │   └── data_analysis/
│   ├── autoregressive/
│   │   ├── comparisons/
│   │   │   ├── figures/
│   │   │   ├── tables/
│   │   │   └── reports/
│   │   ├── lstm/
│   │   │   ├── figures/
│   │   │   ├── tables/
│   │   │   ├── predictions/
│   │   │   └── reports/
│   │   ├── patchtst/
│   │   ├── chronos/
│   │   └── mamba/
│   ├── shapelet/
│   │   └── comparisons/
│   └── survival/
│       └── comparisons/
│
├── reports/
│
└── tests/
```

Do not create all files immediately unless explicitly requested. The structure describes the intended organization of the repository.

Shapelet and survival configs, scripts, model modules, and model-specific
artifact folders should be added later, once their advanced modelling
strategies are explicitly defined. Their top-level `models/` and `results/`
task folders may exist before those strategies are fixed.

The `patchtst`, `chronos`, and `mamba` result folders use the same `figures/`,
`tables/`, `predictions/`, and `reports/` subfolders shown for `lstm`.
Each task-level `comparisons/` folder uses `figures/`, `tables/`, and
`reports/`.

---

## Environment management

This project uses an existing Anaconda/conda environment.

The only dependency and environment specification file is:

- `environment.yml`

Do not create or use:

- `pyproject.toml`;
- `requirements.txt`;
- `requirements-lock.txt`;
- `environment-lock.yml`;
- `uv.lock`.

Do not create a new virtual environment unless explicitly requested.

Use the active conda environment when running Python, tests, analysis scripts, experiments, or any other project command.

When dependencies change, update `environment.yml` by exporting the active conda environment.

### Export the active environment

On macOS/Linux:

```bash
conda env export --no-builds | grep -v "^prefix:" > environment.yml
```

On Windows PowerShell:

```powershell
conda env export --no-builds | Select-String -NotMatch "^prefix:" | Set-Content environment.yml
```

### Recreate the environment from `environment.yml`

Use this only when setting up the project on a new machine or recreating the environment from scratch:

```bash
conda env create -f environment.yml
```

### Update an existing environment from `environment.yml`

Use this when `environment.yml` has changed and the existing conda environment must be synchronized:

```bash
conda env update -f environment.yml --prune
```

### Activate the environment

The environment name should be read from `environment.yml`.

Example:

```bash
conda activate ENVIRONMENT_NAME
```

Do not assume a specific environment name unless it is explicitly defined in `environment.yml`.

---

## Hardware and device policy

Use hardware acceleration when available.

The code should automatically select the best available device in the following order:

1. CUDA GPU, when available;
2. Apple Metal/MPS device, when running on macOS with Apple Silicon and MPS support available;
3. CPU, as a fallback.

For PyTorch code, use a centralized device utility, for example:

```text
src/utils/device.py
```

The device-selection logic should be reused by all neural or deep learning scripts.

Expected behavior:

```text
if CUDA is available:
    use cuda
elif Apple MPS is available:
    use mps
else:
    use cpu
```

Do not hard-code a device inside model scripts.

Do not assume that CUDA is available.

Do not assume that Apple Metal/MPS is available.

All experiments must remain runnable on CPU, even if slower.

When saving experiment metadata, record the selected device.

If a model or external library does not support MPS or CUDA, document the limitation and fall back safely.

For foundation models, GPU acceleration should be used when available, but the code must still provide a clear error message or fallback path if the model is too large for the available hardware.

---

## Interactive analysis policy

Exploratory analysis should preferably be written as Python scripts with cell markers instead of Jupyter notebooks.

Use files such as:

```text
scripts/analysis/00_data_inventory.py
scripts/analysis/01_single_dataset_inspection.py
scripts/analysis/02_sampling_and_gaps.py
```

These scripts may use cell markers:

```python
# %%
```

so that they can be executed interactively in VS Code or another IDE.

Rules:

- Codex or other coding agents should edit `.py` files, not `.ipynb` files;
- Jupyter notebooks are optional and should not be required for the project to run;
- core reusable logic must live in `src/`, not inside analysis scripts;
- analysis scripts should call functions from `src/`;
- shared raw-data analysis tables should be saved under
  `results/tables/data_analysis/`;
- exploratory raw-data analysis figures should be displayed interactively and
  should not be saved automatically;
- figures and tables produced by analysis scripts should be saved under the
  relevant `results/<task>/<model>/figures/` and
  `results/<task>/<model>/tables/` folders, or under
  `results/<task>/comparisons/` for cross-model analysis when they concern a
  specific modelling task.

If notebooks are later added, they must remain lightweight and should only call reusable functions from `src/`.

---

## Common preprocessing rules

All selected task formulations must share the same common preprocessing layer.

The common preprocessing must include:

- data loading;
- timestamp parsing;
- sorting by time;
- duplicate handling;
- missing-value handling;
- strict `Time` and `Signal` schema validation;
- sampling-time checks;
- explicit signal selection;
- optional attenuation construction only when enabled with a configured
  reference method;
- outage-mask construction;
- Perfect Switch construction;
- MATLAB-style switch post-processing;
- chronological train/validation/test split;
- event-window definition.

Do not implement separate versions of these steps inside model scripts.

The common preprocessing should produce a clean time-indexed dataset that can be reused by downstream dataset builders.

---

## No-leakage rules

Strictly avoid future leakage.

Rules:

- inputs for time `t` may only use `Signal` information available up to time
  `t`;
- target construction may use future values, but target values must never enter
  the input window;
- train/validation/test splits must be chronological;
- do not shuffle time series samples before splitting;
- scalers, normalizers, imputers, or transformations with fitted parameters must be fitted only on the training split;
- validation and test data must remain unseen during training;
- the test set must never be used for model selection, hyperparameter tuning, threshold calibration, early stopping, or checkpoint selection.

These rules apply to all task formulations and all models, including foundation models and fine-tuning experiments.

---

## Dataset construction principles

The repository should separate common preprocessing from task-specific dataset construction.

### Windowed forecasting dataset

Used by:

- LSTM;
- PatchTST;
- possibly Mamba-based models if the selected implementation supports the
  required univariate input format.

Expected structure:

```text
X_seq.shape = (n_samples, context_length, 1)
Y.shape     = (n_samples, prediction_length)
```

The input window should contain only past and present `Signal` observations.

The target should contain only the future `Signal` trajectory. The model input
dimension is always one.

### Chronos dataset adapter

Chronos is expected to require a model-specific adapter.

Chronos must use univariate `Signal` histories by default:

```text
[Signal_{t-L+1}, ..., Signal_t]
```

The adapter should not modify the common split or reference-switch logic.

### Mamba dataset adapter

Mamba-based forecasting models may require different input formats depending on the selected implementation.

The Mamba adapter must be isolated from the common preprocessing code.

If the implementation is unstable or incompatible with the data format, document the limitation rather than forcing changes to the common dataset.

### Shapelet dataset

The shapelet branch has been selected but is not fully specified yet.

Do not implement a shapelet dataset builder unless explicitly requested.

When implemented later, it must reuse the common preprocessing, reference switch, event-window, and split logic.

### Survival dataset

The survival branch has been selected but is not fully specified yet.

Do not implement a survival dataset builder unless explicitly requested.

When implemented later, it must reuse the common preprocessing, reference switch, event-window, and split logic.

---

## Reference switch policy

All final models must be evaluated against the same reference switch construction.

The reference switch logic should be centralized in:

```text
src/data/reference_switch.py
```

Do not duplicate reference-switch construction inside model-specific scripts.

The reference policy should include:

- outage mask from an explicitly configured signal threshold;
- optional attenuation-based logic only if attenuation construction and its
  threshold are explicitly configured;
- Perfect Switch construction;
- MATLAB-style minimum switch-time post-processing;
- Simple Switch, if used;
- Smart/Baseline Switch alignment, if the engineering baseline is available.

Any change to the reference switch policy must be explicitly documented because it affects all reported metrics.

---

## Switch-conversion policy

Some models do not directly output a binary switch.

For example:

- autoregressive models output a future trajectory;
- survival models may later output persistence probabilities;
- shapelet models may later output duration or event-persistence scores.

The conversion from model output to binary switch must be explicit, reproducible, and centralized when possible.

Examples:

```text
trajectory -> switch
survival curve -> switch
duration estimate -> switch
score/probability -> switch
```

Switch-conversion thresholds must be selected on the validation set only.

The test set must never be used to choose switch thresholds or decision rules.

---

## Evaluation policy

All final models must be evaluated using the same metric definitions.

The main operational metrics are:

- active duration;
- precision versus Perfect Switch;
- recall versus Perfect Switch;
- F1 versus Perfect Switch;
- IoU versus Perfect Switch;
- balanced accuracy;
- false positive rate;
- false negative rate;
- detected events;
- missed events;
- false switch events;
- onset delay, when available;
- event-window metrics.

Forecast, duration, and survival metrics are diagnostic and should not replace switch-level evaluation.

### Autoregressive metrics

Report:

- forecast MAE;
- forecast RMSE;
- optional horizon-wise forecast metrics;
- switch metrics after trajectory-to-switch conversion;
- event-window switch metrics.

### Shapelet metrics

The final advanced shapelet metrics will be specified when the shapelet modelling strategy is fixed.

For now, any shapelet result must still be evaluated through the common switch-evaluation pipeline.

### Survival metrics

The final advanced survival metrics will be specified when the survival modelling strategy is fixed.

For now, any survival result must still be evaluated through the common switch-evaluation pipeline.

---

## Baseline comparison

Always compare advanced models against appropriate references.

The baseline/reference set may include:

- Perfect Switch;
- Simple Switch;
- Smart/Baseline Switch or engineering baseline, when available;
- persistence-style forecast baseline, when useful;
- simple diagnostic models from the previous exploratory repository, when useful as historical context.

Do not reintroduce all exploratory task formulations from the old repository unless explicitly requested.

The old repository should be treated as the task-selection phase, not as the structure to replicate completely.

---

## Model persistence

Trained models and experiment artifacts should be saved under `models/`.

At the current stage, the main model folder is:

```text
models/autoregressive/<model>/
```

Additional folders such as:

```text
models/shapelet/
models/survival/
```

may exist as empty task roots, but model-specific subfolders should be added
only when the corresponding modelling strategies are fixed.

Each saved model should include enough metadata to make the experiment reproducible.

When saving a trained model, also save when appropriate:

- model configuration;
- task formulation;
- input column, which must be `Signal`;
- target definition;
- context length;
- prediction horizon;
- switch threshold;
- train/validation/test split information;
- selected device;
- random seed;
- evaluation metrics;
- creation timestamp.

For neural models, save PyTorch checkpoints and a separate metadata file.

For scikit-learn-style models, prefer `joblib`.

Do not commit large binary model files unless explicitly requested.

The code should make it possible to regenerate saved models from data and configuration.

---

## Folder conventions

### `data/raw/`

Contains original input data.

Rules:

- never modify files in this folder;
- never overwrite raw data;
- document the source and meaning of each file when data becomes available.

### `data/interim/`

Contains intermediate cleaned or transformed data.

Use this folder for:

- parsed files;
- resampled time series;
- intermediate event tables;
- temporary but reproducible outputs.

### `data/processed/`

Contains final datasets used for modelling.

Use this folder for:

- processed time-indexed datasets;
- train/validation/test split metadata;
- model-ready arrays or tables;
- event-level datasets.

### `scripts/analysis/`

Contains interactive `.py` analysis files.

Rules:

- use `# %%` cell markers when helpful;
- analysis scripts may be run interactively;
- analysis scripts should not contain core reusable logic;
- reusable code must be moved into `src/`.

### `scripts/experiments/`

Contains executable experiment scripts.

At the current stage, experiment scripts should focus on the autoregressive branch.

Examples:

```text
run_autoregressive_lstm.py
run_autoregressive_patchtst.py
run_autoregressive_chronos.py
run_autoregressive_mamba.py
```

Do not add shapelet or survival experiment scripts until their modelling strategies are explicitly defined.

### `src/`

Contains reusable Python code.

Expected modules include:

- shared data-analysis summaries and plots;
- data loading;
- common preprocessing;
- reference switch construction;
- event detection;
- temporal splitting;
- autoregressive dataset builders;
- autoregressive model definitions;
- model adapters;
- evaluation metrics;
- plotting;
- utilities.

Do not place experiment-specific logic directly in reusable modules unless it is clearly general.

### `tests/`

Contains unit tests using `pytest`.

Expected tests may cover:

- data loading;
- preprocessing;
- reference switch construction;
- event detection;
- windowed dataset construction;
- switch conversion;
- metric computation;
- edge cases.

### `models/`

Contains trained models and experiment artifacts.

At the current stage, use:

```text
models/autoregressive/lstm/
models/autoregressive/patchtst/
models/autoregressive/chronos/
models/autoregressive/mamba/
```

Use the hierarchy `models/<task>/<model>/`. Do not add experiment-ID or run-ID
subfolders. Shapelet and survival task roots may exist, but their model folders
must not be added until their strategies are defined.

### `results/`

Contains generated outputs.

Rules:

- save shared raw-data analysis tables under `results/tables/data_analysis/`;
- display exploratory raw-data figures interactively without saving them
  automatically;
- organize model-specific outputs under `results/<task>/<model>/`;
- save final tables in `results/<task>/<model>/tables/`;
- save plots in `results/<task>/<model>/figures/`;
- save model predictions in `results/<task>/<model>/predictions/`;
- save generated result summaries in `results/<task>/<model>/reports/`;
- save cross-model tables, plots, and summaries under
  `results/<task>/comparisons/`;
- do not create experiment-ID or run-ID subfolders;
- use descriptive filenames that identify the relevant configuration, split,
  and seed when necessary;
- do not silently overwrite an existing result produced by a different
  configuration;
- do not manually edit generated result files unless explicitly needed.

### `reports/`

Contains research notes, methodological documentation, and experiment logs.

Expected future files may include:

- research plan;
- methodology notes;
- experiment log;
- model comparison notes;
- data description.

Do not create these files until explicitly requested.

---

## Event detection principles

The project requires transforming the loaded `Signal` time series into
event-level observations and reference switch intervals.

When implementing event detection, clearly define:

- explicitly configured signal threshold used to identify event onset;
- event start time;
- event end time;
- minimum event duration;
- rules for merging close events;
- handling of missing data;
- sampling frequency;
- censoring or incomplete events.

Event-detection assumptions must be documented because they affect event-window metrics, switch evaluation, and future survival targets.

---

## Feature and representation principles

Features and representations must only use information available at prediction time.

Possible input groups include:

- current and recent past `Signal`;
- local temporal patterns derived only from past and present `Signal` values.

Strictly avoid leakage from the future.

All models must use only `Signal` as input. Validate that loading produces
exactly `Time` and `Signal`, and do not pass `Time` or ignored source columns
into a model.

---

## Prediction timing

The project should support predictions at arbitrary timestamps or at operationally meaningful times during an event.

Examples:

- immediately at event onset;
- 30 seconds after onset;
- 1 minute after onset;
- 2 minutes after onset;
- 5 minutes after onset;
- at every sample during an active or candidate event.

When implementing prediction timing, ensure that inputs computed for a prediction time `t` only use observations available up to `t`.

---

## Train/validation/test splitting

Use time-aware splits.

Avoid random splits because they may cause leakage between nearby events or highly correlated time periods.

Preferred splitting strategy:

- chronological train/validation/test split.

Alternative strategies, if explicitly requested:

- split by months, seasons, or years;
- blocked cross-validation for time series.

Document the splitting strategy clearly.

The validation split is used for:

- hyperparameter selection;
- early stopping;
- checkpoint selection;
- decision-threshold calibration;
- switch-conversion calibration.

The test split is used only for final evaluation.

---

## Reproducibility

For every experiment, save or document when appropriate:

- task formulation;
- model configuration;
- train/validation/test split information;
- preprocessing configuration;
- target definition;
- predictions;
- metrics;
- plots;
- selected device;
- random seed;
- relevant assumptions.

Generated outputs should be reproducible by running scripts.

---

## Testing requirements

Add or update tests when implementing:

- data loading;
- preprocessing;
- reference switch construction;
- event detection;
- windowed dataset construction;
- switch conversion;
- evaluation metrics.

Important edge cases include:

- empty input data;
- missing timestamps;
- unsorted timestamps;
- duplicated timestamps;
- missing `Signal` values;
- events at the beginning or end of the dataset;
- very short events;
- long events;
- no detected events;
- irregular sampling frequency;
- all-zero switch arrays;
- all-one switch arrays;
- undefined precision or recall.

---

## Coding rules

Follow these rules when editing or generating code:

- write clean, readable Python;
- prefer simple and explicit code over overly clever solutions;
- use type hints for important functions;
- add docstrings to public functions;
- keep functions small and testable;
- avoid duplicated logic;
- avoid hard-coded constants;
- do not silently change empirical assumptions;
- do not modify unrelated files;
- do not change public function signatures unless explicitly requested;
- prefer deterministic behavior when possible;
- use the active conda environment when running commands;
- centralize device selection instead of hard-coding CUDA, MPS, or CPU inside scripts.

---

## Before editing files

For non-trivial changes, first provide a short plan including:

1. the objective;
2. files or folders to be modified;
3. functions or scripts to be created or changed, if any;
4. tests to be added or updated, if any;
5. possible risks.

Then proceed with focused changes only.

---

## After editing files

After making changes, summarize:

1. what changed;
2. why it changed;
3. which files or folders were modified;
4. which commands or tests were run;
5. whether the commands or tests passed;
6. any remaining limitations or assumptions.

---

## What not to do

Do not:

- create files inside folders unless explicitly requested;
- create placeholder files such as `.gitkeep` unless explicitly requested;
- modify raw data files;
- introduce future information into model inputs;
- silently change the target definition;
- silently change train/validation/test splits;
- use the test set for threshold calibration or model selection;
- duplicate reference switch logic across model scripts;
- overwrite results without making the process reproducible;
- hard-code local absolute paths;
- hard-code CUDA, MPS, or CPU inside model scripts;
- edit unrelated files;
- remove tests to make the test suite pass;
- report performance without specifying the evaluation split and metrics;
- turn the repository into a generic model zoo;
- reintroduce all exploratory task formulations from the old baseline repository unless explicitly requested;
- add detailed shapelet or survival architectures before their modelling strategies are explicitly defined.

---

## Preferred workflow

The preferred workflow is:

1. define or update the methodological assumption;
2. implement the smallest useful reusable component;
3. add tests when appropriate;
4. run tests when available;
5. generate intermediate outputs;
6. inspect results using interactive `.py` analysis scripts;
7. document assumptions and results;
8. commit changes with Git.

The project should evolve from a clean common preprocessing and evaluation layer toward advanced implementations of the selected task formulations.

At the current stage, implementation priority is the autoregressive branch and the shared preprocessing/evaluation infrastructure.
