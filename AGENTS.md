# AGENTS.md

## Project overview

This repository implements the **advanced modelling phase** of a satellite attenuation nowcasting project.

The previous repository was used for an exploratory task-selection phase. In that phase, several alternative task formulations were tested using simple diagnostic models such as Ridge/logistic regression and XGBoost. The goal of the previous phase was not to select a final production model, but to identify which task formulations were empirically promising.

This new repository focuses on the selected task formulations:

1. **Autoregressive forecasting**
2. **Shapelet-pattern modelling**
3. **Current-level persistence**
4. **Long-fade detection**
5. **Probabilistic survival modelling**

At the current stage, the autoregressive branch is the broadest advanced-modelling branch. Current-level persistence, long-fade detection, and the requested survival-persistence XGBoost-AFT and discrete-time TCN baselines have also been implemented. Do not create additional shapelet, survival, or other task-specific model assumptions, scripts, or architecture-specific files unless explicitly requested.

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

- GRU sequence-to-vector baseline;
- GRU encoder-decoder;
- XGBoost tabular baseline with one independent regressor per forecast horizon;
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

### 3. Current-level persistence

The current-level persistence task is separate from autoregressive forecasting.
Given recent `Signal` history up to time `t`, it predicts how long the signal
will remain above `S_t - delta`.

Current default setup:

```text
delta = 0.5
context_length = 30
sampling_time_seconds = 30
```

Input:

```text
X_raw = [S_{t-L+1}, ..., S_t]
X_relative_to_current = X_raw - S_t
```

Learnable-shapelet models may also use explicitly configured scalar context
derived from `X_raw`, including absolute level, window summaries, and recent
slopes. These features must remain auxiliary to `X_relative_to_current`, be
standardized using train-only statistics, and save their ordered names and
scaler parameters in run metadata.

Targets:

```text
remaining_persistence_samples
remaining_persistence_seconds
log1p_remaining_persistence_seconds
```

The duration target must be computed from the full cleaned signal rather than
being truncated at the event-window boundary. Use the actual elapsed time from
`t` to the first future observation below `S_t - delta`. Input contexts must
remain inside one continuous acquisition segment. If recovery is not observed
before that segment ends, mark the target as right-censored and exclude it from
ordinary duration regression; never treat a later observation after a data gap
as proof of continuous persistence through the gap.

When explicitly configured, the full-signal target search may first apply the
shared conservative small-gap imputation policy: expected 30-second sampling,
maximum 90-second gaps, maximum two inserted samples, linear interpolation in
time, and no interpolation across acquisition segments.

Initial model IDs:

- `multiscale_shapelet_mlp`;
- `multiscale_shapelet_transformer`;
- `multiscale_shapelet_convolution`.
- `xgboost`.

For current-level XGBoost, use flattened `X_relative_to_current` plus
explicitly configured scalar context when enabled, predict
`log1p_remaining_persistence_seconds`, select by validation-only duration
metrics, then evaluate the external test split once.

Use dedicated task paths:

```text
src/tasks/current_level_persistence/
scripts/experiments/current_level_persistence/
configs/current_level_persistence/
data/processed/current_level_persistence/
results/runs/current_level_persistence/
models/current_level_persistence/
```

Do not save current-level persistence artifacts under autoregressive folders.

---

### 4. Long-fade detection

The long-fade detection task is a binary grouped-event classification task.
At each timestamp `t` inside a grouped fade event, it predicts whether the
whole grouped fade event duration is at least a configured minimum duration.

Current default setup:

```text
threshold_db = 10.0
min_fade_duration_seconds = 300
context_length = 30
sampling_time_seconds = 30
```

Event grouping starts at the first threshold crossing above `threshold_db`.
Further threshold crossings within 3 hours from that first crossing belong to
the same grouped event. Temporary below-threshold samples between the first and
last grouped crossing remain inside the event and must not split it. Every
timestamp inside one grouped event receives the same `y_long_fade` label.
When explicitly configured, the dataset builder may apply the shared
small-gap imputation pass to the loaded signal source, but it must not bridge
long gaps or distinct loaded signal groups.

Main sequence input:

```text
X_relative_to_threshold = X_raw - threshold_db
```

Allowed scalar context features are derived only from the past/current raw
signal window and event position available at time `t`. Do not use event
duration, final position fraction, inside-event flags, threshold masks, or
labels as model inputs.

Initial model IDs:

- `xgboost_lag_scalar_classifier`;
- `tcn_classifier`;
- `multiscale_shapelet_convolution_classifier`.

Use dedicated task paths:

```text
src/tasks/long_fade_detection/
scripts/experiments/long_fade_detection/
configs/long_fade_detection/
configs/long_fade_detection/grid_search_threshold10_duration300.yaml
data/processed/long_fade_detection/
results/runs/long_fade_detection/
models/long_fade_detection/
```

Do not save long-fade detection artifacts under autoregressive or
current-level persistence folders.

---

### 5. Probabilistic survival modelling

The survival task estimates a distribution over the residual duration of the ongoing attenuation event.

At time `t`, let `R_t` be the remaining duration of the event. The model should eventually estimate a survival curve:

```text
S(u | X_t) = P(R_t > u | X_t)
```

for a discrete or continuous set of future horizons.

The repository contains a model-independent `survival_persistence` dataset
pipeline. That pipeline handles state-machine event construction,
continuous-time survival labels, right-censoring metadata, train-only feature
scaling, and conservative small-gap imputation when explicitly configured.
When explicitly configured, survival event detection and target construction
may use the shared conservative small-gap imputation policy before detecting
fade events. Right-censoring still applies at acquisition segment boundaries.

The implemented survival baselines are XGBoost-AFT and a discrete-time causal
TCN. XGBoost-AFT must use the saved AFT lower/upper bounds directly. The TCN
must convert the same continuous-time labels to discrete event/censoring masks
in memory without modifying the canonical dataset artifacts. Both baselines
must keep censored samples, select hyperparameters or epochs on validation
only, and evaluate the external test split once after model selection. DeepHit
or other survival model families must not be introduced unless explicitly
requested.

Current default survival switch conversion:

```text
model_switch_raw(t) = 1 if S(300s | X_t) >= probability_threshold and Signal(t) >= 10.0
```

Then apply the shared switch post-processing and compare against the
task-scoped Perfect Switch. The uncalibrated default threshold is `0.5`.
The retained exploratory survival-switch output currently uses `0.65` after a
local test-set sweep; do not treat that value as validation-calibrated unless a
separate validation-only calibration step is implemented.

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
│   ├── autoregressive/
│   ├── current_level_persistence/
│   ├── long_fade_detection/
│   ├── survival_persistence/
│   ├── data.yaml
│   ├── autoregressive_gru.yaml
│   ├── autoregressive_patchtst.yaml
│   ├── patchtst_grid_search.yaml
│   ├── xgboost_grid_search.yaml
│   ├── autoregressive_chronos.yaml
│   ├── autoregressive_mamba.yaml
│   ├── data_preparation.yaml
│   ├── autoregressive_dataset.yaml
│   ├── perfect_switch.yaml
│   └── switch_comparison.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   │   ├── clean_signal/
│   │   ├── candidate_events/
│   │   └── event_windows/
│   └── processed/
│       ├── autoregressive/
│       ├── current_level_persistence/
│       ├── long_fade_detection/
│       └── survival_persistence/
│
├── scripts/
│   ├── analysis/
│   │   ├── 00_data_inventory.py
│   │   ├── 01_single_dataset_inspection.py
│   │   ├── 02_sampling_and_gaps.py
│   │   ├── 03_signal_distribution.py
│   │   ├── 04_window_availability.py
│   │   ├── 05_candidate_event_exploration.py
│   │   ├── 06_baseline_drift_and_daily_patterns.py
│   │   ├── 07_signal_change_and_outlier_analysis.py
│   │   ├── 08_normalization_diagnostics.py
│   │   └── autoregressive/
│   │       ├── 09_prepare_event_windows.py
│   │       ├── 10_build_autoregressive_datasets.py
│   │       ├── 11_sanity_check_autoregressive_dataset.py
│   │       ├── 12_compute_perfect_switch.py
│   │       ├── 13_compare_switch_methods.py
│   │       └── 14_compare_model_results.py
│   │
│   └── experiments/
│       ├── current_level_persistence/
│       │   ├── build_dataset.py
│       │   ├── train_learnable_shapelets.py
│       │   ├── run_shapelet_grid_search.py
│       │   ├── run_xgboost_grid_search.py
│       │   └── summarize_runs.py
│       ├── long_fade_detection/
│       │   ├── build_dataset.py
│       │   ├── run_model.py
│       │   ├── run_grid_search.py
│       │   ├── summarize_runs.py
│       │   ├── run_server_smoke.sh
│       │   └── run_server.sh
│       ├── survival_persistence/
│       │   ├── build_dataset.py
│       │   ├── audit_dataset.py
│       │   ├── run_xgboost_aft.py
│       │   ├── run_xgboost_aft_grid_search.py
│       │   ├── run_discrete_time_tcn.py
│       │   ├── run_discrete_time_tcn_controlled.py
│       │   └── run_discrete_time_tcn_grid_search.py
│       └── autoregressive/
│           ├── run_autoregressive_gru.py
│           ├── run_autoregressive_patchtst.py
│           ├── run_patchtst_grid_search.py
│           ├── run_xgboost_grid_search.py
│           ├── run_autoregressive_chronos_zero_shot.py
│           └── run_autoregressive_mamba.py
│
├── src/
│   ├── analysis/
│   │   ├── data_quality.py
│   │   └── plots.py
│   │
│   ├── data/
│   │   ├── loading.py
│   │   ├── preprocessing.py
│   │   ├── events.py
│   │   ├── splits.py
│   │   └── validation.py
│   │
│   ├── datasets/
│   │   └── compatibility wrappers only
│   │
│   ├── models/
│   │   └── autoregressive/
│   │       └── compatibility wrappers only
│   │
│   ├── evaluation/
│   │   ├── switch_metrics.py
│   │   └── forecast_metrics.py compatibility wrapper
│   │
│   ├── switching/
│   │   ├── reference_switch.py
│   │   ├── conversion.py
│   │   ├── comparison.py
│   │   ├── metrics.py
│   │   └── plots.py
│   │
│   ├── utils/
│   │   ├── config.py
│   │   ├── device.py
│   │   ├── paths.py
│   │   ├── logging.py
│   │   └── reproducibility.py
│
│   └── tasks/
│       ├── autoregressive/
│       │   ├── data/
│       │   ├── models/
│       │   ├── evaluation/
│       │   └── utils/
│       ├── current_level_persistence/
│       │   ├── data/
│       │   ├── models/
│       │   ├── evaluation/
│       │   ├── plots/
│       │   └── utils/
│       ├── long_fade_detection/
│       │   ├── data/
│       │   ├── models/
│       │   ├── evaluation/
│       │   └── utils/
│       └── survival_persistence/
│           ├── data/
│           ├── evaluation/
│           ├── models/
│           └── utils/
│
├── models/
│   ├── autoregressive/
│   │   ├── gru/
│   │   ├── patchtst/
│   │   ├── chronos/
│   │   └── xgboost/
│   ├── current_level_persistence/
│   ├── long_fade_detection/
│   └── survival_persistence/
│
├── results/
│   ├── runs/
│   │   ├── autoregressive/
│   │   ├── current_level_persistence/
│   │   ├── long_fade_detection/
│   │   └── survival_persistence/
│   ├── switching/
│   │   └── perfect_switch/
│   ├── comparisons/
│   ├── data_preparation/
│   ├── data_analysis/
│   └── index/
│
├── reports/
│
└── tests/
    ├── autoregressive/
    ├── current_level_persistence/
    ├── long_fade_detection/
    ├── survival_persistence/
    ├── data/
    ├── evaluation/
    ├── switching/
    ├── tuning/
    └── utils/
```

Do not create all files immediately unless explicitly requested. The structure describes the intended organization of the repository.

Shapelet and survival configs, scripts, model modules, and model-specific
artifact folders should be added later, once their advanced modelling
strategies are explicitly defined. Their top-level `models/` and `results/`
task folders may exist before those strategies are fixed.

Run, reference, and comparison IDs define their shallow artifact folders.
Use `src/utils/results_paths.py` instead of constructing result paths manually.

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

All model runners and all grid-search runners must follow this policy. For
grid searches with independent trials, use all available CUDA GPUs when
possible, for example `cuda:0`, `cuda:1`, `cuda:2`. If CUDA is not available,
use MPS when supported by the model/library; otherwise fall back to CPU.

Do not hard-code a device inside model scripts except when assigning a specific
trial to one of the available devices selected by the centralized device
policy.

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
- each analysis script should print one concise run overview stating what it
  prints, displays, and saves;
- write analysis-script output, explanations, plot titles, and code comments in
  English;
- use short methodological comments for assumptions or non-obvious operations,
  but do not narrate self-explanatory code line by line;
- shared raw-data analysis tables should be saved under
  `results/data_analysis/`;
- exploratory raw-data analysis figures should be displayed interactively and
  should not be saved automatically;
- model-specific analysis outputs should be saved under the relevant
  `results/runs/<task_name>/<selection_id>/<run_id>/` folder;
- cross-run, model-vs-reference, and baseline comparisons should be saved under
  `results/comparisons/<comparison_type>/<task_name>/<selection_id>/<comparison_id>/`.

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

- GRU;
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

The `survival_persistence` dataset builder is model-independent and exists
under the task-scoped survival folders.

It must produce continuous-time residual-duration labels, observed/censored
survival bounds, event metadata, and no-leakage past-context inputs. It must
not introduce model-specific discretization masks or DeepHit labels. XGBoost-AFT
training and grid-search code is implemented separately under the survival task
because it was explicitly requested. Neural survival models remain deferred
unless explicitly requested.

---

## Reference switch policy

All final models must be evaluated against the same reference switch construction.

The model-agnostic reference switch logic should be centralized in:

```text
src/switching/reference_switch.py
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

Model checkpoints and their reproducibility metadata must use:

```text
models/<task_name>/<model_family>/<run_id>/
```

Additional folders such as:

```text
models/shapelet/<run_id>/
models/survival/<run_id>/
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
autoregressive/run_autoregressive_gru.py
autoregressive/run_autoregressive_patchtst.py
autoregressive/run_patchtst_grid_search.py
autoregressive/run_xgboost_grid_search.py
autoregressive/run_autoregressive_chronos_zero_shot.py
autoregressive/run_autoregressive_mamba.py
current_level_persistence/build_dataset.py
current_level_persistence/train_learnable_shapelets.py
current_level_persistence/run_shapelet_grid_search.py
current_level_persistence/run_xgboost_grid_search.py
current_level_persistence/summarize_runs.py
long_fade_detection/build_dataset.py
long_fade_detection/run_model.py
long_fade_detection/run_grid_search.py
long_fade_detection/summarize_runs.py
survival_persistence/build_dataset.py
survival_persistence/audit_dataset.py
survival_persistence/run_xgboost_aft.py
survival_persistence/run_xgboost_aft_grid_search.py
survival_persistence/run_discrete_time_tcn.py
survival_persistence/run_discrete_time_tcn_controlled.py
survival_persistence/run_discrete_time_tcn_grid_search.py
```

For survival, the model-independent dataset builder, XGBoost-AFT baseline, and
discrete-time TCN baseline are currently implemented. Do not add DeepHit,
additional survival model families, or alternate target definitions until
explicitly requested.

### `src/`

Contains reusable Python code.

Expected modules include:

- shared data-analysis summaries and plots;
- data loading;
- common preprocessing;
- reference switch construction;
- event detection;
- temporal splitting;
- task-scoped autoregressive dataset builders;
- task-scoped autoregressive model definitions;
- task-scoped model adapters;
- evaluation metrics;
- plotting;
- utilities.

Do not place experiment-specific logic directly in reusable modules unless it is clearly general.

Canonical task-specific code must live under:

```text
src/tasks/autoregressive/
src/tasks/current_level_persistence/
src/tasks/long_fade_detection/
src/tasks/survival_persistence/
```

Legacy namespaces such as `src/datasets/`, `src/evaluation/forecast_metrics.py`,
and `src/analysis/event_preparation.py` may remain only as compatibility
wrappers. Do not add new task logic there.

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

Use:

```text
models/<task_name>/<model_family>/<run_id>/
```

Checkpoints must remain outside `results/`. Each model folder should contain
the checkpoint, model configuration, and training metadata required to
reproduce the corresponding indexed run.

### `results/`

Contains generated outputs.

Rules:

- save shared raw-data analysis tables under `results/data_analysis/`;
- display exploratory raw-data figures interactively without saving them
  automatically;
- organize model outputs under
  `results/runs/<task_name>/<selection_id>/<run_id>/`;
- organize Perfect Switch outputs under
  `results/switching/perfect_switch/<task_name>/<selection_id>/`;
- organize model and baseline comparisons under
  `results/comparisons/<comparison_type>/<task_name>/<selection_id>/<comparison_id>/`;
- save human-readable preparation summaries under
  `results/data_preparation/<task_name>/<selection_id>/`;
- update the relevant CSV index under `results/index/`;
- keep each run or comparison artifact folder shallow, using direct `metrics/`,
  `predictions`, `figures`, and `tables` children;
- use descriptive filenames that identify the relevant configuration, split,
  and seed when necessary;
- do not silently overwrite an existing result produced by a different
  configuration;
- do not manually edit generated result files unless explicitly needed.

## Results organization policy

All generated artifacts must use stable IDs from `src/utils/results_paths.py`.

- Model run outputs go under
  `results/runs/<task_name>/<selection_id>/<run_id>/`.
- Model checkpoints go under
  `models/<task_name>/<model_family>/<run_id>/`.
- Autoregressive artifacts use `task_name=autoregressive`.
- Current-level persistence run outputs go under
  `results/runs/current_level_persistence/<selection_id>/<run_id>/`.
- Current-level persistence checkpoints go under
  `models/current_level_persistence/<model_id>/<run_id>/`.
- Current-level persistence processed datasets go under
  `data/processed/current_level_persistence/`.
- Current-level persistence should use task-local path helpers under
  `src/tasks/current_level_persistence/utils/` when the shared autoregressive
  ID structure does not apply.
- Long-fade detection first-stage runs go under
  `results/runs/long_fade_detection/<selection_id>/<run_id>/`, checkpoints go
  under `models/long_fade_detection/<model_id>/<run_id>/`, and processed
  datasets go under `data/processed/long_fade_detection/`.
- Long-fade detection should use task-local path helpers under
  `src/tasks/long_fade_detection/utils/` because its first-stage run IDs
  already encode threshold, duration, model, and external-holdout selection.
- Perfect Switch outputs go under
  `results/switching/perfect_switch/<task_name>/<selection_id>/`.
- Model-vs-switch comparisons go under
  `results/comparisons/switch_eval/<task_name>/<selection_id>/<comparison_id>/`.
- Cross-model result summaries go under
  `results/comparisons/model_summary/<task_name>/<selection_id>/<comparison_id>/`.
- Model-selection or architecture comparisons go under
  `results/comparisons/model_selection/<task_name>/<selection_id>/<comparison_id>/`.
- Each model-vs-reference comparison folder must contain exactly one method/run
  evaluated against one reference; batch configurations may list multiple
  methods, but their outputs must remain separate.
- Human-readable data-preparation summaries go under
  `results/data_preparation/<task_name>/<selection_id>/`.
- Hyperparameter-search artifacts go under
  `results/grid_searches/<task_name>/<selection_id>/<search_id>/`.
- Grid-search selection must use validation metrics only. Evaluate the test split
  only after the winning trial has been selected.
- Large grids must print their complete trial count and use an explicit safety
  limit or require an explicit configuration change before execution.
- Reusable trial checkpoints must not contain test-derived selection metrics.
- Independent neural-model grid-search trials should use the shared
  device-aware parallel scheduler when multiple CUDA GPUs are available.
- Multi-GPU trial scheduling must detect currently available GPUs dynamically
  and fall back to one worker on a single GPU, MPS, or CPU.
- Keep model-family-specific training logic outside the generic grid expansion,
  ranking, and indexing utilities.
- Every writer must update the appropriate central index in `results/index/`.
- Avoid deeply nested result folders and duplicate copies of the same plot.
- Use path utilities instead of hard-coded result paths.
- During migration, old paths may remain readable, but new outputs must use the
  stable shallow structure.

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

The current shared autoregressive data-preparation pipeline uses configurable
candidate events only to select event-centered data windows. With the default
configuration, a candidate crossing is:

```text
Signal >= 10.0
```

Crossings are grouped relative to the first crossing of the current event, and
the default event-centered window spans 1.5 hours before and after that first
crossing. This candidate-event rule is not the final event definition, Perfect
Switch, Smart Switch, or switch-conversion rule. It does not require minimum
persistence.

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

Organize tests by area:

```text
tests/autoregressive/
tests/current_level_persistence/
tests/data/
tests/evaluation/
tests/switching/
tests/tuning/
tests/utils/
```

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
