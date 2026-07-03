# Current-Level Persistence Server Runs

## Scalar Context

The learnable-shapelet models still encode:

```text
X_relative_to_current = X_raw - S_t
```

They now also receive 12 scalar features derived from `X_raw`: current and
recovery levels, full-window summary statistics, recent slopes, and recent
five-sample statistics. Scalar features are standardized with train-split
statistics only. The fitted mean and standard deviation are saved in run and
checkpoint metadata.

Rebuild the dataset to persist these fields in the split NPZ files and metadata:

```bash
python scripts/experiments/current_level_persistence/build_dataset.py \
  --config configs/current_level_persistence/dataset_delta_0p5_L30_external_holdout.yaml \
  --force
```

Old split files remain readable because the runner can reconstruct scalar
features from `X_raw`, but rebuilding is recommended for complete artifacts.

## Smoke Runs

From the repository root on a machine with three visible CUDA GPUs:

```bash
bash scripts/experiments/run_shapelet_current_level_persistence_scalar_context_server_smoke.sh
```

Override GPU IDs when necessary:

```bash
GPU_IDS="1 2 3" bash scripts/experiments/run_shapelet_current_level_persistence_scalar_context_server_smoke.sh
```

The smoke launcher runs two epochs per model and writes logs under:

```text
logs/current_level_persistence/scalar_context/
```

## Full Initial Runs

```bash
bash scripts/experiments/run_shapelet_current_level_persistence_scalar_context_server.sh
```

The script starts MLP, Transformer, and Convolution concurrently, assigning one
GPU to each process. Existing runs are not overwritten. To intentionally rerun
the same IDs:

```bash
FORCE=1 bash scripts/experiments/run_shapelet_current_level_persistence_scalar_context_server.sh
```

Use the server's global Python by default. Set `PYTHON_BIN` if another executable
is required. To keep a run alive after disconnecting, invoke the script inside a
`tmux` session.
