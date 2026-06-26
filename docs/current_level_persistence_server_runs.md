# Current-Level Persistence Server Runs

This note contains the commands for running the initial learnable-shapelet
experiments on a remote GPU server.

The three runs are:

```text
multiscale_shapelet_mlp
multiscale_shapelet_transformer
multiscale_shapelet_convolution
```

They use:

```text
input:  X_relative_to_current
target: y_log1p_remaining_persistence_seconds
delta:  0.5
L:      30
```

## 1. Copy Or Sync Repository To Server

Use your preferred method. One generic `rsync` pattern is:

```bash
rsync -av \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  ./ user@server:/path/to/Nowcasting-project/
```

The processed dataset must be present on the server:

```text
data/processed/current_level_persistence/delta_0p5/L30/externalHoldout_test_fc_uplink_fade/
```

## 2. Activate Environment

The project environment convention is:

```bash
conda activate Nowcasting
```

If the server uses micromamba:

```bash
micromamba activate Nowcasting
```

All server scripts assume the environment is already active and that `python`
points to that environment.

## 3. Check GPUs

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Expected for the full launcher:

```text
True
3
```

If fewer GPUs are visible, either free GPUs, adjust `CUDA_VISIBLE_DEVICES` in
the launcher scripts, or run the models sequentially.

## 4. Check Dataset Size

```bash
du -sh data/processed/current_level_persistence/delta_0p5/L30/externalHoldout_test_fc_uplink_fade/
ls data/processed/current_level_persistence/delta_0p5/L30/externalHoldout_test_fc_uplink_fade/
```

Required files:

```text
train.npz
val.npz
test.npz
train_metadata.parquet
val_metadata.parquet
test_metadata.parquet
dataset_metadata.yaml
```

## 5. Run Tests On Server

```bash
pytest tests/current_level_persistence -q
```

The tests do not run full training.

## 6. Run Server Smoke Test

Run this first. It launches all three models for only three epochs, using one
GPU per model.

```bash
bash scripts/experiments/run_shapelet_current_level_persistence_server_smoke.sh
```

The smoke runs use `--run-suffix server_smoke` and `--force`, so rerunning the
smoke script replaces previous smoke outputs.

## 7. Monitor Smoke Logs

```bash
tail -f logs/current_level_persistence/mlp_server_smoke.log
tail -f logs/current_level_persistence/transformer_server_smoke.log
tail -f logs/current_level_persistence/convolution_server_smoke.log
```

Each log should print:

```text
Device: cuda
```

If it prints `Device: cpu`, CUDA is not visible to PyTorch in that process.

## 8. Run Full Initial Models

After the smoke test passes:

```bash
bash scripts/experiments/run_shapelet_current_level_persistence_server.sh
```

The full runs use `--run-suffix server_initial`. They do not pass `--force`, so
the runner stops instead of overwriting an existing `server_initial` run.

If an older convolution `server_initial` run exists from before the numerical
stability fixes, remove it or rerun only the convolution command with `--force`.
The current convolution config uses a lower learning rate, disables mixed
precision, and enables gradient clipping.

## 9. Monitor Full Logs

```bash
tail -f logs/current_level_persistence/mlp_server_initial.log
tail -f logs/current_level_persistence/transformer_server_initial.log
tail -f logs/current_level_persistence/convolution_server_initial.log
```

## 10. Check Results

Run outputs:

```text
results/runs/current_level_persistence/externalHoldout_test_fc_uplink_fade/<run_id>/
```

Checkpoint outputs:

```text
models/current_level_persistence/<model_id>/<run_id>/
```

Run index:

```text
results/index/current_level_persistence_runs.csv
```

Logs:

```text
logs/current_level_persistence/
```

The `server_initial` run IDs are:

```text
currentLevelPersistence_delta0p5_L30_multiscale_shapelet_mlp_externalHoldout_test_fc_uplink_fade__server_initial
currentLevelPersistence_delta0p5_L30_multiscale_shapelet_transformer_externalHoldout_test_fc_uplink_fade__server_initial
currentLevelPersistence_delta0p5_L30_multiscale_shapelet_convolution_externalHoldout_test_fc_uplink_fade__server_initial
```

## 11. Build Model Comparison Table

After the full runs finish:

```bash
python scripts/experiments/summarize_current_level_persistence_runs.py
```

Output:

```text
results/comparisons/model_summary/current_level_persistence/externalHoldout_test_fc_uplink_fade/initial_shapelet_model_comparison_delta0p5/tables/initial_shapelet_model_comparison_delta0p5.csv
```

## 12. Common Failure Checks

CUDA visibility:

```bash
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Dataset presence:

```bash
du -sh data/processed/current_level_persistence/delta_0p5/L30/externalHoldout_test_fc_uplink_fade/
```

Disk space:

```bash
df -h .
```

Existing run protection:

```bash
find results/runs/current_level_persistence -maxdepth 3 -type d -name '*server_initial*'
find models/current_level_persistence -maxdepth 3 -type d -name '*server_initial*'
```

If you intentionally want to replace full `server_initial` outputs, add
`--force` to the three commands inside
`scripts/experiments/run_shapelet_current_level_persistence_server.sh`.
