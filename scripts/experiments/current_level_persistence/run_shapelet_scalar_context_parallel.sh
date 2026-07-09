#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_TEXT="${GPU_IDS:-0 1 2}"
read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_TEXT}"
if [[ ${#GPU_IDS_ARRAY[@]} -lt 3 ]]; then
  echo "GPU_IDS must provide at least three device IDs, for example: GPU_IDS='0 1 2'." >&2
  exit 1
fi

LOG_DIR="logs/current_level_persistence/scalar_context"
RUNNER="scripts/experiments/current_level_persistence/train_learnable_shapelets.py"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  "configs/current_level_persistence/shapelet_mlp_delta_0p5.yaml"
  "configs/current_level_persistence/shapelet_transformer_delta_0p5.yaml"
  "configs/current_level_persistence/shapelet_convolution_delta_0p5.yaml"
)
NAMES=("mlp" "transformer" "convolution")
PIDS=()
EXTRA_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--force")
fi

for index in 0 1 2; do
  name="${NAMES[$index]}"
  gpu="${GPU_IDS_ARRAY[$index]}"
  log_path="${LOG_DIR}/${name}_scalar_context.log"
  echo "Starting ${name} scalar-context run on CUDA device ${gpu}: ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${RUNNER}" \
    --config "${CONFIGS[$index]}" \
    "${EXTRA_ARGS[@]}" \
    > "${log_path}" 2>&1 &
  PIDS+=("$!")
done

status=0
for index in 0 1 2; do
  if ! wait "${PIDS[$index]}"; then
    echo "${NAMES[$index]} run failed; inspect its log." >&2
    status=1
  fi
done
exit "${status}"
