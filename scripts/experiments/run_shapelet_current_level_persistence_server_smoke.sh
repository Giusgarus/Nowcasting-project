#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="logs/current_level_persistence"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-current_level_persistence}}"

mkdir -p "${LOG_DIR}" "${MPLCONFIGDIR}"

echo "Starting current_level_persistence shapelet smoke runs..."
echo "Python: ${PYTHON_BIN}"
echo "Logs: ${LOG_DIR}"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" \
  scripts/experiments/run_shapelet_current_level_persistence.py \
  --config configs/current_level_persistence/shapelet_mlp_delta_0p5.yaml \
  --max-epochs 3 \
  --run-suffix server_smoke \
  --force \
  > "${LOG_DIR}/mlp_server_smoke.log" 2>&1 &
PID_MLP=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" \
  scripts/experiments/run_shapelet_current_level_persistence.py \
  --config configs/current_level_persistence/shapelet_transformer_delta_0p5.yaml \
  --max-epochs 3 \
  --run-suffix server_smoke \
  --force \
  > "${LOG_DIR}/transformer_server_smoke.log" 2>&1 &
PID_TRANSFORMER=$!

CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" \
  scripts/experiments/run_shapelet_current_level_persistence.py \
  --config configs/current_level_persistence/shapelet_convolution_delta_0p5.yaml \
  --max-epochs 3 \
  --run-suffix server_smoke \
  --force \
  > "${LOG_DIR}/convolution_server_smoke.log" 2>&1 &
PID_CONVOLUTION=$!

echo "MLP PID: ${PID_MLP}"
echo "Transformer PID: ${PID_TRANSFORMER}"
echo "Convolution PID: ${PID_CONVOLUTION}"

wait "${PID_MLP}"
wait "${PID_TRANSFORMER}"
wait "${PID_CONVOLUTION}"

echo "All current_level_persistence shapelet smoke runs finished."
