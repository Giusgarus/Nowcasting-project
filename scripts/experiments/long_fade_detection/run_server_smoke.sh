#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="logs/long_fade_detection"
mkdir -p "${LOG_DIR}"

echo "Running long-fade grid dry-run..."
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" scripts/experiments/long_fade_detection/run_grid_search.py \
  --config configs/long_fade_detection/grid_search_threshold10_duration300.yaml \
  --dry-run \
  > "${LOG_DIR}/grid_search_smoke.log" 2>&1

echo "Long-fade grid smoke completed. Log: ${LOG_DIR}/grid_search_smoke.log"
