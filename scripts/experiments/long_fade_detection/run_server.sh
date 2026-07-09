#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="logs/long_fade_detection"
mkdir -p "${LOG_DIR}"

echo "Starting long-fade extended grid search..."
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" scripts/experiments/long_fade_detection/run_grid_search.py \
  --config configs/long_fade_detection/grid_search_threshold10_duration300.yaml \
  --force \
  > "${LOG_DIR}/grid_search.log" 2>&1
