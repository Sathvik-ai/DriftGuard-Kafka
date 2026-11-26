#!/usr/bin/env bash
# Run DriftGuard-Kafka M5 LSTM Autoencoder training & comparative evaluation.
#
# Usage: scripts/run_m5_demo.sh [TSB_MAX_PER_DOMAIN]   (default: 1; 0 = all 180 Eva files)
set -euo pipefail
source "$(dirname "$0")/common.sh"

TSB_MAX_PER_DOMAIN="${1:-1}"
CONFIG_PATH="${M5_CONFIG:-${PROJECT_ROOT}/configs/m5_lstm_ae.json}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${PROJECT_ROOT}/reports/m5/${RUN_ID}"
MODELS_DIR="${PROJECT_ROOT}/models/m5/${RUN_ID}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || [[ -x "${PYTHON_BIN}" ]] \
  || fail "Python environment not found at ${PYTHON_BIN}. Run: make setup"

cd "${PROJECT_ROOT}"
log "Running DriftGuard-Kafka M5 LSTM Autoencoder experiment (config=${CONFIG_PATH}, tsb_max_per_domain=${TSB_MAX_PER_DOMAIN})"
"${PYTHON_BIN}" -m app.m5_lstm_autoencoder \
  --config "${CONFIG_PATH}" \
  --run-id "${RUN_ID}" \
  --skab-dir "${PROJECT_ROOT}/data/external/skab" \
  --tsb-dir "${PROJECT_ROOT}/data/external/tsb_ad_m" \
  --tsb-max-per-domain "${TSB_MAX_PER_DOMAIN}" \
  --save-models-dir "${MODELS_DIR}" \
  --output-dir "${REPORT_DIR}"

log "M5 experiment complete: ${REPORT_DIR}/m5_report.md"
