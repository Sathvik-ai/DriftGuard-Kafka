#!/usr/bin/env bash
# Run DriftGuard-Kafka M6 TCN Autoencoder training & comparative evaluation
# against M5 LSTM Autoencoder and M4 baselines without modifying prior reports.
#
# Usage: scripts/run_m6_demo.sh [TSB_MAX_PER_DOMAIN]   (default: 1; 0 = all 180 Eva files)
set -euo pipefail
source "$(dirname "$0")/common.sh"

TSB_MAX_PER_DOMAIN="${1:-1}"
CONFIG_PATH="${M6_CONFIG:-${PROJECT_ROOT}/configs/m6_tcn_ae.json}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${PROJECT_ROOT}/reports/m6/${RUN_ID}"
MODELS_DIR="${PROJECT_ROOT}/models/m6/${RUN_ID}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || [[ -x "${PYTHON_BIN}" ]] \
  || fail "Python environment not found at ${PYTHON_BIN}. Run: make setup"

cd "${PROJECT_ROOT}"

REUSE_ARGS=()
if [[ -d "${PROJECT_ROOT}/reports/m5" ]]; then
  LATEST_M5="$(find "${PROJECT_ROOT}/reports/m5" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1 || true)"
  if [[ -n "${LATEST_M5}" && -f "${LATEST_M5}/m5_report.json" && -f "${LATEST_M5}/per_experiment_metrics.csv" ]]; then
    REUSE_ARGS=(--reuse-m5-dir "${LATEST_M5}")
    log "Detected verified M5 report bundle for split-consistent comparison: ${LATEST_M5}"
  fi
fi

log "Running DriftGuard-Kafka M6 TCN Autoencoder experiment (config=${CONFIG_PATH}, tsb_max_per_domain=${TSB_MAX_PER_DOMAIN})"
"${PYTHON_BIN}" -m app.m6_tcn_autoencoder \
  --config "${CONFIG_PATH}" \
  --run-id "${RUN_ID}" \
  --skab-dir "${PROJECT_ROOT}/data/external/skab" \
  --tsb-dir "${PROJECT_ROOT}/data/external/tsb_ad_m" \
  --tsb-max-per-domain "${TSB_MAX_PER_DOMAIN}" \
  --save-models-dir "${MODELS_DIR}" \
  --output-dir "${REPORT_DIR}" \
  "${REUSE_ARGS[@]+"${REUSE_ARGS[@]}"}"

log "M6 experiment complete: ${REPORT_DIR}/m6_report.md"
