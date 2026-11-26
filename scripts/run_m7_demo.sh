#!/usr/bin/env bash
# Run DriftGuard-Kafka M7 TranAD training & comparative evaluation against
# M6 TCN, M5 LSTM, and M4 baselines while preserving prior reports untouched.
#
# Usage: scripts/run_m7_demo.sh [TSB_MAX_PER_DOMAIN]   (default: 1; 0 = all 180 Eva files)
set -euo pipefail
source "$(dirname "$0")/common.sh"

TSB_MAX_PER_DOMAIN="${1:-1}"
CONFIG_PATH="${M7_CONFIG:-${PROJECT_ROOT}/configs/m7_tranad.json}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${PROJECT_ROOT}/reports/m7/${RUN_ID}"
MODELS_DIR="${PROJECT_ROOT}/models/m7/${RUN_ID}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || [[ -x "${PYTHON_BIN}" ]] \
  || fail "Python environment not found at ${PYTHON_BIN}. Run: make setup"

cd "${PROJECT_ROOT}"

REUSE_ARGS=()
if [[ -d "${PROJECT_ROOT}/reports/m6" ]]; then
  LATEST_M6="$(find "${PROJECT_ROOT}/reports/m6" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1 || true)"
  if [[ -n "${LATEST_M6}" && -f "${LATEST_M6}/m6_report.json" && -f "${LATEST_M6}/per_experiment_metrics.csv" ]]; then
    REUSE_ARGS=(--reuse-m6-dir "${LATEST_M6}")
    log "Detected verified M6 report bundle for split-consistent comparison: ${LATEST_M6}"
  fi
fi

log "Running DriftGuard-Kafka M7 TranAD experiment (config=${CONFIG_PATH}, tsb_max_per_domain=${TSB_MAX_PER_DOMAIN})"
"${PYTHON_BIN}" -m app.m7_tranad \
  --config "${CONFIG_PATH}" \
  --run-id "${RUN_ID}" \
  --skab-dir "${PROJECT_ROOT}/data/external/skab" \
  --tsb-dir "${PROJECT_ROOT}/data/external/tsb_ad_m" \
  --tsb-max-per-domain "${TSB_MAX_PER_DOMAIN}" \
  --save-models-dir "${MODELS_DIR}" \
  --output-dir "${REPORT_DIR}" \
  "${REUSE_ARGS[@]+"${REUSE_ARGS[@]}"}"

log "M7 experiment complete: ${REPORT_DIR}/m7_report.md"
