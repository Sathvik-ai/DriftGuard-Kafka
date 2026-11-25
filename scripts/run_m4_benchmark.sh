#!/usr/bin/env bash
# Run DriftGuard-Kafka M4 comparative ML benchmark on SKAB and TSB-AD-M.
#
# Usage: scripts/run_m4_benchmark.sh [TSB_MAX_PER_DOMAIN]   (default: 2; 0 = all 180 Eva files)
set -euo pipefail
source "$(dirname "$0")/common.sh"

TSB_MAX_PER_DOMAIN="${1:-2}"
SEED="${DEMO_SEED:-42}"
QUANTILE="${M4_CALIBRATION_QUANTILE:-0.99}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${PROJECT_ROOT}/reports/m4/${RUN_ID}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || [[ -x "${PYTHON_BIN}" ]] \
  || fail "Python environment not found at ${PYTHON_BIN}. Run: make setup"

cd "${PROJECT_ROOT}"
log "Running DriftGuard-Kafka M4 benchmark (seed=${SEED}, tsb_max_per_domain=${TSB_MAX_PER_DOMAIN})"
"${PYTHON_BIN}" -m app.m4_benchmark \
  --run-id "${RUN_ID}" \
  --seed "${SEED}" \
  --calibration-quantile "${QUANTILE}" \
  --skab-dir "${PROJECT_ROOT}/data/external/skab" \
  --tsb-dir "${PROJECT_ROOT}/data/external/tsb_ad_m" \
  --tsb-max-per-domain "${TSB_MAX_PER_DOMAIN}" \
  --output-dir "${REPORT_DIR}"

log "M4 benchmark complete: ${REPORT_DIR}/benchmark_report.md"
