#!/usr/bin/env bash
# Run DriftGuard-Kafka M8 zero-shot foundation model benchmark (Amazon Chronos-2
# and Google TimesFM 3.0) against M4–M7 models while preserving prior reports.
#
# Usage: scripts/run_m8_demo.sh [TSB_MAX_PER_DOMAIN]   (default: 1)
set -euo pipefail
source "$(dirname "$0")/common.sh"

TSB_MAX_PER_DOMAIN="${1:-1}"
CONFIG_PATH="${M8_CONFIG:-${PROJECT_ROOT}/configs/m8_tsfm.json}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${PROJECT_ROOT}/reports/m8/${RUN_ID}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || [[ -x "${PYTHON_BIN}" ]] \
  || fail "Python environment not found at ${PYTHON_BIN}. Run: make setup"

cd "${PROJECT_ROOT}"

REUSE_ARGS=()
if [[ -d "${PROJECT_ROOT}/reports/m7" ]]; then
  LATEST_M7="$(find "${PROJECT_ROOT}/reports/m7" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1 || true)"
  if [[ -n "${LATEST_M7}" && -f "${LATEST_M7}/m7_report.json" && -f "${LATEST_M7}/per_experiment_metrics.csv" ]]; then
    REUSE_ARGS=(--reuse-m7-dir "${LATEST_M7}")
    log "Detected verified M7 report bundle for split-consistent comparison: ${LATEST_M7}"
  fi
fi

LOCAL_ARGS=()
if [[ -d "${PROJECT_ROOT}/models/hf_cache" ]] && find "${PROJECT_ROOT}/models/hf_cache" -name "*.safetensors" | grep -q .; then
  LOCAL_ARGS=(--local-files-only)
fi

log "Running DriftGuard-Kafka M8 TSFM experiment (Chronos-2 + TimesFM 3.0, config=${CONFIG_PATH})"
"${PYTHON_BIN}" -m app.m8_benchmark \
  --config "${CONFIG_PATH}" \
  --run-id "${RUN_ID}" \
  --skab-dir "${PROJECT_ROOT}/data/external/skab" \
  --tsb-dir "${PROJECT_ROOT}/data/external/tsb_ad_m" \
  --tsb-max-per-domain "${TSB_MAX_PER_DOMAIN}" \
  --cache-dir "${PROJECT_ROOT}/models/hf_cache" \
  --output-dir "${REPORT_DIR}" \
  "${REUSE_ARGS[@]+"${REUSE_ARGS[@]}"}" \
  "${LOCAL_ARGS[@]+"${LOCAL_ARGS[@]}"}"

log "M8 experiment complete: ${REPORT_DIR}/m8_report.md"
