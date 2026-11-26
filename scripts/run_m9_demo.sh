#!/usr/bin/env bash
# Run DriftGuard-Kafka M9 comprehensive multi-seed comparative evaluation
# across all M4–M8 models without altering historical M1–M8 reports.
#
# Usage: scripts/run_m9_demo.sh [TSB_MAX_PER_DOMAIN]   (default: 1)
set -euo pipefail
source "$(dirname "$0")/common.sh"

TSB_MAX_PER_DOMAIN="${1:-1}"
CONFIG_PATH="${M9_CONFIG:-${PROJECT_ROOT}/configs/m9_eval.json}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${PROJECT_ROOT}/reports/m9/${RUN_ID}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || [[ -x "${PYTHON_BIN}" ]] \
  || fail "Python environment not found at ${PYTHON_BIN}. Run: make setup"

cd "${PROJECT_ROOT}"

LOCAL_ARGS=()
if [[ -d "${PROJECT_ROOT}/models/hf_cache" ]] && find "${PROJECT_ROOT}/models/hf_cache" -name "*.safetensors" | grep -q .; then
  LOCAL_ARGS=(--local-files-only)
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

log "Running DriftGuard-Kafka M9 comprehensive multi-seed evaluation (config=${CONFIG_PATH})"
"${PYTHON_BIN}" -m app.m9_evaluation \
  --config "${CONFIG_PATH}" \
  --run-id "${RUN_ID}" \
  --skab-dir "${PROJECT_ROOT}/data/external/skab" \
  --tsb-dir "${PROJECT_ROOT}/data/external/tsb_ad_m" \
  --tsb-max-per-domain "${TSB_MAX_PER_DOMAIN}" \
  --cache-dir "${PROJECT_ROOT}/models/hf_cache" \
  --output-dir "${REPORT_DIR}" \
  "${LOCAL_ARGS[@]+"${LOCAL_ARGS[@]}"}"

log "M9 evaluation complete: ${REPORT_DIR}/m9_report.md"
