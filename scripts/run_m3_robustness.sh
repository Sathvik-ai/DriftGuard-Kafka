#!/usr/bin/env bash
# Execute 10-seed M3 robustness experiments across normal, drift, and abrupt_drift
# over the live Apache Kafka broker and generate a consolidated Markdown/JSON report.
#
# Usage: scripts/run_m3_robustness.sh [EVENT_COUNT_PER_RUN]   (default: 500)
set -euo pipefail
source "$(dirname "$0")/common.sh"

EVENT_COUNT="${1:-500}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${PROJECT_ROOT}/reports/m3/robustness_${RUN_ID}"
RAW_TOPIC="${KAFKA_TOPIC:-research.telemetry.raw}"
DRIFT_TOPIC="${KAFKA_DRIFT_TOPIC:-research.telemetry.drift}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || [[ -x "${PYTHON_BIN}" ]] \
  || fail "Python environment not found at ${PYTHON_BIN}. Run: make setup"
require_docker
cd "${PROJECT_ROOT}"

log "Step 1/3: ensuring Kafka broker is running and healthy"
compose up -d
"${PROJECT_ROOT}/scripts/verify_kafka.sh" --wait 120

log "Step 2/3: recreating clean raw & drift topics for 10-seed robustness sweep"
"${PROJECT_ROOT}/scripts/create_topic.sh" --recreate --topic "${RAW_TOPIC}" >/dev/null
"${PROJECT_ROOT}/scripts/create_topic.sh" --recreate --topic "${DRIFT_TOPIC}" >/dev/null

log "Step 3/3: running 10 seeds x 3 scenarios (${EVENT_COUNT} events/run) over Kafka"
"${PYTHON_BIN}" -m app.m3_robustness \
  --run-id "${RUN_ID}" \
  --count "${EVENT_COUNT}" \
  --mode kafka \
  --output-dir "${REPORT_DIR}"

log "Consolidated M3 robustness report saved in ${REPORT_DIR}"
