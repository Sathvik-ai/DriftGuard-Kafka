#!/usr/bin/env bash
# End-to-end M3 integration demo across stable normal, gradual drift, and abrupt drift:
#   normal -> drift -> abrupt_drift
#
# Usage: scripts/run_m3_demo.sh [EVENT_COUNT_PER_SCENARIO]   (default: 500)
set -euo pipefail
source "$(dirname "$0")/common.sh"

EVENT_COUNT="${1:-500}"
INTERVAL_MS="${DEMO_INTERVAL_MS:-2}"
SEED="${DEMO_SEED:-42}"
ADWIN_DELTA="${ADWIN_DELTA:-0.002}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${PROJECT_ROOT}/runs/m3_${RUN_ID}"
REPORT_DIR="${PROJECT_ROOT}/reports/m3/${RUN_ID}"
RAW_TOPIC="${KAFKA_TOPIC:-research.telemetry.raw}"
DRIFT_TOPIC="${KAFKA_DRIFT_TOPIC:-research.telemetry.drift}"
SCENARIOS=(normal drift abrupt_drift)

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || [[ -x "${PYTHON_BIN}" ]] \
  || fail "Python environment not found at ${PYTHON_BIN}. Run: make setup"
require_docker
mkdir -p "${RUN_DIR}"
cd "${PROJECT_ROOT}"

monitor_pid=""
cleanup() {
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    kill -TERM "${monitor_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

log "Step 1/4: starting Kafka and checking broker health"
compose up -d
"${PROJECT_ROOT}/scripts/verify_kafka.sh" --wait 120

for scenario in "${SCENARIOS[@]}"; do
  SC_DIR="${RUN_DIR}/${scenario}"
  mkdir -p "${SC_DIR}"
  GROUP_ID="driftguard-m3-${scenario}-${RUN_ID}"
  VERIFY_GROUP_ID="driftguard-m3-verify-${scenario}-${RUN_ID}"

  log "Step 2/4 [${scenario}]: recreating raw & drift topics"
  "${PROJECT_ROOT}/scripts/create_topic.sh" --recreate --topic "${RAW_TOPIC}" >/dev/null
  "${PROJECT_ROOT}/scripts/create_topic.sh" --recreate --topic "${DRIFT_TOPIC}" >/dev/null

  log "Step 3/4 [${scenario}]: starting ADWIN monitor and producing ${EVENT_COUNT} events (seed=${SEED})"
  "${PYTHON_BIN}" -m app.drift_monitor \
    --limit "${EVENT_COUNT}" --idle-timeout-s 60 --group "${GROUP_ID}" \
    --delta "${ADWIN_DELTA}" \
    --summary-file "${SC_DIR}/monitor_summary.json" \
    --drift-file "${SC_DIR}/drift_events.jsonl" \
    --observations-file "${SC_DIR}/observations.jsonl" \
    > "${SC_DIR}/monitor.log" 2>&1 &
  monitor_pid=$!

  for _ in $(seq 1 30); do
    grep -q "partitions assigned" "${SC_DIR}/monitor.log" && break
    kill -0 "${monitor_pid}" 2>/dev/null || { cat "${SC_DIR}/monitor.log"; fail "ADWIN monitor exited early."; }
    sleep 1
  done
  grep -q "partitions assigned" "${SC_DIR}/monitor.log" || fail "ADWIN monitor did not get partitions."

  "${PYTHON_BIN}" -m app.producer \
    --count "${EVENT_COUNT}" --interval-ms "${INTERVAL_MS}" --seed "${SEED}" \
    --scenario "${scenario}" \
    --summary-file "${SC_DIR}/producer_summary.json" \
    --records-file "${SC_DIR}/producer_records.jsonl" \
    --ground-truth-file "${SC_DIR}/ground_truth.jsonl" \
    > "${SC_DIR}/producer.log" 2>&1

  wait "${monitor_pid}"
  monitor_pid=""

  expected_drifts="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${SC_DIR}/monitor_summary.json'))['drift_events_delivered'])")"
  "${PYTHON_BIN}" -m app.drift_monitor \
    --verify-drift "${expected_drifts}" \
    --group "${VERIFY_GROUP_ID}" \
    --summary-file "${SC_DIR}/drift_verified.json"
  log "Scenario '${scenario}' complete (drift_events_delivered=${expected_drifts})"
done

log "Step 4/4: generating M3 evaluation report and dashboard in ${REPORT_DIR}"
"${PYTHON_BIN}" -m app.m3_report \
  --run-id "${RUN_ID}" \
  --seed "${SEED}" \
  --delta "${ADWIN_DELTA}" \
  --run-dir "${RUN_DIR}" \
  --scenarios "${SCENARIOS[@]}" \
  --output-dir "${REPORT_DIR}"

cat "${REPORT_DIR}/report.md"
