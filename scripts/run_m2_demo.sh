#!/usr/bin/env bash
# End-to-end M2 integration demo across all three scenarios:
#   normal -> anomalies (spikes + stuck sensors) -> drift
#
# Usage: scripts/run_m2_demo.sh [EVENT_COUNT_PER_SCENARIO]   (default: 300)
set -euo pipefail
source "$(dirname "$0")/common.sh"

EVENT_COUNT="${1:-300}"
INTERVAL_MS="${DEMO_INTERVAL_MS:-2}"
SEED="${DEMO_SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${PROJECT_ROOT}/runs/m2_${RUN_ID}"
REPORT_DIR="${PROJECT_ROOT}/reports/m2/${RUN_ID}"
RAW_TOPIC="${KAFKA_TOPIC:-research.telemetry.raw}"
ALERTS_TOPIC="${KAFKA_ALERTS_TOPIC:-research.telemetry.alerts}"
SCENARIOS=(normal anomalies drift)

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || [[ -x "${PYTHON_BIN}" ]] \
  || fail "Python environment not found at ${PYTHON_BIN}. Run: make setup"
require_docker
mkdir -p "${RUN_DIR}"
cd "${PROJECT_ROOT}"

detector_pid=""
cleanup() {
  if [[ -n "${detector_pid}" ]] && kill -0 "${detector_pid}" 2>/dev/null; then
    kill -TERM "${detector_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

log "Step 1/4: starting Kafka and checking broker health"
compose up -d
"${PROJECT_ROOT}/scripts/verify_kafka.sh" --wait 120

for scenario in "${SCENARIOS[@]}"; do
  SC_DIR="${RUN_DIR}/${scenario}"
  mkdir -p "${SC_DIR}"
  GROUP_ID="driftguard-m2-${scenario}-${RUN_ID}"
  ALERTS_GROUP_ID="driftguard-m2-alerts-${scenario}-${RUN_ID}"

  log "Step 2/4 [${scenario}]: recreating raw & alerts topics"
  "${PROJECT_ROOT}/scripts/create_topic.sh" --recreate --topic "${RAW_TOPIC}" >/dev/null
  "${PROJECT_ROOT}/scripts/create_topic.sh" --recreate --topic "${ALERTS_TOPIC}" >/dev/null

  log "Step 3/4 [${scenario}]: starting detector and producing ${EVENT_COUNT} events (seed=${SEED})"
  "${PYTHON_BIN}" -m app.detector \
    --limit "${EVENT_COUNT}" --idle-timeout-s 60 --group "${GROUP_ID}" \
    --summary-file "${SC_DIR}/detector_summary.json" \
    --detections-file "${SC_DIR}/detections.jsonl" \
    > "${SC_DIR}/detector.log" 2>&1 &
  detector_pid=$!

  for _ in $(seq 1 30); do
    grep -q "partitions assigned" "${SC_DIR}/detector.log" && break
    kill -0 "${detector_pid}" 2>/dev/null || { cat "${SC_DIR}/detector.log"; fail "Detector exited early."; }
    sleep 1
  done
  grep -q "partitions assigned" "${SC_DIR}/detector.log" || fail "Detector did not get partitions."

  "${PYTHON_BIN}" -m app.producer \
    --count "${EVENT_COUNT}" --interval-ms "${INTERVAL_MS}" --seed "${SEED}" \
    --scenario "${scenario}" \
    --summary-file "${SC_DIR}/producer_summary.json" \
    --records-file "${SC_DIR}/producer_records.jsonl" \
    --ground-truth-file "${SC_DIR}/ground_truth.jsonl" \
    > "${SC_DIR}/producer.log" 2>&1

  wait "${detector_pid}"
  detector_pid=""

  expected_alerts="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${SC_DIR}/detector_summary.json'))['alerts_delivered'])")"
  "${PYTHON_BIN}" -m app.detector \
    --verify-alerts "${expected_alerts}" \
    --group "${ALERTS_GROUP_ID}" \
    --summary-file "${SC_DIR}/alerts_verified.json"
  log "Scenario '${scenario}' complete (alerts_delivered=${expected_alerts})"
done

log "Step 4/4: generating M2 evaluation report in ${REPORT_DIR}"
"${PYTHON_BIN}" -m app.m2_report \
  --run-id "${RUN_ID}" \
  --seed "${SEED}" \
  --run-dir "${RUN_DIR}" \
  --scenarios "${SCENARIOS[@]}" \
  --output-dir "${REPORT_DIR}"

cat "${REPORT_DIR}/report.md"
