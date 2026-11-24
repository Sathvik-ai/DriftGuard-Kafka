#!/usr/bin/env bash
# End-to-end integration demo:
#   start Kafka -> check health -> create topic -> start consumer ->
#   run producer -> compare producer vs consumer statistics -> PASS/FAIL
#
# Usage: scripts/run_demo.sh [EVENT_COUNT]   (default: 1000)
#
# WARNING: the topic is deleted and recreated so that the consumer sees
# exactly the events of this run.
set -euo pipefail
source "$(dirname "$0")/common.sh"

EVENT_COUNT="${1:-1000}"
INTERVAL_MS="${DEMO_INTERVAL_MS:-2}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${PROJECT_ROOT}/runs/${RUN_ID}"
REPORT_DIR="${PROJECT_ROOT}/reports/runs/${RUN_ID}"
CONSUMER_GROUP="driftguard-m1-demo-${RUN_ID}"   # fresh group => reads from offset 0
KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"

[[ -x "${PYTHON_BIN}" ]] || fail "Python environment not found at ${PYTHON_BIN}. Run: make setup"
require_docker
mkdir -p "${RUN_DIR}"
cd "${PROJECT_ROOT}"

consumer_pid=""
cleanup() {
  if [[ -n "${consumer_pid}" ]] && kill -0 "${consumer_pid}" 2>/dev/null; then
    kill -TERM "${consumer_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

log "Step 1/6: starting Kafka"
compose up -d

log "Step 2/6: checking broker health"
"${PROJECT_ROOT}/scripts/verify_kafka.sh" --wait 120

log "Step 3/6: creating a clean topic"
"${PROJECT_ROOT}/scripts/create_topic.sh" --recreate

log "Step 4/6: starting consumer (group=${CONSUMER_GROUP})"
"${PYTHON_BIN}" -m app.consumer \
  --limit "${EVENT_COUNT}" --idle-timeout-s 60 --group "${CONSUMER_GROUP}" \
  --summary-file "${RUN_DIR}/consumer_summary.json" \
  --records-file "${RUN_DIR}/consumer_records.jsonl" \
  > "${RUN_DIR}/consumer.log" 2>&1 &
consumer_pid=$!

# Wait until the consumer has joined its group and owns partitions.
for _ in $(seq 1 30); do
  grep -q "partitions assigned" "${RUN_DIR}/consumer.log" && break
  kill -0 "${consumer_pid}" 2>/dev/null || { cat "${RUN_DIR}/consumer.log"; fail "Consumer exited early."; }
  sleep 1
done
grep -q "partitions assigned" "${RUN_DIR}/consumer.log" || fail "Consumer was not assigned partitions within 30 s."
grep "partitions assigned" "${RUN_DIR}/consumer.log"

log "Step 5/6: producing ${EVENT_COUNT} events (interval=${INTERVAL_MS} ms)"
producer_status=0
"${PYTHON_BIN}" -m app.producer \
  --count "${EVENT_COUNT}" --interval-ms "${INTERVAL_MS}" \
  --summary-file "${RUN_DIR}/producer_summary.json" \
  --records-file "${RUN_DIR}/producer_records.jsonl" \
  > "${RUN_DIR}/producer.log" 2>&1 || producer_status=$?
grep -E "producer (starting|delivery summary)|ERROR" "${RUN_DIR}/producer.log" || true

consumer_status=0
wait "${consumer_pid}" || consumer_status=$?
consumer_pid=""
grep -E "consumer (starting|receipt summary)|ERROR|WARNING" "${RUN_DIR}/consumer.log" || true
log "Producer exit code: ${producer_status}, consumer exit code: ${consumer_status}"

log "Step 6/6: verifying end-to-end results"
verify_status=0
"${PYTHON_BIN}" -m app.run_summary \
  --producer "${RUN_DIR}/producer_summary.json" \
  --consumer "${RUN_DIR}/consumer_summary.json" \
  --expected "${EVENT_COUNT}" > "${RUN_DIR}/verification.txt" 2>&1 || verify_status=$?
cat "${RUN_DIR}/verification.txt"

log "Run artefacts saved in ${RUN_DIR}"
if (( verify_status != 0 || consumer_status != 0 || producer_status != 0 )); then
  exit 1
fi

log "Step 7/7: generating streaming report in ${REPORT_DIR}"
"${PYTHON_BIN}" -m app.report \
  --run-id "${RUN_ID}" \
  --expected "${EVENT_COUNT}" \
  --topic "${KAFKA_TOPIC}" \
  --bootstrap-servers "${KAFKA_BOOTSTRAP_SERVERS}" \
  --producer-summary "${RUN_DIR}/producer_summary.json" \
  --consumer-summary "${RUN_DIR}/consumer_summary.json" \
  --producer-records "${RUN_DIR}/producer_records.jsonl" \
  --consumer-records "${RUN_DIR}/consumer_records.jsonl" \
  --output-dir "${REPORT_DIR}"
