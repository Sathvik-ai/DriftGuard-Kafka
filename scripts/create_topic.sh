#!/usr/bin/env bash
# Create the telemetry topic and verify its partition count.
#
# Usage: scripts/create_topic.sh [--recreate] [--topic TOPIC]
#   --recreate  delete the topic first (DESTROYS its data) for a clean run
#   --topic     override KAFKA_TOPIC from .env
set -euo pipefail
source "$(dirname "$0")/common.sh"

recreate=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate) recreate=true; shift ;;
    --topic) KAFKA_TOPIC="${2:?--topic requires a topic name}"; shift 2 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

require_docker

topic_exists() {
  kafka_cli kafka-topics.sh --list | grep -Fxq "${KAFKA_TOPIC}"
}

if [[ "${recreate}" == "true" ]] && topic_exists; then
  log "Deleting topic '${KAFKA_TOPIC}' for a clean run..."
  kafka_cli kafka-topics.sh --delete --topic "${KAFKA_TOPIC}"
  # Deletion is asynchronous; wait until the broker no longer lists the topic.
  for _ in $(seq 1 30); do
    topic_exists || break
    sleep 1
  done
  topic_exists && fail "Topic '${KAFKA_TOPIC}' still exists after deletion request."
fi

log "Creating topic '${KAFKA_TOPIC}' (partitions=${KAFKA_TOPIC_PARTITIONS}, replication-factor=${KAFKA_TOPIC_REPLICATION_FACTOR})..."
# Retry briefly: right after a deletion the broker may still be cleaning up.
created=false
for _ in $(seq 1 10); do
  if kafka_cli kafka-topics.sh --create --if-not-exists \
      --topic "${KAFKA_TOPIC}" \
      --partitions "${KAFKA_TOPIC_PARTITIONS}" \
      --replication-factor "${KAFKA_TOPIC_REPLICATION_FACTOR}"; then
    created=true
    break
  fi
  sleep 2
done
[[ "${created}" == "true" ]] || fail "Could not create topic '${KAFKA_TOPIC}'."

description="$(kafka_cli kafka-topics.sh --describe --topic "${KAFKA_TOPIC}")"
echo "${description}"

actual_partitions="$(echo "${description}" | grep -c 'Partition: ' || true)"
if [[ "${actual_partitions}" != "${KAFKA_TOPIC_PARTITIONS}" ]]; then
  fail "Topic has ${actual_partitions} partitions, expected ${KAFKA_TOPIC_PARTITIONS}. Recreate with: scripts/create_topic.sh --recreate"
fi
log "Topic '${KAFKA_TOPIC}' ready with ${actual_partitions} partitions."
