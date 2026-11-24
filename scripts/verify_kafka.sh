#!/usr/bin/env bash
# Verify that the Kafka broker is running and healthy.
#
# Usage: scripts/verify_kafka.sh [--wait SECONDS]
#   --wait SECONDS  keep checking until healthy or the timeout expires (default: 0)
set -euo pipefail
source "$(dirname "$0")/common.sh"

wait_seconds=0
if [[ "${1:-}" == "--wait" ]]; then
  wait_seconds="${2:?--wait requires a number of seconds}"
fi

require_docker

container_id="$(compose ps -q "${KAFKA_SERVICE}")"
[[ -n "${container_id}" ]] || fail "Kafka container is not running. Start it with: make kafka-up"

deadline=$(( $(date +%s) + wait_seconds ))
while true; do
  health="$(docker inspect --format '{{.State.Health.Status}}' "${container_id}")"
  [[ "${health}" == "healthy" ]] && break
  if (( $(date +%s) >= deadline )); then
    fail "Kafka health status is '${health}'. Inspect logs with: docker compose logs kafka"
  fi
  log "Waiting for Kafka to become healthy (status: ${health})..."
  sleep 3
done
log "Container health: ${health}"

# Independent check: ask the broker for its supported API versions.
broker_line="$(kafka_cli kafka-broker-api-versions.sh 2>/dev/null | head -n 1)" \
  || fail "Broker did not answer an API versions request."
log "Broker reachable: ${broker_line%% *}"

log "Topics on broker:"
kafka_cli kafka-topics.sh --list | sed 's/^/  - /'
log "Kafka verification: OK"
