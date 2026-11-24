#!/usr/bin/env bash
# Shared helpers for the scripts in this directory. Source it, don't run it.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT

# Load .env (if present) so scripts honour the same settings as the Python apps.
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

KAFKA_TOPIC="${KAFKA_TOPIC:-research.telemetry.raw}"
KAFKA_TOPIC_PARTITIONS="${KAFKA_TOPIC_PARTITIONS:-3}"
KAFKA_TOPIC_REPLICATION_FACTOR="${KAFKA_TOPIC_REPLICATION_FACTOR:-1}"
KAFKA_SERVICE="kafka"
# CLI tools run inside the container, so they use the INTERNAL listener.
KAFKA_INTERNAL_BOOTSTRAP="kafka:19092"
KAFKA_BIN="/opt/kafka/bin"

log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
fail() { printf '[%s] ERROR: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; exit 1; }

require_docker() {
  command -v docker >/dev/null 2>&1 \
    || fail "Docker is not installed. Install Docker Desktop: https://docs.docker.com/desktop/setup/install/mac-install/"
  docker info >/dev/null 2>&1 \
    || fail "Docker daemon is not running. Start Docker Desktop (open -a Docker) and retry."
}

compose() {
  docker compose --project-directory "${PROJECT_ROOT}" -f "${PROJECT_ROOT}/docker-compose.yml" "$@"
}

# Run a Kafka CLI tool inside the broker container, e.g. kafka_cli kafka-topics.sh --list
kafka_cli() {
  local tool="$1"
  shift
  compose exec -T "${KAFKA_SERVICE}" "${KAFKA_BIN}/${tool}" --bootstrap-server "${KAFKA_INTERNAL_BOOTSTRAP}" "$@"
}
