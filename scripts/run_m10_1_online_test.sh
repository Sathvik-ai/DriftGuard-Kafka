#!/usr/bin/env bash
# Run M10.1 continuous online multi-phase virtual IoT pipeline validation.
# Usage: ./scripts/run_m10_1_online_test.sh [duration_scale]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

SCALE="${1:-1.0}"
if [ "$#" -gt 0 ]; then
  shift
fi
echo "Starting DriftGuard-Kafka M10.1 online test (duration_scale=${SCALE})..."
exec "${PYTHON_BIN}" -m app.m10_1_online_runner --duration-scale "${SCALE}" "$@"
