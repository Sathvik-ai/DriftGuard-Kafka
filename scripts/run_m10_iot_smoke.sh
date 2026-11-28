#!/usr/bin/env bash
# Live M10-IoT end-to-end integration smoke test:
#   Simulated Wokwi ESP32 MQTT publisher -> Mosquitto MQTT broker (Docker)
#   -> app.mqtt_bridge -> research.telemetry.raw (Kafka)
#   -> existing app.detector -> research.telemetry.alerts (Kafka)
set -euo pipefail
source "$(dirname "$0")/common.sh"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${PROJECT_ROOT}/reports/m10_iot/smoke_${RUN_ID}"
MQTT_CONTAINER="driftguard-m10-mosquitto-smoke"
MQTT_PORT="${M10_SMOKE_MQTT_PORT:-18830}"
MQTT_TOPIC="driftguard/smoke_${RUN_ID}/esp32-wokwi-01/telemetry"

cleanup() {
  docker rm -f "${MQTT_CONTAINER}" >/dev/null 2>&1 || true
  if [[ -n "${BRIDGE_PID:-}" ]]; then kill "${BRIDGE_PID}" >/dev/null 2>&1 || true; fi
  if [[ -n "${DETECTOR_PID:-}" ]]; then kill "${DETECTOR_PID}" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

cd "${PROJECT_ROOT}"
mkdir -p "${OUT_DIR}"

log "1/5 Ensuring local Kafka broker and topics are ready..."
docker compose -f "${PROJECT_ROOT}/docker-compose.yml" up -d
"${PROJECT_ROOT}/scripts/verify_kafka.sh"
"${PROJECT_ROOT}/scripts/create_topic.sh" --recreate

log "2/5 Starting ephemeral Mosquitto MQTT relay broker on localhost:${MQTT_PORT}..."
docker rm -f "${MQTT_CONTAINER}" >/dev/null 2>&1 || true
MOSQ_CONF="${OUT_DIR}/mosquitto.conf"
cat > "${MOSQ_CONF}" <<'EOF'
listener 1883 0.0.0.0
allow_anonymous true
EOF
docker run -d --rm --name "${MQTT_CONTAINER}" \
  -p "${MQTT_PORT}:1883" \
  -v "${MOSQ_CONF}:/mosquitto/config/mosquitto.conf:ro" \
  eclipse-mosquitto:2 >/dev/null
sleep 2

log "3/5 Starting existing M2 detector on research.telemetry.raw -> research.telemetry.alerts..."
"${PYTHON_BIN}" -m app.detector \
  --group "driftguard-m10-smoke-det-${RUN_ID}" \
  --limit 6 \
  --summary-file "${OUT_DIR}/detector_summary.json" \
  >"${OUT_DIR}/detector.log" 2>&1 &
DETECTOR_PID=$!

log "4/5 Starting M10-IoT MQTT-to-Kafka bridge..."
MQTT_BROKER_HOST="127.0.0.1" \
MQTT_BROKER_PORT="${MQTT_PORT}" \
MQTT_USE_TLS="false" \
MQTT_TOPIC="driftguard/smoke_${RUN_ID}/+/telemetry" \
MQTT_CLIENT_ID="driftguard-m10-smoke-bridge-${RUN_ID}" \
"${PYTHON_BIN}" -m app.mqtt_bridge \
  --max-messages 7 \
  --duration-s 20 \
  --run-id "smoke_${RUN_ID}" \
  --status-dir "${OUT_DIR}" \
  >"${OUT_DIR}/bridge.log" 2>&1 &
BRIDGE_PID=$!
sleep 2

log "5/5 Publishing live virtual ESP32 MQTT telemetry (normal + temperature_spike + seq gap + malformed)..."
"${PYTHON_BIN}" - "${MQTT_PORT}" "${MQTT_TOPIC}" <<'PY'
import json, sys, time
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

port = int(sys.argv[1])
topic = sys.argv[2]
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id="wokwi-smoke-pub")
client.connect("127.0.0.1", port, keepalive=30)
client.loop_start()
time.sleep(0.5)

def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

# 4 normal readings (seq 1..4), 1 malformed JSON, 2 temperature_spike readings (seq 7..8 -> gap of 2!)
payloads = [
    json.dumps({"device_id": "esp32-wokwi-01", "machine_id": "machine_01", "sequence_number": 1, "timestamp": ts(), "temperature_c": 46.2, "vibration_mm_s": 2.1, "pressure_bar": 5.0}),
    json.dumps({"device_id": "esp32-wokwi-01", "machine_id": "machine_01", "sequence_number": 2, "timestamp": ts(), "temperature_c": 46.8, "vibration_mm_s": 2.0, "pressure_bar": 5.1}),
    json.dumps({"device_id": "esp32-wokwi-01", "machine_id": "machine_01", "sequence_number": 3, "timestamp": ts(), "temperature_c": 46.5, "vibration_mm_s": 2.2, "pressure_bar": 5.0}),
    json.dumps({"device_id": "esp32-wokwi-01", "machine_id": "machine_01", "sequence_number": 4, "timestamp": ts(), "temperature_c": 46.9, "vibration_mm_s": 2.1, "pressure_bar": 4.9}),
    "{malformed_json_payload",
    json.dumps({"device_id": "esp32-wokwi-01", "machine_id": "machine_01", "sequence_number": 7, "timestamp": ts(), "temperature_c": 94.5, "vibration_mm_s": 2.2, "pressure_bar": 5.0}),
    json.dumps({"device_id": "esp32-wokwi-01", "machine_id": "machine_01", "sequence_number": 8, "timestamp": ts(), "temperature_c": 96.2, "vibration_mm_s": 2.3, "pressure_bar": 5.1}),
]
for p in payloads:
    client.publish(topic, p, qos=1).wait_for_publish(timeout=5)
    time.sleep(0.15)
client.loop_stop()
client.disconnect()
PY

wait "${BRIDGE_PID}"
wait "${DETECTOR_PID}"

log "Verifying live MQTT-to-Kafka-to-Detector results..."
"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
iot = json.loads((out / "iot_status.json").read_text(encoding="utf-8"))
det = json.loads((out / "detector_summary.json").read_text(encoding="utf-8"))

assert iot["messages_received"] == 7, iot
assert iot["messages_forwarded"] == 6, iot
assert iot["invalid_messages"] == 1, iot
assert iot["sequence_gaps"] == 2, iot
assert det["received"] == 6, det
assert det["valid"] == 6, det
assert det["alerts_triggered"] >= 2, det
assert det["alerts_delivered"] == det["alerts_triggered"], det
print(
    f"M10-IoT LIVE SMOKE PASS: mqtt_received={iot['messages_received']} "
    f"kafka_forwarded={iot['messages_forwarded']} invalid={iot['invalid_messages']} "
    f"seq_gaps={iot['sequence_gaps']} detector_valid={det['valid']} "
    f"alerts_triggered={det['alerts_triggered']} report={out / 'iot_status.md'}"
)
PY
