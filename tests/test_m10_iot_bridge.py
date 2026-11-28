"""Unit tests for DriftGuard-Kafka M10-IoT MQTT-to-Kafka bridge and schema."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from app.config import ConfigError, load_settings
from app.detector import SensorReading
from app.mqtt_bridge import (
    MqttBridgeConfig,
    MqttKafkaBridge,
    convert_mqtt_to_telemetry_event,
    load_mqtt_config,
    parse_mqtt_iot_payload,
)
from app.schemas import (
    EventLabel,
    SchemaValidationError,
    SourceType,
    parse_event_json,
)


class _FakeKafkaMessage:
    def __init__(self, key: bytes | None, partition: int = 0) -> None:
        self._key = key
        self._partition = partition

    def key(self) -> bytes | None:
        return self._key

    def partition(self) -> int:
        return self._partition


class _FakeKafkaProducer:
    def __init__(self) -> None:
        self.produced: list[tuple[str, bytes, bytes]] = []
        self._pending: list[tuple[Any, _FakeKafkaMessage]] = []

    def produce(
        self,
        topic: str,
        key: bytes,
        value: bytes,
        on_delivery: Any = None,
    ) -> None:
        self.produced.append((topic, key, value))
        if on_delivery is not None:
            part = hash(key) % 3
            self._pending.append((on_delivery, _FakeKafkaMessage(key, partition=part)))

    def poll(self, timeout: float = 0.0) -> int:
        del timeout
        delivered = len(self._pending)
        while self._pending:
            cb, msg = self._pending.pop(0)
            cb(None, msg)
        return delivered

    def flush(self, timeout: float = 10.0) -> int:
        del timeout
        self.poll(0.0)
        return 0


def _sample_mqtt_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "device_id": "esp32-wokwi-01",
        "machine_id": "machine_01",
        "sequence_number": 1,
        "timestamp": "2026-11-28T10:00:00.000Z",
        "temperature_c": 46.5,
        "vibration_mm_s": 2.1,
        "pressure_bar": 5.0,
    }
    data.update(overrides)
    return data


def _sample_bridge(tmp_path: Path) -> tuple[MqttKafkaBridge, _FakeKafkaProducer]:
    cfg = MqttBridgeConfig(
        broker_host="broker.example.test",
        broker_port=8883,
        use_tls=True,
        username="iot-user",
        password="super-secret-password",  # noqa: S106 - unit test fixture
        mqtt_topic="driftguard/test/+/telemetry",
        mqtt_client_id="driftguard-test-bridge",
        qos=1,
        status_report_dir=tmp_path,
    )
    kafka_settings = load_settings({})
    fake_prod = _FakeKafkaProducer()
    return MqttKafkaBridge(cfg, kafka_settings, kafka_producer=fake_prod), fake_prod


def test_mqtt_payload_validation_and_schema_conversion() -> None:
    raw = json.dumps(_sample_mqtt_dict(sequence_number=7)).encode("utf-8")
    parsed = parse_mqtt_iot_payload(raw)
    assert parsed.device_id == "esp32-wokwi-01"
    assert parsed.machine_id == "machine_01"
    assert parsed.sequence_number == 7

    event = convert_mqtt_to_telemetry_event(parsed)
    assert event.source_type is SourceType.VIRTUAL_IOT_WOKWI
    assert event.label is EventLabel.NORMAL
    assert event.device_id == "esp32-wokwi-01"
    assert event.sequence_number == 7

    # Round-trips through existing DriftGuard Kafka schema & strips label in SensorReading
    wire = event.to_json_bytes()
    decoded = json.loads(wire)
    assert decoded["source_type"] == "virtual_iot_wokwi"
    assert decoded["device_id"] == "esp32-wokwi-01"
    assert decoded["sequence_number"] == 7
    restored = parse_event_json(wire)
    assert restored == event
    reading = SensorReading.from_telemetry(restored)
    assert reading.machine_id == "machine_01"
    assert not hasattr(reading, "label")


@pytest.mark.parametrize(
    "bad_payload",
    [
        None,
        b"",
        b"\xff\xfe",
        b"{not json",
        b"[]",
        json.dumps({"device_id": "esp32-01"}).encode(),
        json.dumps(_sample_mqtt_dict(extra="forbidden")).encode(),
        json.dumps(_sample_mqtt_dict(temperature_c=999.0)).encode(),
        json.dumps(_sample_mqtt_dict(machine_id="bad_machine")).encode(),
        json.dumps(_sample_mqtt_dict(device_id="bad device with spaces")).encode(),
        json.dumps(_sample_mqtt_dict(sequence_number=-1)).encode(),
        json.dumps(_sample_mqtt_dict(sequence_number=True)).encode(),
        json.dumps(_sample_mqtt_dict(timestamp="2026-11-28 10:00:00")).encode(),
    ],
)
def test_malformed_mqtt_payloads_are_rejected(bad_payload: bytes | None) -> None:
    with pytest.raises(SchemaValidationError):
        parse_mqtt_iot_payload(bad_payload)


def test_bridge_forwarding_sequence_gaps_reconnects_and_status_report(tmp_path: Path) -> None:
    bridge, prod = _sample_bridge(tmp_path)

    # Initial connect -> disconnect -> reconnect increments reconnection_count to 1
    bridge.handle_connect(0)
    assert bridge.connection_state == "connected"
    assert bridge.reconnection_count == 0
    bridge.handle_disconnect(1)
    assert bridge.connection_state == "reconnecting"
    bridge.handle_connect(0)
    assert bridge.connection_state == "connected"
    assert bridge.reconnection_count == 1

    base_dt = datetime(2026, 11, 28, 10, 0, 0, tzinfo=timezone.utc)

    # 1. Valid seq=1 (50ms ingestion latency)
    ev1 = bridge.process_mqtt_message(
        "driftguard/test/esp32-wokwi-01/telemetry",
        json.dumps(_sample_mqtt_dict(sequence_number=1, timestamp="2026-11-28T10:00:00.000Z")),
        received_utc=base_dt + timedelta(milliseconds=50),
    )
    assert ev1 is not None

    # 2. Malformed JSON rejected cleanly
    bad = bridge.process_mqtt_message(
        "driftguard/test/esp32-wokwi-01/telemetry",
        b"{corrupted",
        received_utc=base_dt + timedelta(milliseconds=100),
    )
    assert bad is None

    # 3. Valid seq=4 (skips seq 2 and 3 -> sequence_gaps == 2)
    ev4 = bridge.process_mqtt_message(
        "driftguard/test/esp32-wokwi-01/telemetry",
        json.dumps(_sample_mqtt_dict(sequence_number=4, timestamp="2026-11-28T10:00:02.000Z")),
        received_utc=base_dt + timedelta(seconds=2, milliseconds=120),
    )
    assert ev4 is not None

    # 4. Duplicate / out-of-order seq=4
    ev4_dup = bridge.process_mqtt_message(
        "driftguard/test/esp32-wokwi-01/telemetry",
        json.dumps(_sample_mqtt_dict(sequence_number=4, timestamp="2026-11-28T10:00:03.000Z")),
        received_utc=base_dt + timedelta(seconds=3, milliseconds=30),
    )
    assert ev4_dup is not None

    assert bridge.flush() == 0
    assert bridge.messages_received == 4
    assert bridge.messages_forwarded == 3
    assert bridge.invalid_messages == 1
    assert bridge.sequence_gaps_total == 2
    assert len(prod.produced) == 3
    # Kafka partition key is always `machine_id`
    assert all(k == b"machine_01" for _t, k, _v in prod.produced)

    out_dir = tmp_path / "status_out"
    bridge.write_status_report(out_dir, run_id="test_run")
    rep_json = json.loads((out_dir / "iot_status.json").read_text(encoding="utf-8"))
    rep_md = (out_dir / "iot_status.md").read_text(encoding="utf-8")

    assert rep_json["messages_received"] == 4
    assert rep_json["messages_forwarded"] == 3
    assert rep_json["invalid_messages"] == 1
    assert rep_json["sequence_gaps"] == 2
    assert rep_json["reconnection_count"] == 1
    assert rep_json["ingestion_latency_ms"]["count"] == 3
    assert rep_json["devices"]["esp32-wokwi-01"]["out_of_order_or_duplicate"] == 1
    # Secrets MUST NEVER appear in status report
    assert "super-secret-password" not in json.dumps(rep_json)
    assert "super-secret-password" not in rep_md
    assert "DriftGuard-Kafka M10-IoT Device Status Report" in rep_md


def test_load_mqtt_config_validation() -> None:
    cfg = load_mqtt_config(
        {
            "MQTT_BROKER_HOST": "broker.hivemq.cloud",
            "MQTT_BROKER_PORT": "8883",
            "MQTT_USE_TLS": "true",
            "MQTT_USERNAME": "u1",
            "MQTT_PASSWORD": "p1",
            "MQTT_TOPIC": "driftguard/ns1/+/telemetry",
            "MQTT_QOS": "1",
        }
    )
    assert cfg.broker_host == "broker.hivemq.cloud"
    assert cfg.broker_port == 8883
    assert cfg.use_tls is True
    assert cfg.sanitized_dict()["password_configured"] is True
    assert "p1" not in json.dumps(cfg.sanitized_dict())

    with pytest.raises(ConfigError):
        load_mqtt_config({"MQTT_BROKER_PORT": "99999"})
    with pytest.raises(ConfigError):
        load_mqtt_config({"MQTT_QOS": "3"})
