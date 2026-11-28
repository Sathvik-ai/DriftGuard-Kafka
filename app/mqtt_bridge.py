"""Python MQTT-to-Kafka Bridge & IoT Device Status Monitor (DriftGuard-Kafka M10-IoT).

Architecture:
    Wokwi ESP32 (or virtual IoT device) -> Reachable MQTT Broker ->
    `app.mqtt_bridge` (this module) -> existing `research.telemetry.raw` Kafka topic ->
    existing `app.consumer`, `app.detector`, `app.drift_monitor`, `app.live_dashboard`.

Key guarantees:
  - Reuses `app.producer.build_producer_config`, `app.producer.DeliveryTracker`,
    and `app.schemas.TelemetryEvent` (preserving `machine_id` Kafka partitioning
    and `source_type="virtual_iot_wokwi"` provenance).
  - Never hardcodes MQTT credentials or logs passwords.
  - Validates every incoming MQTT JSON payload, rejecting malformed/out-of-range
    messages cleanly without crashing the subscriber loop.
  - Tracks per-device connection state, last message timestamp, received/forwarded/
    invalid counts, sequence gaps, reconnections, and ingestion latency, writing
    reproducible JSON and Markdown IoT status reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import ssl
import threading
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import paho.mqtt.client as mqtt
from confluent_kafka import Producer
from paho.mqtt.enums import CallbackAPIVersion

from app.config import ConfigError, Settings, load_settings
from app.logging_config import configure_logging
from app.producer import DeliveryTracker, build_producer_config
from app.schemas import (
    EventLabel,
    SchemaValidationError,
    SourceType,
    TelemetryEvent,
    _validate_device_id,
    _validate_machine_id,
    _validate_sensor_value,
    _validate_sequence_number,
    _validate_timestamp,
    format_utc_timestamp,
)

logger = logging.getLogger(__name__)

VALID_DEVICE_PROFILES: frozenset[str] = frozenset(
    {
        "normal",
        "temperature_spike",
        "vibration_anomaly",
        "gradual_drift",
        "abrupt_drift",
        "intermittent_disconnect",
    }
)

MQTT_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "device_id",
        "machine_id",
        "sequence_number",
        "timestamp",
        "temperature_c",
        "vibration_mm_s",
        "pressure_bar",
    }
)


@dataclass(frozen=True)
class MqttBridgeConfig:
    """Runtime configuration for the MQTT-to-Kafka bridge (never hardcodes secrets)."""

    broker_host: str
    broker_port: int
    use_tls: bool
    username: str | None
    password: str | None
    mqtt_topic: str
    mqtt_client_id: str
    qos: int
    status_report_dir: Path
    source_type: SourceType = SourceType.VIRTUAL_IOT_WOKWI
    mqtt_namespace: str | None = None
    heartbeat_topic: str | None = None
    command_token: str | None = None
    publish_device_status_topic: bool = False
    heartbeat_timeout_s: float = 15.0

    def resolved_heartbeat_topic(self) -> str | None:
        """Return explicit or derived MQTT heartbeat subscription topic."""
        if self.heartbeat_topic:
            return self.heartbeat_topic
        if self.mqtt_namespace:
            return f"{self.mqtt_namespace.rstrip('/')}/+/heartbeat"
        if self.mqtt_topic.endswith("/telemetry"):
            return self.mqtt_topic[: -len("/telemetry")] + "/heartbeat"
        return None

    def sanitized_dict(self) -> dict[str, Any]:
        """Return JSON-safe config dictionary with credentials redacted."""
        return {
            "broker_host": self.broker_host,
            "broker_port": self.broker_port,
            "use_tls": self.use_tls,
            "username_configured": bool(self.username),
            "password_configured": bool(self.password),
            "command_token_configured": bool(self.command_token),
            "mqtt_topic": self.mqtt_topic,
            "heartbeat_topic": self.resolved_heartbeat_topic(),
            "mqtt_client_id": self.mqtt_client_id,
            "qos": self.qos,
            "source_type": self.source_type.value,
            "publish_device_status_topic": self.publish_device_status_topic,
        }


def load_mqtt_config(env: Mapping[str, str] | None = None) -> MqttBridgeConfig:
    """Load MQTT bridge settings from environment variables."""
    src: Mapping[str, str] = os.environ if env is None else env
    host = src.get("MQTT_BROKER_HOST", "localhost").strip()
    if not host:
        raise ConfigError("MQTT_BROKER_HOST must not be empty")

    raw_port = src.get("MQTT_BROKER_PORT", "1883").strip()
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ConfigError(f"MQTT_BROKER_PORT must be an integer, got {raw_port!r}") from exc
    if not 1 <= port <= 65535:
        raise ConfigError(f"MQTT_BROKER_PORT must be in [1, 65535], got {port}")

    raw_tls = src.get("MQTT_USE_TLS", "false").strip().lower()
    use_tls = raw_tls in ("1", "true", "yes", "on")

    username = src.get("MQTT_USERNAME", "").strip() or None
    password = src.get("MQTT_PASSWORD", "") or None

    topic = src.get("MQTT_TOPIC", "driftguard/demo/+/telemetry").strip()
    if not topic:
        raise ConfigError("MQTT_TOPIC must not be empty")

    client_id = (
        src.get("MQTT_CLIENT_ID", "driftguard-m10-bridge").strip() or "driftguard-m10-bridge"
    )

    raw_qos = src.get("MQTT_QOS", "1").strip()
    try:
        qos = int(raw_qos)
    except ValueError as exc:
        raise ConfigError(f"MQTT_QOS must be 0, 1, or 2, got {raw_qos!r}") from exc
    if qos not in (0, 1, 2):
        raise ConfigError(f"MQTT_QOS must be 0, 1, or 2, got {qos}")

    report_dir = Path(src.get("MQTT_STATUS_REPORT_DIR", "reports/m10_iot").strip())
    namespace = src.get("MQTT_NAMESPACE", "").strip() or None
    hb_topic = src.get("MQTT_HEARTBEAT_TOPIC", "").strip() or None
    cmd_token = src.get("MQTT_COMMAND_TOKEN", "").strip() or None
    raw_pub_status = src.get("MQTT_PUBLISH_DEVICE_STATUS", "false").strip().lower()
    pub_status = raw_pub_status in ("1", "true", "yes", "on")
    return MqttBridgeConfig(
        broker_host=host,
        broker_port=port,
        use_tls=use_tls,
        username=username,
        password=password,
        mqtt_topic=topic,
        mqtt_client_id=client_id,
        qos=qos,
        status_report_dir=report_dir,
        mqtt_namespace=namespace,
        heartbeat_topic=hb_topic,
        command_token=cmd_token,
        publish_device_status_topic=pub_status,
    )


@dataclass(frozen=True)
class MqttTelemetryPayload:
    """Validated raw payload published by a virtual ESP32 IoT device over MQTT."""

    device_id: str
    machine_id: str
    sequence_number: int
    timestamp: str
    temperature_c: float
    vibration_mm_s: float
    pressure_bar: float


def parse_mqtt_iot_payload(raw_payload: bytes | str | None) -> MqttTelemetryPayload:
    """Decode and strictly validate an incoming MQTT JSON telemetry payload."""
    if raw_payload is None or len(raw_payload) == 0:
        raise SchemaValidationError("MQTT payload is empty")
    try:
        decoded: Any = json.loads(raw_payload)
    except UnicodeDecodeError as exc:
        raise SchemaValidationError(f"MQTT payload is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"MQTT payload is not valid JSON: {exc}") from exc

    if not isinstance(decoded, dict):
        raise SchemaValidationError(
            f"MQTT payload must be a JSON object, got {type(decoded).__name__}"
        )

    missing = MQTT_REQUIRED_FIELDS - decoded.keys()
    if missing:
        raise SchemaValidationError(f"missing required MQTT fields: {sorted(missing)}")
    unexpected = decoded.keys() - MQTT_REQUIRED_FIELDS
    if unexpected:
        raise SchemaValidationError(f"unexpected MQTT fields: {sorted(unexpected)}")

    _validate_device_id(decoded["device_id"])
    _validate_machine_id(decoded["machine_id"])
    _validate_sequence_number(decoded["sequence_number"])
    _validate_timestamp(decoded["timestamp"])
    for sensor_field in ("temperature_c", "vibration_mm_s", "pressure_bar"):
        _validate_sensor_value(sensor_field, decoded[sensor_field])

    return MqttTelemetryPayload(
        device_id=str(decoded["device_id"]),
        machine_id=str(decoded["machine_id"]),
        sequence_number=int(decoded["sequence_number"]),
        timestamp=str(decoded["timestamp"]),
        temperature_c=float(decoded["temperature_c"]),
        vibration_mm_s=float(decoded["vibration_mm_s"]),
        pressure_bar=float(decoded["pressure_bar"]),
    )


def convert_mqtt_to_telemetry_event(
    payload: MqttTelemetryPayload,
    source_type: SourceType = SourceType.VIRTUAL_IOT_WOKWI,
    event_id: str | None = None,
) -> TelemetryEvent:
    """Convert a validated MQTT payload into the canonical DriftGuard `TelemetryEvent`."""
    return TelemetryEvent(
        event_id=event_id or str(uuid.uuid4()),
        machine_id=payload.machine_id,
        timestamp=payload.timestamp,
        temperature_c=payload.temperature_c,
        vibration_mm_s=payload.vibration_mm_s,
        pressure_bar=payload.pressure_bar,
        label=EventLabel.NORMAL,
        source_type=source_type,
        device_id=payload.device_id,
        sequence_number=payload.sequence_number,
    )


@dataclass(frozen=True)
class MqttHeartbeatPayload:
    """Validated device heartbeat or LWT offline status payload."""

    device_id: str
    machine_id: str
    status: str
    timestamp: str | None = None
    sequence_number: int | None = None
    active_profile: str | None = None
    wifi_rssi_dbm: int | None = None
    uptime_s: int | None = None
    reconnect_count: int = 0


def parse_mqtt_heartbeat_payload(raw_payload: bytes | str | None) -> MqttHeartbeatPayload:
    """Decode and validate an incoming MQTT heartbeat or LWT message."""
    if raw_payload is None or len(raw_payload) == 0:
        raise SchemaValidationError("MQTT heartbeat payload is empty")
    try:
        decoded: Any = json.loads(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"MQTT heartbeat payload is invalid JSON: {exc}") from exc

    if not isinstance(decoded, dict):
        raise SchemaValidationError("MQTT heartbeat payload must be a JSON object")
    if "device_id" not in decoded or "machine_id" not in decoded:
        raise SchemaValidationError("heartbeat missing device_id or machine_id")

    _validate_device_id(decoded["device_id"])
    _validate_machine_id(decoded["machine_id"])
    status = str(decoded.get("status", "online")).strip().lower()
    if status not in ("online", "offline"):
        raise SchemaValidationError(f"unsupported heartbeat status {status!r}")

    ts = decoded.get("timestamp")
    if ts is not None:
        _validate_timestamp(ts)

    seq = decoded.get("sequence_number")
    if seq is not None:
        _validate_sequence_number(seq)

    profile = decoded.get("active_profile")
    if profile is not None:
        profile_str = str(profile).strip().lower()
        if profile_str not in VALID_DEVICE_PROFILES:
            raise SchemaValidationError(f"unsupported active_profile {profile!r}")
        profile = profile_str

    rssi = decoded.get("wifi_rssi_dbm")
    uptime = decoded.get("uptime_s")
    reconn = decoded.get("reconnect_count", 0)
    if not isinstance(reconn, int) or isinstance(reconn, bool) or reconn < 0:
        raise SchemaValidationError("reconnect_count must be a non-negative integer")

    return MqttHeartbeatPayload(
        device_id=str(decoded["device_id"]),
        machine_id=str(decoded["machine_id"]),
        status=status,
        timestamp=str(ts) if ts is not None else None,
        sequence_number=int(seq) if seq is not None else None,
        active_profile=profile,
        wifi_rssi_dbm=int(rssi) if isinstance(rssi, int) and not isinstance(rssi, bool) else None,
        uptime_s=int(uptime) if isinstance(uptime, int) and not isinstance(uptime, bool) else None,
        reconnect_count=reconn,
    )


@dataclass
class DeviceHealthState:
    """Per-device sequence continuity, heartbeat liveness, latency, and counters."""

    device_id: str
    machine_id: str
    messages_received: int = 0
    last_sequence_number: int | None = None
    sequence_gaps: int = 0
    out_of_order_or_duplicate: int = 0
    last_message_timestamp: str | None = None
    last_received_utc: str | None = None
    ingestion_latencies_ms: list[float] = field(default_factory=list)
    heartbeat_count: int = 0
    last_heartbeat_timestamp: str | None = None
    last_heartbeat_received_utc: str | None = None
    active_profile_reported: str = "normal"
    device_reconnect_count: int = 0
    wifi_rssi_dbm: int | None = None
    uptime_s: int | None = None
    lwt_offline: bool = False

    def record_heartbeat(self, hb: MqttHeartbeatPayload, received_utc: datetime) -> None:
        """Update per-device heartbeat/LWT status from verified MQTT message."""
        self.machine_id = hb.machine_id
        if hb.status == "offline":
            self.lwt_offline = True
            return
        self.lwt_offline = False
        self.heartbeat_count += 1
        self.last_heartbeat_received_utc = format_utc_timestamp(received_utc)
        self.last_heartbeat_timestamp = hb.timestamp or self.last_heartbeat_received_utc
        if hb.active_profile is not None:
            self.active_profile_reported = hb.active_profile
        if hb.wifi_rssi_dbm is not None:
            self.wifi_rssi_dbm = hb.wifi_rssi_dbm
        if hb.uptime_s is not None:
            self.uptime_s = hb.uptime_s
        self.device_reconnect_count = max(self.device_reconnect_count, hb.reconnect_count)

    def liveness_state(self, now_utc: datetime | None = None, timeout_s: float = 15.0) -> str:
        """Compute explicit ONLINE / DELAYED / OFFLINE device state from real timestamps."""
        if self.lwt_offline:
            return "OFFLINE"
        ref_str = self.last_heartbeat_received_utc or self.last_received_utc
        if ref_str is None:
            return "OFFLINE"
        now = now_utc or datetime.now(timezone.utc)
        last_dt = datetime.fromisoformat(ref_str.replace("Z", "+00:00"))
        age_s = max(0.0, (now - last_dt).total_seconds())
        if age_s <= timeout_s:
            return "ONLINE"
        if age_s <= timeout_s * 3.0:
            return "DELAYED"
        return "OFFLINE"

    def record(self, payload: MqttTelemetryPayload, received_utc: datetime) -> int:
        """Record one validated message and return the number of skipped sequence IDs."""
        self.messages_received += 1
        self.machine_id = payload.machine_id
        self.last_message_timestamp = payload.timestamp
        self.last_received_utc = format_utc_timestamp(received_utc)

        gap = 0
        if self.last_sequence_number is not None:
            if payload.sequence_number > self.last_sequence_number + 1:
                gap = payload.sequence_number - self.last_sequence_number - 1
                self.sequence_gaps += gap
            elif payload.sequence_number <= self.last_sequence_number:
                self.out_of_order_or_duplicate += 1
        self.last_sequence_number = max(
            payload.sequence_number,
            self.last_sequence_number if self.last_sequence_number is not None else -1,
        )

        # Compute ingestion latency from device UTC timestamp to bridge receipt UTC
        dev_dt = datetime.fromisoformat(payload.timestamp.replace("Z", "+00:00"))
        lat_ms = max(0.0, (received_utc - dev_dt).total_seconds() * 1000.0)
        self.ingestion_latencies_ms.append(round(lat_ms, 3))
        return gap

    def to_dict(self, now_utc: datetime | None = None, timeout_s: float = 15.0) -> dict[str, Any]:
        """Return JSON-serialisable per-device status summary."""
        lats = self.ingestion_latencies_ms
        return {
            "device_id": self.device_id,
            "machine_id": self.machine_id,
            "liveness_state": self.liveness_state(now_utc=now_utc, timeout_s=timeout_s),
            "messages_received": self.messages_received,
            "last_sequence_number": self.last_sequence_number,
            "sequence_gaps": self.sequence_gaps,
            "out_of_order_or_duplicate": self.out_of_order_or_duplicate,
            "last_message_timestamp": self.last_message_timestamp,
            "last_received_utc": self.last_received_utc,
            "heartbeat_count": self.heartbeat_count,
            "last_heartbeat_timestamp": self.last_heartbeat_timestamp,
            "last_heartbeat_received_utc": self.last_heartbeat_received_utc,
            "active_profile_reported": self.active_profile_reported,
            "device_reconnect_count": self.device_reconnect_count,
            "wifi_rssi_dbm": self.wifi_rssi_dbm,
            "uptime_s": self.uptime_s,
            "lwt_offline": self.lwt_offline,
            "ingestion_latency_ms": _summarize_latencies(lats),
        }


def _summarize_latencies(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min_ms": None,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min_ms": round(float(np.min(arr)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


class MqttKafkaBridge:
    """Stateful MQTT-to-Kafka bridge reusing the existing DriftGuard Kafka contract."""

    def __init__(
        self,
        mqtt_config: MqttBridgeConfig,
        kafka_settings: Settings,
        kafka_producer: Any | None = None,
    ) -> None:
        """Initialise bridge with MQTT config, Kafka settings, and optional injected producer."""
        self._mqtt_cfg = mqtt_config
        self._kafka_settings = kafka_settings
        self._producer: Any = (
            kafka_producer
            if kafka_producer is not None
            else Producer(build_producer_config(kafka_settings))
        )
        self._delivery = DeliveryTracker()
        self._lock = threading.Lock()

        self.connection_state: str = "disconnected"
        self._has_connected_once: bool = False
        self.reconnection_count: int = 0
        self.messages_received: int = 0
        self.invalid_messages: int = 0
        self.rejection_reasons: Counter[str] = Counter()
        self.last_message_timestamp: str | None = None
        self.devices: dict[str, DeviceHealthState] = {}
        self.all_latencies_ms: list[float] = []
        self._pending_status_devices: set[str] = set()
        self._pending_lock = threading.Lock()
        self._started_utc = format_utc_timestamp(datetime.now(timezone.utc))

    @property
    def messages_forwarded(self) -> int:
        """Broker-confirmed Kafka messages delivered via `DeliveryTracker`."""
        return self._delivery.delivered

    @property
    def sequence_gaps_total(self) -> int:
        """Total missing sequence IDs observed across all connected devices."""
        return sum(d.sequence_gaps for d in self.devices.values())

    def handle_connect(self, rc: int) -> None:
        """Record MQTT connect / reconnect state transition."""
        with self._lock:
            if rc == 0:
                if self._has_connected_once:
                    self.reconnection_count += 1
                self._has_connected_once = True
                self.connection_state = "connected"
                logger.info(
                    "mqtt connected",
                    extra={
                        "fields": {
                            "broker": self._mqtt_cfg.broker_host,
                            "port": self._mqtt_cfg.broker_port,
                            "topic": self._mqtt_cfg.mqtt_topic,
                            "reconnection_count": self.reconnection_count,
                        }
                    },
                )
            else:
                self.connection_state = f"connect_error_{rc}"
                logger.error("mqtt connection rejected", extra={"fields": {"rc": rc}})

    def handle_disconnect(self, rc: int) -> None:
        """Record MQTT disconnect state transition."""
        with self._lock:
            self.connection_state = "reconnecting" if rc != 0 else "disconnected"
            logger.warning(
                "mqtt disconnected",
                extra={"fields": {"rc": rc, "state": self.connection_state}},
            )

    def _publish_device_status_to_kafka(self, device_id: str, *, poll: bool = True) -> None:
        """Publish verified device status to `research.telemetry.device_status`."""
        if not self._mqtt_cfg.publish_device_status_topic:
            return
        with self._lock:
            dev = self.devices.get(device_id)
            if dev is None:
                return
            payload = {
                "event_type": "device_status",
                "bridge_connection_state": self.connection_state,
                "bridge_reconnection_count": self.reconnection_count,
                "messages_received_total": self.messages_received,
                "messages_forwarded_total": self._delivery.delivered,
                "invalid_messages_total": self.invalid_messages,
                "sequence_gaps_total": self.sequence_gaps_total,
                "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
                "device": dev.to_dict(timeout_s=self._mqtt_cfg.heartbeat_timeout_s),
            }
        self._producer.produce(
            topic=self._kafka_settings.device_status_topic,
            key=device_id.encode("utf-8"),
            value=json.dumps(payload).encode("utf-8"),
        )
        if poll:
            self._producer.poll(0)

    def _drain_pending_device_statuses(self, *, poll: bool = True) -> None:
        """Emit verified `device_status` records outside librdkafka C delivery callbacks."""
        with self._pending_lock:
            pending = list(self._pending_status_devices)
            self._pending_status_devices.clear()
        for dev_id in pending:
            self._publish_device_status_to_kafka(dev_id, poll=poll)

    def process_heartbeat_message(
        self,
        topic: str,
        raw_payload: bytes | str | None,
        received_utc: datetime | None = None,
    ) -> MqttHeartbeatPayload | None:
        """Validate one MQTT heartbeat/LWT payload and update verified device state."""
        now_utc = received_utc or datetime.now(timezone.utc)
        try:
            hb = parse_mqtt_heartbeat_payload(raw_payload)
        except SchemaValidationError as exc:
            reason = str(exc).split(":", maxsplit=1)[0]
            with self._lock:
                self.invalid_messages += 1
                self.rejection_reasons[f"heartbeat:{reason}"] += 1
            logger.warning(
                "rejected malformed mqtt heartbeat",
                extra={"fields": {"mqtt_topic": topic, "reason": str(exc)}},
            )
            return None

        with self._lock:
            dev = self.devices.get(hb.device_id)
            if dev is None:
                dev = DeviceHealthState(device_id=hb.device_id, machine_id=hb.machine_id)
                self.devices[hb.device_id] = dev
            dev.record_heartbeat(hb, now_utc)
        self._publish_device_status_to_kafka(hb.device_id)
        return hb

    def process_mqtt_message(
        self,
        topic: str,
        raw_payload: bytes | str | None,
        received_utc: datetime | None = None,
    ) -> TelemetryEvent | None:
        """Validate one MQTT message, update monitoring metrics, and forward to Kafka."""
        if topic.endswith("/heartbeat"):
            self.process_heartbeat_message(
                topic=topic, raw_payload=raw_payload, received_utc=received_utc
            )
            return None

        now_utc = received_utc or datetime.now(timezone.utc)
        with self._lock:
            self.messages_received += 1

        try:
            iot_payload = parse_mqtt_iot_payload(raw_payload)
            event = convert_mqtt_to_telemetry_event(
                iot_payload, source_type=self._mqtt_cfg.source_type
            )
        except SchemaValidationError as exc:
            reason = str(exc).split(":", maxsplit=1)[0]
            with self._lock:
                self.invalid_messages += 1
                self.rejection_reasons[reason] += 1
            logger.warning(
                "rejected malformed mqtt payload",
                extra={"fields": {"mqtt_topic": topic, "reason": str(exc)}},
            )
            return None

        with self._lock:
            dev = self.devices.get(iot_payload.device_id)
            if dev is None:
                dev = DeviceHealthState(
                    device_id=iot_payload.device_id,
                    machine_id=iot_payload.machine_id,
                )
                self.devices[iot_payload.device_id] = dev
            dev.lwt_offline = False
            gap = dev.record(iot_payload, now_utc)
            self.last_message_timestamp = iot_payload.timestamp
            self.all_latencies_ms.append(dev.ingestion_latencies_ms[-1])
            if gap > 0:
                logger.warning(
                    "sequence gap detected",
                    extra={
                        "fields": {
                            "device_id": iot_payload.device_id,
                            "gap": gap,
                            "sequence_number": iot_payload.sequence_number,
                        }
                    },
                )

        def _on_telemetry_delivery(err: Any, msg: Any) -> None:
            self._delivery.on_delivery(err, msg)
            if err is None and self._mqtt_cfg.publish_device_status_topic:
                with self._pending_lock:
                    self._pending_status_devices.add(iot_payload.device_id)

        # Forward to existing `research.telemetry.raw` keyed by `machine_id`
        self._producer.produce(
            topic=self._kafka_settings.topic,
            key=event.machine_id.encode("utf-8"),
            value=event.to_json_bytes(),
            on_delivery=_on_telemetry_delivery,
        )
        self._producer.poll(0)
        self._drain_pending_device_statuses(poll=True)
        return event

    def flush(self, timeout_s: float = 10.0) -> int:
        """Flush pending Kafka messages and return unconfirmed count."""
        unconfirmed = int(self._producer.flush(timeout_s))
        self._drain_pending_device_statuses(poll=False)
        return int(self._producer.flush(timeout_s)) or unconfirmed

    def build_status_report(self, run_id: str | None = None) -> dict[str, Any]:
        """Build structured IoT device & bridge status report dictionary."""
        with self._lock:
            return {
                "run_id": run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "started_utc": self._started_utc,
                "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
                "connection_state": self.connection_state,
                "last_message_timestamp": self.last_message_timestamp,
                "messages_received": self.messages_received,
                "messages_forwarded": self._delivery.delivered,
                "kafka_delivery_failed": self._delivery.failed,
                "invalid_messages": self.invalid_messages,
                "rejection_reasons": dict(self.rejection_reasons),
                "sequence_gaps": self.sequence_gaps_total,
                "reconnection_count": self.reconnection_count,
                "ingestion_latency_ms": _summarize_latencies(self.all_latencies_ms),
                "kafka_target": {
                    "bootstrap_servers": self._kafka_settings.bootstrap_servers,
                    "topic": self._kafka_settings.topic,
                    "per_partition_delivered": {
                        str(k): v for k, v in sorted(self._delivery.per_partition.items())
                    },
                },
                "mqtt_config": self._mqtt_cfg.sanitized_dict(),
                "devices": {
                    k: v.to_dict(timeout_s=self._mqtt_cfg.heartbeat_timeout_s)
                    for k, v in sorted(self.devices.items())
                },
            }

    def write_status_report(self, output_dir: Path, run_id: str | None = None) -> Path:
        """Write `iot_status.json` and `iot_status.md` into `output_dir`."""
        report = self.build_status_report(run_id=run_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "iot_status.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "iot_status.md").write_text(
            render_iot_status_markdown(report), encoding="utf-8"
        )
        return output_dir


def render_iot_status_markdown(report: Mapping[str, Any]) -> str:
    """Render human-readable IoT Device & MQTT-Kafka Bridge Status Report."""
    lat = report["ingestion_latency_ms"]
    lines = [
        f"# DriftGuard-Kafka M10-IoT Device Status Report — `{report['run_id']}`",
        "",
        f"Started: `{report['started_utc']}` | Generated: `{report['generated_utc']}` | "
        f"Connection State: **`{report['connection_state']}`**",
        "",
        "## 1. Bridge Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Connection State | `{report['connection_state']}` |",
        f"| Last Message Timestamp | `{report['last_message_timestamp']}` |",
        f"| Messages Received (MQTT) | `{report['messages_received']}` |",
        f"| Messages Forwarded (Kafka Confirmed) | `{report['messages_forwarded']}` |",
        f"| Kafka Delivery Failed | `{report['kafka_delivery_failed']}` |",
        f"| Invalid Messages Rejected | `{report['invalid_messages']}` |",
        f"| Sequence Gaps Detected | `{report['sequence_gaps']}` |",
        f"| MQTT Reconnection Count | `{report['reconnection_count']}` |",
        f"| Ingestion Latency (mean / p50 / p95 / max) | "
        f"`{lat['mean_ms']} / {lat['p50_ms']} / {lat['p95_ms']} / {lat['max_ms']} ms` |",
        f"| Kafka Topic | `{report['kafka_target']['topic']}` |",
        f"| Kafka Partitions Delivered | `{report['kafka_target']['per_partition_delivered']}` |",
        "",
        "## 2. Connected Virtual IoT Devices",
        "",
        "| Device ID | Machine ID | Received | Last Seq | Seq Gaps | Out-of-Order | "
        "Last Timestamp | Mean Latency (ms) |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for dev in report["devices"].values():
        d_lat = dev["ingestion_latency_ms"]
        lines.append(
            f"| `{dev['device_id']}` | `{dev['machine_id']}` | {dev['messages_received']} | "
            f"{dev['last_sequence_number']} | {dev['sequence_gaps']} | "
            f"{dev['out_of_order_or_duplicate']} | `{dev['last_message_timestamp']}` | "
            f"{d_lat['mean_ms']} |"
        )
    if not report["devices"]:
        lines.append("| *(none)* | — | 0 | — | 0 | 0 | — | — |")

    if report["rejection_reasons"]:
        lines.extend(["", "## 3. Invalid Payload Rejection Reasons", ""])
        for reason, cnt in sorted(report["rejection_reasons"].items()):
            lines.append(f"- `{reason}`: `{cnt}`")
    lines.append("")
    return "\n".join(lines)


def run_bridge_loop(
    bridge: MqttKafkaBridge,
    mqtt_cfg: MqttBridgeConfig,
    max_messages: int | None = None,
    duration_s: float | None = None,
    status_output_dir: Path | None = None,
    run_id: str | None = None,
    external_stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Connect to the MQTT broker, subscribe, and forward telemetry to Kafka."""
    stop_event = external_stop_event if external_stop_event is not None else threading.Event()
    client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id=mqtt_cfg.mqtt_client_id,
    )
    if mqtt_cfg.username is not None:
        client.username_pw_set(mqtt_cfg.username, mqtt_cfg.password)
    if mqtt_cfg.use_tls:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    def _on_connect(
        _client: Any,
        _userdata: Any,
        _flags: Any,
        reason_code: Any,
        _properties: Any,
    ) -> None:
        rc = int(getattr(reason_code, "value", reason_code))
        bridge.handle_connect(rc)
        if rc == 0:
            _client.subscribe(mqtt_cfg.mqtt_topic, qos=mqtt_cfg.qos)
            hb_topic = mqtt_cfg.resolved_heartbeat_topic()
            if hb_topic and hb_topic != mqtt_cfg.mqtt_topic:
                _client.subscribe(hb_topic, qos=mqtt_cfg.qos)

    def _on_disconnect(
        _client: Any,
        _userdata: Any,
        _flags: Any,
        reason_code: Any,
        _properties: Any,
    ) -> None:
        rc = int(getattr(reason_code, "value", reason_code))
        bridge.handle_disconnect(rc)

    def _on_message(_client: Any, _userdata: Any, msg: Any) -> None:
        topic_str = str(msg.topic)
        if "/dashboard/" in topic_str:
            return
        bridge.process_mqtt_message(topic=topic_str, raw_payload=msg.payload)
        if max_messages is not None and bridge.messages_received >= max_messages:
            stop_event.set()

    client.on_connect = _on_connect
    client.on_disconnect = _on_disconnect
    client.on_message = _on_message

    def _handle_sig(_signum: int, _frame: Any) -> None:
        stop_event.set()

    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _handle_sig)
        signal.signal(signal.SIGTERM, _handle_sig)

    t0 = time.monotonic()
    client.connect(mqtt_cfg.broker_host, mqtt_cfg.broker_port, keepalive=60)
    client.loop_start()
    try:
        while not stop_event.is_set():
            if duration_s is not None and (time.monotonic() - t0) >= duration_s:
                break
            stop_event.wait(0.2)
    finally:
        client.loop_stop()
        client.disconnect()
        bridge.flush(10.0)
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)

    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = status_output_dir or (mqtt_cfg.status_report_dir / rid)
    bridge.write_status_report(out_dir, run_id=rid)
    return bridge.build_status_report(run_id=rid)


def build_profile_command_payload(
    profile: str,
    command_token: str | None,
) -> bytes:
    """Validate and serialize an authenticated remote `set_profile` command."""
    norm_profile = profile.strip().lower()
    if norm_profile not in VALID_DEVICE_PROFILES:
        raise SchemaValidationError(
            f"Unsupported profile {profile!r}; expected {sorted(VALID_DEVICE_PROFILES)}"
        )
    token = (command_token or "").strip()
    if not token:
        raise ConfigError("MQTT command_token must not be empty for device profile control")
    payload = {
        "command": "set_profile",
        "profile": norm_profile,
        "command_token": token,
        "issued_utc": format_utc_timestamp(datetime.now(timezone.utc)),
    }
    return json.dumps(payload).encode("utf-8")


def publish_profile_command(
    *,
    broker_host: str,
    broker_port: int,
    namespace: str,
    device_id: str,
    profile: str,
    command_token: str | None,
    use_tls: bool = False,
    username: str | None = None,
    password: str | None = None,
    qos: int = 1,
    ca_certs: str | None = None,
) -> str:
    """Publish one authenticated `set_profile` MQTT command to a target ESP32 device."""
    _validate_device_id(device_id)
    wire = build_profile_command_payload(profile=profile, command_token=command_token)
    topic = f"{namespace.rstrip('/')}/{device_id}/command"

    client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id=f"driftguard-cmd-{uuid.uuid4().hex[:8]}",
    )
    if username is not None:
        client.username_pw_set(username, password)
    if use_tls:
        client.tls_set(ca_certs=ca_certs, cert_reqs=ssl.CERT_REQUIRED)
    client.connect(broker_host, broker_port, keepalive=15)
    client.loop_start()
    try:
        info = client.publish(topic, wire, qos=qos)
        info.wait_for_publish(timeout=5.0)
    finally:
        client.loop_stop()
        client.disconnect()
    return topic


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the M10-IoT MQTT-to-Kafka bridge."""
    parser = argparse.ArgumentParser(description="DriftGuard-Kafka M10-IoT MQTT-to-Kafka Bridge.")
    parser.add_argument("--broker-host", default=None)
    parser.add_argument("--broker-port", type=int, default=None)
    parser.add_argument("--use-tls", action="store_true", default=None)
    parser.add_argument("--mqtt-topic", default=None)
    parser.add_argument("--publish-device-status", action="store_true", default=None)
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--status-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for running the MQTT-to-Kafka bridge."""
    args = parse_args(argv)
    kafka_settings = load_settings()
    configure_logging(kafka_settings.log_level)
    mqtt_cfg = load_mqtt_config()

    if (
        args.broker_host is not None
        or args.broker_port is not None
        or args.use_tls is not None
        or args.mqtt_topic is not None
        or args.publish_device_status is not None
    ):
        mqtt_cfg = MqttBridgeConfig(
            broker_host=args.broker_host or mqtt_cfg.broker_host,
            broker_port=args.broker_port or mqtt_cfg.broker_port,
            use_tls=bool(args.use_tls) if args.use_tls is not None else mqtt_cfg.use_tls,
            username=mqtt_cfg.username,
            password=mqtt_cfg.password,
            mqtt_topic=args.mqtt_topic or mqtt_cfg.mqtt_topic,
            mqtt_client_id=mqtt_cfg.mqtt_client_id,
            qos=mqtt_cfg.qos,
            status_report_dir=mqtt_cfg.status_report_dir,
            mqtt_namespace=mqtt_cfg.mqtt_namespace,
            heartbeat_topic=mqtt_cfg.heartbeat_topic,
            command_token=mqtt_cfg.command_token,
            publish_device_status_topic=(
                bool(args.publish_device_status)
                if args.publish_device_status is not None
                else mqtt_cfg.publish_device_status_topic
            ),
        )

    bridge = MqttKafkaBridge(mqtt_config=mqtt_cfg, kafka_settings=kafka_settings)
    summary = run_bridge_loop(
        bridge=bridge,
        mqtt_cfg=mqtt_cfg,
        max_messages=args.max_messages,
        duration_s=args.duration_s,
        status_output_dir=args.status_dir,
        run_id=args.run_id,
    )
    logger.info(
        "mqtt-kafka bridge summary",
        extra={
            "fields": {
                "received": summary["messages_received"],
                "forwarded": summary["messages_forwarded"],
                "invalid": summary["invalid_messages"],
                "sequence_gaps": summary["sequence_gaps"],
                "reconnections": summary["reconnection_count"],
            }
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
