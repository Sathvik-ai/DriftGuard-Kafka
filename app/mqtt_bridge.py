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

    def sanitized_dict(self) -> dict[str, Any]:
        """Return JSON-safe config dictionary with credentials redacted."""
        return {
            "broker_host": self.broker_host,
            "broker_port": self.broker_port,
            "use_tls": self.use_tls,
            "username_configured": bool(self.username),
            "password_configured": bool(self.password),
            "mqtt_topic": self.mqtt_topic,
            "mqtt_client_id": self.mqtt_client_id,
            "qos": self.qos,
            "source_type": self.source_type.value,
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


@dataclass
class DeviceHealthState:
    """Per-device sequence continuity, latency, and message counters."""

    device_id: str
    machine_id: str
    messages_received: int = 0
    last_sequence_number: int | None = None
    sequence_gaps: int = 0
    out_of_order_or_duplicate: int = 0
    last_message_timestamp: str | None = None
    last_received_utc: str | None = None
    ingestion_latencies_ms: list[float] = field(default_factory=list)

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

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable per-device status summary."""
        lats = self.ingestion_latencies_ms
        return {
            "device_id": self.device_id,
            "machine_id": self.machine_id,
            "messages_received": self.messages_received,
            "last_sequence_number": self.last_sequence_number,
            "sequence_gaps": self.sequence_gaps,
            "out_of_order_or_duplicate": self.out_of_order_or_duplicate,
            "last_message_timestamp": self.last_message_timestamp,
            "last_received_utc": self.last_received_utc,
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

    def process_mqtt_message(
        self,
        topic: str,
        raw_payload: bytes | str | None,
        received_utc: datetime | None = None,
    ) -> TelemetryEvent | None:
        """Validate one MQTT message, update monitoring metrics, and forward to Kafka."""
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

        # Forward to existing `research.telemetry.raw` keyed by `machine_id`
        self._producer.produce(
            topic=self._kafka_settings.topic,
            key=event.machine_id.encode("utf-8"),
            value=event.to_json_bytes(),
            on_delivery=self._delivery.on_delivery,
        )
        self._producer.poll(0)
        return event

    def flush(self, timeout_s: float = 10.0) -> int:
        """Flush pending Kafka messages and return unconfirmed count."""
        return int(self._producer.flush(timeout_s))

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
                "devices": {k: v.to_dict() for k, v in sorted(self.devices.items())},
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
) -> dict[str, Any]:
    """Connect to the MQTT broker, subscribe, and forward telemetry to Kafka."""
    stop_event = threading.Event()
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
        bridge.process_mqtt_message(topic=str(msg.topic), raw_payload=msg.payload)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the M10-IoT MQTT-to-Kafka bridge."""
    parser = argparse.ArgumentParser(description="DriftGuard-Kafka M10-IoT MQTT-to-Kafka Bridge.")
    parser.add_argument("--broker-host", default=None)
    parser.add_argument("--broker-port", type=int, default=None)
    parser.add_argument("--use-tls", action="store_true", default=None)
    parser.add_argument("--mqtt-topic", default=None)
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
