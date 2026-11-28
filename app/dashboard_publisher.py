"""M10.1 Kafka-to-MQTT Dashboard State Publisher.

Consumes confirmed Kafka events from all DriftGuard research topics under an
independent consumer group (`driftguard-m10-1-dashboard-pub`):
  - `research.telemetry.raw`
  - `research.telemetry.alerts`
  - `research.telemetry.drift`
  - `research.telemetry.model_scores`
  - `research.telemetry.device_status`

Re-publishes sanitized JSON updates onto role-separated MQTT dashboard topics:
  - `<namespace>/dashboard/telemetry`
  - `<namespace>/dashboard/alerts`
  - `<namespace>/dashboard/drift`
  - `<namespace>/dashboard/model_scores`
  - `<namespace>/dashboard/device_status`
  - `<namespace>/dashboard/backend_health`
  - `<namespace>/dashboard/snapshot` (retained=True for immediate Streamlit hydration)

Strictly excludes credentials and command tokens from all published payloads.
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
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt
from confluent_kafka import Consumer, KafkaError, KafkaException
from paho.mqtt.enums import CallbackAPIVersion

from app.config import Settings, load_settings
from app.logging_config import configure_logging
from app.schemas import format_utc_timestamp

logger = logging.getLogger(__name__)

DEFAULT_DASHBOARD_PUB_GROUP = "driftguard-m10-1-dashboard-pub"


@dataclass(frozen=True)
class DashboardPublisherConfig:
    """Runtime settings for publishing verified Kafka state to MQTT dashboard topics."""

    broker_host: str = "localhost"
    broker_port: int = 1883
    use_tls: bool = False
    username: str | None = None
    password: str | None = None
    namespace: str = "driftguard/demo"
    qos: int = 1
    consumer_group: str = DEFAULT_DASHBOARD_PUB_GROUP
    auto_offset_reset: str = "latest"
    snapshot_interval_s: float = 2.0
    local_snapshot_path: Path | None = Path("reports/m10_1/latest_snapshot.json")
    ca_certs: str | None = None

    def topic_for(self, suffix: str) -> str:
        """Build full dashboard MQTT topic for a given stream suffix."""
        return f"{self.namespace.rstrip('/')}/dashboard/{suffix}"


class DashboardPublisherState:
    """In-memory bounded mirror of verified Kafka streams for retained MQTT snapshots."""

    def __init__(self, kafka_settings: Settings, pub_cfg: DashboardPublisherConfig) -> None:
        """Initialise bounded buffers and topic mapping."""
        self.kafka_settings = kafka_settings
        self.pub_cfg = pub_cfg
        self._lock = threading.Lock()
        self._started_utc = format_utc_timestamp(datetime.now(timezone.utc))
        self.telemetry: deque[dict[str, Any]] = deque(maxlen=150)
        self.alerts: deque[dict[str, Any]] = deque(maxlen=50)
        self.drifts: deque[dict[str, Any]] = deque(maxlen=50)
        self.model_scores: deque[dict[str, Any]] = deque(maxlen=150)
        self.devices: dict[str, dict[str, Any]] = {}
        self.topic_counts: dict[str, int] = {
            "telemetry": 0,
            "alerts": 0,
            "drift": 0,
            "model_scores": 0,
            "device_status": 0,
        }
        self.last_hardware: dict[str, Any] = {}
        self.last_memory_mib: dict[str, Any] = {}
        self.last_kafka_event_utc: str | None = None

    def route_kafka_record(
        self, kafka_topic: str, raw_value: bytes
    ) -> tuple[str, dict[str, Any]] | None:
        """Decode one Kafka record and return `(dashboard_suffix, sanitized_payload)`."""
        try:
            payload = json.loads(raw_value.decode("utf-8"))
        except Exception:
            logger.warning("dashboard publisher skipped non-JSON record on %s", kafka_topic)
            return None
        if not isinstance(payload, dict):
            return None

        now_iso = format_utc_timestamp(datetime.now(timezone.utc))
        with self._lock:
            self.last_kafka_event_utc = now_iso
            if kafka_topic == self.kafka_settings.topic:
                # Never expose internal ground-truth simulator label if set in non-IoT replay
                sanitized = {k: v for k, v in payload.items() if k != "label"}
                sanitized["kafka_topic"] = kafka_topic
                self.telemetry.append(sanitized)
                self.topic_counts["telemetry"] += 1
                return "telemetry", sanitized

            if kafka_topic == self.kafka_settings.alerts_topic:
                self.alerts.append(payload)
                self.topic_counts["alerts"] += 1
                return "alerts", payload

            if kafka_topic == self.kafka_settings.drift_topic:
                self.drifts.append(payload)
                self.topic_counts["drift"] += 1
                return "drift", payload

            if kafka_topic == self.kafka_settings.model_scores_topic:
                self.model_scores.append(payload)
                self.topic_counts["model_scores"] += 1
                if isinstance(payload.get("hardware"), dict):
                    self.last_hardware = dict(payload["hardware"])
                if isinstance(payload.get("memory_mib"), dict):
                    self.last_memory_mib = dict(payload["memory_mib"])
                return "model_scores", payload

            if kafka_topic == self.kafka_settings.device_status_topic:
                dev = payload.get("device")
                if isinstance(dev, dict) and isinstance(dev.get("device_id"), str):
                    self.devices[dev["device_id"]] = dev
                self.topic_counts["device_status"] += 1
                return "device_status", payload

        return None

    def build_backend_health(self) -> dict[str, Any]:
        """Build verified backend & pipeline health summary."""
        with self._lock:
            return {
                "status": "online",
                "started_utc": self._started_utc,
                "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
                "last_kafka_event_utc": self.last_kafka_event_utc,
                "kafka_bootstrap_servers": self.kafka_settings.bootstrap_servers,
                "topic_counts": dict(self.topic_counts),
                "active_devices": len(self.devices),
                "hardware": dict(self.last_hardware),
                "memory_mib": dict(self.last_memory_mib),
            }

    def build_snapshot(self) -> dict[str, Any]:
        """Build compact retained snapshot for newly joined Streamlit viewers."""
        with self._lock:
            return {
                "snapshot_utc": format_utc_timestamp(datetime.now(timezone.utc)),
                "backend_health": {
                    "status": "online",
                    "started_utc": self._started_utc,
                    "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
                    "last_kafka_event_utc": self.last_kafka_event_utc,
                    "topic_counts": dict(self.topic_counts),
                    "active_devices": len(self.devices),
                    "hardware": dict(self.last_hardware),
                    "memory_mib": dict(self.last_memory_mib),
                },
                "telemetry": list(self.telemetry),
                "model_scores": list(self.model_scores),
                "alerts": list(self.alerts),
                "drifts": list(self.drifts),
                "devices": dict(self.devices),
            }


def run_dashboard_publisher(
    kafka_settings: Settings,
    pub_cfg: DashboardPublisherConfig,
    *,
    duration_s: float | None = None,
    max_records: int | None = None,
    external_stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run Kafka-to-MQTT dashboard relay until duration/max_records or signal."""
    state = DashboardPublisherState(kafka_settings, pub_cfg)
    stop_event = external_stop_event if external_stop_event is not None else threading.Event()
    mqtt_connected = threading.Event()

    mqtt_client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id=f"driftguard-dash-pub-{int(time.time())}",
    )
    if pub_cfg.username is not None:
        mqtt_client.username_pw_set(pub_cfg.username, pub_cfg.password)
    if pub_cfg.use_tls:
        mqtt_client.tls_set(ca_certs=pub_cfg.ca_certs, cert_reqs=ssl.CERT_REQUIRED)

    def _on_connect(_client: Any, _userdata: Any, _flags: Any, rc: Any, _props: Any) -> None:
        code = int(getattr(rc, "value", rc))
        if code == 0:
            mqtt_connected.set()

    mqtt_client.on_connect = _on_connect
    mqtt_client.connect(pub_cfg.broker_host, pub_cfg.broker_port, keepalive=60)
    mqtt_client.loop_start()
    mqtt_connected.wait(timeout=10.0)

    consumer = Consumer(
        {
            "bootstrap.servers": kafka_settings.bootstrap_servers,
            "group.id": pub_cfg.consumer_group,
            "client.id": f"{kafka_settings.client_id}-m10-1-dash-pub",
            "auto.offset.reset": pub_cfg.auto_offset_reset,
            "enable.auto.commit": True,
        }
    )
    subscribed_topics = [
        kafka_settings.topic,
        kafka_settings.alerts_topic,
        kafka_settings.drift_topic,
        kafka_settings.model_scores_topic,
        kafka_settings.device_status_topic,
    ]
    consumer.subscribe(subscribed_topics)

    def _sig(_signum: int, _frame: Any) -> None:
        stop_event.set()

    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)

    t0 = time.monotonic()
    last_snap = 0.0
    total_routed = 0

    def _publish_health_and_snapshot() -> None:
        health = state.build_backend_health()
        snap = state.build_snapshot()
        mqtt_client.publish(
            pub_cfg.topic_for("backend_health"),
            json.dumps(health),
            qos=pub_cfg.qos,
            retain=True,
        )
        mqtt_client.publish(
            pub_cfg.topic_for("snapshot"),
            json.dumps(snap),
            qos=pub_cfg.qos,
            retain=True,
        )
        if pub_cfg.local_snapshot_path is not None:
            pub_cfg.local_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            pub_cfg.local_snapshot_path.write_text(
                json.dumps(snap, indent=2) + "\n", encoding="utf-8"
            )

    try:
        _publish_health_and_snapshot()
        last_snap = time.monotonic()
        while not stop_event.is_set():
            now = time.monotonic()
            if duration_s is not None and (now - t0) >= duration_s:
                break
            if max_records is not None and total_routed >= max_records:
                break
            if (now - last_snap) >= pub_cfg.snapshot_interval_s:
                _publish_health_and_snapshot()
                last_snap = now

            msg = consumer.poll(0.25)
            if msg is None:
                continue
            err = msg.error()
            if err is not None:
                if err.code() != KafkaError._PARTITION_EOF:
                    raise KafkaException(err)
                continue
            raw = msg.value()
            if raw is None:
                continue
            msg_topic = msg.topic()
            if msg_topic is None:
                continue
            routed = state.route_kafka_record(msg_topic, raw)
            if routed is None:
                continue
            suffix, payload = routed
            mqtt_client.publish(
                pub_cfg.topic_for(suffix),
                json.dumps(payload),
                qos=pub_cfg.qos,
                retain=False,
            )
            total_routed += 1
    finally:
        _publish_health_and_snapshot()
        consumer.close()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)

    return state.build_backend_health()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the M10.1 Kafka-to-MQTT dashboard publisher."""
    parser = argparse.ArgumentParser(description="DriftGuard-Kafka M10.1 Dashboard Publisher")
    parser.add_argument("--broker-host", default=os.environ.get("MQTT_BROKER_HOST", "localhost"))
    parser.add_argument(
        "--broker-port", type=int, default=int(os.environ.get("MQTT_BROKER_PORT", "1883"))
    )
    parser.add_argument("--use-tls", action="store_true")
    parser.add_argument(
        "--username",
        default=os.environ.get("MQTT_USERNAME") or None,
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("MQTT_PASSWORD") or None,
    )
    parser.add_argument("--namespace", default=os.environ.get("MQTT_NAMESPACE", "driftguard/demo"))
    parser.add_argument("--consumer-group", default=DEFAULT_DASHBOARD_PUB_GROUP)
    parser.add_argument("--auto-offset-reset", default="latest", choices=("latest", "earliest"))
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for publishing confirmed Kafka topics to MQTT dashboard streams."""
    args = parse_args(argv)
    kafka_settings = load_settings()
    configure_logging(kafka_settings.log_level)
    use_tls = bool(args.use_tls) or (
        os.environ.get("MQTT_USE_TLS", "false").strip().lower() in ("1", "true", "yes", "on")
    )
    cfg = DashboardPublisherConfig(
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        use_tls=use_tls,
        username=args.username,
        password=args.password,
        namespace=args.namespace,
        consumer_group=args.consumer_group,
        auto_offset_reset=args.auto_offset_reset,
    )
    run_dashboard_publisher(
        kafka_settings,
        cfg,
        duration_s=args.duration_s,
        max_records=args.max_records,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
