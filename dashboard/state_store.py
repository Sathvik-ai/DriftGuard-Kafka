"""Thread-safe ring-buffer state store and MQTT subscriber for DriftGuard M10.1 Dashboard."""

from __future__ import annotations

import json
import logging
import ssl
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardMqttSettings:
    """MQTT connection configuration for the read-only dashboard viewer."""

    broker_host: str = "localhost"
    broker_port: int = 1883
    use_tls: bool = False
    username: str | None = None
    password: str | None = None
    namespace: str = "driftguard/demo"
    client_id: str = "driftguard-streamlit-viewer"
    qos: int = 1
    allow_profile_control: bool = False
    command_token: str | None = None
    ca_certs: str | None = None

    @property
    def subscription_topic(self) -> str:
        """Wildcard dashboard topic subscribed by the Streamlit viewer."""
        return f"{self.namespace.rstrip('/')}/dashboard/#"


class DashboardStateStore:
    """Bounded, thread-safe in-memory mirror of verified DriftGuard MQTT streams."""

    def __init__(self, max_events: int = 400, max_incidents: int = 150) -> None:
        """Initialise ring buffers and freshness counters."""
        self._lock = threading.RLock()
        self.telemetry: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.model_scores: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.alerts: deque[dict[str, Any]] = deque(maxlen=max_incidents)
        self.drifts: deque[dict[str, Any]] = deque(maxlen=max_incidents)
        self.devices: dict[str, dict[str, Any]] = {}
        self.backend_health: dict[str, Any] = {}

        self.mqtt_connected: bool = False
        self.mqtt_connection_state: str = "disconnected"
        self.mqtt_error: str | None = None
        self.reconnect_count: int = 0
        self._connected_once: bool = False
        self.messages_ingested: int = 0
        self.demo_mode: bool = False

        self.last_mqtt_message_utc: str | None = None
        self.last_telemetry_utc: str | None = None
        self.last_model_score_utc: str | None = None
        self.last_backend_health_utc: str | None = None

    def set_connection_state(self, connected: bool, state: str, error: str | None = None) -> None:
        """Record MQTT client connection transitions cleanly."""
        with self._lock:
            if connected and self._connected_once and not self.mqtt_connected:
                self.reconnect_count += 1
            if connected:
                self._connected_once = True
            self.mqtt_connected = connected
            self.mqtt_connection_state = state
            self.mqtt_error = error

    def clear_for_live_mode(self) -> None:
        """Clear static demo data when transitioning back to live stream."""
        with self._lock:
            self.demo_mode = False
            self.telemetry.clear()
            self.model_scores.clear()
            self.alerts.clear()
            self.drifts.clear()
            self.devices.clear()
            self.backend_health.clear()
            self.last_telemetry_utc = None
            self.last_model_score_utc = None
            self.last_backend_health_utc = None

    def load_static_demo_sample(self) -> None:
        """Populate an explicitly labeled static offline DEMO dataset (never claimed as live)."""
        with self._lock:
            self.demo_mode = True
            self.telemetry.clear()
            self.model_scores.clear()
            self.alerts.clear()
            self.drifts.clear()
            self.devices.clear()
            base = datetime.now(timezone.utc) - timedelta(seconds=90)
            for idx in range(45):
                ts = (base + timedelta(seconds=idx * 2)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                spike = 35.0 if idx in (22, 23) else 0.0
                drift_shift = max(0, idx - 30) * 1.4
                temp = round(46.5 + spike + drift_shift + (idx % 3) * 0.2, 2)
                vib = round(2.1 + (0.4 if idx >= 30 else 0.05), 3)
                press = round(5.0 + (0.3 if idx >= 30 else 0.03), 3)
                row = {
                    "event_id": f"demo-{idx:04d}",
                    "machine_id": "machine_01",
                    "device_id": "esp32-wokwi-demo",
                    "sequence_number": idx + 1,
                    "timestamp": ts,
                    "source_type": "static_demo_sample",
                    "temperature_c": temp,
                    "vibration_mm_s": vib,
                    "pressure_bar": press,
                }
                self.telemetry.append(row)
                is_anom = spike > 0 or drift_shift > 8.0
                self.model_scores.append(
                    {
                        **row,
                        "telemetry": {
                            "temperature_c": temp,
                            "vibration_mm_s": vib,
                            "pressure_bar": press,
                        },
                        "hardware": {
                            "os_platform": "Demo Static Snapshot",
                            "machine_arch": "arm64",
                            "mps_built": True,
                            "mps_available": False,
                            "selected_dl_device": "cpu",
                        },
                        "memory_mib": {
                            "process_rss_mib": None,
                            "mps_allocated_mib": None,
                            "mps_driver_mib": None,
                            "system_available_mib": None,
                        },
                        "models": {
                            "static_threshold": {
                                "score": 2.4 if is_anom else 0.35,
                                "threshold": 1.0,
                                "normalized_ratio": 2.4 if is_anom else 0.35,
                                "anomaly": is_anom,
                                "latency_ms": 0.05,
                                "device": "cpu",
                                "status": "demo",
                                "channel_mode": "multivariate",
                            },
                            "isolation_forest": {
                                "score": 0.68 if is_anom else 0.42,
                                "threshold": 0.56,
                                "normalized_ratio": 1.21 if is_anom else 0.75,
                                "anomaly": is_anom,
                                "latency_ms": 0.4,
                                "device": "cpu",
                                "status": "demo",
                                "channel_mode": "multivariate",
                            },
                            "half_space_trees": {
                                "score": 0.81 if is_anom else 0.18,
                                "threshold": 0.65,
                                "normalized_ratio": 1.25 if is_anom else 0.28,
                                "anomaly": is_anom,
                                "latency_ms": 0.3,
                                "device": "cpu",
                                "status": "demo",
                                "channel_mode": "multivariate",
                            },
                            "lstm_autoencoder": {
                                "score": 4.2 if is_anom else 0.25,
                                "threshold": 1.2,
                                "normalized_ratio": 3.5 if is_anom else 0.21,
                                "anomaly": is_anom,
                                "latency_ms": 1.1,
                                "device": "demo",
                                "status": "demo",
                                "channel_mode": "multivariate",
                            },
                            "tcn_autoencoder": {
                                "score": 3.9 if is_anom else 0.22,
                                "threshold": 1.1,
                                "normalized_ratio": 3.55 if is_anom else 0.2,
                                "anomaly": is_anom,
                                "latency_ms": 0.9,
                                "device": "demo",
                                "status": "demo",
                                "channel_mode": "multivariate",
                            },
                            "tranad": {
                                "score": 3.6 if is_anom else 0.19,
                                "threshold": 1.05,
                                "normalized_ratio": 3.43 if is_anom else 0.18,
                                "anomaly": is_anom,
                                "latency_ms": 1.2,
                                "device": "demo",
                                "status": "demo",
                                "channel_mode": "multivariate",
                            },
                            "chronos2_zero_shot": {
                                "score": 2.9 if is_anom else 0.31,
                                "threshold": 1.15,
                                "normalized_ratio": 2.52 if is_anom else 0.27,
                                "anomaly": is_anom,
                                "latency_ms": 18.0,
                                "device": "demo",
                                "status": "demo" if idx % 10 == 0 else "cached_skip",
                                "channel_mode": "multivariate",
                            },
                            "timesfm3_zero_shot": {
                                "score": 2.7 if is_anom else 0.33,
                                "threshold": 1.18,
                                "normalized_ratio": 2.29 if is_anom else 0.28,
                                "anomaly": is_anom,
                                "latency_ms": 14.0,
                                "device": "demo",
                                "status": "demo" if idx % 10 == 0 else "cached_skip",
                                "channel_mode": "independent_univariate",
                            },
                        },
                        "adwin": {
                            "drift_detected": idx == 36,
                            "drift_features": ["temperature_c"] if idx == 36 else [],
                            "drift_events_total": 1 if idx >= 36 else 0,
                        },
                    }
                )
            self.alerts.append(
                {
                    "machine_id": "machine_01",
                    "device_id": "esp32-wokwi-demo",
                    "timestamp": self.telemetry[22]["timestamp"],
                    "severity": "HIGH",
                    "reason": "Demo spike sample (not live sensor telemetry)",
                }
            )
            self.drifts.append(
                {
                    "machine_id": "machine_01",
                    "feature": "temperature_c",
                    "timestamp": self.telemetry[36]["timestamp"],
                    "old_mean": 46.5,
                    "new_mean": 54.9,
                    "magnitude": 8.4,
                    "severity": "HIGH",
                }
            )
            self.devices["esp32-wokwi-demo"] = {
                "device_id": "esp32-wokwi-demo",
                "machine_id": "machine_01",
                "liveness_state": "DEMO",
                "messages_received": 45,
                "last_sequence_number": 45,
                "sequence_gaps": 0,
                "heartbeat_count": 9,
                "active_profile_reported": "demo_sample",
                "device_reconnect_count": 0,
                "ingestion_latency_ms": {"mean_ms": 12.0, "p95_ms": 18.0},
            }

    def ingest_mqtt_message(
        self, topic: str, raw_payload: bytes | str | None, received_utc: datetime | None = None
    ) -> str | None:
        """Parse one dashboard MQTT message and route it to the appropriate store buffer."""
        if not raw_payload:
            return None
        try:
            data = json.loads(raw_payload)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        now_iso = (received_utc or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        suffix = topic.rstrip("/").split("/")[-1]

        with self._lock:
            self.demo_mode = False
            self.messages_ingested += 1
            self.last_mqtt_message_utc = now_iso

            if suffix == "telemetry":
                self.telemetry.append(data)
                self.last_telemetry_utc = now_iso
                return "telemetry"
            if suffix == "model_scores":
                self.model_scores.append(data)
                self.last_model_score_utc = now_iso
                return "model_scores"
            if suffix == "alerts":
                self.alerts.append(data)
                return "alerts"
            if suffix == "drift":
                self.drifts.append(data)
                return "drift"
            if suffix == "device_status":
                dev = data.get("device")
                if isinstance(dev, dict) and isinstance(dev.get("device_id"), str):
                    self.devices[dev["device_id"]] = dev
                return "device_status"
            if suffix == "backend_health":
                self.backend_health = data
                self.last_backend_health_utc = now_iso
                return "backend_health"
            if suffix == "snapshot":
                self._hydrate_snapshot_locked(data, now_iso)
                return "snapshot"
        return None

    def _hydrate_snapshot_locked(self, snap: Mapping[str, Any], now_iso: str) -> None:
        if isinstance(snap.get("backend_health"), dict):
            self.backend_health = dict(snap["backend_health"])
            self.last_backend_health_utc = now_iso
        if not self.telemetry and isinstance(snap.get("telemetry"), list):
            for item in snap["telemetry"]:
                if isinstance(item, dict):
                    self.telemetry.append(item)
            if self.telemetry:
                self.last_telemetry_utc = now_iso
        if not self.model_scores and isinstance(snap.get("model_scores"), list):
            for item in snap["model_scores"]:
                if isinstance(item, dict):
                    self.model_scores.append(item)
            if self.model_scores:
                self.last_model_score_utc = now_iso
        if not self.alerts and isinstance(snap.get("alerts"), list):
            for item in snap["alerts"]:
                if isinstance(item, dict):
                    self.alerts.append(item)
        if not self.drifts and isinstance(snap.get("drifts"), list):
            for item in snap["drifts"]:
                if isinstance(item, dict):
                    self.drifts.append(item)
        if not self.devices and isinstance(snap.get("devices"), dict):
            for dev_id, dev_val in snap["devices"].items():
                if isinstance(dev_val, dict):
                    self.devices[str(dev_id)] = dict(dev_val)

    def compute_freshness(
        self,
        *,
        now_utc: datetime | None = None,
        live_threshold_s: float = 12.0,
        delayed_threshold_s: float = 40.0,
    ) -> dict[str, Any]:
        """Return explicit LIVE / DELAYED / OFFLINE / DEMO state from real timestamps."""
        with self._lock:
            if self.demo_mode:
                return {
                    "state": "DEMO",
                    "age_s": None,
                    "reason": "Static offline demo sample active",
                }
            if not self.mqtt_connected:
                return {
                    "state": "OFFLINE",
                    "age_s": None,
                    "reason": f"MQTT disconnected ({self.mqtt_connection_state})",
                }
            ref_str = self.last_telemetry_utc or self.last_mqtt_message_utc
            if ref_str is None:
                return {
                    "state": "OFFLINE",
                    "age_s": None,
                    "reason": "Waiting for first verified MQTT message from backend",
                }
            now = now_utc or datetime.now(timezone.utc)
            last_dt = datetime.fromisoformat(ref_str.replace("Z", "+00:00"))
            age_s = round(max(0.0, (now - last_dt).total_seconds()), 1)
            if age_s <= live_threshold_s:
                return {
                    "state": "LIVE",
                    "age_s": age_s,
                    "reason": f"Last verified event {age_s}s ago",
                }
            if age_s <= delayed_threshold_s:
                return {
                    "state": "DELAYED",
                    "age_s": age_s,
                    "reason": f"No telemetry for {age_s}s (delayed)",
                }
            return {
                "state": "OFFLINE",
                "age_s": age_s,
                "reason": f"Telemetry stale ({age_s}s old)",
            }

    def diagnostic_banners(self, now_utc: datetime | None = None) -> list[dict[str, str]]:
        """Produce honest, actionable diagnostics for disconnected/stale components."""
        with self._lock:
            banners: list[dict[str, str]] = []
            if self.demo_mode:
                banners.append(
                    {
                        "level": "warning",
                        "title": "STATIC DEMO MODE ACTIVE",
                        "detail": (
                            "Showing synthetic offline sample data only. Toggle Demo Mode OFF "
                            "to display real live telemetry from Wokwi ESP32 -> MQTT -> Kafka."
                        ),
                    }
                )
                return banners

            if not self.mqtt_connected:
                err = f" ({self.mqtt_error})" if self.mqtt_error else ""
                banners.append(
                    {
                        "level": "error",
                        "title": "MQTT Broker Disconnected",
                        "detail": (f"Viewer state={self.mqtt_connection_state}{err}. Live paused."),
                    }
                )
            elif self.last_backend_health_utc is None:
                banners.append(
                    {
                        "level": "warning",
                        "title": "Mac Kafka Backend Offline / Not Publishing",
                        "detail": (
                            "Connected to MQTT broker, but no `backend_health` payload has arrived "
                            "from `app.dashboard_publisher`."
                        ),
                    }
                )

            if self.mqtt_connected and self.last_telemetry_utc is None:
                banners.append(
                    {
                        "level": "warning",
                        "title": "No Live Virtual ESP32 Telemetry Yet",
                        "detail": (
                            "Start the Wokwi ESP32 simulation and `make iot-bridge` to stream "
                            "live sensor events into Kafka."
                        ),
                    }
                )
            elif self.mqtt_connected and self.last_model_score_utc is None:
                banners.append(
                    {
                        "level": "info",
                        "title": "Waiting for ML / DL Inference Worker",
                        "detail": (
                            "Telemetry is flowing, but `research.telemetry.model_scores` has not "
                            "arrived yet (`make iot-inference`)."
                        ),
                    }
                )

            now = now_utc or datetime.now(timezone.utc)
            for dev_id, dev in sorted(self.devices.items()):
                state = str(dev.get("liveness_state", "UNKNOWN"))
                if state in ("OFFLINE", "DELAYED"):
                    banners.append(
                        {
                            "level": "warning" if state == "DELAYED" else "error",
                            "title": f"Device {dev_id} is {state}",
                            "detail": (
                                f"Last message={dev.get('last_message_timestamp')}, "
                                f"gaps={dev.get('sequence_gaps', 0)}"
                            ),
                        }
                    )
            _ = now
            return banners

    def snapshot_view(self) -> dict[str, Any]:
        """Return thread-safe copies of all buffers for UI rendering."""
        with self._lock:
            return {
                "telemetry": list(self.telemetry),
                "model_scores": list(self.model_scores),
                "alerts": list(self.alerts),
                "drifts": list(self.drifts),
                "devices": {k: dict(v) for k, v in self.devices.items()},
                "backend_health": dict(self.backend_health),
                "mqtt_connected": self.mqtt_connected,
                "mqtt_connection_state": self.mqtt_connection_state,
                "messages_ingested": self.messages_ingested,
                "reconnect_count": self.reconnect_count,
                "demo_mode": self.demo_mode,
            }


class DashboardMqttSubscriber:
    """Background MQTT subscriber keeping `DashboardStateStore` updated cleanly."""

    def __init__(self, settings: DashboardMqttSettings, store: DashboardStateStore) -> None:
        """Initialise background paho-mqtt subscriber."""
        self.settings = settings
        self.store = store
        self._started = False
        self._client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=f"{settings.client_id}-{id(self) % 100000}",
        )
        if settings.username is not None:
            self._client.username_pw_set(settings.username, settings.password)
        if settings.use_tls:
            self._client.tls_set(ca_certs=settings.ca_certs, cert_reqs=ssl.CERT_REQUIRED)
        self._client.reconnect_delay_set(min_delay=1, max_delay=15)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def _on_connect(
        self, client: Any, _userdata: Any, _flags: Any, reason_code: Any, _props: Any
    ) -> None:
        rc = int(getattr(reason_code, "value", reason_code))
        if rc == 0:
            self.store.set_connection_state(True, "connected")
            client.subscribe(self.settings.subscription_topic, qos=self.settings.qos)
        else:
            self.store.set_connection_state(False, f"rejected_{rc}")

    def _on_disconnect(
        self, _client: Any, _userdata: Any, _flags: Any, reason_code: Any, _props: Any
    ) -> None:
        rc = int(getattr(reason_code, "value", reason_code))
        self.store.set_connection_state(False, "reconnecting" if rc != 0 else "disconnected")

    def _on_message(self, _client: Any, _userdata: Any, msg: Any) -> None:
        self.store.ingest_mqtt_message(str(msg.topic), msg.payload)

    def start(self) -> None:
        """Connect asynchronously and start background MQTT network loop."""
        if self._started:
            return
        self._started = True
        try:
            self._client.connect_async(
                self.settings.broker_host, self.settings.broker_port, keepalive=60
            )
            self._client.loop_start()
        except Exception as exc:
            self.store.set_connection_state(False, "connect_failed", error=str(exc))

    def stop(self) -> None:
        """Stop MQTT background loop cleanly."""
        if not self._started:
            return
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception as exc:
            logger.debug("dashboard mqtt subscriber stop warning: %s", exc)
        self._started = False


def load_reference_reports(repo_root: Path) -> dict[str, Any]:
    """Load latest immutable M9.1 offline benchmark & M10.1 online JSON reports if present."""
    result: dict[str, Any] = {"m9_1": None, "m10_1": None}
    m9_1_dirs = (
        sorted((repo_root / "reports" / "m9_1").glob("*"))
        if (repo_root / "reports" / "m9_1").exists()
        else []
    )
    for cand in reversed(m9_1_dirs):
        rep = cand / "m9_1_report.json"
        if rep.exists():
            try:
                result["m9_1"] = json.loads(rep.read_text(encoding="utf-8"))
                result["m9_1_path"] = str(rep.relative_to(repo_root))
                break
            except Exception as exc:
                logger.debug("skipped unreadable report %s: %s", rep, exc)

    m10_1_dirs = (
        sorted(p for p in (repo_root / "reports" / "m10_1").glob("*") if p.is_dir())
        if (repo_root / "reports" / "m10_1").exists()
        else []
    )
    for cand in reversed(m10_1_dirs):
        rep = cand / "summary_report.json"
        if rep.exists():
            try:
                result["m10_1"] = json.loads(rep.read_text(encoding="utf-8"))
                result["m10_1_path"] = str(rep.relative_to(repo_root))
                break
            except Exception as exc:
                logger.debug("skipped unreadable report %s: %s", rep, exc)
    return result
