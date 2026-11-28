"""Virtual Wokwi ESP32 MQTT emulator matching `iot/wokwi_esp32/sketch.ino` (M10.1).

Simulates an ESP32 industrial IoT device publishing 3-channel telemetry
(`temperature_c`, `vibration_mm_s`, `pressure_bar`), periodic heartbeats,
authenticated `set_profile` command handling, and all six firmware profiles
over MQTT (TLS/username/password configurable via environment or CLI).

Note: All outputs are simulated virtual sensor telemetry, matching the Wokwi
ESP32 sketch behaviour, and NEVER represent physical industrial transducers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import ssl
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

from app.logging_config import configure_logging
from app.schemas import format_utc_timestamp

logger = logging.getLogger(__name__)

VALID_PROFILES: tuple[str, ...] = (
    "normal",
    "temperature_spike",
    "vibration_anomaly",
    "gradual_drift",
    "abrupt_drift",
    "intermittent_disconnect",
)


@dataclass(frozen=True)
class WokwiEmulatorConfig:
    """Configuration for the M10.1 virtual ESP32 MQTT publisher."""

    broker_host: str = "localhost"
    broker_port: int = 1883
    use_tls: bool = False
    username: str | None = None
    password: str | None = None
    namespace: str = "driftguard/demo"
    device_id: str = "esp32-wokwi-01"
    machine_id: str = "machine_01"
    publish_interval_s: float = 1.0
    heartbeat_interval_s: float = 5.0
    qos: int = 1
    command_token: str | None = None
    ca_certs: str | None = None
    seed: int = 42

    @property
    def telemetry_topic(self) -> str:
        """Return MQTT topic used for telemetry payloads."""
        return f"{self.namespace}/{self.device_id}/telemetry"

    @property
    def heartbeat_topic(self) -> str:
        """Return MQTT topic used for periodic heartbeat payloads."""
        return f"{self.namespace}/{self.device_id}/heartbeat"

    @property
    def command_topic(self) -> str:
        """Return MQTT topic subscribed to for authenticated profile commands."""
        return f"{self.namespace}/{self.device_id}/command"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class VirtualWokwiEsp32:
    """Deterministic virtual ESP32 state machine matching `sketch.ino`."""

    def __init__(self, config: WokwiEmulatorConfig) -> None:
        """Initialise firmware state, RNG, and profile tracking."""
        self.config = config
        self._rng = random.Random(config.seed)  # noqa: S311 - deterministic virtual sensor sim
        self.active_profile: str = "normal"
        self.sequence_number: int = 0
        self.profile_step_count: int = 0
        self.reconnect_count: int = 0
        self.boot_monotonic: float = time.monotonic()

    def _noise(self, amplitude: float) -> float:
        return self._rng.uniform(-amplitude, amplitude)

    def set_profile(self, profile: str) -> None:
        """Switch the active virtual device operating profile."""
        if profile not in VALID_PROFILES:
            raise ValueError(f"Unsupported profile {profile!r}; expected one of {VALID_PROFILES}")
        self.active_profile = profile
        self.profile_step_count = 0

    def handle_command_payload(self, raw_payload: bytes | str | None) -> tuple[bool, str]:
        """Validate an incoming authenticated `set_profile` command payload."""
        if not raw_payload:
            return False, "empty command"
        try:
            data = json.loads(raw_payload)
        except Exception:
            return False, "invalid json"
        if not isinstance(data, dict):
            return False, "non-object command"
        expected = self.config.command_token or ""
        supplied = str(data.get("command_token") or "")
        if not expected or expected != supplied:
            return False, "unauthorized token"
        if data.get("command") != "set_profile":
            return False, "unsupported command"
        profile = str(data.get("profile") or "").strip().lower()
        if profile not in VALID_PROFILES:
            return False, "unknown profile"
        self.set_profile(profile)
        return True, profile

    def next_reading(self, now_utc: datetime | None = None) -> tuple[dict[str, Any] | None, str]:
        """Advance device state by one cadence tick.

        Returns `(telemetry_payload, true_profile)` or `(None, "intermittent_disconnect")`
        on simulated every-5th-step disconnect drops while still incrementing
        `sequence_number`.
        """
        self.sequence_number += 1
        self.profile_step_count += 1
        profile = self.active_profile

        if profile == "intermittent_disconnect" and (self.profile_step_count % 5 == 0):
            return None, profile

        temp_c = 46.5 + self._noise(0.60)
        vib_mm_s = 2.10 + self._noise(0.18)
        press_bar = 5.00 + self._noise(0.12)

        if profile == "temperature_spike" and (self.profile_step_count % 4 == 0):
            temp_c += 42.0 + self._noise(4.0)
        elif profile == "vibration_anomaly" and (self.profile_step_count % 4 == 0):
            vib_mm_s += 14.5 + self._noise(2.0)
        elif profile == "gradual_drift":
            ramp = float(self.profile_step_count)
            temp_c += min(ramp * 0.40, 30.0)
            vib_mm_s += min(ramp * 0.05, 4.0)
            press_bar += min(ramp * 0.04, 3.5)
        elif profile == "abrupt_drift":
            temp_c += 18.0
            vib_mm_s += 2.8
            press_bar += 2.4

        ts = format_utc_timestamp(now_utc or datetime.now(timezone.utc))
        payload = {
            "device_id": self.config.device_id,
            "machine_id": self.config.machine_id,
            "sequence_number": self.sequence_number,
            "timestamp": ts,
            "temperature_c": round(_clamp(temp_c, -40.0, 150.0), 2),
            "vibration_mm_s": round(_clamp(vib_mm_s, 0.0, 50.0), 3),
            "pressure_bar": round(_clamp(press_bar, 0.0, 20.0), 3),
        }
        return payload, profile

    def build_heartbeat(
        self, now_utc: datetime | None = None, status: str = "online"
    ) -> dict[str, Any]:
        """Build periodic device heartbeat payload."""
        ts = format_utc_timestamp(now_utc or datetime.now(timezone.utc))
        return {
            "device_id": self.config.device_id,
            "machine_id": self.config.machine_id,
            "sequence_number": self.sequence_number,
            "timestamp": ts,
            "active_profile": self.active_profile,
            "wifi_rssi_dbm": -52,
            "uptime_s": int(max(0.0, time.monotonic() - self.boot_monotonic)),
            "reconnect_count": self.reconnect_count,
            "status": status,
        }


class MqttWokwiRunner:
    """Threaded MQTT client running `VirtualWokwiEsp32` against a broker."""

    def __init__(self, config: WokwiEmulatorConfig) -> None:
        """Initialise MQTT client, LWT, and virtual ESP32 instance."""
        self.config = config
        self.device = VirtualWokwiEsp32(config)
        self._stop = threading.Event()
        self._connected = threading.Event()
        self.telemetry_published: int = 0
        self.heartbeats_published: int = 0
        self.commands_applied: int = 0
        self.commands_rejected: int = 0

        self._client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=f"wokwi-sim-{config.device_id}",
        )
        if config.username is not None:
            self._client.username_pw_set(config.username, config.password)
        if config.use_tls:
            self._client.tls_set(ca_certs=config.ca_certs, cert_reqs=ssl.CERT_REQUIRED)
        lwt_payload = json.dumps(
            {
                "device_id": config.device_id,
                "machine_id": config.machine_id,
                "status": "offline",
            }
        )
        self._client.will_set(config.heartbeat_topic, lwt_payload, qos=config.qos, retain=False)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def _on_connect(
        self,
        client: Any,
        _userdata: Any,
        _flags: Any,
        reason_code: Any,
        _props: Any,
    ) -> None:
        rc = int(getattr(reason_code, "value", reason_code))
        if rc == 0:
            self._connected.set()
            client.subscribe(self.config.command_topic, qos=self.config.qos)
        else:
            logger.error("wokwi emulator connect rejected", extra={"fields": {"rc": rc}})

    def _on_disconnect(
        self,
        _client: Any,
        _userdata: Any,
        _flags: Any,
        _reason_code: Any,
        _props: Any,
    ) -> None:
        self._connected.clear()

    def _on_message(self, _client: Any, _userdata: Any, msg: Any) -> None:
        ok, detail = self.device.handle_command_payload(msg.payload)
        if ok:
            self.commands_applied += 1
            self.publish_heartbeat_now()
        else:
            self.commands_rejected += 1
            logger.warning("rejected device command", extra={"fields": {"reason": detail}})

    def connect(self, timeout_s: float = 10.0) -> None:
        """Connect to MQTT broker and wait for CONNACK."""
        self._client.connect(self.config.broker_host, self.config.broker_port, keepalive=60)
        self._client.loop_start()
        if not self._connected.wait(timeout=timeout_s):
            raise TimeoutError(
                f"Wokwi emulator could not connect to MQTT {self.config.broker_host}:"
                f"{self.config.broker_port}"
            )
        self.publish_heartbeat_now()

    def publish_heartbeat_now(self) -> None:
        """Publish an immediate heartbeat JSON message."""
        hb = self.device.build_heartbeat()
        info = self._client.publish(
            self.config.heartbeat_topic,
            json.dumps(hb),
            qos=self.config.qos,
        )
        info.wait_for_publish(timeout=5.0)
        self.heartbeats_published += 1

    def step_once(self) -> tuple[dict[str, Any] | None, str]:
        """Publish one telemetry tick (simulating disconnect/reconnect when requested)."""
        payload, profile = self.device.next_reading()
        if payload is None:
            # Deliberately drop MQTT connection briefly and reconnect with preserved seq.
            self._client.disconnect()
            self._client.loop_stop()
            time.sleep(0.25)
            self.device.reconnect_count += 1
            self._connected.clear()
            self._client.connect(self.config.broker_host, self.config.broker_port, keepalive=60)
            self._client.loop_start()
            self._connected.wait(timeout=5.0)
            self.publish_heartbeat_now()
            return None, profile

        info = self._client.publish(
            self.config.telemetry_topic,
            json.dumps(payload),
            qos=self.config.qos,
        )
        info.wait_for_publish(timeout=5.0)
        self.telemetry_published += 1
        return payload, profile

    def stop(self) -> None:
        """Stop MQTT loop cleanly."""
        self._stop.set()
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            logger.debug("wokwi emulator disconnect cleanup warning", exc_info=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone Wokwi emulator execution."""
    parser = argparse.ArgumentParser(description="DriftGuard M10.1 Virtual Wokwi ESP32 Emulator")
    parser.add_argument("--broker-host", default=os.environ.get("MQTT_BROKER_HOST", "localhost"))
    parser.add_argument(
        "--broker-port", type=int, default=int(os.environ.get("MQTT_BROKER_PORT", "1883"))
    )
    parser.add_argument("--use-tls", action="store_true")
    parser.add_argument("--username", default=os.environ.get("MQTT_USERNAME") or None)
    parser.add_argument("--password", default=os.environ.get("MQTT_PASSWORD") or None)
    parser.add_argument("--namespace", default=os.environ.get("MQTT_NAMESPACE", "driftguard/demo"))
    parser.add_argument("--device-id", default="esp32-wokwi-01")
    parser.add_argument("--machine-id", default="machine_01")
    parser.add_argument("--profile", choices=VALID_PROFILES, default="normal")
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--command-token", default=os.environ.get("MQTT_COMMAND_TOKEN") or None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the virtual Wokwi ESP32 MQTT publisher for a bounded duration."""
    args = parse_args(argv)
    configure_logging("INFO")
    cfg = WokwiEmulatorConfig(
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        use_tls=bool(args.use_tls),
        username=args.username,
        password=args.password,
        namespace=args.namespace,
        device_id=args.device_id,
        machine_id=args.machine_id,
        publish_interval_s=args.interval_s,
        command_token=args.command_token,
    )
    runner = MqttWokwiRunner(cfg)
    runner.device.set_profile(args.profile)
    runner.connect()
    t0 = time.monotonic()
    last_hb = t0
    try:
        while (time.monotonic() - t0) < args.duration_s:
            runner.step_once()
            now = time.monotonic()
            if (now - last_hb) >= cfg.heartbeat_interval_s:
                runner.publish_heartbeat_now()
                last_hb = now
            time.sleep(cfg.publish_interval_s)
    finally:
        runner.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
