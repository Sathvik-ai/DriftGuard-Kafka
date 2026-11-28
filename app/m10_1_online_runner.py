"""M10.1 Continuous Online Multi-Phase Validation & Stress Orchestrator.

Executes a continuous online multi-phase virtual IoT pipeline run (15 minutes at
default `duration_scale=1.0`, or accelerated via `--duration-scale` for smoke
tests):

    Wokwi Virtual ESP32 (`app.wokwi_emulator`)
        -> Authenticated + ACL-enforced MQTT Broker (`eclipse-mosquitto:2`)
        -> Mac MQTT-Kafka Bridge (`app.mqtt_bridge`)
        -> Apache Kafka (`research.telemetry.raw`, `device_status`)
        -> Streaming Multi-Model Inference (`app.streaming_inference`, MPS/CPU)
           + M2 Static Detector (`app.detector`)
        -> Kafka (`alerts`, `drift`, `model_scores`)
        -> Dashboard Publisher (`app.dashboard_publisher`)
        -> MQTT (`<namespace>/dashboard/*`) -> `DashboardMqttSubscriber`

Generates 9 immutable artefacts inside `reports/m10_1/<run_id>/`:
  1. `summary_report.json`
  2. `summary_report.md`
  3. `events.jsonl`
  4. `scenario_truth.jsonl` (strictly post-hoc; never visible to detectors)
  5. `model_metrics.csv`
  6. `drift_events.csv`
  7. `alerts.csv`
  8. `device_status.json`
  9. `hardware_memory_report.json`
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import paho.mqtt.client as mqtt
from confluent_kafka import Consumer, KafkaError, Producer
from paho.mqtt.enums import CallbackAPIVersion

from app.config import Settings, load_settings
from app.dashboard_publisher import DashboardPublisherConfig, run_dashboard_publisher
from app.detector import DetectorStats, ThresholdConfig, ThresholdDetector, process_messages
from app.logging_config import configure_logging
from app.mqtt_bridge import (
    MqttBridgeConfig,
    MqttKafkaBridge,
    publish_profile_command,
    run_bridge_loop,
)
from app.producer import DeliveryTracker, build_producer_config
from app.schemas import format_utc_timestamp
from app.streaming_inference import (
    LiveModelSuite,
    StreamingInferenceConfig,
    StreamingInferenceEngine,
    collect_memory_metrics_mib,
    generate_normal_iot_splits,
    run_streaming_inference_loop,
)
from app.wokwi_emulator import MqttWokwiRunner, WokwiEmulatorConfig
from dashboard.state_store import (
    DashboardMqttSettings,
    DashboardMqttSubscriber,
    DashboardStateStore,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseSpec:
    """One scenario phase in the M10.1 continuous online protocol."""

    phase_id: str
    profile: str
    duration_s: float
    description: str


DEFAULT_PHASES_15MIN: tuple[PhaseSpec, ...] = (
    PhaseSpec("A_normal_warmup", "normal", 120.0, "Stationary operating baseline"),
    PhaseSpec(
        "B_temperature_spike", "temperature_spike", 120.0, "Periodic +42C spikes (every 4th)"
    ),
    PhaseSpec("C_vibration_anomaly", "vibration_anomaly", 120.0, "Periodic +14.5 mm/s bursts"),
    PhaseSpec("D_gradual_drift", "gradual_drift", 180.0, "Slow linear regime drift"),
    PhaseSpec("E_abrupt_drift", "abrupt_drift", 120.0, "Immediate persistent regime shift"),
    PhaseSpec("F_intermittent_disconnect", "intermittent_disconnect", 120.0, "Drop every 5th tick"),
    PhaseSpec("G_recovery_normal", "normal", 120.0, "Post-fault recovery baseline"),
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_tcp_port(host: str, port: int, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"TCP port {host}:{port} did not become ready within {timeout_s}s")


def _ensure_kafka_topics(settings: Settings) -> None:
    """Create all M10.1 Kafka topics inside the active Docker Kafka broker if needed."""
    topics = (
        settings.topic,
        settings.alerts_topic,
        settings.drift_topic,
        settings.model_scores_topic,
        settings.device_status_topic,
    )
    script = Path("scripts/create_topic.sh")
    bash_bin = shutil.which("bash") or "/bin/bash"
    if not script.exists():
        return
    for topic in topics:
        subprocess.run(  # noqa: S603
            [bash_bin, str(script), "--topic", topic],
            check=False,
            capture_output=True,
            text=True,
        )


class LocalAuthMosquittoBroker:
    """Ephemeral Docker `eclipse-mosquitto:2` broker with strict auth + ACLs."""

    def __init__(self, namespace: str) -> None:
        """Initialise ephemeral credentials and container metadata."""
        self.port = _find_free_port()
        self.namespace = namespace
        self.wokwi_user = "wokwi_device"
        self.wokwi_pass = secrets.token_hex(12)
        self.backend_user = "mac_backend"
        self.backend_pass = secrets.token_hex(12)
        self.viewer_user = "streamlit_readonly"
        self.viewer_pass = secrets.token_hex(12)
        self._tmp_dir: Path | None = None
        self._container_name = f"driftguard-m10-1-mosquitto-{self.port}"
        self._docker_bin = shutil.which("docker") or "docker"

    def __enter__(self) -> LocalAuthMosquittoBroker:
        """Start authenticated + ACL-enforced Mosquitto Docker container."""
        tmp_base = Path("runs") / "mosquitto_tmp"
        tmp_base.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="dg_mosq_", dir=str(tmp_base))).resolve()
        tmp.chmod(0o755)
        self._tmp_dir = tmp
        conf_path = tmp / "mosquitto.conf"
        acl_path = tmp / "acl.conf"

        conf_path.write_text(
            "\n".join(
                [
                    "listener 1883 0.0.0.0",
                    "allow_anonymous false",
                    "password_file /mosquitto/config/passwords.txt",
                    "acl_file /mosquitto/config/acl.conf",
                    "persistence false",
                    "log_type error",
                    "log_type warning",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        acl_path.write_text(
            "\n".join(
                [
                    f"user {self.wokwi_user}",
                    f"topic write {self.namespace}/+/telemetry",
                    f"topic write {self.namespace}/+/heartbeat",
                    f"topic read {self.namespace}/+/command",
                    "",
                    f"user {self.backend_user}",
                    f"topic read {self.namespace}/+/telemetry",
                    f"topic read {self.namespace}/+/heartbeat",
                    f"topic write {self.namespace}/+/command",
                    f"topic write {self.namespace}/dashboard/#",
                    "",
                    f"user {self.viewer_user}",
                    f"topic read {self.namespace}/dashboard/#",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        subprocess.run(  # noqa: S603
            [
                self._docker_bin,
                "run",
                "--rm",
                "-v",
                f"{tmp}:/mosquitto/config",
                "--entrypoint",
                "/bin/sh",
                "eclipse-mosquitto:2",
                "-c",
                (
                    f"mosquitto_passwd -b -c /mosquitto/config/passwords.txt "
                    f"{self.wokwi_user} {self.wokwi_pass} && "
                    f"mosquitto_passwd -b /mosquitto/config/passwords.txt "
                    f"{self.backend_user} {self.backend_pass} && "
                    f"mosquitto_passwd -b /mosquitto/config/passwords.txt "
                    f"{self.viewer_user} {self.viewer_pass} && "
                    "chmod 0644 /mosquitto/config/passwords.txt"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(  # noqa: S603
            [self._docker_bin, "rm", "-f", self._container_name],
            check=False,
            capture_output=True,
        )
        subprocess.run(  # noqa: S603
            [
                self._docker_bin,
                "run",
                "-d",
                "--rm",
                "--name",
                self._container_name,
                "-p",
                f"127.0.0.1:{self.port}:1883",
                "-v",
                f"{tmp}:/mosquitto/config",
                "eclipse-mosquitto:2",
                "mosquitto",
                "-c",
                "/mosquitto/config/mosquitto.conf",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        _wait_tcp_port("127.0.0.1", self.port, timeout_s=15.0)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        """Tear down ephemeral Mosquitto container and temp credentials."""
        subprocess.run(  # noqa: S603
            [self._docker_bin, "rm", "-f", self._container_name],
            check=False,
            capture_output=True,
        )
        if self._tmp_dir is not None and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)


def _is_truth_anomaly(profile: str, profile_step: int) -> bool:
    """Return whether virtual firmware step generated an injected excursion/regime shift."""
    if profile == "temperature_spike":
        return profile_step % 4 == 0
    if profile == "vibration_anomaly":
        return profile_step % 4 == 0
    if profile == "gradual_drift":
        return profile_step >= 5
    return profile == "abrupt_drift"


def _classification_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_true, y_pred, strict=True):
        if yt and yp:
            tp += 1
        elif not yt and yp:
            fp += 1
        elif yt and not yp:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
    }


def _collect_kafka_topic_records(
    settings: Settings,
    topics: Sequence[str],
    stop_event: threading.Event,
    out_buckets: dict[str, list[dict[str, Any]]],
    run_id: str,
) -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": settings.bootstrap_servers,
            "group.id": f"driftguard-m10-1-collector-{run_id}",
            "client.id": f"{settings.client_id}-collector",
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe(list(topics))
    try:
        while not stop_event.is_set():
            msg = consumer.poll(0.25)
            if msg is None:
                continue
            err = msg.error()
            if err is not None:
                if err.code() != KafkaError._PARTITION_EOF:
                    logger.warning("collector error: %s", err)
                continue
            raw = msg.value()
            if raw is None:
                continue
            try:
                doc = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                logger.debug("collector skipped non-JSON record: %s", exc)
                continue
            msg_topic = msg.topic()
            if isinstance(doc, dict) and msg_topic is not None and msg_topic in out_buckets:
                out_buckets[msg_topic].append(doc)
    finally:
        consumer.close()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


def run_m10_1_online_protocol(
    *,
    duration_scale: float = 1.0,
    publish_interval_s: float = 1.0,
    tsfm_every_n_events: int = 10,
    device: str = "auto",
    enable_tsfms: bool = True,
    inject_invalid_count: int = 3,
    output_root: Path = Path("reports/m10_1"),
    run_id: str | None = None,
) -> Path:
    """Execute the end-to-end M10.1 continuous online protocol and write all 9 artefacts."""
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output_root / rid
    out_dir.mkdir(parents=True, exist_ok=True)

    kafka_settings = load_settings()
    _ensure_kafka_topics(kafka_settings)

    namespace = f"driftguard/m10_1_{rid}"
    command_token = secrets.token_urlsafe(12)
    device_id = "esp32-wokwi-01"
    machine_id = "machine_01"

    x_train, x_val = generate_normal_iot_splits()
    inf_cfg = StreamingInferenceConfig(
        device=device,
        tsfm_every_n_events=tsfm_every_n_events,
        enable_tsfms=enable_tsfms,
        consumer_group=f"driftguard-m10-1-inference-{rid}",
        auto_offset_reset="latest",
    )
    suite = LiveModelSuite(inf_cfg, x_train, x_val)
    engine = StreamingInferenceEngine(suite, kafka_settings)

    phases = [
        PhaseSpec(
            p.phase_id,
            p.profile,
            max(1.5, round(p.duration_s * duration_scale, 2)),
            p.description,
        )
        for p in DEFAULT_PHASES_15MIN
    ]
    total_duration_s = sum(p.duration_s for p in phases)
    grace_s = 4.0

    with LocalAuthMosquittoBroker(namespace=namespace) as broker:
        mqtt_cfg = MqttBridgeConfig(
            broker_host="127.0.0.1",
            broker_port=broker.port,
            use_tls=False,
            username=broker.backend_user,
            password=broker.backend_pass,
            mqtt_topic=f"{namespace}/+/telemetry",
            mqtt_client_id=f"driftguard-bridge-{rid}",
            qos=1,
            status_report_dir=out_dir,
            mqtt_namespace=namespace,
            heartbeat_topic=f"{namespace}/+/heartbeat",
            command_token=command_token,
            publish_device_status_topic=True,
        )
        bridge = MqttKafkaBridge(mqtt_config=mqtt_cfg, kafka_settings=kafka_settings)

        pub_cfg = DashboardPublisherConfig(
            broker_host="127.0.0.1",
            broker_port=broker.port,
            use_tls=False,
            username=broker.backend_user,
            password=broker.backend_pass,
            namespace=namespace,
            consumer_group=f"driftguard-m10-1-dashpub-{rid}",
            auto_offset_reset="latest",
            snapshot_interval_s=2.0,
            local_snapshot_path=out_dir.parent / "latest_snapshot.json",
        )

        # Read-only Streamlit subscriber proving role-separated ACL works end-to-end
        viewer_store = DashboardStateStore(max_events=2000, max_incidents=500)
        viewer_sub = DashboardMqttSubscriber(
            DashboardMqttSettings(
                broker_host="127.0.0.1",
                broker_port=broker.port,
                use_tls=False,
                username=broker.viewer_user,
                password=broker.viewer_pass,
                namespace=namespace,
                client_id=f"streamlit-ro-{rid}",
            ),
            viewer_store,
        )

        collector_stop = threading.Event()
        buckets: dict[str, list[dict[str, Any]]] = {
            kafka_settings.alerts_topic: [],
            kafka_settings.drift_topic: [],
            kafka_settings.model_scores_topic: [],
            kafka_settings.device_status_topic: [],
        }

        worker_dur = total_duration_s + 120.0
        workers_stop = threading.Event()
        bridge_res_box: dict[str, Any] = {}
        inf_res_box: dict[str, Any] = {}
        dash_res_box: dict[str, Any] = {}

        def _run_bridge() -> None:
            bridge_res_box["report"] = run_bridge_loop(
                bridge=bridge,
                mqtt_cfg=mqtt_cfg,
                duration_s=worker_dur,
                status_output_dir=out_dir,
                run_id=rid,
                external_stop_event=workers_stop,
            )

        def _run_detector() -> None:
            det_settings = Settings(
                bootstrap_servers=kafka_settings.bootstrap_servers,
                topic=kafka_settings.topic,
                consumer_group=f"driftguard-m10-1-detector-{rid}",
                client_id=f"{kafka_settings.client_id}-m10-1-detector",
                machine_count=kafka_settings.machine_count,
                log_level=kafka_settings.log_level,
                alerts_topic=kafka_settings.alerts_topic,
                drift_topic=kafka_settings.drift_topic,
                model_scores_topic=kafka_settings.model_scores_topic,
                device_status_topic=kafka_settings.device_status_topic,
            )
            det_consumer = Consumer(
                {
                    "bootstrap.servers": det_settings.bootstrap_servers,
                    "group.id": det_settings.consumer_group,
                    "client.id": det_settings.client_id,
                    "auto.offset.reset": "latest",
                    "enable.auto.commit": True,
                    "enable.auto.offset.store": False,
                }
            )
            det_consumer.subscribe([det_settings.topic])
            det_producer = Producer(build_producer_config(det_settings))
            try:
                process_messages(
                    consumer=det_consumer,
                    producer=det_producer,
                    alerts_topic=det_settings.alerts_topic,
                    detector=ThresholdDetector(ThresholdConfig()),
                    stats=DetectorStats(),
                    alert_tracker=DeliveryTracker(),
                    stop_event=workers_stop,
                    limit=None,
                    idle_timeout_s=worker_dur,
                )
            finally:
                det_consumer.close()
                det_producer.flush(5.0)

        def _run_inf() -> None:
            inf_res_box["summary"] = run_streaming_inference_loop(
                engine,
                kafka_settings,
                duration_s=worker_dur,
                external_stop_event=workers_stop,
            )

        def _run_dash_pub() -> None:
            dash_res_box["health"] = run_dashboard_publisher(
                kafka_settings,
                pub_cfg,
                duration_s=worker_dur,
                external_stop_event=workers_stop,
            )

        threads = [
            threading.Thread(
                target=_collect_kafka_topic_records,
                args=(kafka_settings, list(buckets.keys()), collector_stop, buckets, rid),
                daemon=True,
            ),
            threading.Thread(target=_run_bridge, daemon=True),
            threading.Thread(target=_run_detector, daemon=True),
            threading.Thread(target=_run_inf, daemon=True),
            threading.Thread(target=_run_dash_pub, daemon=True),
        ]
        for th in threads:
            th.start()
        viewer_sub.start()
        # Allow Kafka consumer groups & MQTT subscriptions to stabilize
        time.sleep(3.0)

        emu_cfg = WokwiEmulatorConfig(
            broker_host="127.0.0.1",
            broker_port=broker.port,
            use_tls=False,
            username=broker.wokwi_user,
            password=broker.wokwi_pass,
            namespace=namespace,
            device_id=device_id,
            machine_id=machine_id,
            publish_interval_s=publish_interval_s,
            heartbeat_interval_s=5.0,
            command_token=command_token,
            seed=42,
        )
        wokwi = MqttWokwiRunner(emu_cfg)
        wokwi.connect(timeout_s=10.0)

        truth_records: list[dict[str, Any]] = []
        hw_mem_timeline: list[dict[str, Any]] = [
            {
                "stage": "baseline_after_init",
                **collect_memory_metrics_mib(suite.hardware["mps_available"]),
            }
        ]
        run_started_utc = format_utc_timestamp(datetime.now(timezone.utc))
        last_hb = time.monotonic()

        try:
            for p_idx, phase in enumerate(phases):
                if p_idx == 0:
                    wokwi.device.set_profile(phase.profile)
                    wokwi.publish_heartbeat_now()
                else:
                    # Issue authenticated remote MQTT command like real Dashboard UI -> Wokwi ESP32!
                    publish_profile_command(
                        broker_host="127.0.0.1",
                        broker_port=broker.port,
                        namespace=namespace,
                        device_id=device_id,
                        profile=phase.profile,
                        command_token=command_token,
                        username=broker.backend_user,
                        password=broker.backend_pass,
                        qos=1,
                    )
                    time.sleep(0.35)

                p_t0 = time.monotonic()
                while (time.monotonic() - p_t0) < phase.duration_s:
                    payload, actual_prof = wokwi.step_once()
                    truth_records.append(
                        {
                            "phase_id": phase.phase_id,
                            "expected_profile": phase.profile,
                            "reported_profile": actual_prof,
                            "sequence_number": wokwi.device.sequence_number,
                            "timestamp": payload["timestamp"]
                            if payload
                            else format_utc_timestamp(datetime.now(timezone.utc)),
                            "published": payload is not None,
                            "true_anomaly": (
                                _is_truth_anomaly(phase.profile, wokwi.device.profile_step_count)
                                if payload is not None
                                else False
                            ),
                            "drift_regime": phase.profile in ("gradual_drift", "abrupt_drift"),
                        }
                    )
                    now = time.monotonic()
                    if (now - last_hb) >= emu_cfg.heartbeat_interval_s:
                        wokwi.publish_heartbeat_now()
                        last_hb = now
                    time.sleep(publish_interval_s)

                hw_mem_timeline.append(
                    {
                        "stage": f"end_{phase.phase_id}",
                        "timestamp": format_utc_timestamp(datetime.now(timezone.utc)),
                        **collect_memory_metrics_mib(suite.hardware["mps_available"]),
                    }
                )

            # Inject `inject_invalid_count` malformed MQTT payloads from wokwi_device
            if inject_invalid_count > 0:
                bad_cli = mqtt.Client(
                    callback_api_version=CallbackAPIVersion.VERSION2,
                    client_id=f"wokwi-bad-{rid}",
                )
                bad_cli.username_pw_set(broker.wokwi_user, broker.wokwi_pass)
                bad_cli.connect("127.0.0.1", broker.port, keepalive=15)
                bad_cli.loop_start()
                samples = [
                    b"{not_valid_json",
                    json.dumps({"device_id": device_id}).encode(),
                    json.dumps(
                        {
                            "device_id": device_id,
                            "machine_id": machine_id,
                            "sequence_number": 999999,
                            "timestamp": format_utc_timestamp(datetime.now(timezone.utc)),
                            "temperature_c": 999.0,
                            "vibration_mm_s": 2.0,
                            "pressure_bar": 5.0,
                        }
                    ).encode(),
                ]
                for i in range(inject_invalid_count):
                    info = bad_cli.publish(
                        emu_cfg.telemetry_topic, samples[i % len(samples)], qos=1
                    )
                    info.wait_for_publish(timeout=3.0)
                bad_cli.loop_stop()
                bad_cli.disconnect()

            # Allow downstream Kafka & MQTT streams to drain before signaling workers
            time.sleep(grace_s)
        finally:
            wokwi.stop()
            workers_stop.set()
            for th in threads[1:]:
                th.join(timeout=15.0)
            collector_stop.set()
            threads[0].join(timeout=5.0)
            viewer_sub.stop()

    run_finished_utc = format_utc_timestamp(datetime.now(timezone.utc))

    # Build strict post-hoc alignment by `sequence_number`
    truth_by_seq: dict[int, dict[str, Any]] = {
        int(r["sequence_number"]): r for r in truth_records if r["published"]
    }
    score_events = buckets[kafka_settings.model_scores_topic]
    alert_events = buckets[kafka_settings.alerts_topic]
    drift_events = buckets[kafka_settings.drift_topic]

    events_path = out_dir / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as handle:
        for ev in score_events:
            handle.write(json.dumps(ev) + "\n")

    truth_path = out_dir / "scenario_truth.jsonl"
    with truth_path.open("w", encoding="utf-8") as handle:
        for tr in truth_records:
            handle.write(json.dumps(tr) + "\n")

    # Compute per-model online precision / recall / F1 / FPR + latency (post-hoc)
    model_ids = sorted(suite.slots.keys())
    model_metric_rows: list[dict[str, Any]] = []
    for mid in model_ids:
        slot = suite.slots[mid]
        y_true: list[bool] = []
        y_pred: list[bool] = []
        lats: list[float] = []
        normal_fp = normal_tot = 0
        drift_tp = drift_tot = 0
        for ev in score_events:
            seq = ev.get("sequence_number")
            if seq is None or int(seq) not in truth_by_seq:
                continue
            tr = truth_by_seq[int(seq)]
            m_info = ev.get("models", {}).get(mid, {})
            if m_info.get("status") == "unavailable":
                continue
            if m_info.get("evaluated_this_event") or "anomaly" in m_info:
                y_true.append(bool(tr["true_anomaly"]))
                pred = bool(m_info.get("anomaly", False))
                y_pred.append(pred)
                if m_info.get("evaluated_this_event"):
                    lats.append(float(m_info.get("latency_ms", 0.0)))
                if not tr["true_anomaly"] and not tr["drift_regime"]:
                    normal_tot += 1
                    if pred:
                        normal_fp += 1
                if tr["drift_regime"]:
                    drift_tot += 1
                    if pred:
                        drift_tp += 1
        cls_m = (
            _classification_metrics(y_true, y_pred)
            if y_true
            else {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "fpr": 0.0,
            }
        )
        lat_arr = np.asarray(lats, dtype=np.float64)
        model_metric_rows.append(
            {
                "model_id": mid,
                "family": slot.family,
                "channel_mode": slot.channel_mode,
                "device": slot.device,
                "status": slot.status,
                "calibrated_threshold": round(slot.last_threshold, 6),
                "events_scored": len(y_true),
                "evaluations_executed": len(lats),
                **cls_m,
                "normal_phase_fpr": round(normal_fp / normal_tot, 4) if normal_tot else None,
                "drift_phase_trigger_rate": round(drift_tp / drift_tot, 4) if drift_tot else None,
                "mean_latency_ms": round(float(np.mean(lat_arr)), 3) if lat_arr.size else None,
                "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 3)
                if lat_arr.size
                else None,
            }
        )

    _write_csv(
        out_dir / "model_metrics.csv",
        model_metric_rows,
        [
            "model_id",
            "family",
            "channel_mode",
            "device",
            "status",
            "calibrated_threshold",
            "events_scored",
            "evaluations_executed",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "f1",
            "fpr",
            "normal_phase_fpr",
            "drift_phase_trigger_rate",
            "mean_latency_ms",
            "p95_latency_ms",
        ],
    )
    _write_csv(
        out_dir / "drift_events.csv",
        drift_events,
        [
            "drift_id",
            "event_id",
            "machine_id",
            "feature",
            "feature_value",
            "adwin_estimation",
            "adwin_width",
            "delta",
            "event_timestamp",
            "detected_utc",
            "stream_position",
            "machine_position",
        ],
    )
    _write_csv(
        out_dir / "alerts.csv",
        alert_events,
        [
            "alert_id",
            "event_id",
            "machine_id",
            "timestamp",
            "detected_utc",
            "detector",
            "reason",
            "triggered_rules",
            "temperature_c",
            "vibration_mm_s",
            "pressure_bar",
        ],
    )

    bridge_report = bridge_res_box.get("report") or bridge.build_status_report(run_id=rid)
    (out_dir / "device_status.json").write_text(
        json.dumps(bridge_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    hw_mem_report = {
        "run_id": rid,
        "hardware": suite.hardware,
        "memory_timeline_mib": hw_mem_timeline,
        "inference_summary": inf_res_box.get("summary") or engine.build_runtime_summary(),
        "legacy_checkpoint_audit": suite.legacy_audit,
    }
    (out_dir / "hardware_memory_report.json").write_text(
        json.dumps(hw_mem_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    viewer_snap = viewer_store.snapshot_view()
    summary = {
        "run_id": rid,
        "started_utc": run_started_utc,
        "finished_utc": run_finished_utc,
        "duration_scale": duration_scale,
        "planned_duration_s": round(total_duration_s, 2),
        "scenario_summary": [
            {
                "phase_id": p.phase_id,
                "profile": p.profile,
                "duration_s": p.duration_s,
                "description": p.description,
            }
            for p in phases
        ],
        "wokwi_emulator": {
            "telemetry_published": wokwi.telemetry_published,
            "heartbeats_published": wokwi.heartbeats_published,
            "commands_applied": wokwi.commands_applied,
            "commands_rejected": wokwi.commands_rejected,
            "reconnect_count": wokwi.device.reconnect_count,
            "final_sequence_number": wokwi.device.sequence_number,
        },
        "bridge_status": {
            "messages_received": bridge_report["messages_received"],
            "messages_forwarded": bridge_report["messages_forwarded"],
            "invalid_messages": bridge_report["invalid_messages"],
            "sequence_gaps": bridge_report["sequence_gaps"],
            "reconnection_count": bridge_report["reconnection_count"],
            "ingestion_latency_ms": bridge_report["ingestion_latency_ms"],
        },
        "pipeline_counts": {
            "model_score_events_kafka": len(score_events),
            "alert_events_kafka": len(alert_events),
            "drift_events_kafka": len(drift_events),
            "viewer_messages_ingested_ro": viewer_snap["messages_ingested"],
            "viewer_telemetry_ro": len(viewer_snap["telemetry"]),
            "viewer_model_scores_ro": len(viewer_snap["model_scores"]),
        },
        "hardware": suite.hardware,
        "model_metrics": model_metric_rows,
        "dashboard_health": dash_res_box.get("health", {}),
        "note": (
            "All sensor outputs are SIMULATED virtual ESP32 telemetry; "
            "offline M9.1 and live M10.1 scopes are kept strictly separate."
        ),
    }
    (out_dir / "summary_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "summary_report.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
    return out_dir


def _render_summary_markdown(summary: Mapping[str, Any]) -> str:
    br = summary["bridge_status"]
    wk = summary["wokwi_emulator"]
    hw = summary["hardware"]
    pc = summary["pipeline_counts"]
    lat = br["ingestion_latency_ms"]
    lines = [
        f"# DriftGuard-Kafka M10.1 Online End-to-End Validation Report — `{summary['run_id']}`",
        "",
        f"Started: `{summary['started_utc']}` | Finished: `{summary['finished_utc']}` | "
        f"Duration: `{summary['planned_duration_s']}s` (`scale={summary['duration_scale']}`)",
        "",
        "> **Scope Note**: All sensor values originate from the virtual ESP32 state machine "
        "matching `iot/wokwi_esp32/sketch.ino`. Offline SKAB/TSB-AD-M M9.1 metrics stay separate.",
        "",
        "## 1. End-to-End Pipeline & Device Health",
        "",
        "| Metric | Observed Value |",
        "|---|---|",
        f"| Wokwi Telemetry Published | `{wk['telemetry_published']}` |",
        f"| Wokwi Heartbeats Published | `{wk['heartbeats_published']}` |",
        f"| Authenticated `set_profile` Commands Applied | `{wk['commands_applied']}` |",
        f"| Bridge Messages Received (MQTT) | `{br['messages_received']}` |",
        f"| Bridge Forwarded (Kafka Confirmed) | `{br['messages_forwarded']}` |",
        f"| Invalid Messages Rejected | `{br['invalid_messages']}` |",
        f"| Sequence Gaps Observed | `{br['sequence_gaps']}` |",
        f"| Reconnects (Bridge/Dev) | `{br['reconnection_count']}` / `{wk['reconnect_count']}` |",
        f"| Ingestion Latency Mean / P95 | `{lat['mean_ms']} / {lat['p95_ms']} ms` |",
        f"| Kafka Model Score Events | `{pc['model_score_events_kafka']}` |",
        f"| Alerts / ADWIN Drifts | `{pc['alert_events_kafka']}` / `{pc['drift_events_kafka']}` |",
        f"| RO Streamlit Subscriber Ingested | `{pc['viewer_messages_ingested_ro']}` |",
        f"| Compute Accelerator | `{hw['selected_dl_device']}` (mps={hw['mps_available']}) |",
        "",
        "## 2. Live 3-Channel Model Performance (`model_metrics.csv`)",
        "",
        "| Model | Family | Mode | Device | Scored | Prec | Recall | F1 | Normal FPR | Lat (ms) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["model_metrics"]:
        lines.append(
            f"| `{row['model_id']}` | `{row['family']}` | `{row['channel_mode']}` | "
            f"`{row['device']}` | {row['events_scored']} | {row['precision']} | "
            f"{row['recall']} | {row['f1']} | {row['normal_phase_fpr']} | "
            f"{row['mean_latency_ms']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the M10.1 continuous online protocol."""
    parser = argparse.ArgumentParser(description="DriftGuard-Kafka M10.1 Online Validation Runner")
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--publish-interval-s", type=float, default=1.0)
    parser.add_argument("--tsfm-every-n-events", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--disable-tsfms", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("reports/m10_1"))
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for M10.1 online validation."""
    args = parse_args(argv)
    configure_logging("INFO")
    out_dir = run_m10_1_online_protocol(
        duration_scale=args.duration_scale,
        publish_interval_s=args.publish_interval_s,
        tsfm_every_n_events=args.tsfm_every_n_events,
        device=args.device,
        enable_tsfms=not args.disable_tsfms,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(f"M10.1 online validation complete -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
