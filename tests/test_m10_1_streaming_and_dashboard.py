"""Unit and integration tests for DriftGuard-Kafka M10.1 streaming & dashboard modules."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from app.config import ConfigError, load_settings
from app.dashboard_publisher import DashboardPublisherConfig, DashboardPublisherState
from app.m5_lstm_autoencoder import DeviceInfo
from app.m8_tsfm_base import ForecastBatchOutput
from app.mqtt_bridge import (
    MqttBridgeConfig,
    MqttKafkaBridge,
    build_profile_command_payload,
    parse_mqtt_heartbeat_payload,
)
from app.schemas import EventLabel, SchemaValidationError, SourceType, TelemetryEvent
from app.streaming_inference import (
    LIVE_FEATURE_NAMES,
    LiveModelSuite,
    SchemaMismatchError,
    StreamingInferenceConfig,
    StreamingInferenceEngine,
    audit_legacy_checkpoints,
    collect_hardware_diagnostics,
    collect_memory_metrics_mib,
    generate_normal_iot_splits,
    validate_checkpoint_schema,
)
from app.wokwi_emulator import VirtualWokwiEsp32, WokwiEmulatorConfig
from dashboard.state_store import DashboardStateStore


class _FakeProducer:
    def __init__(self) -> None:
        self.produced: list[tuple[str, bytes, bytes]] = []
        self._cbs: list[tuple[Any, Any]] = []

    def produce(self, topic: str, key: bytes, value: bytes, on_delivery: Any = None) -> None:
        self.produced.append((topic, key, value))
        if on_delivery is not None:

            class _Msg:
                def __init__(self, k: bytes) -> None:
                    self._k = k

                def key(self) -> bytes:
                    return self._k

                def partition(self) -> int:
                    return 0

            self._cbs.append((on_delivery, _Msg(key)))

    def poll(self, timeout: float = 0.0) -> int:
        del timeout
        n = len(self._cbs)
        while self._cbs:
            cb, msg = self._cbs.pop(0)
            cb(None, msg)
        return n

    def flush(self, timeout: float = 10.0) -> int:
        del timeout
        return self.poll(0.0)


class _DummyForecastBackend:
    """Fast deterministic ForecastBackend stub for offline CI TSFM unit tests."""

    def __init__(self, supports_mv: bool = True) -> None:
        self._mv = supports_mv
        self._dev = DeviceInfo(
            requested="cpu",
            selected="cpu",
            is_apple_silicon=False,
            mps_built=False,
            mps_available=False,
            cuda_available=False,
            fallback_reason=None,
        )

    @property
    def model_id(self) -> str:
        return "dummy/tsfm"

    @property
    def license_name(self) -> str:
        return "Apache-2.0"

    @property
    def supports_multivariate(self) -> bool:
        return self._mv

    @property
    def device_info(self) -> DeviceInfo:
        return self._dev

    def forecast_multivariate_blocks(
        self, contexts_bdl: np.ndarray, horizon: int, batch_size: int
    ) -> ForecastBatchOutput:
        del batch_size
        means = np.mean(contexts_bdl[:, :, -min(8, contexts_bdl.shape[2]) :], axis=2, keepdims=True)
        pt = np.repeat(means, horizon, axis=2).astype(np.float32)
        return ForecastBatchOutput(
            point_forecast=pt,
            q_low=(pt - 1.5).astype(np.float32),
            q_high=(pt + 1.5).astype(np.float32),
        )


def test_checkpoint_schema_validation_rejects_8ch_and_1ch_checkpoints(tmp_path: Path) -> None:
    skab_ckpt = tmp_path / "m5" / "SKAB" / "lstm.pt"
    skab_ckpt.parent.mkdir(parents=True)
    torch.save(
        {
            "feature_names": [f"f{i}" for i in range(8)],
            "scaler_mean": np.zeros(8),
            "config": {"channel_mode": "multivariate"},
        },
        skab_ckpt,
    )
    tsb_ckpt = tmp_path / "m7" / "TSB-AD-M" / "tranad.pt"
    tsb_ckpt.parent.mkdir(parents=True)
    torch.save(
        {
            "feature_names": ["ch0"],
            "scaler_min": np.zeros(1),
            "config": {"channel_mode": "multivariate"},
        },
        tsb_ckpt,
    )

    with pytest.raises(SchemaMismatchError):
        validate_checkpoint_schema(
            LIVE_FEATURE_NAMES,
            torch.load(skab_ckpt, map_location="cpu", weights_only=False),
        )

    audit = audit_legacy_checkpoints(models_root=tmp_path, expected_features=LIVE_FEATURE_NAMES)
    assert len(audit) == 2
    assert all(not item["compatible_with_3ch_iot"] for item in audit)


def test_streaming_inference_3ch_label_isolation_and_tsfm_cadence(tmp_path: Path) -> None:
    x_train, x_val = generate_normal_iot_splits(n_train=90, n_val=90)
    cfg = StreamingInferenceConfig(
        device="cpu",
        models_dir=tmp_path / "m10_1_models",
        tsfm_every_n_events=3,
        tsfm_context_length=16,
        ae_window_size=8,
        tranad_window_size=6,
    )
    suite = LiveModelSuite(
        cfg,
        x_train,
        x_val,
        chronos_backend=_DummyForecastBackend(supports_mv=True),
        timesfm_backend=_DummyForecastBackend(supports_mv=False),
    )
    assert (cfg.models_dir / "live_3ch_manifest.json").exists()
    assert (cfg.models_dir / "lstm_autoencoder_3ch.pt").exists()
    assert (cfg.models_dir / "tcn_autoencoder_3ch.pt").exists()
    assert (cfg.models_dir / "tranad_3ch.pt").exists()

    kafka_settings = load_settings({})
    prod = _FakeProducer()
    engine = StreamingInferenceEngine(suite, kafka_settings, kafka_producer=prod)

    # Identical payload with different label MUST produce identical model scores
    ev_norm = TelemetryEvent(
        event_id=str(uuid.uuid4()),
        machine_id="machine_01",
        timestamp="2026-11-28T10:00:00.000Z",
        temperature_c=46.5,
        vibration_mm_s=2.1,
        pressure_bar=5.0,
        label=EventLabel.NORMAL,
        source_type=SourceType.VIRTUAL_IOT_WOKWI,
        device_id="esp32-wokwi-01",
        sequence_number=1,
    )
    out1 = engine.process_telemetry_bytes(ev_norm.to_json_bytes())
    assert out1 is not None
    assert out1["feature_schema"] == ["temperature_c", "vibration_mm_s", "pressure_bar"]
    # Event 1 evaluates TSFMs cleanly
    assert out1["models"]["chronos2_zero_shot"]["evaluated_this_event"] is True
    assert out1["models"]["timesfm3_zero_shot"]["evaluated_this_event"] is True
    assert set(out1["memory_mib"].keys()) == {
        "process_rss_mib",
        "mps_allocated_mib",
        "mps_driver_mib",
        "system_available_mib",
    }

    # Event 2 uses cached_skip for heavy TSFMs
    ev_2 = TelemetryEvent(
        event_id=str(uuid.uuid4()),
        machine_id="machine_01",
        timestamp="2026-11-28T10:00:01.000Z",
        temperature_c=46.6,
        vibration_mm_s=2.12,
        pressure_bar=5.01,
        label=EventLabel.SPIKE,  # Must NEVER alter inference scores!
        source_type=SourceType.VIRTUAL_IOT_WOKWI,
        device_id="esp32-wokwi-01",
        sequence_number=2,
    )
    out2 = engine.process_telemetry_bytes(ev_2.to_json_bytes())
    assert out2 is not None
    assert out2["models"]["chronos2_zero_shot"]["status"] == "cached_skip"
    assert out2["models"]["chronos2_zero_shot"]["evaluated_this_event"] is False

    # Inject sustained abrupt drift events and ensure ADWIN triggers & produces Kafka drift topic
    base_ts = datetime(2026, 11, 28, 10, 0, 0, tzinfo=timezone.utc)
    for seq in range(3, 95):
        ts_iso = (base_ts + timedelta(seconds=seq)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        ev_d = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            machine_id="machine_01",
            timestamp=ts_iso,
            temperature_c=90.0 if seq >= 40 else 46.5,
            vibration_mm_s=15.0 if seq >= 40 else 2.1,
            pressure_bar=9.0 if seq >= 40 else 5.0,
            label=EventLabel.NORMAL,
            source_type=SourceType.VIRTUAL_IOT_WOKWI,
            device_id="esp32-wokwi-01",
            sequence_number=seq,
        )
        engine.process_telemetry_bytes(ev_d.to_json_bytes())

    assert engine.process_telemetry_bytes(b"{malformed") is None
    engine.flush()
    assert engine.drift_events_emitted > 0
    assert any(t == kafka_settings.model_scores_topic for t, _k, _v in prod.produced)
    assert any(t == kafka_settings.drift_topic for t, _k, _v in prod.produced)
    summary = engine.build_runtime_summary()
    assert summary["events_processed"] == 94
    assert isinstance(collect_hardware_diagnostics("cpu"), dict)
    assert "process_rss_mib" in collect_memory_metrics_mib(use_mps=False)


def test_wokwi_profiles_heartbeat_and_authenticated_commands(tmp_path: Path) -> None:
    emu_cfg = WokwiEmulatorConfig(
        namespace="driftguard/unit",
        device_id="esp32-wokwi-01",
        machine_id="machine_01",
        command_token="secret-cmd-tok",  # noqa: S106
    )
    dev = VirtualWokwiEsp32(emu_cfg)

    ok, _ = dev.handle_command_payload(
        json.dumps(
            {"command": "set_profile", "profile": "temperature_spike", "command_token": "wrong"}
        )
    )
    assert not ok
    assert dev.active_profile == "normal"

    cmd_wire = build_profile_command_payload("temperature_spike", "secret-cmd-tok")
    ok2, prof2 = dev.handle_command_payload(cmd_wire)
    assert ok2 and prof2 == "temperature_spike"

    with pytest.raises(ConfigError):
        build_profile_command_payload("normal", "")
    with pytest.raises(SchemaValidationError):
        build_profile_command_payload("unknown_mode", "tok")

    # Intermittent disconnect skips every 5th tick while advancing sequence_number
    dev.set_profile("intermittent_disconnect")
    seqs = []
    drops = 0
    for _ in range(5):
        payload, _ = dev.next_reading()
        seqs.append(dev.sequence_number)
        if payload is None:
            drops += 1
    assert seqs == [1, 2, 3, 4, 5]
    assert drops == 1

    # Bridge processes heartbeat + LWT + confirmed device_status topic
    bridge_cfg = MqttBridgeConfig(
        broker_host="localhost",
        broker_port=1883,
        use_tls=False,
        username=None,
        password=None,
        mqtt_topic="driftguard/unit/+/telemetry",
        mqtt_client_id="bridge-test",
        qos=1,
        status_report_dir=tmp_path,
        publish_device_status_topic=True,
    )
    kafka_settings = load_settings({})
    prod = _FakeProducer()
    bridge = MqttKafkaBridge(bridge_cfg, kafka_settings, kafka_producer=prod)

    hb = dev.build_heartbeat()
    parsed_hb = parse_mqtt_heartbeat_payload(json.dumps(hb))
    assert parsed_hb.status == "online"
    bridge.process_mqtt_message("driftguard/unit/esp32-wokwi-01/heartbeat", json.dumps(hb))
    assert bridge.devices["esp32-wokwi-01"].liveness_state() == "ONLINE"

    lwt = json.dumps(
        {"device_id": "esp32-wokwi-01", "machine_id": "machine_01", "status": "offline"}
    )
    bridge.process_mqtt_message("driftguard/unit/esp32-wokwi-01/heartbeat", lwt)
    assert bridge.devices["esp32-wokwi-01"].liveness_state() == "OFFLINE"

    telemetry_msg, _ = VirtualWokwiEsp32(emu_cfg).next_reading()
    bridge.process_mqtt_message(
        "driftguard/unit/esp32-wokwi-01/telemetry", json.dumps(telemetry_msg)
    )
    bridge.flush()
    assert bridge.devices["esp32-wokwi-01"].liveness_state() == "ONLINE"
    assert any(t == kafka_settings.device_status_topic for t, _k, _v in prod.produced)


def test_dashboard_publisher_and_state_store_freshness() -> None:
    settings = load_settings({})
    pub_cfg = DashboardPublisherConfig(namespace="driftguard/pubtest")
    pub_state = DashboardPublisherState(settings, pub_cfg)

    raw_ev = TelemetryEvent(
        event_id=str(uuid.uuid4()),
        machine_id="machine_01",
        timestamp="2026-11-28T10:00:00.000Z",
        temperature_c=47.0,
        vibration_mm_s=2.2,
        pressure_bar=5.1,
        label=EventLabel.SPIKE,
        source_type=SourceType.VIRTUAL_IOT_WOKWI,
        device_id="esp32-wokwi-01",
        sequence_number=99,
    ).to_json_bytes()
    routed = pub_state.route_kafka_record(settings.topic, raw_ev)
    assert routed is not None
    assert routed[0] == "telemetry"
    assert "label" not in routed[1]

    # Route remaining Kafka topics cleanly
    assert pub_state.route_kafka_record(settings.alerts_topic, b'{"alert_id":"a1"}') is not None
    assert pub_state.route_kafka_record(settings.drift_topic, b'{"drift_id":"d1"}') is not None
    ms_bytes = json.dumps(
        {"hardware": {"selected_dl_device": "cpu"}, "memory_mib": {"process_rss_mib": 120.0}}
    ).encode()
    assert pub_state.route_kafka_record(settings.model_scores_topic, ms_bytes) is not None
    ds_bytes = json.dumps(
        {"device": {"device_id": "esp32-wokwi-01", "liveness_state": "ONLINE"}}
    ).encode()
    assert pub_state.route_kafka_record(settings.device_status_topic, ds_bytes) is not None
    assert pub_state.route_kafka_record(settings.topic, b"{bad_json") is None
    assert pub_state.route_kafka_record("unknown.topic", b'{"x":1}') is None
    health = pub_state.build_backend_health()
    assert health["active_devices"] == 1

    snap = pub_state.build_snapshot()
    store = DashboardStateStore()
    assert store.compute_freshness()["state"] == "OFFLINE"
    assert any("Disconnected" in b["title"] for b in store.diagnostic_banners())

    t0 = datetime(2026, 11, 28, 10, 0, 0, tzinfo=timezone.utc)
    store.set_connection_state(True, "connected")
    store.ingest_mqtt_message(
        "driftguard/pubtest/dashboard/snapshot", json.dumps(snap), received_utc=t0
    )
    assert store.compute_freshness(now_utc=t0 + timedelta(seconds=3))["state"] == "LIVE"
    assert store.compute_freshness(now_utc=t0 + timedelta(seconds=20))["state"] == "DELAYED"
    assert store.compute_freshness(now_utc=t0 + timedelta(seconds=60))["state"] == "OFFLINE"

    store.load_static_demo_sample()
    assert store.compute_freshness()["state"] == "DEMO"
    store.clear_for_live_mode()
    assert not store.demo_mode


def test_m10_1_helper_functions_and_cli_parsers(tmp_path: Path) -> None:
    from app.dashboard_publisher import parse_args as pub_parse_args
    from app.m10_1_online_runner import (
        _classification_metrics,
        _find_free_port,
        _is_truth_anomaly,
        _render_summary_markdown,
        _write_csv,
    )
    from app.m10_1_online_runner import parse_args as runner_parse_args
    from app.streaming_inference import parse_args as inf_parse_args
    from app.wokwi_emulator import VALID_PROFILES
    from app.wokwi_emulator import parse_args as emu_parse_args

    assert pub_parse_args(["--namespace", "n1"]).namespace == "n1"
    assert inf_parse_args(["--disable-tsfms"]).disable_tsfms is True
    assert emu_parse_args(["--profile", "abrupt_drift"]).profile == "abrupt_drift"
    assert runner_parse_args(["--duration-scale", "0.5"]).duration_scale == 0.5
    assert _find_free_port() > 0

    emu = VirtualWokwiEsp32(WokwiEmulatorConfig(command_token="tok"))  # noqa: S106
    for prof in VALID_PROFILES:
        emu.set_profile(prof)
        for step in range(1, 7):
            _p, reported = emu.next_reading()
            assert reported == prof
            _ = _is_truth_anomaly(prof, step)
    with pytest.raises(ValueError):
        emu.set_profile("nonexistent")

    cls_m = _classification_metrics([True, False, True, False], [True, False, False, True])
    assert cls_m["tp"] == 1 and cls_m["fp"] == 1 and cls_m["fn"] == 1 and cls_m["tn"] == 1
    csv_out = tmp_path / "metrics.csv"
    _write_csv(csv_out, [{"a": 1, "b": 2}], ["a", "b"])
    assert "a,b" in csv_out.read_text(encoding="utf-8")

    md = _render_summary_markdown(
        {
            "run_id": "unit_r1",
            "started_utc": "2026-11-28T00:00:00Z",
            "finished_utc": "2026-11-28T00:15:00Z",
            "planned_duration_s": 900.0,
            "duration_scale": 1.0,
            "wokwi_emulator": {
                "telemetry_published": 10,
                "heartbeats_published": 2,
                "commands_applied": 1,
                "reconnect_count": 0,
            },
            "bridge_status": {
                "messages_received": 10,
                "messages_forwarded": 10,
                "invalid_messages": 0,
                "sequence_gaps": 0,
                "reconnection_count": 0,
                "ingestion_latency_ms": {"mean_ms": 5.0, "p95_ms": 9.0},
            },
            "pipeline_counts": {
                "model_score_events_kafka": 10,
                "alert_events_kafka": 1,
                "drift_events_kafka": 0,
                "viewer_messages_ingested_ro": 12,
            },
            "hardware": {"selected_dl_device": "mps", "mps_available": True},
            "model_metrics": [
                {
                    "model_id": "lstm_autoencoder",
                    "family": "deep_learning",
                    "channel_mode": "multivariate",
                    "device": "mps",
                    "events_scored": 10,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "normal_phase_fpr": 0.0,
                    "mean_latency_ms": 1.2,
                }
            ],
        }
    )
    assert "DriftGuard-Kafka M10.1" in md
    cfg_no_tsfm = StreamingInferenceConfig(
        device="cpu",
        models_dir=tmp_path / "m_no",
        enable_tsfms=False,
        ae_window_size=6,
        tranad_window_size=6,
    )
    xt, xv = generate_normal_iot_splits(n_train=90, n_val=90)
    suite_no_tsfm = LiveModelSuite(cfg_no_tsfm, xt, xv)
    assert suite_no_tsfm.slots["chronos2_zero_shot"].status == "unavailable"


def test_loops_and_mqtt_runner_with_monkeypatched_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket
    import subprocess
    import threading

    import app.dashboard_publisher as dp_mod
    import app.m10_1_online_runner as runner_mod
    import app.streaming_inference as si_mod
    import app.wokwi_emulator as we_mod
    from dashboard.state_store import (
        DashboardMqttSettings,
        DashboardMqttSubscriber,
        load_reference_reports,
    )

    # 1. Mock Kafka Consumer + MQTT Client for inference & publisher loops
    ev_bytes = TelemetryEvent(
        event_id=str(uuid.uuid4()),
        machine_id="machine_01",
        timestamp="2026-11-28T10:00:00.000Z",
        temperature_c=46.5,
        vibration_mm_s=2.1,
        pressure_bar=5.0,
        label=EventLabel.NORMAL,
        source_type=SourceType.VIRTUAL_IOT_WOKWI,
        device_id="esp32-wokwi-01",
        sequence_number=1,
    ).to_json_bytes()

    class _FakeMsg:
        def __init__(self, topic: str, val: bytes) -> None:
            self._t = topic
            self._v = val

        def error(self) -> None:
            return None

        def topic(self) -> str:
            return self._t

        def value(self) -> bytes:
            return self._v

    class _FakeConsumer:
        def __init__(self, _conf: dict[str, Any]) -> None:
            self._items = [
                _FakeMsg("research.telemetry.raw", ev_bytes),
                None,
            ]

        def subscribe(self, _topics: list[str]) -> None:
            return None

        def poll(self, _t: float = 0.25) -> Any:
            return self._items.pop(0) if self._items else None

        def close(self) -> None:
            return None

    class _FakePubInfo:
        def wait_for_publish(self, timeout: float = 5.0) -> None:
            del timeout

    class _FakeMqttClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.on_connect: Any = None
            self.on_disconnect: Any = None
            self.on_message: Any = None
            self.published: list[tuple[str, Any, bool]] = []
            self.subscribed: list[str] = []

        def username_pw_set(self, _u: str, _p: str | None = None) -> None:
            return None

        def tls_set(self, **_kwargs: Any) -> None:
            return None

        def will_set(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def reconnect_delay_set(self, **_kwargs: Any) -> None:
            return None

        def connect(self, *_args: Any, **_kwargs: Any) -> None:
            if self.on_connect is not None:
                self.on_connect(self, None, None, 0, None)

        def connect_async(self, *_args: Any, **_kwargs: Any) -> None:
            self.connect()

        def loop_start(self) -> None:
            return None

        def loop_stop(self) -> None:
            return None

        def disconnect(self) -> None:
            if self.on_disconnect is not None:
                self.on_disconnect(self, None, None, 0, None)

        def subscribe(self, topic: str, qos: int = 1) -> None:
            del qos
            self.subscribed.append(topic)

        def publish(self, topic: str, payload: Any, qos: int = 1, retain: bool = False) -> Any:
            del qos
            self.published.append((topic, payload, retain))
            return _FakePubInfo()

    import paho.mqtt.client as paho_mqtt

    monkeypatch.setattr(si_mod, "Consumer", _FakeConsumer)
    monkeypatch.setattr(dp_mod, "Consumer", _FakeConsumer)
    monkeypatch.setattr(runner_mod, "Consumer", _FakeConsumer)
    monkeypatch.setattr(paho_mqtt, "Client", _FakeMqttClient)

    xt, xv = generate_normal_iot_splits(n_train=90, n_val=90)
    cfg = StreamingInferenceConfig(
        device="cpu",
        models_dir=tmp_path / "m_loop",
        enable_tsfms=False,
        ae_window_size=6,
        tranad_window_size=6,
    )
    suite = LiveModelSuite(cfg, xt, xv)
    settings = load_settings({})
    engine = StreamingInferenceEngine(suite, settings, kafka_producer=_FakeProducer())
    summary_path = tmp_path / "inf_summary.json"
    loop_sum = si_mod.run_streaming_inference_loop(
        engine, settings, max_events=1, duration_s=1.0, summary_output_path=summary_path
    )
    assert loop_sum["events_processed"] == 1
    assert summary_path.exists()

    pub_cfg = DashboardPublisherConfig(
        namespace="driftguard/unitpub",
        local_snapshot_path=tmp_path / "snap.json",
        snapshot_interval_s=0.1,
    )
    pub_health = dp_mod.run_dashboard_publisher(settings, pub_cfg, max_records=1, duration_s=1.0)
    assert pub_health["topic_counts"]["telemetry"] == 1
    assert (tmp_path / "snap.json").exists()

    # 2. Wokwi runner step + command handling + disconnect/reconnect path
    emu_cfg = WokwiEmulatorConfig(
        namespace="driftguard/unitwokwi",
        command_token="cmd-tok",  # noqa: S106
        publish_interval_s=0.01,
    )
    wokwi = we_mod.MqttWokwiRunner(emu_cfg)
    wokwi.connect(timeout_s=1.0)
    cmd_wire = build_profile_command_payload("intermittent_disconnect", "cmd-tok")

    class _FakeMqttMsg:
        def __init__(
            self, payload: bytes, topic: str = "driftguard/unitpub/dashboard/telemetry"
        ) -> None:
            self.payload = payload
            self.topic = topic

    wokwi._on_message(None, None, _FakeMqttMsg(cmd_wire))
    assert wokwi.commands_applied == 1
    for _ in range(5):
        wokwi.step_once()
    assert wokwi.device.reconnect_count == 1
    assert wokwi.telemetry_published == 4
    wokwi.stop()

    # 3. Online runner utility helpers (_wait_tcp_port, _ensure_kafka_topics, collector)
    attempts = {"count": 0}

    class _DummySockCtx:
        def __enter__(self) -> _DummySockCtx:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    def _flaky_create_connection(_addr: tuple[str, int], timeout: float = 1.0) -> _DummySockCtx:
        del timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("not ready yet")
        return _DummySockCtx()

    monkeypatch.setattr(socket, "create_connection", _flaky_create_connection)
    runner_mod._wait_tcp_port("127.0.0.1", 18830, timeout_s=1.0)
    assert attempts["count"] == 2

    def _always_fail_conn(_addr: tuple[str, int], timeout: float = 1.0) -> Any:
        del timeout
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", _always_fail_conn)
    with pytest.raises(TimeoutError):
        runner_mod._wait_tcp_port("127.0.0.1", runner_mod._find_free_port(), timeout_s=0.05)

    called_cmds: list[list[str]] = []

    def _fake_subproc_run(cmd: list[str], **_kwargs: Any) -> Any:
        called_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_subproc_run)
    runner_mod._ensure_kafka_topics(settings)
    assert len(called_cmds) == 5

    stop_ev = threading.Event()
    buckets: dict[str, list[dict[str, Any]]] = {"research.telemetry.raw": []}

    def _stop_soon() -> None:
        stop_ev.set()

    timer = threading.Timer(0.05, _stop_soon)
    timer.start()
    runner_mod._collect_kafka_topic_records(
        settings, ["research.telemetry.raw"], stop_ev, buckets, "u1"
    )
    assert len(buckets["research.telemetry.raw"]) == 1

    # 4. DashboardMqttSubscriber & load_reference_reports
    store = DashboardStateStore()
    sub = DashboardMqttSubscriber(DashboardMqttSettings(namespace="driftguard/unitpub"), store)
    sub._on_connect(_FakeMqttClient(), None, None, 0, None)
    assert store.mqtt_connected is True
    sub._on_message(
        None,
        None,
        _FakeMqttMsg(b'{"device_id":"esp32-wokwi-01","temperature_c":46.5}'),
    )
    assert load_reference_reports(tmp_path)["m10_1"] is None


def test_m10_1_online_protocol_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import socket
    import subprocess
    import time as time_mod

    import paho.mqtt.client as paho_mqtt

    import app.m10_1_online_runner as runner_mod
    import app.mqtt_bridge as bridge_mod
    import app.streaming_inference as si_mod

    class _DummySockCtx:
        def __enter__(self) -> _DummySockCtx:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    class _FakePubInfo:
        def wait_for_publish(self, timeout: float = 5.0) -> None:
            del timeout

    class _FakeMqttClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.on_connect: Any = None
            self.on_disconnect: Any = None
            self.on_message: Any = None

        def username_pw_set(self, _u: str, _p: str | None = None) -> None:
            return None

        def tls_set(self, **_kwargs: Any) -> None:
            return None

        def will_set(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def reconnect_delay_set(self, **_kwargs: Any) -> None:
            return None

        def connect(self, *_args: Any, **_kwargs: Any) -> None:
            if self.on_connect is not None:
                self.on_connect(self, None, None, 0, None)

        def connect_async(self, *_args: Any, **_kwargs: Any) -> None:
            self.connect()

        def loop_start(self) -> None:
            return None

        def loop_stop(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

        def subscribe(self, _topic: str, qos: int = 1) -> None:
            del qos

        def publish(self, _topic: str, _payload: Any, qos: int = 1, retain: bool = False) -> Any:
            del qos, retain
            return _FakePubInfo()

    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: _DummySockCtx())
    monkeypatch.setattr(subprocess, "run", lambda cmd, **_k: subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(paho_mqtt, "Client", _FakeMqttClient)
    monkeypatch.setattr(runner_mod, "Producer", lambda _c: _FakeProducer())
    monkeypatch.setattr(si_mod, "Producer", lambda _c: _FakeProducer())
    monkeypatch.setattr(bridge_mod, "Producer", lambda _c: _FakeProducer())
    monkeypatch.setattr(runner_mod, "_collect_kafka_topic_records", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runner_mod,
        "run_bridge_loop",
        lambda bridge, **_k: bridge.build_status_report(run_id="mocked"),
    )
    monkeypatch.setattr(runner_mod, "process_messages", lambda **_k: "stopped")
    monkeypatch.setattr(
        runner_mod,
        "run_streaming_inference_loop",
        lambda engine, *_a, **_k: engine.build_runtime_summary(),
    )
    monkeypatch.setattr(
        runner_mod, "run_dashboard_publisher", lambda *_a, **_k: {"status": "online"}
    )
    monkeypatch.setattr(runner_mod, "Consumer", lambda _c: None)
    monkeypatch.setattr(runner_mod, "publish_profile_command", lambda **_k: "cmd/topic")
    # Speed up phase sleeps inside test
    monkeypatch.setattr(time_mod, "sleep", lambda _s: None)
    monkeypatch.setattr(
        si_mod,
        "generate_normal_iot_splits",
        lambda **_k: generate_normal_iot_splits(n_train=90, n_val=90),
    )

    out_dir = runner_mod.run_m10_1_online_protocol(
        duration_scale=0.001,
        publish_interval_s=0.01,
        device="cpu",
        enable_tsfms=False,
        inject_invalid_count=1,
        output_root=tmp_path,
        run_id="mock_run",
    )
    for fname in (
        "summary_report.json",
        "summary_report.md",
        "events.jsonl",
        "scenario_truth.jsonl",
        "model_metrics.csv",
        "drift_events.csv",
        "alerts.csv",
        "device_status.json",
        "hardware_memory_report.json",
    ):
        assert (out_dir / fname).exists()
