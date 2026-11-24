"""Unit tests for M2 scenarios, threshold detector, and evaluation reports."""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from app.detector import (
    DetectorStats,
    SensorReading,
    ThresholdConfig,
    ThresholdDetector,
    build_alert_payload,
    parse_args,
    process_messages,
    verify_alerts_on_topic,
)
from app.m2_report import (
    compute_classification_metrics,
)
from app.m2_report import (
    main as m2_report_main,
)
from app.producer import DeliveryTracker
from app.run_summary import write_summary
from app.schemas import EventLabel
from app.simulator import Scenario, TelemetrySimulator


class TestScenarios:
    def test_wire_events_never_expose_ground_truth_labels(self) -> None:
        for scenario in (Scenario.NORMAL, Scenario.ANOMALIES, Scenario.DRIFT):
            sim = TelemetrySimulator(machine_count=3, seed=42, scenario=scenario, anomaly_rate=0.25)
            pairs = list(sim.generate_scenario_events(200))
            assert len(pairs) == 200
            for ev, _ in pairs:
                assert ev.label is EventLabel.NORMAL
                wire_payload = json.loads(ev.to_json_bytes())
                assert wire_payload["label"] == "normal"
                assert "is_anomaly" not in wire_payload
                assert "ground_truth_label" not in wire_payload

    def test_gradual_drift_retains_normal_ground_truth_labels(self) -> None:
        sim = TelemetrySimulator(machine_count=5, seed=42, scenario=Scenario.DRIFT)
        pairs = list(sim.generate_scenario_events(300))
        # Drift shifts sensor values upward in the second half of the stream,
        # yet every event represents normal operating-pattern drift (not a fault).
        assert pairs[-1][0].temperature_c > pairs[0][0].temperature_c
        assert any(ev.temperature_c > 70.0 for ev, _ in pairs)
        assert all(gt.ground_truth_label is EventLabel.NORMAL for _, gt in pairs)
        assert all(gt.is_anomaly is False for _, gt in pairs)
        assert all(gt.scenario == "drift" for _, gt in pairs)

    def test_drift_is_not_an_anomaly_event_label(self) -> None:
        from app.schemas import SchemaValidationError, TelemetryEvent

        ev_dict = TelemetrySimulator(1, seed=1).generate_event().to_dict()
        ev_dict["label"] = "drift"
        with pytest.raises(SchemaValidationError, match="label"):
            TelemetryEvent.from_dict(ev_dict)

    def test_spikes_and_stuck_sensors_are_independently_labelled(self) -> None:
        sim = TelemetrySimulator(
            machine_count=5, seed=42, scenario=Scenario.ANOMALIES, anomaly_rate=0.12
        )
        pairs = list(sim.generate_scenario_events(300))
        labels = {gt.ground_truth_label for _, gt in pairs}
        assert labels == {EventLabel.NORMAL, EventLabel.SPIKE, EventLabel.STUCK}
        for _, gt in pairs:
            if gt.ground_truth_label is EventLabel.NORMAL:
                assert gt.is_anomaly is False
            else:
                assert gt.is_anomaly is True

    def test_scenario_generation_is_reproducible_with_seed(self) -> None:
        s1 = TelemetrySimulator(3, seed=99, scenario=Scenario.ANOMALIES, anomaly_rate=0.2)
        s2 = TelemetrySimulator(3, seed=99, scenario=Scenario.ANOMALIES, anomaly_rate=0.2)
        seq1 = [
            (e.temperature_c, gt.ground_truth_label) for e, gt in s1.generate_scenario_events(40)
        ]
        seq2 = [
            (e.temperature_c, gt.ground_truth_label) for e, gt in s2.generate_scenario_events(40)
        ]
        assert seq1 == seq2

    def test_invalid_anomaly_rate_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="anomaly_rate"):
            TelemetrySimulator(1, anomaly_rate=1.5)


class TestThresholdDetector:
    @staticmethod
    def _reading(
        machine_id: str = "machine_01",
        temp: float = 50.0,
        vib: float = 2.5,
        press: float = 5.0,
    ) -> SensorReading:
        return SensorReading(
            event_id="3f2b8c1e-9a4d-4e7b-8c2a-1d5e6f7a8b9c",
            machine_id=machine_id,
            timestamp="2026-11-24T10:30:00.000Z",
            temperature_c=temp,
            vibration_mm_s=vib,
            pressure_bar=press,
        )

    def test_sensor_reading_strips_label(self) -> None:
        import app.detector as detector_mod

        ev = TelemetrySimulator(1, seed=1).generate_event()
        reading = SensorReading.from_telemetry(ev)
        assert not hasattr(reading, "label")
        assert "GroundTruthRecord" not in dir(detector_mod)
        assert "EventLabel" not in dir(detector_mod)

    def test_normal_reading_passes(self) -> None:
        detector = ThresholdDetector()
        res = detector.evaluate(self._reading())
        assert not res.is_anomaly
        assert res.reason == "normal"

    @pytest.mark.parametrize(
        ("kwargs", "expected_rule"),
        [
            ({"temp": 80.0}, "temperature_c>70.0"),
            ({"temp": 20.0}, "temperature_c<30.0"),
            ({"vib": 6.5}, "vibration_mm_s>5.0"),
            ({"vib": 0.1}, "vibration_mm_s<0.5"),
            ({"press": 9.0}, "pressure_bar>7.5"),
            ({"press": 1.0}, "pressure_bar<2.5"),
        ],
    )
    def test_threshold_breaches_are_detected(
        self, kwargs: dict[str, float], expected_rule: str
    ) -> None:
        res = ThresholdDetector().evaluate(self._reading("machine_01", **kwargs))
        assert res.is_anomaly
        assert res.reason == "threshold_breach"
        assert expected_rule in res.triggered_rules

    def test_stuck_sensor_is_tracked_per_machine(self) -> None:
        detector = ThresholdDetector(ThresholdConfig(stuck_window=3))
        assert not detector.evaluate(self._reading(machine_id="machine_01")).is_anomaly
        assert not detector.evaluate(self._reading(machine_id="machine_02")).is_anomaly
        assert not detector.evaluate(self._reading(machine_id="machine_01")).is_anomaly
        stuck = detector.evaluate(self._reading(machine_id="machine_01"))
        assert stuck.is_anomaly
        assert stuck.reason == "stuck_sensor"

    def test_combined_threshold_and_stuck(self) -> None:
        detector = ThresholdDetector(ThresholdConfig(stuck_window=2))
        detector.evaluate(self._reading(temp=95.0))
        both = detector.evaluate(self._reading(temp=95.0))
        assert both.reason == "threshold_and_stuck"


@dataclass
class _FakeMsg:
    payload: bytes | None
    err: Any = None

    def value(self) -> bytes | None:
        return self.payload

    def error(self) -> Any:
        return self.err

    def partition(self) -> int:
        return 0


@dataclass
class _FakeConsumer:
    msgs: list[_FakeMsg | None]
    stored: list[_FakeMsg] = field(default_factory=list)

    def poll(self, _t: float) -> _FakeMsg | None:
        return self.msgs.pop(0) if self.msgs else None

    def store_offsets(self, message: _FakeMsg) -> None:
        self.stored.append(message)


@dataclass
class _FakeProducer:
    sent: list[dict[str, Any]] = field(default_factory=list)

    def produce(self, topic: str, key: bytes, value: bytes, on_delivery: Any) -> None:
        self.sent.append({"topic": topic, "key": key, "value": value})
        on_delivery(None, _FakeMsg(value))

    def poll(self, _t: float) -> int:
        return 0


class TestDetectorPipeline:
    def test_process_messages_publishes_alerts_and_handles_malformed(self) -> None:
        sim = TelemetrySimulator(1, seed=1)
        normal_ev = sim.generate_event()
        spike_dict = normal_ev.to_dict()
        spike_dict["temperature_c"] = 120.0
        spike_bytes = json.dumps(spike_dict).encode("utf-8")

        consumer = _FakeConsumer(
            [
                _FakeMsg(normal_ev.to_json_bytes()),
                _FakeMsg(spike_bytes),
                _FakeMsg(b"{bad"),
            ]
        )
        producer = _FakeProducer()
        stats = DetectorStats()
        tracker = DeliveryTracker()
        detections: list[dict[str, Any]] = []

        reason = process_messages(
            consumer,  # type: ignore[arg-type]
            producer,  # type: ignore[arg-type]
            "research.telemetry.alerts",
            ThresholdDetector(),
            stats,
            tracker,
            threading.Event(),
            limit=3,
            idle_timeout_s=None,
            on_detection=detections.append,
        )
        assert reason == "limit_reached"
        assert stats.received == 3
        assert stats.malformed == 1
        assert stats.alerts_triggered == 1
        assert tracker.delivered == 1
        assert len(producer.sent) == 1
        assert len(detections) == 2

    def test_alert_payload_and_cli_args(self) -> None:
        reading = TestThresholdDetector._reading(temp=99.0)
        res = ThresholdDetector().evaluate(reading)
        payload = build_alert_payload(reading, res)
        assert payload["reason"] == "threshold_breach"
        assert "alert_id" in payload
        args = parse_args(["--limit", "10", "--group", "g"])
        assert (args.limit, args.group) == (10, "g")

    def test_process_messages_idle_timeout_and_kafka_error(self) -> None:
        class _Err:
            def fatal(self) -> bool:
                return False

            def str(self) -> str:
                return "transient"

        consumer = _FakeConsumer([_FakeMsg(None, err=_Err()), None])
        stats = DetectorStats()
        reason = process_messages(
            consumer,  # type: ignore[arg-type]
            _FakeProducer(),  # type: ignore[arg-type]
            "alerts",
            ThresholdDetector(),
            stats,
            DeliveryTracker(),
            threading.Event(),
            limit=None,
            idle_timeout_s=0.001,
        )
        assert reason == "idle_timeout"
        assert stats.kafka_errors == 1

    def test_process_messages_raises_on_fatal_kafka_error(self) -> None:
        from confluent_kafka import KafkaException

        class _FatalErr:
            def fatal(self) -> bool:
                return True

            def str(self) -> str:
                return "fatal"

        with pytest.raises(KafkaException):
            process_messages(
                _FakeConsumer([_FakeMsg(None, err=_FatalErr())]),  # type: ignore[arg-type]
                _FakeProducer(),  # type: ignore[arg-type]
                "alerts",
                ThresholdDetector(),
                DetectorStats(),
                DeliveryTracker(),
                threading.Event(),
                limit=None,
                idle_timeout_s=None,
            )

    def test_verify_alerts_on_topic_with_mocked_consumer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from app.config import load_settings

        settings = load_settings({})
        assert verify_alerts_on_topic(settings, 0) == {"received": 0, "valid": 0, "malformed": 0}

        valid_alert = json.dumps({"alert_id": "a1", "event_id": "e1"}).encode("utf-8")

        class _MockConsumer:
            def __init__(self, _cfg: Any) -> None:
                self.msgs = [_FakeMsg(valid_alert), _FakeMsg(b"{bad")]

            def subscribe(self, _topics: list[str]) -> None:
                return None

            def poll(self, _t: float) -> _FakeMsg | None:
                return self.msgs.pop(0) if self.msgs else None

            def store_offsets(self, message: Any) -> None:
                return None

            def close(self) -> None:
                return None

        monkeypatch.setattr("app.detector.Consumer", _MockConsumer)
        summary = verify_alerts_on_topic(settings, 2, timeout_s=2.0)
        assert summary == {"received": 2, "valid": 1, "malformed": 1}

    def test_publish_events_with_ground_truth_callback(self) -> None:
        from app.producer import publish_events

        truths: list[Any] = []
        sim = TelemetrySimulator(2, seed=5, scenario=Scenario.ANOMALIES, anomaly_rate=0.3)
        attempted, interrupted = publish_events(
            _FakeProducer(),  # type: ignore[arg-type]
            "raw",
            sim,
            10,
            0,
            DeliveryTracker(),
            on_ground_truth=truths.append,
        )
        assert (attempted, interrupted) == (10, False)
        assert len(truths) == 10

        from app.producer import parse_args as parse_prod_args

        prod_args = parse_prod_args(["--scenario", "drift", "--anomaly-rate", "0.2"])
        assert prod_args.scenario is Scenario.DRIFT
        assert prod_args.anomaly_rate == 0.2


class TestM2Report:
    def test_classification_metrics_handle_zero_denominators_and_drift_fps(self) -> None:
        # Pure normal scenario: 0 alerts, 0 anomalies
        m_normal = compute_classification_metrics(0, 0, 300, 0)
        assert m_normal == {
            "tp": 0,
            "fp": 0,
            "tn": 300,
            "fn": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "false_positive_rate": 0.0,
        }

        # Pure drift scenario: 0 true anomalies, 100 false-positive alerts on 300 normal events
        # 2*TP + FP + FN = 100 > 0 -> F1 is 0.0 (unavailable only when 2*TP + FP + FN == 0)
        m_drift = compute_classification_metrics(0, 100, 200, 0)
        assert m_drift == {
            "tp": 0,
            "fp": 100,
            "tn": 200,
            "fn": 0,
            "precision": 0.0,
            "recall": None,
            "f1": 0.0,
            "false_positive_rate": 0.3333,
        }

        # Detector predicted 0 anomalies when 10 anomalies existed -> 2*TP + FP + FN = 10 > 0
        m_all_missed = compute_classification_metrics(0, 0, 90, 10)
        assert m_all_missed == {
            "tp": 0,
            "fp": 0,
            "tn": 90,
            "fn": 10,
            "precision": None,
            "recall": 0.0,
            "f1": 0.0,
            "false_positive_rate": 0.0,
        }

        # Detector fired false alarms AND missed all true anomalies -> F1 must be 0.0, not None
        m_zero_tp = compute_classification_metrics(0, 5, 85, 10)
        assert m_zero_tp == {
            "tp": 0,
            "fp": 5,
            "tn": 85,
            "fn": 10,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_positive_rate": 0.0556,
        }

        m = compute_classification_metrics(8, 2, 88, 2)
        assert m["precision"] == 0.8
        assert m["recall"] == 0.8
        assert m["f1"] == 0.8
        assert m["false_positive_rate"] == 0.0222

        with pytest.raises(ValueError, match="non-negative"):
            compute_classification_metrics(-1, 0, 10, 0)

    def test_evaluate_scenario_rejects_inconsistent_ground_truth(self) -> None:
        from app.m2_report import evaluate_scenario
        from app.run_summary import SummaryError

        with pytest.raises(SummaryError, match="inconsistent ground truth"):
            evaluate_scenario(
                scenario="drift",
                producer_summary={"delivered": 1},
                detector_summary={
                    "received": 1,
                    "malformed": 0,
                    "alerts_triggered": 0,
                    "alerts_delivered": 0,
                    "alerts_failed": 0,
                },
                producer_records=[{"event_id": "e1"}],
                ground_truth_records=[
                    {
                        "event_id": "e1",
                        "machine_id": "machine_01",
                        "ground_truth_label": "normal",
                        "is_anomaly": True,
                    }
                ],
                detection_records=[
                    {"event_id": "e1", "predicted_anomaly": False, "reason": "normal"}
                ],
            )

    def test_evaluate_scenario_and_write_bundle(self, tmp_path: Path) -> None:
        sc_dir = tmp_path / "run" / "anomalies"
        sc_dir.mkdir(parents=True)
        write_summary(sc_dir / "producer_summary.json", {"delivered": 2, "duration_s": 0.5})
        write_summary(
            sc_dir / "detector_summary.json",
            {
                "received": 2,
                "malformed": 0,
                "alerts_triggered": 1,
                "alerts_delivered": 1,
                "alerts_failed": 0,
                "duration_s": 1.0,
            },
        )
        write_summary(sc_dir / "alerts_verified.json", {"received": 1})

        prod_rec = [
            {"event_id": "e1", "enqueued_monotonic_ns": 1_000_000},
            {"event_id": "e2", "enqueued_monotonic_ns": 2_000_000},
        ]
        gt_rec = [
            {
                "event_id": "e1",
                "machine_id": "machine_01",
                "scenario": "anomalies",
                "ground_truth_label": "normal",
                "is_anomaly": False,
            },
            {
                "event_id": "e2",
                "machine_id": "machine_01",
                "scenario": "anomalies",
                "ground_truth_label": "spike",
                "is_anomaly": True,
            },
        ]
        det_rec = [
            {
                "event_id": "e1",
                "predicted_anomaly": False,
                "reason": "normal",
                "detected_monotonic_ns": 3_000_000,
            },
            {
                "event_id": "e2",
                "predicted_anomaly": True,
                "reason": "threshold_breach",
                "detected_monotonic_ns": 6_000_000,
            },
        ]
        for fname, rows in (
            ("producer_records.jsonl", prod_rec),
            ("ground_truth.jsonl", gt_rec),
            ("detections.jsonl", det_rec),
        ):
            (sc_dir / fname).write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )

        out_dir = tmp_path / "reports"
        rc = m2_report_main(
            [
                "--run-id",
                "m2-test",
                "--seed",
                "42",
                "--run-dir",
                str(tmp_path / "run"),
                "--scenarios",
                "anomalies",
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        parsed = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        assert parsed["overall_metrics"]["tp"] == 1
        assert parsed["overall_metrics"]["tn"] == 1
        assert parsed["overall_metrics"]["f1"] == 1.0
        assert (out_dir / "report.md").exists()
        with (out_dir / "evaluations.csv").open(encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == 2
