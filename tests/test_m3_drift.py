"""Unit tests for M3 ADWIN concept drift monitor, reports, and dashboard."""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from app.config import load_settings
from app.detector import SensorReading, ThresholdDetector
from app.drift_monitor import (
    MONITORED_FEATURES,
    AdwinConfig,
    DriftMonitorStats,
    MachineAdwinMonitor,
    parse_args,
    process_drift_stream,
    verify_drift_events_on_topic,
)
from app.drift_monitor import (
    main as drift_monitor_main,
)
from app.m3_report import (
    evaluate_m3_scenario,
)
from app.m3_report import (
    main as m3_report_main,
)
from app.producer import DeliveryTracker
from app.run_summary import SummaryError, write_summary
from app.schemas import EventLabel
from app.simulator import Scenario, TelemetrySimulator


class TestAdwinConfigAndIsolation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"delta": 0.0},
            {"delta": 1.0},
            {"clock": 0},
            {"min_window_length": 0},
            {"grace_period": 0},
        ],
    )
    def test_invalid_adwin_config_is_rejected(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            AdwinConfig(**kwargs)

    def test_drift_monitor_never_imports_ground_truth(self) -> None:
        import app.drift_monitor as dm_mod

        assert "GroundTruthRecord" not in dir(dm_mod)
        assert "EventLabel" not in dir(dm_mod)

    def test_abrupt_drift_retains_normal_ground_truth_labels(self) -> None:
        sim = TelemetrySimulator(3, seed=42, scenario=Scenario.ABRUPT_DRIFT)
        pairs = list(sim.generate_scenario_events(200))
        assert len(pairs) == 200
        assert all(ev.label is EventLabel.NORMAL for ev, _ in pairs)
        assert all(gt.ground_truth_label is EventLabel.NORMAL for _, gt in pairs)
        assert all(not gt.is_anomaly for _, gt in pairs)
        assert not pairs[99][1].drift_active
        assert pairs[100][1].drift_active
        assert pairs[-1][0].temperature_c > pairs[0][0].temperature_c


class TestMachineAdwinMonitor:
    def test_monitors_are_scoped_per_machine_and_feature(self) -> None:
        mon = MachineAdwinMonitor()
        # Drift only temperature_c on machine_01; keep machine_02 completely stable
        for idx in range(120):
            temp_m1 = 48.0 if idx < 50 else 75.0
            r1 = SensorReading(
                event_id=f"00000000-0000-4000-8000-{idx:012d}",
                machine_id="machine_01",
                timestamp="2026-11-24T10:00:00.000Z",
                temperature_c=temp_m1,
                vibration_mm_s=2.5,
                pressure_bar=5.0,
            )
            r2 = SensorReading(
                event_id=f"00000000-0000-4000-8001-{idx:012d}",
                machine_id="machine_02",
                timestamp="2026-11-24T10:00:00.000Z",
                temperature_c=48.0,
                vibration_mm_s=2.5,
                pressure_bar=5.0,
            )
            _, m1_pos, d1 = mon.update(r1)
            _, m2_pos, d2 = mon.update(r2)
            assert m1_pos == idx
            assert m2_pos == idx
            assert d2 == []
            for d in d1:
                assert d.machine_id == "machine_01"
                assert d.feature == "temperature_c"
                assert d.machine_position >= 50

        assert len(mon.monitor_keys) == 2 * len(MONITORED_FEATURES)

    def test_stable_normal_stream_produces_zero_false_alarms_and_drift_detects_all(self) -> None:
        normal_sim = TelemetrySimulator(5, seed=42, scenario=Scenario.NORMAL)
        normal_mon = MachineAdwinMonitor()
        normal_drifts = [
            d
            for ev, _ in normal_sim.generate_scenario_events(500)
            for d in normal_mon.update(SensorReading.from_telemetry(ev))[2]
        ]
        assert normal_drifts == []

        for sc in (Scenario.DRIFT, Scenario.ABRUPT_DRIFT):
            sim = TelemetrySimulator(5, seed=42, scenario=sc)
            mon = MachineAdwinMonitor()
            detected_channels: set[tuple[str, str]] = set()
            for ev, gt in sim.generate_scenario_events(500):
                _, _, drifts = mon.update(SensorReading.from_telemetry(ev))
                for d in drifts:
                    assert gt.drift_active
                    detected_channels.add((d.machine_id, d.feature))
            assert len(detected_channels) == 15


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


class TestDriftPipelineAndReport:
    def test_process_drift_stream_and_m3_report_bundle(self, tmp_path: Path) -> None:
        sim = TelemetrySimulator(2, seed=42, scenario=Scenario.ABRUPT_DRIFT)
        pairs = list(sim.generate_scenario_events(160))
        msgs: list[_FakeMsg | None] = [_FakeMsg(ev.to_json_bytes()) for ev, _ in pairs]
        msgs.append(_FakeMsg(b"{malformed"))

        consumer = _FakeConsumer(msgs)
        producer = _FakeProducer()
        monitor = MachineAdwinMonitor()
        stats = DriftMonitorStats()
        tracker = DeliveryTracker()
        drift_records: list[dict[str, Any]] = []
        obs_records: list[dict[str, Any]] = []

        stop_reason = process_drift_stream(
            consumer,  # type: ignore[arg-type]
            producer,  # type: ignore[arg-type]
            "research.telemetry.drift",
            monitor,
            ThresholdDetector(),
            stats,
            tracker,
            threading.Event(),
            limit=161,
            idle_timeout_s=None,
            on_drift=drift_records.append,
            on_observation=obs_records.append,
        )
        assert stop_reason == "limit_reached"
        assert stats.received == 161
        assert stats.malformed == 1
        assert stats.drift_events_triggered > 0
        assert tracker.delivered == stats.drift_events_triggered

        sc_dir = tmp_path / "runs" / "abrupt_drift"
        sc_dir.mkdir(parents=True)
        write_summary(sc_dir / "producer_summary.json", {"delivered": 160, "duration_s": 0.4})
        write_summary(
            sc_dir / "monitor_summary.json",
            stats.to_dict(tracker, 0.5, stop_reason, monitor.config.delta),
        )
        write_summary(sc_dir / "drift_verified.json", {"received": tracker.delivered})

        prod_recs = [
            {"event_id": ev.event_id, "enqueued_monotonic_ns": 1_000_000} for ev, _ in pairs
        ]
        gt_recs = [gt.to_dict() for _, gt in pairs]
        for fname, rows in (
            ("producer_records.jsonl", prod_recs),
            ("ground_truth.jsonl", gt_recs),
            ("drift_events.jsonl", drift_records),
            ("observations.jsonl", obs_records),
        ):
            (sc_dir / fname).write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )

        out_dir = tmp_path / "reports" / "m3"
        rc = m3_report_main(
            [
                "--run-id",
                "m3-unit",
                "--seed",
                "42",
                "--delta",
                "0.002",
                "--run-dir",
                str(tmp_path / "runs"),
                "--scenarios",
                "abrupt_drift",
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        for name in ("report.json", "report.md", "channel_evaluations.csv", "dashboard.html"):
            assert (out_dir / name).exists() and (out_dir / name).stat().st_size > 0

        report_data = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        assert report_data["overall_metrics"]["expected_drift_channels"] == 6
        assert report_data["overall_metrics"]["detected_drift_channels"] == 6
        assert report_data["overall_metrics"]["missed_drift_channels"] == 0
        assert report_data["overall_metrics"]["false_alarms_stable"] == 0
        with (out_dir / "channel_evaluations.csv").open(encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == 6

        dashboard_html = (out_dir / "dashboard.html").read_text(encoding="utf-8")
        assert "m3-unit" in dashboard_html
        assert "streamCanvas" in dashboard_html

    def test_evaluate_m3_rejects_anomaly_labelled_drift(self) -> None:
        with pytest.raises(SummaryError, match="normal ground-truth label"):
            evaluate_m3_scenario(
                scenario="drift",
                producer_summary={"delivered": 1},
                monitor_summary={
                    "received": 1,
                    "malformed": 0,
                    "static_alerts_observed": 0,
                    "drift_events_triggered": 0,
                    "drift_events_delivered": 0,
                    "drift_events_failed": 0,
                },
                producer_records=[{"event_id": "e1"}],
                ground_truth_records=[
                    {
                        "event_id": "e1",
                        "machine_id": "machine_01",
                        "ground_truth_label": "spike",
                        "is_anomaly": True,
                        "drift_active": True,
                    }
                ],
                drift_records=[],
                observation_records=[],
            )

    def test_verify_drift_events_and_cli_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = load_settings({})
        assert verify_drift_events_on_topic(settings, 0) == {
            "received": 0,
            "valid": 0,
            "malformed": 0,
        }
        good = json.dumps(
            {
                "drift_id": "d1",
                "event_id": "e1",
                "machine_id": "machine_01",
                "feature": "temperature_c",
                "stream_position": 10,
            }
        ).encode("utf-8")

        class _MockCons:
            def __init__(self, _cfg: Any) -> None:
                self.msgs = [_FakeMsg(good), _FakeMsg(b"{bad")]

            def subscribe(self, _t: list[str]) -> None:
                return None

            def poll(self, _t: float) -> _FakeMsg | None:
                return self.msgs.pop(0) if self.msgs else None

            def store_offsets(self, message: Any) -> None:
                return None

            def close(self) -> None:
                return None

        monkeypatch.setattr("app.drift_monitor.Consumer", _MockCons)
        assert verify_drift_events_on_topic(settings, 2, timeout_s=2.0) == {
            "received": 2,
            "valid": 1,
            "malformed": 1,
        }
        args = parse_args(["--limit", "100", "--delta", "0.005", "--clock", "2"])
        assert (args.limit, args.delta, args.clock) == (100, 0.005, 2)

    def test_drift_monitor_main_and_report_edge_cases(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sim = TelemetrySimulator(1, seed=1, scenario=Scenario.NORMAL)
        pairs = list(sim.generate_scenario_events(10))

        class _MainCons:
            def __init__(self, _cfg: Any) -> None:
                self.msgs = [_FakeMsg(ev.to_json_bytes()) for ev, _ in pairs]

            def subscribe(self, _t: list[str], on_assign: Any = None) -> None:
                return None

            def poll(self, _t: float) -> _FakeMsg | None:
                return self.msgs.pop(0) if self.msgs else None

            def store_offsets(self, message: Any) -> None:
                return None

            def close(self) -> None:
                return None

        class _MainProd:
            def __init__(self, _cfg: Any) -> None:
                return None

            def produce(self, **kwargs: Any) -> None:
                kwargs["on_delivery"](None, _FakeMsg(b""))

            def poll(self, _t: float) -> int:
                return 0

            def flush(self, _t: float) -> int:
                return 0

        monkeypatch.setattr("app.drift_monitor.Consumer", _MainCons)
        monkeypatch.setattr("app.drift_monitor.Producer", _MainProd)

        sum_file = tmp_path / "mon_sum.json"
        drift_file = tmp_path / "drifts.jsonl"
        obs_file = tmp_path / "obs.jsonl"
        rc = drift_monitor_main(
            [
                "--limit",
                "10",
                "--group",
                "g-m3",
                "--summary-file",
                str(sum_file),
                "--drift-file",
                str(drift_file),
                "--observations-file",
                str(obs_file),
            ]
        )
        assert rc == 0
        assert sum_file.exists()

        ver_sum = tmp_path / "ver_sum.json"
        assert drift_monitor_main(["--verify-drift", "0", "--summary-file", str(ver_sum)]) == 0

        # Evaluate normal scenario (0 expected channels, 1 simulated false alarm) + missed channel
        gt_normal = [gt.to_dict() for _, gt in pairs]
        obs_recs = [json.loads(line) for line in obs_file.read_text().splitlines() if line]
        fake_false_alarm = [
            {
                "event_id": pairs[0][0].event_id,
                "machine_id": pairs[0][0].machine_id,
                "feature": "temperature_c",
                "machine_position": 0,
                "stream_position": 0,
            }
        ]
        sc_rep, ch_rows, _ = evaluate_m3_scenario(
            scenario="normal",
            producer_summary={"delivered": 10, "duration_s": 0.1},
            monitor_summary=json.loads(sum_file.read_text()),
            producer_records=[],
            ground_truth_records=gt_normal,
            drift_records=fake_false_alarm,
            observation_records=obs_recs,
            drift_verified=0,
        )
        assert sc_rep["drift_metrics"]["expected_drift_channels"] == 0
        assert sc_rep["drift_metrics"]["false_alarms_stable"] == 1
        assert any(r["status"] == "stable_false_alarm" for r in ch_rows)

        # Missed drift channel (drift_active=True, no ADWIN detections)
        gt_missed = [dict(r, drift_active=True) for r in gt_normal]
        sc_missed, ch_missed, _ = evaluate_m3_scenario(
            scenario="drift",
            producer_summary={"delivered": 10, "duration_s": 0.1},
            monitor_summary=json.loads(sum_file.read_text()),
            producer_records=[],
            ground_truth_records=gt_missed,
            drift_records=[],
            observation_records=obs_recs,
            drift_verified=0,
        )
        assert sc_missed["drift_metrics"]["missed_drift_channels"] == 3
        assert all(r["status"] == "missed" for r in ch_missed)

    def test_episode_vs_repeated_alerts_and_independent_fa_missed(self) -> None:
        # 4 observations on machine_01: pos 0,1 stable; pos 2,3 drifted
        gt = [
            {
                "event_id": f"e{i}",
                "machine_id": "machine_01",
                "ground_truth_label": "normal",
                "is_anomaly": False,
                "stream_index": i,
                "drift_active": i >= 2,
            }
            for i in range(4)
        ]
        obs = [
            {
                "event_id": f"e{i}",
                "machine_id": "machine_01",
                "stream_position": i,
                "machine_position": i,
                "temperature_c": 50.0 + 10.0 * i,
                "vibration_mm_s": 2.0,
                "pressure_bar": 5.0,
                "static_alert": False,
                "drift_features": [],
            }
            for i in range(4)
        ]
        # temperature_c: 2 post-onset alerts (pos 2 first, pos 3 repeated)
        # vibration_mm_s: 1 pre-onset false alarm (pos 1) AND 0 post-onset alerts (missed!)
        # pressure_bar: 0 alerts (missed!)
        drifts = [
            {
                "event_id": "e2",
                "machine_id": "machine_01",
                "feature": "temperature_c",
                "machine_position": 2,
                "stream_position": 2,
            },
            {
                "event_id": "e3",
                "machine_id": "machine_01",
                "feature": "temperature_c",
                "machine_position": 3,
                "stream_position": 3,
            },
            {
                "event_id": "e1",
                "machine_id": "machine_01",
                "feature": "vibration_mm_s",
                "machine_position": 1,
                "stream_position": 1,
            },
        ]
        rep, rows, _ = evaluate_m3_scenario(
            scenario="drift",
            producer_summary={"delivered": 4, "duration_s": 0.1},
            monitor_summary={
                "received": 4,
                "malformed": 0,
                "static_alerts_observed": 0,
                "drift_events_triggered": 3,
                "drift_events_delivered": 3,
                "drift_events_failed": 0,
            },
            producer_records=[],
            ground_truth_records=gt,
            drift_records=drifts,
            observation_records=obs,
            drift_verified=3,
        )
        dm = rep["drift_metrics"]
        assert dm["expected_drift_channels"] == 3
        assert dm["first_episode_detections"] == 1
        assert dm["repeated_episode_alerts"] == 1
        assert dm["false_alarms_stable"] == 1
        assert dm["total_adwin_alerts"] == 3
        assert dm["missed_drift_channels"] == 2
        by_feat = {r["feature"]: r for r in rows}
        assert by_feat["temperature_c"]["delay_machine_samples"] == 0
        assert by_feat["temperature_c"]["repeated_episode_alerts"] == 1
        assert by_feat["vibration_mm_s"]["status"] == "missed"
        assert by_feat["vibration_mm_s"]["false_alarms_stable"] == 1

    def test_missing_drift_gt_and_out_of_order_obs_raise_summary_error(self) -> None:
        gt = [
            {
                "event_id": "e0",
                "machine_id": "machine_01",
                "ground_truth_label": "normal",
                "is_anomaly": False,
                "stream_index": 0,
                "drift_active": False,
            },
            {
                "event_id": "e1",
                "machine_id": "machine_01",
                "ground_truth_label": "normal",
                "is_anomaly": False,
                "stream_index": 1,
                "drift_active": False,
            },
        ]
        obs = [
            {
                "event_id": "e1",
                "machine_id": "machine_01",
                "stream_position": 0,
                "machine_position": 0,
                "temperature_c": 50.0,
                "vibration_mm_s": 2.0,
                "pressure_bar": 5.0,
            },
            {
                "event_id": "e0",
                "machine_id": "machine_01",
                "stream_position": 1,
                "machine_position": 1,
                "temperature_c": 50.0,
                "vibration_mm_s": 2.0,
                "pressure_bar": 5.0,
            },
        ]
        mon_sum = {
            "received": 2,
            "malformed": 0,
            "static_alerts_observed": 0,
            "drift_events_triggered": 0,
            "drift_events_delivered": 0,
            "drift_events_failed": 0,
        }
        with pytest.raises(SummaryError, match="out-of-order"):
            evaluate_m3_scenario("normal", {"delivered": 2}, mon_sum, [], gt, [], obs)

        with pytest.raises(SummaryError, match="missing ground truth for drift event"):
            evaluate_m3_scenario(
                "normal",
                {"delivered": 2},
                mon_sum,
                [],
                gt[:1],
                [
                    {
                        "event_id": "unknown",
                        "machine_id": "machine_01",
                        "feature": "temperature_c",
                        "machine_position": 0,
                        "stream_position": 0,
                    }
                ],
                obs[1:],
            )

    def test_multi_seed_robustness_inprocess_cli(self, tmp_path: Path) -> None:
        from app.m3_robustness import main as robustness_main

        out_dir = tmp_path / "robustness"
        rc = robustness_main(
            [
                "--run-id",
                "rob-test",
                "--seeds",
                "42",
                "43",
                "44",
                "--count",
                "300",
                "--mode",
                "inprocess",
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        for name in ("robustness_report.json", "robustness_report.md", "seed_runs.csv"):
            assert (out_dir / name).exists() and (out_dir / name).stat().st_size > 0
        data = json.loads((out_dir / "robustness_report.json").read_text(encoding="utf-8"))
        assert data["adwin_parameters"]["delta"] == 0.002
        assert data["static_threshold_parameters"]["temp_max_c"] == 70.0
        assert data["scenarios"]["normal"]["detection_rate"]["mean"] is None
        assert data["scenarios"]["abrupt_drift"]["detection_rate"]["count"] == 3
        with (out_dir / "seed_runs.csv").open(encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == 9
