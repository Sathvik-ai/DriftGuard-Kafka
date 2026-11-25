"""Unit and smoke tests for DriftGuard-Kafka M4 comparative ML benchmarking."""

from __future__ import annotations

import csv
import inspect
import io
import json
import queue
import tarfile
import threading
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from confluent_kafka import KafkaError, KafkaException

from app.config import Settings
from app.datasets import (
    DatasetFormatError,
    DatasetNotAvailableError,
    compute_sha256,
    ensure_skab_downloaded,
    ensure_tsb_ad_m_downloaded,
    load_skab_suite,
    load_tsb_ad_m_suite,
    parse_skab_csv,
    parse_tsb_train_index,
    split_skab_experiment,
    split_tsb_csv,
)
from app.live_dashboard import make_handler, stream_scenario_events
from app.live_dashboard import parse_args as parse_live_args
from app.m4_benchmark import compute_ranking_metrics
from app.m4_benchmark import main as m4_main
from app.m4_models import (
    HalfSpaceTreesDetector,
    IsolationForestDetector,
    StaticThresholdBaseline,
    build_default_detectors,
)
from app.simulator import Scenario


def _write_synthetic_skab_tree(skab_root: Path) -> Path:
    rng = np.random.default_rng(123)
    data_dir = skab_root / "SKAB-master" / "data"
    (data_dir / "valve1").mkdir(parents=True, exist_ok=True)
    (data_dir / "other").mkdir(parents=True, exist_ok=True)
    (data_dir / "anomaly-free").mkdir(parents=True, exist_ok=True)

    # 1. Standard fault experiment: normal [:110], anomalous [110:150], cp at 110 & 150.
    lines_v1 = ["datetime;Accelerometer1RMS;Pressure;Temperature;anomaly;changepoint"]
    for i in range(180):
        anom = 1 if 110 <= i < 150 else 0
        cp = 1 if i in (110, 150) else 0
        shift = 9.0 if anom else 0.0
        a1 = 0.4 + 0.02 * rng.normal() + 0.5 * shift
        pr = 5.0 + 0.05 * rng.normal() + shift
        tp = 50.0 + 0.10 * rng.normal() + 2.5 * shift
        lines_v1.append(
            f"2026-01-01 00:{i // 60:02d}:{i % 60:02d};{a1:.5f};{pr:.5f};{tp:.5f};{anom};{cp}"
        )
    (data_dir / "valve1" / "0.csv").write_text("\n".join(lines_v1) + "\n", encoding="utf-8")

    # 2. Early-anomaly SKAB file (mirrors real SKAB other/2.csv where anomaly starts < train_end).
    lines_o2 = ["datetime;Accelerometer1RMS;Pressure;Temperature;anomaly;changepoint"]
    for i in range(180):
        anom = 1 if (40 <= i < 85 or 120 <= i < 155) else 0
        cp = 1 if i in (40, 85, 120, 155) else 0
        shift = 8.0 if anom else 0.0
        a1 = 0.4 + 0.02 * rng.normal() + 0.4 * shift
        pr = 5.0 + 0.05 * rng.normal() + shift
        tp = 50.0 + 0.10 * rng.normal() + 2.0 * shift
        lines_o2.append(
            f"2026-01-02 00:{i // 60:02d}:{i % 60:02d};{a1:.5f};{pr:.5f};{tp:.5f};{anom};{cp}"
        )
    (data_dir / "other" / "2.csv").write_text("\n".join(lines_o2) + "\n", encoding="utf-8")

    # 3. Anomaly-free baseline (no anomaly or changepoint columns).
    lines_af = ["datetime;Accelerometer1RMS;Pressure;Temperature"]
    for i in range(150):
        a1 = 0.4 + 0.02 * rng.normal()
        pr = 5.0 + 0.05 * rng.normal()
        tp = 50.0 + 0.10 * rng.normal()
        lines_af.append(f"2026-01-03 00:{i // 60:02d}:{i % 60:02d};{a1:.5f};{pr:.5f};{tp:.5f}")
    (data_dir / "anomaly-free" / "anomaly-free.csv").write_text(
        "\n".join(lines_af) + "\n", encoding="utf-8"
    )
    return data_dir


def _write_synthetic_tsb_tree(tsb_root: Path) -> Path:
    rng = np.random.default_rng(456)
    data_dir = tsb_root / "TSB-AD-M"
    data_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("001_MSL_id_1_Sensor_tr_100_1st_130.csv", 180, 100, [(130, 155)]),
        ("002_MSL_id_2_Sensor_tr_90_1st_120.csv", 170, 90, [(120, 145)]),
        # Early anomaly inside [:tr_100] (mirrors TAO/SMD files with 1st < tr)
        ("003_TAO_id_1_Environment_tr_100_1st_60.csv", 180, 100, [(60, 80), (135, 160)]),
    ]
    eva_lines = ["file_name"]
    for fname, n_rows, _tr, windows in specs:
        eva_lines.append(fname)
        csv_lines = ["f0,f1,f2,Label"]
        for i in range(n_rows):
            lbl = 1 if any(s <= i < e for s, e in windows) else 0
            bump = 7.5 if lbl else 0.0
            f0 = float(np.sin(i / 6.0)) + 0.05 * rng.normal() + bump
            f1 = float(np.cos(i / 6.0)) + 0.05 * rng.normal() + bump
            f2 = 2.0 + 0.05 * rng.normal() + bump
            csv_lines.append(f"{f0:.5f},{f1:.5f},{f2:.5f},{lbl}")
        (data_dir / fname).write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    (tsb_root / "TSB-AD-M-Eva.csv").write_text("\n".join(eva_lines) + "\n", encoding="utf-8")
    return data_dir


def test_skab_preserves_anomaly_and_changepoint_separately_and_filters_normal_train(
    tmp_path: Path,
) -> None:
    data_dir = _write_synthetic_skab_tree(tmp_path / "skab")
    exp = parse_skab_csv(data_dir / "other" / "2.csv", data_dir)
    assert exp.experiment_id == "other/2.csv"
    assert not exp.is_anomaly_free
    assert int(exp.y_anomaly.sum()) == 80
    assert int(exp.y_changepoint.sum()) == 4
    assert not np.array_equal(exp.y_anomaly, exp.y_changepoint)

    # With default_train_size=100, indices [40:85] are anomalous in [:90] (since n_rows//2 == 90).
    split = split_skab_experiment(exp, default_train_size=100, val_fraction=0.2)
    assert split.train_boundary_index == 90
    assert split.dropped_train_anomalies == 45
    assert split.x_train_normal.shape[0] + split.x_val_normal.shape[0] == 45
    # Test split is strictly chronological [90:]
    assert np.allclose(split.x_test, exp.features[90:])
    assert np.array_equal(split.y_test_anomaly, exp.y_anomaly[90:])
    assert split.y_test_changepoint is not None
    assert np.array_equal(split.y_test_changepoint, exp.y_changepoint[90:])


def test_tsb_ad_m_adapter_parses_tr_split_and_stratifies_domains(tmp_path: Path) -> None:
    domain, tr = parse_tsb_train_index("018_Daphnet_id_1_HumanActivity_tr_9693_1st_20732.csv")
    assert domain == "Daphnet"
    assert tr == 9693

    with pytest.raises(DatasetFormatError):
        parse_tsb_train_index("invalid_filename_without_split.csv")

    tsb_root = tmp_path / "tsb"
    _write_synthetic_tsb_tree(tsb_root)

    full_bundle = load_tsb_ad_m_suite(tsb_root, max_per_domain=None)
    assert len(full_bundle.splits) == 3
    assert full_bundle.provenance["domains_covered_count"] == 2

    strat_bundle = load_tsb_ad_m_suite(tsb_root, max_per_domain=1)
    assert len(strat_bundle.splits) == 2
    assert {s.domain for s in strat_bundle.splits} == {"MSL", "TAO"}

    tao_split = next(s for s in full_bundle.splits if s.domain == "TAO")
    assert tao_split.train_boundary_index == 100
    assert tao_split.dropped_train_anomalies == 20
    assert tao_split.x_train_normal.shape[0] + tao_split.x_val_normal.shape[0] == 80


def test_missing_datasets_fail_loudly_never_silently_skipped(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotAvailableError):
        load_skab_suite(tmp_path / "does_not_exist_skab")
    with pytest.raises(DatasetNotAvailableError):
        load_tsb_ad_m_suite(tmp_path / "does_not_exist_tsb")

    rc = m4_main(
        [
            "--run-id",
            "missing_check",
            "--skab-dir",
            str(tmp_path / "missing_skab"),
            "--tsb-dir",
            str(tmp_path / "missing_tsb"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2


def test_models_exclude_labels_and_hst_enforces_warmup_and_score_before_learn() -> None:
    rng = np.random.default_rng(7)
    x_train = rng.normal(loc=10.0, scale=0.5, size=(40, 3))
    x_val = rng.normal(loc=10.0, scale=0.5, size=(15, 3))
    x_test = np.vstack(
        [
            rng.normal(loc=10.0, scale=0.5, size=(20, 3)),
            np.full((6, 3), 10.0),  # stuck sensor streak >= 4
            rng.normal(loc=25.0, scale=0.5, size=(10, 3)),  # large multivariate spike
        ]
    )
    feats = ("f0", "f1", "f2")

    for det in build_default_detectors(seed=42, calibration_quantile=0.95):
        fit_params = list(inspect.signature(det.fit_and_calibrate).parameters)
        score_params = list(inspect.signature(det.score_and_predict).parameters)
        assert fit_params == ["x_train_normal", "x_val_normal", "feature_names"]
        assert score_params == ["x_test", "feature_names"]

    # Half-Space Trees: requested window_size=100 > len(x_train)=40 -> shrinks warm-up window
    # so warm-up completes before validation, and strictly executes score before learn.
    hst = HalfSpaceTreesDetector(
        n_trees=15, height=5, window_size=100, seed=42, calibration_quantile=0.95
    )
    hst.enable_trace(True)
    cal = hst.fit_and_calibrate(x_train, x_val, feats)
    assert cal.parameters["effective_window_size"] == 40
    # Clear validation trace and verify test trace order item-by-item.
    hst.enable_trace(False)
    hst.enable_trace(True)
    inf = hst.score_and_predict(x_test, feats)
    assert inf.extra_diagnostics["warmup_complete_before_test"] is True
    assert inf.extra_diagnostics["score_before_learn"] is True

    expected_trace: list[tuple[str, int]] = []
    for idx in range(x_test.shape[0]):
        expected_trace.append(("score", idx))
        expected_trace.append(("learn", idx))
    assert hst.operation_trace == expected_trace
    assert np.mean(inf.scores[-10:]) > np.mean(inf.scores[:20])

    # Static threshold baseline detects both stuck streaks and out-of-band spikes.
    st = StaticThresholdBaseline(z_bound=3.0, stuck_window=4, calibration_quantile=0.95)
    st.fit_and_calibrate(x_train, x_val, feats)
    st_inf = st.score_and_predict(x_test, feats)
    assert st_inf.extra_diagnostics["stuck_sensor_triggers"] >= 3
    assert int(np.sum(st_inf.predictions[-10:])) == 10

    # Isolation Forest threshold depends strictly on x_val, never test data.
    ifo = IsolationForestDetector(n_estimators=50, seed=42, calibration_quantile=0.95)
    ifo.fit_and_calibrate(x_train, x_val, feats)
    ifo_inf = ifo.score_and_predict(x_test, feats)
    assert int(np.sum(ifo_inf.predictions[-10:])) >= 8
    assert ifo_inf.throughput_events_per_sec is not None and ifo_inf.throughput_events_per_sec > 0
    assert ifo_inf.rss_memory_mib > 0


def test_ranking_metrics_return_none_for_anomaly_free_and_valid_vus_pr_for_anomalous() -> None:
    y_norm = np.zeros(50, dtype=np.int64)
    scores_norm = np.linspace(0.0, 1.0, 50)
    preds_norm = np.zeros(50, dtype=np.int64)
    assert compute_ranking_metrics(y_norm, scores_norm, preds_norm, sliding_window=10) == {
        "pr_auc": None,
        "vus_pr": None,
        "vus_roc": None,
    }

    y_anom = np.array([0] * 30 + [1] * 10 + [0] * 20, dtype=np.int64)
    scores_anom = np.array([0.1] * 30 + [0.95] * 10 + [0.15] * 20, dtype=np.float64)
    preds_anom = (scores_anom > 0.5).astype(np.int64)
    res = compute_ranking_metrics(y_anom, scores_anom, preds_anom, sliding_window=10)
    assert res["pr_auc"] == 1.0
    assert res["vus_pr"] is not None and res["vus_pr"] > 0.9


def test_end_to_end_m4_benchmark_generates_reproducible_csv_json_md(tmp_path: Path) -> None:
    skab_root = tmp_path / "skab"
    tsb_root = tmp_path / "tsb"
    _write_synthetic_skab_tree(skab_root)
    _write_synthetic_tsb_tree(tsb_root)

    out_dir = tmp_path / "reports_m4"
    rc = m4_main(
        [
            "--run-id",
            "test_m4_run",
            "--seed",
            "42",
            "--calibration-quantile",
            "0.98",
            "--skab-dir",
            str(skab_root),
            "--tsb-dir",
            str(tsb_root),
            "--tsb-max-per-domain",
            "0",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    report_json = out_dir / "benchmark_report.json"
    report_md = out_dir / "benchmark_report.md"
    summary_csv = out_dir / "comparison_summary.csv"
    per_exp_csv = out_dir / "per_experiment_metrics.csv"
    for fpath in (report_json, report_md, summary_csv, per_exp_csv):
        assert fpath.exists() and fpath.stat().st_size > 0

    assert len(compute_sha256(report_json)) == 64
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["reproducibility"]["seed"] == 42
    assert report["reproducibility"]["adwin_integration"] == "independent_no_adaptive_retraining"
    assert set(report["dataset_provenance"]) == {"SKAB", "TSB-AD-M"}
    assert report["dataset_provenance"]["SKAB"]["labels_preserved_separately"] == [
        "anomaly",
        "changepoint",
    ]
    assert set(report["tsb_ad_m_domain_breakdown"]) == {"MSL", "TAO"}

    with per_exp_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # (3 SKAB files + 3 TSB-AD-M files) * 3 models = 18 experiment-model evaluations
    assert len(rows) == 18

    md_text = report_md.read_text(encoding="utf-8")
    assert "DriftGuard-Kafka M4 Comparative ML Benchmark Report" in md_text
    assert "SKAB Separate `changepoint` Preservation Summary" in md_text
    assert "TSB-AD-M Per-Domain VUS-PR & F1 Breakdown" in md_text


def test_split_tsb_csv_validation_errors(tmp_path: Path) -> None:
    bad_col = tmp_path / "001_MSL_id_1_Sensor_tr_10_1st_15.csv"
    bad_col.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(DatasetFormatError):
        split_tsb_csv(bad_col)

    bad_tr = tmp_path / "002_MSL_id_2_Sensor_tr_50_1st_55.csv"
    bad_tr.write_text("a,Label\n1,0\n2,0\n3,1\n", encoding="utf-8")
    with pytest.raises(DatasetFormatError):
        split_tsb_csv(bad_tr)


def test_live_dashboard_in_memory_fallback_and_http_routes() -> None:
    args = parse_live_args(["--port", "8799", "--count", "15", "--interval-ms", "1"])
    assert args.port == 8799
    assert args.count == 15

    settings = Settings(
        bootstrap_servers="localhost:19999",
        topic="research.telemetry.raw",
        alerts_topic="research.telemetry.alerts",
        drift_topic="research.telemetry.drift",
        consumer_group="test-live",
        client_id="test-client",
        machine_count=5,
        log_level="WARNING",
    )
    q: queue.Queue[dict[str, object] | None] = queue.Queue()
    stop_flag = threading.Event()
    with patch(
        "app.live_dashboard.Consumer",
        side_effect=KafkaException(KafkaError(KafkaError._TRANSPORT)),
    ):
        stream_scenario_events(settings, Scenario.ANOMALIES, 12, 0.0, q, stop_flag)

    items: list[dict[str, object]] = []
    while True:
        msg = q.get_nowait()
        if msg is None:
            break
        items.append(msg)
    assert items[0]["type"] == "meta"
    assert items[0]["transport"] == "In-Memory Simulator"
    assert items[-1]["type"] == "done"
    assert sum(1 for x in items if x["type"] == "point") == 12

    handler_cls = make_handler(settings, count=5, interval_s=0.0)

    class _DummySocket:
        def __init__(self, req_bytes: bytes) -> None:
            self._rfile = io.BytesIO(req_bytes)
            self._wfile = io.BytesIO()

        def makefile(self, mode: str, *_args: object) -> io.BytesIO:
            return self._rfile if "r" in mode else self._wfile

        def sendall(self, data: bytes) -> None:
            self._wfile.write(data)

    class _DummyServer:
        server_name = "127.0.0.1"
        server_port = 8799

    sock_root = _DummySocket(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
    handler_cls(sock_root, ("127.0.0.1", 55555), _DummyServer())  # type: ignore[arg-type]
    resp_root = sock_root._wfile.getvalue().decode("utf-8")
    assert "200 OK" in resp_root
    assert "DriftGuard-Kafka Live Streaming Dashboard" in resp_root

    sock_404 = _DummySocket(b"GET /missing HTTP/1.1\r\nHost: localhost\r\n\r\n")
    handler_cls(sock_404, ("127.0.0.1", 55556), _DummyServer())  # type: ignore[arg-type]
    assert "404" in sock_404._wfile.getvalue().decode("utf-8")

    with patch(
        "app.live_dashboard.Consumer",
        side_effect=KafkaException(KafkaError(KafkaError._TRANSPORT)),
    ):
        sock_sse = _DummySocket(
            b"GET /api/stream?scenario=invalid_sc HTTP/1.1\r\nHost: localhost\r\n\r\n"
        )
        handler_cls(sock_sse, ("127.0.0.1", 55557), _DummyServer())  # type: ignore[arg-type]
        sse_text = sock_sse._wfile.getvalue().decode("utf-8")
        assert "text/event-stream" in sse_text
        assert '"type":"meta"' in sse_text
        assert '"type":"done"' in sse_text


def test_archive_extraction_and_model_input_guards(tmp_path: Path) -> None:
    # Build local skab-master.tar.gz and verify ensure_skab_downloaded extracts safely.
    src_skab = tmp_path / "src_skab"
    _write_synthetic_skab_tree(src_skab)
    dest_skab = tmp_path / "dest_skab"
    dest_skab.mkdir()
    with tarfile.open(dest_skab / "skab-master.tar.gz", "w:gz") as tar:
        tar.add(src_skab / "SKAB-master", arcname="SKAB-master")
    extracted_skab = ensure_skab_downloaded(dest_skab)
    assert extracted_skab.is_dir()

    # Build local TSB-AD-M.zip and verify ensure_tsb_ad_m_downloaded extracts safely.
    src_tsb = tmp_path / "src_tsb"
    _write_synthetic_tsb_tree(src_tsb)
    dest_tsb = tmp_path / "dest_tsb"
    dest_tsb.mkdir()
    (dest_tsb / "TSB-AD-M-Eva.csv").write_text(
        (src_tsb / "TSB-AD-M-Eva.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with zipfile.ZipFile(dest_tsb / "TSB-AD-M.zip", "w") as zf:
        for f in (src_tsb / "TSB-AD-M").glob("*.csv"):
            zf.write(f, arcname=f"TSB-AD-M/{f.name}")
    extracted_tsb = ensure_tsb_ad_m_downloaded(dest_tsb)
    assert extracted_tsb.is_dir()

    with pytest.raises(ValueError):
        StaticThresholdBaseline(calibration_quantile=1.0)
    with pytest.raises(ValueError):
        StaticThresholdBaseline(z_bound=-1.0)
    with pytest.raises(ValueError):
        StaticThresholdBaseline(stuck_window=1)

    det = StaticThresholdBaseline()
    with pytest.raises(RuntimeError):
        _ = det.calibrated_threshold
    with pytest.raises(RuntimeError):
        det.score_and_predict(np.ones((5, 2)), ("a", "b"))
    with pytest.raises(ValueError):
        det.fit_and_calibrate(np.array([[np.nan, 1.0]]), np.ones((4, 2)), ("a", "b"))
