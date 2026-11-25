"""Live real-time streaming web dashboard for DriftGuard-Kafka.

Serves a clean, interactive browser UI on `http://127.0.0.1:8765` backed by a
live Server-Sent Events (`/api/stream`) stream. When the user clicks a scenario
button ("Normal", "Injected Faults", "Gradual Drift", "Abrupt Drift"), the
backend streams events through Apache Kafka (when reachable, falling back to
in-memory streaming if Kafka is down), runs both the M2 threshold detector and
M3 River ADWIN drift monitor live, and pushes every point to the graph as it
happens with active pulsing markers.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from confluent_kafka import Consumer, KafkaException, Producer, TopicPartition

from app.config import Settings, load_settings, positive_int
from app.consumer import build_consumer_config
from app.detector import SensorReading, ThresholdConfig, ThresholdDetector
from app.drift_monitor import AdwinConfig, MachineAdwinMonitor
from app.logging_config import configure_logging
from app.producer import DeliveryTracker, build_producer_config
from app.schemas import parse_event_json
from app.simulator import Scenario, TelemetrySimulator

logger = logging.getLogger(__name__)

_LIVE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DriftGuard-Kafka — Live Streaming Monitor</title>
  <style>
    :root {
      --bg: #0b1120; --panel: #131c31; --border: #24324d;
      --text: #f1f5f9; --muted: #94a3b8; --blue: #38bdf8;
      --green: #22c55e; --amber: #f59e0b; --red: #ef4444; --purple: #c084fc;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 20px; background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .shell { max-width: 1180px; margin: 0 auto; display: grid; gap: 16px; }
    .bar {
      background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
      padding: 14px 18px; display: flex; flex-wrap: wrap; gap: 12px;
      align-items: center; justify-content: space-between;
    }
    .title { font-size: 18px; font-weight: 700; display: flex; align-items: center; gap: 10px; }
    .live-dot {
      width: 10px; height: 10px; border-radius: 50%; background: var(--green);
      box-shadow: 0 0 10px var(--green); animation: blink 1.2s infinite;
    }
    @keyframes blink { 50% { opacity: 0.35; } }
    .subtitle { font-size: 12px; color: var(--muted); margin-top: 3px; }
    .btns { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    button, select {
      background: #0f172a; color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: 8px 14px; font-size: 13px; cursor: pointer;
    }
    button:hover { border-color: var(--blue); }
    button.active {
      background: #0284c7; border-color: var(--blue); color: #fff; font-weight: 600;
    }
    .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 14px 16px;
    }
    .lbl { font-size: 12px; color: var(--muted); }
    .val { font-size: 26px; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
    .grid-main { display: grid; grid-template-columns: 1fr 320px; gap: 16px; }
    @media (max-width: 920px) {
      .cards { grid-template-columns: repeat(2, 1fr); }
      .grid-main { grid-template-columns: 1fr; }
    }
    canvas {
      width: 100%; height: 370px; display: block;
      background: #090f1d; border-radius: 10px; border: 1px solid var(--border);
    }
    .legend {
      display: flex; flex-wrap: wrap; gap: 16px; margin-top: 10px;
      font-size: 12px; color: var(--muted);
    }
    .dot {
      display: inline-block; width: 10px; height: 10px;
      border-radius: 50%; margin-right: 6px; vertical-align: middle;
    }
    .feed {
      height: 370px; overflow-y: auto; display: flex;
      flex-direction: column; gap: 8px;
    }
    .ev {
      background: #0f172a; border-left: 3px solid var(--purple);
      border-radius: 6px; padding: 8px 10px; font-size: 12px;
    }
    .ev.alert { border-left-color: var(--amber); }
    .ev-top { font-weight: 600; display: flex; justify-content: space-between; }
    .ev-body { color: var(--muted); margin-top: 3px; font-family: ui-monospace, monospace; }
  </style>
</head>
<body>
  <div class="shell">
    <div class="bar">
      <div>
        <div class="title">
          <span class="live-dot" id="liveDot"></span>
          DriftGuard-Kafka Live Streaming Dashboard
        </div>
        <div class="subtitle" id="transportInfo">Connecting to live stream...</div>
      </div>
      <div class="btns">
        <span style="font-size:12px;color:var(--muted)">1. Pick Scenario:</span>
        <button data-sc="normal" class="sc">Normal</button>
        <button data-sc="anomalies" class="sc">Faults (Spikes/Stuck)</button>
        <button data-sc="drift" class="sc active">Gradual Drift</button>
        <button data-sc="abrupt_drift" class="sc">Abrupt Drift</button>
      </div>
    </div>

    <div class="bar">
      <div class="btns">
        <span style="font-size:12px;color:var(--muted)">2. Sensor on Graph:</span>
        <button data-f="temperature_c" class="ft active">Temperature (°C)</button>
        <button data-f="vibration_mm_s" class="ft">Vibration (mm/s)</button>
        <button data-f="pressure_bar" class="ft">Pressure (bar)</button>
      </div>
      <div class="btns">
        <button id="pauseBtn">Pause</button>
        <button id="restartBtn">Restart Stream</button>
      </div>
    </div>

    <div class="cards">
      <div class="card">
        <div class="lbl">Current Sensor Reading</div>
        <div class="val" id="curVal" style="color:var(--blue)">—</div>
      </div>
      <div class="card">
        <div class="lbl">Live Events Received</div>
        <div class="val" id="cntVal">0</div>
      </div>
      <div class="card">
        <div class="lbl">ADWIN Drift Detections</div>
        <div class="val" id="driftCnt" style="color:var(--purple)">0</div>
      </div>
      <div class="card">
        <div class="lbl">Threshold Alerts Marked</div>
        <div class="val" id="alertCnt" style="color:var(--amber)">0</div>
      </div>
    </div>

    <div class="grid-main">
      <div class="card">
        <canvas id="chart" width="820" height="370"></canvas>
        <div class="legend">
          <span>
            <span class="dot" style="background:var(--blue)"></span>Live Telemetry (`machine_01`)
          </span>
          <span><span class="dot" style="background:var(--amber)"></span>Threshold Alert</span>
          <span><span class="dot" style="background:var(--purple)"></span>ADWIN Drift Cut</span>
          <span><span class="dot" style="background:var(--red)"></span>Safe Operating Limits</span>
        </div>
      </div>
      <div class="card">
        <div style="font-weight:600;margin-bottom:10px;font-size:14px">
          Live Detection Feed
        </div>
        <div class="feed" id="feed"></div>
      </div>
    </div>
  </div>

  <script>
    const BOUNDS = {
      temperature_c: { unit: "°C", min: 30.0, max: 70.0 },
      vibration_mm_s: { unit: "mm/s", min: 0.5, max: 5.0 },
      pressure_bar: { unit: "bar", min: 2.5, max: 7.5 }
    };
    let scenario = "drift", feature = "temperature_c", paused = false;
    let points = [], pulses = [], driftTotal = 0, alertTotal = 0, es = null;

    function startStream() {
      if (es) es.close();
      points = []; pulses = []; driftTotal = 0; alertTotal = 0; paused = false;
      document.getElementById("pauseBtn").textContent = "Pause";
      document.getElementById("feed").innerHTML = "";
      document.getElementById("cntVal").textContent = "0";
      document.getElementById("driftCnt").textContent = "0";
      document.getElementById("alertCnt").textContent = "0";

      es = new EventSource("/api/stream?scenario=" + encodeURIComponent(scenario));
      es.onmessage = (ev) => {
        if (paused) return;
        const msg = JSON.parse(ev.data);
        if (msg.type === "meta") {
          document.getElementById("transportInfo").textContent =
            "Source: " + msg.transport + " | Scenario: " + msg.scenario +
            " | Plotting machine_01 in real time";
          return;
        }
        if (msg.type === "done") {
          document.getElementById("transportInfo").textContent +=
            " — Stream complete (click Restart or pick another scenario)";
          es.close();
          return;
        }
        document.getElementById("cntVal").textContent = String(msg.stream_pos + 1);
        if (msg.drift_features.length > 0) {
          driftTotal += msg.drift_features.length;
          document.getElementById("driftCnt").textContent = String(driftTotal);
          logEvent(
            "drift",
            msg.machine_id + " ADWIN Drift",
            msg.drift_features.join(", ") + " shifted"
          );
        }
        if (msg.is_alert) {
          alertTotal += 1;
          document.getElementById("alertCnt").textContent = String(alertTotal);
          logEvent("alert", msg.machine_id + " " + msg.alert_reason, msg.alert_rules.join(", "));
        }
        if (msg.machine_id === "machine_01") {
          points.push(msg);
          if (points.length > 140) points.shift();
          if (msg.drift_features.length > 0) {
            pulses.push({ pt: msg, color: "#c084fc", born: performance.now() });
          } else if (msg.is_alert) {
            pulses.push({ pt: msg, color: "#f59e0b", born: performance.now() });
          }
          const b = BOUNDS[feature];
          document.getElementById("curVal").textContent = msg[feature].toFixed(2) + " " + b.unit;
        }
      };
    }

    function logEvent(kind, title, detail) {
      const feed = document.getElementById("feed");
      const div = document.createElement("div");
      div.className = kind === "alert" ? "ev alert" : "ev";
      const col = kind === "alert" ? "#f59e0b" : "#c084fc";
      div.innerHTML = '<div class="ev-top"><span style="color:' + col + '">' + title +
        '</span></div><div class="ev-body">' + detail + '</div>';
      feed.prepend(div);
      while (feed.children.length > 40) feed.removeChild(feed.lastChild);
    }

    function renderLoop() {
      draw();
      requestAnimationFrame(renderLoop);
    }

    function draw() {
      const canvas = document.getElementById("chart");
      const ctx = canvas.getContext("2d");
      const W = canvas.width, H = canvas.height;
      const pad = { l: 48, r: 20, t: 20, b: 26 };
      ctx.clearRect(0, 0, W, H);

      const cfg = BOUNDS[feature];
      const vals = points.map(p => p[feature]);
      const yMin = (vals.length ? Math.min(...vals, cfg.min) : cfg.min) - 2.5;
      const yMax = (vals.length ? Math.max(...vals, cfg.max) : cfg.max) + 2.5;
      const slots = Math.max(60, points.length);
      const xOf = (i) => pad.l + (i / Math.max(1, slots - 1)) * (W - pad.l - pad.r);
      const ySpan = Math.max(1e-6, yMax - yMin);
      const yOf = (v) => H - pad.b - ((v - yMin) / ySpan) * (H - pad.t - pad.b);

      ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1;
      ctx.fillStyle = "#94a3b8"; ctx.font = "11px monospace";
      for (let g = 0; g <= 4; g++) {
        const v = yMin + (g / 4) * (yMax - yMin);
        const y = yOf(v);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
        ctx.fillText(v.toFixed(1), 6, y + 4);
      }

      ctx.save();
      ctx.strokeStyle = "#ef4444"; ctx.setLineDash([6, 4]);
      [cfg.min, cfg.max].forEach(th => {
        const y = yOf(th);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      });
      ctx.restore();

      if (!points.length) return;
      ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2.2;
      ctx.beginPath();
      points.forEach((p, i) => {
        const x = xOf(i), y = yOf(p[feature]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      points.forEach((p, i) => {
        const x = xOf(i), y = yOf(p[feature]);
        const hasDrift = p.drift_features.includes(feature);
        const outBounds = p[feature] < cfg.min || p[feature] > cfg.max || p.is_alert;
        if (hasDrift) {
          ctx.strokeStyle = "#c084fc"; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
          ctx.fillStyle = "#c084fc";
          ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
        } else if (outBounds) {
          ctx.fillStyle = "#f59e0b";
          ctx.beginPath(); ctx.arc(x, y, 4.2, 0, Math.PI * 2); ctx.fill();
        } else if (i === points.length - 1) {
          ctx.fillStyle = "#38bdf8";
          ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
        }
      });

      const now = performance.now();
      pulses = pulses.filter(pl => now - pl.born < 1000);
      pulses.forEach(pl => {
        const idx = points.indexOf(pl.pt);
        if (idx >= 0) {
          const age = (now - pl.born) / 1000;
          const x = xOf(idx), y = yOf(pl.pt[feature]);
          ctx.save();
          ctx.strokeStyle = pl.color;
          ctx.globalAlpha = 1 - age;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.arc(x, y, 6 + age * 24, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        }
      });
    }

    document.querySelectorAll(".sc").forEach(b => {
      b.onclick = () => {
        scenario = b.dataset.sc;
        document.querySelectorAll(".sc").forEach(
          x => x.classList.toggle("active", x.dataset.sc === scenario)
        );
        startStream();
      };
    });
    document.querySelectorAll(".ft").forEach(b => {
      b.onclick = () => {
        feature = b.dataset.f;
        document.querySelectorAll(".ft").forEach(
          x => x.classList.toggle("active", x.dataset.f === feature)
        );
      };
    });
    document.getElementById("pauseBtn").onclick = () => {
      paused = !paused;
      document.getElementById("pauseBtn").textContent = paused ? "Resume" : "Pause";
    };
    document.getElementById("restartBtn").onclick = startStream;

    startStream();
    requestAnimationFrame(renderLoop);
  </script>
</body>
</html>
"""


def stream_scenario_events(
    settings: Settings,
    scenario: Scenario,
    count: int,
    interval_s: float,
    out_queue: queue.Queue[dict[str, Any] | None],
    stop_flag: threading.Event,
) -> None:
    """Stream events through live Kafka when reachable, else fallback in-memory."""
    sim = TelemetrySimulator(machine_count=5, seed=42, scenario=scenario)
    detector = ThresholdDetector(ThresholdConfig())
    adwin = MachineAdwinMonitor(AdwinConfig(delta=0.002, clock=1))

    use_kafka = False
    consumer: Consumer | None = None
    producer: Producer | None = None
    try:
        tag = f"live-{int(time.time() * 1000)}"
        c_cfg = build_consumer_config(
            Settings(**{**settings.__dict__, "consumer_group": f"dg-{tag}"})
        )
        consumer = Consumer(c_cfg)
        tps = []
        for part in range(3):
            _low, high = consumer.get_watermark_offsets(
                TopicPartition(settings.topic, part), timeout=1.5, cached=False
            )
            tps.append(TopicPartition(settings.topic, part, high))
        consumer.assign(tps)
        producer = Producer(build_producer_config(settings))
        use_kafka = True
    except KafkaException:
        if consumer is not None:
            consumer.close()
        consumer = None
        producer = None

    out_queue.put(
        {
            "type": "meta",
            "transport": f"Apache Kafka ({settings.topic})" if use_kafka else "In-Memory Simulator",
            "scenario": scenario.value,
        }
    )

    tracker = DeliveryTracker()
    try:
        for idx, (ev, _gt) in enumerate(sim.generate_scenario_events(count)):
            if stop_flag.is_set():
                break
            if use_kafka and producer is not None and consumer is not None:
                producer.produce(
                    topic=settings.topic,
                    key=ev.machine_id.encode("utf-8"),
                    value=ev.to_json_bytes(),
                    on_delivery=tracker.on_delivery,
                )
                producer.poll(0)
                msg = consumer.poll(0.2)
                wire_ev = (
                    parse_event_json(msg.value()) if msg is not None and msg.error() is None else ev
                )
            else:
                wire_ev = ev

            reading = SensorReading.from_telemetry(wire_ev)
            det_res = detector.evaluate(reading)
            _s_pos, m_pos, drift_evs = adwin.update(reading)
            out_queue.put(
                {
                    "type": "point",
                    "stream_pos": idx,
                    "machine_pos": m_pos,
                    "machine_id": reading.machine_id,
                    "temperature_c": reading.temperature_c,
                    "vibration_mm_s": reading.vibration_mm_s,
                    "pressure_bar": reading.pressure_bar,
                    "is_alert": det_res.is_anomaly,
                    "alert_reason": det_res.reason,
                    "alert_rules": list(det_res.triggered_rules),
                    "drift_features": [d.feature for d in drift_evs],
                }
            )
            time.sleep(interval_s)
        out_queue.put({"type": "done"})
    finally:
        out_queue.put(None)
        if producer is not None:
            producer.flush(2.0)
        if consumer is not None:
            consumer.close()


def make_handler(settings: Settings, count: int, interval_s: float) -> type[BaseHTTPRequestHandler]:
    """Create HTTP handler class serving `/` and SSE `/api/stream`."""

    class LiveDashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug(fmt, *args)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = _LIVE_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/stream":
                qs = parse_qs(parsed.query)
                sc_raw = qs.get("scenario", ["drift"])[0]
                try:
                    scenario = Scenario(sc_raw)
                except ValueError:
                    scenario = Scenario.DRIFT

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                q: queue.Queue[dict[str, Any] | None] = queue.Queue()
                stop_flag = threading.Event()
                worker = threading.Thread(
                    target=stream_scenario_events,
                    args=(settings, scenario, count, interval_s, q, stop_flag),
                    daemon=True,
                )
                worker.start()
                try:
                    while True:
                        item = q.get(timeout=10.0)
                        if item is None:
                            break
                        payload = f"data: {json.dumps(item, separators=(',', ':'))}\n\n"
                        self.wfile.write(payload.encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, queue.Empty):
                    stop_flag.set()
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    return LiveDashboardHandler


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the live streaming dashboard server."""
    parser = argparse.ArgumentParser(description="Run DriftGuard-Kafka live web dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive_int, default=8765)
    parser.add_argument("--count", type=positive_int, default=500)
    parser.add_argument("--interval-ms", type=positive_int, default=35)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Start the live HTTP + SSE streaming dashboard server."""
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)
    handler = make_handler(settings, args.count, args.interval_ms / 1000.0)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    logger.info(
        "live dashboard listening",
        extra={"fields": {"url": f"http://{args.host}:{args.port}"}},
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
