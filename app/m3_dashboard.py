"""Interactive animated HTML dashboard for DriftGuard-Kafka streaming reports.

Generates a clean, plain-English `dashboard.html` that actively streams and
animates telemetry points across the chart in real time, highlighting threshold
alerts and ADWIN concept-drift detections as they happen.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.detector import ThresholdConfig

_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DriftGuard-Kafka Live Replay — __RUN_ID__</title>
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
    .title { font-size: 18px; font-weight: 700; }
    .subtitle { font-size: 12px; color: var(--muted); margin-top: 2px; }
    .pills { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    button, select {
      background: #0f172a; color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: 7px 12px; font-size: 13px; cursor: pointer;
      transition: all 0.15s ease;
    }
    button:hover { border-color: var(--blue); }
    button.active {
      background: rgba(56, 189, 248, 0.16); border-color: var(--blue);
      color: var(--blue); font-weight: 600;
    }
    button.play-btn {
      background: #0284c7; border-color: #38bdf8; color: #fff; font-weight: 600;
    }
    .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 14px 16px;
    }
    .card-lbl { font-size: 12px; color: var(--muted); }
    .card-val {
      font-size: 26px; font-weight: 700; margin-top: 4px;
      font-variant-numeric: tabular-nums;
    }
    .main-grid { display: grid; grid-template-columns: 1fr 310px; gap: 16px; }
    @media (max-width: 920px) {
      .cards { grid-template-columns: repeat(2, 1fr); }
      .main-grid { grid-template-columns: 1fr; }
    }
    canvas {
      width: 100%; height: 360px; display: block;
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
      height: 360px; overflow-y: auto; display: flex;
      flex-direction: column; gap: 8px; padding-right: 4px;
    }
    .ev-item {
      background: #0f172a; border-left: 3px solid var(--purple);
      border-radius: 6px; padding: 8px 10px; font-size: 12px;
    }
    .ev-item.warn { border-left-color: var(--amber); }
    .ev-head { font-weight: 600; display: flex; justify-content: space-between; }
    .ev-sub { color: var(--muted); margin-top: 2px; font-family: ui-monospace, monospace; }
  </style>
</head>
<body>
  <div class="shell">
    <div class="bar">
      <div>
        <div class="title">DriftGuard-Kafka — Live Telemetry &amp; Drift Visualizer</div>
        <div class="subtitle" id="statusText">Streaming Run __RUN_ID__</div>
      </div>
      <div class="pills">
        <button id="playBtn" class="play-btn">Pause</button>
        <button id="restartBtn">Replay</button>
        <select id="speedSel" title="Playback speed">
          <option value="80">1x Speed</option>
          <option value="30" selected>3x Speed</option>
          <option value="8">10x Fast</option>
        </select>
      </div>
    </div>

    <div class="bar">
      <div class="pills" id="scenarioPills"></div>
      <div class="pills">
        <select id="sensorSel">
          <option value="temperature_c">Temperature (°C)</option>
          <option value="vibration_mm_s">Vibration (mm/s)</option>
          <option value="pressure_bar">Pressure (bar)</option>
        </select>
        <select id="machineSel"></select>
      </div>
    </div>

    <div class="cards">
      <div class="card">
        <div class="card-lbl">Live Reading</div>
        <div class="card-val" id="liveVal" style="color:var(--blue)">—</div>
      </div>
      <div class="card">
        <div class="card-lbl">Points Streamed</div>
        <div class="card-val" id="ptsVal">0 / 0</div>
      </div>
      <div class="card">
        <div class="card-lbl">ADWIN Drift Cuts Marked</div>
        <div class="card-val" id="driftVal" style="color:var(--purple)">0</div>
      </div>
      <div class="card">
        <div class="card-lbl">Static Threshold Alerts</div>
        <div class="card-val" id="alertVal" style="color:var(--amber)">0</div>
      </div>
    </div>

    <div class="main-grid">
      <div class="card">
        <canvas id="streamCanvas" width="820" height="360"></canvas>
        <div class="legend">
          <span><span class="dot" style="background:var(--blue)"></span>Normal Sensor Stream</span>
          <span>
            <span class="dot" style="background:var(--amber)"></span>Static Threshold Alert
          </span>
          <span>
            <span class="dot" style="background:var(--purple)"></span>ADWIN Drift Detected
          </span>
          <span><span class="dot" style="background:var(--green)"></span>True Drift Start</span>
          <span><span class="dot" style="background:var(--red)"></span>Safe Limits</span>
        </div>
      </div>
      <div class="card">
        <div style="font-weight:600;margin-bottom:10px;font-size:14px">
          Active Event Log
        </div>
        <div class="feed" id="eventFeed"></div>
      </div>
    </div>
  </div>

  <script>
    const DATA = __DASHBOARD_JSON__;
    const LABELS = {
      normal: "Normal Operation",
      drift: "Gradual Drift",
      abrupt_drift: "Abrupt Drift",
      anomalies: "Injected Faults"
    };
    const SENSORS = {
      temperature_c: {
        unit: "°C", min: DATA.thresholds.temp_min_c, max: DATA.thresholds.temp_max_c
      },
      vibration_mm_s: {
        unit: "mm/s", min: DATA.thresholds.vib_min_mm_s, max: DATA.thresholds.vib_max_mm_s
      },
      pressure_bar: {
        unit: "bar", min: DATA.thresholds.press_min_bar, max: DATA.thresholds.press_max_bar
      }
    };

    const scenarios = DATA.report.scenarios.map(s => s.scenario);
    let sc = scenarios.includes("drift") ? "drift" : scenarios[0];
    let machine = "machine_01", sensor = "temperature_c";
    let cursor = 0, playing = true, timer = null, pulses = [];

    function series() {
      return (DATA.streams[sc] || []).filter(p => p.machine_id === machine);
    }

    function setupUI() {
      const box = document.getElementById("scenarioPills");
      box.innerHTML = scenarios.map(s => {
        const cls = s === sc ? "active" : "";
        return '<button data-s="' + s + '" class="' + cls + '">' +
          (LABELS[s] || s) + '</button>';
      }).join("");
      box.querySelectorAll("button").forEach(b => {
        b.onclick = () => {
          sc = b.dataset.s;
          box.querySelectorAll("button").forEach(
            x => x.classList.toggle("active", x.dataset.s === sc)
          );
          syncMachines();
          resetStream();
        };
      });

      document.getElementById("sensorSel").onchange = (e) => {
        sensor = e.target.value;
        drawFrame();
      };
      document.getElementById("playBtn").onclick = () => {
        playing = !playing;
        document.getElementById("playBtn").textContent = playing ? "Pause" : "Play Stream";
      };
      document.getElementById("restartBtn").onclick = resetStream;
      document.getElementById("speedSel").onchange = startLoop;

      syncMachines();
      resetStream();
      startLoop();
      requestAnimationFrame(animatePulses);
    }

    function syncMachines() {
      const pts = DATA.streams[sc] || [];
      const ms = [...new Set(pts.map(p => p.machine_id))].sort();
      if (!ms.includes(machine) && ms.length) machine = ms[0];
      const sel = document.getElementById("machineSel");
      sel.innerHTML = ms.map(m => '<option value="' + m + '">' + m + '</option>').join("");
      sel.value = machine;
      sel.onchange = () => { machine = sel.value; resetStream(); };
    }

    function resetStream() {
      cursor = 0; pulses = []; playing = true;
      document.getElementById("playBtn").textContent = "Pause";
      document.getElementById("eventFeed").innerHTML = "";
      drawFrame();
    }

    function startLoop() {
      if (timer) clearInterval(timer);
      const ms = Number(document.getElementById("speedSel").value);
      timer = setInterval(() => {
        if (!playing) return;
        const pts = series();
        if (cursor < pts.length) {
          const pt = pts[cursor];
          cursor++;
          checkEvents(pt, cursor - 1);
          drawFrame();
        } else {
          playing = false;
          document.getElementById("playBtn").textContent = "Play Stream";
        }
      }, ms);
    }

    function checkEvents(pt, idx) {
      const feed = document.getElementById("eventFeed");
      const drifts = pt.drift_features || [];
      if (drifts.length > 0) {
        pulses.push({ idx, color: "#c084fc", born: performance.now() });
        const div = document.createElement("div");
        div.className = "ev-item";
        div.innerHTML = '<div class="ev-head"><span style="color:#c084fc">ADWIN Drift Cut</span>' +
          '<span>#' + pt.machine_position + '</span></div>' +
          '<div class="ev-sub">' + drifts.join(", ") + ' shifted</div>';
        feed.prepend(div);
      } else if (pt.static_alert) {
        pulses.push({ idx, color: "#f59e0b", born: performance.now() });
        const div = document.createElement("div");
        div.className = "ev-item warn";
        div.innerHTML = '<div class="ev-head"><span style="color:#f59e0b">Threshold Alert</span>' +
          '<span>#' + pt.machine_position + '</span></div>' +
          '<div class="ev-sub">Temp=' + pt.temperature_c + '°C Vib=' +
          pt.vibration_mm_s + '</div>';
        feed.prepend(div);
      }
    }

    function animatePulses() {
      if (pulses.length > 0) drawFrame();
      requestAnimationFrame(animatePulses);
    }

    function drawFrame() {
      const pts = series();
      const visible = pts.slice(0, cursor);
      const cfg = SENSORS[sensor];
      const canvas = document.getElementById("streamCanvas");
      const ctx = canvas.getContext("2d");
      const W = canvas.width, H = canvas.height;
      const pad = { l: 48, r: 20, t: 22, b: 28 };
      ctx.clearRect(0, 0, W, H);

      if (!pts.length) return;
      const allVals = pts.map(p => p[sensor]);
      const yMin = Math.min(...allVals, cfg.min) - 2.5;
      const yMax = Math.max(...allVals, cfg.max) + 2.5;
      const xOf = (i) => pad.l + (i / Math.max(1, pts.length - 1)) * (W - pad.l - pad.r);
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

      const onset = pts.findIndex(p => p.drift_active);
      if (onset >= 0 && cursor > onset) {
        const x = xOf(onset);
        ctx.save();
        ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
        ctx.fillStyle = "#4ade80";
        ctx.fillText("True Drift Start", Math.min(x + 5, W - 110), pad.t + 12);
        ctx.restore();
      }

      if (visible.length > 0) {
        ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2;
        ctx.beginPath();
        visible.forEach((p, i) => {
          const x = xOf(i), y = yOf(p[sensor]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();

        let driftCnt = 0, alertCnt = 0;
        visible.forEach((p, i) => {
          const x = xOf(i), y = yOf(p[sensor]);
          const v = p[sensor];
          const isDrift = (p.drift_features || []).includes(sensor);
          const isAlert = v < cfg.min || v > cfg.max;
          if ((p.drift_features || []).length > 0) driftCnt++;
          if (p.static_alert) alertCnt++;

          if (isDrift) {
            ctx.strokeStyle = "#c084fc"; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
            ctx.fillStyle = "#c084fc";
            ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
          } else if (isAlert) {
            ctx.fillStyle = "#f59e0b";
            ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
          } else if (i === visible.length - 1) {
            ctx.fillStyle = "#38bdf8";
            ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
          }
        });

        const now = performance.now();
        pulses = pulses.filter(pl => now - pl.born < 900);
        pulses.forEach(pl => {
          if (pl.idx < visible.length) {
            const age = (now - pl.born) / 900;
            const x = xOf(pl.idx), y = yOf(visible[pl.idx][sensor]);
            ctx.save();
            ctx.strokeStyle = pl.color;
            ctx.globalAlpha = 1 - age;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.arc(x, y, 6 + age * 22, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
          }
        });

        const last = visible[visible.length - 1];
        document.getElementById("liveVal").textContent = last[sensor].toFixed(2) + " " + cfg.unit;
        document.getElementById("ptsVal").textContent = visible.length + " / " + pts.length;
        document.getElementById("driftVal").textContent = String(driftCnt);
        document.getElementById("alertVal").textContent = String(alertCnt);
      }
    }

    setupUI();
  </script>
</body>
</html>
"""


def write_m3_dashboard(
    output_path: Path,
    report: Mapping[str, Any],
    scenario_streams: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds: ThresholdConfig | None = None,
) -> None:
    """Write self-contained animated `dashboard.html` for an M3 run."""
    cfg = thresholds or ThresholdConfig()
    payload: dict[str, Any] = {
        "report": report,
        "thresholds": {
            "temp_min_c": cfg.temp_min_c,
            "temp_max_c": cfg.temp_max_c,
            "vib_min_mm_s": cfg.vib_min_mm_s,
            "vib_max_mm_s": cfg.vib_max_mm_s,
            "press_min_bar": cfg.press_min_bar,
            "press_max_bar": cfg.press_max_bar,
        },
        "streams": {sc: list(points) for sc, points in scenario_streams.items()},
    }
    json_blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    rendered = _DASHBOARD_TEMPLATE.replace(
        "__RUN_ID__", html.escape(str(report.get("run_id", "m3")))
    ).replace("__DASHBOARD_JSON__", json_blob)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
