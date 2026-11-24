"""Interactive HTML report dashboard for DriftGuard-Kafka M3 streaming runs.

Generates a self-contained `dashboard.html` inside `reports/m3/<run_id>/` that
plots actual Kafka-consumed telemetry points (`temperature_c`, `vibration_mm_s`,
`pressure_bar`) per scenario and machine, overlaid with:
  - Static M2 anomaly threshold bands (unchanged in M3)
  - Ground-truth drift onset boundary (for verification only)
  - River ADWIN concept drift detection markers
  - Interactive streaming point inspector table
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
  <title>DriftGuard-Kafka M3 Streaming Dashboard — __RUN_ID__</title>
  <style>
    :root {
      --bg: #0f172a; --card: #1e293b; --border: #334155;
      --fg: #f8fafc; --muted: #94a3b8; --accent: #38bdf8;
      --ok: #22c55e; --warn: #f59e0b; --err: #ef4444; --drift: #a855f7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 20px; background: var(--bg); color: var(--fg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px; line-height: 1.45;
    }
    .wrap { max-width: 1240px; margin: 0 auto; display: grid; gap: 16px; }
    .card {
      background: var(--card); border: 1px solid var(--border);
      border-radius: 10px; padding: 16px;
    }
    h1 { margin: 0 0 4px; font-size: 20px; }
    h2 { margin: 0 0 12px; font-size: 15px; color: var(--fg); }
    .sub { color: var(--muted); font-size: 12px; font-family: ui-monospace, monospace; }
    .kpis {
      display: grid; gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    }
    .kpi {
      background: rgba(15,23,42,0.55); border: 1px solid var(--border);
      border-radius: 8px; padding: 12px;
    }
    .kpi .lbl { color: var(--muted); font-size: 12px; }
    .kpi .val {
      font-size: 22px; font-weight: 700; margin-top: 4px;
      font-variant-numeric: tabular-nums;
    }
    .controls {
      display: flex; flex-wrap: wrap; gap: 12px;
      align-items: center; margin-bottom: 12px;
    }
    select, button {
      background: var(--bg); color: var(--fg); border: 1px solid var(--border);
      border-radius: 6px; padding: 6px 10px; font-size: 13px; cursor: pointer;
    }
    button.active { border-color: var(--accent); color: var(--accent); font-weight: 600; }
    canvas {
      width: 100%; height: 320px; display: block;
      background: rgba(15,23,42,0.45); border-radius: 8px;
    }
    .legend {
      display: flex; flex-wrap: wrap; gap: 16px;
      margin-top: 10px; font-size: 12px; color: var(--muted);
    }
    .dot {
      display: inline-block; width: 10px; height: 10px;
      border-radius: 50%; margin-right: 5px; vertical-align: middle;
    }
    table {
      width: 100%; border-collapse: collapse;
      font-size: 12px; font-variant-numeric: tabular-nums;
    }
    th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); font-weight: 600; }
    .badge { padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .b-ok { background: rgba(34,197,94,0.18); color: #4ade80; }
    .b-drift { background: rgba(168,85,247,0.22); color: #c084fc; }
    .b-warn { background: rgba(245,158,11,0.2); color: #fbbf24; }
    .scroll { max-height: 260px; overflow-y: auto; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>DriftGuard-Kafka M3 — ADWIN Streaming Telemetry &amp; Drift Dashboard</h1>
      <div class="sub" id="meta"></div>
    </div>

    <div class="kpis" id="kpis"></div>

    <div class="card">
      <div class="controls">
        <label>Scenario: <select id="scenarioSelect"></select></label>
        <label>Machine: <select id="machineSelect"></select></label>
        <div id="featureBtns"></div>
        <span class="sub" id="hoverInfo" style="margin-left:auto">
          Hover chart to inspect point
        </span>
      </div>
      <canvas id="streamCanvas" width="1180" height="320"></canvas>
      <div class="legend">
        <span><span class="dot" style="background:#38bdf8"></span>Streaming point</span>
        <span><span class="dot" style="background:#f59e0b"></span>Static M2 alert</span>
        <span><span class="dot" style="background:#ef4444"></span>Static bounds</span>
        <span><span class="dot" style="background:#22c55e"></span>True drift onset</span>
        <span><span class="dot" style="background:#a855f7"></span>ADWIN drift detected</span>
      </div>
    </div>

    <div class="card">
      <h2>Scenario Summary &amp; Channel Drift Metrics</h2>
      <div class="scroll"><table id="scenarioTable"></table></div>
    </div>

    <div class="card">
      <h2>Streaming Points &amp; ADWIN Events Inspector (Selected Scenario &amp; Machine)</h2>
      <div class="scroll"><table id="pointsTable"></table></div>
    </div>
  </div>

  <script>
    const DATA = __DASHBOARD_JSON__;
    const TH = DATA.thresholds;
    const FEATURES = [
      { key: "temperature_c", label: "Temperature (°C)", min: TH.temp_min_c, max: TH.temp_max_c },
      {
        key: "vibration_mm_s", label: "Vibration (mm/s)",
        min: TH.vib_min_mm_s, max: TH.vib_max_mm_s
      },
      { key: "pressure_bar", label: "Pressure (bar)", min: TH.press_min_bar, max: TH.press_max_bar }
    ];
    const defaultSc = DATA.report.scenarios[1]
      ? DATA.report.scenarios[1].scenario
      : DATA.report.scenarios[0].scenario;
    let state = { scenario: defaultSc, machine: "machine_01", feature: "temperature_c" };

    function fmt(v, s = "") { return v === null || v === undefined ? "unavailable" : v + s; }

    function init() {
      const r = DATA.report;
      document.getElementById("meta").textContent =
        "Run ID: " + r.run_id + " | Generated: " + r.generated_utc +
        " | ADWIN delta: " + r.adwin_delta + " | Seed: " + fmt(r.reproducibility.seed);

      const ov = r.overall_metrics;
      const pct = ov.detection_rate !== null ? (ov.detection_rate * 100).toFixed(1) : null;
      const kpis = [
        ["Expected Drift Channels", ov.expected_drift_channels],
        ["Detected Drift Channels", ov.detected_drift_channels + " (" + fmt(pct, "%") + ")"],
        ["Missed Drift Channels", ov.missed_drift_channels],
        ["Stable False Alarms", ov.false_alarms_stable],
        ["Mean Delay (Samples)", fmt(ov.mean_delay_machine_samples, " samples")],
        [
          "Kafka Drift Verified",
          ov.total_drift_events_verified + " / " + ov.total_drift_events_triggered
        ]
      ];
      document.getElementById("kpis").innerHTML = kpis.map(([l, v]) =>
        '<div class="kpi"><div class="lbl">' + l + '</div><div class="val">' + v + '</div></div>'
      ).join("");

      const scSel = document.getElementById("scenarioSelect");
      scSel.innerHTML = r.scenarios.map(
        s => '<option value="' + s.scenario + '">' + s.scenario + '</option>'
      ).join("");
      scSel.value = state.scenario;
      scSel.onchange = () => { state.scenario = scSel.value; syncMachines(); render(); };

      const fBtns = document.getElementById("featureBtns");
      fBtns.innerHTML = FEATURES.map(f => {
        const cls = f.key === state.feature ? "active" : "";
        return '<button data-f="' + f.key + '" class="' + cls + '">' + f.label + '</button>';
      }).join(" ");
      fBtns.querySelectorAll("button").forEach(btn => {
        btn.onclick = () => {
          state.feature = btn.dataset.f;
          fBtns.querySelectorAll("button").forEach(
            b => b.classList.toggle("active", b.dataset.f === state.feature)
          );
          render();
        };
      });

      renderScenarioTable();
      syncMachines();
      render();
    }

    function syncMachines() {
      const pts = DATA.streams[state.scenario] || [];
      const machines = [...new Set(pts.map(p => p.machine_id))].sort();
      const mSel = document.getElementById("machineSelect");
      if (!machines.includes(state.machine) && machines.length) state.machine = machines[0];
      mSel.innerHTML = machines.map(m => '<option value="' + m + '">' + m + '</option>').join("");
      mSel.value = state.machine;
      mSel.onchange = () => { state.machine = mSel.value; render(); };
    }

    function renderScenarioTable() {
      const rows = DATA.report.scenarios.map(s => {
        const dm = s.drift_metrics;
        const dly = s.delay_machine_samples;
        const dt = s.drift_topic;
        const dtStr = dt.triggered + " / " + dt.delivered + " / " + fmt(dt.verified_on_topic);
        return "<tr>" +
          "<td><strong>" + s.scenario + "</strong></td>" +
          "<td>" + s.events.received + "</td>" +
          "<td>" + dm.expected_drift_channels + "</td>" +
          "<td>" + dm.detected_drift_channels + "</td>" +
          "<td>" + dm.missed_drift_channels + "</td>" +
          "<td>" + fmt(dm.detection_rate) + "</td>" +
          "<td>" + dm.false_alarms_stable + "</td>" +
          "<td>" + fmt(dly.mean) + " / " + fmt(dly.p50) + " / " + fmt(dly.p95) + "</td>" +
          "<td>" + s.events.static_alerts_observed + "</td>" +
          "<td>" + dtStr + "</td></tr>";
      }).join("");
      document.getElementById("scenarioTable").innerHTML = "<thead><tr>" +
        "<th>Scenario</th><th>Events</th><th>Expected Ch</th>" +
        "<th>Detected Ch</th><th>Missed Ch</th>" +
        "<th>Detection Rate</th><th>Stable False Alarms</th><th>Delay Mean/p50/p95</th>" +
        "<th>Unchanged Static Alerts</th><th>Kafka Drift (Trig/Del/Ver)</th>" +
        "</tr></thead><tbody>" + rows + "</tbody>";
    }

    let currentPlotPoints = [];

    function render() {
      const rawPts = DATA.streams[state.scenario] || [];
      const allPts = rawPts.filter(p => p.machine_id === state.machine);
      const featCfg = FEATURES.find(f => f.key === state.feature);
      const canvas = document.getElementById("streamCanvas");
      const ctx = canvas.getContext("2d");
      const W = canvas.width, H = canvas.height;
      const pad = { l: 52, r: 20, t: 20, b: 32 };
      ctx.clearRect(0, 0, W, H);

      if (!allPts.length) return;
      const vals = allPts.map(p => p[state.feature]);
      const yMin = Math.min(...vals, featCfg.min) - 2;
      const yMax = Math.max(...vals, featCfg.max) + 2;
      const xOf = (i) => pad.l + (i / Math.max(1, allPts.length - 1)) * (W - pad.l - pad.r);
      const span = Math.max(1e-6, yMax - yMin);
      const yOf = (v) => H - pad.b - ((v - yMin) / span) * (H - pad.t - pad.b);

      ctx.strokeStyle = "#334155"; ctx.lineWidth = 1;
      ctx.fillStyle = "#94a3b8"; ctx.font = "11px monospace";
      for (let g = 0; g <= 4; g++) {
        const v = yMin + (g / 4) * (yMax - yMin);
        const y = yOf(v);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
        ctx.fillText(v.toFixed(1), 6, y + 4);
      }

      ctx.save();
      ctx.strokeStyle = "#ef4444"; ctx.setLineDash([5, 4]);
      [featCfg.min, featCfg.max].forEach(th => {
        const y = yOf(th);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      });
      ctx.restore();

      const onsetIdx = allPts.findIndex(p => p.drift_active);
      if (onsetIdx >= 0) {
        const x = xOf(onsetIdx);
        ctx.save();
        ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
        ctx.fillStyle = "#4ade80";
        ctx.fillText("onset #" + onsetIdx, Math.min(x + 4, W - 90), pad.t + 12);
        ctx.restore();
      }

      ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 1.6;
      ctx.beginPath();
      allPts.forEach((p, i) => {
        const x = xOf(i), y = yOf(p[state.feature]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      currentPlotPoints = [];
      allPts.forEach((p, i) => {
        const x = xOf(i), y = yOf(p[state.feature]);
        const v = p[state.feature];
        const breach = v < featCfg.min || v > featCfg.max;
        const drifted = (p.drift_features || []).includes(state.feature);
        currentPlotPoints.push({ x, y, p, i });

        if (drifted) {
          ctx.strokeStyle = "#a855f7"; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
          ctx.fillStyle = "#c084fc";
          ctx.beginPath(); ctx.arc(x, y, 5.5, 0, Math.PI * 2); ctx.fill();
        } else {
          ctx.fillStyle = breach ? "#f59e0b" : "#38bdf8";
          ctx.beginPath(); ctx.arc(x, y, breach ? 3.5 : 2.3, 0, Math.PI * 2); ctx.fill();
        }
      });

      canvas.onmousemove = (ev) => {
        const rect = canvas.getBoundingClientRect();
        const mx = (ev.clientX - rect.left) * (W / rect.width);
        let best = null, bestDist = 1e9;
        for (const item of currentPlotPoints) {
          const d = Math.abs(item.x - mx);
          if (d < bestDist) { bestDist = d; best = item; }
        }
        if (best) {
          const p = best.p;
          const adw = (p.drift_features || []).join(",") || "none";
          document.getElementById("hoverInfo").textContent =
            "machine_pos=" + p.machine_position + " | stream_pos=" + p.stream_position +
            " | " + state.feature + "=" + p[state.feature] +
            " | gt=" + p.ground_truth_label + " | drift=" + p.drift_active + " | adwin=" + adw;
        }
      };

      const interesting = allPts.filter((p, idx) =>
        (p.drift_features && p.drift_features.length > 0) || idx % 5 === 0
      );
      const bodyRows = interesting.map(p => {
        const reg = p.drift_active
          ? '<span class="badge b-warn">drifted normal</span>'
          : "baseline normal";
        const al = p.static_alert ? '<span class="badge b-warn">alert (FP)</span>' : "—";
        const df = (p.drift_features || []).length
          ? '<span class="badge b-drift">' + p.drift_features.join(", ") + "</span>"
          : "—";
        return "<tr><td>" + p.machine_position + "</td><td>" + p.stream_position + "</td>" +
          "<td>" + p.temperature_c + "</td><td>" + p.vibration_mm_s + "</td>" +
          "<td>" + p.pressure_bar + "</td>" +
          '<td><span class="badge b-ok">' + p.ground_truth_label + "</span></td>" +
          "<td>" + reg + "</td><td>" + al + "</td><td>" + df + "</td></tr>";
      }).join("");
      document.getElementById("pointsTable").innerHTML = "<thead><tr>" +
        "<th>Machine Pos</th><th>Stream Pos</th><th>Temp (°C)</th>" +
        "<th>Vib (mm/s)</th><th>Press (bar)</th>" +
        "<th>GT Label</th><th>Drift Regime</th><th>Static Alert</th><th>ADWIN Triggered</th>" +
        "</tr></thead><tbody>" + bodyRows + "</tbody>";
    }

    init();
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
    """Write self-contained interactive `dashboard.html` for an M3 run."""
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
