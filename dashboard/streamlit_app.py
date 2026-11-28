"""DriftGuard-Kafka M10.1 Live Industrial IoT & Multi-Model Drift Observatory."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.mqtt_bridge import VALID_DEVICE_PROFILES, publish_profile_command  # noqa: E402
from dashboard.state_store import (  # noqa: E402
    DashboardMqttSettings,
    DashboardMqttSubscriber,
    DashboardStateStore,
    load_reference_reports,
)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "static_threshold": "Static Threshold (M2)",
    "isolation_forest": "Isolation Forest (M4)",
    "half_space_trees": "Half-Space Trees (M4)",
    "lstm_autoencoder": "LSTM Autoencoder (M5)",
    "tcn_autoencoder": "TCN Autoencoder (M6)",
    "tranad": "TranAD Transformer (M7)",
    "chronos2_zero_shot": "Chronos-2 Zero-Shot (M8)",
    "timesfm3_zero_shot": "TimesFM 3.0 Zero-Shot (M8)",
}

st.set_page_config(
    page_title="DriftGuard-Kafka | Live IoT & Drift Observatory",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background-color: #0B1120; color: #F8FAFC; }
      div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 12px 16px;
      }
      .dg-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
      }
      .dg-live { background: rgba(16,185,129,0.16); color: #34D399; border: 1px solid #10B981; }
      .dg-delayed { background: rgba(245,158,11,0.16); color: #FBBF24; border: 1px solid #F59E0B; }
      .dg-offline { background: rgba(239,68,68,0.16); color: #F87171; border: 1px solid #EF4444; }
      .dg-demo { background: rgba(139,92,246,0.16); color: #C4B5FD; border: 1px solid #8B5CF6; }
      .dg-card {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_dashboard_settings() -> DashboardMqttSettings:
    """Load read-only MQTT viewer configuration from `st.secrets` or environment."""
    sec: dict[str, Any] = {}
    try:
        if "mqtt" in st.secrets:
            sec = dict(st.secrets["mqtt"])
    except Exception:
        sec = {}

    def _get(sec_key: str, env_key: str, default: str) -> str:
        if sec_key in sec and sec[sec_key] is not None:
            return str(sec[sec_key]).strip()
        return os.environ.get(env_key, default).strip()

    use_tls_val = (
        sec.get("use_tls") if "use_tls" in sec else os.environ.get("MQTT_USE_TLS", "false")
    )
    use_tls = (
        bool(use_tls_val)
        if isinstance(use_tls_val, bool)
        else str(use_tls_val).strip().lower() in ("1", "true", "yes", "on")
    )
    allow_ctrl_val = sec.get(
        "allow_profile_control", os.environ.get("ALLOW_PROFILE_CONTROL", "false")
    )
    allow_ctrl = (
        bool(allow_ctrl_val)
        if isinstance(allow_ctrl_val, bool)
        else str(allow_ctrl_val).strip().lower() in ("1", "true", "yes", "on")
    )
    return DashboardMqttSettings(
        broker_host=_get("broker_host", "MQTT_BROKER_HOST", "localhost") or "localhost",
        broker_port=int(_get("broker_port", "MQTT_BROKER_PORT", "1883") or "1883"),
        use_tls=use_tls,
        username=_get("username", "MQTT_DASHBOARD_USERNAME", "")
        or _get("username", "MQTT_USERNAME", "")
        or None,
        password=_get("password", "MQTT_DASHBOARD_PASSWORD", "")
        or _get("password", "MQTT_PASSWORD", "")
        or None,
        namespace=_get("namespace", "MQTT_NAMESPACE", "driftguard/demo") or "driftguard/demo",
        client_id=_get("client_id", "MQTT_CLIENT_ID", "driftguard-streamlit-viewer"),
        qos=int(_get("qos", "MQTT_QOS", "1") or "1"),
        allow_profile_control=allow_ctrl,
        command_token=_get("command_token", "MQTT_COMMAND_TOKEN", "") or None,
    )


@st.cache_resource
def _get_store_and_subscriber() -> tuple[
    DashboardStateStore, DashboardMqttSubscriber, DashboardMqttSettings
]:
    settings = _read_dashboard_settings()
    store = DashboardStateStore()
    sub = DashboardMqttSubscriber(settings, store)
    sub.start()
    return store, sub, settings


def _badge_html(state: str) -> str:
    cls_map = {
        "LIVE": "dg-live",
        "ONLINE": "dg-live",
        "DELAYED": "dg-delayed",
        "OFFLINE": "dg-offline",
        "DEMO": "dg-demo",
    }
    cls = cls_map.get(state, "dg-offline")
    return f"<span class='dg-badge {cls}'>{state}</span>"


def _render_diagnostics(store: DashboardStateStore) -> None:
    for b in store.diagnostic_banners():
        msg = f"**{b['title']}** — {b['detail']}"
        if b["level"] == "error":
            st.error(msg)
        elif b["level"] == "warning":
            st.warning(msg)
        else:
            st.info(msg)


def _render_live_overview(
    view: dict[str, Any], freshness: dict[str, Any], store: DashboardStateStore
) -> None:
    st.markdown(
        f"### System Health & Pipeline Telemetry {_badge_html(freshness['state'])}",
        unsafe_allow_html=True,
    )
    st.caption(f"Freshness status: {freshness['reason']}")
    _render_diagnostics(store)

    telemetry = view["telemetry"]
    scores = view["model_scores"]
    alerts = view["alerts"]
    drifts = view["drifts"]
    devices = view["devices"]
    health = view["backend_health"]
    hw = health.get("hardware") or (scores[-1].get("hardware") if scores else {}) or {}

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Ingested Telemetry", len(telemetry))
    c2.metric("Active Alerts", len(alerts))
    c3.metric("ADWIN Drifts", len(drifts))
    online_cnt = sum(1 for d in devices.values() if d.get("liveness_state") in ("ONLINE", "LIVE"))
    c4.metric("IoT Devices Online", f"{online_cnt}/{len(devices)}")
    dl_dev = str(hw.get("selected_dl_device", "unavailable")).upper()
    c5.metric("DL Accelerator", dl_dev)
    c6.metric("MQTT Reconnects", view["reconnect_count"])

    st.divider()
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("Latest Multi-Model Ensemble Snapshot")
        if scores:
            latest = scores[-1]
            rows = []
            for mid, info in latest.get("models", {}).items():
                rows.append(
                    {
                        "Model": MODEL_DISPLAY_NAMES.get(mid, mid),
                        "Status": info.get("status"),
                        "Device": info.get("device"),
                        "Mode": info.get("channel_mode"),
                        "Score": info.get("score"),
                        "Threshold": info.get("threshold"),
                        "Ratio": info.get("normalized_ratio"),
                        "Flagged": bool(info.get("anomaly")),
                        "Latency (ms)": info.get("latency_ms"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No `research.telemetry.model_scores` events ingested yet.")

    with col_r:
        st.subheader("Mac Backend & Memory Footprint")
        mem = health.get("memory_mib") or (scores[-1].get("memory_mib") if scores else {}) or {}
        st.json(
            {
                "mqtt_viewer_state": view["mqtt_connection_state"],
                "backend_last_kafka_utc": health.get("last_kafka_event_utc"),
                "topic_counts": health.get("topic_counts", {}),
                "hardware": {
                    "os": hw.get("os_platform"),
                    "arch": hw.get("machine_arch"),
                    "mps_built": hw.get("mps_built"),
                    "mps_available": hw.get("mps_available"),
                    "selected_dl_device": hw.get("selected_dl_device"),
                },
                "memory_mib": mem,
            }
        )


def _render_sensor_monitoring(view: dict[str, Any], selected_device: str) -> None:
    st.subheader("Live Virtual ESP32 Sensor Telemetry (3-Channel Stream)")
    st.caption(
        "Simulated virtual ESP32 sensor readings (temperature_c, vibration_mm_s, pressure_bar); "
        "not physical industrial transducers."
    )
    rows = [
        r
        for r in view["telemetry"]
        if selected_device == "ALL" or r.get("device_id") == selected_device
    ]
    if not rows:
        st.warning("No telemetry events available for the selected filter.")
        return

    df = pd.DataFrame(rows)
    df["ts_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("ts_dt")

    win = st.slider("Rolling Window (events)", min_value=20, max_value=400, value=min(150, len(df)))
    df_win = df.tail(win)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Temperature (°C)", "Vibration RMS (mm/s)", "Pressure (bar)"),
    )
    fig.add_trace(
        go.Scatter(
            x=df_win["ts_dt"], y=df_win["temperature_c"], name="temp_c", line={"color": "#38BDF8"}
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=80.0, line_dash="dot", line_color="#EF4444", row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=df_win["ts_dt"],
            y=df_win["vibration_mm_s"],
            name="vib_mm_s",
            line={"color": "#34D399"},
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=10.0, line_dash="dot", line_color="#EF4444", row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=df_win["ts_dt"], y=df_win["pressure_bar"], name="press_bar", line={"color": "#FBBF24"}
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#111827",
        height=620,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_model_monitoring(view: dict[str, Any]) -> None:
    st.subheader("ML / DL / TSFM Model Monitoring (All 9 Integrated Methods)")
    scores = view["model_scores"]
    if not scores:
        st.info("No live `model_scores` events recorded yet.")
        return

    trend_rows = []
    for ev in scores[-120:]:
        ts = ev.get("timestamp")
        for mid, mdata in ev.get("models", {}).items():
            trend_rows.append(
                {
                    "timestamp": ts,
                    "model_id": mid,
                    "model_label": MODEL_DISPLAY_NAMES.get(mid, mid),
                    "normalized_ratio": mdata.get("normalized_ratio", 0.0),
                    "score": mdata.get("score", 0.0),
                    "threshold": mdata.get("threshold", 1.0),
                    "latency_ms": mdata.get("latency_ms", 0.0),
                    "device": mdata.get("device", "unknown"),
                    "status": mdata.get("status", "unknown"),
                }
            )
    tdf = pd.DataFrame(trend_rows)
    if tdf.empty:
        st.info("Model score payload contained no model entries.")
        return
    tdf["ts_dt"] = pd.to_datetime(tdf["timestamp"], errors="coerce")

    fig = go.Figure()
    for mid, grp in tdf.groupby("model_id"):
        fig.add_trace(
            go.Scatter(
                x=grp["ts_dt"],
                y=grp["normalized_ratio"],
                mode="lines",
                name=MODEL_DISPLAY_NAMES.get(str(mid), str(mid)),
            )
        )
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Calibrated Threshold Ratio = 1.0",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#111827",
        height=420,
        yaxis_title="Score / Threshold Ratio (1.0 = Alert Boundary)",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    st.plotly_chart(fig, use_container_width=True)

    summary = (
        tdf.groupby("model_label", as_index=False)
        .agg(
            latest_ratio=("normalized_ratio", "last"),
            mean_latency_ms=("latency_ms", "mean"),
            p95_latency_ms=("latency_ms", lambda s: float(s.quantile(0.95))),
            device=("device", "last"),
            status=("status", "last"),
        )
        .round(3)
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


def _render_drift_monitoring(view: dict[str, Any]) -> None:
    st.subheader("River ADWIN Concept Drift Monitoring")
    drifts = view["drifts"]
    if not drifts:
        st.success("Zero ADWIN concept drift transitions recorded in the current buffer.")
        return

    ddf = pd.DataFrame(drifts)
    ts_series = ddf["event_timestamp"] if "event_timestamp" in ddf.columns else ddf.get("timestamp")
    ddf["ts_dt"] = pd.to_datetime(ts_series, errors="coerce")
    y_col = "adwin_estimation" if "adwin_estimation" in ddf.columns else "magnitude"
    if y_col not in ddf.columns:
        ddf[y_col] = 0.0
    fig = go.Figure(
        data=[
            go.Scatter(
                x=ddf["ts_dt"],
                y=ddf[y_col],
                mode="markers+text",
                text=ddf.get("feature", ""),
                textposition="top center",
                marker={"size": 12, "color": "#F59E0B"},
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#111827",
        height=340,
        yaxis_title=f"ADWIN {y_col}",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ddf.drop(columns=["ts_dt"]), use_container_width=True, hide_index=True)


def _render_alerts_tab(view: dict[str, Any]) -> None:
    st.subheader("Anomaly Alerts & Pipeline Incidents")
    alerts = view["alerts"]
    if not alerts:
        st.info("No anomaly alerts emitted on `research.telemetry.alerts` in the current window.")
        return
    adf = pd.DataFrame(alerts)
    filter_col = (
        "severity" if "severity" in adf.columns else ("reason" if "reason" in adf.columns else None)
    )
    if filter_col is not None:
        opts = ["ALL"] + sorted({str(x) for x in adf[filter_col].dropna()})
        pick = st.selectbox(f"Filter by {filter_col}", opts)
        if pick != "ALL":
            adf = adf[adf[filter_col].astype(str) == pick]
    st.dataframe(adf.iloc[::-1], use_container_width=True, hide_index=True)
    st.download_button(
        "Export Alerts CSV",
        data=adf.to_csv(index=False).encode("utf-8"),
        file_name="driftguard_live_alerts.csv",
        mime="text/csv",
    )


def _render_device_management(view: dict[str, Any], settings: DashboardMqttSettings) -> None:
    st.subheader("Virtual ESP32 Device Health & Remote Profile Control")
    devices = view["devices"]
    if devices:
        rows = []
        for dev in devices.values():
            lat = dev.get("ingestion_latency_ms") or {}
            rows.append(
                {
                    "Device ID": dev.get("device_id"),
                    "Machine": dev.get("machine_id"),
                    "State": dev.get("liveness_state"),
                    "Active Profile": dev.get("active_profile_reported"),
                    "Messages": dev.get("messages_received"),
                    "Heartbeats": dev.get("heartbeat_count"),
                    "Seq Gaps": dev.get("sequence_gaps"),
                    "Reconnects": dev.get("device_reconnect_count"),
                    "Mean Lat (ms)": lat.get("mean_ms"),
                    "Last Message UTC": dev.get("last_message_timestamp"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No device status reports received yet.")

    st.divider()
    st.markdown("#### Remote Wokwi Profile Switching (`set_profile`)")
    if not settings.allow_profile_control or not settings.command_token:
        st.info(
            "Read-only dashboard mode is active. Set `allow_profile_control = true` and a valid "
            "`command_token` under `[mqtt]` in Streamlit secrets to unlock authenticated commands."
        )
        return

    dev_ids = sorted(devices.keys()) or ["esp32-wokwi-01"]
    col1, col2, col3 = st.columns([2, 2, 1])
    target_dev = col1.selectbox("Target ESP32 Device", dev_ids)
    target_profile = col2.selectbox("Target Profile", sorted(VALID_DEVICE_PROFILES))
    if col3.button("Send Command", type="primary"):
        try:
            topic = publish_profile_command(
                broker_host=settings.broker_host,
                broker_port=settings.broker_port,
                namespace=settings.namespace,
                device_id=target_dev,
                profile=target_profile,
                command_token=settings.command_token,
                use_tls=settings.use_tls,
                username=settings.username,
                password=settings.password,
                qos=settings.qos,
            )
            st.success(f"Published authenticated `set_profile={target_profile}` to `{topic}`")
        except Exception as exc:
            st.error(f"Command failed: {exc}")


def _render_research_results(repo_root: Path) -> None:
    st.subheader("Research & Benchmark Results (Strictly Separated Scopes)")
    st.info(
        "**Methodology Separation Notice**: Offline M9.1 results evaluate SKAB (8 channels) and "
        "TSB-AD-M (2–248 channels) benchmarks, whereas Live M10.1 evaluates the 3-channel Wokwi "
        "virtual IoT Kafka stream (`temperature_c`, `vibration_mm_s`, `pressure_bar`)."
    )
    reports = load_reference_reports(repo_root)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### M10.1 Online 15-Minute Validation Summary")
        if reports["m10_1"]:
            st.caption(f"Source: `{reports.get('m10_1_path')}`")
            st.json(reports["m10_1"].get("scenario_summary", reports["m10_1"]))
        else:
            st.warning(
                "No local M10.1 report artefact found (`reports/m10_1/*/summary_report.json`)."
            )

    with col_b:
        st.markdown("#### M9.1 Offline Multi-Model Reference Summary")
        if reports["m9_1"]:
            st.caption(f"Source: `{reports.get('m9_1_path')}`")
            st.json(
                {
                    "scope": "Offline SKAB & TSB-AD-M (M9.1)",
                    "run_id": reports["m9_1"].get("run_id"),
                    "datasets": reports["m9_1"].get("dataset_summary", {}),
                }
            )
        else:
            st.info("Offline M9.1 report artifacts are stored locally and excluded from Git.")


def main() -> None:
    """Render the DriftGuard-Kafka M10.1 Streamlit dashboard."""
    store, _sub, settings = _get_store_and_subscriber()

    with st.sidebar:
        st.title("🛡️ DriftGuard-Kafka")
        st.caption("M10.1 Online Virtual IoT & Multi-Model Observatory")
        fresh = store.compute_freshness()
        st.markdown(f"Status: {_badge_html(fresh['state'])}", unsafe_allow_html=True)
        st.caption(
            f"Broker: `{settings.broker_host}:{settings.broker_port}` (`TLS={settings.use_tls}`)"
        )
        st.caption(f"Topic: `{settings.subscription_topic}`")

        auto_refresh = st.toggle("Live Auto-Refresh (2s)", value=True)
        demo_toggle = st.toggle("Static Offline Demo Sample", value=store.demo_mode)
        if demo_toggle and not store.demo_mode:
            store.load_static_demo_sample()
        elif not demo_toggle and store.demo_mode:
            store.clear_for_live_mode()

        view_sb = store.snapshot_view()
        device_opts = ["ALL"] + sorted(view_sb["devices"].keys())
        selected_device = st.selectbox("Device Filter", device_opts)

    tabs = st.tabs(
        [
            "Live Overview",
            "Live Sensor Monitoring",
            "ML / DL Model Monitoring",
            "Drift Monitoring",
            "Alerts & Incidents",
            "Device Management",
            "Experiment / Research Results",
        ]
    )

    @st.fragment(run_every=2 if auto_refresh else None)
    def _live_panel() -> None:
        view = store.snapshot_view()
        freshness = store.compute_freshness()
        with tabs[0]:
            _render_live_overview(view, freshness, store)
        with tabs[1]:
            _render_sensor_monitoring(view, selected_device)
        with tabs[2]:
            _render_model_monitoring(view)
        with tabs[3]:
            _render_drift_monitoring(view)
        with tabs[4]:
            _render_alerts_tab(view)
        with tabs[5]:
            _render_device_management(view, settings)

    _live_panel()
    with tabs[6]:
        _render_research_results(REPO_ROOT)


if __name__ == "__main__":
    main()
