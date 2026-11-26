"""Zero-dependency pure-SVG performance chart generator for DriftGuard-Kafka M9.

Generates crisp, publication-grade vector charts without requiring matplotlib:
1. `skab_multivariate_metrics.svg` — Grouped bar chart (Micro F1, Macro F1, PR-AUC, VUS-PR)
   with ±1 SD error bars for trained models on SKAB multivariate fault experiments.
2. `tsb_univariate_metrics.svg` — Grouped bar chart (Macro F1, PR-AUC, VUS-PR)
   with ±1 SD error bars on the TSB-AD-M 9-domain common univariate suite.
3. `f1_vs_latency_pareto.svg` — Log-latency vs. F1 scatter plot highlighting
   streaming SLA thresholds and high-F1 / high-latency models.
4. `resource_efficiency.svg` — Throughput (log ev/s) and Peak RSS (MiB) side-by-side
   comparison across all 10 detector variants.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MODEL_FAMILY_COLORS: dict[str, str] = {
    "static_threshold": "#64748b",
    "isolation_forest": "#475569",
    "half_space_trees": "#94a3b8",
    "lstm_autoencoder": "#2563eb",
    "tcn_autoencoder": "#0891b2",
    "tranad": "#7c3aed",
    "chronos2_zero_shot": "#d97706",
    "chronos2_interval": "#f59e0b",
    "timesfm3_zero_shot": "#dc2626",
    "timesfm3_interval": "#f87171",
}

METRIC_COLORS: dict[str, str] = {
    "micro_f1": "#2563eb",
    "macro_f1": "#0891b2",
    "mean_pr_auc": "#7c3aed",
    "mean_vus_pr": "#059669",
}


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _metric_mean(row: Mapping[str, Any], key: str) -> float | None:
    val = row.get(f"{key}_mean", row.get(key))
    if val is None:
        return None
    return float(val)


def _metric_std(row: Mapping[str, Any], key: str) -> float:
    val = row.get(f"{key}_std")
    return 0.0 if val is None else float(val)


def render_grouped_metric_bar_svg(
    title: str,
    subtitle: str,
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[tuple[str, str]],
) -> str:
    """Render a horizontal grouped bar chart with ±1 SD error bars as standalone SVG."""
    n_models = max(1, len(rows))
    n_metrics = max(1, len(metrics))
    bar_h = 11
    bar_gap = 3
    group_h = n_metrics * (bar_h + bar_gap) + 14
    margin_left = 170
    margin_right = 70
    margin_top = 82
    margin_bottom = 44
    plot_w = 640
    plot_h = n_models * group_h
    width = margin_left + plot_w + margin_right
    height = margin_top + plot_h + margin_bottom

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="28" font-size="16" font-weight="700" '
        f'fill="#0f172a">{_esc(title)}</text>',
        f'<text x="{margin_left}" y="46" font-size="11.5" fill="#475569">{_esc(subtitle)}</text>',
    ]

    # Legend
    lx = margin_left
    ly = 66
    for m_key, m_label in metrics:
        color = METRIC_COLORS.get(m_key, "#334155")
        lines.append(f'<rect x="{lx}" y="{ly - 9}" width="11" height="11" rx="2" fill="{color}"/>')
        lines.append(
            f'<text x="{lx + 15}" y="{ly}" font-size="11" fill="#1e293b">{_esc(m_label)}</text>'
        )
        lx += 120

    # Grid lines 0.0 -> 1.0
    for tick_idx in range(6):
        val = tick_idx * 0.2
        x = margin_left + val * plot_w
        lines.append(
            f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{margin_top + plot_h + 18}" text-anchor="middle" '
            f'font-size="10.5" fill="#64748b">{val:.1f}</text>'
        )

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{margin_top + plot_h}" stroke="#94a3b8" stroke-width="1.2"/>'
    )

    for idx, row in enumerate(rows):
        y0 = margin_top + idx * group_h + 6
        model_label = str(row["model"])
        mid_y = y0 + (n_metrics * (bar_h + bar_gap)) / 2.0 + 3
        lines.append(
            f'<text x="{margin_left - 10}" y="{mid_y:.1f}" text-anchor="end" '
            f'font-size="11" font-weight="600" fill="#1e293b">{_esc(model_label)}</text>'
        )
        if idx > 0:
            sep_y = margin_top + idx * group_h
            lines.append(
                f'<line x1="{margin_left}" y1="{sep_y:.1f}" x2="{margin_left + plot_w}" '
                f'y2="{sep_y:.1f}" stroke="#f1f5f9" stroke-width="1"/>'
            )

        for m_i, (m_key, _m_label) in enumerate(metrics):
            by = y0 + m_i * (bar_h + bar_gap)
            mean_v = _metric_mean(row, m_key)
            std_v = _metric_std(row, m_key)
            if mean_v is None:
                continue
            clamped = max(0.0, min(1.0, mean_v))
            bw = clamped * plot_w
            color = METRIC_COLORS.get(m_key, "#334155")
            lines.append(
                f'<rect x="{margin_left}" y="{by:.1f}" width="{bw:.1f}" height="{bar_h}" '
                f'rx="2" fill="{color}" opacity="0.88"/>'
            )
            if std_v > 1e-6:
                lo_x = margin_left + max(0.0, clamped - std_v) * plot_w
                hi_x = margin_left + min(1.0, clamped + std_v) * plot_w
                cy = by + bar_h / 2.0
                lines.append(
                    f'<line x1="{lo_x:.1f}" y1="{cy:.1f}" x2="{hi_x:.1f}" y2="{cy:.1f}" '
                    f'stroke="#0f172a" stroke-width="1.3"/>'
                )
                lines.append(
                    f'<line x1="{lo_x:.1f}" y1="{cy - 3:.1f}" x2="{lo_x:.1f}" y2="{cy + 3:.1f}" '
                    f'stroke="#0f172a" stroke-width="1.3"/>'
                )
                lines.append(
                    f'<line x1="{hi_x:.1f}" y1="{cy - 3:.1f}" x2="{hi_x:.1f}" y2="{cy + 3:.1f}" '
                    f'stroke="#0f172a" stroke-width="1.3"/>'
                )
            txt_x = margin_left + min(1.0, clamped + std_v) * plot_w + 5
            lbl = f"{mean_v:.3f}±{std_v:.3f}" if std_v > 1e-6 else f"{mean_v:.3f}"
            lines.append(
                f'<text x="{txt_x:.1f}" y="{by + bar_h - 1.5:.1f}" font-size="9.5" '
                f'fill="#334155">{_esc(lbl)}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_f1_vs_latency_svg(
    title: str,
    subtitle: str,
    skab_rows: Sequence[Mapping[str, Any]],
    tsb_rows: Sequence[Mapping[str, Any]],
    max_latency_ms: float = 0.1,
) -> str:
    """Render side-by-side log-latency vs F1 Pareto scatter plots for SKAB & TSB."""
    width = 960
    height = 460
    panel_w = 390
    panel_h = 300
    top_y = 85
    left_panels = [
        (70, "SKAB Multivariate (Micro F1 vs. Latency)", skab_rows, "micro_f1"),
        (530, "TSB-AD-M Univariate 9-Domain (Macro F1 vs. Latency)", tsb_rows, "macro_f1"),
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="70" y="30" font-size="16" font-weight="700" fill="#0f172a">{_esc(title)}</text>',
        f'<text x="70" y="49" font-size="11.5" fill="#475569">{_esc(subtitle)}</text>',
    ]

    min_log = -3.0  # 0.001 ms
    max_log = 1.0  # 10.0 ms

    def _x_map(lat_ms: float, x0: float) -> float:
        lv = math.log10(max(1e-3, min(10.0, lat_ms)))
        frac = (lv - min_log) / (max_log - min_log)
        return x0 + frac * panel_w

    for x0, p_title, p_rows, f1_key in left_panels:
        lines.append(
            f'<text x="{x0}" y="{top_y - 12}" font-size="12.5" font-weight="700" '
            f'fill="#1e293b">{_esc(p_title)}</text>'
        )
        lines.append(
            f'<rect x="{x0}" y="{top_y}" width="{panel_w}" height="{panel_h}" '
            f'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>'
        )
        # SLA latency threshold vertical line
        sla_x = _x_map(max_latency_ms, x0)
        lines.append(
            f'<rect x="{sla_x:.1f}" y="{top_y}" width="{x0 + panel_w - sla_x:.1f}" '
            f'height="{panel_h}" fill="#fef2f2" opacity="0.65"/>'
        )
        lines.append(
            f'<line x1="{sla_x:.1f}" y1="{top_y}" x2="{sla_x:.1f}" y2="{top_y + panel_h}" '
            f'stroke="#dc2626" stroke-width="1.3" stroke-dasharray="4,3"/>'
        )
        lines.append(
            f'<text x="{sla_x + 4:.1f}" y="{top_y + 14}" font-size="9.5" fill="#b91c1c">'
            f"SLA &gt; {max_latency_ms} ms</text>"
        )

        # Y grid (0.0 .. 1.0)
        for t in range(6):
            yv = t * 0.2
            py = top_y + panel_h - yv * panel_h
            lines.append(
                f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + panel_w}" y2="{py:.1f}" '
                f'stroke="#e2e8f0" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{x0 - 8}" y="{py + 3.5:.1f}" text-anchor="end" '
                f'font-size="10" fill="#64748b">{yv:.1f}</text>'
            )

        # X log ticks
        for exp in (-3, -2, -1, 0, 1):
            val = 10.0**exp
            px = _x_map(val, x0)
            lines.append(
                f'<line x1="{px:.1f}" y1="{top_y}" x2="{px:.1f}" y2="{top_y + panel_h}" '
                f'stroke="#e2e8f0" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{px:.1f}" y="{top_y + panel_h + 16}" text-anchor="middle" '
                f'font-size="10" fill="#64748b">{val:g} ms</text>'
            )
        lines.append(
            f'<text x="{x0 + panel_w / 2:.1f}" y="{top_y + panel_h + 34}" text-anchor="middle" '
            f'font-size="11" fill="#334155">Inference Latency (ms / event, log scale)</text>'
        )

        # Plot points
        placed: list[tuple[float, float]] = []
        for row in p_rows:
            f1 = _metric_mean(row, f1_key)
            lat = _metric_mean(row, "mean_latency_ms")
            if f1 is None or lat is None or lat <= 0:
                continue
            px = _x_map(lat, x0)
            py = top_y + panel_h - max(0.0, min(1.0, f1)) * panel_h
            m_name = str(row["model"])
            color = MODEL_FAMILY_COLORS.get(m_name, "#334155")
            lines.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{color}" '
                f'stroke="#ffffff" stroke-width="1.2"/>'
            )
            # Nudge overlapping labels vertically
            lbl_y = py - 8
            for ox, oy in placed:
                if abs(ox - px) < 75 and abs(oy - lbl_y) < 12:
                    lbl_y += 14
            placed.append((px, lbl_y))
            anchor = "end" if px > x0 + panel_w - 85 else "start"
            dx = -7 if anchor == "end" else 7
            lines.append(
                f'<text x="{px + dx:.1f}" y="{lbl_y:.1f}" text-anchor="{anchor}" '
                f'font-size="9.5" font-weight="600" fill="#0f172a">{_esc(m_name)}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_resource_efficiency_svg(
    title: str,
    subtitle: str,
    rows: Sequence[Mapping[str, Any]],
    max_rss_mib: float = 1024.0,
) -> str:
    """Render side-by-side Throughput (log ev/s) and Max RSS (MiB) horizontal bars."""
    n = max(1, len(rows))
    row_h = 28
    top_y = 75
    plot_h = n * row_h + 12
    width = 960
    height = top_y + plot_h + 48
    left_x = 175
    tp_w = 320
    rss_x = 575
    rss_w = 320

    max_rss_val = max([max_rss_mib * 1.2] + [_metric_mean(r, "max_rss_mib") or 0.0 for r in rows])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left_x}" y="28" font-size="16" font-weight="700" fill="#0f172a">'
        f"{_esc(title)}</text>",
        f'<text x="{left_x}" y="47" font-size="11.5" fill="#475569">{_esc(subtitle)}</text>',
        f'<text x="{left_x}" y="66" font-size="12" font-weight="700" fill="#1e293b">'
        "Throughput (events/sec, log10 scale)</text>",
        f'<text x="{rss_x}" y="66" font-size="12" font-weight="700" fill="#1e293b">'
        f"Max RSS Memory (MiB, SLA line = {max_rss_mib:.0f} MiB)</text>",
    ]

    # Throughput log grid: 10^2 .. 10^6
    for exp in (2, 3, 4, 5, 6):
        gx = left_x + ((exp - 2) / 4.0) * tp_w
        lines.append(
            f'<line x1="{gx:.1f}" y1="{top_y}" x2="{gx:.1f}" y2="{top_y + plot_h}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{gx:.1f}" y="{top_y + plot_h + 16}" text-anchor="middle" '
            f'font-size="10" fill="#64748b">10^{exp}</text>'
        )

    # RSS SLA dashed line
    sla_rx = rss_x + min(1.0, max_rss_mib / max_rss_val) * rss_w
    lines.append(
        f'<line x1="{sla_rx:.1f}" y1="{top_y}" x2="{sla_rx:.1f}" y2="{top_y + plot_h}" '
        f'stroke="#dc2626" stroke-width="1.3" stroke-dasharray="4,3"/>'
    )

    for idx, row in enumerate(rows):
        y = top_y + 8 + idx * row_h
        m_name = str(row["model"])
        color = MODEL_FAMILY_COLORS.get(m_name, "#334155")
        lines.append(
            f'<text x="{left_x - 10}" y="{y + 13}" text-anchor="end" font-size="11" '
            f'font-weight="600" fill="#1e293b">{_esc(m_name)}</text>'
        )
        tp = _metric_mean(row, "throughput_events_per_sec")
        if tp is not None and tp > 0:
            lt = max(2.0, min(6.0, math.log10(tp)))
            bw = ((lt - 2.0) / 4.0) * tp_w
            lines.append(
                f'<rect x="{left_x}" y="{y}" width="{bw:.1f}" height="16" rx="2" '
                f'fill="{color}" opacity="0.88"/>'
            )
            lines.append(
                f'<text x="{left_x + bw + 5:.1f}" y="{y + 12}" font-size="9.5" '
                f'fill="#334155">{tp:,.0f}</text>'
            )

        rss = _metric_mean(row, "max_rss_mib")
        if rss is not None and rss > 0:
            rw = min(1.0, rss / max_rss_val) * rss_w
            r_col = "#dc2626" if rss > max_rss_mib else color
            lines.append(
                f'<rect x="{rss_x}" y="{y}" width="{rw:.1f}" height="16" rx="2" '
                f'fill="{r_col}" opacity="0.82"/>'
            )
            lines.append(
                f'<text x="{rss_x + rw + 5:.1f}" y="{y + 12}" font-size="9.5" '
                f'fill="#334155">{rss:.0f} MiB</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def generate_m9_charts(
    charts_dir: Path,
    skab_fault_summary: Sequence[Mapping[str, Any]],
    tsb_common_summary: Sequence[Mapping[str, Any]],
    max_latency_ms: float = 0.1,
    max_rss_mib: float = 1024.0,
) -> list[str]:
    """Write all four M9 SVG performance charts and return their relative filenames."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    p1 = "skab_multivariate_metrics.svg"
    (charts_dir / p1).write_text(
        render_grouped_metric_bar_svg(
            title="SKAB Multivariate Anomaly Detection (34 Fault Experiments)",
            subtitle="Mean ± 1 SD across seeds for trained models; deterministic for static/TSFM",
            rows=skab_fault_summary,
            metrics=(
                ("micro_f1", "Micro F1"),
                ("macro_f1", "Macro F1"),
                ("mean_pr_auc", "Mean PR-AUC"),
                ("mean_vus_pr", "Mean VUS-PR"),
            ),
        ),
        encoding="utf-8",
    )
    generated.append(p1)

    p2 = "tsb_univariate_metrics.svg"
    (charts_dir / p2).write_text(
        render_grouped_metric_bar_svg(
            title="TSB-AD-M Independent Univariate (9-Domain TSFM-Common Suite)",
            subtitle="Mean ± 1 SD across seeds for trained models; deterministic for static/TSFM",
            rows=tsb_common_summary,
            metrics=(
                ("macro_f1", "Macro F1"),
                ("mean_pr_auc", "Mean PR-AUC"),
                ("mean_vus_pr", "Mean VUS-PR"),
            ),
        ),
        encoding="utf-8",
    )
    generated.append(p2)

    p3 = "f1_vs_latency_pareto.svg"
    (charts_dir / p3).write_text(
        render_f1_vs_latency_svg(
            title="Accuracy vs. Inference Latency Pareto Frontier (M9)",
            subtitle="Shaded region flags models exceeding streaming latency SLA",
            skab_rows=skab_fault_summary,
            tsb_rows=tsb_common_summary,
            max_latency_ms=max_latency_ms,
        ),
        encoding="utf-8",
    )
    generated.append(p3)

    p4 = "resource_efficiency.svg"
    (charts_dir / p4).write_text(
        render_resource_efficiency_svg(
            title="Resource Efficiency: Throughput & Peak Memory (TSB 9-Domain Common)",
            subtitle="Red RSS bars exceed streaming memory SLA",
            rows=tsb_common_summary or skab_fault_summary,
            max_rss_mib=max_rss_mib,
        ),
        encoding="utf-8",
    )
    generated.append(p4)

    return generated
