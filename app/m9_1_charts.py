"""Revised zero-dependency pure-SVG charts for DriftGuard-Kafka M9.1.

Generates 5 publication-ready vector charts:
1. `track_a_multivariate.svg` — Track A (Multivariate) SKAB & TSB-AD-M Macro F1 + PR-AUC
   with 95% bootstrap confidence intervals.
2. `track_b_independent_channel.svg` — Track B (Independent-Channel) SKAB & TSB-AD-M
   Macro F1 + PR-AUC with 95% bootstrap confidence intervals.
3. `protocol_gap_comparison.svg` — Direct Track A (Multivariate) vs. Track B
   (Independent-Channel) PR-AUC delta across models on TSB-AD-M.
4. `calibration_fpr_tradeoff.svg` — Validation-only calibration strategies on SKAB:
   Anomaly-free False Alarms per 1,000 Normal vs. Fault Micro F1 (highlighting TranAD).
5. `isolated_resource_profile.svg` — Isolated subprocess memory breakdown
   (baseline vs model-load delta vs inference delta + MPS) and offline-batched latency.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.m9_charts import MODEL_FAMILY_COLORS


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def render_ci_bar_chart_svg(
    title: str,
    subtitle: str,
    rows: Sequence[Mapping[str, Any]],
    metric_key: str,
    ci_low_key: str,
    ci_high_key: str,
    metric_label: str,
) -> str:
    """Render horizontal bar chart with 95% bootstrap CI whiskers."""
    n = max(1, len(rows))
    row_h = 28
    margin_l = 175
    margin_r = 125
    margin_t = 72
    margin_b = 42
    plot_w = 560
    plot_h = n * row_h + 10
    width = margin_l + plot_w + margin_r
    height = margin_t + plot_h + margin_b

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{margin_l}" y="28" font-size="15.5" font-weight="700" fill="#0f172a">'
        f"{_esc(title)}</text>",
        f'<text x="{margin_l}" y="47" font-size="11" fill="#475569">{_esc(subtitle)}</text>',
        f'<text x="{margin_l + plot_w / 2:.1f}" y="{margin_t + plot_h + 34}" text-anchor="middle" '
        f'font-size="11" fill="#334155">{_esc(metric_label)} (with 95% bootstrap CI)</text>',
    ]

    for t in range(6):
        val = t * 0.2
        gx = margin_l + val * plot_w
        lines.append(
            f'<line x1="{gx:.1f}" y1="{margin_t}" x2="{gx:.1f}" y2="{margin_t + plot_h}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{gx:.1f}" y="{margin_t + plot_h + 16}" text-anchor="middle" '
            f'font-size="10" fill="#64748b">{val:.1f}</text>'
        )

    for idx, row in enumerate(rows):
        y = margin_t + 6 + idx * row_h
        m_name = str(row["model"])
        color = MODEL_FAMILY_COLORS.get(m_name, "#334155")
        lines.append(
            f'<text x="{margin_l - 10}" y="{y + 13}" text-anchor="end" font-size="11" '
            f'font-weight="600" fill="#1e293b">{_esc(m_name)}</text>'
        )
        v = row.get(metric_key)
        if v is None:
            continue
        fv = max(0.0, min(1.0, float(v)))
        bw = fv * plot_w
        lines.append(
            f'<rect x="{margin_l}" y="{y}" width="{bw:.1f}" height="16" rx="2" '
            f'fill="{color}" opacity="0.86"/>'
        )
        lo = row.get(ci_low_key)
        hi = row.get(ci_high_key)
        txt_x = margin_l + bw + 6
        if lo is not None and hi is not None:
            lo_x = margin_l + max(0.0, min(1.0, float(lo))) * plot_w
            hi_x = margin_l + max(0.0, min(1.0, float(hi))) * plot_w
            cy = y + 8.0
            lines.append(
                f'<line x1="{lo_x:.1f}" y1="{cy:.1f}" x2="{hi_x:.1f}" y2="{cy:.1f}" '
                f'stroke="#0f172a" stroke-width="1.4"/>'
            )
            lines.append(
                f'<line x1="{lo_x:.1f}" y1="{cy - 4:.1f}" x2="{lo_x:.1f}" y2="{cy + 4:.1f}" '
                f'stroke="#0f172a" stroke-width="1.4"/>'
            )
            lines.append(
                f'<line x1="{hi_x:.1f}" y1="{cy - 4:.1f}" x2="{hi_x:.1f}" y2="{cy + 4:.1f}" '
                f'stroke="#0f172a" stroke-width="1.4"/>'
            )
            txt_x = max(txt_x, hi_x + 6)
            lbl = f"{fv:.3f} [{float(lo):.3f}, {float(hi):.3f}]"
        else:
            lbl = f"{fv:.3f}"
        lines.append(
            f'<text x="{txt_x:.1f}" y="{y + 12}" font-size="9.5" fill="#334155">{_esc(lbl)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_protocol_gap_svg(
    title: str,
    subtitle: str,
    pairs: Sequence[Mapping[str, Any]],
) -> str:
    """Render Track A (Multivariate) vs Track B (Independent-Channel) PR-AUC comparison."""
    n = max(1, len(pairs))
    row_h = 34
    margin_l = 175
    margin_r = 90
    margin_t = 76
    margin_b = 40
    plot_w = 560
    plot_h = n * row_h + 10
    width = margin_l + plot_w + margin_r
    height = margin_t + plot_h + margin_b

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{margin_l}" y="28" font-size="15.5" font-weight="700" fill="#0f172a">'
        f"{_esc(title)}</text>",
        f'<text x="{margin_l}" y="46" font-size="11" fill="#475569">{_esc(subtitle)}</text>',
        f'<rect x="{margin_l}" y="55" width="11" height="11" rx="2" fill="#2563eb"/>',
        f'<text x="{margin_l + 15}" y="64" font-size="10.5" fill="#1e293b">'
        "Track A: Multivariate</text>",
        f'<rect x="{margin_l + 170}" y="55" width="11" height="11" rx="2" fill="#d97706"/>',
        f'<text x="{margin_l + 185}" y="64" font-size="10.5" fill="#1e293b">'
        "Track B: Independent-Channel</text>",
    ]

    for t in range(6):
        val = t * 0.2
        gx = margin_l + val * plot_w
        lines.append(
            f'<line x1="{gx:.1f}" y1="{margin_t}" x2="{gx:.1f}" y2="{margin_t + plot_h}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{gx:.1f}" y="{margin_t + plot_h + 16}" text-anchor="middle" '
            f'font-size="10" fill="#64748b">{val:.1f}</text>'
        )

    for idx, p in enumerate(pairs):
        y = margin_t + 6 + idx * row_h
        lines.append(
            f'<text x="{margin_l - 10}" y="{y + 16}" text-anchor="end" font-size="11" '
            f'font-weight="600" fill="#1e293b">{_esc(str(p["model"]))}</text>'
        )
        va = p.get("track_a_pr_auc")
        vb = p.get("track_b_pr_auc")
        if va is not None:
            wa = max(0.0, min(1.0, float(va))) * plot_w
            lines.append(
                f'<rect x="{margin_l}" y="{y}" width="{wa:.1f}" height="11" rx="2" '
                f'fill="#2563eb" opacity="0.88"/>'
            )
            lines.append(
                f'<text x="{margin_l + wa + 5:.1f}" y="{y + 9.5}" font-size="9.5" '
                f'fill="#1e3a8a">{float(va):.3f}</text>'
            )
        if vb is not None:
            wb = max(0.0, min(1.0, float(vb))) * plot_w
            lines.append(
                f'<rect x="{margin_l}" y="{y + 13}" width="{wb:.1f}" height="11" rx="2" '
                f'fill="#d97706" opacity="0.88"/>'
            )
            lines.append(
                f'<text x="{margin_l + wb + 5:.1f}" y="{y + 22.5}" font-size="9.5" '
                f'fill="#92400e">{float(vb):.3f}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_calibration_tradeoff_svg(
    title: str,
    subtitle: str,
    cal_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Render SKAB Anomaly-Free FA/1k Normal vs Fault Micro F1 across validation-only strategies."""
    width = 920
    height = 420
    x0, y0, pw, ph = 80, 75, 760, 275
    max_fa = max([50.0] + [float(r["anomaly_free_fa_per_1k_normal"] or 0.0) for r in cal_rows])
    strat_colors = {
        "empirical_q99": "#dc2626",
        "empirical_q995": "#d97706",
        "mad_z35": "#2563eb",
        "iqr_k30": "#059669",
        "evt_pot_q99": "#7c3aed",
    }

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{x0}" y="28" font-size="15.5" font-weight="700" fill="#0f172a">'
        f"{_esc(title)}</text>",
        f'<text x="{x0}" y="46" font-size="11" fill="#475569">{_esc(subtitle)}</text>',
        f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f8fafc" '
        f'stroke="#cbd5e1" stroke-width="1"/>',
    ]

    # Legend
    lx = x0
    for st, col in strat_colors.items():
        lines.append(f'<circle cx="{lx + 5}" cy="62" r="4.5" fill="{col}"/>')
        lines.append(f'<text x="{lx + 14}" y="66" font-size="10.5" fill="#1e293b">{st}</text>')
        lx += 130

    for t in range(6):
        yv = t * 0.2
        py = y0 + ph - yv * ph
        lines.append(
            f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + pw}" y2="{py:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x0 - 8}" y="{py + 3.5:.1f}" text-anchor="end" font-size="10" '
            f'fill="#64748b">{yv:.1f}</text>'
        )

    for t in range(5):
        xv = (t / 4.0) * max_fa
        px = x0 + (t / 4.0) * pw
        lines.append(
            f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0 + ph}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{px:.1f}" y="{y0 + ph + 16}" text-anchor="middle" font-size="10" '
            f'fill="#64748b">{xv:.0f}</text>'
        )

    lines.append(
        f'<text x="{x0 + pw / 2:.1f}" y="{y0 + ph + 34}" text-anchor="middle" font-size="11" '
        f'fill="#334155">SKAB Anomaly-Free False Alarms per 1,000 Normal Observations '
        "(lower is better)</text>"
    )
    lines.append(
        f'<text x="24" y="{y0 + ph / 2:.1f}" text-anchor="middle" font-size="11" fill="#334155" '
        f'transform="rotate(-90 24 {y0 + ph / 2:.1f})">'
        "SKAB Fault Micro F1 (higher is better)</text>"
    )

    for r in cal_rows:
        fa = r.get("anomaly_free_fa_per_1k_normal")
        f1 = r.get("fault_micro_f1")
        if fa is None or f1 is None:
            continue
        px = x0 + min(1.0, float(fa) / max_fa) * pw
        py = y0 + ph - max(0.0, min(1.0, float(f1))) * ph
        col = strat_colors.get(str(r["strategy"]), "#334155")
        lines.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}" '
            f'stroke="#ffffff" stroke-width="1"/>'
        )
        if str(r["model"]) == "tranad":
            lines.append(
                f'<text x="{px + 7:.1f}" y="{py + 3:.1f}" font-size="9" font-weight="600" '
                f'fill="#0f172a">tranad:{_esc(str(r["strategy"]))}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_isolated_resource_svg(
    title: str,
    subtitle: str,
    prof_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Render isolated subprocess stacked RSS delta + offline-batched log latency."""
    ok_rows = [r for r in prof_rows if r.get("status") == "ok"]
    n = max(1, len(ok_rows))
    row_h = 28
    top_y = 76
    plot_h = n * row_h + 12
    width = 960
    height = top_y + plot_h + 44
    left_x = 175
    mem_w = 340
    lat_x = 585
    lat_w = 310

    max_mem = max(
        [100.0]
        + [
            float(r.get("model_load_rss_delta_mib") or 0.0)
            + float(r.get("inference_rss_delta_mib") or 0.0)
            + float(r.get("mps_current_allocated_mib") or 0.0)
            for r in ok_rows
        ]
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left_x}" y="28" font-size="15.5" font-weight="700" fill="#0f172a">'
        f"{_esc(title)}</text>",
        f'<text x="{left_x}" y="46" font-size="11" fill="#475569">{_esc(subtitle)}</text>',
        f'<text x="{left_x}" y="66" font-size="11.5" font-weight="700" fill="#1e293b">'
        "Isolated Incremental Memory (Model Load + Inference + MPS MiB)</text>",
        f'<text x="{lat_x}" y="66" font-size="11.5" font-weight="700" fill="#1e293b">'
        "Offline-Batched Latency (ms/event, log10)</text>",
    ]

    for exp in (-3, -2, -1, 0):
        gx = lat_x + ((exp + 3) / 3.0) * lat_w
        lines.append(
            f'<line x1="{gx:.1f}" y1="{top_y}" x2="{gx:.1f}" y2="{top_y + plot_h}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{gx:.1f}" y="{top_y + plot_h + 16}" text-anchor="middle" font-size="10" '
            f'fill="#64748b">{10.0**exp:g} ms</text>'
        )

    for idx, r in enumerate(ok_rows):
        y = top_y + 8 + idx * row_h
        m_name = str(r["model"])
        lines.append(
            f'<text x="{left_x - 10}" y="{y + 13}" text-anchor="end" font-size="11" '
            f'font-weight="600" fill="#1e293b">{_esc(m_name)}</text>'
        )
        load_d = float(r.get("model_load_rss_delta_mib") or 0.0)
        inf_d = float(r.get("inference_rss_delta_mib") or 0.0)
        mps_m = float(r.get("mps_current_allocated_mib") or 0.0)
        w1 = (load_d / max_mem) * mem_w
        w2 = (inf_d / max_mem) * mem_w
        w3 = (mps_m / max_mem) * mem_w
        lines.append(
            f'<rect x="{left_x}" y="{y}" width="{w1:.1f}" height="15" '
            'fill="#2563eb" opacity="0.85"/>'
        )
        lines.append(
            f'<rect x="{left_x + w1:.1f}" y="{y}" width="{w2:.1f}" height="15" '
            f'fill="#0891b2" opacity="0.85"/>'
        )
        lines.append(
            f'<rect x="{left_x + w1 + w2:.1f}" y="{y}" width="{w3:.1f}" height="15" '
            f'fill="#d97706" opacity="0.85"/>'
        )
        lines.append(
            f'<text x="{left_x + w1 + w2 + w3 + 5:.1f}" y="{y + 11.5}" font-size="9" '
            f'fill="#334155">+{load_d + inf_d:.0f} MiB (MPS {mps_m:.0f})</text>'
        )

        lat = float(r.get("offline_batched_mean_latency_ms") or 0.001)
        lv = max(-3.0, min(0.0, math.log10(max(1e-3, lat))))
        lw = ((lv + 3.0) / 3.0) * lat_w
        col = MODEL_FAMILY_COLORS.get(m_name, "#334155")
        lines.append(
            f'<rect x="{lat_x}" y="{y}" width="{max(2.0, lw):.1f}" height="15" rx="2" '
            f'fill="{col}" opacity="0.85"/>'
        )
        lines.append(
            f'<text x="{lat_x + max(2.0, lw) + 5:.1f}" y="{y + 11.5}" font-size="9" '
            f'fill="#334155">{lat:.4f} ms</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def generate_m9_1_charts(
    charts_dir: Path,
    track_a_tsb_rows: Sequence[Mapping[str, Any]],
    track_b_tsb_rows: Sequence[Mapping[str, Any]],
    protocol_pairs: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    isolated_profile_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Write all 5 revised M9.1 SVG charts and return filenames."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    p1 = "track_a_multivariate.svg"
    (charts_dir / p1).write_text(
        render_ci_bar_chart_svg(
            title="Track A (Multivariate): TSB-AD-M Mean PR-AUC with 95% Bootstrap CI",
            subtitle="All compatible models receive identical multivariate feature matrices",
            rows=track_a_tsb_rows,
            metric_key="mean_pr_auc_mean",
            ci_low_key="mean_pr_auc_ci_low",
            ci_high_key="mean_pr_auc_ci_high",
            metric_label="Mean PR-AUC",
        ),
        encoding="utf-8",
    )
    generated.append(p1)

    p2 = "track_b_independent_channel.svg"
    (charts_dir / p2).write_text(
        render_ci_bar_chart_svg(
            title="Track B (Independent-Channel): TSB-AD-M Mean PR-AUC with 95% Bootstrap CI",
            subtitle="Identical 1D normalization, mean aggregation & validation calibration",
            rows=track_b_tsb_rows,
            metric_key="mean_pr_auc_mean",
            ci_low_key="mean_pr_auc_ci_low",
            ci_high_key="mean_pr_auc_ci_high",
            metric_label="Mean PR-AUC",
        ),
        encoding="utf-8",
    )
    generated.append(p2)

    p3 = "protocol_gap_comparison.svg"
    (charts_dir / p3).write_text(
        render_protocol_gap_svg(
            title="Protocol Audit: Track A (Multivariate) vs. Track B (Independent-Channel)",
            subtitle="Direct PR-AUC comparison on identical TSB-AD-M multivariate series",
            pairs=protocol_pairs,
        ),
        encoding="utf-8",
    )
    generated.append(p3)

    p4 = "calibration_fpr_tradeoff.svg"
    (charts_dir / p4).write_text(
        render_calibration_tradeoff_svg(
            title="Validation-Only Threshold Calibration: SKAB Anomaly-Free FA/1k vs Fault F1",
            subtitle="Zero test-label tuning; highlights TranAD FPR under robust calibration",
            cal_rows=calibration_rows,
        ),
        encoding="utf-8",
    )
    generated.append(p4)

    skab_prof = [r for r in isolated_profile_rows if r.get("dataset_suite") == "SKAB"]
    p5 = "isolated_resource_profile.svg"
    (charts_dir / p5).write_text(
        render_isolated_resource_svg(
            title="Isolated Subprocess Resource Profile (SKAB Multivariate)",
            subtitle="Fresh Python process per model separating load delta, inference delta & MPS",
            prof_rows=skab_prof or isolated_profile_rows,
        ),
        encoding="utf-8",
    )
    generated.append(p5)

    return generated
