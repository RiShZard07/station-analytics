"""Self-contained HTML report: inline CSS and inline SVG charts, no external assets or scripts.

The report renders the analysis outputs (metrics.json, CSV tables, charts.json) and
alerts.json. It never reads raw data and never cleans anything; all canonicalisation
happened at ingest.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

STYLE = """
body{font-family:Georgia,serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222;line-height:1.45}
h1{font-size:1.7em;margin-bottom:0}h2{border-bottom:1px solid #ccc;padding-bottom:4px;margin-top:36px}
.banner{background:#fff3cd;border:1px solid #e0c060;padding:10px 14px;margin:16px 0}
table{border-collapse:collapse;font-size:0.84em;margin:10px 0}th,td{border:1px solid #ddd;padding:3px 7px;text-align:right}
th{background:#f3f3f3}td:first-child,th:first-child{text-align:left}.flag{background:#fde2e2}
.critical{color:#a01818;font-weight:bold}.warning{color:#9a6a00;font-weight:bold}svg{display:block;margin:12px 0}
.kv span{display:inline-block;margin-right:22px}
"""
COLORS = {"data": "#1f5fa8", "center": "#444444", "limit": "#c0392b"}
DASHES = {"data": "none", "center": "6 3", "limit": "4 4"}


def fmt(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else f"{value:,.1f}" if abs(value) >= 100 else f"{value:.4g}"
    return html.escape(str(value))


def html_table(df: pd.DataFrame, highlight: str | None = None) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cls = ' class="flag"' if highlight and bool(row[highlight]) else ""
        rows.append(f"<tr{cls}>" + "".join(f"<td>{fmt(v)}</td>" for v in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def svg_chart(spec: dict, width: int = 760, height: int = 250) -> str:
    """Line chart with constant or series limits. Styles: data, center, limit."""
    left, right, top, bottom = 64, 16, 44, 40
    xs = np.asarray(spec["x"], dtype=float)
    all_y = np.concatenate([np.asarray(s["values"], dtype=float) for s in spec["series"]])
    lo, hi = float(all_y.min()), float(all_y.max())
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    sx = lambda x: left + (x - xs.min()) / max(xs.max() - xs.min(), 1) * (width - left - right)  # noqa: E731
    sy = lambda y: top + (hi - y) / (hi - lo) * (height - top - bottom)  # noqa: E731
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img" aria-label="{html.escape(spec["title"])}">',
             f'<text x="{left}" y="18" font-size="13" font-family="sans-serif">{html.escape(spec["title"])}</text>',
             f'<rect x="{left}" y="{top}" width="{width - left - right}" height="{height - top - bottom}" fill="#fafafa" stroke="#999"/>']
    for frac in (0.0, 0.5, 1.0):
        y = lo + frac * (hi - lo)
        parts.append(f'<text x="{left - 6}" y="{sy(y) + 4:.1f}" font-size="11" text-anchor="end" font-family="sans-serif">{y:.4g}</text>')
    for x in (xs.min(), xs.max()):
        parts.append(f'<text x="{sx(x):.1f}" y="{height - 24}" font-size="11" text-anchor="middle" font-family="sans-serif">{x:g}</text>')
    parts.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 8}" font-size="11" text-anchor="middle" font-family="sans-serif">day or sample</text>')
    legend_x = left + 8
    for s in spec["series"]:
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, s["values"]))
        color, dash = COLORS[s["style"]], DASHES[s["style"]]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" stroke-dasharray="{dash}"/>')
        parts.append(f'<text x="{legend_x}" y="{top - 8}" font-size="11" fill="{color}" font-family="sans-serif">{html.escape(s["name"])}</text>')
        legend_x += 9 * len(s["name"]) + 14
    parts.append("</svg>")
    return "\n".join(parts)


def _kv(items: dict) -> str:
    return '<p class="kv">' + "".join(f"<span><b>{html.escape(str(k))}:</b> {fmt(v)}</span>" for k, v in items.items()) + "</p>"


def render(metrics: dict, tables: dict[str, pd.DataFrame], matrices: dict[str, pd.DataFrame], charts: list[dict], alerts: dict) -> str:
    ing = metrics["ingest"]
    reasons = pd.DataFrame({"reason": list(ing["quarantine_reasons"]), "records": list(ing["quarantine_reasons"].values())})
    alert_items = "".join(
        f'<li><span class="{a["severity"]}">{a["severity"].upper()}</span> [{html.escape(a["kind"])}] {html.escape(a["message"])} '
        f'<i>Action: {html.escape(a["recommended_action"])}</i></li>' for a in alerts["alerts"]) or "<li>No alerts.</li>"
    matrix_html = "".join(f"<h3>{html.escape(m)}</h3>{html_table(mat.round(3).reset_index())}" for m, mat in matrices.items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Station analytics report (synthetic data)</title>
<style>{STYLE}</style></head><body>
<h1>Station analytics report</h1>
<p>Measurement-system analytics for a synthetic multi-station production test line. Last batch loaded {html.escape(ing["last_loaded_at"])}.</p>
<div class="banner"><b>Synthetic data.</b> Every number in this report comes from a seeded generator with documented injected faults.
No real product, instrument, site or person is represented. This is an educational prototype, not a production deployment.</div>
<h2>1. Ingestion summary</h2>
{_kv({"batches": ing["batches"], "source rows in": ing["rows_in"], "measurements accepted": ing["accepted"], "quarantine records": ing["quarantined"],
      "fact rows in warehouse": metrics["fact_rows"], "stations": metrics["stations"], "units": metrics["units"], "days": metrics["days"]})}
{_kv({f"batches on contract {k}": v for k, v in ing["batches_by_version"].items()})}
<h3>Quarantine reasons</h3>{html_table(reasons)}
<h2>2. Gauge R&amp;R (ANOVA, crossed parts x operators with repeats)</h2>
<p>Verdict thresholds on %GRR of total variation: under 10% acceptable, 10 to 30% marginal, over 30% unacceptable.</p>
{html_table(tables["grr"])}
<h2>3. Process capability per station and metric (production units)</h2>
<p>Cp and Cpk use within-subgroup sigma (R-bar over d2, subgroups of 5 in time order); Pp and Ppk use the overall sample sd.</p>
{html_table(tables["capability"])}
<h2>4. Drift summary (daily means, EWMA lambda 0.2 L 3, CUSUM k 0.5 h 5)</h2>
<p>Baseline: first {metrics["baseline_days"]} days. A series is flagged when a chart alarms and the fitted step or ramp is at least 2 sigma of a daily mean.
Alarms without a flag are marginal and are shown for transparency.</p>
{html_table(tables["drift_summary"], highlight="flagged")}
<h2>5. Station-to-station agreement on golden units (reference {html.escape(metrics["reference_station"])})</h2>
{html_table(tables["agreement"])}
<h3>Pairwise Pearson correlation</h3>{matrix_html}
<h2>6. Alerts (as of {html.escape(alerts["as_of"])}, due window {alerts["due_within_days"]} days)</h2>
<ul>{alert_items}</ul>
<h2>7. Control charts</h2>
{"".join(svg_chart(c) for c in charts)}
<p>Generated by station-analytics on synthetic data. Tables are also written as CSV next to metrics.json.</p>
</body></html>"""


def build_report(analysis_dir: Path, alerts_path: Path, out_path: Path) -> Path:
    """Assemble report.html from the analysis directory and alerts.json."""
    metrics = json.loads((analysis_dir / "metrics.json").read_text())
    tables = {name: pd.read_csv(analysis_dir / f"{name}.csv") for name in ("grr", "capability", "drift_summary", "agreement")}
    matrices = {p.stem.removeprefix("correlation_"): pd.read_csv(p, index_col=0) for p in sorted(analysis_dir.glob("correlation_*.csv"))}
    charts = json.loads((analysis_dir / "charts.json").read_text())
    alerts = json.loads(alerts_path.read_text())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(metrics, tables, matrices, charts, alerts))
    return out_path
