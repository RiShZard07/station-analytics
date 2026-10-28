"""Run the measurement-system analysis over the warehouse.

Writes CSV tables, chart data and metrics.json to an output directory. Only canonical
data from the warehouse is read here; nothing is cleaned or renamed at this stage.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import msa
from .ingest import connect, read_frame
from .specs import METRICS

BASELINE_FRACTION = 1 / 3
SUBGROUP_SIZE = 5
IMR_POINTS = 60


def load_facts(conn) -> pd.DataFrame:
    """All accepted measurements in time order with a 1-based day index."""
    df = read_frame(conn, "SELECT station_key, unit_id, metric, value, measured_at, operator, repeat_no, study "
                          "FROM fact_measurement ORDER BY measured_at, measurement_id")
    df["measured_at"] = pd.to_datetime(df["measured_at"])
    df["day"] = (df["measured_at"].dt.normalize() - df["measured_at"].min().normalize()).dt.days + 1
    return df


def grr_table(facts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (station, metric), sub in facts[facts["study"] == "grr"].groupby(["station_key", "metric"]):
        g = msa.gauge_rr(sub)
        rows.append({"station_key": station, "metric": metric, "n_parts": g.n_parts, "n_operators": g.n_operators,
                     "n_repeats": g.n_repeats, "var_repeatability": g.var_repeatability, "var_operator": g.var_operator,
                     "var_interaction": g.var_interaction, "var_part": g.var_part, "pct_grr_total": g.pct_grr_total,
                     "pct_grr_tolerance": g.pct_grr_tolerance(METRICS[metric].tolerance), "ndc": g.ndc,
                     "verdict": msa.grr_verdict(g.pct_grr_total)})
    return pd.DataFrame(rows)


def capability_table(production: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (station, metric), sub in production.groupby(["station_key", "metric"]):
        spec = METRICS[metric]
        c = msa.capability(sub["value"].to_numpy(), spec.lsl, spec.usl, SUBGROUP_SIZE)
        rows.append({"station_key": station, "metric": metric, "lsl": spec.lsl, "usl": spec.usl, **asdict(c)})
    return pd.DataFrame(rows)


def daily_series(production: pd.DataFrame) -> pd.DataFrame:
    """Per station, metric and day: mean, sd and count of production measurements."""
    return production.groupby(["station_key", "metric", "day"])["value"].agg(["mean", "std", "count"]).reset_index()


def _daily_sigma(base: pd.DataFrame) -> float:
    """Standard error of a daily mean from pooled within-day variance over the baseline days."""
    weights = base["count"] - 1
    pooled = np.sqrt((weights * base["std"] ** 2).sum() / weights.sum())
    return float(pooled / np.sqrt(base["count"].mean()))


def drift_table(daily: pd.DataFrame, baseline_days: int, min_segment: int) -> pd.DataFrame:
    rows = []
    for (station, metric), sub in daily.groupby(["station_key", "metric"]):
        sub = sub.sort_values("day")
        days = sub["day"].to_numpy()
        d = msa.drift_summary(sub["mean"].to_numpy(), baseline_days, _daily_sigma(sub.iloc[:baseline_days]),
                              min_segment=min_segment)
        day_of = lambda i: None if i is None else int(days[i])  # noqa: E731
        rows.append({"station_key": station, "metric": metric, "n_days": d.n, "baseline_mean": d.baseline_mean,
                     "sigma": d.sigma, "cusum_alarm_day": day_of(d.cusum_alarm_index), "ewma_alarm_day": day_of(d.ewma_alarm_index),
                     "pattern": d.pattern, "change_day": day_of(d.change_index), "magnitude_sigma": d.magnitude_sigma,
                     "slope_per_day": d.slope_per_step, "flagged": d.flagged})
    return pd.DataFrame(rows)


def correlation_tables(facts: pd.DataFrame, reference: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    matrices, rows = {}, []
    for metric, sub in facts[facts["study"] == "xstation"].groupby("metric"):
        wide = sub.pivot_table(index="unit_id", columns="station_key", values="value").dropna()
        matrices[metric] = wide.corr()
        for station in wide.columns:
            a = msa.agreement(wide[reference], wide[station])
            rows.append({"metric": metric, "station_key": station, "reference": reference, **asdict(a)})
    return matrices, pd.DataFrame(rows)


def ingest_summary(conn) -> dict:
    log = read_frame(conn, "SELECT * FROM ingest_log")
    reasons = read_frame(conn, "SELECT reason, COUNT(*) AS n FROM quarantine GROUP BY reason ORDER BY n DESC")
    return {"batches": int(len(log)), "rows_in": int(log["rows_in"].sum()), "accepted": int(log["accepted"].sum()),
            "quarantined": int(log["quarantined"].sum()),
            "batches_by_version": {k: int(v) for k, v in log["source_version"].value_counts().sort_index().items()},
            "quarantine_reasons": dict(zip(reasons["reason"], reasons["n"].astype(int))),
            "last_loaded_at": str(log["loaded_at"].max())}


def _chart(title: str, y_label: str, x, series: list[tuple[str, object, str]]) -> dict:
    return {"title": title, "y_label": y_label, "x": [int(v) for v in x],
            "series": [{"name": n, "values": [float(v) for v in np.broadcast_to(vals, (len(x),))], "style": s}
                       for n, vals, s in series]}


def _pick_series(drift: pd.DataFrame) -> list:
    """One drift summary row per pattern (drift, shift, stable), falling back to any unused series."""
    chosen: list = []
    for pattern in ("drift", "shift", "stable"):
        unused = [r for r in drift.sort_values(["station_key", "metric"]).itertuples()
                  if all((r.station_key, r.metric) != (c.station_key, c.metric) for c in chosen)]
        chosen.append(next((r for r in unused if r.pattern == pattern), unused[0]))
    return chosen


def chart_specs(daily: pd.DataFrame, drift: pd.DataFrame, production: pd.DataFrame) -> list[dict]:
    """X-bar chart for a drifting series, EWMA for a shifted series, I-MR for a stable one."""
    charts = []
    drift_row, shift_row, stable_row = _pick_series(drift)
    for row, kind in ((drift_row, "xbar"), (shift_row, "ewma")):
        sub = daily[(daily["station_key"] == row.station_key) & (daily["metric"] == row.metric)].sort_values("day")
        unit, label = METRICS[row.metric].unit, f"{row.station_key} {row.metric}"
        if kind == "xbar":
            charts.append(_chart(f"X-bar chart of daily means: {label}", unit, sub["day"], [
                ("daily mean", sub["mean"].to_numpy(), "data"), ("CL", row.baseline_mean, "center"),
                ("UCL", row.baseline_mean + 3 * row.sigma, "limit"), ("LCL", row.baseline_mean - 3 * row.sigma, "limit")]))
        else:
            e = msa.ewma(sub["mean"].to_numpy(), row.baseline_mean, row.sigma)
            charts.append(_chart(f"EWMA chart (lambda 0.2, L 3) of daily means: {label}", unit, sub["day"], [
                ("EWMA", e.z, "data"), ("CL", row.baseline_mean, "center"), ("UCL", e.ucl, "limit"), ("LCL", e.lcl, "limit")]))
    sub = production[(production["station_key"] == stable_row.station_key) & (production["metric"] == stable_row.metric)]
    chart = msa.imr_chart(sub["value"].to_numpy()[:IMR_POINTS])
    label, unit = f"{stable_row.station_key} {stable_row.metric}", METRICS[stable_row.metric].unit
    charts.append(_chart(f"Individuals chart, first {len(chart.x)} units: {label}", unit, range(1, len(chart.x) + 1), [
        ("value", chart.x, "data"), ("CL", chart.x_cl, "center"), ("UCL", chart.x_ucl, "limit"), ("LCL", chart.x_lcl, "limit")]))
    charts.append(_chart(f"Moving range chart: {label}", unit, range(2, len(chart.x) + 1), [
        ("moving range", chart.mr, "data"), ("CL", chart.mr_cl, "center"), ("UCL", chart.mr_ucl, "limit")]))
    return charts


def run_analysis(db_path: Path, out_dir: Path) -> dict:
    """Compute every table, write them under out_dir and return the metrics dictionary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        facts = load_facts(conn)
        reference = conn.execute("SELECT station_key FROM dim_station WHERE is_reference = 1").fetchone()[0]
        summary = ingest_summary(conn)
    finally:
        conn.close()
    production = facts[facts["study"] == "production"]
    n_days = int(facts["day"].max())
    baseline_days = max(3, int(n_days * BASELINE_FRACTION))
    min_segment = max(5, n_days // 10)
    daily = daily_series(production)
    grr, cap, drift = grr_table(facts), capability_table(production), drift_table(daily, baseline_days, min_segment)
    matrices, agree = correlation_tables(facts, reference)
    for name, table in (("grr", grr), ("capability", cap), ("drift_summary", drift), ("agreement", agree)):
        table.to_csv(out_dir / f"{name}.csv", index=False)
    for metric, matrix in matrices.items():
        matrix.to_csv(out_dir / f"correlation_{metric}.csv")
    (out_dir / "charts.json").write_text(json.dumps(chart_specs(daily, drift, production), indent=1))
    flagged = drift[drift["flagged"]]
    worst_cpk = cap.loc[cap["cpk"].idxmin()]
    means = facts.groupby(["station_key", "metric"])["value"].mean()
    metrics = {
        "synthetic": True,
        "ingest": summary,
        "fact_rows": int(len(facts)),
        "stations": int(facts["station_key"].nunique()),
        "units": int(facts["unit_id"].nunique()),
        "days": n_days,
        "baseline_days": baseline_days,
        "min_segment_days": min_segment,
        "reference_station": reference,
        "rows_by_station": {k: int(v) for k, v in facts["station_key"].value_counts().sort_index().items()},
        "station_metric_means": {f"{s}|{m}": float(v) for (s, m), v in means.items()},
        "last_measured_at": facts["measured_at"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "flagged_series": [{"station_key": r.station_key, "metric": r.metric, "pattern": r.pattern, "change_day": int(r.change_day),
                            "magnitude_sigma": round(float(r.magnitude_sigma), 2)} for r in flagged.itertuples()],
        "grr": {"studies": int(len(grr)), "worst_pct_grr_total": round(float(grr["pct_grr_total"].max()), 2),
                "verdicts": {k: int(v) for k, v in grr["verdict"].value_counts().items()}},
        "capability": {"min_cpk": round(float(worst_cpk["cpk"]), 3), "min_cpk_series": f"{worst_cpk['station_key']}|{worst_cpk['metric']}"},
        "agreement": {"min_r_vs_reference": round(float(agree.loc[agree["station_key"] != reference, "r"].min()), 4)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics
