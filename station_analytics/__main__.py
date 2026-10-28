"""Command line: generate, ingest, analyze, alerts, report and demo [--check].

demo --check is the regression gate: it exits non-zero unless every injected fault in the
synthetic data was recovered and the report numbers reconcile with direct SQL.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .alerts import DEFAULT_DUE_WITHIN_DAYS, run_alerts
from .analysis import run_analysis
from .generate import GenConfig, generate, station_keys
from .ingest import canonical_station, connect, ingest_dir
from .report import build_report
from .specs import REFERENCE_DATA_DIR

DRIFT_ONSET_TOLERANCE_DAYS = 10
CHANGE_POINT_TOLERANCE_DAYS = 3


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _ingest_totals(results) -> str:
    loaded = [r for r in results if not r.skipped]
    return (f"{len(loaded)} batches loaded, {len(results) - len(loaded)} skipped, "
            f"{sum(r.rows_in for r in loaded):,} source rows, {sum(r.accepted for r in loaded):,} measurements accepted, "
            f"{sum(r.quarantined for r in loaded):,} quarantine records")


def run_checks(out: Path) -> list[Check]:
    """Compare pipeline outputs with the truth recorded in manifest.json and with direct SQL."""
    manifest = json.loads((out / "data" / "manifest.json").read_text())
    metrics = json.loads((out / "analysis" / "metrics.json").read_text())
    drift = pd.read_csv(out / "analysis" / "drift_summary.csv")
    cap = pd.read_csv(out / "analysis" / "capability.csv")
    exp, inj = manifest["expected"], manifest["injected"]
    conn = connect(out / "warehouse.sqlite")
    q = lambda sql, *p: conn.execute(sql, p).fetchall()  # noqa: E731
    reasons = dict(q("SELECT reason, COUNT(*) FROM quarantine GROUP BY reason"))
    checks: list[Check] = []

    d, s = inj["drift"], inj["shift"]
    row = drift[(drift["station_key"] == d["station"]) & (drift["metric"] == d["metric"])].iloc[0]
    ok = bool(row["flagged"]) and row["pattern"] == "drift" and abs(row["change_day"] - d["start_day"]) <= DRIFT_ONSET_TOLERANCE_DAYS
    checks.append(Check("drifting station flagged", ok, f"{d['station']} {d['metric']} pattern={row['pattern']} onset day "
                        f"{row['change_day']:.0f} vs injected {d['start_day']} (tolerance {DRIFT_ONSET_TOLERANCE_DAYS} days), "
                        f"{row['magnitude_sigma']:.1f} sigma, CUSUM alarm day {row['cusum_alarm_day']:.0f}"))
    row = drift[(drift["station_key"] == s["station"]) & (drift["metric"] == s["metric"])].iloc[0]
    ok = bool(row["flagged"]) and row["pattern"] == "shift" and abs(row["change_day"] - s["day"]) <= CHANGE_POINT_TOLERANCE_DAYS
    checks.append(Check("change point within tolerance", ok, f"{s['station']} {s['metric']} estimated day {row['change_day']:.0f} vs "
                        f"injected {s['day']} (tolerance {CHANGE_POINT_TOLERANCE_DAYS} days), {row['magnitude_sigma']:.1f} sigma"))
    flagged = {(r.station_key, r.metric) for r in drift[drift["flagged"]].itertuples()}
    checks.append(Check("only injected series flagged", flagged == {(d["station"], d["metric"]), (s["station"], s["metric"])},
                        f"flagged {sorted(flagged)}"))

    variants_ok = all(canonical_station(p.format(n=int(k[-2:]))) == k for k in station_keys() for p in inj["name_variants"])
    by_station = dict(q("SELECT station_key, COUNT(*) FROM fact_measurement GROUP BY station_key"))
    bad = reasons.get("bad_station", 0) + reasons.get("unknown_station", 0)
    checks.append(Check("station names canonicalised", variants_ok and by_station == exp["accepted_by_station"] and bad == 0,
                        f"{exp['variant_rows']:,} rows had variant spellings; per-station counts match the generator; "
                        f"{bad} rows quarantined for station names"))
    v1_lum = q("SELECT COUNT(*) FROM fact_measurement WHERE metric='luminance_nits' AND source_version='v1'")[0][0]
    stray = q("SELECT COUNT(*) FROM fact_measurement WHERE metric='lum_nits'")[0][0]
    checks.append(Check("renamed column mapped", v1_lum > 0 and stray == 0, f"{v1_lum:,} v1 lum_nits values stored as luminance_nits"))
    unknown = q("SELECT COUNT(*) FROM quarantine WHERE reason='unknown_column' AND detail=?", inj["unexpected_column"])[0][0]
    checks.append(Check("unexpected column quarantined", unknown == exp["v2_batches"],
                        f"{unknown} quarantine records for {inj['unexpected_column']} across {exp['v2_batches']} v2 batches"))
    dupes_left = q("SELECT COUNT(*) FROM (SELECT 1 FROM fact_measurement GROUP BY station_key, unit_id, metric, measured_at, "
                   "repeat_no, operator, value HAVING COUNT(*) > 1)")[0][0]
    checks.append(Check("exact duplicates removed", reasons.get("exact_duplicate", 0) == exp["duplicate_rows"] and dupes_left == 0,
                        f"{exp['duplicate_rows']} duplicates quarantined, {dupes_left} duplicate groups left in the fact table"))
    ok = (reasons.get("sentinel_value", 0) == exp["sentinel_neg_cells"] + exp["sentinel_pos_cells"]
          and reasons.get("out_of_range", 0) == exp["impossible_cells"] and reasons.get("type_mismatch", 0) == exp["type_error_cells"])
    checks.append(Check("sentinel, impossible and non-numeric values quarantined", ok,
                        f"sentinel {reasons.get('sentinel_value', 0)}, out_of_range {reasons.get('out_of_range', 0)}, "
                        f"type_mismatch {reasons.get('type_mismatch', 0)}"))

    fact_rows = q("SELECT COUNT(*) FROM fact_measurement")[0][0]
    log_total = q("SELECT SUM(accepted) FROM ingest_log")[0][0]
    sql_means = {f"{a}|{b}": c for a, b, c in q("SELECT station_key, metric, AVG(value) FROM fact_measurement GROUP BY 1, 2")}
    means_ok = all(abs(sql_means[k] - v) < 1e-6 for k, v in metrics["station_metric_means"].items())
    prod_means = {(a, b): c for a, b, c in q("SELECT station_key, metric, AVG(value) FROM fact_measurement "
                                             "WHERE study='production' GROUP BY 1, 2")}
    cap_ok = all(abs(prod_means[(r.station_key, r.metric)] - r.mean) < 1e-6 for r in cap.itertuples())
    html_ok = f"{fact_rows:,}" in (out / "report.html").read_text()
    ok = metrics["fact_rows"] == fact_rows == log_total and metrics["rows_by_station"] == by_station and means_ok and cap_ok and html_ok
    checks.append(Check("report reconciles with direct SQL", ok, f"{fact_rows:,} fact rows = ingest_log total = metrics.json; "
                        f"{len(sql_means)} station-metric means and {len(cap)} capability means match SQL; count appears in report.html"))
    before = fact_rows
    again = ingest_dir(out / "warehouse.sqlite", out / "data")
    after = q("SELECT COUNT(*) FROM fact_measurement")[0][0]
    checks.append(Check("re-ingest is idempotent", all(r.skipped for r in again) and before == after,
                        f"{len(again)} batches skipped on second load, fact rows unchanged at {after:,}"))
    conn.close()
    return checks


def run_demo(out: Path, seed: int = 42, days: int = 90, rows: int | None = None, check: bool = False) -> int:
    """Run the whole pipeline into out/ and optionally the regression checks."""
    data_dir, db, analysis_dir = out / "data", out / "warehouse.sqlite", out / "analysis"
    if db.exists():
        db.unlink()
    cfg = GenConfig.from_rows(rows, days, seed)
    manifest = generate(cfg, data_dir)
    print(f"generated {len(manifest['batches'])} synthetic batches ({cfg.days} days, {cfg.units_per_station_day} units per station-day) in {data_dir}")
    print("ingest: " + _ingest_totals(ingest_dir(db, data_dir)))
    metrics = run_analysis(db, analysis_dir)
    print(f"analysis: {metrics['fact_rows']:,} fact rows, {len(metrics['flagged_series'])} flagged series, "
          f"worst %GRR {metrics['grr']['worst_pct_grr_total']}, min Cpk {metrics['capability']['min_cpk']} ({metrics['capability']['min_cpk_series']})")
    alerts = run_alerts(REFERENCE_DATA_DIR / "instruments.csv", analysis_dir, out / "alerts.json")
    print(f"alerts: {alerts['counts']} as of {alerts['as_of']}")
    print(f"report: {build_report(analysis_dir, out / 'alerts.json', out / 'report.html')}")
    if not check:
        return 0
    checks = run_checks(out)
    print("\ndemo --check (synthetic data, injected faults recovered?)")
    for c in checks:
        print(f"  {'PASS' if c.passed else 'FAIL'}  {c.name}: {c.detail}")
    failed = [c for c in checks if not c.passed]
    print(f"{len(checks) - len(failed)} of {len(checks)} checks passed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="station_analytics", description="Measurement-system analytics on a synthetic multi-station test line.")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="write synthetic CSV batches and manifest.json")
    g.add_argument("--out", type=Path, default=Path("data/generated"))
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--days", type=int, default=90)
    g.add_argument("--rows", type=int, default=None, help="approximate number of measurement values to generate")
    i = sub.add_parser("ingest", help="load batches into the SQLite warehouse")
    i.add_argument("--data", type=Path, default=Path("data/generated"))
    i.add_argument("--db", type=Path, default=Path("output/warehouse.sqlite"))
    a = sub.add_parser("analyze", help="run Gauge R&R, capability, drift and agreement analysis")
    a.add_argument("--db", type=Path, default=Path("output/warehouse.sqlite"))
    a.add_argument("--out", type=Path, default=Path("output/analysis"))
    al = sub.add_parser("alerts", help="combine calibration status and drift flags into alerts.json")
    al.add_argument("--analysis", type=Path, default=Path("output/analysis"))
    al.add_argument("--instruments", type=Path, default=REFERENCE_DATA_DIR / "instruments.csv")
    al.add_argument("--as-of", type=date.fromisoformat, default=None, help="ISO date; defaults to the last measurement date")
    al.add_argument("--due-within", type=int, default=DEFAULT_DUE_WITHIN_DAYS)
    al.add_argument("--out", type=Path, default=Path("output/alerts.json"))
    r = sub.add_parser("report", help="render the self-contained HTML report")
    r.add_argument("--analysis", type=Path, default=Path("output/analysis"))
    r.add_argument("--alerts", type=Path, default=Path("output/alerts.json"))
    r.add_argument("--out", type=Path, default=Path("output/report.html"))
    dm = sub.add_parser("demo", help="run everything into one directory")
    dm.add_argument("--out", type=Path, default=Path("output"))
    dm.add_argument("--seed", type=int, default=42)
    dm.add_argument("--days", type=int, default=90)
    dm.add_argument("--rows", type=int, default=None)
    dm.add_argument("--check", action="store_true", help="exit non-zero unless all injected faults were recovered")
    args = p.parse_args(argv)

    if args.command == "generate":
        m = generate(GenConfig.from_rows(args.rows, args.days, args.seed), args.out)
        print(f"wrote {len(m['batches'])} batches and manifest.json to {args.out}")
    elif args.command == "ingest":
        args.db.parent.mkdir(parents=True, exist_ok=True)
        print("ingest: " + _ingest_totals(ingest_dir(args.db, args.data)))
    elif args.command == "analyze":
        m = run_analysis(args.db, args.out)
        print(f"analysis written to {args.out}: {m['fact_rows']:,} fact rows, {len(m['flagged_series'])} flagged series")
    elif args.command == "alerts":
        a = run_alerts(args.instruments, args.analysis, args.out, args.as_of, args.due_within)
        print(f"{sum(a['counts'].values())} alerts written to {args.out} (as of {a['as_of']})")
    elif args.command == "report":
        print(f"report written to {build_report(args.analysis, args.alerts, args.out)}")
    else:
        return run_demo(args.out, args.seed, args.days, args.rows, args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
