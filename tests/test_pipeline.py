import json
import subprocess
import sys

import pandas as pd

from conftest import ROOT
from station_analytics.__main__ import run_checks
from station_analytics.generate import GenConfig, generate
from station_analytics.ingest import connect


def run_cli(*args, cwd=ROOT):
    return subprocess.run([sys.executable, "-m", "station_analytics", *args], cwd=cwd, capture_output=True, text=True, timeout=300)


def test_cli_help_and_generate_smoke(tmp_path):
    assert "demo" in run_cli("--help").stdout
    result = run_cli("generate", "--out", str(tmp_path / "g"), "--days", "3", "--rows", "3000", "--seed", "1")
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "g" / "manifest.json").read_text())
    assert manifest["synthetic"] is True and len(manifest["batches"]) == 3
    assert all((tmp_path / "g" / b["file"]).exists() for b in manifest["batches"])


def test_generator_is_deterministic(tmp_path):
    cfg = GenConfig(seed=5, days=3, units_per_station_day=6)
    m1, m2 = generate(cfg, tmp_path / "a"), generate(cfg, tmp_path / "b")
    assert m1["expected"] == m2["expected"]
    assert (tmp_path / "a" / "batches" / "B20260105.csv").read_bytes() == (tmp_path / "b" / "batches" / "B20260105.csv").read_bytes()


def test_demo_check_passes_on_reduced_run(demo_dir):
    stdout = (demo_dir / "demo_stdout.txt").read_text()
    assert "10 of 10 checks passed" in stdout and "FAIL" not in stdout
    assert all(c.passed for c in run_checks(demo_dir))


def test_manifest_truth_matches_quarantine_counts(demo_dir):
    expected = json.loads((demo_dir / "data" / "manifest.json").read_text())["expected"]
    conn = connect(demo_dir / "warehouse.sqlite")
    reasons = dict(conn.execute("SELECT reason, COUNT(*) FROM quarantine GROUP BY reason").fetchall())
    flagged = conn.execute("SELECT COUNT(DISTINCT unit_id) FROM fact_measurement WHERE quality_flags LIKE '%missing_operator%'").fetchone()[0]
    conn.close()
    assert reasons["exact_duplicate"] == expected["duplicate_rows"]
    assert reasons["sentinel_value"] == expected["sentinel_neg_cells"] + expected["sentinel_pos_cells"]
    assert reasons["out_of_range"] == expected["impossible_cells"] and reasons["type_mismatch"] == expected["type_error_cells"]
    assert reasons["unknown_column"] == expected["v2_batches"] and flagged == expected["missing_operator_rows"]


def test_report_metrics_reconcile_with_sql_and_pandas(demo_dir):
    metrics = json.loads((demo_dir / "analysis" / "metrics.json").read_text())
    conn = connect(demo_dir / "warehouse.sqlite")
    frame = pd.read_sql_query("SELECT station_key, metric, value, study FROM fact_measurement", conn)
    sql_count = conn.execute("SELECT COUNT(*) FROM fact_measurement").fetchone()[0]
    sql_by_station = dict(conn.execute("SELECT station_key, COUNT(*) FROM fact_measurement GROUP BY 1").fetchall())
    conn.close()
    assert metrics["fact_rows"] == sql_count == len(frame) == metrics["ingest"]["accepted"]
    assert metrics["rows_by_station"] == sql_by_station == frame["station_key"].value_counts().to_dict()
    pandas_means = frame.groupby(["station_key", "metric"])["value"].mean()
    for key, value in metrics["station_metric_means"].items():
        station, metric = key.split("|")
        assert abs(pandas_means[(station, metric)] - value) < 1e-9
    cap = pd.read_csv(demo_dir / "analysis" / "capability.csv")
    prod_means = frame[frame["study"] == "production"].groupby(["station_key", "metric"])["value"].mean()
    assert all(abs(prod_means[(r.station_key, r.metric)] - r.mean) < 1e-9 for r in cap.itertuples())


def test_report_html_is_self_contained_and_complete(demo_dir):
    html = (demo_dir / "report.html").read_text()
    metrics = json.loads((demo_dir / "analysis" / "metrics.json").read_text())
    assert "<script" not in html and "http://" not in html.replace("http://www.w3.org/2000/svg", "") and "https://" not in html
    assert html.count("<svg") == 4 and "Synthetic data" in html
    for heading in ("Ingestion summary", "Gauge R", "capability", "Drift summary", "agreement", "Alerts", "Control charts"):
        assert heading in html
    assert f"{metrics['fact_rows']:,}" in html and "STA-04" in html


def test_flagged_series_match_injected_faults(demo_dir):
    injected = json.loads((demo_dir / "data" / "manifest.json").read_text())["injected"]
    drift = pd.read_csv(demo_dir / "analysis" / "drift_summary.csv")
    flagged = {(r.station_key, r.metric): r for r in drift[drift["flagged"]].itertuples()}
    d, s = injected["drift"], injected["shift"]
    assert set(flagged) == {(d["station"], d["metric"]), (s["station"], s["metric"])}
    assert flagged[(s["station"], s["metric"])].pattern == "shift" and abs(flagged[(s["station"], s["metric"])].change_day - s["day"]) <= 3
    assert flagged[(d["station"], d["metric"])].pattern == "drift" and abs(flagged[(d["station"], d["metric"])].change_day - d["start_day"]) <= 10
    grr = pd.read_csv(demo_dir / "analysis" / "grr.csv")
    assert len(grr) == 16 and set(grr["station_key"]) == set(injected["grr"]["stations"])
