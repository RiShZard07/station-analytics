"""Shared fixtures. The demo fixture runs the whole pipeline once per session on a reduced
synthetic data set (30 days) so the pipeline tests stay fast and fully offline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from station_analytics.ingest import connect, init_warehouse

ROOT = Path(__file__).resolve().parent.parent


def base_row(**over) -> dict:
    """One valid v2 source row for station STA-07; override fields to inject a fault."""
    row = {"station": "STA-07", "unit_id": "U1", "measured_at": "2026-01-05 09:00:00", "operator": "OP-A",
           "instrument_serial": "OM200-2202", "repeat_no": 1, "study": "production", "luminance_nits": 500.0,
           "color_delta_e": 1.5, "mtf_50": 0.55, "binocular_misalignment_arcmin": 2.0}
    row.update(over)
    return row


def write_batch(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "w.sqlite")
    init_warehouse(connection)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("demo")
    cmd = [sys.executable, "-m", "station_analytics", "demo", "--check", "--out", str(out), "--days", "30", "--rows", "40000", "--seed", "7"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    (out / "demo_stdout.txt").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return out
