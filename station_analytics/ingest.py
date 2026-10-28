"""Load raw CSV batches into the SQLite warehouse with explicit schema contracts.

Policies (each one is tested):
- Unknown columns are quarantined (column name plus values) and reported, never dropped
  silently. Missing declared columns refuse the batch with SchemaError.
- Known drift is handled by the per-version rename map (v1 lum_nits -> luminance_nits).
- Exact duplicate source rows are quarantined; the first occurrence wins.
- Station names are canonicalised to STA-NN at ingest. Unparseable or unknown stations
  quarantine the whole source row.
- Missing operator or instrument serial keeps the row but records a quality flag.
- Each measurement value is checked for type, sentinel (-999, 9999) and plausible range.
  Bad values quarantine that single measurement, not the whole row.
- A batch is loaded in one transaction and keyed by batch id in ingest_log, so loading
  the same batch twice is a no-op.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .specs import METRIC_NAMES, METRICS, REFERENCE_DATA_DIR, SENTINELS


class SchemaError(ValueError):
    """The batch does not match its declared contract in a way we cannot repair."""


class StationNameError(ValueError):
    """A station name could not be reduced to a canonical STA-NN key."""


@dataclass(frozen=True)
class SchemaContract:
    """Declared columns (source name -> type) and the rename map to canonical names."""

    version: str
    columns: dict[str, str]
    rename: dict[str, str]
    units: dict[str, str]


BASE_COLUMNS = {"station": "str", "unit_id": "str", "measured_at": "datetime", "operator": "str",
                "instrument_serial": "str", "repeat_no": "int", "study": "str"}
METRIC_UNITS = {m.name: m.unit for m in METRICS.values()}
CONTRACTS: dict[str, SchemaContract] = {
    "v1": SchemaContract("v1", {**BASE_COLUMNS, "lum_nits": "float", "color_delta_e": "float", "mtf_50": "float",
                                "binocular_misalignment_arcmin": "float"}, {"lum_nits": "luminance_nits"}, METRIC_UNITS),
    "v2": SchemaContract("v2", {**BASE_COLUMNS, **{m: "float" for m in METRIC_NAMES}}, {}, METRIC_UNITS),
}

_STATION_RE = re.compile(r"^(?:STATION|STA|ST)?[\s_-]*(\d{1,2})$")


def canonical_station(raw: object) -> str:
    """Map tolerant spellings (STA-07, sta07, 'Station 7 ', STA_07) to STA-07.

    Raises StationNameError when the text does not reduce to a station number 1..99.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        raise StationNameError("missing station name")
    match = _STATION_RE.match(str(raw).strip().upper())
    if match is None or int(match.group(1)) == 0:
        raise StationNameError(f"unrecognised station name: {raw!r}")
    return f"STA-{int(match.group(1)):02d}"


DDL = """
CREATE TABLE IF NOT EXISTS dim_site(site_key TEXT PRIMARY KEY, site_name TEXT NOT NULL, site_type TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dim_station(station_key TEXT PRIMARY KEY, site_key TEXT NOT NULL REFERENCES dim_site(site_key),
    is_reference INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS dim_instrument(instrument_id TEXT PRIMARY KEY,
    station_key TEXT NOT NULL REFERENCES dim_station(station_key), serial TEXT NOT NULL, model TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dim_unit(unit_id TEXT PRIMARY KEY, unit_type TEXT NOT NULL, first_seen TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fact_measurement(measurement_id INTEGER PRIMARY KEY, batch_id TEXT NOT NULL,
    station_key TEXT NOT NULL REFERENCES dim_station(station_key), unit_id TEXT NOT NULL, metric TEXT NOT NULL,
    value REAL NOT NULL, measured_at TEXT NOT NULL, operator TEXT, instrument_serial TEXT, repeat_no INTEGER NOT NULL,
    study TEXT NOT NULL, source_version TEXT NOT NULL, quality_flags TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS ix_fact_station_metric_time ON fact_measurement(station_key, metric, measured_at);
CREATE TABLE IF NOT EXISTS quarantine(quarantine_id INTEGER PRIMARY KEY, batch_id TEXT NOT NULL, reason TEXT NOT NULL,
    detail TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ingest_log(batch_id TEXT PRIMARY KEY, source_version TEXT NOT NULL, rows_in INTEGER NOT NULL,
    accepted INTEGER NOT NULL, quarantined INTEGER NOT NULL, loaded_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS schema_versions(source_version TEXT PRIMARY KEY, contract TEXT NOT NULL, registered_at TEXT NOT NULL);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_warehouse(conn: sqlite3.Connection, ref_dir: Path = REFERENCE_DATA_DIR) -> None:
    """Create tables and load the station, site and instrument dimensions."""
    conn.executescript(DDL)
    stations = pd.read_csv(ref_dir / "stations.csv", dtype=str)
    instruments = pd.read_csv(ref_dir / "instruments.csv", dtype=str)
    with conn:
        conn.executemany("INSERT OR REPLACE INTO dim_site VALUES (?,?,?)",
                         stations[["site_key", "site_name", "site_type"]].drop_duplicates().values.tolist())
        conn.executemany("INSERT OR REPLACE INTO dim_station VALUES (?,?,?)",
                         [(r.station_key, r.site_key, int(r.is_reference)) for r in stations.itertuples()])
        conn.executemany("INSERT OR REPLACE INTO dim_instrument VALUES (?,?,?,?)",
                         instruments[["instrument_id", "station", "serial", "model"]].values.tolist())
        conn.executemany("INSERT OR REPLACE INTO schema_versions VALUES (?,?,?)",
                         [(v, json.dumps(c.__dict__, sort_keys=True), _now()) for v, c in CONTRACTS.items()])


def known_stations(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT station_key FROM dim_station")}


@dataclass
class BatchResult:
    """Outcome of loading one batch; counts match the ingest_log row."""

    batch_id: str
    source_version: str
    rows_in: int = 0
    accepted: int = 0
    quarantined: int = 0
    skipped: bool = False
    reasons: dict[str, int] = field(default_factory=dict)


def _check_columns(raw: pd.DataFrame, contract: SchemaContract, q: list) -> pd.DataFrame:
    unknown = [c for c in raw.columns if c not in contract.columns]
    for col in unknown:
        q.append(("unknown_column", col, json.dumps({"column": col, "values": raw[col].tolist()})))
    missing = [c for c in contract.columns if c not in raw.columns]
    if missing:
        raise SchemaError(f"columns {missing} declared in {contract.version} are missing from the batch")
    return raw.drop(columns=unknown).rename(columns=contract.rename)


def _drop_duplicates(frame: pd.DataFrame, q: list) -> pd.DataFrame:
    dup = frame.duplicated(keep="first").to_numpy()
    q.extend(("exact_duplicate", rec["unit_id"], json.dumps(rec)) for rec in frame[dup].to_dict("records"))
    return frame[~dup]


def _validate_rows(frame: pd.DataFrame, stations: set[str], q: list) -> pd.DataFrame:
    """Row-level checks on the wide frame: station, timestamp, repeat number, unit id."""
    mapping: dict[str, str | None] = {}
    for raw in frame["station"].unique():
        try:
            mapping[raw] = canonical_station(raw)
        except StationNameError:
            mapping[raw] = None
    key = frame["station"].map(mapping)
    ts = pd.to_datetime(frame["measured_at"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
    rep = pd.to_numeric(frame["repeat_no"], errors="coerce")
    reason = np.select(
        [key.isna().to_numpy(), ~key.isin(stations).to_numpy(), ts.isna().to_numpy(), rep.isna().to_numpy(),
         (frame["unit_id"].str.strip() == "").to_numpy()],
        ["bad_station", "unknown_station", "bad_timestamp", "bad_repeat", "missing_unit"], default="")
    bad = reason != ""
    q.extend((why, rec["station"], json.dumps(rec)) for why, rec in zip(reason[bad].tolist(), frame[bad].to_dict("records")))
    keep = frame[~bad].copy()
    keep["station_key"] = key[~bad]
    keep["measured_at"] = ts[~bad].dt.strftime("%Y-%m-%d %H:%M:%S")
    keep["repeat_no"] = rep[~bad].astype(int)
    flags = np.where(keep["operator"].str.strip() == "", "missing_operator", "")
    serial_flag = np.where(keep["instrument_serial"].str.strip() == "", "missing_serial", "")
    keep["quality_flags"] = [";".join(f for f in (a, b) if f) for a, b in zip(flags, serial_flag)]
    return keep


def _melt_values(frame: pd.DataFrame, q: list) -> pd.DataFrame:
    """Reshape to long format and check each value for type, sentinel and range."""
    ids = ["station_key", "unit_id", "measured_at", "operator", "instrument_serial", "repeat_no", "study", "quality_flags"]
    long = frame.melt(id_vars=ids, value_vars=list(METRIC_NAMES), var_name="metric", value_name="raw_value")
    raw = long["raw_value"].astype(str).str.strip()
    num = pd.to_numeric(raw, errors="coerce")
    lo = long["metric"].map({m: s.plausible_min for m, s in METRICS.items()}).astype(float)
    hi = long["metric"].map({m: s.plausible_max for m, s in METRICS.items()}).astype(float)
    reason = np.select(
        [(raw == "").to_numpy(), num.isna().to_numpy(), num.isin(SENTINELS).to_numpy(), ((num < lo) | (num > hi)).to_numpy()],
        ["missing_value", "type_mismatch", "sentinel_value", "out_of_range"], default="")
    bad = reason != ""
    cols = ["station_key", "unit_id", "measured_at", "metric", "raw_value"]
    q.extend((why, rec["metric"], json.dumps(rec)) for why, rec in zip(reason[bad].tolist(), long.loc[bad, cols].to_dict("records")))
    good = long[~bad].copy()
    good["value"] = num[~bad].astype(float)
    return good


def _write_batch(conn, batch_id: str, version: str, good: pd.DataFrame, q: list, rows_in: int) -> None:
    opt = lambda s: [v or None for v in s.tolist()]  # noqa: E731 - blank metadata becomes NULL
    rows = list(zip([batch_id] * len(good), good["station_key"].tolist(), good["unit_id"].tolist(), good["metric"].tolist(),
                    good["value"].tolist(), good["measured_at"].tolist(), opt(good["operator"]), opt(good["instrument_serial"]),
                    good["repeat_no"].tolist(), good["study"].tolist(), [version] * len(good), good["quality_flags"].tolist()))
    first_seen = good.groupby("unit_id")["measured_at"].min()
    units = [(u, "golden" if u.startswith("GOLD-") else "production", t) for u, t in first_seen.items()]
    with conn:
        conn.executemany("INSERT INTO fact_measurement(batch_id, station_key, unit_id, metric, value, measured_at, operator, "
                         "instrument_serial, repeat_no, study, source_version, quality_flags) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.executemany("INSERT OR IGNORE INTO dim_unit VALUES (?,?,?)", units)
        conn.executemany("INSERT INTO quarantine(batch_id, reason, detail, payload) VALUES (?,?,?,?)",
                         [(batch_id, *rec) for rec in q])
        conn.execute("INSERT INTO ingest_log VALUES (?,?,?,?,?,?)", (batch_id, version, rows_in, len(rows), len(q), _now()))


def ingest_batch(conn: sqlite3.Connection, csv_path: Path, batch_id: str, source_version: str) -> BatchResult:
    """Validate and load one batch. Re-running with the same batch id is a no-op."""
    if conn.execute("SELECT 1 FROM ingest_log WHERE batch_id = ?", (batch_id,)).fetchone():
        return BatchResult(batch_id, source_version, skipped=True)
    if source_version not in CONTRACTS:
        raise SchemaError(f"no contract declared for source version {source_version!r}")
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    q: list[tuple[str, str, str]] = []
    frame = _drop_duplicates(_check_columns(raw, CONTRACTS[source_version], q), q)
    good = _melt_values(_validate_rows(frame, known_stations(conn), q), q)
    _write_batch(conn, batch_id, source_version, good, q, len(raw))
    reasons = pd.Series([r[0] for r in q]).value_counts().to_dict() if q else {}
    return BatchResult(batch_id, source_version, len(raw), len(good), len(q), False, {k: int(v) for k, v in reasons.items()})


def ingest_dir(db_path: Path, data_dir: Path, ref_dir: Path = REFERENCE_DATA_DIR) -> list[BatchResult]:
    """Load every batch listed in data_dir/manifest.json in manifest order."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    conn = connect(db_path)
    try:
        init_warehouse(conn, ref_dir)
        return [ingest_batch(conn, data_dir / b["file"], b["batch_id"], b["source_version"]) for b in manifest["batches"]]
    finally:
        conn.close()


def read_frame(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)
