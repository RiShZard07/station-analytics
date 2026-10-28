"""Seeded synthetic data generator for a multi-station optical display-module test line.

Everything written by this module is synthetic. Faults are injected on purpose so the
ingest and analysis layers can be checked against the truth recorded in manifest.json.

Injected signals (deterministic for a given seed and configuration):

a. Per-station bias offsets (STATION_BIAS).
b. A slow linear drift on DRIFT_STATION / DRIFT_METRIC that starts on
   GenConfig.drift_start_day and reaches DRIFT_TOTAL by the last day.
c. A sudden mean shift of SHIFT_DELTA on SHIFT_STATION / SHIFT_METRIC from
   GenConfig.shift_day onward.
d. A Gauge R&R block: 10 golden units x 3 operators x 3 repeats on GRR_STATIONS.
e. A station-to-station block: 30 golden units measured once on all 12 stations.
f. Data-quality faults on production rows: inconsistent station spellings, missing
   operator and instrument serial, exact duplicate rows, sentinel values, impossible
   values, non-numeric cells, and a schema change where batches from
   GenConfig.schema_change_day onward rename lum_nits to luminance_nits and add an
   undeclared fixture_temp_c column.

Study rows (grr, xstation) only receive name variants and duplicates, so the Gauge R&R
design stays balanced.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .specs import METRIC_NAMES, METRICS, REFERENCE_DATA_DIR

N_STATIONS = 12
REFERENCE_STATION = "STA-12"
OPERATORS = ("OP-A", "OP-B", "OP-C")
OPERATOR_BIAS_SD_UNITS = {"OP-A": 0.0, "OP-B": 0.4, "OP-C": -0.4}  # in units of meas_sd
GRR_STATIONS = ("STA-01", "STA-06", "STA-11", "STA-12")
GOLDEN_UNITS = 30
GRR_UNITS, GRR_REPEATS = 10, 3

STATION_BIAS: dict[tuple[str, str], float] = {
    ("STA-02", "luminance_nits"): 4.0,
    ("STA-10", "luminance_nits"): -3.0,
    ("STA-06", "color_delta_e"): 0.15,
    ("STA-09", "mtf_50"): -0.012,
    ("STA-05", "binocular_misalignment_arcmin"): 0.3,
}
DRIFT_STATION, DRIFT_METRIC, DRIFT_TOTAL = "STA-04", "luminance_nits", 20.0
SHIFT_STATION, SHIFT_METRIC, SHIFT_DELTA = "STA-08", "mtf_50", -0.03

NAME_VARIANTS = ("sta{n:02d}", "Station {n} ", "STA_{n:02d}", "STA-{n}", " sta-{n:02d}")
ROW_FAULT_RATES = {"name_variant": 0.12, "missing_operator": 0.03, "missing_serial": 0.02, "duplicate": 0.005}
CELL_FAULT_RATES = {"sentinel_neg": 0.0015, "sentinel_pos": 0.001, "impossible": 0.0005, "type_error": 0.0005}
IMPOSSIBLE_VALUES = {"luminance_nits": -5.0, "color_delta_e": 500.0, "mtf_50": 2.5, "binocular_misalignment_arcmin": -3.0}
V1_RENAMES = {"luminance_nits": "lum_nits"}  # canonical -> v1 source name
UNEXPECTED_COLUMN = "fixture_temp_c"


@dataclass(frozen=True)
class GenConfig:
    """Size and timing of one synthetic run. Event days scale with the horizon."""

    seed: int = 42
    days: int = 90
    units_per_station_day: int = 45
    start_date: date = date(2026, 1, 5)

    @classmethod
    def from_rows(cls, rows: int | None, days: int, seed: int) -> "GenConfig":
        """Pick units per station-day so the run yields about rows measurement values."""
        if rows is None:
            return cls(seed=seed, days=days)
        units = max(4, round(rows / (days * N_STATIONS * len(METRIC_NAMES))))
        return cls(seed=seed, days=days, units_per_station_day=units)

    def _scaled(self, day_on_90: int) -> int:
        return max(2, round(self.days * day_on_90 / 90))

    @property
    def drift_start_day(self) -> int:
        return self._scaled(50)

    @property
    def shift_day(self) -> int:
        return self._scaled(60)

    @property
    def schema_change_day(self) -> int:
        return self._scaled(55)

    @property
    def grr_day(self) -> int:
        return self._scaled(15)

    @property
    def xstation_day(self) -> int:
        return self._scaled(22)

    @property
    def drift_per_day(self) -> float:
        return DRIFT_TOTAL / max(1, self.days - self.drift_start_day)

    def date_of(self, day: int) -> date:
        return self.start_date + timedelta(days=day - 1)


def station_keys() -> list[str]:
    return [f"STA-{n:02d}" for n in range(1, N_STATIONS + 1)]


def load_serials(ref_dir: Path = REFERENCE_DATA_DIR) -> dict[str, str]:
    inst = pd.read_csv(ref_dir / "instruments.csv", dtype=str)
    return dict(zip(inst["station"], inst["serial"]))


def effect(cfg: GenConfig, station: str, metric: str, day: int) -> float:
    """Deterministic offset added to true values: bias plus drift or shift."""
    out = STATION_BIAS.get((station, metric), 0.0)
    if station == DRIFT_STATION and metric == DRIFT_METRIC and day > cfg.drift_start_day:
        out += cfg.drift_per_day * (day - cfg.drift_start_day)
    if station == SHIFT_STATION and metric == SHIFT_METRIC and day >= cfg.shift_day:
        out += SHIFT_DELTA
    return out


def _timestamps(rng: np.random.Generator, cfg: GenConfig, day: int, n: int) -> list[str]:
    secs = np.sort(rng.integers(8 * 3600, 20 * 3600, size=n))
    base = pd.Timestamp(cfg.date_of(day))
    return list((base + pd.to_timedelta(secs, unit="s")).strftime("%Y-%m-%d %H:%M:%S"))


def _frame(rng, cfg, day, station, unit_ids, operators, repeats, study, serials, true_values) -> pd.DataFrame:
    """Assemble one wide block: true values plus station effects plus noise."""
    n = len(unit_ids)
    cols: dict[str, object] = {
        "station": station, "unit_id": unit_ids, "measured_at": _timestamps(rng, cfg, day, n),
        "operator": operators, "instrument_serial": serials[station], "repeat_no": repeats, "study": study,
    }
    for name, spec in METRICS.items():
        op_bias = np.array([OPERATOR_BIAS_SD_UNITS[o] for o in operators]) * spec.meas_sd
        noise = rng.normal(0.0, spec.meas_sd, n)
        measured = true_values[name] + effect(cfg, station, name, day) + op_bias + noise
        cols[name] = np.abs(measured) if spec.plausible_min == 0.0 else measured  # magnitudes cannot go negative
    return pd.DataFrame(cols)


def _production_day(rng, cfg, day, serials) -> list[pd.DataFrame]:
    n = cfg.units_per_station_day
    blocks = []
    for station in station_keys():
        sn = int(station[-2:])
        ids = [f"U{day:03d}-{sn:02d}-{i:04d}" for i in range(1, n + 1)]
        true = {m: rng.normal(s.target, s.unit_sd, n) for m, s in METRICS.items()}
        ops = list(rng.choice(OPERATORS, size=n))
        blocks.append(_frame(rng, cfg, day, station, ids, ops, np.ones(n, dtype=int), "production", serials, true))
    return blocks


def _grr_block(rng, cfg, gold: dict[str, np.ndarray], serials) -> list[pd.DataFrame]:
    unit_idx = np.repeat(np.arange(GRR_UNITS), len(OPERATORS) * GRR_REPEATS)
    ops = list(np.tile(np.repeat(OPERATORS, GRR_REPEATS), GRR_UNITS))
    reps = np.tile(np.arange(1, GRR_REPEATS + 1), GRR_UNITS * len(OPERATORS))
    ids = [f"GOLD-{i + 1:02d}" for i in unit_idx]
    true = {m: v[unit_idx] for m, v in gold.items()}
    return [_frame(rng, cfg, cfg.grr_day, s, ids, ops, reps, "grr", serials, true) for s in GRR_STATIONS]


def _xstation_block(rng, cfg, gold: dict[str, np.ndarray], serials) -> list[pd.DataFrame]:
    ids = [f"GOLD-{i + 1:02d}" for i in range(GOLDEN_UNITS)]
    ops, reps = ["OP-A"] * GOLDEN_UNITS, np.ones(GOLDEN_UNITS, dtype=int)
    return [_frame(rng, cfg, cfg.xstation_day, s, ids, ops, reps, "xstation", serials, gold) for s in station_keys()]


def _metadata_faults(rng, frame: pd.DataFrame, prod: np.ndarray, expected: dict) -> None:
    """Blank operator or instrument serial on a fraction of production rows."""
    for col, key in (("operator", "missing_operator"), ("instrument_serial", "missing_serial")):
        mask = prod & (rng.random(len(frame)) < ROW_FAULT_RATES[key])
        frame.loc[mask, col] = ""
        expected[f"{key}_rows"] += int(mask.sum())


def _cell_faults(rng, frame: pd.DataFrame, prod: np.ndarray, expected: dict) -> None:
    """Overwrite a disjoint random sample of production cells with sentinels, impossible values and ERR."""
    metrics = list(METRIC_NAMES)
    values = frame[metrics].to_numpy(dtype=float).round(4)
    cells = np.flatnonzero(np.repeat(prod, len(metrics)))
    counts = {k: int(round(r * len(cells))) for k, r in CELL_FAULT_RATES.items()}
    chosen = rng.choice(cells, size=sum(counts.values()), replace=False)
    parts = dict(zip(counts, np.split(chosen, np.cumsum(list(counts.values()))[:-1])))
    faulted = expected["faulted_cells_by_station"]
    for s, k in pd.Series(frame["station"].to_numpy()[chosen // len(metrics)]).value_counts().items():
        faulted[s] = faulted.get(s, 0) + int(k)
    fill = {"sentinel_neg": lambda c: -999.0, "sentinel_pos": lambda c: 9999.0,
            "impossible": lambda c: [IMPOSSIBLE_VALUES[metrics[j]] for j in c]}
    for key, flat in parts.items():
        expected[f"{key}_cells"] += len(flat)
        r, c = np.divmod(flat, len(metrics))
        if key in fill:
            values[r, c] = fill[key](c)
    err_r, err_c = np.divmod(parts["type_error"], len(metrics))
    for j, name in enumerate(metrics):
        col = values[:, j].astype(object)
        col[err_r[err_c == j]] = "ERR"
        frame[name] = col


def _name_variants(rng, frame: pd.DataFrame, expected: dict) -> None:
    """Rewrite a fraction of station names in the documented inconsistent spellings."""
    variant = rng.random(len(frame)) < ROW_FAULT_RATES["name_variant"]
    patterns = rng.choice(NAME_VARIANTS, size=len(frame))
    numbers = frame["station"].str[-2:].astype(int).to_numpy()
    frame["station"] = [p.format(n=k) if v else s for s, v, p, k in zip(frame["station"], variant, patterns, numbers)]
    expected["variant_rows"] += int(variant.sum())


def _inject_faults(rng: np.random.Generator, frame: pd.DataFrame, expected: dict) -> pd.DataFrame:
    """Apply the documented data-quality faults; study rows only get name variants."""
    prod = (frame["study"] == "production").to_numpy()
    _metadata_faults(rng, frame, prod, expected)
    _cell_faults(rng, frame, prod, expected)
    for s, k in frame["station"].value_counts().items():
        expected["accepted_by_station"][s] = expected["accepted_by_station"].get(s, 0) + int(k) * len(METRIC_NAMES)
    _name_variants(rng, frame, expected)
    return frame


def _apply_schema_version(rng, frame: pd.DataFrame, version: str) -> pd.DataFrame:
    if version == "v1":
        return frame.rename(columns=V1_RENAMES)
    frame[UNEXPECTED_COLUMN] = rng.normal(35.0, 1.5, len(frame)).round(2)
    return frame


def _duplicate_rows(rng, frame: pd.DataFrame, expected: dict) -> pd.DataFrame:
    idx = rng.choice(len(frame), size=int(round(ROW_FAULT_RATES["duplicate"] * len(frame))), replace=False)
    expected["duplicate_rows"] += len(idx)
    return pd.concat([frame, frame.iloc[idx]], ignore_index=True)


def generate(cfg: GenConfig, out_dir: Path, ref_dir: Path = REFERENCE_DATA_DIR) -> dict:
    """Write one CSV batch per day plus manifest.json; return the manifest."""
    rng = np.random.default_rng(cfg.seed)
    serials = load_serials(ref_dir)
    batch_dir = out_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    gold = {m: rng.normal(s.target, s.unit_sd, GOLDEN_UNITS) for m, s in METRICS.items()}
    expected: dict = {
        "accepted_by_station": {}, "faulted_cells_by_station": {}, "missing_operator_rows": 0, "missing_serial_rows": 0,
        "sentinel_neg_cells": 0, "sentinel_pos_cells": 0, "impossible_cells": 0, "type_error_cells": 0,
        "variant_rows": 0, "duplicate_rows": 0, "v2_batches": 0,
    }
    batches = []
    for day in range(1, cfg.days + 1):
        blocks = _production_day(rng, cfg, day, serials)
        if day == cfg.grr_day:
            blocks += _grr_block(rng, cfg, gold, serials)
        if day == cfg.xstation_day:
            blocks += _xstation_block(rng, cfg, gold, serials)
        frame = _inject_faults(rng, pd.concat(blocks, ignore_index=True), expected)
        version = "v1" if day < cfg.schema_change_day else "v2"
        expected["v2_batches"] += version == "v2"
        frame = _duplicate_rows(rng, _apply_schema_version(rng, frame, version), expected)
        batch_id = "B" + cfg.date_of(day).strftime("%Y%m%d")
        frame.to_csv(batch_dir / f"{batch_id}.csv", index=False)
        batches.append({"batch_id": batch_id, "file": f"batches/{batch_id}.csv", "day": day,
                        "date": cfg.date_of(day).isoformat(), "source_version": version, "rows": len(frame)})
    for s, faulted in expected["faulted_cells_by_station"].items():
        expected["accepted_by_station"][s] -= faulted
    manifest = {
        "synthetic": True,
        "description": "Synthetic multi-station production test line data with injected faults. "
                       "No real product, instrument, site or person.",
        "config": {**asdict(cfg), "start_date": cfg.start_date.isoformat()},
        "injected": {
            "station_bias": {f"{s}|{m}": v for (s, m), v in STATION_BIAS.items()},
            "drift": {"station": DRIFT_STATION, "metric": DRIFT_METRIC, "start_day": cfg.drift_start_day,
                      "per_day": cfg.drift_per_day, "total": DRIFT_TOTAL},
            "shift": {"station": SHIFT_STATION, "metric": SHIFT_METRIC, "day": cfg.shift_day, "delta": SHIFT_DELTA},
            "grr": {"day": cfg.grr_day, "stations": list(GRR_STATIONS), "units": GRR_UNITS,
                    "operators": list(OPERATORS), "repeats": GRR_REPEATS},
            "xstation": {"day": cfg.xstation_day, "units": GOLDEN_UNITS, "reference_station": REFERENCE_STATION},
            "schema_change_day": cfg.schema_change_day, "v1_renames": V1_RENAMES, "unexpected_column": UNEXPECTED_COLUMN,
            "name_variants": list(NAME_VARIANTS), "row_fault_rates": ROW_FAULT_RATES, "cell_fault_rates": CELL_FAULT_RATES,
        },
        "expected": expected,
        "batches": batches,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
