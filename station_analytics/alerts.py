"""Calibration system of record and the alert feed a scheduled job would emit.

Inputs: the calibration register (data/instruments.csv) and the drift summary written by
the analysis. Output: alerts.json with a severity and a one-line recommended action per
alert. There is no email, chat or ticketing integration; a real deployment would hand
this file to a notifier.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .specs import METRICS

DEFAULT_DUE_WITHIN_DAYS = 30


@dataclass(frozen=True)
class Alert:
    severity: str  # critical or warning
    kind: str  # calibration_overdue, calibration_due_soon, shift, drift
    station_key: str
    subject: str  # instrument id or metric name
    message: str
    recommended_action: str


def load_instruments(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"interval_days": int})
    frame["last_certified"] = pd.to_datetime(frame["last_certified"]).dt.date
    return frame


def calibration_status(instruments: pd.DataFrame, as_of: date, due_within_days: int = DEFAULT_DUE_WITHIN_DAYS) -> pd.DataFrame:
    """due_date = last_certified + interval_days. Negative days_until_due is overdue;
    0..due_within_days is due_soon; anything later is ok."""
    out = instruments.copy()
    out["due_date"] = [d + timedelta(days=int(n)) for d, n in zip(out["last_certified"], out["interval_days"])]
    out["days_until_due"] = [(d - as_of).days for d in out["due_date"]]
    out["status"] = ["overdue" if n < 0 else "due_soon" if n <= due_within_days else "ok" for n in out["days_until_due"]]
    return out


def _calibration_alerts(status: pd.DataFrame) -> list[Alert]:
    alerts = []
    for r in status.itertuples():
        if r.status == "overdue":
            alerts.append(Alert("critical", "calibration_overdue", r.station, r.instrument_id,
                                f"{r.instrument_id} on {r.station} is {-r.days_until_due} days past its certification due date {r.due_date}.",
                                f"Hold or quarantine {r.station} results until {r.instrument_id} is recertified by {r.service_vendor}."))
        elif r.status == "due_soon":
            alerts.append(Alert("warning", "calibration_due_soon", r.station, r.instrument_id,
                                f"{r.instrument_id} on {r.station} is due for certification in {r.days_until_due} days ({r.due_date}).",
                                f"Book recertification of {r.instrument_id} with {r.service_vendor} before {r.due_date}."))
    return alerts


def _drift_alerts(drift: pd.DataFrame, status: pd.DataFrame) -> list[Alert]:
    overdue = set(status.loc[status["status"] == "overdue", "station"])
    alerts = []
    for r in drift[drift["flagged"]].itertuples():
        unit, day = METRICS[r.metric].unit, int(r.change_day)
        extra = " The station's instrument is also overdue for calibration; recalibrate first." if r.station_key in overdue else ""
        if r.pattern == "shift":
            alerts.append(Alert("critical", "shift", r.station_key, r.metric,
                                f"Mean shift of {r.magnitude_sigma:.1f} sigma in {r.metric} on {r.station_key} from day {day}.",
                                f"Hold {r.station_key} {r.metric} results from day {day}; re-measure golden units and look for a "
                                f"setup or fixture change on that day.{extra}"))
        else:
            alerts.append(Alert("warning", "drift", r.station_key, r.metric,
                                f"Drift of {r.slope_per_day:+.3g} {unit}/day in {r.metric} on {r.station_key} since day {day} "
                                f"({r.magnitude_sigma:.1f} sigma so far).",
                                f"Re-verify {r.station_key} against golden units and schedule recalibration.{extra}"))
    return alerts


def build_alerts(status: pd.DataFrame, drift: pd.DataFrame) -> list[Alert]:
    rank = {"critical": 0, "warning": 1}
    alerts = _calibration_alerts(status) + _drift_alerts(drift, status)
    return sorted(alerts, key=lambda a: (rank[a.severity], a.station_key, a.kind))


def run_alerts(instruments_path: Path, analysis_dir: Path, out_path: Path, as_of: date | None = None,
               due_within_days: int = DEFAULT_DUE_WITHIN_DAYS) -> dict:
    """Combine the calibration register with the drift summary and write alerts.json."""
    if as_of is None:
        metrics = json.loads((analysis_dir / "metrics.json").read_text())
        as_of = date.fromisoformat(metrics["last_measured_at"][:10])
    status = calibration_status(load_instruments(instruments_path), as_of, due_within_days)
    drift = pd.read_csv(analysis_dir / "drift_summary.csv")
    alerts = build_alerts(status, drift)
    payload = {
        "synthetic": True,
        "as_of": as_of.isoformat(),
        "due_within_days": due_within_days,
        "counts": {s: sum(a.severity == s for a in alerts) for s in ("critical", "warning")},
        "instrument_status": {k: int(v) for k, v in status["status"].value_counts().sort_index().items()},
        "alerts": [asdict(a) for a in alerts],
        "delivery": "none; this file is the job output and no email, chat or ticket integration exists",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload
