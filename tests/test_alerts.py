import json
from datetime import date

import pandas as pd
import pytest

from station_analytics.alerts import build_alerts, calibration_status, load_instruments, run_alerts
from station_analytics.specs import REFERENCE_DATA_DIR

AS_OF = date(2026, 4, 4)


def instrument(days_until_due: int) -> pd.DataFrame:
    # Certified 100 days before as_of with an interval that lands the due date at as_of + days_until_due.
    return pd.DataFrame([{"instrument_id": "INST-X", "station": "STA-01", "serial": "S", "model": "M",
                          "last_certified": date(2025, 12, 25), "interval_days": 100 + days_until_due, "service_vendor": "v"}])


@pytest.mark.parametrize("days, status", [(-1, "overdue"), (0, "due_soon"), (30, "due_soon"), (31, "ok")])
def test_calibration_status_boundaries(days, status):
    out = calibration_status(instrument(days), AS_OF, due_within_days=30)
    assert out.loc[0, "status"] == status and out.loc[0, "days_until_due"] == days


def test_build_alerts_severity_kinds_and_actions():
    status = calibration_status(pd.concat([instrument(-5), instrument(10).assign(instrument_id="INST-Y", station="STA-08")]), AS_OF)
    drift = pd.DataFrame([
        {"station_key": "STA-01", "metric": "luminance_nits", "pattern": "drift", "change_day": 50, "magnitude_sigma": 12.0, "slope_per_day": 0.5, "flagged": True},
        {"station_key": "STA-08", "metric": "mtf_50", "pattern": "shift", "change_day": 60, "magnitude_sigma": 7.0, "slope_per_day": 0.0, "flagged": True},
        {"station_key": "STA-02", "metric": "mtf_50", "pattern": "stable", "change_day": None, "magnitude_sigma": 0.3, "slope_per_day": 0.0, "flagged": False},
    ])
    alerts = build_alerts(status, drift)
    assert [(a.severity, a.kind) for a in alerts] == [("critical", "calibration_overdue"), ("critical", "shift"),
                                                      ("warning", "drift"), ("warning", "calibration_due_soon")]
    drift_alert = next(a for a in alerts if a.kind == "drift")
    assert "overdue for calibration" in drift_alert.recommended_action and "day 50" in drift_alert.message
    assert all(a.recommended_action for a in alerts)


def test_reference_register_as_of_last_measurement(demo_dir):
    status = calibration_status(load_instruments(REFERENCE_DATA_DIR / "instruments.csv"), AS_OF)
    assert status["status"].value_counts().to_dict() == {"ok": 8, "overdue": 2, "due_soon": 2}
    payload = json.loads((demo_dir / "alerts.json").read_text())
    assert payload["synthetic"] is True and payload["delivery"].startswith("none")
    assert {a["kind"] for a in payload["alerts"]} >= {"drift", "shift", "calibration_overdue"}


def test_run_alerts_respects_explicit_as_of(demo_dir, tmp_path):
    payload = run_alerts(REFERENCE_DATA_DIR / "instruments.csv", demo_dir / "analysis", tmp_path / "a.json", as_of=date(2027, 1, 1))
    assert payload["instrument_status"] == {"overdue": 12} and payload["counts"]["critical"] >= 12
