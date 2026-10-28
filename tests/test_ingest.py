import json

import pytest

from conftest import base_row, write_batch
from station_analytics.ingest import SchemaError, StationNameError, canonical_station, ingest_batch


def facts(conn):
    return conn.execute("SELECT station_key, unit_id, metric, value, operator, quality_flags, source_version "
                        "FROM fact_measurement ORDER BY metric").fetchall()


def quarantine(conn):
    return conn.execute("SELECT reason, detail, payload FROM quarantine").fetchall()


@pytest.mark.parametrize("raw, expected", [
    ("STA-07", "STA-07"), ("sta07", "STA-07"), ("Station 7 ", "STA-07"), ("STA_07", "STA-07"),
    (" sta-7 ", "STA-07"), ("STATION_12", "STA-12"), ("7", "STA-07"), ("st 3", "STA-03"),
])
def test_canonical_station_accepts_known_variants(raw, expected):
    assert canonical_station(raw) == expected


@pytest.mark.parametrize("raw", ["", "STA", "STA-", "STA-7A", "STA-100", "station seven", None, float("nan"), "STA-00"])
def test_canonical_station_rejects_garbage(raw):
    with pytest.raises(StationNameError):
        canonical_station(raw)


def test_v1_rename_is_mapped_to_canonical_metric(conn, tmp_path):
    row = base_row()
    row["lum_nits"] = row.pop("luminance_nits")
    result = ingest_batch(conn, write_batch(tmp_path / "b.csv", [row]), "B1", "v1")
    metrics = {f[2] for f in facts(conn)}
    assert result.accepted == 4 and "luminance_nits" in metrics and "lum_nits" not in metrics


def test_unknown_column_is_quarantined_not_dropped(conn, tmp_path):
    rows = [base_row(fixture_temp_c=35.1), base_row(unit_id="U2", fixture_temp_c=36.2)]
    result = ingest_batch(conn, write_batch(tmp_path / "b.csv", rows), "B1", "v2")
    assert result.accepted == 8 and result.reasons == {"unknown_column": 1}
    reason, detail, payload = quarantine(conn)[0]
    assert (reason, detail) == ("unknown_column", "fixture_temp_c") and json.loads(payload)["values"] == ["35.1", "36.2"]


def test_missing_declared_column_refuses_batch(conn, tmp_path):
    row = base_row()
    del row["mtf_50"]
    with pytest.raises(SchemaError):
        ingest_batch(conn, write_batch(tmp_path / "b.csv", [row]), "B1", "v2")
    assert conn.execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0] == 0


def test_unknown_source_version_raises(conn, tmp_path):
    with pytest.raises(SchemaError):
        ingest_batch(conn, write_batch(tmp_path / "b.csv", [base_row()]), "B1", "v9")


def test_type_mismatch_quarantines_only_that_value(conn, tmp_path):
    result = ingest_batch(conn, write_batch(tmp_path / "b.csv", [base_row(mtf_50="ERR")]), "B1", "v2")
    assert result.accepted == 3 and result.reasons == {"type_mismatch": 1}
    assert json.loads(quarantine(conn)[0][2])["raw_value"] == "ERR"


def test_sentinel_range_and_missing_values(conn, tmp_path):
    rows = [base_row(luminance_nits=-999), base_row(unit_id="U2", color_delta_e=9999),
            base_row(unit_id="U3", binocular_misalignment_arcmin=-3.0), base_row(unit_id="U4", mtf_50="")]
    result = ingest_batch(conn, write_batch(tmp_path / "b.csv", rows), "B1", "v2")
    assert result.accepted == 12
    assert result.reasons == {"sentinel_value": 2, "out_of_range": 1, "missing_value": 1}


def test_missing_metadata_is_kept_and_flagged(conn, tmp_path):
    rows = [base_row(operator=""), base_row(unit_id="U2", instrument_serial=""), base_row(unit_id="U3", operator="", instrument_serial="")]
    result = ingest_batch(conn, write_batch(tmp_path / "b.csv", rows), "B1", "v2")
    assert result.accepted == 12 and result.quarantined == 0
    flags = dict(conn.execute("SELECT unit_id, quality_flags FROM fact_measurement GROUP BY unit_id").fetchall())
    assert flags == {"U1": "missing_operator", "U2": "missing_serial", "U3": "missing_operator;missing_serial"}
    assert conn.execute("SELECT operator FROM fact_measurement WHERE unit_id='U1'").fetchone()[0] is None


def test_exact_duplicates_removed_first_wins(conn, tmp_path):
    rows = [base_row(luminance_nits=501.0), base_row(luminance_nits=501.0), base_row(unit_id="U2")]
    result = ingest_batch(conn, write_batch(tmp_path / "b.csv", rows), "B1", "v2")
    assert result.rows_in == 3 and result.accepted == 8 and result.reasons == {"exact_duplicate": 1}
    assert conn.execute("SELECT COUNT(*) FROM fact_measurement WHERE unit_id='U1'").fetchone()[0] == 4


def test_bad_or_unknown_station_quarantines_whole_row(conn, tmp_path):
    rows = [base_row(station="garbage"), base_row(unit_id="U2", station="STA-99"), base_row(unit_id="U3", station="sta_07")]
    result = ingest_batch(conn, write_batch(tmp_path / "b.csv", rows), "B1", "v2")
    assert result.accepted == 4 and result.reasons == {"bad_station": 1, "unknown_station": 1}
    assert {f[0] for f in facts(conn)} == {"STA-07"}


def test_reingest_same_batch_is_noop(conn, tmp_path):
    path = write_batch(tmp_path / "b.csv", [base_row(), base_row(unit_id="U2", mtf_50=-999)])
    first = ingest_batch(conn, path, "B1", "v2")
    second = ingest_batch(conn, path, "B1", "v2")
    assert second.skipped and not first.skipped
    assert conn.execute("SELECT COUNT(*) FROM fact_measurement").fetchone()[0] == 7
    assert conn.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0] == 1


def test_ingest_log_matches_batch_result(conn, tmp_path):
    rows = [base_row(), base_row(), base_row(unit_id="U2", luminance_nits="n/a", fixture_temp_c=1)]
    result = ingest_batch(conn, write_batch(tmp_path / "b.csv", rows), "B1", "v2")
    log = conn.execute("SELECT rows_in, accepted, quarantined, source_version FROM ingest_log").fetchone()
    assert log == (3, 7, 3, "v2") == (result.rows_in, result.accepted, result.quarantined, result.source_version)
    assert conn.execute("SELECT unit_type, COUNT(*) FROM dim_unit GROUP BY 1").fetchall() == [("production", 2)]
