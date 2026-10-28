# Verified results

All numbers below come from synthetic data produced by `station_analytics/generate.py`
with seed 42. They describe the generator and the pipeline, not any real test line.

Environment used: Linux, Python 3.13, numpy 2.5, pandas 3.0, pytest 9.2. The CI
workflow pins Python 3.12 with the loose requirement ranges in `requirements.txt`.

## Commands

```
pip install -r requirements.txt
python -m pytest -q
python -m station_analytics demo --check
```

## Test run

```
61 passed in 3.89s
```

The suite is fully offline. It includes one reduced end-to-end run
(`demo --check --days 30 --rows 40000 --seed 7`) shared by the pipeline tests. The
regression gate was also run by hand at 30 days with seeds 1 to 8 and at 90 days with
seeds 1 to 5 and 42; all passed 10 of 10.

## demo --check output (seed 42, 90 days, about 8 seconds wall clock)

```
generated 90 synthetic batches (90 days, 45 units per station-day) in output/data
ingest: 90 batches loaded, 0 skipped, 49,592 source rows, 196,650 measurements accepted, 938 quarantine records
analysis: 196,650 fact rows, 2 flagged series, worst %GRR 44.93, min Cpk 1.064 (STA-02|binocular_misalignment_arcmin)
alerts: {'critical': 3, 'warning': 3} as of 2026-04-04
report: output/report.html

demo --check (synthetic data, injected faults recovered?)
  PASS  drifting station flagged: STA-04 luminance_nits pattern=drift onset day 49 vs injected 50 (tolerance 10 days), 12.1 sigma, CUSUM alarm day 56
  PASS  change point within tolerance: STA-08 mtf_50 estimated day 60 vs injected 60 (tolerance 3 days), 7.2 sigma
  PASS  only injected series flagged: flagged [('STA-04', 'luminance_nits'), ('STA-08', 'mtf_50')]
  PASS  station names canonicalised: 6,006 rows had variant spellings; per-station counts match the generator; 0 rows quarantined for station names
  PASS  renamed column mapped: 29,785 v1 lum_nits values stored as luminance_nits
  PASS  unexpected column quarantined: 36 quarantine records for fixture_temp_c across 36 v2 batches
  PASS  exact duplicates removed: 272 duplicates quarantined, 0 duplicate groups left in the fact table
  PASS  sentinel, impossible and non-numeric values quarantined: sentinel 450, out_of_range 90, type_mismatch 90
  PASS  report reconciles with direct SQL: 196,650 fact rows = ingest_log total = metrics.json; 48 station-metric means and 48 capability means match SQL; count appears in report.html
  PASS  re-ingest is idempotent: 90 batches skipped on second load, fact rows unchanged at 196,650
10 of 10 checks passed
```

## Key numbers (synthetic)

- Quarantine records by reason: sentinel_value 450, exact_duplicate 272, type_mismatch 90,
  out_of_range 90, unknown_column 36. Rows kept with a quality flag: 1,402 source rows
  missing an operator and 971 missing an instrument serial.
- Gauge R&R (10 golden units x 3 operators x 3 repeats on STA-01, STA-06, STA-11,
  STA-12): colour difference 9.1 to 10.2 %GRR (acceptable to marginal, 13 to 15 distinct
  categories), luminance 11.3 to 12.4% (marginal), misalignment 30 to 35% and MTF50 37 to
  45% (unacceptable, 2 to 4 distinct categories). Verdicts: 3 acceptable, 5 marginal,
  8 unacceptable.
- Capability: minimum Cpk 1.064 (STA-02 misalignment, limited by the zero lower bound).
  The drifting station shows the expected gap between within and overall sigma:
  STA-04 luminance Cpk 1.503 versus Ppk 1.261; STA-08 MTF50 Cpk 1.132 versus Ppk 0.991.
- Drift summary: 2 of 48 series flagged (the two injected ones). 8 of the 46 unflagged
  series had a marginal CUSUM or EWMA alarm below the 2 sigma flag rule, which is
  consistent with h 5 over 90 points.
- Alerts as of 2026-04-04: 2 instruments overdue (INST-04 on STA-04, INST-09 on STA-09),
  2 due within 30 days (INST-02, INST-05), 1 shift alert, 1 drift alert; the STA-04 drift
  alert notes that its instrument is also overdue.

## Size

1,965 lines of Python across the package (1,533) and tests (432). The warehouse for the
default run is about 38 MB on disk and the report is about 63 KB with four inline SVG
charts and no external assets.
