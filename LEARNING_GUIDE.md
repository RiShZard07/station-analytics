# Learning guide: twelve things to be able to explain

Each item is a question an interviewer could ask about this project, with the answer the
code actually implements. Read the referenced function before answering.

1. **What are the variance components in a Gauge R&R study and what do the %GRR
   thresholds mean?** (`msa.gauge_rr`) Total variation splits into part-to-part
   variation (what you want to see) and measurement variation. Measurement variation is
   repeatability (same operator, same part, repeat measurements: the error mean square)
   plus reproducibility (operator effect plus part-by-operator interaction). %GRR of total
   variation is sigma_GRR over sigma_total; under 10% the gauge is fine, 10 to 30% is
   marginal and depends on cost and risk, over 30% the gauge cannot separate parts. %GRR
   of tolerance asks the same question against the spec width, which is what matters
   for pass/fail decisions. Number of distinct categories (1.41 sigma_part / sigma_GRR)
   says how many groups of parts the gauge can tell apart; 5 or more is the usual bar.

2. **Cp versus Cpk versus Ppk.** (`msa.capability`) Cp compares the spec width with six
   within-subgroup sigmas and ignores centring. Cpk uses the nearer spec limit, so a
   centred process has Cp equal to Cpk and an off-centre process has Cpk lower. Pp and
   Ppk use the overall standard deviation, which includes between-subgroup drift and
   shifts, so Ppk below Cpk is a sign the process is not stable over time. In the demo,
   the drifting station shows Ppk falling below Cpk for luminance.

3. **X-bar/R versus I-MR.** (`msa.xbar_r_chart`, `msa.imr_chart`) X-bar/R needs rational
   subgroups (several units measured close together); the R chart watches within-subgroup
   spread and the X-bar chart watches the subgroup mean with limits A2 Rbar. I-MR is for
   one value at a time, using the moving range of consecutive values as the spread
   estimate; it is less sensitive to small shifts because each point is a single noisy
   value. The report uses daily means as subgroups for the X-bar view and the first 60
   production units for the I-MR view.

4. **EWMA versus CUSUM, and the k and h trade-off.** (`msa.ewma`, `msa.cusum`) Both
   accumulate evidence across points so they catch small sustained shifts that a
   Shewhart chart misses. EWMA is a weighted average with memory controlled by lambda;
   smaller lambda means more memory and more sensitivity to small shifts. CUSUM adds the
   standardised deviation minus a reference k, resets at zero, and alarms above h. k is
   half the shift size you want to detect quickly (0.5 for a one sigma shift); h sets
   the false alarm rate: h 5 with k 0.5 gives an average run length of roughly 465
   points in control, so with 46 series of 90 days you expect several marginal alarms,
   which is exactly what the drift table shows.

5. **Change point versus drift.** (`msa.change_point`, `msa.ramp_fit`) A change point is
   a step: two levels, one switch day, found here by the least-squares split that is
   equivalent to the argmax of the scaled cumulative sum. A drift is a ramp: a slope that
   starts on some day. The code fits both and keeps the one with the lower residual sum
   of squares, which is a fair comparison because both models have the same number of
   free parameters. The recommended actions differ: a step points at an event (setup
   change, fixture swap), a ramp points at wear or an instrument losing calibration.

6. **Why does the drift flag need more than an alarm?** (`msa.drift_summary`) Alarms
   happen on stable series at the rate set by h. The flag additionally requires the
   current offset (mean of the last few days minus baseline) to be at least 2 sigma of a
   daily mean. This keeps marginal alarms visible in the table but out of the alert
   feed. An earlier version used the fitted ramp size instead and produced false flags
   when a steep ramp was fitted to the last three noisy days; the fix was a minimum
   segment length that scales with the horizon and a magnitude rule that does not
   depend on the fit.

7. **Why canonicalise station names at ingest and never in the report?**
   (`ingest.canonical_station`) Every downstream query groups by station. If the report
   fixed names, each report would need the same fix, the warehouse would still be wrong
   for anyone else, and two spellings of one station would split its history. Fixing at
   the boundary once means one tested parser, one place to extend, and a fact table that
   SQL users can trust. Rows the parser cannot map are quarantined with the raw text so
   nothing is silently lost.

8. **How is schema drift detected and why are unknown columns quarantined rather than
   dropped?** (`ingest.SchemaContract`, `ingest._check_columns`) Each source version has a
   declared contract: column names, types and a rename map for known changes. Columns
   not in the contract are moved to the quarantine table with their values and a batch
   id, so the data is kept and the drift is visible in the ingest summary. Dropping
   would hide the signal that the source changed; failing the whole batch would lose
   good data. A missing declared column does fail the batch, because the contract is
   then broken rather than extended.

9. **What makes the batch load idempotent?** (`ingest.ingest_batch`) The batch id is the
   primary key of `ingest_log`, and the fact rows, quarantine rows and log row are
   written in one transaction. A batch that is in the log is skipped before its file is
   even read. If the transaction fails part way, nothing is logged and the next run
   repeats the batch. `demo --check` runs the ingest twice and asserts the fact count is
   unchanged.

10. **What does the reconciliation test prove, and what does it not prove?**
    (`__main__.run_checks`, `tests/test_pipeline.py`) It proves that `metrics.json`, the
    capability means and the number printed in the report equal direct SQL aggregates
    and an independent pandas recomputation over the same warehouse, so the report is
    not lying about the warehouse. It does not prove the warehouse is right about the
    world: that is what the manifest checks do for synthetic data, and what nothing can
    do for real data without an external reference. It also does not check every number
    in the report, only the ones listed.

11. **What is Bland-Altman style agreement adding to a correlation?**
    (`msa.agreement`) Correlation only says two stations rank units the same way. Bias
    and limits of agreement say how far apart their readings are in the metric's units,
    which is what a transfer decision needs. The regression slope against the lab
    reference shows whether a station reads proportionally high or low rather than by a
    constant offset.

12. **What would the alerts job need to become real?** (`alerts.run_alerts`) A scheduler
    to run ingest, analysis and alerts on each new batch; a notifier that routes by
    severity and de-duplicates repeated alerts; an acknowledgement state so an alert
    stops firing once someone owns it; a calibration register with certificates and
    history rather than a CSV; thresholds agreed with the people who own the stations;
    and a feedback loop recording whether each alert was a true problem, so k, h and the
    flag rule can be tuned on evidence.
