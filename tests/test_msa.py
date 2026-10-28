import numpy as np
import pandas as pd
import pytest

from station_analytics import msa

# Hand-computed Gauge R&R data set: 2 parts x 2 operators x 2 repeats.
#   P1/A: 10, 12   P1/B: 11, 13   P2/A: 20, 22   P2/B: 21, 23
# Cell means 11, 12, 21, 22; part means 11.5, 21.5; operator means 16, 17; grand mean 16.5.
# SS_P = 2*2*((-5)^2 + 5^2) = 200 (df 1)     SS_O = 2*2*((-0.5)^2 + 0.5^2) = 2 (df 1)
# SS_PO = 0 (every interaction residual is 0, df 1)     SS_E = 4 cells * 2 = 8 (df 4)
# MS: part 200, operator 2, interaction 0, repeat 2.
# var_repeat = 2; var_inter = max(0, (0-2)/2) = 0; var_oper = (2-0)/(2*2) = 0.5; var_part = (200-0)/(2*2) = 50.
# GRR = 2.5, total = 52.5, %GRR = sqrt(2.5/52.5) = 21.82%; ndc = floor(1.41*sqrt(50/2.5)) = floor(6.31) = 6.
# With tolerance 60: %GRR_tol = 6*sqrt(2.5)/60 = 15.81%.
GRR_DF = pd.DataFrame({"unit_id": ["P1"] * 4 + ["P2"] * 4, "operator": ["A", "A", "B", "B"] * 2,
                       "value": [10, 12, 11, 13, 20, 22, 21, 23]})


def test_gauge_rr_variance_components_match_hand_computation():
    g = msa.gauge_rr(GRR_DF)
    assert (g.n_parts, g.n_operators, g.n_repeats) == (2, 2, 2)
    assert (g.ms_part, g.ms_operator, g.ms_interaction, g.ms_repeat) == (200.0, 2.0, 0.0, 2.0)
    assert (g.var_repeatability, g.var_interaction, g.var_operator, g.var_part) == (2.0, 0.0, 0.5, 50.0)
    assert g.var_reproducibility == 0.5 and g.var_grr == 2.5 and g.var_total == 52.5


def test_gauge_rr_ratios_and_distinct_categories():
    g = msa.gauge_rr(GRR_DF)
    assert g.pct_grr_total == pytest.approx(100 * np.sqrt(2.5 / 52.5))
    assert g.pct_grr_tolerance(60.0) == pytest.approx(100 * 6 * np.sqrt(2.5) / 60)
    assert g.ndc == 6
    assert msa.grr_verdict(g.pct_grr_total) == "marginal"
    assert msa.grr_verdict(5.0) == "acceptable" and msa.grr_verdict(31.0) == "unacceptable"


def test_gauge_rr_rejects_unbalanced_design():
    with pytest.raises(ValueError):
        msa.gauge_rr(GRR_DF.iloc[:-1])


def test_capability_within_sigma_from_ranges():
    # Repeating 0..4 gives every subgroup of 5 a range of 4, so sigma_w = 4/d2(5) = 4/2.326.
    c = msa.capability(np.tile([0, 1, 2, 3, 4], 20), lsl=-10, usl=14, subgroup_size=5)
    assert c.sigma_within == pytest.approx(4 / 2.326)
    assert c.cp == pytest.approx(24 * 2.326 / 24)
    assert c.cpk == pytest.approx(min(14 - 2, 2 + 10) * 2.326 / 12)


def test_capability_overall_sigma_matches_numpy():
    x = np.random.default_rng(0).normal(10, 2, 500)
    c = msa.capability(x, lsl=4, usl=18)
    s = x.std(ddof=1)
    assert c.pp == pytest.approx(14 / (6 * s)) and c.ppk == pytest.approx(min(18 - x.mean(), x.mean() - 4) / (3 * s))
    assert c.sigma_within == pytest.approx(2.0, rel=0.1)


@pytest.mark.parametrize("n, a2, d3, d4", [(2, 1.880, 0.0, 3.267), (3, 1.023, 0.0, 2.574), (4, 0.729, 0.0, 2.282), (5, 0.577, 0.0, 2.114)])
def test_control_chart_constants(n, a2, d3, d4):
    assert msa.CONTROL_CONSTANTS[n][:3] == (a2, d3, d4)


def test_xbar_r_limits():
    chart = msa.xbar_r_chart(np.tile([0, 1, 2, 3, 4], 10), subgroup_size=5)
    assert chart.xbar_cl == 2.0 and chart.r_cl == 4.0
    assert chart.xbar_ucl == pytest.approx(2 + 0.577 * 4) and chart.xbar_lcl == pytest.approx(2 - 0.577 * 4)
    assert chart.r_ucl == pytest.approx(2.114 * 4) and chart.r_lcl == 0.0


def test_imr_limits():
    chart = msa.imr_chart([1, 3, 1, 3, 1, 3])
    assert chart.mr_cl == 2.0 and chart.x_cl == 2.0
    assert chart.x_ucl == pytest.approx(2 + 2.66 * 2) and chart.mr_ucl == pytest.approx(3.267 * 2)


def shifted(seed=1, before=40, after=30, size=2.0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 1, before), rng.normal(size, 1, after)])


def test_ewma_detects_shift_within_tolerance():
    e = msa.ewma(shifted(), mu0=0.0, sigma=1.0)
    assert e.alarm_index is not None and 40 <= e.alarm_index <= 45
    assert e.ucl[-1] == pytest.approx(3 * np.sqrt(0.2 / 1.8), rel=1e-3)


def test_ewma_quiet_on_stable_data():
    assert msa.ewma(np.random.default_rng(2).normal(0, 1, 200), 0.0, 1.0).alarm_index is None


def test_cusum_detects_shift_and_direction():
    c = msa.cusum(shifted(), mu0=0.0, sigma=1.0)
    assert c.alarm_index is not None and 40 <= c.alarm_index <= 46 and c.direction == "up"
    assert msa.cusum(-shifted(), 0.0, 1.0).direction == "down"


def test_cusum_quiet_on_stable_data():
    assert msa.cusum(np.random.default_rng(3).normal(0, 1, 100), 0.0, 1.0).alarm_index is None


def test_change_point_estimate_is_accurate():
    cp = msa.change_point(shifted(size=4.0), min_segment=5)
    assert abs(cp.index - 40) <= 1 and cp.mean_after - cp.mean_before == pytest.approx(4.0, abs=0.6)


def test_ramp_fit_recovers_onset_and_slope():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.3, 60)
    x[30:] += 0.5 * np.arange(30)
    ramp = msa.ramp_fit(x, min_segment=5)
    assert abs(ramp.index - 30) <= 2 and ramp.slope == pytest.approx(0.5, abs=0.05)


def test_agreement_on_constructed_pair():
    ref = np.arange(1.0, 11.0)
    test = 2 * ref + 1 + np.array([0.5, -0.5] * 5)
    a = msa.agreement(ref, test)
    assert a.slope == pytest.approx(2.0, abs=0.05) and a.intercept == pytest.approx(1.0, abs=0.3)
    diff = test - ref
    assert a.bias == pytest.approx(diff.mean()) and a.loa_high == pytest.approx(diff.mean() + 1.96 * diff.std(ddof=1))
    assert a.r > 0.99


def test_drift_summary_classifies_shift_drift_and_stable():
    rng = np.random.default_rng(5)
    stable = rng.normal(0, 1, 60)
    ramp = stable.copy()
    ramp[40:] += 0.6 * np.arange(20)
    step = stable.copy()
    step[40:] += 5
    assert msa.drift_summary(stable, 20, min_segment=6).pattern == "stable"
    s_shift, s_drift = msa.drift_summary(step, 20, min_segment=6), msa.drift_summary(ramp, 20, min_segment=6)
    assert (s_shift.pattern, s_shift.change_index) == ("shift", 40) and s_shift.flagged
    assert s_drift.pattern == "drift" and abs(s_drift.change_index - 40) <= 2 and s_drift.slope_per_step > 0
