"""Measurement systems analysis and statistical process control in plain numpy and pandas.

Every function documents the formula it implements. Nothing here knows about the
warehouse; callers pass arrays or small DataFrames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Shewhart constants by subgroup size n: (A2, D3, D4, d2).
CONTROL_CONSTANTS: dict[int, tuple[float, float, float, float]] = {
    2: (1.880, 0.0, 3.267, 1.128), 3: (1.023, 0.0, 2.574, 1.693), 4: (0.729, 0.0, 2.282, 2.059),
    5: (0.577, 0.0, 2.114, 2.326), 6: (0.483, 0.0, 2.004, 2.534), 7: (0.419, 0.076, 1.924, 2.704),
    8: (0.373, 0.136, 1.864, 2.847), 9: (0.337, 0.184, 1.816, 2.970), 10: (0.308, 0.223, 1.777, 3.078),
}


@dataclass(frozen=True)
class GaugeRR:
    """Variance components of a crossed Gauge R&R study (ANOVA method)."""

    n_parts: int
    n_operators: int
    n_repeats: int
    ms_part: float
    ms_operator: float
    ms_interaction: float
    ms_repeat: float
    var_repeatability: float
    var_operator: float
    var_interaction: float
    var_part: float

    @property
    def var_reproducibility(self) -> float:
        return self.var_operator + self.var_interaction

    @property
    def var_grr(self) -> float:
        return self.var_repeatability + self.var_reproducibility

    @property
    def var_total(self) -> float:
        return self.var_grr + self.var_part

    @property
    def pct_grr_total(self) -> float:
        """%GRR of total variation: 100 * sigma_GRR / sigma_total (the 6-sigma widths cancel)."""
        return 100.0 * math.sqrt(self.var_grr / self.var_total) if self.var_total > 0 else float("nan")

    def pct_grr_tolerance(self, tolerance: float) -> float:
        """%GRR of tolerance: 100 * 6 sigma_GRR / (USL - LSL)."""
        return 100.0 * 6.0 * math.sqrt(self.var_grr) / tolerance

    @property
    def ndc(self) -> int:
        """Number of distinct categories: floor(1.41 * sigma_part / sigma_GRR)."""
        return int(math.floor(1.41 * math.sqrt(self.var_part / self.var_grr))) if self.var_grr > 0 else 0


def grr_verdict(pct_grr: float) -> str:
    """Common rule of thumb: under 10% acceptable, 10 to 30% marginal, over 30% unacceptable."""
    return "acceptable" if pct_grr < 10 else "marginal" if pct_grr <= 30 else "unacceptable"


def gauge_rr(df: pd.DataFrame, part_col: str = "unit_id", operator_col: str = "operator", value_col: str = "value") -> GaugeRR:
    """ANOVA Gauge R&R for a balanced crossed design with n parts, k operators, r repeats.

    SS_P  = k r sum_i (ybar_i.. - ybar)^2                   df n-1
    SS_O  = n r sum_j (ybar_.j. - ybar)^2                   df k-1
    SS_PO = r sum_ij (ybar_ij. - ybar_i.. - ybar_.j. + ybar)^2   df (n-1)(k-1)
    SS_E  = sum_ijk (y_ijk - ybar_ij.)^2                    df n k (r-1)
    Variance components (negative estimates are clipped to zero):
    repeatability = MS_E; interaction = (MS_PO - MS_E)/r; operator = (MS_O - MS_PO)/(n r);
    part = (MS_P - MS_PO)/(k r). The interaction term is always kept, never pooled.
    """
    cell = df.groupby([part_col, operator_col])[value_col].agg(["mean", "count", "var"])
    counts = cell["count"].unique()
    if len(counts) != 1 or counts[0] < 2:
        raise ValueError("gauge_rr needs a balanced design with at least two repeats per part-operator cell")
    means = cell["mean"].unstack()
    if means.isna().any().any():
        raise ValueError("gauge_rr needs every part measured by every operator")
    m, r = means.to_numpy(), int(counts[0])
    n, k = m.shape
    grand, part_means, op_means = m.mean(), m.mean(axis=1), m.mean(axis=0)
    ss_p = k * r * ((part_means - grand) ** 2).sum()
    ss_o = n * r * ((op_means - grand) ** 2).sum()
    ss_po = r * ((m - part_means[:, None] - op_means[None, :] + grand) ** 2).sum()
    ss_e = float((cell["var"] * (r - 1)).sum())
    ms_p, ms_o, ms_po = float(ss_p / (n - 1)), float(ss_o / (k - 1)), float(ss_po / ((n - 1) * (k - 1)))
    ms_e = ss_e / (n * k * (r - 1))
    return GaugeRR(n, k, r, ms_p, ms_o, ms_po, ms_e, ms_e, max(0.0, (ms_o - ms_po) / (n * r)),
                   max(0.0, (ms_po - ms_e) / r), max(0.0, (ms_p - ms_po) / (k * r)))


def _subgroups(values: np.ndarray, size: int) -> np.ndarray:
    full = len(values) // size * size
    if full < 2 * size:
        raise ValueError("need at least two complete subgroups")
    return values[:full].reshape(-1, size)


def sigma_within(values: np.ndarray, subgroup_size: int = 5) -> float:
    """Within-subgroup sigma: Rbar/d2 over consecutive subgroups, or MRbar/1.128 for size 1."""
    if subgroup_size == 1:
        return float(np.abs(np.diff(values)).mean() / CONTROL_CONSTANTS[2][3])
    groups = _subgroups(values, subgroup_size)
    return float((groups.max(axis=1) - groups.min(axis=1)).mean() / CONTROL_CONSTANTS[subgroup_size][3])


@dataclass(frozen=True)
class Capability:
    n: int
    mean: float
    sigma_within: float
    sigma_overall: float
    cp: float
    cpk: float
    pp: float
    ppk: float


def capability(values, lsl: float, usl: float, subgroup_size: int = 5) -> Capability:
    """Cp = (USL-LSL)/(6 sigma_w), Cpk = min(USL-mean, mean-LSL)/(3 sigma_w) with sigma_w from
    sigma_within(); Pp and Ppk use the same formulas with the overall sample sd (ddof=1)."""
    x = np.asarray(values, dtype=float)
    mean, s_overall, s_within = float(x.mean()), float(x.std(ddof=1)), sigma_within(x, subgroup_size)
    nearest = min(usl - mean, mean - lsl)
    return Capability(len(x), mean, s_within, s_overall, (usl - lsl) / (6 * s_within), nearest / (3 * s_within),
                      (usl - lsl) / (6 * s_overall), nearest / (3 * s_overall))


@dataclass(frozen=True)
class XbarR:
    xbar: np.ndarray
    r: np.ndarray
    xbar_cl: float
    xbar_ucl: float
    xbar_lcl: float
    r_cl: float
    r_ucl: float
    r_lcl: float


def xbar_r_chart(values, subgroup_size: int = 5) -> XbarR:
    """Consecutive subgroups. Xbar limits: xbarbar +/- A2 Rbar. R limits: D4 Rbar and D3 Rbar."""
    groups = _subgroups(np.asarray(values, dtype=float), subgroup_size)
    a2, d3, d4, _ = CONTROL_CONSTANTS[subgroup_size]
    xbar, r = groups.mean(axis=1), groups.max(axis=1) - groups.min(axis=1)
    cl, rbar = float(xbar.mean()), float(r.mean())
    return XbarR(xbar, r, cl, cl + a2 * rbar, cl - a2 * rbar, rbar, d4 * rbar, d3 * rbar)


@dataclass(frozen=True)
class IMR:
    x: np.ndarray
    mr: np.ndarray
    x_cl: float
    x_ucl: float
    x_lcl: float
    mr_cl: float
    mr_ucl: float


def imr_chart(values) -> IMR:
    """Individuals: xbar +/- 2.66 MRbar (2.66 = 3/d2 for n=2). Moving range: UCL 3.267 MRbar, LCL 0."""
    x = np.asarray(values, dtype=float)
    mr = np.abs(np.diff(x))
    cl, mrbar = float(x.mean()), float(mr.mean())
    return IMR(x, mr, cl, cl + 2.66 * mrbar, cl - 2.66 * mrbar, mrbar, 3.267 * mrbar)


@dataclass(frozen=True)
class EWMAResult:
    z: np.ndarray
    ucl: np.ndarray
    lcl: np.ndarray
    alarm_index: int | None


def ewma(values, mu0: float, sigma: float, lam: float = 0.2, L: float = 3.0) -> EWMAResult:
    """z_i = lam x_i + (1-lam) z_{i-1}, z_0 = mu0.
    Limits: mu0 +/- L sigma sqrt(lam/(2-lam) (1-(1-lam)^(2i))); alarm at the first point outside."""
    x = np.asarray(values, dtype=float)
    z = np.empty(len(x))
    prev = mu0
    for i, xi in enumerate(x):
        prev = z[i] = lam * xi + (1 - lam) * prev
    i = np.arange(1, len(x) + 1)
    half = L * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))
    out = np.flatnonzero((z > mu0 + half) | (z < mu0 - half))
    return EWMAResult(z, mu0 + half, mu0 - half, int(out[0]) if len(out) else None)


@dataclass(frozen=True)
class CusumResult:
    c_plus: np.ndarray
    c_minus: np.ndarray
    alarm_index: int | None
    direction: str | None


def cusum(values, mu0: float, sigma: float, k: float = 0.5, h: float = 5.0) -> CusumResult:
    """Tabular two-sided CUSUM on standardised data z_i = (x_i - mu0)/sigma.
    C+_i = max(0, C+_{i-1} + z_i - k), C-_i = max(0, C-_{i-1} - z_i - k); alarm when either exceeds h."""
    z = (np.asarray(values, dtype=float) - mu0) / sigma
    cp, cm = np.zeros(len(z)), np.zeros(len(z))
    up = dn = 0.0
    for i, zi in enumerate(z):
        up, dn = max(0.0, up + zi - k), max(0.0, dn - zi - k)
        cp[i], cm[i] = up, dn
    hits = np.flatnonzero((cp > h) | (cm > h))
    if not len(hits):
        return CusumResult(cp, cm, None, None)
    first = int(hits[0])
    return CusumResult(cp, cm, first, "up" if cp[first] > h else "down")


@dataclass(frozen=True)
class ChangePoint:
    index: int  # first index of the new regime
    mean_before: float
    mean_after: float
    sse: float


def change_point(values, min_segment: int = 3) -> ChangePoint:
    """Single least-squares split in the mean (the first step of binary segmentation).
    With tau points before the split, SSE_total - SSE_split = S_tau^2 T / (tau (T - tau)) where
    S_tau = sum_{i<tau} (x_i - xbar); the best split maximises S_tau^2 / (tau (T - tau))."""
    x = np.asarray(values, dtype=float)
    T = len(x)
    if T < 2 * min_segment:
        raise ValueError("series too short for a change point")
    taus = np.arange(min_segment, T - min_segment + 1)
    s = np.cumsum(x - x.mean())[taus - 1]
    tau = int(taus[np.argmax(s**2 / (taus * (T - taus)))])
    before, after = x[:tau], x[tau:]
    sse = float(((before - before.mean()) ** 2).sum() + ((after - after.mean()) ** 2).sum())
    return ChangePoint(tau, float(before.mean()), float(after.mean()), sse)


@dataclass(frozen=True)
class Ramp:
    index: int  # onset of the ramp
    slope: float
    intercept: float
    sse: float


def ramp_fit(values, min_segment: int = 3) -> Ramp:
    """Hinge model x_t = a + b max(0, t - tau) fitted by least squares for each tau; lowest SSE wins."""
    x = np.asarray(values, dtype=float)
    t = np.arange(len(x))
    best: Ramp | None = None
    for tau in range(min_segment, len(x) - min_segment + 1):
        design = np.column_stack([np.ones(len(x)), np.maximum(0, t - tau)])
        coef, *_ = np.linalg.lstsq(design, x, rcond=None)
        sse = float(((x - design @ coef) ** 2).sum())
        if best is None or sse < best.sse:
            best = Ramp(tau, float(coef[1]), float(coef[0]), sse)
    assert best is not None
    return best


@dataclass(frozen=True)
class Agreement:
    n: int
    r: float
    bias: float
    loa_low: float
    loa_high: float
    slope: float
    intercept: float


def agreement(reference, test) -> Agreement:
    """Pearson r; bias = mean(test - reference); limits of agreement = bias +/- 1.96 sd(diff, ddof=1);
    slope and intercept from least squares of test on reference."""
    ref, tst = np.asarray(reference, dtype=float), np.asarray(test, dtype=float)
    diff = tst - ref
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    slope, intercept = np.polyfit(ref, tst, 1)
    return Agreement(len(ref), float(np.corrcoef(ref, tst)[0, 1]), bias, bias - 1.96 * sd, bias + 1.96 * sd,
                     float(slope), float(intercept))


@dataclass(frozen=True)
class DriftSummary:
    n: int
    baseline_mean: float
    sigma: float
    cusum_alarm_index: int | None
    ewma_alarm_index: int | None
    pattern: str  # stable, shift or drift
    change_index: int | None
    magnitude_sigma: float
    slope_per_step: float
    flagged: bool


def drift_summary(values, baseline_n: int, sigma: float | None = None, lam: float = 0.2, L: float = 3.0,
                  k: float = 0.5, h: float = 5.0, magnitude_threshold: float = 2.0, min_segment: int = 3) -> DriftSummary:
    """Monitor one series of subgroup statistics (for example daily means).

    mu0 and (unless given) sigma come from the first baseline_n points. EWMA and CUSUM run over the
    whole series. The magnitude is the current offset: the mean of the last min_segment points
    minus mu0, in sigma units. The series is flagged when either chart alarms and the magnitude is
    at least magnitude_threshold sigma, which keeps marginal alarms on stable series out of the
    alert list. For a flagged series a step fit (change_point) and a hinge fit (ramp_fit) are
    compared by SSE to label the pattern shift or drift and to estimate the change day.
    """
    x = np.asarray(values, dtype=float)
    base = x[:baseline_n]
    mu0 = float(base.mean())
    sigma = float(base.std(ddof=1)) if sigma is None else float(sigma)
    cu, ew = cusum(x, mu0, sigma, k, h), ewma(x, mu0, sigma, lam, L)
    step, ramp = change_point(x, min_segment), ramp_fit(x, min_segment)
    is_ramp = ramp.sse < step.sse
    magnitude = abs(float(x[-min_segment:].mean()) - mu0)
    flagged = (cu.alarm_index is not None or ew.alarm_index is not None) and magnitude / sigma >= magnitude_threshold
    pattern = "stable" if not flagged else "drift" if is_ramp else "shift"
    return DriftSummary(len(x), mu0, sigma, cu.alarm_index, ew.alarm_index, pattern,
                        (ramp.index if is_ramp else step.index) if flagged else None,
                        magnitude / sigma, ramp.slope if is_ramp else 0.0, flagged)
