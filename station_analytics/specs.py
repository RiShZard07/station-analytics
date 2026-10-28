"""Metric definitions shared by the generator, the ingest contracts and the analysis.

All values are synthetic design choices for a made-up optical display-module test
line. They are not taken from any real product or instrument.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetricSpec:
    """One measured quantity with its spec limits and physically plausible range."""

    name: str
    unit: str
    target: float
    lsl: float
    usl: float
    plausible_min: float
    plausible_max: float
    unit_sd: float  # unit-to-unit (true) variation used by the generator
    meas_sd: float  # repeatability noise of one measurement used by the generator

    @property
    def tolerance(self) -> float:
        return self.usl - self.lsl


METRICS: dict[str, MetricSpec] = {
    m.name: m
    for m in (
        MetricSpec("luminance_nits", "nits", 500.0, 450.0, 550.0, 0.0, 5000.0, 10.0, 1.0),
        MetricSpec("color_delta_e", "dE2000", 1.5, 0.0, 3.0, 0.0, 100.0, 0.35, 0.03),
        MetricSpec("mtf_50", "cycles/pixel", 0.55, 0.45, 0.70, 0.0, 1.0, 0.025, 0.008),
        MetricSpec("binocular_misalignment_arcmin", "arcmin", 2.0, 0.0, 5.0, 0.0, 60.0, 0.6, 0.15),
    )
}
METRIC_NAMES: tuple[str, ...] = tuple(METRICS)

SENTINELS: tuple[float, ...] = (-999.0, 9999.0)

REFERENCE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
