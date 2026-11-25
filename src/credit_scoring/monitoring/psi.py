"""Population Stability Index, the standard drift metric in credit risk.

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

Bins are fixed from the **reference** (training) population: deciles of a continuous
variable, plus a separate bin for missing values. Applied to the model score it's called
PSI. Applied to an individual input feature it's usually called CSI (Characteristic
Stability Index). The maths is the same.

Industry rule of thumb:
    PSI < 0.10        stable
    0.10 - 0.25       monitor: investigate the cause
    > 0.25            action: significant shift, consider recalibration or retraining
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

Status = Literal["stable", "monitor", "action"]

MONITOR_AT = 0.10
ACTION_AT = 0.25
EPS = 1e-4  # floor for empty bins, so ln() stays finite


def status_for(psi: float) -> Status:
    if psi >= ACTION_AT:
        return "action"
    if psi >= MONITOR_AT:
        return "monitor"
    return "stable"


def quantile_edges(values: np.ndarray, n_bins: int = 10) -> list[float]:
    """Interior bin edges from reference quantiles. Duplicate edges (count data) collapse."""
    v = values[~np.isnan(values)]
    if v.size == 0:
        return []
    qs = np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])
    return [float(x) for x in np.unique(qs)]


def bin_shares(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Share of rows per bin: len(edges) + 1 numeric bins, then one missing bin."""
    values = np.asarray(values, dtype=float)
    missing = np.isnan(values)
    idx = np.searchsorted(np.asarray(edges), values[~missing], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, missing.sum())
    return counts / max(len(values), 1)


@dataclass
class PSIResult:
    psi: float
    status: Status
    expected: list[float]
    actual: list[float]


def psi(expected_shares: np.ndarray, actual_shares: np.ndarray) -> PSIResult:
    e = np.clip(np.asarray(expected_shares, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual_shares, dtype=float), EPS, None)
    value = float(np.sum((a - e) * np.log(a / e)))
    return PSIResult(
        psi=round(value, 5),
        status=status_for(value),
        expected=[round(float(x), 5) for x in expected_shares],
        actual=[round(float(x), 5) for x in actual_shares],
    )
