"""Business decision threshold.

A 0.5 cut-off assumes both kinds of mistake cost the same and that defaults are as
common as repayments. In lending neither is true. The cut-off is the PD that minimises
expected cost on the validation split:

    cost = cost_FN * (#approved who default) + cost_FP * (#declined who would have repaid)

For calibrated PDs the optimum is exactly cost_FP / (cost_FP + cost_FN), which is
1/6 ≈ 16.7% with the default 5:1 cost ratio. The empirical cost curve is flat near its
minimum (on validation, costs within ~1% across cut-offs 0.15-0.19), so a raw grid
search picks noise and the cut-off would jump between retrains. We therefore use the
analytical threshold, since calibration is verified separately, and report the
empirical optimum as a check. The strategy table shows the trade-off a credit
committee would discuss: approval rate against the bad rate of the approved book.
"""

import numpy as np
import pandas as pd


def bayes_threshold(cost_fn: float, cost_fp: float) -> float:
    """Cost-minimising PD cut-off for calibrated probabilities.

    Decline when the expected loss of approving, PD * cost_FN, exceeds the expected loss
    of declining, (1 - PD) * cost_FP. That is, when PD > cost_FP / (cost_FP + cost_FN).
    """
    return cost_fp / (cost_fp + cost_fn)


def expected_cost(
    y_true: pd.Series | np.ndarray, pd_: np.ndarray, threshold: float, c_fn: float, c_fp: float
) -> float:
    return _cost_at(np.asarray(y_true), np.asarray(pd_), threshold, c_fn, c_fp)


def _cost_at(y: np.ndarray, p: np.ndarray, threshold: float, c_fn: float, c_fp: float) -> float:
    approve = p < threshold
    fn = np.sum(approve & (y == 1))
    fp = np.sum(~approve & (y == 0))
    return float((c_fn * fn + c_fp * fp) / len(y))


def cost_optimal_threshold(
    y_true: pd.Series | np.ndarray,
    pd_: np.ndarray,
    cost_fn: float,
    cost_fp: float,
    n_candidates: int = 400,
) -> tuple[float, float]:
    """Return (threshold, expected cost per applicant)."""
    y = np.asarray(y_true)
    p = np.asarray(pd_)
    candidates = np.unique(np.quantile(p, np.linspace(0, 1, n_candidates)))
    costs = np.array([_cost_at(y, p, t, cost_fn, cost_fp) for t in candidates])
    best = int(np.argmin(costs))
    return float(candidates[best]), float(costs[best])


def strategy_table(
    y_true: pd.Series | np.ndarray,
    pd_: np.ndarray,
    cost_fn: float,
    cost_fp: float,
    approval_rates: tuple[float, ...] = tuple(np.round(np.arange(0.5, 1.0001, 0.05), 2)),
) -> pd.DataFrame:
    """Approval rate vs bad rate and expected cost, as the cut-off moves."""
    y = np.asarray(y_true)
    p = np.asarray(pd_)
    rows = []
    for rate in approval_rates:
        cut = float(np.quantile(p, rate)) if rate < 1 else np.inf
        approved = p <= cut if rate < 1 else np.ones_like(p, dtype=bool)
        rows.append(
            {
                "approval_rate": round(float(approved.mean()), 4),
                "pd_cutoff": round(cut, 4) if np.isfinite(cut) else None,
                "bad_rate_approved": round(float(y[approved].mean()), 4),
                "defaults_caught": round(float((~approved & (y == 1)).sum() / max(y.sum(), 1)), 4),
                "expected_cost": round(_cost_at(y, p, cut + 1e-12, cost_fn, cost_fp), 4),
            }
        )
    return pd.DataFrame(rows)
