"""Per-applicant explanations in log-odds space.

Each feature gets a contribution to the applicant's log-odds of default, relative to
the average applicant. Positive values push risk *up*. Contributions add up exactly
to the model's raw output:

    log_odds(default) = base_value + sum(contributions)

* **LightGBM**: TreeSHAP (exact Shapley values for tree ensembles).
* **Logistic scorecard**: the model is linear in its WoE inputs, so its exact Shapley
  value is beta_j * (x_j - mean_j), with means taken over a training background sample.

A probability-calibration wrapper (Platt) is a monotone transformation of these log-odds.
It changes the size of the PD, but not which factors drive it or their ranking, so we
explain the underlying model.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_scoring.explain.reasons import FEATURE_TO_GROUP, REASON_GROUPS

MAX_REASONS = 4


@dataclass
class Reason:
    code: str
    message: str
    impact: float  # contribution to log-odds of default (> 0 = increases risk)
    features: list[str] = field(default_factory=list)


@dataclass
class Explanation:
    base_log_odds: float
    contributions: dict[str, float]  # per model feature
    reasons: list[Reason]  # risk-increasing groups, strongest first


def unwrap_pipeline(estimator: Any) -> Pipeline:
    """Return the fitted sklearn Pipeline inside any calibration wrapper."""
    if isinstance(estimator, CalibratedClassifierCV):
        estimator = estimator.calibrated_classifiers_[0].estimator
    if isinstance(estimator, FrozenEstimator):
        estimator = estimator.estimator
    if not isinstance(estimator, Pipeline):
        raise TypeError(f"Expected a Pipeline, got {type(estimator).__name__}")
    return estimator


class ReasonCodeExplainer:
    def __init__(self, estimator: Any, background: pd.DataFrame | None = None) -> None:
        pipe = unwrap_pipeline(estimator)
        self.preprocess = pipe[:-1]
        self.model = pipe[-1]

        if isinstance(self.model, LogisticRegression):
            if background is None:
                raise ValueError("A background sample is required to explain a linear model")
            self.kind = "linear"
            means = self.preprocess.transform(background).mean()
            self._coef = pd.Series(self.model.coef_[0], index=means.index)
            self._means = means
            self.base_value = float(self.model.intercept_[0] + (self._coef * means).sum())
        elif hasattr(self.model, "booster_"):  # LightGBM
            self.kind = "tree"
            self._tree = shap.TreeExplainer(self.model)
            self.base_value = float(np.ravel(self._tree.expected_value)[-1])
        else:
            raise TypeError(f"Unsupported model type: {type(self.model).__name__}")

    # ------------------------------------------------------------------ core

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution to log-odds of default, one row per applicant."""
        Z = self.preprocess.transform(X)
        if self.kind == "linear":
            values = (Z - self._means) * self._coef
        else:
            with warnings.catch_warnings():  # shap notes a format change we handle below
                warnings.simplefilter("ignore", UserWarning)
                raw = self._tree.shap_values(Z)
            arr = raw[1] if isinstance(raw, list) else np.asarray(raw)
            values = pd.DataFrame(arr, columns=Z.columns)
        values.index = X.index
        result: pd.DataFrame = values
        return result

    def explain(self, X: pd.DataFrame, max_reasons: int = MAX_REASONS) -> list[Explanation]:
        contrib = self.contributions(X)
        groups = group_contributions(contrib)
        out = []
        for idx in contrib.index:
            g = groups.loc[idx]
            positive = g[g > 0].sort_values(ascending=False).head(max_reasons)
            reasons = [
                Reason(
                    code=code,
                    message=_GROUP_BY_CODE[code].message,
                    impact=round(float(impact), 4),
                    features=[f for f in _GROUP_BY_CODE[code].features if f in contrib.columns],
                )
                for code, impact in positive.items()
            ]
            out.append(
                Explanation(
                    base_log_odds=round(self.base_value, 4),
                    contributions={k: round(float(v), 4) for k, v in contrib.loc[idx].items()},
                    reasons=reasons,
                )
            )
        return out

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean |contribution| per feature, with its reason group."""
        contrib = self.contributions(X)
        imp = contrib.abs().mean().sort_values(ascending=False)
        return pd.DataFrame(
            {
                "feature": imp.index,
                "mean_abs_contribution": imp.to_numpy().round(5),
                "reason_code": [_code_for(f) for f in imp.index],
            }
        )


_GROUP_BY_CODE = {g.code: g for g in REASON_GROUPS}


def _code_for(feature: str) -> str:
    group = FEATURE_TO_GROUP.get(feature)
    return group.code if group else "R99"


def group_contributions(contrib: pd.DataFrame) -> pd.DataFrame:
    """Sum feature contributions into reason groups (columns = reason codes)."""
    codes = [_code_for(f) for f in contrib.columns]
    return contrib.T.groupby(codes).sum().T
