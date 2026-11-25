"""Fairness analysis by age band.

Age is a protected characteristic in most lending regulation. US ECOA / Regulation B
permits it in an "empirically derived, demonstrably and statistically sound" scoring
system, but it must not be used to the disadvantage of older applicants. Two questions
matter:

1. **Is the model equally accurate for every group?** Per-group AUC, and calibration:
   is the predicted PD close to the actual default rate in each group? A model that
   overstates risk for one group treats it unfairly even if the averages look fine.
2. **What is the impact on outcomes?** Approval rate per group, and the *adverse impact
   ratio* (AIR): a group's approval rate divided by the most-approved group's. The
   "four-fifths rule" flags AIR < 0.8.

``without_feature`` refits the champion LightGBM, with the same tuned hyperparameters,
minus a feature (age). That shows how much accuracy age actually buys, and whether
dropping it changes the outcome gap or simply moves it onto correlated features.
"""

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from credit_scoring.features.transformers import (
    OUTPUT_COLUMNS,
    ColumnSelector,
    CreditFeatureEngineer,
)

AGE_BINS = (17, 25, 35, 45, 60, 200)
AGE_LABELS = ("18-25", "26-35", "36-45", "46-60", "61+")
FOUR_FIFTHS = 0.8


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=list(AGE_BINS), labels=list(AGE_LABELS)).astype("string")


def group_metrics(
    y_true: pd.Series | np.ndarray,
    pd_: np.ndarray,
    threshold: float,
    groups: pd.Series,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {"y": np.asarray(y_true), "p": np.asarray(pd_), "g": np.asarray(groups, dtype=object)}
    )
    df["approved"] = df["p"] < threshold
    rows = []
    for g, part in df.dropna(subset=["g"]).groupby("g", sort=False):
        bads = part["y"] == 1
        rows.append(
            {
                "group": g,
                "count": len(part),
                "default_rate": part["y"].mean(),
                "mean_pd": part["p"].mean(),
                "calibration_gap": part["p"].mean() - part["y"].mean(),
                "auc": roc_auc_score(part["y"], part["p"]) if part["y"].nunique() == 2 else np.nan,
                "approval_rate": part["approved"].mean(),
                # share of the group's defaulters who are declined (good: higher)
                "defaults_declined": (~part["approved"] & bads).sum() / max(bads.sum(), 1),
                # share of the group's good payers who are declined (bad: higher)
                "goods_declined": (~part["approved"] & ~bads).sum() / max((~bads).sum(), 1),
            }
        )
    out = pd.DataFrame(rows)
    order = {label: i for i, label in enumerate(AGE_LABELS)}
    out = out.sort_values("group", key=lambda s: s.map(order).fillna(len(order)))
    out["adverse_impact_ratio"] = out["approval_rate"] / out["approval_rate"].max()
    out["passes_four_fifths"] = out["adverse_impact_ratio"] >= FOUR_FIFTHS
    return out.reset_index(drop=True).round(4)


def without_feature(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lgbm_meta: dict[str, Any],
    drop: str,
    cap_quantile: float,
    monotone_constraints: dict[str, int],
) -> Pipeline:
    """Refit the champion LightGBM without one feature, reusing its tuned parameters."""
    columns = [c for c in OUTPUT_COLUMNS if c != drop]
    skip = {"objective", "metric", "cv_auc", "n_trees", "monotone_constraints"}
    params = {k: v for k, v in lgbm_meta.items() if k not in skip}
    if "monotone_constraints_method" in params:
        params["monotone_constraints"] = [monotone_constraints.get(c, 0) for c in columns]

    engineer = CreditFeatureEngineer(cap_quantile=cap_quantile).set_output(transform="pandas")
    selector = ColumnSelector(columns)
    X_eng = selector.fit_transform(engineer.fit_transform(X_train))
    model = lgb.LGBMClassifier(n_estimators=int(lgbm_meta["n_trees"]), **params).fit(X_eng, y_train)
    return Pipeline([("engineer", engineer), ("select", selector), ("model", model)])
