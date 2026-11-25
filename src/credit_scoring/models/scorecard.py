"""Champion candidate: a logistic-regression scorecard on WoE features.

Feature selection follows standard scorecard practice:

1. **IV filter**: drop features with too little information (IV < min_iv).
2. **Correlation filter**: of two highly correlated WoE features, keep the one with the
   higher IV. Correlated inputs make coefficients unstable and hard to interpret.
3. **Sign check**: with WoE = ln(%good / %bad), every coefficient predicting *default*
   must be negative. A positive coefficient means the feature is fighting the others
   (collinearity), and its points would run backwards for customers. Such features are
   dropped one at a time, the worst first, and the model is refit.

The fitted model becomes a **points table**: each bin of each feature carries a fixed
number of points, and a customer's score is the sum. Anyone can read it and audit it.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_scoring.features.transformers import ColumnSelector, CreditFeatureEngineer
from credit_scoring.features.woe import WoEEncoder
from credit_scoring.models.config import TrainConfig
from credit_scoring.models.scoring import ScoreScaler

log = logging.getLogger(__name__)


def select_features(
    X_woe: pd.DataFrame, y: pd.Series, iv: pd.Series, cfg: TrainConfig
) -> tuple[list[str], dict[str, str]]:
    """Return the selected features and the reason each dropped feature was dropped."""
    sc = cfg.scorecard
    dropped: dict[str, str] = {}

    candidates = [f for f in iv.sort_values(ascending=False).index if iv[f] >= sc.min_iv]
    dropped |= {f: f"IV {iv[f]:.3f} < {sc.min_iv}" for f in iv.index if iv[f] < sc.min_iv}

    corr = X_woe[candidates].corr().abs()
    kept: list[str] = []
    for f in candidates:  # highest IV first
        clash = next((k for k in kept if corr.loc[f, k] > sc.max_correlation), None)
        if clash:
            dropped[f] = f"corr {corr.loc[f, clash]:.2f} with {clash} (higher IV)"
        else:
            kept.append(f)

    while kept:
        lr = LogisticRegression(C=sc.C, max_iter=1000).fit(X_woe[kept], y)
        coefs = pd.Series(lr.coef_[0], index=kept)
        wrong = coefs[coefs > 0]
        if wrong.empty:
            break
        worst = str(wrong.idxmax())
        dropped[worst] = f"wrong-sign coefficient {wrong[worst]:+.3f}"
        kept.remove(worst)

    for f, why in dropped.items():
        log.info("scorecard drops %-26s %s", f, why)
    return kept, dropped


def build_scorecard(
    X: pd.DataFrame, y: pd.Series, cfg: TrainConfig
) -> tuple[Pipeline, dict[str, str]]:
    """Fit the full scorecard pipeline: engineer -> WoE -> select -> logistic regression."""
    fc = cfg.features
    engineer = CreditFeatureEngineer(cap_quantile=fc.cap_quantile).set_output(transform="pandas")
    woe = WoEEncoder(max_bins=fc.woe_max_bins, min_bin_frac=fc.woe_min_bin_frac)
    X_eng = engineer.fit_transform(X)
    X_woe = woe.fit_transform(X_eng, y)
    iv = woe.iv_table().set_index("feature")["iv"]

    selected, dropped = select_features(X_woe, y, iv, cfg)
    lr = LogisticRegression(C=cfg.scorecard.C, max_iter=1000).fit(X_woe[selected], y)

    pipe = Pipeline(
        [
            ("engineer", engineer),
            ("woe", woe),
            ("select", ColumnSelector(selected).fit(X_woe)),
            ("model", lr),
        ]
    )
    return pipe, dropped


def points_table(pipe: Pipeline, scaler: ScoreScaler) -> pd.DataFrame:
    """Scorecard points per bin.

    ln(odds_good) = -(intercept + sum_j beta_j * WoE_j), so with n features, bin b of
    feature j is worth  offset/n - factor * (intercept/n + beta_j * WoE_jb).
    Summing one bin per feature reproduces the (unclipped) model score, up to rounding.
    """
    woe: WoEEncoder = pipe.named_steps["woe"]
    lr: LogisticRegression = pipe.named_steps["model"]
    features = list(pipe.named_steps["select"].columns)
    n = len(features)
    intercept = float(lr.intercept_[0])
    bins = woe.bin_table()

    rows = []
    for feature, beta in zip(features, lr.coef_[0], strict=True):
        fb = bins[bins["feature"] == feature]
        pts = scaler.offset / n - scaler.factor * (intercept / n + beta * fb["woe"])
        rows.append(fb.assign(coefficient=round(float(beta), 4), points=np.round(pts, 1)))
    table: pd.DataFrame = pd.concat(rows, ignore_index=True)
    return table[["feature", "bin", "count", "bad_rate", "woe", "coefficient", "points"]]
