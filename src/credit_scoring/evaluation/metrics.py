"""Credit-risk model metrics.

* **ROC-AUC / Gini** (Gini = 2*AUC - 1): how well the model *ranks* risk. Gini is the
  figure banks usually quote.
* **KS**: the largest gap between the score distributions of goods and bads. A classic
  scorecard acceptance metric.
* **PR-AUC**: ranking quality focused on the rare default class.
* **Brier / ECE**: whether predicted PDs match observed default rates. Capital, pricing
  and expected-loss calculations use the PD itself, so calibration matters as much as
  ranking.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

ArrayLike = np.ndarray | pd.Series


def ks_statistic(y_true: ArrayLike, y_score: ArrayLike) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def calibration_table(y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted default rate per quantile bin of predicted PD."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    return (
        df.groupby("bin")
        .agg(count=("y", "size"), predicted_pd=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index(drop=True)
    )


def expected_calibration_error(y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10) -> float:
    """Average |observed - predicted|, weighted by bin size."""
    t = calibration_table(y_true, y_prob, n_bins)
    weights = t["count"] / t["count"].sum()
    return float((weights * (t["observed_rate"] - t["predicted_pd"]).abs()).sum())


def classification_summary(y_true: ArrayLike, y_prob: ArrayLike) -> dict[str, Any]:
    auc = float(roc_auc_score(y_true, y_prob))
    return {
        "auc": round(auc, 5),
        "gini": round(2 * auc - 1, 5),
        "ks": round(ks_statistic(y_true, y_prob), 5),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 5),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 6),
        "ece": round(expected_calibration_error(y_true, y_prob), 6),
        "mean_pd": round(float(np.mean(y_prob)), 6),
        "default_rate": round(float(np.mean(y_true)), 6),
    }
