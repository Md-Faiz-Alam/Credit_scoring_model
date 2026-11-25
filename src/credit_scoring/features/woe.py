"""Weight-of-Evidence (WoE) binning and Information Value (IV).

This is the standard way banks build application scorecards:

1. Each numeric feature is cut into a few bins. The cut points come from a shallow
   decision tree fit on the training data, with a minimum share of rows per bin.
2. Adjacent bins are merged until the risk trend across bins is **monotonic**. A
   scorecard where risk goes up, then down, then up with age can't be explained to a
   customer or a regulator, so monotonicity is a hard requirement.
3. Missing values get their own bin, because "missing" is often informative.
4. Each bin is replaced by its WoE:

       WoE = ln( share_of_all_goods_in_bin / share_of_all_bads_in_bin )

   A positive WoE means the bin has *lower* risk than average (Siddiqi's convention).
5. A feature's IV sums its predictive power across bins:

       IV = sum( (share_goods - share_bads) * WoE )

   Rule of thumb: < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong,
   > 0.5 very strong. Above 0.5 is normally a reason to check for leakage. Here it's
   expected for delinquency history, which is recorded *before* the 2-year outcome window
   and is the strongest known predictor in credit-bureau data.

Logistic regression on WoE features is linear in the log-odds and directly convertible
into scorecard points (see models/scorecard.py).
"""

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

MISSING_BIN = "Missing"


@dataclass
class FeatureBinning:
    edges: np.ndarray  # bin i covers (edges[i], edges[i+1]]; edges[0] = -inf, edges[-1] = inf
    woe: np.ndarray  # WoE of each numeric bin
    missing_woe: float
    iv: float
    table: pd.DataFrame


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "very_strong"


class WoEEncoder(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Supervised binning + WoE encoding for every column of a numeric DataFrame.

    Parameters
    ----------
    max_bins: maximum number of numeric bins per feature (before monotonic merging).
    min_bin_frac: minimum share of non-missing training rows in each bin. 3% (rather than
        the textbook 5%) lets sparse delinquency counts, which are non-zero for only
        ~5% of applicants, form their own bin.
    monotonic: merge adjacent bins until WoE is monotonic.
    smoothing: additive (Laplace) smoothing so empty classes don't give infinite WoE.
    min_missing_frac: a missing bin with fewer rows than this share gets WoE 0 (neutral)
        instead of an unreliable estimate.
    """

    def __init__(
        self,
        max_bins: int = 6,
        min_bin_frac: float = 0.03,
        monotonic: bool = True,
        smoothing: float = 0.5,
        min_missing_frac: float = 0.001,
    ) -> None:
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac
        self.monotonic = monotonic
        self.smoothing = smoothing
        self.min_missing_frac = min_missing_frac

    # ------------------------------------------------------------------ sklearn API

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> Self:
        y_arr = np.asarray(y, dtype=int)
        if set(np.unique(y_arr)) - {0, 1}:
            raise ValueError("WoEEncoder needs a binary 0/1 target")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.total_good_ = int((y_arr == 0).sum())
        self.total_bad_ = int((y_arr == 1).sum())
        self.binnings_: dict[str, FeatureBinning] = {
            col: self._fit_feature(X[col].to_numpy(dtype=float), y_arr) for col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "binnings_")
        out = {}
        for col, b in self.binnings_.items():
            x = X[col].to_numpy(dtype=float)
            idx = np.clip(np.searchsorted(b.edges, x, side="left") - 1, 0, len(b.woe) - 1)
            out[col] = np.where(np.isnan(x), b.missing_woe, b.woe[idx])
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "binnings_")
        return np.asarray(list(self.binnings_), dtype=object)

    # ------------------------------------------------------------------ reporting

    def iv_table(self) -> pd.DataFrame:
        check_is_fitted(self, "binnings_")
        rows = [
            {"feature": f, "iv": round(b.iv, 4), "strength": iv_strength(b.iv), "bins": len(b.woe)}
            for f, b in self.binnings_.items()
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)

    def bin_table(self) -> pd.DataFrame:
        check_is_fitted(self, "binnings_")
        return pd.concat(
            [b.table.assign(feature=f) for f, b in self.binnings_.items()], ignore_index=True
        )[["feature", "bin", "count", "bads", "bad_rate", "woe", "iv_contribution"]]

    # ------------------------------------------------------------------ internals

    def _woe(self, good: np.ndarray, bad: np.ndarray) -> np.ndarray:
        s = self.smoothing
        k = len(good)
        dist_good = (good + s) / (self.total_good_ + s * k)
        dist_bad = (bad + s) / (self.total_bad_ + s * k)
        return np.asarray(np.log(dist_good / dist_bad))

    def _initial_edges(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        tree = DecisionTreeClassifier(
            max_leaf_nodes=self.max_bins,
            min_samples_leaf=max(1, int(self.min_bin_frac * len(x))),
            random_state=0,
        )
        tree.fit(x.reshape(-1, 1), y)
        is_split = tree.tree_.children_left != tree.tree_.children_right
        thresholds = np.unique(tree.tree_.threshold[is_split])
        return np.concatenate([[-np.inf], thresholds, [np.inf]])

    @staticmethod
    def _counts(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.clip(np.searchsorted(edges, x, side="left") - 1, 0, len(edges) - 2)
        n_bins = len(edges) - 1
        bad = np.bincount(idx, weights=y, minlength=n_bins)
        total = np.bincount(idx, minlength=n_bins)
        return total - bad, bad

    def _make_monotonic(self, x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Merge the adjacent pair with the closest WoE until the trend is monotonic.

        The direction (increasing / decreasing) is the sign of the overall correlation.
        """
        corr = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 else 0.0
        increasing_woe = corr < 0  # higher x -> lower risk -> higher WoE
        while len(edges) > 2:
            woe = self._woe(*self._counts(x, y, edges))
            diffs = np.diff(woe)
            violations = diffs < 0 if increasing_woe else diffs > 0
            if not violations.any():
                break
            # merge the violating pair with the smallest WoE gap: remove the shared edge
            candidates = np.flatnonzero(violations)
            worst = candidates[np.argmin(np.abs(diffs[candidates]))]
            edges = np.delete(edges, worst + 1)
        return edges

    def _fit_feature(self, x_all: np.ndarray, y_all: np.ndarray) -> FeatureBinning:
        present = ~np.isnan(x_all)
        x, y = x_all[present], y_all[present]

        edges = self._initial_edges(x, y)
        if self.monotonic:
            edges = self._make_monotonic(x, y, edges)
        good, bad = self._counts(x, y, edges)
        woe = self._woe(good, bad)

        miss_good = int(((~present) & (y_all == 0)).sum())
        miss_bad = int(((~present) & (y_all == 1)).sum())
        miss_n = miss_good + miss_bad
        if miss_n >= self.min_missing_frac * len(x_all):
            missing_woe = float(self._woe(np.array([miss_good]), np.array([miss_bad]))[0])
        else:
            missing_woe = 0.0

        all_good = np.append(good, miss_good)
        all_bad = np.append(bad, miss_bad)
        all_woe = np.append(woe, missing_woe)
        contrib = (all_good / self.total_good_ - all_bad / self.total_bad_) * all_woe

        labels = [_bin_label(lo, hi) for lo, hi in pairwise(edges)]
        table = pd.DataFrame(
            {
                "bin": [*labels, MISSING_BIN],
                "count": (all_good + all_bad).astype(int),
                "bads": all_bad.astype(int),
                "bad_rate": np.round(all_bad / np.maximum(all_good + all_bad, 1), 4),
                "woe": np.round(all_woe, 4),
                "iv_contribution": np.round(contrib, 4),
            }
        )
        return FeatureBinning(edges, woe, missing_woe, float(contrib.sum()), table)


def _bin_label(lo: float, hi: float) -> str:
    if np.isinf(lo):
        return f"<= {hi:.4g}"
    if np.isinf(hi):
        return f"> {lo:.4g}"
    return f"({lo:.4g}, {hi:.4g}]"
