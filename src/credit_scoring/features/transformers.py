"""Domain feature engineering as a scikit-learn transformer.

All learned values (outlier caps) are fit in ``fit`` on the training data and reused
unchanged at prediction time. Because the same object runs in training and in the API,
training/serving skew can't happen.

Evidence for each rule (training split, see reports/data_quality.json):

* **Revolving utilization** should be 0-1. Values of 1-3 are genuine over-limit borrowers
  (~40% default rate), but values above 10 are data errors that behave like ordinary
  customers (~5% default rate). Simple capping would put them in the riskiest group, so
  they're set to missing and flagged instead.
* **DebtRatio** is only a ratio when monthly income is positive. When income is missing or
  zero the column holds the monthly debt *amount* (median ~1,000 vs ~0.3), so ratio and
  amount are split into two features.
* Heavy right tails (income, debt) are capped at a high training quantile. This keeps
  linear models stable and doesn't affect tree models.
"""

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from credit_scoring.data import columns as c

IMPLAUSIBLE_UTILIZATION = 10.0

INPUT_COLUMNS = (*c.RAW_FEATURES, c.DELINQUENCY_CODE_FLAG)

OUTPUT_COLUMNS = (
    "revolving_utilization",
    "utilization_implausible",
    "age",
    "past_due_30_59",
    "past_due_60_89",
    "past_due_90_plus",
    "total_past_due",
    c.DELINQUENCY_CODE_FLAG,
    "monthly_income",
    "no_income_reported",
    "debt_ratio",
    "monthly_debt",
    "income_per_person",
    "open_credit_lines",
    "real_estate_loans",
    "dependents",
)

# Columns capped at a fitted upper quantile
CAPPED_COLUMNS = ("revolving_utilization", "debt_ratio", "monthly_debt", "monthly_income")


class CreditFeatureEngineer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Clean dataset columns -> model features.

    Parameters
    ----------
    cap_quantile:
        Upper quantile (fit on training data) at which heavy-tailed columns are capped.
    """

    def __init__(self, cap_quantile: float = 0.99) -> None:
        self.cap_quantile = cap_quantile

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        self._check_input(X)
        derived = self._derive(X)
        self.caps_: dict[str, float] = {
            col: float(derived[col].quantile(self.cap_quantile)) for col in CAPPED_COLUMNS
        }
        self.n_features_in_ = len(INPUT_COLUMNS)
        self.feature_names_in_ = np.array(INPUT_COLUMNS, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "caps_")
        self._check_input(X)
        out = self._derive(X)
        for col, cap in self.caps_.items():
            out[col] = out[col].clip(upper=cap)
        return out[list(OUTPUT_COLUMNS)]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(OUTPUT_COLUMNS, dtype=object)

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _check_input(X: pd.DataFrame) -> None:
        missing = set(INPUT_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")

    @staticmethod
    def _derive(X: pd.DataFrame) -> pd.DataFrame:
        """Stateless feature derivation (no learned values)."""
        out = X[list(INPUT_COLUMNS)].astype("float64").copy()

        util = out["revolving_utilization"]
        implausible = util > IMPLAUSIBLE_UTILIZATION
        out["utilization_implausible"] = implausible.astype("float64")
        out["revolving_utilization"] = util.mask(implausible)

        dues = out[list(c.DELINQUENCY_COLS)]
        # min_count=1: stays missing when the counts are unknown (special codes)
        out["total_past_due"] = dues.sum(axis=1, min_count=1)

        income = out["monthly_income"]
        has_income = income > 0
        raw_debt = out["debt_ratio"]
        out["no_income_reported"] = (~has_income).astype("float64")
        out["debt_ratio"] = raw_debt.where(has_income)
        out["monthly_debt"] = np.where(has_income, raw_debt * income, raw_debt)
        out["income_per_person"] = income / (out["dependents"].fillna(0) + 1)
        return out


class ColumnSelector(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Keep a fixed list of columns (chosen during feature selection)."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        missing = set(self.columns or []) - set(X.columns)
        if missing:
            raise ValueError(f"Columns not found: {sorted(missing)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)  # marks it fitted
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        return X[list(self.columns or [])]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.columns or [], dtype=object)
