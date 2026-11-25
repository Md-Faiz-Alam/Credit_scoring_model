"""Data-quality profile.

Records the facts about the data that later modelling decisions rely on, so that each
decision can be traced back to evidence. It also gives a baseline to compare future
data deliveries against.
"""

from typing import Any

import pandas as pd

from credit_scoring.data import columns as c


def _rate(mask: pd.Series) -> float:
    return round(float(mask.mean()), 6) if len(mask) else 0.0


def profile(df: pd.DataFrame) -> dict[str, Any]:
    income_missing = df["monthly_income"].isna()
    return {
        "rows": len(df),
        "default_rate": _rate(df[c.TARGET] == 1),
        "missing_rate": {col: _rate(df[col].isna()) for col in c.RAW_FEATURES},
        "findings": {
            # Utilization should be 0-1. Values far above that are data errors.
            "revolving_utilization_gt_1": int((df["revolving_utilization"] > 1).sum()),
            "revolving_utilization_gt_10": int((df["revolving_utilization"] > 10).sum()),
            # When income is missing, DebtRatio holds the monthly debt amount rather than a
            # ratio (median ~1,159 vs ~0.30). Feature engineering must treat these separately.
            "debt_ratio_median_income_missing": float(
                df.loc[income_missing, "debt_ratio"].median()
            ),
            "debt_ratio_median_income_present": float(
                df.loc[~income_missing, "debt_ratio"].median()
            ),
            "monthly_income_zero": int((df["monthly_income"] == 0).sum()),
            "default_rate_income_missing": _rate(df.loc[income_missing, c.TARGET] == 1),
            "default_rate_delinquency_code": _rate(
                df.loc[df[c.DELINQUENCY_CODE_FLAG] == 1, c.TARGET] == 1
            ),
        },
    }
