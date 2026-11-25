"""Shared fixtures: a small synthetic dataset in the raw Kaggle format.

Tests never depend on the real dataset, so they run in CI without Kaggle credentials.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAW_HEADER = [
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def make_raw(n: int = 400, seed: int = 0, signal: bool = False) -> pd.DataFrame:
    """Synthetic raw data. With ``signal=True`` the default risk depends on utilization,
    delinquency and age (like the real data), so models can learn something."""
    rng = np.random.default_rng(seed)
    util = rng.random(n)
    late = rng.poisson(0.4, n).astype(float)
    age = rng.integers(21, 90, n).astype(float)
    if signal:
        logit = -3.0 + 3.0 * util + 1.2 * late - 0.02 * (age - 50)
        target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    else:
        target = (rng.random(n) < 0.2).astype(float)
    df = pd.DataFrame(
        {
            "ID": np.arange(1, n + 1, dtype=float),
            "SeriousDlqin2yrs": target,
            "RevolvingUtilizationOfUnsecuredLines": util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late,
            "DebtRatio": rng.random(n),
            "MonthlyIncome": rng.integers(1000, 15000, n).astype(float),
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n).astype(float),
            "NumberOfTimes90DaysLate": rng.integers(0, 2, n).astype(float),
            "NumberRealEstateLoansOrLines": rng.integers(0, 3, n).astype(float),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 2, n).astype(float),
            "NumberOfDependents": rng.integers(0, 4, n).astype(float),
        }
    )
    df.loc[0:9, "MonthlyIncome"] = np.nan
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Raw frame (Kaggle names) with a known set of planted data problems."""
    df = make_raw()
    # Planted problems, each at a known row:
    df.loc[1, "age"] = 0  # invalid age
    for col in (
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ):
        df.loc[2, col] = 98  # special reporting code
    dup = df.iloc[[3]].copy()
    dup["ID"] = 9999  # same record, different id
    junk = pd.DataFrame([{"MonthlyIncome": 29731.0}])  # spreadsheet footer row
    return pd.concat([df, dup, junk], ignore_index=True)


@pytest.fixture
def raw_csv(tmp_path: Path, raw_df: pd.DataFrame) -> Path:
    path = tmp_path / "raw" / "cs-training.csv"
    path.parent.mkdir(parents=True)
    raw_df.to_csv(path, index=False)
    return path
