"""Single source of truth for column names.

The Kaggle column names (``NumberOfTime30-59DaysPastDueNotWorse``...) are long and contain
characters that are awkward in code, formulas and API payloads. They are mapped once, at
load time, to short snake_case names that the rest of the codebase uses.
"""

from typing import Final

ID: Final = "id"
TARGET: Final = "default_2y"

RAW_TO_CLEAN: Final[dict[str, str]] = {
    "ID": ID,
    "Unnamed: 0": ID,  # original Kaggle cs-training.csv has an unnamed index column
    "SeriousDlqin2yrs": TARGET,
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "past_due_90_plus",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfDependents": "dependents",
}

DELINQUENCY_COLS: Final = ("past_due_30_59", "past_due_60_89", "past_due_90_plus")

# Raw input features, in a stable order.
RAW_FEATURES: Final = (
    "revolving_utilization",
    "age",
    "past_due_30_59",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "past_due_90_plus",
    "real_estate_loans",
    "past_due_60_89",
    "dependents",
)

# Flags added by cleaning
DELINQUENCY_CODE_FLAG: Final = "delinquency_code_reported"

CLEAN_COLUMNS: Final = (*RAW_FEATURES, DELINQUENCY_CODE_FLAG, TARGET)
