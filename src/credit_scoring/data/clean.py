"""Row-level cleaning rules.

These rules are deterministic and use **no statistics learned from the data**. Medians,
caps and other fitted values belong in the feature pipeline, which is fit on the training
split only. That keeps cleaning safe to run before the train/test split without leaking
information.

Each rule is documented with the evidence behind it, and the cleaning report records
how many rows each rule touched.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from credit_scoring.data import columns as c
from credit_scoring.data.schema import CLEAN_SCHEMA

# 96 / 98 in the past-due counts aren't counts. They're reporting codes, and they always
# appear in all three columns at once (269 rows). Defaults run at ~55% in those rows
# against ~6.7% overall, so the code itself is highly informative. We keep it as a flag
# and blank out the meaningless "count".
DELINQUENCY_SPECIAL_CODES = (96, 98)

MIN_ADULT_AGE = 18


@dataclass
class CleaningReport:
    rows_in: int = 0
    dropped_missing_target: int = 0
    dropped_duplicates: int = 0
    delinquency_codes_flagged: int = 0
    invalid_age_set_missing: int = 0
    rows_out: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(rows_in=len(raw))
    df = raw.copy()

    # 1. Rows without a label are useless for training. In the supplied file these are
    #    junk footer rows added when the CSV was edited in a spreadsheet.
    has_target = df[c.TARGET].notna()
    report.dropped_missing_target = int((~has_target).sum())
    df = df.loc[has_target]

    # 2. Exact duplicate records (ignoring the id) would let the same applicant appear in
    #    both train and test, which inflates test scores.
    df = df.drop(columns=[c.ID], errors="ignore")
    dupes = df.duplicated()
    report.dropped_duplicates = int(dupes.sum())
    df = df.loc[~dupes]

    # 3. Special delinquency codes become a flag, and the count is set to missing.
    special = df[list(c.DELINQUENCY_COLS)].isin(DELINQUENCY_SPECIAL_CODES).any(axis=1)
    report.delinquency_codes_flagged = int(special.sum())
    df[c.DELINQUENCY_CODE_FLAG] = special.astype("int8")
    for col in c.DELINQUENCY_COLS:
        df[col] = df[col].mask(df[col].isin(DELINQUENCY_SPECIAL_CODES))

    # 4. Applicants must be adults. age == 0 is a data-entry error, so treat it as missing.
    bad_age = df["age"] < MIN_ADULT_AGE
    report.invalid_age_set_missing = int(bad_age.sum())
    df["age"] = df["age"].mask(bad_age, np.nan)

    df[c.TARGET] = df[c.TARGET].astype("int8")
    df = df[list(c.CLEAN_COLUMNS)].reset_index(drop=True)

    report.rows_out = len(df)
    return CLEAN_SCHEMA.validate(df), report
