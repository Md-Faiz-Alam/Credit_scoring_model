"""Data contracts (pandera).

Two contracts:

* ``RAW_SCHEMA`` is what we accept from the source system. It only checks structure and
  types, because raw data is allowed to be messy and cleaning handles that.
* ``CLEAN_SCHEMA`` is what downstream feature/model code may assume. It checks value
  ranges and types, so a bad upstream change fails loudly here instead of silently
  degrading the model.
"""

import pandera.pandas as pa

from credit_scoring.data import columns as c

_count = pa.Column(float, pa.Check.in_range(0, 50), nullable=True)

RAW_SCHEMA = pa.DataFrameSchema(
    {name: pa.Column(float, nullable=True, coerce=True) for name in (*c.RAW_FEATURES, c.TARGET)},
    strict=False,  # extra columns (e.g. id) are fine at this stage
    name="raw_credit_data",
)

CLEAN_SCHEMA = pa.DataFrameSchema(
    {
        "revolving_utilization": pa.Column(float, pa.Check.ge(0)),
        "age": pa.Column(float, pa.Check.in_range(18, 120), nullable=True),
        "past_due_30_59": _count,
        "debt_ratio": pa.Column(float, pa.Check.ge(0)),
        "monthly_income": pa.Column(float, pa.Check.ge(0), nullable=True),
        "open_credit_lines": pa.Column(float, pa.Check.in_range(0, 100)),
        "past_due_90_plus": _count,
        "real_estate_loans": pa.Column(float, pa.Check.in_range(0, 100)),
        "past_due_60_89": _count,
        "dependents": pa.Column(float, pa.Check.in_range(0, 30), nullable=True),
        c.DELINQUENCY_CODE_FLAG: pa.Column("int8", pa.Check.isin([0, 1])),
        c.TARGET: pa.Column("int8", pa.Check.isin([0, 1])),
    },
    strict=True,
    ordered=True,
    unique=list(c.CLEAN_COLUMNS),  # no exact duplicate records
    name="clean_credit_data",
)
