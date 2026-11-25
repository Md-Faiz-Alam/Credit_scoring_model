"""Stratified train / validation / test split.

* **train** is used to fit imputers, binning and models.
* **valid** is used for model selection, probability calibration and the decision threshold.
* **test** is touched once, for the final reported numbers.

The dataset has no application date, so a true out-of-time split isn't possible. That
limitation is documented in the model card. Stratifying on the target keeps the ~6.7%
default rate identical across the three sets.
"""

from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scoring.data import columns as c


class Splits(NamedTuple):
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split_data(
    df: pd.DataFrame, *, valid_size: float, test_size: float, random_state: int
) -> Splits:
    train_valid, test = train_test_split(
        df, test_size=test_size, stratify=df[c.TARGET], random_state=random_state
    )
    # Rescale so valid_size is a fraction of the *whole* dataset
    relative_valid = valid_size / (1 - test_size)
    train, valid = train_test_split(
        train_valid,
        test_size=relative_valid,
        stratify=train_valid[c.TARGET],
        random_state=random_state,
    )
    return Splits(
        train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
    )
