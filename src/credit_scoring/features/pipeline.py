"""Feature pipelines shared by training and serving.

* ``tree_features()``: domain features only. Gradient boosting handles missing values
  and non-linearity natively.
* ``scorecard_features()``: domain features followed by WoE encoding, the input to the
  logistic-regression scorecard.
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from credit_scoring.data import columns as c
from credit_scoring.features.transformers import CreditFeatureEngineer
from credit_scoring.features.woe import WoEEncoder


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the target."""
    return df.drop(columns=[c.TARGET]), df[c.TARGET]


def tree_features(cap_quantile: float = 0.99) -> Pipeline:
    return Pipeline([("engineer", CreditFeatureEngineer(cap_quantile=cap_quantile))]).set_output(
        transform="pandas"
    )


def scorecard_features(
    cap_quantile: float = 0.99, max_bins: int = 6, min_bin_frac: float = 0.03
) -> Pipeline:
    return Pipeline(
        [
            ("engineer", CreditFeatureEngineer(cap_quantile=cap_quantile)),
            ("woe", WoEEncoder(max_bins=max_bins, min_bin_frac=min_bin_frac)),
        ]
    ).set_output(transform="pandas")
