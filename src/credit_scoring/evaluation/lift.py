"""Decile lift / gains table: the standard way to present a risk model to the business.

Applicants are sorted from riskiest to safest and split into ten equal groups.
"Cumulative capture" answers questions like: *if we decline the riskiest 10%, what share
of all future defaults do we avoid?*
"""

import numpy as np
import pandas as pd


def decile_table(y_true: pd.Series | np.ndarray, pd_: np.ndarray, n: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(pd_)})
    # rank(first) breaks ties so deciles are equal-sized; decile 1 = highest PD
    df["decile"] = n - pd.qcut(df["p"].rank(method="first"), n, labels=False)
    overall = df["y"].mean()
    total_bads = df["y"].sum()

    t = (
        df.groupby("decile")
        .agg(
            count=("y", "size"),
            min_pd=("p", "min"),
            max_pd=("p", "max"),
            mean_pd=("p", "mean"),
            defaults=("y", "sum"),
        )
        .sort_index()
    )
    t["default_rate"] = t["defaults"] / t["count"]
    t["lift"] = t["default_rate"] / overall
    t["cum_capture"] = t["defaults"].cumsum() / total_bads
    t["cum_population"] = t["count"].cumsum() / t["count"].sum()
    t["cum_lift"] = t["cum_capture"] / t["cum_population"]
    return t.reset_index().round(4)
