import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from credit_scoring.data import columns as c
from credit_scoring.data.clean import clean
from credit_scoring.data.load import load_raw
from credit_scoring.features.pipeline import scorecard_features, split_xy, tree_features
from credit_scoring.features.transformers import OUTPUT_COLUMNS, CreditFeatureEngineer
from credit_scoring.features.woe import WoEEncoder, iv_strength

from .conftest import make_raw


@pytest.fixture
def clean_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Cleaned synthetic data (no file I/O)."""
    raw = make_raw(n=2000, seed=1).rename(columns=c.RAW_TO_CLEAN)
    df, _ = clean(raw)
    return split_xy(df)


def _row(**overrides: float) -> pd.DataFrame:
    base = {
        "revolving_utilization": 0.3,
        "age": 40.0,
        "past_due_30_59": 0.0,
        "debt_ratio": 0.4,
        "monthly_income": 5000.0,
        "open_credit_lines": 5.0,
        "past_due_90_plus": 0.0,
        "real_estate_loans": 1.0,
        "past_due_60_89": 0.0,
        "dependents": 1.0,
        c.DELINQUENCY_CODE_FLAG: 0,
    }
    return pd.DataFrame([{**base, **overrides}])


# ------------------------------------------------------------------ engineer


@pytest.fixture
def engineer(clean_xy: tuple[pd.DataFrame, pd.Series]) -> CreditFeatureEngineer:
    return CreditFeatureEngineer().fit(clean_xy[0])


def test_engineer_requires_fit() -> None:
    with pytest.raises(NotFittedError):
        CreditFeatureEngineer().transform(_row())


def test_engineer_rejects_missing_columns(engineer: CreditFeatureEngineer) -> None:
    with pytest.raises(ValueError, match="age"):
        engineer.transform(_row().drop(columns="age"))


def test_engineer_output_columns(engineer: CreditFeatureEngineer) -> None:
    out = engineer.transform(_row())
    assert tuple(out.columns) == OUTPUT_COLUMNS
    assert list(engineer.get_feature_names_out()) == list(OUTPUT_COLUMNS)


def test_implausible_utilization_becomes_missing_and_flagged(
    engineer: CreditFeatureEngineer,
) -> None:
    out = engineer.transform(_row(revolving_utilization=5000.0))
    assert np.isnan(out.loc[0, "revolving_utilization"])
    assert out.loc[0, "utilization_implausible"] == 1


@pytest.mark.parametrize("income", [np.nan, 0.0])
def test_debt_ratio_is_an_amount_without_income(
    engineer: CreditFeatureEngineer, income: float
) -> None:
    out = engineer.transform(_row(monthly_income=income, debt_ratio=1200.0))
    assert np.isnan(out.loc[0, "debt_ratio"])  # not a ratio
    assert out.loc[0, "monthly_debt"] == pytest.approx(min(1200.0, engineer.caps_["monthly_debt"]))
    assert out.loc[0, "no_income_reported"] == 1


def test_monthly_debt_from_ratio_with_income(engineer: CreditFeatureEngineer) -> None:
    out = engineer.transform(_row(monthly_income=5000.0, debt_ratio=0.4))
    assert out.loc[0, "monthly_debt"] == pytest.approx(2000.0)
    assert out.loc[0, "income_per_person"] == pytest.approx(2500.0)


def test_total_past_due_missing_when_counts_unknown(engineer: CreditFeatureEngineer) -> None:
    out = engineer.transform(
        _row(
            past_due_30_59=np.nan,
            past_due_60_89=np.nan,
            past_due_90_plus=np.nan,
            **{c.DELINQUENCY_CODE_FLAG: 1},
        )
    )
    assert np.isnan(out.loc[0, "total_past_due"])
    assert engineer.transform(_row(past_due_30_59=2, past_due_90_plus=1)).loc[
        0, "total_past_due"
    ] == pytest.approx(3)


def test_caps_are_learned_from_fit_data_only(
    clean_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    eng = CreditFeatureEngineer(cap_quantile=0.9).fit(clean_xy[0])
    cap = eng.caps_["monthly_income"]
    out = eng.transform(_row(monthly_income=10_000_000.0))
    assert out.loc[0, "monthly_income"] == pytest.approx(cap)
    assert cap < clean_xy[0]["monthly_income"].max()


def test_engineer_does_not_mutate_input(engineer: CreditFeatureEngineer) -> None:
    row = _row(revolving_utilization=5000.0)
    before = row.copy()
    engineer.transform(row)
    pd.testing.assert_frame_equal(row, before)


# ------------------------------------------------------------------ WoE


@pytest.fixture
def monotone_data() -> tuple[pd.DataFrame, pd.Series]:
    """Risk falls steadily with x. 10% of x is missing, with very high risk."""
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.uniform(0, 100, n)
    p = 0.4 - 0.35 * x / 100
    missing = rng.random(n) < 0.1
    p = np.where(missing, 0.6, p)
    y = (rng.random(n) < p).astype(int)
    x = np.where(missing, np.nan, x)
    return pd.DataFrame({"x": x}), pd.Series(y)


def test_woe_is_monotonic_and_signed(monotone_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = monotone_data
    enc = WoEEncoder().fit(X, y)
    woe = enc.binnings_["x"].woe
    assert len(woe) >= 3
    assert np.all(np.diff(woe) > 0)  # higher x -> safer -> higher WoE
    assert woe[0] < 0 < woe[-1]


def test_woe_missing_gets_own_bin(monotone_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = monotone_data
    enc = WoEEncoder().fit(X, y)
    assert enc.binnings_["x"].missing_woe < enc.binnings_["x"].woe.min()  # riskiest
    out = enc.transform(pd.DataFrame({"x": [np.nan]}))
    assert out.loc[0, "x"] == pytest.approx(enc.binnings_["x"].missing_woe)


def test_woe_unseen_missing_is_neutral() -> None:
    X = pd.DataFrame({"x": np.arange(1000, dtype=float)})
    y = pd.Series((np.arange(1000) < 300).astype(int))
    enc = WoEEncoder().fit(X, y)
    assert enc.transform(pd.DataFrame({"x": [np.nan]})).loc[0, "x"] == 0.0


def test_woe_out_of_range_values_use_edge_bins(
    monotone_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = monotone_data
    enc = WoEEncoder().fit(X, y)
    out = enc.transform(pd.DataFrame({"x": [-1e9, 1e9]}))
    woe = enc.binnings_["x"].woe
    assert out["x"].tolist() == pytest.approx([woe[0], woe[-1]])


def test_iv_table_and_bins(monotone_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = monotone_data
    enc = WoEEncoder().fit(X, y)
    iv = enc.iv_table()
    assert iv.loc[0, "iv"] > 0.1
    bins = enc.bin_table()
    assert bins["count"].sum() == len(X)
    assert bins["bin"].iloc[-1] == "Missing"


def test_woe_rejects_non_binary_target() -> None:
    with pytest.raises(ValueError, match="binary"):
        WoEEncoder().fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), [0, 1, 2])


@pytest.mark.parametrize(
    ("iv", "label"),
    [(0.01, "useless"), (0.05, "weak"), (0.2, "medium"), (0.4, "strong"), (0.9, "very_strong")],
)
def test_iv_strength(iv: float, label: str) -> None:
    assert iv_strength(iv) == label


# ------------------------------------------------------------------ pipelines


@pytest.mark.parametrize("factory", [tree_features, scorecard_features])
def test_pipeline_is_row_independent_after_fit(
    factory: object, clean_xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Scoring an applicant must not depend on who else is in the batch."""
    X, y = clean_xy
    pipe = factory().fit(X, y)  # type: ignore[operator]
    full = pipe.transform(X)
    part = pipe.transform(X.iloc[:50])
    pd.testing.assert_frame_equal(full.iloc[:50], part)


def test_scorecard_pipeline_roundtrips_through_pickle(
    clean_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = clean_xy
    pipe = scorecard_features().fit(X, y)
    restored = pickle.loads(pickle.dumps(pipe))
    pd.testing.assert_frame_equal(pipe.transform(X), restored.transform(X))
    assert not restored.transform(X).isna().any(axis=None)  # WoE output is complete


def test_pipelines_are_cloneable() -> None:
    for pipe in (tree_features(), scorecard_features()):
        clone(pipe)


def test_feature_pipeline_on_cleaned_file(raw_csv: object) -> None:
    df, _ = clean(load_raw(raw_csv))  # type: ignore[arg-type]
    X, y = split_xy(df)
    assert tree_features().fit_transform(X, y).shape == (len(X), len(OUTPUT_COLUMNS))
