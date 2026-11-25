import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from credit_scoring.data import columns as c
from credit_scoring.data.clean import clean
from credit_scoring.evaluation import plots
from credit_scoring.evaluation.fairness import age_band, group_metrics, without_feature
from credit_scoring.evaluation.lift import decile_table
from credit_scoring.evaluation.metrics import calibration_table
from credit_scoring.evaluation.report import evaluate
from credit_scoring.explain.explainer import ReasonCodeExplainer, unwrap_pipeline
from credit_scoring.explain.reasons import FEATURE_TO_GROUP, REASON_GROUPS
from credit_scoring.features.pipeline import split_xy
from credit_scoring.features.transformers import OUTPUT_COLUMNS, CreditFeatureEngineer
from credit_scoring.models.bundle import CreditRiskModel
from credit_scoring.models.config import TrainConfig
from credit_scoring.models.scorecard import build_scorecard
from credit_scoring.models.scoring import RiskGrades, ScoreScaler
from credit_scoring.models.train import train

from .conftest import make_raw
from .test_models import FAST, _prepared_settings


@pytest.fixture(scope="module")
def xy() -> tuple[pd.DataFrame, pd.Series]:
    df, _ = clean(make_raw(n=4000, seed=11, signal=True).rename(columns=c.RAW_TO_CLEAN))
    return split_xy(df)


@pytest.fixture(scope="module")
def tree_pipe(xy: tuple[pd.DataFrame, pd.Series]) -> Pipeline:
    X, y = xy
    eng = CreditFeatureEngineer().set_output(transform="pandas")
    Z = eng.fit_transform(X)
    model = lgb.LGBMClassifier(n_estimators=60, num_leaves=8, verbose=-1).fit(Z, y)
    return Pipeline([("engineer", eng), ("model", model)])


@pytest.fixture(scope="module")
def linear_pipe(xy: tuple[pd.DataFrame, pd.Series]) -> Pipeline:
    pipe, _ = build_scorecard(*xy, TrainConfig())
    return pipe


# ------------------------------------------------------------------ reason codes


def test_every_feature_has_a_reason_group() -> None:
    assert set(OUTPUT_COLUMNS) <= set(FEATURE_TO_GROUP), "unmapped feature -> no customer text"
    codes = [g.code for g in REASON_GROUPS]
    assert len(codes) == len(set(codes))


def test_tree_contributions_are_additive(
    tree_pipe: Pipeline, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    X = xy[0].head(50)
    ex = ReasonCodeExplainer(tree_pipe)
    contrib = ex.contributions(X)
    raw = logit(tree_pipe.predict_proba(X)[:, 1])
    np.testing.assert_allclose(contrib.sum(axis=1) + ex.base_value, raw, atol=1e-6)


def test_linear_contributions_are_additive(
    linear_pipe: Pipeline, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    X = xy[0]
    ex = ReasonCodeExplainer(linear_pipe, background=X.sample(500, random_state=0))
    contrib = ex.contributions(X.head(50))
    raw = logit(linear_pipe.predict_proba(X.head(50))[:, 1])
    np.testing.assert_allclose(contrib.sum(axis=1) + ex.base_value, raw, atol=1e-6)


def test_linear_explainer_needs_background(linear_pipe: Pipeline) -> None:
    with pytest.raises(ValueError, match="background"):
        ReasonCodeExplainer(linear_pipe)


def test_reasons_are_risk_increasing_sorted_and_capped(
    tree_pipe: Pipeline, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    X = xy[0].head(100)
    for expl in ReasonCodeExplainer(tree_pipe).explain(X, max_reasons=3):
        impacts = [r.impact for r in expl.reasons]
        assert len(impacts) <= 3
        assert all(i > 0 for i in impacts)
        assert impacts == sorted(impacts, reverse=True)


def test_grouped_impact_equals_sum_of_member_features(
    tree_pipe: Pipeline, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    X = xy[0].head(30)
    ex = ReasonCodeExplainer(tree_pipe)
    contrib = ex.contributions(X)
    for (idx, row), expl in zip(contrib.iterrows(), ex.explain(X, max_reasons=8), strict=True):
        for reason in expl.reasons:
            assert reason.impact == pytest.approx(row[reason.features].sum(), abs=1e-3), idx


def test_high_utilization_applicant_gets_utilization_reason(
    tree_pipe: Pipeline, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    applicant = xy[0].head(1).assign(revolving_utilization=0.99, past_due_30_59=0.0)
    codes = [r.code for r in ReasonCodeExplainer(tree_pipe).explain(applicant)[0].reasons]
    assert "R02" in codes


def test_explainer_unwraps_calibration(
    tree_pipe: Pipeline, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = xy
    calibrated = CalibratedClassifierCV(FrozenEstimator(tree_pipe), method="sigmoid").fit(X, y)
    assert unwrap_pipeline(calibrated) is tree_pipe
    ReasonCodeExplainer(calibrated).explain(X.head(2))


def test_bundle_explain_survives_pickle(
    tree_pipe: Pipeline, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    bundle = CreditRiskModel(tree_pipe, ScoreScaler(), RiskGrades(), 0.15)
    before = bundle.explain(xy[0].head(3))
    restored = pickle.loads(pickle.dumps(bundle))
    assert restored._explainer is None  # explainer is rebuilt, not pickled
    assert restored.explain(xy[0].head(3)) == before


# ------------------------------------------------------------------ lift


def test_decile_table(tree_pipe: Pipeline, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    t = decile_table(y, tree_pipe.predict_proba(X)[:, 1])
    assert len(t) == 10
    assert t["count"].sum() == len(X)
    assert t["cum_capture"].iloc[-1] == pytest.approx(1.0)
    assert t.loc[0, "default_rate"] > t.loc[9, "default_rate"]  # decile 1 = riskiest
    assert t.loc[0, "lift"] > 1


# ------------------------------------------------------------------ fairness


def test_age_band_boundaries() -> None:
    bands = age_band(pd.Series([18, 25, 26, 35, 36, 60, 61, 99]))
    assert bands.tolist() == ["18-25", "18-25", "26-35", "26-35", "36-45", "46-60", "61+", "61+"]


def test_group_metrics_adverse_impact() -> None:
    y = np.array([0, 0, 1, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9])
    groups = pd.Series(["18-25"] * 4 + ["61+"] * 4)
    t = group_metrics(y, p, threshold=0.5, groups=groups).set_index("group")
    assert t.loc["18-25", "approval_rate"] == pytest.approx(0.75)
    assert t.loc["61+", "approval_rate"] == pytest.approx(0.25)
    assert t["adverse_impact_ratio"].max() == pytest.approx(1.0)
    assert not t.loc["61+", "passes_four_fifths"]


def test_without_feature_drops_age(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    meta = {"n_trees": 30, "num_leaves": 8, "verbose": -1, "monotone_constraints_method": "basic"}
    pipe = without_feature(X, y, meta, "age", 0.99, {"revolving_utilization": 1})
    assert "age" not in pipe.named_steps["model"].feature_name_
    assert pipe.predict_proba(X.head(5)).shape == (5, 2)


# ------------------------------------------------------------------ plots + report


def test_all_charts_render(
    tmp_path: Path, tree_pipe: Pipeline, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = xy
    p = tree_pipe.predict_proba(X)[:, 1]
    yv = y.to_numpy()
    fair = group_metrics(yv, p, 0.2, age_band(X["age"]))
    imp = ReasonCodeExplainer(tree_pipe).global_importance(X.head(200))
    paths = [
        plots.roc_chart(yv, {"a": p, "b": p**2}, tmp_path / "roc.png"),
        plots.calibration_chart(calibration_table(yv, p), tmp_path / "cal.png"),
        plots.gains_chart(decile_table(yv, p), tmp_path / "gains.png"),
        plots.importance_chart(imp, tmp_path / "imp.png"),
        plots.score_distribution_chart(ScoreScaler().score(p), yv, 550, tmp_path / "s.png"),
        plots.fairness_chart(fair, fair, tmp_path / "fair.png"),
    ]
    for path in paths:
        assert path.stat().st_size > 5_000


def test_evaluate_writes_model_card(tmp_path: Path) -> None:
    settings = _prepared_settings(tmp_path, signal=True)
    cfg = TrainConfig.model_validate(
        {**FAST, "quality_gate": {"min_valid_auc": 0.6, "max_valid_ece": 0.1}}
    )
    train(settings, cfg, tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    docs = tmp_path / "docs"
    result = evaluate(settings, docs_dir=docs)

    card = (docs / "model_card.md").read_text(encoding="utf-8")
    for section in ("## Performance", "## Business impact", "## Fairness: age", "## Limitations"):
        assert section in card
    assert len(list((docs / "figures").glob("*.png"))) == 6
    assert 0 < result["decision"]["approval_rate"] <= 1
    assert (settings.reports_dir / "fairness_age.csv").exists()
