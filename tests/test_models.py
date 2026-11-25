from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from credit_scoring.config import Settings
from credit_scoring.data import columns as c
from credit_scoring.data.clean import clean
from credit_scoring.data.make_dataset import build
from credit_scoring.evaluation.metrics import (
    classification_summary,
    expected_calibration_error,
    ks_statistic,
)
from credit_scoring.features.pipeline import split_xy
from credit_scoring.models.bundle import METADATA_FILE, CreditRiskModel
from credit_scoring.models.calibrate import calibrate_if_needed
from credit_scoring.models.config import DEFAULT_CONFIG_PATH, TrainConfig
from credit_scoring.models.scorecard import build_scorecard, points_table
from credit_scoring.models.scoring import RiskGrades, ScoreScaler, decide
from credit_scoring.models.threshold import cost_optimal_threshold, strategy_table
from credit_scoring.models.train import QualityGateError, train

from .conftest import make_raw

# Small, fast settings for tests
FAST = {"lightgbm": {"n_trials": 1, "cv_folds": 2, "early_stopping_rounds": 10}}


@pytest.fixture(scope="module")
def signal_xy() -> tuple[pd.DataFrame, pd.Series]:
    df, _ = clean(make_raw(n=4000, seed=3, signal=True).rename(columns=c.RAW_TO_CLEAN))
    return split_xy(df)


def _calibrated_sample(n: int = 50_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 12, n)
    y = (rng.random(n) < p).astype(int)
    return y, p


# ------------------------------------------------------------------ config


def test_repo_config_is_valid() -> None:
    cfg = TrainConfig.from_yaml(DEFAULT_CONFIG_PATH)
    assert cfg.decision.cost_false_negative > cfg.decision.cost_false_positive


def test_config_rejects_unknown_keys_and_bad_grades() -> None:
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"scorecard": {"typo_key": 1}})
    with pytest.raises(ValidationError, match="increasing"):
        TrainConfig.model_validate({"risk_grades": {"A": 0.5, "B": 0.1, "C": 1.0}})
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"lightgbm": {"monotone_constraints": {"age": 2}}})


# ------------------------------------------------------------------ metrics


def test_ks_extremes() -> None:
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_ece_small_for_calibrated_large_for_biased() -> None:
    y, p = _calibrated_sample()
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p * 3, 0, 1)) > 0.05


def test_summary_gini_matches_auc() -> None:
    y, p = _calibrated_sample()
    s = classification_summary(y, p)
    assert s["gini"] == pytest.approx(2 * s["auc"] - 1, abs=1e-4)


# ------------------------------------------------------------------ scoring


def test_score_scaling_anchor_and_pdo() -> None:
    s = ScoreScaler(base_score=650, base_odds=30, pdo=40)
    pd_at_base = 1 / 31  # odds 30:1
    pd_double = 1 / 61  # odds 60:1
    assert s.score(np.array([pd_at_base]))[0] == 650
    assert s.score(np.array([pd_double]))[0] == 690
    assert s.score(np.array([0.999999]))[0] == 300  # clipped
    assert s.score(np.array([1e-9]))[0] == 850


def test_score_is_monotone() -> None:
    s = ScoreScaler()
    pds = np.array([0.01, 0.05, 0.1, 0.3])
    scores = s.score(pds)
    assert np.all(np.diff(scores) < 0)


def test_grades_and_decision() -> None:
    g = RiskGrades({"A": 0.02, "B": 0.05, "C": 1.0})
    assert g.grade(np.array([0.0, 0.02, 0.021, 0.5])).tolist() == ["A", "A", "B", "C"]
    assert decide(np.array([0.1, 0.2]), 0.15).tolist() == ["approve", "decline"]


# ------------------------------------------------------------------ threshold


def test_cost_optimal_threshold_near_theory() -> None:
    y, p = _calibrated_sample(200_000)
    t, _ = cost_optimal_threshold(y, p, cost_fn=5, cost_fp=1)
    assert t == pytest.approx(1 / 6, abs=0.03)


def test_strategy_table_trade_off() -> None:
    y, p = _calibrated_sample()
    table = strategy_table(y, p, 5, 1)
    assert table["approval_rate"].is_monotonic_increasing
    assert table["bad_rate_approved"].is_monotonic_increasing
    assert table["approval_rate"].iloc[-1] == 1.0


# ------------------------------------------------------------------ scorecard


def test_scorecard_selection_and_signs(signal_xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = signal_xy
    pipe, dropped = build_scorecard(X, y, TrainConfig())
    lr: LogisticRegression = pipe.named_steps["model"]
    assert np.all(lr.coef_[0] <= 0), "every WoE coefficient must be negative"
    assert "revolving_utilization" in pipe.named_steps["select"].columns
    assert all(isinstance(reason, str) for reason in dropped.values())


def test_points_table_reproduces_model_score(signal_xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = signal_xy
    pipe, _ = build_scorecard(X, y, TrainConfig())
    scaler = ScoreScaler(min_score=-10_000, max_score=10_000)  # no clipping
    points = points_table(pipe, scaler)

    applicant = X.iloc[[0]]
    woe_values = pipe[:-1].transform(applicant).iloc[0]
    total = sum(
        points[(points.feature == f) & np.isclose(points.woe, round(w, 4))]["points"].iloc[0]
        for f, w in woe_values.items()
    )
    model_score = scaler.score(pipe.predict_proba(applicant)[:, 1])[0]
    assert total == pytest.approx(model_score, abs=2)


# ------------------------------------------------------------------ calibration


class _Inflated(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Wraps a fitted model and predicts 3x its PD: a deliberately miscalibrated model."""

    def __init__(self, base: Any = None) -> None:
        self.base = base

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_Inflated":
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.base.predict_proba(X)[:, 1] * 3, 0, 0.999)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def test_calibration_skipped_when_already_calibrated(
    signal_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = signal_xy
    pipe, _ = build_scorecard(X, y, TrainConfig())
    est, info = calibrate_if_needed(pipe, X, y, max_ece=0.05)
    assert est is pipe
    assert info["calibrated"] is False


def test_calibration_fixes_biased_model(signal_xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = signal_xy
    pipe, _ = build_scorecard(X, y, TrainConfig())

    _, info = calibrate_if_needed(_Inflated(pipe).fit(X, y), X, y, max_ece=0.01)
    assert info["calibrated"] is True
    assert info["ece_after"] < info["ece_before"]


# ------------------------------------------------------------------ bundle


def test_bundle_roundtrip(tmp_path: Path, signal_xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = signal_xy
    pipe, _ = build_scorecard(X, y, TrainConfig())
    bundle = CreditRiskModel(pipe, ScoreScaler(), RiskGrades(), 0.15, {"model_version": "t"})
    bundle.save(tmp_path)
    loaded = CreditRiskModel.load(tmp_path)

    out = loaded.predict(X.head(20))
    assert list(out.columns) == ["pd", "score", "grade", "decision"]
    assert out["pd"].between(0, 1).all()
    assert set(out["decision"]) <= {"approve", "decline"}
    assert (tmp_path / METADATA_FILE).exists()
    np.testing.assert_allclose(out["pd"], bundle.predict_pd(X.head(20)))


# ------------------------------------------------------------------ end to end


def _prepared_settings(tmp_path: Path, signal: bool) -> Settings:
    raw = tmp_path / "data" / "raw" / "cs-training.csv"
    raw.parent.mkdir(parents=True)
    make_raw(n=4000, seed=5, signal=signal).to_csv(raw, index=False)
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )
    build(settings)
    return settings


def test_train_end_to_end(tmp_path: Path) -> None:
    settings = _prepared_settings(tmp_path, signal=True)
    # Loose gate: 600 validation rows make ECE noisy
    cfg = TrainConfig.model_validate(
        {**FAST, "quality_gate": {"min_valid_auc": 0.6, "max_valid_ece": 0.1}}
    )
    path = train(settings, cfg, tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    model = CreditRiskModel.load(path.parent)
    meta = model.metadata
    assert meta["champion"] in {"scorecard", "lightgbm"}
    assert 0 < model.threshold < 1
    assert meta["metrics"]["test"]["champion"]["auc"] > 0.6
    for report in ("model_comparison", "scorecard_points", "strategy_table", "calibration_valid"):
        assert (settings.reports_dir / f"{report}.csv").exists()


def test_quality_gate_blocks_useless_model(tmp_path: Path) -> None:
    settings = _prepared_settings(tmp_path, signal=False)  # target is pure noise
    cfg = TrainConfig.model_validate(FAST)
    with pytest.raises(QualityGateError, match="AUC"):
        train(settings, cfg, tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    assert not (settings.models_dir / "credit_model.joblib").exists()


def test_bayes_threshold() -> None:
    from credit_scoring.models.threshold import bayes_threshold

    assert bayes_threshold(cost_fn=5, cost_fp=1) == pytest.approx(1 / 6)
    assert bayes_threshold(cost_fn=1, cost_fp=1) == pytest.approx(0.5)
