import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from credit_scoring.api.main import create_app
from credit_scoring.api.schemas import MAX_BATCH, Applicant, to_model_frame
from credit_scoring.api.service import PredictionLog, ScoringService
from credit_scoring.config import get_settings
from credit_scoring.data import columns as c
from credit_scoring.data.clean import clean
from credit_scoring.features.pipeline import split_xy
from credit_scoring.features.transformers import CreditFeatureEngineer
from credit_scoring.models.bundle import CreditRiskModel
from credit_scoring.models.scoring import RiskGrades, ScoreScaler

from .conftest import make_raw

APPLICANT: dict[str, Any] = {
    "application_id": "APP-1",
    "age": 42,
    "monthly_income": 5400,
    "monthly_debt_payments": 1600,
    "revolving_utilization": 0.35,
    "open_credit_lines": 8,
    "real_estate_loans": 1,
    "dependents": 2,
    "past_due_30_59": 0,
    "past_due_60_89": 0,
    "past_due_90_plus": 0,
}


@pytest.fixture(scope="module")
def model() -> CreditRiskModel:
    df, _ = clean(make_raw(n=4000, seed=21, signal=True).rename(columns=c.RAW_TO_CLEAN))
    X, y = split_xy(df)
    eng = CreditFeatureEngineer().set_output(transform="pandas")
    clf = lgb.LGBMClassifier(n_estimators=60, num_leaves=8, verbose=-1).fit(eng.fit_transform(X), y)
    return CreditRiskModel(
        Pipeline([("engineer", eng), ("model", clf)]),
        ScoreScaler(),
        RiskGrades(),
        threshold=0.3,
        metadata={
            "model_version": "test-1",
            "champion": "lightgbm",
            "created_at": "2026-01-01",
            "metrics": {"test": {"champion": {"auc": 0.8, "gini": 0.6, "ks": 0.5, "ece": 0.01}}},
        },
    )


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "predictions.jsonl"


@pytest.fixture
def client(model: CreditRiskModel, log_path: Path) -> Iterator[TestClient]:
    service = ScoringService(model, PredictionLog(log_path))
    with TestClient(create_app(service)) as tc:
        yield tc


# ------------------------------------------------------------------ input conversion


def test_debt_ratio_rebuilt_from_payments_and_income() -> None:
    with_income = Applicant(**{**APPLICANT, "monthly_income": 5000, "monthly_debt_payments": 1000})
    no_income = Applicant(**{**APPLICANT, "monthly_income": None, "monthly_debt_payments": 1000})
    zero_income = Applicant(**{**APPLICANT, "monthly_income": 0, "monthly_debt_payments": 1000})
    frame = to_model_frame([with_income, no_income, zero_income])
    # dataset convention: a ratio when income > 0, otherwise the debt amount itself
    assert frame["debt_ratio"].tolist() == pytest.approx([0.2, 1000, 1000])
    assert np.isnan(frame.loc[1, "monthly_income"])


def test_special_delinquency_code_matches_cleaning() -> None:
    frame = to_model_frame([Applicant(**{**APPLICANT, "delinquency_code_reported": True})])
    assert frame.loc[0, c.DELINQUENCY_CODE_FLAG] == 1
    assert frame[list(c.DELINQUENCY_COLS)].isna().all(axis=None)


def test_model_frame_has_training_columns() -> None:
    frame = to_model_frame([Applicant(**APPLICANT)])
    assert list(frame.columns) == [*c.RAW_FEATURES, c.DELINQUENCY_CODE_FLAG]


# ------------------------------------------------------------------ endpoints


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_version": "test-1"}


def test_model_info(client: TestClient, model: CreditRiskModel) -> None:
    info = client.get("/model-info").json()
    assert info["decision_threshold_pd"] == pytest.approx(0.3)
    assert info["decision_threshold_score"] == model.scaler.score(np.array([0.3]))[0]
    assert info["reason_codes"]["R01"].startswith("Serious delinquency")
    assert info["test_metrics"]["auc"] == pytest.approx(0.8)


def test_predict(client: TestClient) -> None:
    resp = client.post("/predict", json=APPLICANT)
    assert resp.status_code == 200
    body = resp.json()
    assert body["application_id"] == "APP-1"
    assert 300 <= body["score"] <= 850
    assert body["risk_grade"] in "ABCDE"
    expected = "decline" if body["probability_of_default"] >= 0.3 else "approve"
    assert body["decision"] == expected
    assert len(body["reasons"]) <= 4
    assert all(r["impact"] > 0 for r in body["reasons"])
    assert body["model_version"] == "test-1"


def test_api_matches_offline_model(client: TestClient, model: CreditRiskModel) -> None:
    """Serving must give exactly what the model gives offline: no training/serving skew."""
    body = client.post("/predict", json=APPLICANT).json()
    offline = model.predict(to_model_frame([Applicant(**APPLICANT)])).iloc[0]
    assert body["probability_of_default"] == pytest.approx(offline["pd"], abs=1e-6)
    assert body["score"] == offline["score"]


def test_riskier_profile_scores_lower(client: TestClient) -> None:
    safe = client.post("/predict", json=APPLICANT).json()
    risky = client.post(
        "/predict",
        json={**APPLICANT, "revolving_utilization": 0.99, "past_due_30_59": 4, "age": 23},
    ).json()
    assert risky["score"] < safe["score"]


def test_request_id_propagates(client: TestClient) -> None:
    resp = client.post("/predict", json=APPLICANT, headers={"X-Request-ID": "trace-123"})
    assert resp.headers["X-Request-ID"] == "trace-123"
    assert float(resp.headers["X-Process-Time-Ms"]) >= 0


@pytest.mark.parametrize(
    "change",
    [
        {"age": 17},
        {"monthly_income": -1},
        {"revolving_utilization": -0.1},
        {"past_due_30_59": 96},  # special codes go through delinquency_code_reported
        {"unexpected_field": 1},
    ],
)
def test_invalid_input_rejected(client: TestClient, change: dict[str, Any]) -> None:
    assert client.post("/predict", json={**APPLICANT, **change}).status_code == 422


def test_missing_required_field_rejected(client: TestClient) -> None:
    body = {k: v for k, v in APPLICANT.items() if k != "age"}
    assert client.post("/predict", json=body).status_code == 422


def test_batch(client: TestClient) -> None:
    applicants = [{**APPLICANT, "application_id": f"A{i}"} for i in range(25)]
    body = client.post("/predict/batch", json={"applicants": applicants}).json()
    assert body["count"] == 25
    assert [r["application_id"] for r in body["results"]] == [f"A{i}" for i in range(25)]
    assert body["approved"] == sum(r["decision"] == "approve" for r in body["results"])


@pytest.mark.parametrize("n", [0, MAX_BATCH + 1])
def test_batch_size_limits(client: TestClient, n: int) -> None:
    resp = client.post("/predict/batch", json={"applicants": [APPLICANT] * n})
    assert resp.status_code == 422


def test_prediction_log(client: TestClient, log_path: Path) -> None:
    client.post("/predict", json=APPLICANT, headers={"X-Request-ID": "r1"})
    client.post("/predict/batch", json={"applicants": [APPLICANT, APPLICANT]})
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    first = records[0]
    assert first["request_id"] == "r1"
    assert first["model_version"] == "test-1"
    assert set(first["features"]) == {*c.RAW_FEATURES, c.DELINQUENCY_CODE_FLAG}
    assert "APP-1" not in json.dumps(first)  # no application identifiers in the log


def test_app_loads_model_from_disk(
    model: CreditRiskModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model.save(tmp_path / "models")
    monkeypatch.setenv("CREDIT_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CREDIT_PREDICTION_LOG", "")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as tc:
            assert tc.get("/health").json()["model_version"] == "test-1"
            assert tc.post("/predict", json=APPLICANT).status_code == 200
    finally:
        get_settings.cache_clear()


def test_app_fails_fast_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDIT_MODELS_DIR", str(tmp_path / "empty"))
    get_settings.cache_clear()
    try:
        with pytest.raises(FileNotFoundError), TestClient(create_app()):
            pass
    finally:
        get_settings.cache_clear()
