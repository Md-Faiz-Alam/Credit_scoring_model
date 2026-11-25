"""UI tests: the HTTP client, and the Streamlit app driven headlessly with AppTest."""

from typing import Any

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from credit_scoring.config import PROJECT_ROOT
from credit_scoring.ui.client import ApiClient, ApiError

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")


def _client(handler: Any) -> ApiClient:
    return ApiClient("http://api.test", transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------ client


def test_client_returns_json() -> None:
    api = _client(lambda req: httpx.Response(200, json={"status": "ok", "model_version": "v"}))
    assert api.health()["model_version"] == "v"


def test_client_formats_validation_errors() -> None:
    detail = {"detail": [{"loc": ["body", "age"], "msg": "Input should be >= 18"}]}
    api = _client(lambda req: httpx.Response(422, json=detail))
    with pytest.raises(ApiError, match="age: Input should be >= 18"):
        api.predict({"age": 10})


def test_client_wraps_connection_errors() -> None:
    def refuse(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(ApiError, match="Cannot reach"):
        _client(refuse).health()


def test_client_reports_server_errors() -> None:
    with pytest.raises(ApiError, match="500"):
        _client(lambda req: httpx.Response(500, text="boom")).model_info()


# ------------------------------------------------------------------ streamlit app

INFO = {
    "model_version": "v-test",
    "champion": "lightgbm",
    "created_at": "2026-01-01",
    "decision_threshold_pd": 0.1667,
    "decision_threshold_score": 547,
    "score_range": [300, 850],
    "risk_grades": {"A": 0.02, "B": 0.05, "C": 0.1, "D": 0.2, "E": 1.0},
    "test_metrics": {"auc": 0.865, "gini": 0.73, "ks": 0.589, "ece": 0.002},
    "reason_codes": {"R01": "Serious delinquency", "R02": "High utilization"},
}


def _fake_result(applicant: dict[str, Any]) -> dict[str, Any]:
    risky = applicant["revolving_utilization"] > 0.9
    return {
        "application_id": None,
        "probability_of_default": 0.31 if risky else 0.02,
        "score": 500 if risky else 700,
        "risk_grade": "E" if risky else "A",
        "decision": "decline" if risky else "approve",
        "reasons": [{"code": "R02", "message": "High utilization", "impact": 1.1}] if risky else [],
        "model_version": "v-test",
        "scored_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ApiClient, "health", lambda self: {"status": "ok", "model_version": "v-test"}
    )
    monkeypatch.setattr(ApiClient, "model_info", lambda self: INFO)
    monkeypatch.setattr(ApiClient, "predict", lambda self, a: _fake_result(a))


def test_app_shows_error_when_api_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDIT_API_URL", "http://127.0.0.1:9")  # nothing listens here
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Cannot reach" in e.value for e in at.error)


def test_app_scores_a_sample_profile(fake_api: None) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.selectbox[0].select("Stretched borrower (high risk)").run()
    at.button[0].click().run()  # "Assess risk"
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Probability of default"] == "31.0%"
    assert metrics["Risk grade"] == "E"
    assert any("Reasons for decline" in s.value for s in at.subheader)


def test_app_model_tab(fake_api: None) -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    labels = [m.label for m in at.metric]
    assert {"ROC-AUC", "Gini", "KS"} <= set(labels)


def test_deep_link_scores_profile(fake_api: None) -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["profile"] = "high-risk"
    at.run()
    assert not at.exception
    assert at.selectbox[0].value == "Stretched borrower (high risk)"
    assert {m.label: m.value for m in at.metric}["Risk grade"] == "E"


def test_unknown_deep_link_is_ignored(fake_api: None) -> None:
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["profile"] = "nonsense"
    at.run()
    assert not at.exception
    assert at.selectbox[0].value == "(blank)"
