"""Thin, typed HTTP client for the scoring API.

The UI talks to the model **only** through the API, like any other consumer (a loan
origination system, a batch job). That keeps a single source of truth for scoring and
lets the UI and API be deployed and scaled separately.
"""

from typing import Any

import httpx


class ApiError(RuntimeError):
    """The API is unreachable, or rejected the request (message is user-presentable)."""


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 30.0, **kwargs: Any) -> None:
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, **kwargs)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(
                f"Cannot reach the scoring API at {self._http.base_url} ({exc})"
            ) from exc
        if resp.status_code == 422:
            raise ApiError("Invalid input: " + _validation_message(resp.json()))
        if resp.is_error:
            raise ApiError(f"API error {resp.status_code}: {resp.text[:200]}")
        data: dict[str, Any] = resp.json()
        return data

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/model-info")

    def predict(self, applicant: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predict", json=applicant)

    def predict_batch(self, applicants: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/predict/batch", json={"applicants": applicants})


def _validation_message(body: dict[str, Any]) -> str:
    """Turn FastAPI's 422 detail list into one readable line."""
    parts = []
    for err in body.get("detail", [])[:5]:
        loc = ".".join(str(p) for p in err.get("loc", []) if p not in ("body",))
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts) or "request rejected"
