"""FastAPI service.

Endpoints
  GET  /health          liveness + loaded model version
  GET  /model-info      version, metrics, threshold, grades, reason-code catalogue
  POST /predict         score one applicant
  POST /predict/batch   score up to 1,000 applicants

Interactive docs at /docs. Run locally:  uv run poe api
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from credit_scoring import __version__
from credit_scoring.api.schemas import (
    Applicant,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfo,
    ScoreResponse,
)
from credit_scoring.api.service import PredictionLog, ScoringService
from credit_scoring.config import get_settings
from credit_scoring.models.bundle import CreditRiskModel
from credit_scoring.models.fetch import ensure_model

log = logging.getLogger("credit_scoring.api")


def _configure_logging() -> None:
    """Uvicorn configures only its own loggers, so without this the app's INFO logs
    (request lines, model download/verification) would be silently dropped."""
    app_logger = logging.getLogger("credit_scoring")
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        app_logger.addHandler(handler)
        app_logger.setLevel(logging.INFO)
        app_logger.propagate = False


def create_app(service: ScoringService | None = None) -> FastAPI:
    """App factory. Tests inject a ready ``service``. Otherwise the model is loaded from
    ``CREDIT_MODELS_DIR`` at startup, and startup fails loudly if it's missing."""
    _configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            settings = get_settings()
            ensure_model(settings)  # download + verify a released model if none is local
            model = CreditRiskModel.load(settings.models_dir)
            app.state.service = ScoringService(model, PredictionLog(settings.prediction_log))
            _ = app.state.service.model.explainer  # build SHAP explainer before traffic
        else:
            app.state.service = service
        log.info("Model %s ready", app.state.service.version)
        yield

    app = FastAPI(
        title="Credit Scoring API",
        version=__version__,
        description=(
            "Probability of default, 300-850 credit score, risk grade, approve/decline "
            "decision and reason codes for consumer credit applications."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        log.info(
            "%s %s -> %d (%.1f ms) request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    def _service(request: Request) -> ScoringService:
        svc: ScoringService = request.app.state.service
        return svc

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        return HealthResponse(status="ok", model_version=_service(request).version)

    @app.get("/model-info", response_model=ModelInfo, tags=["ops"])
    def model_info(request: Request) -> ModelInfo:
        return _service(request).info()

    @app.post("/predict", response_model=ScoreResponse, tags=["scoring"])
    def predict(applicant: Applicant, request: Request) -> ScoreResponse:
        return _service(request).score([applicant], request.state.request_id)[0]

    @app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        results = _service(request).score(batch.applicants, request.state.request_id)
        return BatchResponse(
            count=len(results),
            approved=sum(r.decision == "approve" for r in results),
            results=results,
        )

    return app


app = create_app()
