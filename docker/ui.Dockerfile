# syntax=docker/dockerfile:1.7
# Web UI image. The UI only talks HTTP to the API, so it gets the `ui` dependency
# group (Streamlit, Plotly, httpx) and the tiny `credit_scoring.ui` client, but none of the
# model stack (LightGBM, SHAP, scikit-learn).

ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --only-group ui --no-install-project


FROM python:${PYTHON_VERSION}-slim AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Only the files the UI imports: the package root and the HTTP client
COPY src/credit_scoring/__init__.py ./src/credit_scoring/__init__.py
COPY src/credit_scoring/ui ./src/credit_scoring/ui
COPY app ./app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    CREDIT_API_URL=http://api:8000 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
USER app

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status != 200)"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
