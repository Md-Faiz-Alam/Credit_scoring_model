# syntax=docker/dockerfile:1.7
# Scoring API image.
#   builder: resolves the locked dependencies into /opt/venv (uv, frozen lockfile)
#   runtime: slim Python + that venv + source. No compilers, no uv, runs as non-root.
# The model isn't baked in. Mount it at /app/models (see docker-compose.yml), so a
# retrained model ships without rebuilding the image.

ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim AS builder
# uv pinned to the version that writes uv.lock locally
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build

# 1) dependencies only: cached until pyproject.toml / uv.lock change
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --extra api --no-install-project

# 2) the project itself (non-editable, so the runtime stage needs no source tree)
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --extra api --no-editable


FROM python:${PYTHON_VERSION}-slim AS runtime
# libgomp: OpenMP runtime that LightGBM needs at import time
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CREDIT_MODELS_DIR=/app/models \
    CREDIT_PREDICTION_LOG=/app/logs/predictions.jsonl \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba

WORKDIR /app
RUN mkdir -p /app/models /app/logs && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,sys,urllib.request; sys.exit(urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", 8000)}/health', timeout=4).status != 200)"

# PORT is injected by hosting platforms (e.g. Render); uvicorn reads WEB_CONCURRENCY for the
# number of worker processes. exec keeps uvicorn as PID 1 so it receives stop signals.
ENV WEB_CONCURRENCY=2
CMD ["sh", "-c", "exec uvicorn credit_scoring.api.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
