# ── Stage 1: dependências de produção ────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY app/ app/
COPY gunicorn.conf.py .
RUN uv pip install --system --no-cache .


# ── Stage 2: lint + testes ────────────────────────────────────────────────────
FROM deps AS test

RUN uv pip install --system --no-cache ruff pytest httpx

COPY tests/ tests/

RUN ruff check app/
RUN pytest tests/ -v


# ── Stage 3: produção ─────────────────────────────────────────────────────────
FROM deps AS production

RUN mkdir -p models_store logs

ENV MODELS_DIR=models_store
ENV LOGS_DIR=logs
ENV LOG_LEVEL=INFO

EXPOSE 8000

CMD ["gunicorn", "app.main:app", "--config", "gunicorn.conf.py"]
