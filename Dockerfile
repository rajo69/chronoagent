# Pinned image versions for reproducibility.  Bump these intentionally.
ARG PYTHON_TAG=3.11.13-slim-bookworm
ARG UV_VERSION=0.7.12

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:${PYTHON_TAG} AS builder
ARG UV_VERSION

WORKDIR /build

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

COPY requirements.lock pyproject.toml ./
COPY src ./src

RUN uv pip install --system --no-cache -r requirements.lock && \
    uv pip install --system --no-cache -e .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:${PYTHON_TAG} AS runtime

WORKDIR /app

# make is needed for the reproduce-* targets
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN addgroup --system chrono && adduser --system --ingroup chrono chrono

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/src ./src
COPY configs ./configs
COPY Makefile ./

USER chrono

EXPOSE 8000

CMD ["uvicorn", "chronoagent.main:app", "--host", "0.0.0.0", "--port", "8000"]
