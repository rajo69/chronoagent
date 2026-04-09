# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir uv

COPY requirements.lock pyproject.toml ./
COPY src ./src

RUN uv pip install --system --no-cache -r requirements.lock && \
    uv pip install --system --no-cache -e .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user
RUN addgroup --system chrono && adduser --system --ingroup chrono chrono

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/src ./src
COPY configs ./configs

USER chrono

EXPOSE 8000

CMD ["uvicorn", "chronoagent.main:app", "--host", "0.0.0.0", "--port", "8000"]
