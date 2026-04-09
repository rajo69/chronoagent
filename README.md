# ChronoAgent

> Temporal health monitoring for LLM multi-agent systems.

**Status:** Phase 0 — Bootstrap

---

## Quick Start

```bash
cp .env.example .env          # fill in CHRONO_TOGETHER_API_KEY
docker compose up --build     # starts app + redis + postgres + chroma
curl http://localhost:8000/health
```

## Installation (local dev)

```bash
pip install uv
uv pip install -e ".[dev]"
pre-commit install
make test
```

## Environment Variables

See `.env.example` for the full list of `CHRONO_*` variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `CHRONO_ENV` | `dev` | Runtime environment: dev / prod / test |
| `CHRONO_LLM_BACKEND` | `together` | LLM backend: together / mock / ollama |
| `CHRONO_TOGETHER_API_KEY` | — | API key for Together.ai |
| `CHRONO_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CHRONO_DATABASE_URL` | `sqlite:///./chronoagent.db` | SQLAlchemy DB URL |

## Project Structure

```
src/chronoagent/
├── config.py          # Pydantic settings
├── main.py            # FastAPI app factory
├── cli.py             # Typer CLI
├── api/
│   └── health.py      # /health endpoint
└── observability/
    └── logging.py     # structlog configuration
```

## Commands

```bash
make dev              # FastAPI with hot reload
make test             # pytest (MockBackend, no API calls)
make lint             # ruff + mypy
make docker-up        # full stack
cz commit             # conventional commit
```
