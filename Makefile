.PHONY: dev test lint docker-up docker-down update-readme install

# ── Development ───────────────────────────────────────────────────────────────
dev:
	uvicorn chronoagent.main:app --reload --host 0.0.0.0 --port 8000

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -x --no-cov

# ── Linting ───────────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

lint-fix:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f app

# ── Dependencies ──────────────────────────────────────────────────────────────
install:
	pip install uv
	uv pip install -e ".[dev]"
	pre-commit install

lock:
	uv pip compile pyproject.toml -o requirements.lock

# ── Docs ──────────────────────────────────────────────────────────────────────
update-readme:
	@echo "Update README.md manually after each phase — see PLAN.md."
