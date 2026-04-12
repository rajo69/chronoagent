.PHONY: dev test test-unit test-integration lint docker-up docker-down update-readme install \
       reproduce reproduce-signal reproduce-main reproduce-ablations reproduce-figures

# ── Development ───────────────────────────────────────────────────────────────
dev:
	uvicorn chronoagent.main:app --reload --host 0.0.0.0 --port 8000

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration --no-cov

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

# ── Reproducibility ──────────────────────────────────────────────────────────
# Re-run the full experiment suite and regenerate every figure and table
# referenced in the paper. These targets mirror the protocol documented
# in paper/sections/05_experiments.tex.

reproduce-signal:
	chronoagent run-experiment phase1 \
		--config configs/experiments/signal_validation.yaml \
		--output results/

reproduce-main:
	chronoagent run-experiment phase10 \
		--config configs/experiments/main_experiment.yaml \
		--output results/
	chronoagent run-experiment phase10 \
		--config configs/experiments/agentpoison_experiment.yaml \
		--output results/
	chronoagent run-experiment phase10 \
		--config configs/experiments/baseline_sentinel.yaml \
		--output results/

reproduce-ablations:
	chronoagent run-experiment phase10 \
		--config configs/experiments/ablation_no_bocpd.yaml \
		--output results/
	chronoagent run-experiment phase10 \
		--config configs/experiments/ablation_no_forecaster.yaml \
		--output results/
	chronoagent run-experiment phase10 \
		--config configs/experiments/ablation_no_health_scores.yaml \
		--output results/

reproduce-figures:
	chronoagent compare-experiments \
		--output results/ \
		--experiment main_experiment \
		--experiment agentpoison_experiment \
		--experiment baseline_sentinel \
		--ablation ablation_no_bocpd \
		--ablation ablation_no_forecaster \
		--ablation ablation_no_health_scores

reproduce: reproduce-signal reproduce-main reproduce-ablations reproduce-figures

# ── Docs ──────────────────────────────────────────────────────────────────────
update-readme:
	@echo "Update README.md manually after each phase — see PLAN.md."
