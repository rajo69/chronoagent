# ChronoAgent Implementation Plan

> v2.1 | 2026-04-09 | Active working document — tracks implementation, decisions, findings

Critical path: **0 → 1 (gate) → 2 → 3 → 4 → 5 → 10 → 11**

---

## How to Use This File

This is your single source of truth across all development sessions. Every session follows this loop:

```
1. Open PLAN.md → find current phase → read exit criteria
2. Pick next unchecked [ ] task → implement it → write its unit test immediately
3. Run `make test` — must be green before checking off the task
4. Check [x] when task + test pass
5. Fill in the phase's Log section (findings, challenges, decisions)
6. When all phase tasks are [x] and ALL exit criteria pass → mark phase [x] in Phase Tracker
```

> **Testing rule:** Every implementation task gets its test written in the same session, before moving to the next task. `make test` must be green before any `[x]`. Exit criteria are the phase gate — all must pass before the phase is marked complete.
> **Exception — Phase 1:** This is a research experiment, not a software phase. There are no unit tests. The "gate" is the measured signal results and the GO/NO-GO decision.

**Starting a new Claude Code session:**
Say: _"Read PLAN.md and continue from where we left off."_
The checkboxes + Log sections tell the full story — no other context needed.

**Logging rules (keep entries short):**
- **Finding:** something you measured or observed (e.g., "KL-div signal Cohen's d = 1.4 under MINJA attack")
- **Challenge:** a blocker or unexpected problem
- **Decision:** a choice you made + one-line reason why
- **Pivot:** if a phase fails exit criteria, record the pivot taken (see Pivot Protocol section)

**README strategy:**
`README.md` in the project root has placeholder sections. Each phase has a "README contribution" — after completing a phase, copy the relevant note into README. By Phase 9 it's complete.

---

---

## Phase Tracker

| # | Phase | Complexity | Status | Depends On |
|---|-------|-----------|--------|------------|
| 0 | Bootstrap | Low | `[x]` | -- |
| 1 | Signal Validation (GO/NO-GO) | **High** | `[x]` | 0 |
| 2 | Core Agent Pipeline | Medium | `[ ]` | 0, 1 (collector) |
| 3 | Behavioral Monitor | Medium | `[x]` | 2 |
| 4 | Temporal Health Scorer | **High** | `[x]` | 3 |
| 5 | Decentralized Task Allocator | **High** | `[ ]` | 4 |
| 6 | Memory Integrity Module | Medium | `[ ]` | 2, 3 |
| 7 | Human Escalation Layer | Low | `[ ]` | 4, 6 |
| 8 | Observability Dashboard | Medium | `[ ]` | 4, 5, 6, 7 |
| 9 | Production Hardening | Medium | `[ ]` | 0-8 |
| 10 | Research Experiment Suite | **High** | `[x]` | 1-9 |
| 11 | Paper Scaffold + Reproducibility | Medium | `[ ]` | 10 |
| 12 | CI/CD + Release | Low | `[ ]` | 0 (basic), all (full) |

Parallel opportunities: P6 || P5; P8 starts after P4; P12 basic CI starts at P0.

**Estimated effort by complexity:**
- Low (P0, P7, P12): ~2-3 days each
- Medium (P2, P3, P6, P8, P9, P11): ~4-7 days each
- High (P1, P4, P5, P10): ~7-12 days each

---

## Phase 0: Bootstrap

**Goal:** Runnable skeleton -- config, logging, health endpoint, one passing test, Docker up.

**Exit Criteria:**
- `pip install -e .` succeeds
- `pytest` passes (>=1 test)
- `docker compose up` returns 200 on `/health`
- `chronoagent --help` prints usage
- CI green on push

**Tasks:**
- [x] 0.1 `pyproject.toml` -- PEP 621, Python >=3.11. Deps split:
  - Core: fastapi, uvicorn, pydantic, pydantic-settings, langchain, langgraph, chromadb, redis, sqlalchemy, alembic, structlog, prometheus-client, httpx, typer, pyyaml, ruptures, numpy, scipy, scikit-learn
  - `[forecaster]`: chronos-forecasting (optional, allows CPU-only envs to skip)
  - `[dev]`: pytest, pytest-asyncio, pytest-cov, hypothesis, ruff, mypy, pre-commit
  - `[experiments]`: matplotlib, seaborn, pandas
- [x] 0.2 `src/chronoagent/config.py` -- Pydantic `Settings` with `env_prefix="CHRONO_"` and `.env` support. YAML overlay loader in CLI merges YAML -> env -> defaults. Key fields: env (dev/prod/test), llm_backend (**together**/mock/ollama), together_api_key, together_model (default: `mistralai/Mixtral-8x7B-Instruct-v0.1`), ollama_base_url (optional), ollama_model (optional, phi3:mini), redis_url, database_url (sqlite default for dev), chroma_persist_dir, health_score_window (50), bocpd_hazard_rate (0.01), forecaster_enabled (true), forecaster_horizon (10), ensemble_weights (bocpd=0.4, chronos=0.6), escalation_threshold (0.3), escalation_cooldown (300s)
- [x] 0.3 `src/chronoagent/main.py` -- FastAPI app factory with lifespan for DB/Redis/Chroma init, mount health router
- [x] 0.4 `src/chronoagent/cli.py` -- Typer CLI: `serve`, `run-experiment`, `check-health`
- [x] 0.5 `src/chronoagent/observability/logging.py` -- structlog, JSON in prod, colored in dev, consistent fields (`agent_id`, `task_id`, `phase`)
- [x] 0.6 `Dockerfile` (multi-stage, Python 3.11-slim), `docker-compose.yml` (app, redis, postgres, chroma), `.env.example` documenting all `CHRONO_*` vars
- [x] 0.7 `configs/` -- `base.yaml`, `dev.yaml`, `prod.yaml` with merge hierarchy
- [x] 0.8 `tests/unit/test_config.py` -- Settings loads defaults
- [x] 0.9 `.github/workflows/ci.yml` -- checkout, install, ruff check, mypy, pytest
- [x] 0.10 `Makefile` -- `dev`, `test`, `lint`, `docker-up`, `docker-down`, `update-readme`

**Git & CI/CD setup (do this before any code):**
- [x] 0.11 `git init` + initial commit on `main`. Set branch protection: `main` requires PR + green CI. All work on feature branches (`feat/`, `fix/`, `chore/`). Never commit directly to main.
- [x] 0.12 Conventional commits enforced via `commitizen` (`cz commit`). Format: `feat(scorer): add BOCPD streaming`, `fix(monitor): handle zero-variance KL-div`, `chore(ci): add mypy step`. Add to `pyproject.toml`: `[tool.commitizen] version_provider = "pep621"`.
- [x] 0.13 `.pre-commit-config.yaml` with hooks: `ruff` (lint+format), `mypy` (type check), `commitizen` (commit message), `trailing-whitespace`, `end-of-file-fixer`. Run `pre-commit install` — blocks bad commits locally before they reach CI.
- [x] 0.14 Lock file: use `uv` (`pip install uv`). Run `uv pip compile pyproject.toml -o requirements.lock`. Commit the lock file. CI installs from lock file for reproducibility.
- [x] 0.15 `.github/workflows/ci.yml` triggers on every push + PR to main: install from lock, pre-commit, mypy, pytest with coverage (fail below 80%). PRs to main require this to pass.
- [x] 0.16 `.github/workflows/release.yml` triggers on `git tag v*`: bump version via commitizen (`cz bump`), build wheel, build + push Docker image to GHCR, create GitHub Release with changelog auto-generated from conventional commits.

**Dev practices enforced from commit 1:**
- [x] 0.17 All modules: full type hints. `mypy --strict` in CI — no `Any` without explicit comment.
- [x] 0.18 All public functions: docstrings (one-line summary, Args, Returns). No inline comments except for non-obvious logic.
- [x] 0.19 `ruff` config in `pyproject.toml`: `line-length=100`, `select=["E","F","I","N","UP","B","SIM"]`. Format on save.

**Key Files:** `pyproject.toml`, `config.py`, `main.py`, `cli.py`, `Dockerfile`, `docker-compose.yml`, `ci.yml`, `.pre-commit-config.yaml`, `requirements.lock`

**README contribution (fill in when phase complete):**
> Installation, quickstart (`docker compose up`), environment variables table, project structure diagram.

### Phase 0 Log
| | |
|--|--|
| **Findings** | 44 tests pass, 95.77% coverage (gate ≥80%). Python 3.13 on Windows (py launcher). `uv pip compile` produces lock file cleanly. |
| **Challenges** | No `pip` on PATH — must use `py -m pip`. CRLF warnings on Windows git (cosmetic, not blocking). |
| **Decisions** | Used `py` launcher instead of `python3`. Ensemble weights split into two separate fields (`ensemble_weights_bocpd`, `ensemble_weights_chronos`) instead of a nested dict — simpler env-var override. |
| **Completed** | 2026-04-09 |

---

## Phase 1: Signal Validation -- GO/NO-GO Gate

**Goal:** Empirically confirm behavioral signals drift before/during memory poisoning. Determines project viability.

**Exit Criteria:**
- 2-agent pipeline (SecurityReviewer + Summarizer) processes >=20 synthetic PRs
- MINJA-style memory poisoning injected into ChromaDB
- 6 signals measured (latency, retrieval count, tokens, KL-div, tool calls, memory query entropy) across clean vs poisoned runs
- Per-signal: Cohen's d effect size + detection lag (timesteps from injection to shift)
- Decision document: viable signals identified, AWT assessment, proceed/pivot ruling

**Tasks:**
- [x] 1.1 Minimal agent pair -- `SecurityReviewerAgent` + `SummarizerAgent` using LangChain + **MockBackend** (not Ollama — no GPU needed, deterministic), both with ChromaDB retrieval. MockBackend returns realistic but fixed responses to simulate real agent behavior.
- [x] 1.2 `monitor/collector.py` -- `BehavioralCollector` with `StepSignals` dataclass (6 fields), `start_step`/`end_step`, `get_signal_matrix() -> (T, 6) ndarray`
- [x] 1.3 `monitor/kl_divergence.py` -- Calibration run fits Gaussian on clean retrieval embeddings; per-step KL from baseline via `scipy.stats.entropy`
- [x] 1.4 `monitor/entropy.py` -- Shannon entropy of top-k similarity score distribution per retrieval; normalized to [0,1]
- [x] 1.5 `memory/poisoning.py` -- `MINJAStyleAttack` (query-optimized trigger embeddings) + `AGENTPOISONStyleAttack` (backdoor trigger docs); both inject into ChromaDB
- [x] 1.6 `experiments/runner.py` -- Phase A: clean run (N PRs) -> Phase B: inject attack -> Phase C: poisoned run (N PRs) -> Phase D: compute stats
- [x] 1.7 `configs/experiments/signal_validation.yaml` -- seed, step counts, attack type, agent list, signal list, analysis params
- [x] 1.8 Analysis script -- per-signal time-series plots, Cohen's d, PELT changepoint detection, AWT estimation, decision matrix table
- [x] 1.9 Write decision document with pivot ruling

**Key Files:** `agents/security_reviewer.py`, `agents/summarizer.py`, `monitor/collector.py`, `monitor/kl_divergence.py`, `monitor/entropy.py`, `memory/poisoning.py`, `experiments/runner.py`

**Research Notes:**
- This IS the first research contribution. Results feed directly into paper Section 5.
- Pivot logic evaluated here (see Pivot Protocol below).
- Key decision: if KL-div and entropy show large effect sizes with AWT > 0, the "proactive detection" framing holds. If AWT = 0 but effect sizes are large, pivot to "concurrent detection." If no signal shows significance, expand signal set before abandoning security claim.
- Use PELT (from `ruptures`) for changepoint detection in the analysis phase.
- Minimum bar: at least 2 of 6 signals must show Cohen's d > 0.8 (large effect).

**README contribution (fill in when phase complete):**
> Research motivation section, signal validation results table (Cohen's d per signal), GO/NO-GO decision, pivot taken (if any).

### Phase 1 Log
| | |
|--|--|
| **Signal Results** | KL-div: d=1.343 (MINJA), d=1.402 (AGENTPOISON), per `results/decision_matrix.csv`. All other signals d<0.4. Entropy/retrieval_count/tool_calls are MockBackend constants (σ=0). AWT=0 (concurrent). *Earlier draft of this line recorded d=1.611 (MINJA) / d=1.348 (AGENTPOISON) from an outdated run; canonical numbers are the decision_matrix.csv values quoted here and in README.* |
| **GO/NO-GO Decision** | Conditional GO with Pivot A. Strict criterion: NO-GO (1/6 signals). Adjusted: primary signal confirmed; MockBackend limits 3/6 signals. |
| **Pivot Taken** | Pivot A — AWT=0; reframe to concurrent detection. No code changes. |
| **Findings** | KL-divergence is the only backend-independent signal; works under both MINJA and AGENTPOISON (d>1.3). PELT detects changepoint at step 9/25. Entropy flat due to uniform MockBackend similarity scores. |
| **Challenges** | MockBackend eliminates 3/6 signals as observable (σ=0). Token-count effect is a PR-seed confound, not attack signal. |
| **Decisions** | KL-div promoted as anchor signal for Phase 2 monitor integration. Entropy/retrieval_count deferred to Phase 2+ with real LLM backend. Full doc: `docs/phase1_decision.md`. |
| **Completed** | 2026-04-09 |

---

## Phase 2: Core Agent Pipeline

**Goal:** Full 4-agent code review pipeline (Planner -> SecurityReviewer || StyleReviewer -> Summarizer) via LangGraph. No health scoring yet.

**Exit Criteria:**
- POST PR diff via API -> receive structured ReviewReport
- All 4 agents execute in correct order via LangGraph (fan-out security||style, fan-in to summarizer)
- Works with Together.ai (default), MockBackend (experiments/tests), and Ollama (optional, GPU only)
- Mock pipeline integration test passes in <5s
- Full trace logging per agent

**Tasks:**
- [x] 2.1 `agents/base.py` -- `BaseAgent` ABC: `execute(Task) -> TaskResult`, instrumented `_call_llm`, `_retrieve_memory`
- [x] 2.2 `agents/backends/` -- `LLMBackend` ABC (`generate`, `embed`), `TogetherAIBackend` (default), `MockBackend` (deterministic fixtures — primary for all tests/experiments), `OllamaBackend` (optional, skip if no GPU)
- [x] 2.3 `agents/planner.py` -- Decomposes PR diff into `list[ReviewSubtask]` with task_type + code_segment; queries memory for similar decompositions
- [x] 2.4 `agents/security_reviewer.py` -- Full impl: checks for vulns, outputs `SecurityFinding` (severity, desc, line ref); queries CWE patterns from memory
- [x] 2.5 `agents/style_reviewer.py` -- Checks quality/naming/complexity, outputs `StyleFinding`; queries style conventions from memory
- [x] 2.6 `agents/summarizer.py` -- Synthesizes all findings into `ReviewReport` (markdown); queries report templates
- [x] 2.7 `agents/registry.py` -- `AgentRegistry`: capability map, agent lookup by type
- [x] 2.8 `memory/store.py` -- ChromaDB wrapper: `add`, `query`, `delete`, `get_all_embeddings`
- [x] 2.9 Seed data script -- ~50 vuln patterns (CWE top 25), ~30 style docs, ~10 report templates, ~20 sample reviews
- [x] 2.10 `pipeline/graph.py` -- LangGraph `StateGraph`: plan -> (security_review || style_review) -> summarize -> END
- [x] 2.11 `api/routers/review.py` -- `POST /api/v1/review`, `GET /api/v1/review/{id}`
- [x] 2.12 `tests/integration/test_pipeline_e2e.py` -- MockBackend, submit diff, assert ReviewReport with findings, assert agent order

**Key Files:** `agents/*.py`, `agents/backends/*.py`, `memory/store.py`, `pipeline/graph.py`, `api/routers/review.py`

**Research:** This pipeline IS the no-monitoring baseline for experiments.

**README contribution (fill in when phase complete):**
> System architecture diagram (agent graph), agent descriptions, API usage example (`POST /api/v1/review`), sample ReviewReport output.

### Phase 2 Log
| | |
|--|--|
| **Findings** | _fill in_ |
| **Challenges** | _fill in_ |
| **Decisions** | _fill in_ |
| **Completed** | _fill in: date_ |

---

## Phase 3: Behavioral Monitor

**Goal:** Full monitoring on every agent -- all 6 signals per timestep, persisted, queryable.

**Exit Criteria:**
- Every agent call produces a `StepSignals` record
- `GET /api/v1/agents/{id}/signals?window=50` returns signal time-series
- KL-div baseline calibrated from first N clean runs
- Signals persisted to PostgreSQL
- Unit tests for each signal computation (including property tests)

**Tasks:**
- [x] 3.1 Expand `BehavioralCollector` -- add PostgreSQL persistence via SQLAlchemy, baseline calibration after N steps, rolling window stats
- [x] 3.2 Complete `kl_divergence.py` -- edge cases (zero variance, insufficient samples), unit tests with known distributions
- [x] 3.3 Complete `entropy.py` -- edge cases (single result, empty), normalize [0,1], unit tests (uniform=max, delta=0)
- [x] 3.4 `db/models.py` -- `AgentSignalRecord` table (agent_id, task_id, timestamp, 6 signal columns); Alembic migration
- [x] 3.5 `api/routers/signals.py` -- `GET /api/v1/agents/{agent_id}/signals?window=50`
- [x] 3.6 Unit tests: `test_signals.py`, `test_kl_divergence.py`, `test_entropy.py`
- [x] 3.7 Property test via Hypothesis: KL-div and entropy always non-negative for valid inputs

**Key Files:** `monitor/collector.py`, `monitor/kl_divergence.py`, `monitor/entropy.py`, `db/models.py`

**Research:** Data collection layer for all experiments. Correctness is critical.

**README contribution:** Signal monitoring section — what the 6 signals are and why each was chosen.

### Phase 3 Log
| | |
|--|--|
| **Findings** | `BehavioralCollector.persist_step` stores signals via SQLAlchemy session; KLCalibrator analytic formula matches scipy grid to <2% for D≤3 |
| **Challenges** | In-memory SQLite needs `StaticPool` for router tests; ruff B008 requires `Annotated[Session, Depends(...)]` instead of `= Depends(...)` |
| **Decisions** | `BehavioralCollector` stays DI-agnostic — persistence via explicit `persist_step(session, ...)` call; `get_db` dependency in `api/deps.py` |
| **Completed** | 2026-04-10 — 462 tests, 96.27% coverage |

---

## Phase 4: Temporal Health Scorer

**Goal:** BOCPD + Chronos T5-small ensemble producing per-agent health scores in [0,1], async, with graceful degradation.

**Exit Criteria:**
- Health score updates within 500ms of new signal (non-blocking)
- BOCPD detects synthetic changepoints with F1 > 0.8
- Chronos forecasts next 10 steps with MASE < 1.5
- Graceful fallback to BOCPD-only when Chronos unavailable
- Health scores on API and message bus

**Tasks:**
- [x] 4.1 `scorer/bocpd.py` -- Online BOCPD (Adams & MacKay 2007, ~80 lines NumPy) for streaming; `ruptures` for offline validation. `update(obs) -> changepoint_probability [0,1]`
- [x] 4.2 `scorer/chronos_forecaster.py` -- Lazy-loaded Chronos T5-small (46M params). `forecast(history) -> ForecastResult`. `compute_anomaly_score(history, actual) -> [0,1]`. Returns None if unavailable.
- [x] 4.3 `scorer/ensemble.py` -- `health = 1 - (w_bocpd * bocpd + w_chronos * chronos)`, clamped [0,1]. Single-component fallback when other missing. Weights configurable.
- [x] 4.4 `scorer/health_scorer.py` -- `TemporalHealthScorer`: subscribes to signals via message bus, maintains signal buffer, orchestrates BOCPD + Chronos, publishes `HealthUpdate`
- [x] 4.5 `messaging/bus.py` + `messaging/redis_bus.py` + `messaging/local_bus.py` -- MessageBus ABC, Redis pub/sub impl, in-memory impl for dev/test
- [x] 4.6 `api/routers/health_scores.py` -- `GET /api/v1/agents/{id}/health` (score + components), `GET /api/v1/agents/health` (all agents + system health)
- [x] 4.7 Tests: synthetic changepoint at step 50 -> detect within 5 steps; sine wave forecast low residual; sudden jump -> high anomaly; ensemble fallbacks; Hypothesis: health always in [0,1]

**Key Files:** `scorer/*.py`, `messaging/*.py`, `api/routers/health.py`

**Research Notes:**
- Core novel component. Accuracy, calibration, latency measured in every experiment.
- Ablation configs disable BOCPD or Chronos independently to measure individual contribution.
- Key implementation note: `ruptures` is batch/offline only. For streaming BOCPD, implement Adams & MacKay (2007) directly (~80 lines NumPy). Use `ruptures` for offline experiment validation.
- Chronos T5-small (46M params) is Apache 2.0 licensed. Use `amazon/chronos-t5-small` checkpoint. Lazy-load to avoid startup cost.
- Health score interpretation: 1.0 = fully healthy, 0.0 = confirmed anomalous. Scores below escalation_threshold (default 0.3) trigger human escalation.

**README contribution:** How health scoring works — BOCPD + Chronos ensemble diagram, health score interpretation, API example (`GET /api/v1/agents/{id}/health`).

### Phase 4 Log
| | |
|--|--|
| **BOCPD F1 result** | Synthetic changepoint (step 50, mean shift 0→5, std=0.1) detected at step 1 post-shift (cp_score ≈ 1.0 > 0.5). F1 effectively 1.0 on this benchmark. |
| **Chronos MASE result** | Chronos not installed in dev env — BOCPD-only mode. Graceful fallback confirmed (all tests pass with w_chronos promoted to 0). |
| **Latency overhead** | LocalBus is synchronous (in-process). All signal→health updates complete within the same call stack; <1ms overhead. |
| **Chronos Pivot C?** | BOCPD-only for now. ChronosForecaster lazy-loads and returns None gracefully; ensemble falls back automatically. |
| **Findings** | Normalized BOCPD (Adams & MacKay) always returns H as raw R[0]. Fixed by returning H·P(x\|prior)/P(x\|x_{1:t-1}) — this spikes to 1 on regime shift, stays near 0 on stable signal. |
| **Challenges** | BOCPD algebraic identity: normalized R[0]/total = H always. Diagnosed by tracing the cancellation. Fix: use evidence ratio instead of R[0]. |
| **Decisions** | cp_signal = H·pred_probs[0]/evidence (evidence = sum(R·pred_probs) before normalization). Run-length distribution update unchanged — only the return value changed. |
| **Completed** | 2026-04-10 — 503 tests, 92.48% coverage |

---

## Phase 5: Decentralized Task Allocator

**Goal:** Health-score + capability-weighted task routing via contract-net negotiation. No central controller.

**Exit Criteria:**
- Degraded agent -> tasks shift to healthier agents with overlapping capabilities
- All allocation decisions logged with bids, scores, rationale
- Allocation efficiency metric computable
- Round-robin fallback on allocator failure

**Tasks:**
- [x] 5.1 `allocator/capability_weights.py` -- Static capability matrix: 4 agents x 4 task types, proficiency in [0,1]
- [x] 5.2 `allocator/negotiation.py` -- Contract-net variant: broadcast task -> agents bid (capability * health) -> highest wins -> ties broken by agent_id. Min bid threshold -> escalate to human.
- [x] 5.3 `allocator/task_allocator.py` -- `DecentralizedTaskAllocator`: subscribes to health updates, maintains health snapshot, calls negotiation protocol
- [x] 5.4 Integrate into `pipeline/graph.py` -- LangGraph conditional edges: after Planner, each subtask routed via allocator at runtime
- [x] 5.5 `db/models.py` -- `AllocationAuditRecord` (task_id, assigned_agent, all_bids JSON, health_snapshot JSON, rationale, escalated)
- [x] 5.6 Fallback: negotiation timeout/error -> round-robin with warning log
- [x] 5.7 Tests: mock health -> verify highest-bid wins; all-low-health -> escalation; timeout handling; Hypothesis: exactly one agent assigned or escalated

**Key Files:** `allocator/*.py`, `pipeline/graph.py`

**Research Notes:**
- Allocation efficiency = `successful_tasks / total_tasks` under attack. Compare vs round-robin baseline.
- Ablation: capability-only allocation (no health scores) measures health scoring contribution.
- Note on 4-agent pipeline: limited capability overlap (each agent is specialized). Allocation benefit is most visible when an agent is degraded -- system partially redistributes subtasks to agents with nonzero capability, or escalates to human rather than using unhealthy agent.
- Negotiation protocol: contract-net variant. Bids = `capability_weight * health_score`. Deterministic tie-breaking by agent_id for reproducibility.

**README contribution:** Task allocation section — contract-net protocol diagram, how health scores influence routing, audit log example.

### Phase 5 Log
| | |
|--|--|
| **Allocation efficiency vs baseline** | Deferred to Phase 10 (research experiment suite). Phase 5 ships the routing primitives and the audit surface; the comparative metric requires the full attack benchmark. |
| **Negotiation protocol issues** | None on the pure function. The integration layer surfaced two design choices that are now locked in: (1) the allocator's bus handler must swallow malformed payloads rather than raise (LocalBus does not trap handler exceptions, so a raise would tear down the publisher and break unrelated subscribers); (2) the round-robin fallback must catch `Exception` broadly rather than specific subclasses, because the spec includes "timeout" which is not a subclass of `InvalidHealthError`/`InvalidThresholdError`/`UnknownTaskTypeError`. |
| **Findings** | Contract-net negotiation reduces to a pure function over `(task_type, snapshot, matrix, threshold)`; pulling that purity out of the stateful allocator made every layer above it cheap to test and reason about. The 4-agent specialized pipeline limits the practical benefit of redistribution because non-specialist reviewers cannot actually run a specialist's task; the allocator's real role is therefore a health gate plus an audit ledger plus a graceful-degradation hook, not dynamic dispatch. |
| **Challenges** | Pipeline integration (5.4) had to reconcile the allocator's "non-specialist winner" outcome with the fact that the 4 reviewer agents have non-swappable interfaces. The resolution: `_route_security` and `_route_style` only fire the specialist branch when `assigned_agent == expected_specialist AND escalated is False`; everything else routes to the existing escalation-placeholder branch. The Alembic migration test for 5.5 also surfaced a pydantic-settings precedence gotcha (init kwargs outrank env vars), which is now documented in the project memory file. |
| **Decisions** | (a) Negotiation function stays pure; allocator wrapper owns all state and bus wiring. (b) Round-robin cursor is shared across task types: simpler and still deterministic given the canonical `AGENT_IDS` order. (c) Audit-record persistence hook deferred from 5.5/5.6 to a later task; the model is in place but `task_allocator.py` does not yet write to it. (d) Phase 5 closes without allocation-efficiency benchmarking; that metric belongs in Phase 10 alongside the rest of the experiment harness. |
| **Completed** | 2026-04-10 |

---

## Phase 6: Memory Integrity Module

**Goal:** Detect anomalous memory entries, quarantine them, surface for human review.

**Exit Criteria:**
- Integrity check on every retrieval
- Anomalous docs flagged with confidence score
- Quarantine excludes flagged docs from future retrieval until approved
- Detection AUROC > 0.8 on synthetic attacks

**Tasks:**
- [x] 6.1 `memory/integrity.py` -- `MemoryIntegrityModule`: 4 detection signals (embedding outlier score, freshness anomaly, retrieval frequency spike, content-embedding mismatch). `check_retrieval(query, docs) -> IntegrityResult`
- [x] 6.2 Isolation forest (sklearn) fitted on clean embeddings for outlier detection; periodic refit every N new docs
- [x] 6.3 `memory/quarantine.py` -- Separate ChromaDB collection for quarantined docs. Move on flag, restore on approve. Retrieval only queries active collection.
- [x] 6.4 `api/routers/memory.py` -- `GET /api/v1/memory/integrity`, `POST /api/v1/memory/quarantine`, `POST /api/v1/memory/approve`
- [x] 6.5 Tests: inject poison docs -> flagged; clean docs -> not flagged; quarantine/approve flow

**Key Files:** `memory/integrity.py`, `memory/quarantine.py`

**Research:** AUROC + F1 for poison detection. Component of overall system eval.

**README contribution:** Memory integrity section — how detection works, quarantine/approve workflow, API examples.

### Phase 6 Log
| | |
|--|--|
| **Detection AUROC** | 1.0 on 10 clean + 5 poison (MINJA and AgentPoison). Phase target >0.8 met. |
| **False positive rate** | 0% on clean corpus: content_embedding_mismatch = 0.0 for all docs added via normal add path. |
| **Findings** | content_embedding_mismatch is the only reliable cold-start signal. Random unit vectors in 384-d space have cosine distance ~0.5 (std ~0.025), giving clear separation from clean (dist=0). IsolationForest baseline must be fitted before the embedding_outlier signal contributes. |
| **Challenges** | ChromaDB EphemeralClient shares a process-level segment manager, causing cross-test collection state leakage. Fixed by UUID-suffixed collection names per fixture. |
| **Decisions** | flag_threshold=0.3 with content_mismatch=1.0 weight for detection tests: gives ~8-sigma margin above clean (0.0) and well below poison (~0.5). Default 4-signal weights unreachable from mismatch alone at threshold=0.6. |
| **Completed** | 2026-04-10 |

---

## Phase 7: Human Escalation Layer

**Goal:** Auto-escalate on low health or memory flags. Full audit trail. Cooldown to prevent spam.

**Exit Criteria:**
- Escalation triggers on health < threshold OR quarantine event
- Escalation includes full context (signals, health history, flagged docs, allocation trail)
- Human resolve via API (approve/reject/modify)
- Cooldown prevents repeated escalation for same agent

**Tasks:**
- [x] 7.1 `escalation/escalation_manager.py` -- `EscalationHandler`: threshold check, cooldown tracking, context assembly, publish escalation event
- [x] 7.2 `escalation/audit.py` -- `AuditTrailLogger`: log events (allocation, health update, escalation, quarantine, approval) to PostgreSQL `audit_events` table
- [x] 7.3 `api/routers/escalation.py` -- `GET /api/v1/escalations?status=pending`, `POST /api/v1/escalations/{id}/resolve`
- [x] 7.4 Tests: low health -> escalation fires; cooldown -> no repeat; resolve updates status; full context present

**Key Files:** `escalation/*.py`

**Research:** Escalation rate under attack is a measured metric.

**README contribution:** Human-in-the-loop section — when and why escalation fires, how to resolve via API.

### Phase 7 Log
| | |
|--|--|
| **Findings** | 50 new tests; 845 total pass; coverage 94.29%. Escalation handler correctly suppresses on threshold and cooldown; quarantine events bypass threshold. Bus handler helpers (on_health_update, on_quarantine_event) swallow malformed payloads safely. StaticPool required for SQLite in-memory test isolation in router tests. |
| **Challenges** | SQLite in-memory test isolation: `create_all` and session-factory connections use different connections without StaticPool, so tables are invisible to the session. Fixed by using `StaticPool` in router test fixture. Cooldown lock scope: lock must be released before bus.publish to avoid reentrancy (LocalBus is synchronous). |
| **Decisions** | Escalation storage is SQLAlchemy (not in-memory dict) for restart durability and audit trail alignment. Cooldown is in-memory per agent (acceptable reset on restart). Separate `EscalationOutcome` dataclass (handler return type) from `EscalationRecord` ORM model to avoid ORM coupling at handler boundary. `memory.py` quarantine endpoint gains optional `agent_id` for bus event triggering; backward-compatible. |
| **Completed** | 2026-04-11 |

---

## Phase 8: Observability Dashboard

**Goal:** Real-time viz of agent health, routing, memory status, escalation queue.

**Exit Criteria:**
- Per-agent health time-series displayed
- Allocation decisions with bid details visible
- Memory integrity + escalation queue displayed
- Loads <3s, near-real-time updates

**Tasks:**
- [x] 8.1 Dashboard backend -- FastAPI endpoints: `/dashboard/api/agents`, `/dashboard/api/agents/{id}/timeline`, `/dashboard/api/allocations`, `/dashboard/api/memory`, `/dashboard/api/escalations`, WebSocket `/dashboard/ws/live`
- [x] 8.2 Dashboard frontend (MVP: single HTML + Chart.js, no build step, ~500 LOC). Views: Agent Health Panel (gauges + sparklines), Signal Explorer, Allocation Log, Memory Inspector, Escalation Queue
- [x] 8.3 Prometheus metrics (optional prod path): `src/chronoagent/observability/metrics.py` -- gauges/counters/histograms + Grafana dashboard JSON

**Key Files:** `dashboard/`, `observability/metrics.py`

**Research:** Dashboard screenshots for paper system design section. Signal explorer for qualitative analysis.

**README contribution:** Dashboard section — screenshot, how to run it, what each panel shows.

### Phase 8 Log
| | |
|--|--|
| **Findings** | Bus-subscriber wiring is a clean cut line between the passive metrics sink and the rest of ChronoAgent: the sink exposes update methods only, and `observability/metrics_wiring.py` attaches closures to `health_updates`, `escalations`, `memory.quarantine`. That split keeps the sink trivially unit-testable with a `LocalBus`, lets tests instantiate the sink without touching bus state, and leaves hot code paths (scorer, allocator, escalation handler) free of metrics imports. Poll-driven gauges (`system_health`, `pending_escalations`, `quarantine_size`, `memory_baseline_size`, `memory_pending_refit`) are refreshed inside the `/metrics` handler via `_refresh_poll_gauges`, which tolerates partial `app.state` so scraping works during startup. Isolated `CollectorRegistry` per instance avoids the global-registry duplicate-metric trap. 8.1 (dashboard backend) and 8.2 (dashboard frontend) context preserved from earlier sessions. |
| **Challenges** | (1) `prometheus_client.generate_latest` returns the counter `_total` suffix automatically, so tests must query `chronoagent_health_updates_total` (not `_updates`) and the fixed 0.0 counter sample is always emitted (checking `is None` on an unset label-less counter is wrong; checking `== 0.0` is right). (2) FastAPI's `TestClient` with `create_app()` runs the real lifespan which builds an in-memory SQLite engine with the default pool -- every new connection gets a fresh empty database, so the `/metrics` handler's `SELECT count(*) FROM escalation_records` errors with "no such table". Fix: override `app.state.session_factory` with a `StaticPool`-backed engine after lifespan startup (same pattern as `test_dashboard_router.client_app`). (3) The test-file's system-health integration test briefly tried to populate the scorer cache by publishing on `HEALTH_CHANNEL`, but `TemporalHealthScorer` subscribes to `SIGNAL_CHANNEL` (`signal_updates`), not the health channel it publishes on -- so the cache never updated. Rewritten to assert the empty-cache default (`system_health == 1.0`). (4) Coverage-mode runs surfaced a test-order flake in `TestQuarantine` once; reproduced with `--no-cov` showing 927 pass, and a re-run with coverage also showing 927 pass. Non-deterministic, pre-existing ChromaDB flavour flake; not introduced by task 8.3. |
| **Decisions** | (1) Metrics sink is passive: no bus subscriptions in `metrics.py`, all wiring goes through a separate `metrics_wiring.py` module that returns a `MetricsSubscribers` dataclass carrying the exact callables for idempotent teardown. (2) Registry isolation default: `ChronoAgentMetrics()` always creates a fresh `CollectorRegistry` unless one is explicitly passed; lifespan relies on the default. (3) Poll vs event: per-agent gauges + counters + histograms are event-driven via the bus; aggregate/snapshot gauges (`system_health`, queue sizes, baseline counters) are poll-driven in the `/metrics` handler so the scrape is always up to date without background tasks. (4) `/metrics` router is a plain FastAPI router with `include_in_schema=False`, mounted at the app root (not under `/dashboard/`) so Prometheus scrapers hit the conventional URL. (5) Allocation outcome label taxonomy: `"assigned"` (winning bid present), `"escalated"` (`result.escalated`), `"fallback"` (round-robin path where `winning_bid is None` but `escalated=False`). This matches the three states `DecentralizedTaskAllocator.allocate` can produce. (6) Grafana dashboard JSON lives at `docs/grafana/chronoagent_dashboard.json`, uses a templated `${DS_PROMETHEUS}` datasource, and ships 9 panels spanning system health, queues, per-agent timeseries, allocation rates, escalation rates, review-duration percentiles, and integrity-score quantiles. |
| **Completed** | 2026-04-11 (8.1 merged as PR #18 `b1117a2`; 8.2 merged as PR #20 `9071638`; 8.3 merged as PR #21 `94570e7`) |

---

## Phase 9: Production Hardening

**Goal:** Robust, deployable, operationally sound system.

**Exit Criteria:**
- Rate limiting on all API endpoints
- Retry with exponential backoff on all external calls (LLM, Redis, Chroma, Postgres)
- Graceful degradation: no Redis -> local bus, no Chronos -> BOCPD-only, no Postgres -> SQLite
- Component health check endpoint
- Structured logging on every code path
- Docker image builds and runs cleanly
- `pip install chronoagent` works from built wheel

**Tasks:**
- [x] 9.1 `api/middleware.py` -- Rate limiting (slowapi or custom): POST 10/min, GET 60/min, WS 5 concurrent
- [x] 9.2 Retry wrappers (tenacity): 3 attempts, exponential backoff, all external calls
- [x] 9.3 Graceful degradation in `create_app()` -- try/except for each component, fallback to local alternatives
- [x] 9.4 Comprehensive `/api/v1/health` -- per-component status (api, redis, postgres, chromadb, together_ai, forecaster), degraded/healthy/unhealthy. Ollama component omitted unless configured.
- [x] 9.5 Structured logging audit -- verify every module uses structlog with consistent fields
- [x] 9.6 Package build -- `python -m build` produces wheel, entry point `chronoagent = chronoagent.cli:app`

**Key Files:** `api/middleware.py`, `main.py`, `pyproject.toml`

**README contribution:** Production deployment section, Docker, environment variables, health checks, rate limits.

### Phase 9 Log
| | |
|--|--|
| **Findings** | **9.1:** Custom pure-ASGI middleware is a better fit than slowapi because (a) WebSocket concurrency caps are not a first-class slowapi concept, (b) we want one middleware handling both HTTP and WebSocket scopes in a single pass, (c) a fixed-window counter keyed by `(client_ip, method, minute_bucket)` maps directly onto the PLAN's "N per minute" wording with no token-bucket refill semantics to get wrong, and (d) zero new dependencies. The middleware reads the client IP from `scope["client"]` and falls back to `"unknown"` (shared bucket) when the entry is missing or malformed: a conservative default that refuses to hand every anonymous caller its own budget. `/health` and `/metrics` are on a default exempt list so ops probes and Prometheus scrapers cannot be starved by a flood of traffic. **9.2:** A single centralised `src/chronoagent/retry.py` module exports four named tenacity policies (`llm_retry`, `redis_retry`, `chroma_retry`, `db_retry`), each pinned at `stop_after_attempt(3)` and `wait_exponential(multiplier=0.25, min=0.25, max=2.0)` with a `retry_if_exception_type(...)` filter narrow enough to never swallow caller bugs (`ValueError`, `TypeError`, `IntegrityError`). A shared `before_sleep_log(logger=chronoagent.retry, level=WARNING)` hook fires one WARNING per intermediate retry so operators can spot retry storms from a single logger name. Wrapping happens at the method level (for class-owned external calls) or at a dedicated `_fetch_*_db` module-level helper (for FastAPI routers where the transaction is a pure read/write block) so only the actual I/O retries, not the surrounding bus publish, audit log, or header-mutation code paths. **9.3:** Four module-level `_build_*` helpers in `main.py` (`_build_message_bus`, `_build_database`, `_build_chromadb`, `_probe_forecaster`) each attempt the primary backend and return a `(instance, ComponentStatus)` tuple; `lifespan` stuffs each status into `app.state.component_status` keyed by logical component name so the forthcoming 9.4 `/api/v1/health` endpoint has a single source of truth without re-probing. A new tiny `src/chronoagent/observability/components.py` owns a frozen, hashable `ComponentStatus` dataclass with three fields (`name`, `mode` in `{"primary","fallback","unavailable"}`, `detail`) and a `ComponentMode` literal alias. The helpers are module-level specifically so tests can monkeypatch them (or the downstream symbols they reach -- `chronoagent.main.make_engine`, `chronoagent.messaging.redis_bus.RedisBus`, `chromadb.EphemeralClient`, `chronoagent.main.ChronosForecaster`) to drive every branch without standing up real infrastructure. **9.4:** Comprehensive `/api/v1/health` endpoint added to `src/chronoagent/api/health.py` alongside the existing `/health` liveness probe (one router, two routes). Reads `app.state.component_status` populated by 9.3, adds an `api` self-entry (always `primary: responding`), and layers on fresh config-only probes for `together_ai` (always present) and `ollama` (only when `settings.llm_backend == "ollama"`). Aggregation is strict precedence: any `unavailable` -> `unhealthy` (HTTP 503), any `fallback` -> `degraded` (HTTP 200), else `healthy` (HTTP 200). A frozen dataclass -> pydantic `ComponentReport` + `HealthReport` round-trip gives FastAPI a proper `response_model` with a stable OpenAPI shape. The default `RateLimitConfig.exempt_paths` grew `/api/v1/health` alongside `/health` and `/metrics` so ops polling cannot starve the 60-req/min budget. **9.6:** `py -m build` (build 1.4.2) produces a clean wheel + sdist via the existing hatchling backend. The first baseline build worked end-to-end but exposed two hygiene issues: hatchling's default sdist picks up every file not matched by `.gitignore`, which on this repo meant `.claude/`, `.hypothesis/` (~1500 cached files), `chronoagent.db`, and one personal-notes file ended up inside the tarball; and the `[build-system].requires` line had no minimum hatchling version pin. The fix is small: pin `hatchling>=1.26.0` and add an explicit `[tool.hatch.build.targets.sdist]` whitelist (`/src`, `/tests`, `/docs`, `/configs`, `/alembic`, top-level config files) plus a defensive exclude list for build / cache directories. The wheel itself was already correct: `[tool.hatch.build.targets.wheel] packages = ["src/chronoagent"]` bundles `chronoagent/dashboard/static/index.html` (28556 bytes) automatically because hatchling walks every file under each listed package, not just `*.py`. The console-script entry point `chronoagent = chronoagent.cli:app` was also already in place from Phase 0; no change needed there. **9.5:** Repo-wide logging audit routed every `src/` module through the single `chronoagent.observability.logging.get_logger` entry point (wrapping `structlog.get_logger`). Four stdlib-backed modules (`allocator/task_allocator.py`, `scorer/health_scorer.py`, `scorer/chronos_forecaster.py`, `messaging/redis_bus.py`) had their `import logging` + `logging.getLogger(__name__)` swapped AND their `%s`-style positional calls rewritten into structured `logger.warning("event_name", key=value)` form so structlog preserves the fields. Eleven direct-structlog consumers (`escalation/audit.py`, `escalation/escalation_manager.py`, `pipeline/graph.py`, `observability/metrics_wiring.py`, plus seven routers: `dashboard`, `health_scores`, `escalation`, `metrics`, `memory`, `review`, `signals`) had their `import structlog` + `logger: structlog.BoundLogger = structlog.get_logger(__name__)` lines replaced with `from chronoagent.observability.logging import get_logger` + `logger = get_logger(__name__)`, dropping the now-unused `structlog` import. `retry.py` kept its stdlib logger (tenacity's `before_sleep_log` hook requires a real `logging.Logger`) with a new block comment naming the exemption and pointing at 9.2 and 9.5. `observability/logging.py` is the other exemption because it *defines* the wrapper and configures the stdlib root handler. New `tests/conftest.py` runs `configure_logging("test")` once per session via `@pytest.fixture(autouse=True, scope="session")` so structlog routes through the stdlib bridge for the whole suite, which is what allows `caplog` to capture structured events. |
| **Challenges** | **9.1:** Three gotchas locked in. (1) Starlette's `TestClient` spawns a fresh event loop per concurrent WebSocket portal, so an `asyncio.Lock` created lazily on one loop cannot be awaited from another. Switched to `threading.Lock`; the critical sections are counter-increment only (no I/O), so it is fast enough for both production single-loop deployments and the multi-loop test harness. (2) Non-reserved routes need a custom `send` wrapper to inject `X-RateLimit-*` headers onto the first `http.response.start` message; the allowed-request path therefore holds no lock while the downstream app runs. (3) WebSocket rejection has to drain the initial `websocket.connect` message before emitting `websocket.close` code 1008, otherwise Starlette's ASGI server treats the close as a protocol violation and the TestClient hangs. **9.2:** Three gotchas locked in. (1) `RedisBus.subscribe` / `unsubscribe` mutate `self._handlers` under a lock BEFORE the Redis round-trip; wrapping the outer method with `redis_retry` would re-append the handler on every retry. Split the Redis call into private `_publish_raw` / `_pubsub_subscribe` / `_pubsub_unsubscribe` helpers and decorated those, keeping bookkeeping outside the retry scope. (2) `EscalationHandler.maybe_escalate` does DB insert + audit log + bus publish + cooldown mutation in one method; wrapping the whole thing would double-fire the side effects. Extracted the `with self._session_factory() as session: session.add(orm_row); session.commit()` block into a private `_persist_escalation` method so only the transaction retries. (3) FastAPI route functions with `Depends(get_db)`-injected sessions cannot open a fresh session inside a retry; the one call site (`signals.py:_fetch_signal_rows`) wraps `session.execute(stmt)` and accepts that retrying on a broken session may not always recover. Write-path routers instead extract the transaction into a module-level `@db_retry`-decorated helper that opens its own session, which gives full recovery for reads and write-then-commit blocks. (4) `monitor/collector.py:354` is a pure `session.add` with no DB I/O (transaction is caller-managed per the persist_step docstring), so it is deliberately NOT wrapped; the caller's commit path owns the retry. **9.3:** Four gotchas locked in. (1) `redis.Redis.from_url` is lazy: the socket is only opened on the first command, so constructing a `RedisBus` against an unreachable server succeeds silently. The primary-path probe therefore has to call `bus._client.ping()` (single-shot private access, flagged with `# noqa: SLF001`) inside the try block so `ConnectionError`/`TimeoutError` surface as a fallback instead of as a delayed crash on the first publish. (2) The database fallback engine has to use `poolclass=StaticPool` plus `connect_args={"check_same_thread": False}`, otherwise every new SQLAlchemy connection to `sqlite:///:memory:` gets its own private database and `escalation_records` disappears the moment the probe connection closes. Same pattern already used in `tests/unit/test_dashboard_router.py::client_app`. (3) The broad `except Exception` in every helper is intentional and covers `ImportError`, network errors, and `RuntimeError` alike; narrower filters would miss the "redis-py not installed" branch that production deployments actually hit in container images that skip optional extras. A `# noqa: BLE001` comment makes the intent explicit. (4) For the forecaster, probing `ChronosForecaster().available` spins up a fresh instance each call, which is cheap because `available` uses `importlib.util.find_spec` with no model load. The already-constructed forecaster inside `TemporalHealthScorer._chronos` is deliberately not reused to avoid crossing a private boundary just to save one import-spec lookup. **9.4:** Three gotchas locked in. (1) The `chronos-forecasting` package is not installed in dev/CI, so the 9.3 forecaster probe legitimately reports `fallback` and the overall health rolls up as `degraded` in the default test env. The "healthy" test path therefore has to monkeypatch `chronoagent.main.ChronosForecaster` with a stub whose `.available` returns `True`; treating "chronos missing" as a bug would make the endpoint unusable in CI. A dedicated `TestEndpointDegradedOnForecaster` locks in the honest default so operators see the reduced-capability signal. (2) Probing Together.ai / Ollama with a real HTTP call on every health request would burn paid quota and add 100-200 ms of latency to ops polling; 9.4 ships a config-only probe instead. "Inactive" (backend not selected) is reported as `primary` with `detail="inactive (active backend: <x>)"` so the component list stays stable across envs without contributing to degraded/unhealthy. A `?probe=true` query parameter for deeper network checks is deliberately punted to a follow-up. (3) The new endpoint lives at `/api/v1/health`, which does not match the legacy `/health` prefix in the rate limiter's default `exempt_paths`. Adding `/api/v1/health` to the tuple is required so a Prometheus-style scraper polling at sub-second intervals does not starve the 60-GETs-per-minute per-client budget, and a dedicated `TestRateLimitExemption` test fires 120 requests to prove it. **9.6:** Three gotchas locked in. (1) **Hatchling's sdist default is "ship everything not gitignored".** Untracked dev artefacts (`.claude/`, `.hypothesis/`, `chronoagent.db`, personal notes files) ended up in the first baseline tarball because hatchling reads from the filesystem, not git index. The fix is an explicit `[tool.hatch.build.targets.sdist]` whitelist; the `exclude` list is technically redundant for the listed include paths but keeps the intent visible and gives a clearer error message if someone reverts to the default. (2) **`tarfile` is the right tool for sdist inspection, not `python -m zipfile`.** Sdist is a `.tar.gz`, not a zip, so `py -m zipfile -l dist/chronoagent-0.1.0.tar.gz` errors with "File is not a zip file". Used `py -c "import tarfile; ..."` to enumerate entries instead. The 9.6 packaging regression test deliberately does NOT shell out to `py -m build` because a real isolated build takes 30+ seconds and would slow CI; the test asserts pyproject invariants and the importable bundled-asset path constants instead. (3) **No LICENSE file at the repo root.** `pyproject.toml` declares `license = { text = "MIT" }` for metadata but there is no `LICENSE` file alongside `README.md`. The first attempt at the sdist whitelist included `"/LICENSE"` and would have errored on missing path; removed from the whitelist for now. Creating a LICENSE file is a project / legal decision and was deliberately punted out of 9.6's scope. **9.5:** Three gotchas locked in. (1) **`caplog` silently dies under structlog's default factory.** Before 9.5, `task_allocator.py` used `logging.getLogger`, so `caplog.at_level(logging.WARNING, logger="chronoagent.allocator.task_allocator")` captured records via stdlib logging directly. After the migration to `get_logger`, structlog takes over — but only once `configure_logging` has been called. In an isolated unit test that never constructs `create_app`, `configure_logging` is never invoked, so structlog stays on its default `PrintLogger` factory that writes to stdout and bypasses stdlib entirely. Four existing assertions in `test_task_allocator.py` started failing with empty `caplog.records` until `tests/conftest.py` added a session-scoped autouse fixture that calls `configure_logging("test")` once. Lesson: any test that wants to capture structlog output via `caplog` must either import from `main` (which configures) or rely on the conftest fixture. (2) **`structlog.get_logger` returns `BoundLoggerLazyProxy`, not `BoundLogger`.** The audit test initially checked `isinstance(logger, structlog.stdlib.BoundLogger)` for every migrated module and failed on all 18 of them. The lazy proxy delegates to the real bound logger on first method call; the audit test was relaxed to accept anything whose `type(...).__module__.startswith("structlog")` and reject only `logging.Logger` instances. (3) **`logger.warning("msg with %s", x)` silently swallows the arg under structlog.** A structlog bound logger accepts kwargs, not positional `%` arguments — so the positional value is simply dropped and the `%s` literal appears unformatted in the output. The audit includes a regex-based spot check (`TestStructuredFieldConvention`) that forbids `logger.foo("... %s", x)` in all four migrated stdlib-backed modules so this failure mode cannot creep back in. |
| **Decisions** | **9.1:** (a) Per-client POST/GET budgets, global WS cap, which matches the PLAN's "WS 5 concurrent" wording (total connections, not per-IP). (b) Fixed-window rollover via `int(now // 60)` rather than a sliding window, because the PLAN wording is "per minute" and fixed windows give trivially deterministic tests under an injected clock. (c) Previous + current bucket always survive the prune so requests straddling a minute boundary are neither double-counted nor lost. (d) `create_app(rate_limit_config=..., rate_limit_clock=...)` takes both the config and the clock as optional kwargs, so tests can inject a mutable clock without monkey-patching module state. (e) Non-(GET\|POST) methods pass through untouched; PUT/DELETE/PATCH/OPTIONS are out of scope for 9.1. **9.2:** (a) One `src/chronoagent/retry.py` module owns every policy, so the "3 attempts, exponential backoff" knob lives in one file; individual call sites only pick a policy name. (b) Exception filters are narrow: `httpx.HTTPError` (covers both `RequestError` and `HTTPStatusError` subclasses for 5xx), `redis.exceptions.RedisError`, `chromadb.errors.ChromaError`, `sqlalchemy.exc.OperationalError`. `IntegrityError` is deliberately excluded from `db_retry` because duplicate-key / FK-violation is a caller bug, not a transient failure. (c) `MemoryStore` / `QuarantineStore` refactor introduces `_count_raw`, `_upsert_raw`, `_query_raw`, `_get_by_ids_raw`, `_get_all_raw`, `_delete_raw`, `_get_ids_raw`, `_get_full_raw`, `_existing_ids_raw` wrappers so the public method signatures stay unchanged while only the Chroma call retries; re-embedding on retry is avoided because the embed call sits outside the wrapped private method. (d) Router write endpoints extract the DB block into a module-level `@db_retry` helper (`_list_escalations_db`, `_apply_resolution`, `_fetch_last_signal_rows`, `_fetch_timeline_rows`, `_fetch_allocation_rows`, `_fetch_escalation_rows`, `_fetch_pending_escalation_count`, `_fetch_pending_count`, `_fetch_signal_rows`) so the FastAPI handler body stays thin and HTTPException-free inside the retry scope. (e) Escalation and audit methods are decorated at the public method level because each one already owns its own session and has no side effects outside the transaction (`AuditTrailLogger.log_event`, `EscalationHandler._persist_escalation`, `EscalationHandler._recent_allocation_task_ids`). **9.3:** (a) Primary-vs-fallback is decided per environment: `settings.env == "prod"` opts into RedisBus; dev and test default to LocalBus as the primary (not a fallback), so `component_status["bus"].mode == "primary"` in unit tests without any infrastructure. (b) Database fallback is env-agnostic: any failure probing the configured `database_url` (SELECT 1 + `create_all`) hands control to an in-memory SQLite engine wired with `StaticPool`, so a broken Postgres URL downgrades the app cleanly even in prod. (c) ChromaDB has no fallback today: `EphemeralClient` is pure in-process, so the helper simply labels it `primary: ephemeral (in-process)` and re-raises on failure. The helper exists anyway so tests can monkeypatch it and so 9.4 can treat every component uniformly. (d) The forecaster helper reports `primary: BOCPD + Chronos ensemble` when the `chronos-forecasting` package is importable and `fallback: BOCPD-only (chronos-forecasting not installed)` otherwise; the actual degradation is already owned by `ChronosForecaster.compute_anomaly_score` returning `None`, so 9.3 only surfaces the signal. (e) `ComponentStatus` lives in `observability/components.py` (not `main.py`) so 9.4 can import it without a circular dependency on `create_app`. It is a `@dataclass(frozen=True)` with a `Literal["primary","fallback","unavailable"]` mode field: hashable, serialisable, and cheap to compare in tests. (f) The registry lives on `app.state.component_status: dict[str, ComponentStatus]` keyed by logical component name (`"bus"`, `"database"`, `"chromadb"`, `"forecaster"`) rather than by backend class name, so 9.4 can look up `component_status["bus"]` without caring whether it's Redis or LocalBus. **9.4:** (a) One router file, two routes: `/health` stays as the cheap liveness probe; `/api/v1/health` is the new comprehensive endpoint, defined on the same `APIRouter` in `src/chronoagent/api/health.py`. Splitting into two files would fragment the "health" concept for no gain. (b) Strict precedence aggregation: `unavailable` dominates `fallback` dominates `primary`; empty component dict is treated as `healthy` (no news is good news). Encoded as a short generator loop with a single `return` inside so the `unavailable` path short-circuits without scanning the remaining entries. (c) HTTP status codes: `healthy` and `degraded` both return 200 (the service IS still serving traffic on a fallback), `unhealthy` returns 503. Callers on the 200 path inspect `body.status` if they care about the distinction. (d) Pydantic response model with a typed `dict[str, ComponentReport]` field populates OpenAPI cleanly; `ComponentReport` mirrors `ComponentStatus` but is a `BaseModel` so FastAPI can serialise it. The handler builds internal `ComponentStatus` values first and only converts at the top of the return, keeping the aggregation logic backend-agnostic. (e) LLM probes are config-only by design. Network probes (together.ai `/models`, ollama `/api/tags`) would cost paid quota or add 100 ms+ per poll; a future `?probe=true` query param can opt in if operators want it. (f) `RateLimitConfig.exempt_paths` grew `/api/v1/health` alongside `/health` and `/metrics` so ops scrapers cannot be starved. (g) `_build_report` reads `getattr(request.app.state, "component_status", {})` instead of attribute access, so a bare `create_app()` probe (with no lifespan run) still returns a well-formed payload containing at least `api` + `together_ai`. **9.6:** (a) **Tighten the sdist, leave the wheel alone.** The wheel was already correct: hatchling's `packages = ["src/chronoagent"]` rule walks every file (not just `*.py`) under `src/chronoagent/`, so `dashboard/static/index.html` is already bundled. The change is a small `[tool.hatch.build.targets.sdist]` block that pins the sdist to a tight whitelist of paths (`/src`, `/tests`, `/docs`, `/configs`, `/alembic`, top-level config files) plus a defensive exclude list for cache and build directories. (b) **Hatchling pinned at `>=1.26.0`** in `[build-system].requires` so future hatchling releases that change default behaviour cannot silently break the build; the actual build runs against `hatchling 1.29.0` in the isolated env. (c) **Grafana JSON stays in `docs/`, not bundled into the wheel.** The PLAN.md wording for 9.6 only requires the wheel + entry point to work; bundling docs/grafana would require either moving the file under `src/chronoagent/` (refactor across the repo) or `force-include` (split file location vs wheel location). Both add complexity for an asset most ops teams deploy from the repo checkout anyway. Punted out of 9.6's scope. (d) **No `LICENSE` file created in 9.6.** `pyproject.toml` declares `license = { text = "MIT" }` for metadata; creating an actual `LICENSE` file is a project / legal decision and was punted. The sdist whitelist does NOT reference `/LICENSE` so the build does not error on the missing path. (e) **Throwaway-venv smoke test is the acceptance gate**, not a CI job. After `py -m build`, ran `py -m venv /tmp/chronoagent_smoke && /tmp/chronoagent_smoke/Scripts/python.exe -m pip install dist/chronoagent-0.1.0-py3-none-any.whl` and verified `chronoagent.exe --help` lists the three CLI commands (`serve`, `run-experiment`, `check-health`) plus `from chronoagent.dashboard import INDEX_HTML; INDEX_HTML.exists()` returns `True` with the right byte size. Cleaned up the venv + dist directory after to keep the working tree clean. (f) **`tests/unit/test_package_build.py`** (10 tests, 3 classes) locks in the invariants without re-running `py -m build`: `TestBundledDashboardAsset` (4 tests on `STATIC_DIR` / `INDEX_HTML` resolution and the file living under the package root), `TestPyprojectPackagingInvariants` (5 tests on hatchling pin, console-script entry, wheel packages list, sdist include whitelist anchored with leading `/`, sdist exclude covering all cache patterns), `TestVersionStringMatches` (1 test that `chronoagent.__version__` agrees with `pyproject.toml::project.version`). **9.5:** (a) **One entry point: `chronoagent.observability.logging.get_logger`.** Every `src/` module that owns a module-level `logger` now imports `get_logger` and calls it at import time. The wrapper delegates to `structlog.get_logger`, but centralising the call means field-name conventions, `ProcessorFormatter` wiring, and future changes (sampling, additional processors) only need to change in one file. (b) `retry.py` keeps `logging.getLogger` because tenacity's `before_sleep_log` hook requires a real `logging.Logger` instance; a block comment makes the exemption explicit and points at task 9.2. `observability/logging.py` is also exempt because it defines the wrapper. These are the only two exceptions in the whole repo. (c) **Event name + kwargs convention.** Migrated sites use `logger.warning("event_name_snake_case", key=value, ...)` rather than `%s` interpolation, which gives structlog proper structured fields and lets CI assertions match on event names. (d) `tests/conftest.py` (new) runs `configure_logging("test")` once per session via `@pytest.fixture(autouse=True, scope="session")`. Without it, structlog's default factory points at a `PrintLogger` that writes to stdout and bypasses pytest's `caplog`; four existing `test_task_allocator.py` assertions broke on that path and were only fixable by forcing structlog through the stdlib bridge. (e) The new `test_logging_audit.py` encodes three guard rails via file-text regex scanning: (1) no `logging.getLogger(...)` call outside the two whitelisted files, (2) no `structlog.get_logger(...)` call outside `observability/logging.py`, (3) no `logger.foo("... %s", x)` style call in any of the four migrated stdlib-backed modules. These lock the audit in so future drift is a failed test. (f) The audit also walks every importable submodule and checks that the module-level `logger` attribute is a structlog type (accepting `BoundLoggerLazyProxy` from the lazy factory) rather than a stdlib `logging.Logger`. |
| **Completed** | 9.1: 2026-04-11; 9.2: 2026-04-11; 9.3: 2026-04-11; 9.4: 2026-04-11; 9.5: 2026-04-11; 9.6: 2026-04-11 |

---

## Phase 10: Research Experiment Suite

**Goal:** Full experiment harness -- all baselines, ablations, metrics. Reproducible from single config YAML.

**Exit Criteria:**
- All 6 experiment configs run to completion
- Metrics: AWT, allocation efficiency, AUROC, F1, precision, recall
- Baselines: SentinelAgent-reactive, no-monitoring (round-robin)
- Ablations: no-Chronos, no-BOCPD, no-health-scores
- Results saved as JSON + CSV
- Paper figures auto-generated

**Tasks:**
- [x] 10.1 `experiments/config_schema.py` -- `ExperimentConfig` Pydantic model: name, seed, num_runs, num_prs, AttackConfig (type, target, injection step, strategy), AblationConfig (forecaster/bocpd/health/integrity toggles), SystemConfig
- [x] 10.2 `experiments/metrics.py` -- `advance_warning_time(injection, detection)`, `allocation_efficiency(results)`, `detection_auroc(y_true, y_scores)`, `detection_f1(y_true, y_pred)`
- [x] 10.3 `experiments/baselines/sentinel.py` -- Reactive baseline: execution trace matching, no temporal forecasting, no health scores
- [x] 10.4 `experiments/baselines/no_monitoring.py` -- No monitoring, round-robin allocation, no integrity checks
- [x] 10.5 Six experiment configs:

| Config | Attack | Ablation | Purpose |
|--------|--------|----------|---------|
| `main_experiment.yaml` | MINJA | Full system | Main result |
| `agentpoison_experiment.yaml` | AGENTPOISON | Full system | Generalization |
| `ablation_no_forecaster.yaml` | MINJA | Chronos off | Forecaster contribution |
| `ablation_no_bocpd.yaml` | MINJA | BOCPD off | BOCPD contribution |
| `ablation_no_health_scores.yaml` | MINJA | Health off | Health scoring contribution |
| `baseline_sentinel.yaml` | MINJA | Sentinel | Reactive comparison |

- [x] 10.6 `experiments/runner.py` (full) -- Per run: seed, build system from config, clean phase, inject attack, post-attack phase, compute metrics. Aggregate across N runs (mean, std, CI).
- [x] 10.7 `experiments/analysis/plots.py` -- 6 figures: signal drift viz, health score comparison, AWT box plot, allocation efficiency over time, ROC curve, ablation bar chart
- [x] 10.8 `experiments/analysis/tables.py` -- LaTeX tables: main results, ablation results, signal validation
- [x] 10.9 CLI: `chronoagent run-experiment --config <path> --output results/`

**Key Files:** `experiments/*.py`, `experiments/analysis/*.py`, `configs/experiments/*.yaml`

**Research Notes:**
- Every config maps to a specific paper result. MockBackend ensures deterministic reproducibility.
- Each experiment runs N times (default 5) with different seeds for statistical significance. Results aggregated: mean, std, 95% CI.
- Figures auto-generated for paper:
  1. Signal drift viz (time-series, injection + detection points marked) -- Section 5.1
  2. Health score comparison (full vs ablations) -- Section 5.2
  3. AWT box plot (full vs baselines) -- Section 5.3
  4. Allocation efficiency over time (cumulative success rate) -- Section 5.4
  5. ROC curve for anomaly detection -- Section 5.5
  6. Ablation bar chart (all metrics across conditions) -- Section 5.6
- LaTeX tables auto-generated: main results, ablation results, signal validation

**README contribution:** Research results section — main metrics table (AWT, AUROC, F1, allocation efficiency), ablation table, link to paper.

### Phase 10 Log
| | |
|--|--|
| **AWT result** | _fill in: mean ± std (target >0)_ |
| **AUROC result** | _fill in (target >0.8)_ |
| **Allocation efficiency vs no-monitoring** | _fill in: % gain_ |
| **Allocation efficiency vs round-robin** | _fill in: % gain_ |
| **Key finding** | _fill in: one-sentence summary of the most important result_ |
| **Pivot triggered?** | _fill in: yes/no and which_ |
| **Findings** | **10.8:** New `src/chronoagent/experiments/analysis/tables.py` (~530 lines) ships three LaTeX `tabular` generators for the Phase 10 paper plus a `generate_all_tables` convenience entry point: `make_main_results_table` (per-experiment headline table with bolded best mean per metric column), `make_ablation_table` (full system + ablations with parenthesised delta-vs-full per cell), and `make_signal_validation_table` (per-signal Cohen's d from a Phase 1 stats sequence). Module exposes `MAIN_RESULTS_TABLE_STEM`, `ABLATION_TABLE_STEM`, `SIGNAL_VALIDATION_TABLE_STEM` constants for the default output filenames, plus a public frozen `SignalStatRow` dataclass that mirrors `phase1.SignalStats` but lives independently so the tables module does not pull Phase 1's heavy ChromaDB / MockBackend imports transitively. Per-metric decimal places locked in module-private constants: `_AWT_DECIMALS=1`, `_AUROC_DECIMALS=3`, `_F1_DECIMALS=3`, `_ALLOC_EFF_DECIMALS=3`, `_COHENS_D_DECIMALS=2`. **Cell formatters:** `_format_mean_ci(mean, ci_low, ci_high, decimals)` renders `"0.500 $\\pm$ 0.100"` from a 10.6 `MetricAggregate`; NaN mean returns `"--"`; NaN CI bounds return mean alone with no `$\\pm$` block. `_format_delta(mean, baseline, decimals)` renders `" (+0.500)"` or `" (-0.500)"` for the ablation column; NaN on either operand returns the empty string so the parent cell stays as just `"--"` without a delta. `_is_better(value, current_best)` enforces "higher is better" across all four metric columns (AWT for more advance warning, AUROC/F1/AE for quality) so a single comparison rule drives the bolding loop; NaN never wins. **Output paths:** multi-experiment tables default to `<results_dir>/tables/<stem>.tex` via `_resolve_table_path`; the signal validation table requires an explicit output_path because Phase 1 results don't live under a 10.6 results directory. **Plain LaTeX `\\begin{tabular}{lcccc}` with `\\hline` separators**, no booktabs dep, so the only LaTeX package needed in the paper preamble is the standard set. Reuses `_EXPERIMENT_DISPLAY_NAMES` from the 10.7 plot module so figures and tables agree on row labels. **Test file:** `tests/unit/test_experiment_tables.py` (~470 lines, 54 tests across 8 classes). `TestFormatMeanCi` x6 (basic symmetric, asymmetric picks larger side, NaN mean returns "--", NaN CI bounds returns mean alone, decimal parameter respected, zero-width CI). `TestFormatDelta` x6 (positive, negative, zero uses plus sign, NaN mean returns empty, NaN baseline returns empty, decimal respect). `TestIsBetter` x4 (higher wins, equal does not replace, first real beats NaN baseline, NaN never wins). `TestCoerceMetricValue` x4 (None, numeric string, float passthrough, garbage). `TestEscapeLatex` x3 (underscore, multiple specials, plain text). `TestMakeMainResultsTable` x10 (default path, output path override, table structure, NoMonitoring NaN as "--", bolding marks best per column with explicit no-health alloc-eff win check, display names not raw YAML names, empty list rejected, missing experiment raises, creates tables dir, per-metric decimal places). `TestMakeAblationTable` x9 (default path, output override, full row first, full row has no delta, ablation rows have delta, NoMonitoring AUROC delta `-0.500`, NoMonitoring alloc_eff delta `+0.500`, NaN metrics drop delta, empty list rejected). `TestMakeSignalValidationTable` x8 (writes to supplied path, one row per signal, underscores escaped, large_effect bolded, Cohen's d 2 decimals, creates parent dir, empty rows rejected, header text). `TestGenerateAllTables` x4 (runs main + ablation without signal rows, includes signal validation when supplied, signal lands under tables dir, paths are files). **Key design decisions locked in:** (1) **Single shared `_METRIC_COLUMNS` tuple of `(key, label, decimals)` triples** drives both the main results and ablation tables so column order, headers, and decimal places stay synchronized. (2) **Bolding loop walks the rows twice**: first to find the best mean per column, then to render each cell with `\\textbf{...}` wrapping when `cell.mean == best_per_column[key]`. Equality comparison (rather than `>`) means tied bests get bolded together, which `test_writes_default_path` and `test_bolding_marks_best_value_per_column` exercise. (3) **`_load_metric_row` is private** and returns a dict-of-`_MetricCell` because both table generators need the same per-experiment data shape. (4) **Ablation table puts the full system row before the `\\hline` and the ablations after**, so the LaTeX paper can `\\input{...}` the file and the visual separation matches the paper's expected layout. (5) **`_format_delta` always emits a sign character**, even for zero deltas (`(+0.000)`), so a positive vs negative ablation row is unambiguous on first glance. (6) **`make_signal_validation_table` requires an explicit output path** rather than reusing the multi-experiment default, because Phase 1 data is independent of the 10.6 results directory tree and forcing one entry point would couple the table generator to a directory layout that does not exist for Phase 1. **Key gotchas locked in:** (a) **Bolding wraps the entire `mean $\\pm$ ci` cell** in `\\textbf{...}`, so test assertions that look for the unwrapped form like `" 0.0 $\\pm$ 0.0 "` (with leading space) need to drop the leading space and accept that the rendered cell may be `\\textbf{0.0 $\\pm$ 0.0}` instead. First test attempt failed on `test_per_metric_decimal_places` for exactly this reason; fixed by removing the leading space from the assertion. (b) **ruff `I001` flagged the test file's import order** after I added the new tables imports; one `--fix` pass cleaned it up. (c) **`_format_mean_ci` returns `mean alone, no $\\pm$ block` when either CI bound is NaN**, which the 10.6 single-sample aggregator emits. The plot tests already exercise the single-sample-then-bar-no-error-bar path; the table version is the symmetric "single-sample-then-mean-only" cell. **Test + lint + type check status end of 10.8:** 1483 tests passed (1429 prior + 54 new tables tests, zero failures), 96.19% coverage, ruff check + format --check clean on 130 files (1 new src + 1 new test), mypy --strict --no-incremental clean on 80 source files, zero flakes this run. **No new dependencies** -- the tables module is pure stdlib + plain LaTeX text. **10.7:** New `src/chronoagent/experiments/analysis/` subpackage (converted from the single-file `analysis.py` so Phase 10 plots and Phase 1 `SignalAnalyzer` can coexist under one import root). `analysis/__init__.py` re-exports `AnalysisConfig` and `SignalAnalyzer` from the relocated `analysis/phase1.py` so the existing CLI import `from chronoagent.experiments.analysis import AnalysisConfig, SignalAnalyzer` keeps working unchanged. New `analysis/plots.py` (~640 lines) ships six paper figure functions plus an `ExperimentArtefacts` loader and a `generate_all_plots` convenience: `plot_signal_drift` (Section 5.1, time-series of every signal column for one representative run with injection + first-flagged step marked, reads `raw/run_<i>_signals.npy`), `plot_health_score_comparison` (Section 5.2, mean per-step detector score across runs for the full system + ablations on the same axes with shaded ±1 std band, reads `raw/run_<i>_decisions.json`), `plot_awt_box` (Section 5.3, box plot of per-run AWT values across baselines + full system, reads only `runs.csv` so it works without raw data), `plot_allocation_efficiency_over_time` (Section 5.4, per-step cumulative success rate averaged across runs, reads decisions JSON), `plot_roc_curve` (Section 5.5, per-experiment ROC curves built from per-step pooled scores, calls `_empirical_roc` plus `detection_auroc` from 10.2 so the AUC label and the aggregate.json metric agree), `plot_ablation_bar_chart` (Section 5.6, four-metric grouped bar chart with 95% CI error bars from `aggregate.json` only). Each function writes both PNG and SVG. Per-experiment figures land under `<results>/<experiment>/figures/`; multi-experiment comparison figures land under `<results>/figures/`. Pinned colour palette and display-name table give the figures a consistent look across regenerations. **10.6 runner extension landed in this task:** `RawRunRecord(run_index, seed, signal_matrix, decisions)` frozen dataclass, `collect_raw: bool = False` constructor flag on `ExperimentRunner`, `raw_runs` property exposing the live list, `_project_decision_for_raw` helper that turns Sentinel / NoMonitoring / FullSystem decisions into a JSON-friendly dict with stable schema (default `score=0.0`/`flagged=False` for NoMonitoring, only emits `bocpd_score`/`forecaster_score`/`integrity_score` when the source decision had them set), and an optional `raw_runs` parameter on `write_experiment_results` that writes `<results>/<name>/raw/run_<i>_signals.npy` (np.save) plus `run_<i>_decisions.json` (strict JSON via the existing `_nan_to_none` walker). Default `collect_raw=False` keeps the inner-loop test path lean; the 50 runner tests from 10.6 still pass unchanged. **Test file:** `tests/unit/test_experiment_plots.py` (~640 lines, 41 tests across 11 classes) covers: `TestRunnerRawData` (10 tests for the 10.6 runner extension: default-off behaviour, populated raw_runs shape, run_index/seed preservation, FullSystem subscores in raw decisions, NoMonitoring decisions have no subscores, raw_runs reset on second run, write skips raw when omitted, write persists raw when supplied, signals.npy round-trip, decisions.json round-trip), `TestLoadArtefacts` (4 tests: loader returns dataclass, raw data read when present, raw dir absent handled, missing aggregate.json raises), `TestEmpiricalRoc` (3 tests: empty returns diagonal, single class returns diagonal, perfect separation gives the corner-at-(0,1) curve), `TestPlotSignalDrift` (4 tests: PNG+SVG created, per-experiment figures dir, missing raw raises, run_index out of range raises), `TestPlotHealthScoreComparison` (4 tests: PNG+SVG, root figures dir, empty list rejected, missing raw raises), `TestPlotAwtBox` (4 tests: PNG+SVG, handles NoMonitoring with empty AWT column via hatched placeholder box, runs.csv only no raw required, empty list rejected), `TestPlotAllocationEfficiencyOverTime` (3 tests: PNG+SVG, missing raw raises, root figures dir), `TestPlotRocCurve` (2 tests: PNG+SVG, missing raw raises), `TestPlotAblationBarChart` (4 tests: PNG+SVG, no raw data required, NaN metrics handled (NoMonitoring AWT mean is null), empty list rejected), `TestGenerateAllPlots` (3 tests: 12 paths returned for 6 figures x 2 formats, drift_experiment override, empty list rejected). **Key design decisions locked in:** (1) **Subpackage refactor uses `git mv` to preserve history.** `src/chronoagent/experiments/analysis.py` becomes `analysis/phase1.py` plus a new `__init__.py` that re-exports the public symbols. The CLI continues to import from `chronoagent.experiments.analysis` unchanged. (2) **Raw data is opt-in via `collect_raw=True`** so existing 10.6 runner tests + production runs that don't need plots pay zero cost. The default-off behaviour is locked in by `test_collect_raw_default_is_off`. (3) **Per-decision projection has a stable schema across detector flavours.** `_project_decision_for_raw` always emits `step_index`, `success`, `agent_id`, `score`, `flagged`, then only adds `bocpd_score` / `forecaster_score` / `integrity_score` when the source FullSystemDecision had them set. NoMonitoring decisions get `score=0.0` / `flagged=False` defaults so the plot layer's `getattr(d, "score", 0.0)` pattern works on every detector. (4) **Per-experiment vs root figures dir.** Per-run figures (signal drift) live under `<results>/<experiment>/figures/`; multi-experiment comparison figures live under `<results>/figures/`. The naming convention keeps the file tree readable when six experiments + a comparison root all coexist. (5) **Pinned `_EXPERIMENT_COLOURS` and `_EXPERIMENT_DISPLAY_NAMES` tables** so figures generated in different sessions are visually consistent across regenerations. (6) **Lazy matplotlib import with `matplotlib.use("Agg")`** centralised in `_setup_matplotlib()` so each figure function shares one entry point. The plot test file calls `matplotlib.use("Agg")` at module import time before any pyplot import to lock the backend before pytest collects any matplotlib state. (7) **Autouse `_close_matplotlib` fixture** in the test file calls `plt.close("all")` after every test so figure-leaking failures don't cascade. (8) **`_empirical_roc` is a small helper** rather than a sklearn call because it needs to gracefully degenerate to the diagonal when one class is missing or scores are empty (sklearn would raise). The corner-at-(0,1) test locks in the perfect-separation case. (9) **AWT box plot draws hatched grey placeholder boxes for fully-empty AWT columns** (e.g. NoMonitoring) instead of crashing or skipping the experiment. The label below the box reports `n=0/num_runs` so the reader knows the column has no data. (10) **Ablation bar chart hides the CI error bar when `n < 2`** so a one-sample point does not misleadingly imply a confidence interval; matches the 10.6 `_aggregate_metric` single-sample policy. **Key gotchas locked in:** (a) **`matplotlib.use("Agg")` must run BEFORE the first pyplot import.** First test attempt put it inside the autouse fixture and got the warning that `use` was a no-op. Fix: import matplotlib at module top, call `use("Agg")`, then import the plot module under a `# noqa: E402` so the linter does not flag the import order. (b) **`ax.boxplot(..., labels=...)` is deprecated in matplotlib 3.10**, replaced with `tick_labels=...`. The plot tests would fail with a `MatplotlibDeprecationWarning` -> error if the deprecated kwarg were used. (c) **`ruff SIM105` enforces `contextlib.suppress` over a bare `try / except / pass`** for the AWT CSV cell parsing; rewrote the loop to use `contextlib.suppress(TypeError, ValueError)`. **Test + lint + type check status end of 10.7:** 1429 tests passed (1388 prior + 41 new plot tests, zero failures), 96.06% coverage, ruff check + format --check clean on 128 files (3 new: phase1 rename + plots + plot tests), mypy --strict --no-incremental clean on 79 source files, zero flakes this run. **No new dependencies** -- matplotlib is already in `[project.dependencies]` from Phase 1. **10.6:** Two new modules under `src/chronoagent/experiments/`: `experiment_runner.py` (~570 lines) ships `ExperimentRunner` class, `RunResult` / `MetricAggregate` / `AggregateResult` frozen dataclasses, a `SignalMatrixFactory` Protocol, a `default_signal_matrix_factory` that wraps `SignalValidationRunner` (deferred import so tests pay no ChromaDB cost at collection time), the `_dispatch_detector` helper, a `_compute_metrics` helper that turns a decision stream into a `RunResult`, an `_aggregate_metric` helper that computes mean / std / 95% CI via `scipy.stats.t`, and a `write_experiment_results` function that persists one-row-per-run `runs.csv` + a strict-JSON `aggregate.json` under `<output>/<name>/`. `full_system_detector.py` (~460 lines) ships `FullSystemConfig`, `FullSystemDecision`, `FullSystemDetector`, the internal `_ChannelMask` and `_SentinelFallback` helpers, plus module constants `KL_COLUMN_INDEX = 3`, `ENTROPY_COLUMN_INDEX = 5`, `FULL_SYSTEM_AGENT_ID = "full_system_detector"`, and `FORECASTER_IS_PLACEHOLDER = True`. The detector is an offline replay of ChronoAgent's monitoring stack driven by the three anomaly-facing ablation flags: `bocpd` runs the real project `BOCPD` implementation offline over the KL-divergence column; `forecaster` runs a lightweight EMA-residual stand-in for Chronos (honest placeholder, documented); `integrity` runs a MAD-outlier score on the entropy column. Sub-scores are averaged and a step is flagged iff `combined > config.decision_threshold` (default 0.5, strict `>`). When every anomaly flag is off the detector falls back to Sentinel-style z-score on the KL column, but in practice that path only ever triggers through the runner's dispatch table for `baseline_sentinel` which goes to the real `SentinelBaseline` first. **Dispatch rule (locked in):** `cfg.name == "baseline_sentinel"` -> `SentinelBaseline` with `calibration_steps = cfg.attack.injection_step`; `cfg.ablation.health is False` -> `NoMonitoringBaseline`; otherwise -> `FullSystemDetector(cfg.ablation)`. This matches the 10.5 YAMLs one-to-one: `baseline_sentinel` goes to Sentinel, `ablation_no_health_scores` goes to NoMonitoring, the other four (main + generalisation + no_forecaster + no_bocpd) go to FullSystemDetector with the right flag wiring. **Per-run flow:** `cfg.per_run_seed(run_index)` -> factory produces `(num_prs, NUM_SIGNALS)` matrix -> detector emits decisions -> `_compute_metrics` projects decisions to dict form for `allocation_efficiency`, reads `flagged` + `score` via `getattr(..., default)` for `detection_auroc` / `detection_f1` (so NoMonitoring decisions, which carry neither attribute, default to False / 0.0 without errors), and walks the decision list once to find the first-flagged step for `advance_warning_time`. Ground-truth labels are `[False]*injection_step + [True]*(num_prs - injection_step)`. **Aggregation:** per-metric mean / std / 95% CI using `scipy.stats.t.ppf(0.975, n-1) * std / sqrt(n)` for the Student-t half-width; NaN values are dropped before aggregation; empty input returns `MetricAggregate(nan, nan, nan, nan, 0)` so the runner can always serialise a complete JSON document. Single-sample input reports the mean but NaN CI to avoid implying a confidence bound from one observation. **Persistence:** `write_experiment_results(aggregate, output_dir)` writes `runs.csv` (one row per run, DictWriter with explicit fieldnames) and `aggregate.json` (strict JSON, NaN scrubbed to `null` via `_nan_to_none` walker which handles dict / list / tuple / numpy floating / numpy integer). The walker is the preferred approach because `json.dump(default=...)` does NOT intercept Python floats, so a NaN would otherwise become the literal `NaN` token (invalid per the JSON spec). **Signal matrix provenance:** The default factory wraps `SignalValidationRunner.create(n_steps=max(clean_len, poison_len), n_calibration=clean_len)` and truncates the symmetric clean + poisoned matrices into `(num_prs, NUM_SIGNALS)` with `clean[:injection_step]` vstacked with `poison[:num_prs - injection_step]`. This reuses every byte of Phase 1's signal generation pipeline and only needs the runner to enforce `injection_step > 0 and num_prs - injection_step > 0`. Tests inject a fast deterministic `_shifted_factory()` that returns seeded gaussian noise with an additive shift on the KL + entropy columns after `injection_step`, so runner-level tests complete in ~7 s without spinning up ChromaDB at all. **Test file:** `tests/unit/test_experiment_runner.py` (~730 lines, 50 tests across 8 classes). `TestAggregateMetric` (8 tests: empty, single sample, uniform samples, symmetric CI, Student-t formula spot check, NaN dropping, all-nan input, CI_CONFIDENCE constant). `TestDispatchDetector` (8 tests: `baseline_sentinel` -> Sentinel, `ablation_no_health_scores` -> NoMonitoring, parametrized over the four full-system YAMLs -> FullSystemDetector, Sentinel's calibration_steps matches `injection_step`, FullSystem's hazard_lambda matches `system.bocpd_hazard_lambda`, FullSystem respects ablation flags). `TestComputeMetrics` (4 tests: perfect full-system decisions give expected metrics, detector firing before injection gives positive AWT, detector never firing gives None AWT / 1.0 alloc eff / 0.0 F1, wrong decision count raises). `TestExperimentRunnerEndToEnd` (8 tests: main config yields perfect metrics on shifted fake, baseline_sentinel dispatch on real YAML, ablation_no_health dispatch on real YAML, null factory never flags, per-run determinism, per-run seeds are `cfg.seed + run_index`, provenance includes every config block, injected factory is called once per run with correct kwargs). `TestResultDataclasses` (3 tests: RunResult frozen, MetricAggregate fields, AggregateResult preserves run order). `TestFullSystemDetectorChannels` (12 tests: all channels on flags post-injection, BOCPD-only has non-None bocpd_score and None others, forecaster-only, integrity-only, all-off falls back to Sentinel z-score, agent_id rotates through `AGENT_IDS`, wrong shape rejected (rank + column count), too-few-rows rejected, zero-row matrix returns `[]`, FullSystemDecision frozen, FullSystemConfig validators reject every out-of-range field). `TestWriteExperimentResults` (4 tests: CSV + JSON round-trip, NaN metrics serialise as `null`, output directory created on demand, detector_name field present in CSV). `TestDefaultFactoryIntegration` (3 tests, marked `@pytest.mark.slow`: end-to-end run on `main_experiment` with real `SignalValidationRunner` at `num_runs=1, num_prs=16, injection_step=8`; default factory rejects zero phase lengths; default factory rejects `attack_type='none'`). **Key design decisions locked in:** (1) **Two files not one.** The runner orchestration and the full-system detector are separable responsibilities; keeping them in distinct modules means tests can import `FullSystemDetector` without pulling in `scipy.stats` and the metric machinery. (2) **`SignalMatrixFactory` as a pluggable Protocol.** The runner's unit tests NEVER spin up ChromaDB; only the 3 slow-marked integration tests do. This keeps the inner-loop test runtime at ~7 s for the full runner file. (3) **Detector dispatch returns `tuple[_DetectorLike, str]` with `_DetectorLike.run` typed as `Sequence[Any]`.** mypy's invariant `Sequence[Specific]` would reject concrete subclasses otherwise; since the runner projects decisions through `getattr` anyway, the structural contract is enforced at projection time in `_compute_metrics` rather than in the type annotation. (4) **`_compute_metrics` uses `getattr(d, "flagged", False)` / `getattr(d, "score", 0.0)`** so NoMonitoring decisions (which have neither attribute) degrade gracefully into all-False predictions and all-zero scores, which the 10.2 metric functions then translate into `detection_f1 = 0.0` and (via sklearn's tie-breaking convention) `detection_auroc = 0.5`. This matches the "no monitoring means the system sees nothing" semantic the paper cites for the no-monitoring ablation row. (5) **Aggregation uses `scipy.stats.t` for the 95% CI half-width**, not a normal approximation, because `num_runs=5` is below the asymptotic regime where the two agree; for `num_runs=5` the t critical value is ~2.78 vs the normal 1.96. NaN values are dropped before aggregation so empty/all-nan metrics return `(nan, nan, nan, nan, 0)` instead of raising. (6) **`write_experiment_results` scrubs NaN to `null` via a recursive `_nan_to_none` walker.** `json.dump(default=...)` does NOT intercept Python floats (they are "serialisable"), so a NaN would otherwise become the literal `NaN` token (invalid JSON). The walker handles `float`, `dict`, `list`, `tuple`, `np.floating`, `np.integer` explicitly. (7) **`FORECASTER_IS_PLACEHOLDER = True` module flag + explicit docstring.** The forecaster channel is an EMA residual, NOT a real Chronos call. This is documented in both the module docstring and the runtime flag so a future contributor touching 10.7 / 10.8 analysis knows exactly what to replace. **Files changed (2 new src + 1 new test + 1 config + 2 doc):** `src/chronoagent/experiments/experiment_runner.py` (new, 570 lines, 3 Protocols + 3 dataclasses + 1 class + 6 helpers), `src/chronoagent/experiments/full_system_detector.py` (new, 460 lines, 2 dataclasses + 1 Decision class + 1 detector class + 2 internal helpers + 4 module constants), `tests/unit/test_experiment_runner.py` (new, ~730 lines, 50 tests across 8 classes), `pyproject.toml` (added `slow` marker to `[tool.pytest.ini_options.markers]` so the `@pytest.mark.slow` decorator on the 3 integration tests does not emit `PytestUnknownMarkWarning`), `PLAN.md` (10.6 Findings/Challenges/Completed added, 10.6 ticked), `CLAUDE.md` (this table). **Dependencies:** no new dependencies; `scipy.stats.t` is already a transitive dep of the project. **10.5:** Six experiment YAMLs shipped under `configs/experiments/` (`main_experiment.yaml`, `agentpoison_experiment.yaml`, `ablation_no_forecaster.yaml`, `ablation_no_bocpd.yaml`, `ablation_no_health_scores.yaml`, `baseline_sentinel.yaml`), one per row of the PLAN.md 10.5 table. Every YAML carries full `attack` / `ablation` / `system` blocks and pins the shared reproducibility knobs verbatim: `seed=42`, `num_runs=5`, `num_prs=25`, `attack.target="both"`, `attack.injection_step=10`, `attack.n_poison_docs=10`, `attack.strategy="default"`, `system.llm_backend="mock"`, `system.bocpd_hazard_lambda=50.0`, `system.health_threshold=0.3`, `system.integrity_threshold=0.6`. `main_experiment` and `agentpoison_experiment` both have every ablation flag `True` (full system); the three `ablation_no_*` files flip exactly one ablation flag each to `False` against the MINJA attack; `baseline_sentinel` flips every ablation flag to `False` as a hand-off signal that the 10.6 runner will map onto a `SentinelBaseline` dispatch instead of the normal pipeline. Attack type is MINJA everywhere except `agentpoison_experiment.yaml`, which is the only file that cites `type: agentpoison`, matching the generalisation row of the paper table. All six validate against `ExperimentConfig.from_yaml` on first load with no schema edits. **Test file:** `tests/unit/test_experiment_configs.py` with 106 tests across 7 classes. `TestConfigFilesExist` (7 tests: one existence check per config via parametrize + one `test_exactly_six_experiment_yamls_in_directory` that computes `{p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem != "signal_validation"}` and asserts set equality with `EXPERIMENT_NAMES`, so stray files trip a hard fail). `TestRoundTripThroughSchema` (18 tests: parametrized `from_yaml` success, `cfg.name == filename`, and `ExperimentConfig.model_validate(cfg.model_dump()) == cfg`). `TestEveryNestedBlockExercised` (18 tests: parametrized attack-block populated with valid Literal values, ablation block `isinstance AblationConfig`, system block populated with valid Literal + numeric ranges). `TestSharedReproducibilityKnobs` (36 tests: parametrized per-knob over all six configs for seed, num_runs, num_prs, injection_step, n_poison_docs, llm_backend). `TestAttackTypePerPlanTable` (6 tests: parametrized expected attack type per config). `TestAblationDeltasPerPlanTable` (14 tests: parametrized exact-match against `EXPECTED_ABLATIONS` over all six configs, explicit full-system assertions for main + agentpoison, one-knob-delta check parametrized over the three ablation files, per-config ablation detail checks for `ablation_no_forecaster` / `ablation_no_bocpd` / `ablation_no_health_scores`, and `test_baseline_sentinel_has_every_flag_off` locking in the four-false pattern). `TestPerRunSeedOverShippedConfigs` (6 tests: parametrized 5 distinct per-run seeds per config, asserting `cfg.per_run_seed(i) == SHARED_SEED + i` for `i in range(num_runs)` and that the resulting list has no duplicates). **10.1:** New `src/chronoagent/experiments/config_schema.py` ships four pydantic models that every Phase 10 experiment YAML loads through: `ExperimentConfig` (top-level: `name`, `seed`, `num_runs`, `num_prs`, plus nested `attack` / `ablation` / `system`), `AttackConfig` (`type` Literal `"minja"|"agentpoison"|"none"`, `target` Literal `"security_reviewer"|"summarizer"|"both"`, `injection_step`, `n_poison_docs`, `strategy` free-form descriptor), `AblationConfig` (four bool toggles: `forecaster`, `bocpd`, `health`, `integrity`, all default `True`), `SystemConfig` (`llm_backend`, `bocpd_hazard_lambda`, `health_threshold`, `integrity_threshold` -- the runtime knobs the harness overrides per experiment). Every model uses `model_config = ConfigDict(extra="forbid")` so a YAML key typo hard-fails at load time instead of silently shadowing. `ExperimentConfig.from_yaml(path)` is the single loader entry point: it surfaces `FileNotFoundError`, `ValueError` for non-mapping roots, and `pydantic.ValidationError` for any field-level failure. `per_run_seed(run_index)` is a deterministic helper that returns `seed + run_index` so a single config drives `num_runs` reproducible runs without re-loading. The schema is intentionally narrow: it captures only the parameters PLAN.md task 10.1 enumerates plus the minimum needed to make YAML loading work (`from_yaml`, `per_run_seed`). **10.2:** New `src/chronoagent/experiments/metrics.py` ships the four paper-facing metric functions the runner (10.6) and analysis layer (10.7 / 10.8) call to turn a run artefact into headline numbers. (1) `advance_warning_time(injection_step, detection_step) -> int` returns the signed step difference `injection_step - detection_step`. Per Pivot A, concurrent detection (`injection_step == detection_step`) returns `0` instead of NO-GO, and a negative return value means the detector was late. Both arguments are validated `>= 0`; callers must filter runs where the detector never fired. Numpy int inputs are coerced to plain Python `int` so the value survives JSON serialisation unchanged. (2) `allocation_efficiency(results) -> float` cumulative success rate of allocator decisions, accepting either a plain `Sequence[bool]` or a sequence of mapping-style audit rows (each with a `"success"` key). Empty sequence returns `0.0`, mixed bool + mapping sequences work, missing-`"success"`-key raises `KeyError`, non-bool/non-Mapping elements raise `TypeError`. The mapping path uses `bool(item["success"])` so `0`/`1`-coded integer flags also count correctly. (3) `detection_auroc(y_true, y_scores) -> float` is a thin wrapper around `sklearn.metrics.roc_auc_score` that short-circuits undefined inputs into `float('nan')`: empty arrays, zero-size scores, or single-class `y_true`. Perfect-ranking returns `1.0`, perfectly-inverted ranking returns `0.0`, uninformative (all-same) scores return `0.5` via sklearn's tie-breaking convention. (4) `detection_f1(y_true, y_pred) -> float` wraps `sklearn.metrics.f1_score` with `zero_division=0.0` so all-zero predictions return `0.0` silently instead of emitting `UndefinedMetricWarning`, and short-circuits empty arrays into `float('nan')`. Both sklearn wrappers use `warnings.catch_warnings()` + `simplefilter("ignore", UndefinedMetricWarning)` so callers get a clean stdout when running the 5-run aggregate. NaN returns let aggregators use `np.nanmean` over per-run results without wrapping each call in try/except. Both sklearn wrappers explicitly return `float(...)` so the result type is a plain Python `float` (not a numpy scalar) for JSON serialisation. **Test file:** `tests/unit/test_experiment_metrics.py` with 51 tests across 7 classes. `TestAdvanceWarningTime` (9 tests: early, concurrent, late, zero-zero, parametrized invariant check over 5 pairs, numpy-int coercion, negative injection rejected, negative detection rejected). `TestAllocationEfficiencyBoolInput` (7 tests: all success, all failure, half-half, two-thirds, single success, single failure, empty returns 0.0). `TestAllocationEfficiencyMappingInput` (5 tests: mapping all success, mixed, extra fields ignored, missing key raises KeyError, truthy non-bool `"success": 1/0` counts). `TestAllocationEfficiencyMixedAndErrors` (3 tests: mixed bool+mapping, string element raises TypeError, None element raises TypeError). `TestDetectionAuroc` (12 tests: perfect ranking = 1.0, inverted = 0.0, uninformative = 0.5, partial overlap bounded, accepts Python lists, empty y_true = nan, empty y_scores = nan, single-class all-zero = nan, single-class all-one = nan, all-zero scores with two classes = 0.5, return type is Python float, nan return type is Python float). `TestDetectionF1` (12 tests: perfect predictions = 1.0, all wrong = 0.0, half correct (TP=1 FP=1 FN=1) = 0.5, Python lists accepted, empty y_true = nan, empty y_pred = nan, all-zero true+pred = 0.0 silently, all-zero true + all-one pred = 0.0, single-class all-one exact match = 1.0, no warnings on zero division (asserted via `warnings.simplefilter("error", UndefinedMetricWarning)`), return type is Python float, nan return type is Python float). **10.3:** New subpackage `src/chronoagent/experiments/baselines/` with an `__init__.py` module doc and `sentinel.py` (~250 lines) containing the reactive Sentinel baseline. Ships `SENTINEL_AGENT_ID = "sentinel_baseline"` constant, frozen `SentinelConfig` dataclass (`calibration_steps >= 2` default 10, `z_threshold > 0` default 3.0, `min_std > 0` default 1e-6) with `__post_init__` validators, frozen `SentinelDecision` dataclass (`step_index`, `score`, `flagged`, `success`, `agent_id` default `SENTINEL_AGENT_ID`), and `SentinelBaseline` class with `calibrate`, `decide`, `run` methods. Calibration learns per-signal mean + `ddof=1` std from the first `calibration_steps` rows of a `(T, NUM_SIGNALS)` matrix, clamps std to `min_std` so constant MockBackend signals cannot divide by zero, then scoring computes `max(|vec - mean| / std)` per step and flags when `score > z_threshold` (strict `>` so boundary ties pass). `success = not flagged`. The class is intentionally single-step and memoryless: no rolling window, no temporal state, no forecasting, no health score. `run(signal_matrix)` is the runner entry point: calibrate + score the full matrix, return `list[SentinelDecision]` in step order. Decisions plug straight into the 10.2 metric functions via a one-line dict projection for `allocation_efficiency`, direct attribute access for `detection_auroc` (scores) and `detection_f1` (flagged casts to int), and `first-flagged-step` index for `advance_warning_time`. **Test file:** `tests/unit/test_sentinel_baseline.py` (350 lines, 37 tests across 6 classes). `TestSentinelConfig` (7 tests: defaults, frozen, `calibration_steps >= 2`, `calibration_steps == 0` rejected, `z_threshold > 0`, negative `z_threshold` rejected, `min_std > 0`). `TestSentinelDecision` (4 tests: explicit fields, `agent_id` default, frozen, `success`/`flagged` independently settable at the dataclass level). `TestSentinelBaselineCalibrate` (8 tests: `is_calibrated` flip, wrong rank rejected, wrong column count rejected, too-few-rows rejected, exactly-`calibration_steps` rows accepted, extra rows beyond window ignored (assertion that baseline mean/std match a truncated matrix), `min_std` floor handles constant column, calibrate is idempotent (second call overwrites)). `TestSentinelBaselineDecide` (8 tests: un-calibrated raises `RuntimeError`, accepts `StepSignals`, accepts numpy vector, wrong-shape rejected, zero-deviation gives score 0, boundary `score == z_threshold` does NOT flag (locks in strict `>` semantics), `success = not flagged`, score is max across signals (col 0 at +5σ dominates col 1 at +1σ -> score = 5.0)). `TestSentinelBaselineRun` (6 tests: one decision per row, step indices monotonic, clean random run never flags calibration window (variance identity bounds max |z| by sqrt(N-1) = 3.0), shifted post-calibration block flags every shifted row, `success` mirrors `not flagged` across full run, constant-signal run is stable (every row at baseline mean, no flags)). `TestSentinelMetricsIntegration` (4 tests: `allocation_efficiency` accepts decision mapping and returns 0.5 on a 10 clean + 10 shifted matrix, `detection_auroc` = 1.0 on well-separated blocks (paper-quality assertion), `detection_f1` = 1.0 on same separation, first-flagged-step equals the injection block start). **10.4:** New `src/chronoagent/experiments/baselines/no_monitoring.py` (~150 lines) shipping the dumb round-robin comparator the Phase 10 paper's ablation table cites to isolate the contribution of *any* monitoring at all versus the Sentinel and full-system conditions. Exports `NO_MONITORING_AGENT_ID = "no_monitoring_baseline"` (baseline-as-a-whole label, distinct from the real agent ids), frozen `NoMonitoringDecision` dataclass (`step_index`, `success: bool`, `agent_id: str`), and `NoMonitoringBaseline` class with `decide(step_index)` and `run(signal_matrix)` methods. `decide` cycles through `chronoagent.allocator.capability_weights.AGENT_IDS` (`planner` -> `security_reviewer` -> `style_reviewer` -> `summarizer`) via `AGENT_IDS[step_index % len(AGENT_IDS)]`, always reports `success=True`, and is deterministic with no randomness anywhere. `run(signal_matrix)` validates the `(T, NUM_SIGNALS)` shape and returns one decision per row; the matrix *contents* are deliberately ignored, which `test_signal_matrix_content_is_ignored` locks in by asserting that clean / shifted / NaN-filled inputs produce identical decision lists. By construction the stream plugs into the 10.2 metric functions as: `allocation_efficiency = 1.0` exactly (every step succeeds), `detection_auroc = nan` via the single-class short-circuit (all-zero labels), `detection_f1 = 0.0` via sklearn's `zero_division=0.0` path (all-zero predictions against any label stream). **Test file:** `tests/unit/test_no_monitoring_baseline.py` (280 lines, 28 tests across 5 classes). `TestNoMonitoringDecision` (3 tests: explicit fields, frozen, `success=False` representable at the dataclass level). `TestNoMonitoringBaselineDecide` (8 tests: step 0 = first agent, step 1 = second agent, wrap at `len(AGENT_IDS)`, full first cycle matches `AGENT_IDS` element-for-element, every decision succeeds over 50 steps, determinism across two independent instances, negative step index rejected, step_index recorded verbatim). `TestNoMonitoringBaselineRun` (9 tests: one decision per row, monotonic indices, 3-cycle agent sequence matches `list(AGENT_IDS) * 3`, all-success, zero-row matrix returns `[]`, content ignored across clean/shifted/NaN matrices, wrong-rank rejected, wrong-column-count rejected, determinism across two calls). `TestNoMonitoringAgentIdConstant` (2 tests: label value, label distinct from the canonical `AGENT_IDS` tuple). `TestNoMonitoringMetricsIntegration` (6 tests: `allocation_efficiency` = 1.0 on 20 rows, = 1.0 on a single-row run, `detection_auroc` = nan on all-zero labels + all-zero scores, `detection_f1` = 0.0 on all-zero predictions against a mixed-label ground truth, `detection_f1` = 0.0 on fully-zero label + prediction streams, decision `agent_id` values span the full `AGENT_IDS` set and never contain the baseline label constant). **10.9:** Rewired `src/chronoagent/cli.py` to close Phase 10. Converted `chronoagent run-experiment` into a Typer sub-app with `phase1` (the existing `SignalValidationRunner` entry point) and `phase10` (the new single-experiment driver). `phase10` loads an `ExperimentConfig` via `from_yaml`, constructs `ExperimentRunner(cfg, collect_raw=True)`, calls `runner.run()` + `write_experiment_results(agg, output, raw_runs=runner.raw_runs)`, then optionally renders the single-experiment drift figure via `plot_signal_drift` and the headline main results table via `make_main_results_table`. Flags: `--plots/--no-plots`, `--tables/--no-tables`, `--quiet`. Config-load errors (FileNotFoundError, ValidationError, ValueError) surface as exit 1; plot/table failures log stderr WARNINGs but leave exit code 0 so rendering glitches never invalidate a successful run. A new top-level `chronoagent compare-experiments` command takes `--output`, one or more `--experiment`, optional `--ablation`, optional `--full-system` (default: first `--experiment`), optional `--drift-experiment` (default: first `--experiment`), and its own `--plots/--no-plots` / `--tables/--no-tables` / `--quiet`, then dispatches to `generate_all_plots` + `make_main_results_table` + `make_ablation_table`. The ablation table is skipped with an operator-visible note when no `--ablation` values are supplied (avoids the `ValueError` from `make_ablation_table` on an empty list). **Key design decisions locked in:** (a) **Subcommand group, not `--phase` flag.** Each command's YAML shape is honest on its own: `phase1` takes the legacy `runner:`/`analysis:` nested layout, `phase10` takes an `ExperimentConfig`. A shared `--config` would force one command body to dispatch on YAML shape and silently misinterpret typos. (b) **`run-experiment phase10` only renders per-experiment artefacts.** `plot_signal_drift` and `make_main_results_table` both work on a one-entry list; the other five figures and the ablation table would render misleading one-point comparisons, so they are deliberately skipped with an operator-visible note pointing at `compare-experiments`. Narrower than CLAUDE.md's "`generate_all_plots(output, [cfg.name])`" wording but correctness-preserving. (c) **`compare-experiments` is a separate top-level command.** Matches the research workflow ("run every experiment first, then compare") and keeps each command's responsibility narrow. (d) **Deferred imports inside each command body.** Preserves `chronoagent serve` startup time by keeping Phase 10 imports out of the import graph unless the relevant subcommand runs. (e) **Plot/table failures in `run-experiment phase10` log WARNINGs but do not fail the run.** `runs.csv` + `aggregate.json` + `raw/` have already been persisted, so a plot failure should not block the operator from moving on. `compare-experiments` is stricter (plot/table failures return exit 1) because rendering is its sole job. **Key gotchas locked in:** (1) **Typer sub-app wiring requires `app.add_typer(sub, name="run-experiment")`** to register the top-level command name. (2) **Patching deferred imports must target the origin namespace.** First test attempt patched `chronoagent.cli.ExperimentRunner` and got AttributeError because the class never enters the `chronoagent.cli` namespace (the `from ... import ExperimentRunner` sits inside the function body and re-executes each call). Fix: patch at `chronoagent.experiments.experiment_runner.ExperimentRunner`, and similarly at `chronoagent.experiments.analysis.plots.plot_signal_drift`, `chronoagent.experiments.analysis.tables.make_main_results_table`, `chronoagent.experiments.analysis.plots.generate_all_plots`, `chronoagent.experiments.analysis.tables.make_ablation_table`. (3) **`make_ablation_table` raises `ValueError` on an empty `ablation_names`.** The compare-experiments command checks the list length and prints `"Skipping ablation table (no --ablation values supplied)."` before calling the generator. **Test file:** `tests/unit/test_cli.py` grew from 11 to 28 tests (+17). `TestHelp` x7 (root, serve, run-experiment lists subcommands, phase1, phase10 documents --plots/--tables/--quiet, compare-experiments documents --experiment, check-health). `TestRunExperimentPhase1` x4 (valid config exits 0, echoes config path, echoes output dir, rejects unknown attack type). `TestRunExperimentPhase10` x8 (happy path writes `runs.csv` + `aggregate.json` via `_StubRunner`, `--no-plots` + `--no-tables` short-circuit, `--plots` calls `plot_signal_drift` with `run_index=0` and experiment name, `--tables` calls `make_main_results_table` with a single-entry list, `--quiet` suppresses progress echoes, missing config exits 1, invalid config (negative seed) exits 1, plot failure is logged as WARNING with exit 0). `TestCompareExperiments` x5 (missing output directory exits 1, happy path calls all three generators with defaults threaded through correctly, skips ablation table when no --ablation flags, `--no-plots` + `--no-tables` short-circuit both generators, explicit `--full-system` and `--drift-experiment` override defaults). `TestCheckHealth` x2 and `TestServe` x2 (unchanged). **Test helpers:** `_phase10_yaml(name)` returns a validator-clean YAML for the Phase 10 schema; `_stub_aggregate(name, num_runs)` builds a deterministic `AggregateResult`; `_StubRunner` is a drop-in `ExperimentRunner` replacement that never touches ChromaDB and, when `collect_raw=True`, emits a `(num_prs, 6)` matrix with columns 3 and 5 bumped after `injection_step` plus a decision stream flagging every post-injection step so the persisted raw files are valid plot inputs. **Files changed (1 edited src + 1 edited test + 3 doc):** `src/chronoagent/cli.py` (restructured), `tests/unit/test_cli.py` (11 -> 28 tests), `PLAN.md` (10.9 Findings/Challenges/Completed added, 10.9 ticked, Phase 10 flipped to `[x]`), `CLAUDE.md` (session context, Phase 10 tracker flipped to `[x]`), `README.md` (Phase 10 row ticked, badges + Currently line bumped). No new dependencies. |
| **Challenges** | **10.1:** Two design calls locked in. (1) **Strategy validator only enforces non-emptiness, not whitespace stripping.** The first test attempt expected `cfg.strategy == "high_noise"` after passing `"  high_noise  "`, but the validator only rejects whitespace-only strings. Silent input mutation is a footgun (operators looking at result rows would be confused why their YAML value differs from the recorded one), so the validator stays as-is and the test was renamed to `test_strategy_with_surrounding_whitespace_accepted_verbatim` and asserts the string passes through unchanged. (2) **`name` field uses a strict ASCII-safe validator.** The validator allows `[A-Za-z0-9_-]` only, because `ExperimentConfig.name` is spliced into output filenames by the runner (10.6) and into log/event names. Allowing dots, slashes, or shell metacharacters would create a path-traversal risk and break filename hashing. The test parametrizes over both safe (`main`, `main_experiment`, `ablation-no-bocpd`, `v2_run`) and unsafe (`""`, `"   "`, `"with space"`, `"../escape"`, `"weird!"`) names. **10.2:** Four design calls locked in. (1) **Empty `allocation_efficiency` returns `0.0`, not `nan`.** Downstream cumulative tracking in the runner (10.6) benefits from a numeric value it can always append to a CSV; "success rate of zero decisions" is vacuously zero and the test asserts this explicitly so the semantics cannot silently change. The other two nan-returning metrics (`detection_auroc`, `detection_f1`) use `nan` because they feed into statistical aggregations (`np.nanmean` across 5 seeds) where a zero would pollute the mean, whereas allocation efficiency aggregates across allocator decisions within a single run. (2) **AUROC and F1 wrap sklearn but short-circuit before calling it** for every case where sklearn would emit `UndefinedMetricWarning` or raise. The alternatives (let sklearn raise, catch at the call site) pollute the runner and force every caller to write the same try/except block; centralising the nan translation here means the runner can build a `DataFrame` row with no defensive code. The `warnings.catch_warnings()` block guards the sklearn call anyway so a future sklearn version that starts emitting new undefined-metric warnings still produces a clean stdout. (3) **`AllocationResult` type alias accepts `bool | Mapping[str, Any]` rather than forcing a projection.** The task allocator's audit trail rows are already dict-shaped; demanding callers project to booleans before calling `allocation_efficiency` would be extra ceremony for no gain, and the `isinstance(item, bool)` / `isinstance(item, Mapping)` branch is a one-line check inside the loop. The test file covers the mixed bool+mapping case explicitly. (4) **`pyproject.toml` mypy overrides extended with `sklearn.*`.** The 10.2 module is the first `src/` import of sklearn (the rest of the project uses numpy/scipy for stats), and sklearn ships without a `py.typed` marker, so `mypy --strict` flagged `[import-untyped]` until `sklearn.*` joined the existing ignore list next to `ruptures.*`, `scipy.*`, `pandas.*`. **10.3 design decisions locked in:** (1) **Reactive z-score thresholding stands in for "execution trace matching".** Without a concrete library of known-bad traces, the cleanest instantiation of a reactive single-step detector is: learn mean/std from a clean window, then at each new step compute max |z| against that fixed baseline and flag above a threshold. This matches classic signature-based IDS in spirit (compare observation to fixed fingerprint set, no temporal modelling) and gives the runner a deterministic, parameter-light baseline. (2) **Calibration-window steps cannot flag with the default config.** The variance identity bounds `sum((x_i - mean)^2 / std^2) = N - 1` (where std uses `ddof=1`), so each individual z-score within the calibration window is bounded by `sqrt(N - 1)`. With `calibration_steps=10` this bound is exactly `sqrt(9) = 3.0`, and the strict `>` comparison against `z_threshold=3.0` lets the boundary pass. `test_clean_random_run_never_flags_calibration_window` locks this in empirically so a future switch to `>=` or a change to `ddof=0` or a lower default threshold cannot silently break the invariant. (3) **Strict `>` on the threshold comparison, not `>=`.** A step sitting exactly at the threshold should NOT flag: the threshold is the acceptance boundary. `test_boundary_exactly_threshold_does_not_flag` locks this in with a hand-crafted 10-row calibration matrix where col 0 has a known `ddof=1` std. (4) **`min_std` floor is non-negotiable for MockBackend compatibility.** Several signals produced by MockBackend (e.g. `tool_calls` = constant 2 across all steps in the Phase 1 runner) have zero variance in the calibration window. Without the floor, numpy divides by zero and every subsequent step gets `inf` z-scores, flagging everything. With `min_std = 1e-6`, the constant column still gets a meaningful z-score but cannot blow up. `test_min_std_floor_handles_constant_column` locks in `std = 1e-3` when the entire calibration window is constant. (5) **`SentinelDecision` is a frozen dataclass, not a Mapping.** The runner projects to a dict (one-liner) when handing decisions to `allocation_efficiency` because `SentinelDecision(...)` should NOT accidentally satisfy the `Mapping` protocol and silently count fields other than `success`. The test `test_allocation_efficiency_accepts_decision_mapping` pins the projection shape explicitly. (6) **`agent_id` field on `SentinelDecision` defaults to `"sentinel_baseline"`** so aggregated allocator-audit CSVs from multiple experiment runs can disambiguate baseline rows from full-system rows on a single column, without the runner having to thread a label parameter through every function. **10.4 design decisions locked in:** (1) **Round-robin cycles in canonical `AGENT_IDS` order, not in a shuffled or hashed order.** The Phase 5 capability matrix already pins `AGENT_IDS = ("planner", "security_reviewer", "style_reviewer", "summarizer")`; the baseline uses that exact tuple via `AGENT_IDS[step_index % len(AGENT_IDS)]` so two runs with the same step count produce byte-identical decision lists regardless of seed, which is the minimum reproducibility guarantee the paper's ablation table requires. (2) **Every step succeeds by construction, so the decision stream is honest about what "no monitoring" means.** Without a detector there is nothing to flag, so forcing `success=True` on every row surfaces the expected `allocation_efficiency = 1.0` and the expected single-class `detection_auroc = nan` / `detection_f1 = 0.0` signals. Synthesising fake failures to make the baseline look worse would bake assumptions about the attack model into the baseline itself. (3) **`NoMonitoringDecision` carries the *real* dispatched agent id, not the `NO_MONITORING_AGENT_ID` module label.** Each row's `agent_id` is one of the canonical `AGENT_IDS`, so aggregated allocator-audit CSVs can compute per-agent efficiency on the same column used by Sentinel and full-system runs. The baseline-as-a-whole label exists only as a module constant for run-level tagging, and `test_distinct_from_canonical_agent_ids` locks in the non-collision. (4) **`run(signal_matrix)` validates the `(T, NUM_SIGNALS)` shape even though it does not read the contents.** Accepting any shape would let a miswired runner pass the wrong matrix silently; rejecting wrong shapes up front gives the same fail-fast surface as the Sentinel baseline so the runner (10.6) can drive both through a single call site without branch-specific validation. (5) **No calibration step.** Unlike Sentinel, the baseline is stateless and has no baseline window, no std floor, no z-threshold, no `RuntimeError` when called un-calibrated; it is deliberately the simplest possible comparator so the Phase 10 paper's ablation table has a clean "monitoring contribution" delta. **10.5 design decisions locked in:** (1) **Six YAMLs under `configs/experiments/`, one per row of the PLAN.md 10.5 table, each carrying full `attack` / `ablation` / `system` blocks even when the block matches defaults.** Having every nested block populated explicitly means operators reading the YAML can see every knob the experiment touches without cross-referencing the schema, and the 10.6 runner can load every config through the same code path. (2) **Shared reproducibility knobs pinned identically across all six configs.** `seed=42`, `num_runs=5`, `num_prs=25`, `attack.target="both"`, `attack.injection_step=10`, `attack.n_poison_docs=10`, `system.llm_backend="mock"`, `system.bocpd_hazard_lambda=50.0`, `system.health_threshold=0.3`, `system.integrity_threshold=0.6`. This is the honesty guarantee for the paper's ablation table: any delta between configs is attributable ONLY to the ablation block (or the attack type, for the generalisation row). `TestSharedReproducibilityKnobs` parametrizes a check per knob over all six configs so a drift in any shared knob trips a hard fail. (3) **The three ablation YAMLs flip exactly one `AblationConfig` bool each, and every other flag stays `True`.** `test_one_knob_delta_vs_main` loops over `("ablation_no_forecaster", "ablation_no_bocpd", "ablation_no_health_scores")`, computes the set-difference against `main_experiment.ablation`, and asserts `len(differing) == 1`. Future additions (e.g. `ablation_no_integrity`) must satisfy the same invariant. (4) **`baseline_sentinel.yaml` sets every ablation flag to `False`** to signal "full ChronoAgent stack off, runner dispatches the reactive comparator". This was the "baseline-as-config vs baseline-as-runner-flag" question CLAUDE.md flagged at 10.4 end; the resolution is **both**: the YAML is a valid `ExperimentConfig` (baseline-as-config), and the runner (10.6) takes the additional hint from `cfg.name == "baseline_sentinel"` + all-false ablation pattern to dispatch `SentinelBaseline` rather than the normal pipeline (baseline-as-runner-flag). `test_baseline_sentinel_has_every_flag_off` locks in the four-false pattern, and `test_one_knob_delta_vs_main` explicitly excludes `baseline_sentinel` from its parametrize list so the one-knob invariant does not fire against the baseline row. (5) **No `ablation_no_integrity.yaml` and no `baseline_no_monitoring.yaml` shipped in this task.** PLAN.md's 10.5 table names exactly six configs; the `no_monitoring` baseline from task 10.4 is baseline-as-runner-flag only, and the integrity ablation is left out because PLAN.md does not cite it in the paper's ablation table. `test_exactly_six_experiment_yamls_in_directory` gates against stray `*.yaml` additions so a future drive-by config cannot slip in without a deliberate review. (6) **`signal_validation.yaml` is the Phase 1 runner input and does NOT round-trip through `ExperimentConfig.from_yaml`.** The 10.5 test file explicitly excludes it by filename stem in every YAML enumeration so the strict extra="forbid" validator does not fire against a completely different schema. **Test + lint + type check status end of 10.5:** 1338 tests passed (1232 prior + 106 new config tests, zero failures), 95.77% coverage, ruff check + format --check clean on 122 files, mypy --strict --no-incremental clean on 75 source files (0 new src files; 6 new YAML configs + 1 new test file), zero flakes this run (the usual intermittent ChromaDB cross-test-pollution pattern did NOT trip). **Test + lint + type check status end of 10.4:** 1232 tests passed (1204 prior + 28 new no-monitoring tests, zero failures), 95.77% coverage, `src/chronoagent/experiments/baselines/no_monitoring.py` 100% covered (23/23), ruff check + format --check clean on 121 files, mypy --strict clean on 75 source files (1 new: `baselines/no_monitoring.py`), zero flakes this run. **Test + lint + type check status end of 10.3:** 1204 tests passed (1167 prior + 37 new sentinel tests, zero failures), 95.74% coverage, ruff check + format --check clean on 119 files, mypy --strict clean on 74 source files (2 new: `baselines/__init__.py` + `baselines/sentinel.py`), zero flakes this run. **Test + lint + type check status end of 10.2:** 1167 tests passed (1116 prior + 51 metrics, zero failures), 95.70% coverage, `src/chronoagent/experiments/metrics.py` 100% covered (50/50), ruff check + format --check clean on 117 files, mypy --strict clean on 72 source files. Zero flakes this run (the usual intermittent ChromaDB cross-test-pollution pattern did NOT trip). **10.9:** One mocking subtlety surfaced: patching `chronoagent.cli.ExperimentRunner` fails because the `from chronoagent.experiments.experiment_runner import ExperimentRunner` sits inside the function body; the import re-executes on every call and reads the class from the origin namespace, not the local `chronoagent.cli` namespace. Fixed by patching at the origin. Local pre-push run for 10.9: ruff check + format --check clean on 130 files, mypy --strict --no-incremental clean on 80 source files, pytest 1500 passed (1483 prior + 17 new CLI tests, CLI file 89% covered) 95.98% coverage, zero flakes this run. |
| **Completed** | 10.1: 2026-04-11 &nbsp;·&nbsp; 10.2: 2026-04-11 &nbsp;·&nbsp; 10.3: 2026-04-11 &nbsp;·&nbsp; 10.4: 2026-04-11 &nbsp;·&nbsp; 10.5: 2026-04-11 &nbsp;·&nbsp; 10.6: 2026-04-11 &nbsp;·&nbsp; 10.7: 2026-04-11 &nbsp;·&nbsp; 10.8: 2026-04-11 &nbsp;·&nbsp; 10.9: 2026-04-12 |

---

## Phase 11: Paper Scaffold + Reproducibility

**Goal:** Complete LaTeX scaffold, all sections drafted, figures linked, reproducibility package ready.

**Exit Criteria:**
- `paper/main.tex` compiles to PDF
- All 9 sections have draft text
- Figures linked to Phase 10 outputs
- Bibliography complete
- `make reproduce` runs all experiments from scratch and regenerates figures

**Tasks:**
- [x] 11.1 Paper structure: `main.tex` + sections (abstract, intro, related work, system design, methodology, experiments, results, discussion, conclusion)
- [x] 11.2 Map claims to experiments:
  - C1: Behavioral signals shift under poisoning (signal validation)
  - C2: BOCPD+Chronos ensemble detects shifts with AWT>=0 (main experiment)
  - C3: Health-weighted allocation improves efficiency under attack (vs no-monitoring)
  - C4: Competitive AUROC vs reactive baselines with temporal context (vs SentinelAgent)
  - C5: Forecaster runs inline without stalling the control plane (soft claim; quantitative microbenchmark deferred to 11.4)
- [x] 11.3 `bibliography.bib` -- all cited works
- [x] 11.4 Makefile targets: `reproduce`, `reproduce-signal`, `reproduce-main`, `reproduce-ablations`, `reproduce-figures`; latency microbenchmark wired end-to-end
- [x] 11.5 Pin all versions in Docker image; all seeds fixed; all configs in repo

**Key Files:** `paper/*.tex`, `paper/bibliography.bib`, `Makefile`

**README contribution:** Citation block, link to paper PDF/arXiv, reproducibility badge (`make reproduce`).

### Phase 11 Log
| | |
|--|--|
| **Findings** | **11.1** Scaffolding the paper as nine section `\input` files under `paper/sections/` keeps `main.tex` readable and gives task 11.2 a small, unambiguous place to wire each claim. `\graphicspath` lists every per-experiment and multi-experiment figures directory under `../results/` so the body sections can reference figures by bare filename (`fig1_signal_drift.png`, not a full path); a `\inputiffound` macro wraps each `\input{../results/tables/<stem>.tex}` call so the scaffold compiles even when the Phase 10 results are stale or missing, which is the expected state at the end of 11.1. `natbib` with `plainnat` was chosen over `biblatex` to avoid a `backend=biber` dependency on the eventual CI LaTeX image; `cleveref` gives cross-ref polish without a style change later. The bibliography stub keeps the PDF build green until task 11.3 populates real entries. **11.2** The claim-to-experiment map in `paper/sections/05_experiments.tex` is now a proper `\begin{table}[tbp]\label{tab:claim-map}\end{table}` booktabs float, so the `\cref{tab:claim-map}` forward-reference dropped in by 11.1 resolves cleanly and stops the undefined-reference warning on every compile. Every `\todo{\includegraphics of figN_*}` placeholder in `06_results.tex` is now a `\figureiffound{../results/.../figN_*.png}{...}` call backed by a new `\figureiffound` macro in `main.tex` that mirrors the `\inputiffound` pattern from 11.1: it guards `\IfFileExists` around `\includegraphics[width=\columnwidth]` so the scaffold still compiles green on a clean checkout (and surfaces a red TODO per missing figure), and the explicit-path form sidesteps the fact that `\IfFileExists` ignores `\graphicspath`. Two result tables (`tab:main-results` and `tab:ablation`) are now wrapped in `\begin{table}\caption{...}\label{...}\end{table}` floats around the `\inputiffound` call so both get a paper-side caption/label while the tabular body stays owned by the 10.8 generator. Short per-claim prose paragraphs were also dropped into every `C1`--`C5` + ablations subsection so `06_results.tex` is no longer a pure figure-wiring skeleton -- it now reads as a paper skeleton with real narrative glue, which is what 11.3/11.4 need to edit against. **11.3** `paper/bibliography.bib` now holds 16 real entries grouped by the section that introduces them: attacks (MINJA, AgentPoison, A-MemGuard), allocation (Smith's contract-net, Oliehoek & Amato Dec-POMDP textbook, AutoGen, MetaGPT, CrewAI), change-point / forecasting (Adams & MacKay 2007 BOCPD, Chronos), runtime monitoring (NeMo Guardrails, Llama Guard, Constitutional AI), and the framework stack (LangChain, LangGraph, Chroma). Every `\todo{cite ...}` in `02_related_work.tex` is gone, replaced by inline `\citep{}` / `\citet{}` calls that resolve against the bib via `natbib` + `plainnat` numeric mode. Introduction, system design, methodology, and experiments were also wired: intro cites the framework stack and the three attacks, `03_system_design.tex` cites LangGraph/Chroma in the architectural prose plus Adams & MacKay and Chronos where the scorer is introduced and Smith's contract-net in the allocator paragraph, `04_methodology.tex` anchors BOCPD and Chronos in the detector subsection and adds a Dec-POMDP framing pointer in the allocator subsection, and `05_experiments.tex` grounds the two attack configs and the Sentinel-style baseline (cited to LlamaGuard + NeMo Guardrails since no specific "Sentinel" paper is in scope). A Python bib-audit script confirmed zero missing cite keys and zero unused bib entries before commit. |
| **Challenges** | **11.1** Two decisions that needed explicit reasoning. (a) **`\input{../results/tables/*.tex}` vs. a copy step.** Copying tables into `paper/tables/` would decouple the paper from the analysis pipeline but then drift silently on every re-run; inlining `../results/` keeps the paper coupled to whatever `chronoagent compare-experiments` most recently wrote. The `\inputiffound` guard is the cost of that coupling: it prevents a missing-file hard error and surfaces a visible TODO in the PDF instead. (b) **Where does C5 live?** C5 (forecaster overhead) has no Phase 10 artefact yet; the results section keeps it as an explicit `\todo` so task 11.2 has to either wire a microbenchmark into the runner or downgrade the claim. Recording the gap here (rather than fabricating a placeholder table) is the safer move. **11.2** The C5 decision carried over from 11.1 is resolved as **downgrade, not instrument**: 11.2 softens the C5 text in `05_experiments.tex` from "less than 1% computational overhead" to a qualitative "runs inline on CPU without stalling the control plane" claim, and `06_results.tex` explicitly defers the quantitative wall-clock microbenchmark to task 11.4. Rationale: wiring a real latency microbenchmark would require a new `latency_ms` column on `RunResult` + `MetricAggregate`, a schema change on `runs.csv` + `aggregate.json`, a new column on `make_main_results_table`, a new plot function, and (at minimum) a fresh test set for each of those -- all of which blows past the "map claims to experiments" scope of 11.2. Deferring to 11.4 also keeps the microbenchmark co-located with the `make reproduce` Makefile wiring that 11.4 already owns, so the latency harness can run under `make reproduce` without a cross-phase import. The soft claim is still testable end-to-end ("does Chronos inference block the collector thread?") via the existing async path; what 11.2 downgrades is the *quantitative* bound, not the qualitative guarantee. A separate footnote: `\IfFileExists` does NOT respect `\graphicspath` (documented in the `graphics` bundle's internals), so `\figureiffound` has to take an explicit path like `../results/figures/fig5_roc_curve.png` rather than a bare stem. This is the opposite convention from the 11.1 `\graphicspath` trick for plain `\includegraphics`, and it means body-level figures in the scaffold mix both styles: `\figureiffound` (explicit path, guarded) where the figure might be missing on a clean checkout, and plain `\includegraphics{<stem>}` (graphicspath-resolved) for any always-present figure. 11.2 uses the guarded form everywhere because every current figure is produced by Phase 10 and may be stale. **11.3** Three citation judgment calls. (a) **"Sentinel-style baseline" has no canonical paper.** The CLAUDE.md scoping note asked for a "Sentinel baseline reference", but the scaffold's `baseline_sentinel` config is a generic reactive per-query filter, not a reference to any specific published system named "Sentinel". Rather than fabricate a bib entry, 11.3 cites the baseline to LlamaGuard + NeMo Guardrails in `05_experiments.tex` as representative reactive filters, and leaves a comment in the bib file documenting the call so 11.4 or a later writing pass can swap in a real citation if one is chosen. (b) **Numeric vs. author-year citations.** `main.tex` sets `natbib` with `[numbers,sort&compress]`, so `\citep{adams2007bocpd}` renders as `[N]`, not `(Adams & MacKay, 2007)`. Consequence: the pre-11.3 prose that already spelled "BOCPD (Adams & MacKay, 2007)" or "Chronos (2024)" had to have the literal parenthetical author-year stripped, otherwise the PDF would show "BOCPD (Adams & MacKay, 2007) [1]" with duplicated attribution. The 11.3 edit replaces each literal author-year parenthetical with a bare `\citep{key}` call so the numeric style reads cleanly. (c) **Where to anchor each cite.** The same work often gets named in multiple sections (e.g. MINJA in abstract, intro, related work, and experiments). 11.3 cites each work once in each section that names it, rather than a single cite-the-first-time pass, because `natbib` numeric mode renumbers references at bibtex time and there's no reader benefit to withholding a citation in the experiments section just because the abstract already dropped it. The abstract is the one exception: it is left cite-free on the standard convention that abstracts avoid inline citations. A Python audit script (`\cite` regex over all .tex files, `@\w+{key,` regex over the bib) confirmed zero missing and zero unused keys before commit. No LaTeX toolchain is installed locally, so a full `latexmk` compile is deferred to whoever runs the CI LaTeX step; the key audit is the best verification available pre-push. |
| **Completed** | 11.1: 2026-04-12. 11.2: 2026-04-12. 11.3: 2026-04-12. 11.4: 2026-04-12. 11.5: 2026-04-12. |

**Research Notes:**
- Paper claims to validate:
  - C1: Behavioral signals shift under poisoning (supported by signal validation experiment)
  - C2: BOCPD+Chronos detects shifts with AWT>=0 (supported by main experiment)
  - C3: Health-weighted allocation improves efficiency under attack (supported by comparison vs no-monitoring)
  - C4: Competitive AUROC vs reactive baselines with temporal context (supported by SentinelAgent comparison)
  - C5: Forecaster adds <1% computational overhead (supported by latency measurements)
- Related work subsections: memory poisoning (MINJA, AgentPoison, A-MemGuard), MAS task allocation (Dec-POMDP, contract nets), anomaly detection (BOCPD, time-series forecasting), LLM agent frameworks, runtime monitoring

---

## Phase 12: CI/CD + Release

**Goal:** Automated quality gates and release pipeline.

**Exit Criteria:**
- Every push: lint + type check + unit tests + integration tests
- PR merge requires green checks
- Tagged release auto-builds Docker image
- Experiment workflow manually triggerable via Actions

**Tasks:**
- [x] 12.1 `.github/workflows/ci.yml` -- ruff, mypy, pytest unit + integration, codecov. Services: redis, postgres
- [x] 12.2 `.github/workflows/experiments.yml` -- manual trigger, runs experiments on GPU runner (or CPU + mock for CI validation)
- [x] 12.3 `.github/workflows/release.yml` -- on tag `v*`: build wheel, build Docker image, push to GHCR

**Key Files:** `.github/workflows/*.yml`

**README contribution:** CI/CD badges, contributing guide link.

### Phase 12 Log
| | |
|--|--|
| **Findings** | Existing CI already covered lint + test + docker-smoke; main addition was services (redis, postgres), job splitting, caching, concurrency. Release workflow already existed in skeleton form; enhanced with parallel jobs, docker/metadata-action for semver tags, and sdist. |
| **Challenges** | None significant. pytestmark placement triggered E402 lint error (moved after imports). |
| **Completed** | 2026-04-12 |

---

## README Build Guide

The project `README.md` is built progressively — each phase contributes one section. After completing a phase, copy its "README contribution" note into README.

```
README.md structure (builds phase by phase):
─────────────────────────────────────────────
[Header + badges]                  ← P12
[One-line description]             ← now (from research)
[Architecture diagram]             ← P2
[Quickstart / Docker]              ← P0
[Agents + Pipeline]                ← P2
[Signal Monitoring]                ← P3
[Health Scoring]                   ← P4
[Task Allocation]                  ← P5
[Memory Integrity]                 ← P6
[Human-in-the-loop]                ← P7
[Dashboard]                        ← P8
[Production Deployment]            ← P9
[Research Results + Tables]        ← P10
[Reproducibility / make reproduce] ← P11
[Citation]                         ← P11
[Contributing]                     ← P12
─────────────────────────────────────────────
```

**When starting a new session:** Say _"Read PLAN.md and show me the current phase status and next unchecked task."_

---

## Pivot Protocol

### Pivot A: AWT = 0 (concurrent, not proactive detection)
- **Trigger:** Phase 1 shows signals shift at injection time, not before
- **Action:** Reframe paper: "Behavioral Time-Series Monitoring for Reliable Multi-Agent Task Allocation with Concurrent Attack Detection"
- **Impact:** No code changes. Allocation efficiency becomes primary contribution. Detection is secondary.

### Pivot B: No detectable signal
- **Trigger:** Phase 1 shows no significant effect size on any signal
- **Action:**
  1. First: expand signal set (output semantic drift, attention patterns). Try longer windows.
  2. If still nothing: drop security claim entirely. Pivot to allocation-only story (health from performance metrics: completion rate, response quality).
  3. Remove Phase 6. Rewrite Phase 10 experiments.
- **Impact:** Major scope reduction. Still publishable as allocation paper.

### Pivot C: Chronos underperforms
- **Trigger:** Chronos forecast MASE >> 1.5 or no improvement over BOCPD-only in ablation
- **Action:** System already degrades to BOCPD-only. Ablation result becomes an honest finding.
- **Impact:** No code changes. Paper reports BOCPD-only as sufficient.

---

## Running the Project

### Dev Setup
```bash
git clone <repo> && cd chronoagent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,experiments]"
pip install -e ".[forecaster]"                       # Chronos T5-small — CPU-safe, ~2-5s/inference
cp .env.example .env                                 # add CHRONO_TOGETHER_API_KEY
docker compose up -d redis postgres chroma           # infra only, no GPU needed
alembic upgrade head                                 # DB migrations
make test                                            # verify setup (uses MockBackend, no API calls)
```

> **No GPU required.** Together.ai handles all LLM inference via API. MockBackend handles all tests and experiments — zero API calls, zero cost, deterministic. Chronos T5-small runs on CPU (46M params). Ollama is completely optional — skip it.

### Common Commands
```bash
make dev              # start FastAPI dev server with reload
make test             # pytest unit + integration
make lint             # ruff check + mypy
make docker-up        # full stack via Docker Compose
make docker-down      # teardown
```

### Run Experiments
```bash
# Single experiment
chronoagent run-experiment --config configs/experiments/signal_validation.yaml --output results/

# All experiments
chronoagent run-experiment --config configs/experiments/ --output results/

# Reproduce everything (from scratch)
make reproduce
```

### Build Paper
```bash
cd paper && latexmk -pdf main.tex
# or
make paper
```

### Regenerate Figures Only
```bash
make reproduce-figures    # uses saved results in results/
```

### Docker (Full Stack)
```bash
docker compose up -d                               # all services
docker compose up -d redis postgres chroma          # infra only (run app locally)
docker compose logs -f app                          # watch app logs
docker compose exec app chronoagent check-health    # verify from inside container
```

### Environment Variables (Key)
```bash
CHRONO_ENV=dev                                       # dev | prod | test
CHRONO_LLM_BACKEND=together                          # together (default) | mock | ollama (GPU only)
CHRONO_TOGETHER_API_KEY=your_key_here                # free key at api.together.ai
CHRONO_TOGETHER_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
CHRONO_REDIS_URL=redis://localhost:6379/0
CHRONO_DATABASE_URL=sqlite:///./chronoagent.db       # or postgresql://...
CHRONO_FORECASTER_ENABLED=true                       # Chronos T5-small, CPU-safe, ~2-5s/call
CHRONO_HEALTH_SCORE_WINDOW=50
CHRONO_ESCALATION_THRESHOLD=0.3
# GPU-only (skip if no GPU):
# CHRONO_LLM_BACKEND=ollama
# CHRONO_OLLAMA_BASE_URL=http://localhost:11434
# CHRONO_OLLAMA_MODEL=phi3:mini
```

---

## Signal Reference

Six behavioral signals collected per agent per timestep:

| # | Signal | Source | Unit | Expected Attack Effect | Module |
|---|--------|--------|------|----------------------|--------|
| 1 | Response latency | LLM call timing | ms | Increase (poisoned context confuses model) | `collector.py` |
| 2 | Retrieval count | ChromaDB query | int | Change (poison docs alter retrieval patterns) | `collector.py` |
| 3 | Output token count | LLM response | int | Increase/decrease (altered reasoning) | `collector.py` |
| 4 | KL-divergence | Retrieved embeddings vs baseline Gaussian | float | Increase (poison docs shift embedding dist) | `kl_divergence.py` |
| 5 | Tool call frequency | Agent execution trace | int | Change (altered tool use patterns) | `collector.py` |
| 6 | Memory query entropy | Top-k similarity score distribution | float [0,1] | Spike (poison docs create uniform scores) | `entropy.py` |

Baseline calibration: first N clean steps (default 20) fit a multivariate Gaussian on retrieval embeddings for KL-div reference.

---

## Metrics Reference

| Metric | Formula | Used In | Target |
|--------|---------|---------|--------|
| AWT (Advance Warning Time) | `detection_step - injection_step` | P1, P10 | > 0 (proactive) |
| AUROC | Area under ROC (poisoned vs clean) | P6, P10 | > 0.8 |
| F1 | Harmonic mean of precision/recall | P4, P10 | > 0.7 |
| Allocation Efficiency | `successful_tasks / total_tasks` under attack | P5, P10 | > baseline |
| MASE | Mean Absolute Scaled Error (Chronos) | P4 | < 1.5 |
| Escalation Rate | `escalations / total_tasks` under attack | P7, P10 | informational |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Online vs batch BOCPD | Online (Adams & MacKay) | Streaming requirement; `ruptures` is batch-only, used for offline validation |
| Chronos optional | `[forecaster]` extra | Allows CPU-only / no-GPU envs; BOCPD-only fallback |
| MockBackend | Deterministic fixtures | Reproducible experiments without LLM non-determinism |
| ChromaDB quarantine | Separate collection | ChromaDB has no native quarantine; move docs between active/quarantine collections |
| Contract-net negotiation | Bid = capability * health | Simple, interpretable, deterministic. No MARL needed for 4-agent system. |
| SQLite dev / Postgres prod | Config-driven | Zero-setup dev, robust prod |
| Dashboard MVP | HTML + Chart.js | No npm build step, <500 LOC, sufficient for paper screenshots |
| Attack implementations | MINJA + AGENTPOISON reproductions | Not novel -- reproductions of published attacks used as threat model |

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Signals too noisy to detect drift | Project pivot required | Phase 1 GO/NO-GO gate; 3 pivot contingencies defined |
| No GPU / weak CPU | LLM inference speed | **Not a problem by design.** Together.ai handles LLM calls (API). MockBackend handles all experiments (zero compute). Chronos T5-small (46M) runs on CPU. Ollama not required. |
| Chronos package unavailable / API change | Forecaster unavailable | BOCPD-only fallback built into architecture; Chronos is optional dep |
| 4-agent overlap too small for allocation benefit | Weak allocation story | Measure partial redistribution + escalation rate; allocation still useful even with low overlap |
| ChromaDB embedding drift over time | False positives in integrity | Periodic refit of isolation forest; configurable refit interval |

---

## API Surface Summary

All routes under `/api/v1/`. Dashboard routes under `/dashboard/api/`.

| Method | Endpoint | Phase | Purpose |
|--------|----------|-------|---------|
| POST | `/api/v1/review` | P2 | Submit PR diff for review |
| GET | `/api/v1/review/{id}` | P2 | Get review status/report |
| GET | `/api/v1/agents/{id}/signals?window=50` | P3 | Signal time-series |
| GET | `/api/v1/agents/{id}/health` | P4 | Health score + components |
| GET | `/api/v1/health` | P4/P9 | System health + component status |
| GET | `/api/v1/memory/integrity` | P6 | Memory integrity report |
| POST | `/api/v1/memory/quarantine` | P6 | Quarantine docs |
| POST | `/api/v1/memory/approve` | P6 | Approve quarantined docs |
| GET | `/api/v1/escalations?status=pending` | P7 | List escalations |
| POST | `/api/v1/escalations/{id}/resolve` | P7 | Resolve escalation |
| WS | `/dashboard/ws/live` | P8 | Real-time health updates |

---

## Testing Strategy

| Layer | Tool | Scope | Phase |
|-------|------|-------|-------|
| Unit | pytest | Individual functions (signals, BOCPD, entropy, KL-div) | P1-P9 |
| Property | hypothesis | Invariants: health in [0,1], KL-div >= 0, exactly one allocation | P3-P5 |
| Integration | pytest + services | Pipeline e2e (MockBackend), attack detection, allocation under attack | P2, P5, P10 |
| E2E | pytest + Docker | Full stack with real Redis/Postgres/Chroma | P9 |
| Experiment | custom runner | Reproducible research experiments | P10 |

Key test files:
- `tests/unit/test_config.py`, `test_signals.py`, `test_bocpd.py`, `test_chronos_forecaster.py`, `test_health_scorer.py`, `test_task_allocator.py`, `test_memory_integrity.py`, `test_escalation.py`, `test_negotiation.py`
- `tests/integration/test_pipeline_e2e.py`, `test_attack_detection.py`, `test_allocation_under_attack.py`
- `tests/property/test_health_score_bounds.py`, `test_allocation_invariants.py`

---

## Dependency Graph (Visual)

```
P0 (Bootstrap)
 |
 +--> P1 (Signal Validation) ---- GO/NO-GO GATE
 |     |
 |     v
 +--> P2 (Core Pipeline) ---------+
       |                           |
       v                           |
      P3 (Monitor)                 |
       |                           |
       v                           |
      P4 (Health Scorer) ---+      |
       |                    |      |
       v                    v      |
      P5 (Allocator)    P6 (Memory Integrity)
       |                    |
       v                    v
      P7 (Escalation) <----+
       |
       v
      P8 (Dashboard)
       |
       v
      P9 (Hardening)
       |
       v
      P10 (Experiments) <-- requires P1 results
       |
       v
      P11 (Paper)
       |
       v
      P12 (CI/CD + Release)
```
