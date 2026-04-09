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
| 1 | Signal Validation (GO/NO-GO) | **High** | `[ ]` | 0 |
| 2 | Core Agent Pipeline | Medium | `[ ]` | 0, 1 (collector) |
| 3 | Behavioral Monitor | Medium | `[ ]` | 2 |
| 4 | Temporal Health Scorer | **High** | `[ ]` | 3 |
| 5 | Decentralized Task Allocator | **High** | `[ ]` | 4 |
| 6 | Memory Integrity Module | Medium | `[ ]` | 2, 3 |
| 7 | Human Escalation Layer | Low | `[ ]` | 4, 6 |
| 8 | Observability Dashboard | Medium | `[ ]` | 4, 5, 6, 7 |
| 9 | Production Hardening | Medium | `[ ]` | 0-8 |
| 10 | Research Experiment Suite | **High** | `[ ]` | 1-9 |
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
- [ ] 1.2 `monitor/collector.py` -- `BehavioralCollector` with `StepSignals` dataclass (6 fields), `start_step`/`end_step`, `get_signal_matrix() -> (T, 6) ndarray`
- [ ] 1.3 `monitor/kl_divergence.py` -- Calibration run fits Gaussian on clean retrieval embeddings; per-step KL from baseline via `scipy.stats.entropy`
- [ ] 1.4 `monitor/entropy.py` -- Shannon entropy of top-k similarity score distribution per retrieval; normalized to [0,1]
- [ ] 1.5 `memory/poisoning.py` -- `MINJAStyleAttack` (query-optimized trigger embeddings) + `AGENTPOISONStyleAttack` (backdoor trigger docs); both inject into ChromaDB
- [ ] 1.6 `experiments/runner.py` -- Phase A: clean run (N PRs) -> Phase B: inject attack -> Phase C: poisoned run (N PRs) -> Phase D: compute stats
- [ ] 1.7 `configs/experiments/signal_validation.yaml` -- seed, step counts, attack type, agent list, signal list, analysis params
- [ ] 1.8 Analysis script -- per-signal time-series plots, Cohen's d, PELT changepoint detection, AWT estimation, decision matrix table
- [ ] 1.9 Write decision document with pivot ruling

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
| **Signal Results** | _fill in: Cohen's d per signal, AWT estimate_ |
| **GO/NO-GO Decision** | _fill in: GO / PIVOT A / PIVOT B_ |
| **Pivot Taken** | _fill in: none / Pivot A / Pivot B — and why_ |
| **Findings** | _fill in_ |
| **Challenges** | _fill in_ |
| **Decisions** | _fill in_ |
| **Completed** | _fill in: date_ |

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
- [ ] 2.1 `agents/base.py` -- `BaseAgent` ABC: `execute(Task) -> TaskResult`, instrumented `_call_llm`, `_retrieve_memory`
- [ ] 2.2 `agents/backends/` -- `LLMBackend` ABC (`generate`, `embed`), `TogetherAIBackend` (default), `MockBackend` (deterministic fixtures — primary for all tests/experiments), `OllamaBackend` (optional, skip if no GPU)
- [ ] 2.3 `agents/planner.py` -- Decomposes PR diff into `list[ReviewSubtask]` with task_type + code_segment; queries memory for similar decompositions
- [ ] 2.4 `agents/security_reviewer.py` -- Full impl: checks for vulns, outputs `SecurityFinding` (severity, desc, line ref); queries CWE patterns from memory
- [ ] 2.5 `agents/style_reviewer.py` -- Checks quality/naming/complexity, outputs `StyleFinding`; queries style conventions from memory
- [ ] 2.6 `agents/summarizer.py` -- Synthesizes all findings into `ReviewReport` (markdown); queries report templates
- [ ] 2.7 `agents/registry.py` -- `AgentRegistry`: capability map, agent lookup by type
- [ ] 2.8 `memory/store.py` -- ChromaDB wrapper: `add`, `query`, `delete`, `get_all_embeddings`
- [ ] 2.9 Seed data script -- ~50 vuln patterns (CWE top 25), ~30 style docs, ~10 report templates, ~20 sample reviews
- [ ] 2.10 `pipeline/graph.py` -- LangGraph `StateGraph`: plan -> (security_review || style_review) -> summarize -> END
- [ ] 2.11 `api/routers/review.py` -- `POST /api/v1/review`, `GET /api/v1/review/{id}`
- [ ] 2.12 `tests/integration/test_pipeline_e2e.py` -- MockBackend, submit diff, assert ReviewReport with findings, assert agent order

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
- [ ] 3.1 Expand `BehavioralCollector` -- add PostgreSQL persistence via SQLAlchemy, baseline calibration after N steps, rolling window stats
- [ ] 3.2 Complete `kl_divergence.py` -- edge cases (zero variance, insufficient samples), unit tests with known distributions
- [ ] 3.3 Complete `entropy.py` -- edge cases (single result, empty), normalize [0,1], unit tests (uniform=max, delta=0)
- [ ] 3.4 `db/models.py` -- `AgentSignalRecord` table (agent_id, task_id, timestamp, 6 signal columns); Alembic migration
- [ ] 3.5 `api/routers/signals.py` -- `GET /api/v1/agents/{agent_id}/signals?window=50`
- [ ] 3.6 Unit tests: `test_signals.py`, `test_kl_divergence.py`, `test_entropy.py`
- [ ] 3.7 Property test via Hypothesis: KL-div and entropy always non-negative for valid inputs

**Key Files:** `monitor/collector.py`, `monitor/kl_divergence.py`, `monitor/entropy.py`, `db/models.py`

**Research:** Data collection layer for all experiments. Correctness is critical.

**README contribution:** Signal monitoring section — what the 6 signals are and why each was chosen.

### Phase 3 Log
| | |
|--|--|
| **Findings** | _fill in_ |
| **Challenges** | _fill in_ |
| **Decisions** | _fill in_ |
| **Completed** | _fill in: date_ |

---

## Phase 4: Temporal Health Scorer

**Goal:** BOCPD + Chronos-2-Small ensemble producing per-agent health scores in [0,1], async, with graceful degradation.

**Exit Criteria:**
- Health score updates within 500ms of new signal (non-blocking)
- BOCPD detects synthetic changepoints with F1 > 0.8
- Chronos forecasts next 10 steps with MASE < 1.5
- Graceful fallback to BOCPD-only when Chronos unavailable
- Health scores on API and message bus

**Tasks:**
- [ ] 4.1 `scorer/bocpd.py` -- Online BOCPD (Adams & MacKay 2007, ~80 lines NumPy) for streaming; `ruptures` for offline validation. `update(obs) -> changepoint_probability [0,1]`
- [ ] 4.2 `scorer/chronos_forecaster.py` -- Lazy-loaded Chronos-2-Small (46M params). `forecast(history) -> ForecastResult`. `compute_anomaly_score(history, actual) -> [0,1]`. Returns None if unavailable.
- [ ] 4.3 `scorer/ensemble.py` -- `health = 1 - (w_bocpd * bocpd + w_chronos * chronos)`, clamped [0,1]. Single-component fallback when other missing. Weights configurable.
- [ ] 4.4 `scorer/health_scorer.py` -- `TemporalHealthScorer`: subscribes to signals via message bus, maintains signal buffer, orchestrates BOCPD + Chronos, publishes `HealthUpdate`
- [ ] 4.5 `messaging/bus.py` + `messaging/redis_bus.py` + `messaging/local_bus.py` -- MessageBus ABC, Redis pub/sub impl, in-memory impl for dev/test
- [ ] 4.6 `api/routers/health.py` -- `GET /api/v1/agents/{id}/health` (score + components), `GET /api/v1/health` (all agents + system health)
- [ ] 4.7 Tests: synthetic changepoint at step 50 -> detect within 5 steps; sine wave forecast low residual; sudden jump -> high anomaly; ensemble fallbacks; Hypothesis: health always in [0,1]

**Key Files:** `scorer/*.py`, `messaging/*.py`, `api/routers/health.py`

**Research Notes:**
- Core novel component. Accuracy, calibration, latency measured in every experiment.
- Ablation configs disable BOCPD or Chronos independently to measure individual contribution.
- Key implementation note: `ruptures` is batch/offline only. For streaming BOCPD, implement Adams & MacKay (2007) directly (~80 lines NumPy). Use `ruptures` for offline experiment validation.
- Chronos-2-Small (46M params) is Apache 2.0 licensed. Use `amazon/chronos-t5-small` checkpoint. Lazy-load to avoid startup cost.
- Health score interpretation: 1.0 = fully healthy, 0.0 = confirmed anomalous. Scores below escalation_threshold (default 0.3) trigger human escalation.

**README contribution:** How health scoring works — BOCPD + Chronos ensemble diagram, health score interpretation, API example (`GET /api/v1/agents/{id}/health`).

### Phase 4 Log
| | |
|--|--|
| **BOCPD F1 result** | _fill in: synthetic changepoint test_ |
| **Chronos MASE result** | _fill in_ |
| **Latency overhead** | _fill in: async update time (target <500ms)_ |
| **Chronos Pivot C?** | _fill in: BOCPD-only or ensemble_ |
| **Findings** | _fill in_ |
| **Challenges** | _fill in_ |
| **Decisions** | _fill in_ |
| **Completed** | _fill in: date_ |

---

## Phase 5: Decentralized Task Allocator

**Goal:** Health-score + capability-weighted task routing via contract-net negotiation. No central controller.

**Exit Criteria:**
- Degraded agent -> tasks shift to healthier agents with overlapping capabilities
- All allocation decisions logged with bids, scores, rationale
- Allocation efficiency metric computable
- Round-robin fallback on allocator failure

**Tasks:**
- [ ] 5.1 `allocator/capability_weights.py` -- Static capability matrix: 4 agents x 4 task types, proficiency in [0,1]
- [ ] 5.2 `allocator/negotiation.py` -- Contract-net variant: broadcast task -> agents bid (capability * health) -> highest wins -> ties broken by agent_id. Min bid threshold -> escalate to human.
- [ ] 5.3 `allocator/task_allocator.py` -- `DecentralizedTaskAllocator`: subscribes to health updates, maintains health snapshot, calls negotiation protocol
- [ ] 5.4 Integrate into `pipeline/graph.py` -- LangGraph conditional edges: after Planner, each subtask routed via allocator at runtime
- [ ] 5.5 `db/models.py` -- `AllocationAuditRecord` (task_id, assigned_agent, all_bids JSON, health_snapshot JSON, rationale, escalated)
- [ ] 5.6 Fallback: negotiation timeout/error -> round-robin with warning log
- [ ] 5.7 Tests: mock health -> verify highest-bid wins; all-low-health -> escalation; timeout handling; Hypothesis: exactly one agent assigned or escalated

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
| **Allocation efficiency vs baseline** | _fill in: % improvement_ |
| **Negotiation protocol issues** | _fill in: any edge cases hit_ |
| **Findings** | _fill in_ |
| **Challenges** | _fill in_ |
| **Decisions** | _fill in_ |
| **Completed** | _fill in: date_ |

---

## Phase 6: Memory Integrity Module

**Goal:** Detect anomalous memory entries, quarantine them, surface for human review.

**Exit Criteria:**
- Integrity check on every retrieval
- Anomalous docs flagged with confidence score
- Quarantine excludes flagged docs from future retrieval until approved
- Detection AUROC > 0.8 on synthetic attacks

**Tasks:**
- [ ] 6.1 `memory/integrity.py` -- `MemoryIntegrityModule`: 4 detection signals (embedding outlier score, freshness anomaly, retrieval frequency spike, content-embedding mismatch). `check_retrieval(query, docs) -> IntegrityResult`
- [ ] 6.2 Isolation forest (sklearn) fitted on clean embeddings for outlier detection; periodic refit every N new docs
- [ ] 6.3 `memory/quarantine.py` -- Separate ChromaDB collection for quarantined docs. Move on flag, restore on approve. Retrieval only queries active collection.
- [ ] 6.4 `api/routers/memory.py` -- `GET /api/v1/memory/integrity`, `POST /api/v1/memory/quarantine`, `POST /api/v1/memory/approve`
- [ ] 6.5 Tests: inject poison docs -> flagged; clean docs -> not flagged; quarantine/approve flow

**Key Files:** `memory/integrity.py`, `memory/quarantine.py`

**Research:** AUROC + F1 for poison detection. Component of overall system eval.

**README contribution:** Memory integrity section — how detection works, quarantine/approve workflow, API examples.

### Phase 6 Log
| | |
|--|--|
| **Detection AUROC** | _fill in (target >0.8)_ |
| **False positive rate** | _fill in_ |
| **Findings** | _fill in_ |
| **Challenges** | _fill in_ |
| **Decisions** | _fill in_ |
| **Completed** | _fill in: date_ |

---

## Phase 7: Human Escalation Layer

**Goal:** Auto-escalate on low health or memory flags. Full audit trail. Cooldown to prevent spam.

**Exit Criteria:**
- Escalation triggers on health < threshold OR quarantine event
- Escalation includes full context (signals, health history, flagged docs, allocation trail)
- Human resolve via API (approve/reject/modify)
- Cooldown prevents repeated escalation for same agent

**Tasks:**
- [ ] 7.1 `escalation/escalation_manager.py` -- `EscalationHandler`: threshold check, cooldown tracking, context assembly, publish escalation event
- [ ] 7.2 `escalation/audit.py` -- `AuditTrailLogger`: log events (allocation, health update, escalation, quarantine, approval) to PostgreSQL `audit_events` table
- [ ] 7.3 `api/routers/escalation.py` -- `GET /api/v1/escalations?status=pending`, `POST /api/v1/escalations/{id}/resolve`
- [ ] 7.4 Tests: low health -> escalation fires; cooldown -> no repeat; resolve updates status; full context present

**Key Files:** `escalation/*.py`

**Research:** Escalation rate under attack is a measured metric.

**README contribution:** Human-in-the-loop section — when and why escalation fires, how to resolve via API.

### Phase 7 Log
| | |
|--|--|
| **Findings** | _fill in_ |
| **Challenges** | _fill in_ |
| **Decisions** | _fill in_ |
| **Completed** | _fill in: date_ |

---

## Phase 8: Observability Dashboard

**Goal:** Real-time viz of agent health, routing, memory status, escalation queue.

**Exit Criteria:**
- Per-agent health time-series displayed
- Allocation decisions with bid details visible
- Memory integrity + escalation queue displayed
- Loads <3s, near-real-time updates

**Tasks:**
- [ ] 8.1 Dashboard backend -- FastAPI endpoints: `/dashboard/api/agents`, `/dashboard/api/agents/{id}/timeline`, `/dashboard/api/allocations`, `/dashboard/api/memory`, `/dashboard/api/escalations`, WebSocket `/dashboard/ws/live`
- [ ] 8.2 Dashboard frontend (MVP: single HTML + Chart.js, no build step, ~500 LOC). Views: Agent Health Panel (gauges + sparklines), Signal Explorer, Allocation Log, Memory Inspector, Escalation Queue
- [ ] 8.3 Prometheus metrics (optional prod path): `src/chronoagent/observability/metrics.py` -- gauges/counters/histograms + Grafana dashboard JSON

**Key Files:** `dashboard/`, `observability/metrics.py`

**Research:** Dashboard screenshots for paper system design section. Signal explorer for qualitative analysis.

**README contribution:** Dashboard section — screenshot, how to run it, what each panel shows.

### Phase 8 Log
| | |
|--|--|
| **Findings** | _fill in_ |
| **Challenges** | _fill in_ |
| **Decisions** | _fill in_ |
| **Completed** | _fill in: date_ |

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
- [ ] 9.1 `api/middleware.py` -- Rate limiting (slowapi or custom): POST 10/min, GET 60/min, WS 5 concurrent
- [ ] 9.2 Retry wrappers (tenacity): 3 attempts, exponential backoff, all external calls
- [ ] 9.3 Graceful degradation in `create_app()` -- try/except for each component, fallback to local alternatives
- [ ] 9.4 Comprehensive `/api/v1/health` -- per-component status (api, redis, postgres, chromadb, together_ai, forecaster), degraded/healthy/unhealthy. Ollama component omitted unless configured.
- [ ] 9.5 Structured logging audit -- verify every module uses structlog with consistent fields
- [ ] 9.6 Package build -- `python -m build` produces wheel, entry point `chronoagent = chronoagent.cli:app`

**Key Files:** `api/middleware.py`, `main.py`, `pyproject.toml`

**README contribution:** Production deployment section — Docker, environment variables, health checks, rate limits.

### Phase 9 Log
| | |
|--|--|
| **Findings** | _fill in_ |
| **Challenges** | _fill in_ |
| **Decisions** | _fill in_ |
| **Completed** | _fill in: date_ |

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
- [ ] 10.1 `experiments/config_schema.py` -- `ExperimentConfig` Pydantic model: name, seed, num_runs, num_prs, AttackConfig (type, target, injection step, strategy), AblationConfig (forecaster/bocpd/health/integrity toggles), SystemConfig
- [ ] 10.2 `experiments/metrics.py` -- `advance_warning_time(injection, detection)`, `allocation_efficiency(results)`, `detection_auroc(y_true, y_scores)`, `detection_f1(y_true, y_pred)`
- [ ] 10.3 `experiments/baselines/sentinel.py` -- Reactive baseline: execution trace matching, no temporal forecasting, no health scores
- [ ] 10.4 `experiments/baselines/no_monitoring.py` -- No monitoring, round-robin allocation, no integrity checks
- [ ] 10.5 Six experiment configs:

| Config | Attack | Ablation | Purpose |
|--------|--------|----------|---------|
| `main_experiment.yaml` | MINJA | Full system | Main result |
| `agentpoison_experiment.yaml` | AGENTPOISON | Full system | Generalization |
| `ablation_no_forecaster.yaml` | MINJA | Chronos off | Forecaster contribution |
| `ablation_no_bocpd.yaml` | MINJA | BOCPD off | BOCPD contribution |
| `ablation_no_health_scores.yaml` | MINJA | Health off | Health scoring contribution |
| `baseline_sentinel.yaml` | MINJA | Sentinel | Reactive comparison |

- [ ] 10.6 `experiments/runner.py` (full) -- Per run: seed, build system from config, clean phase, inject attack, post-attack phase, compute metrics. Aggregate across N runs (mean, std, CI).
- [ ] 10.7 `experiments/analysis/plots.py` -- 6 figures: signal drift viz, health score comparison, AWT box plot, allocation efficiency over time, ROC curve, ablation bar chart
- [ ] 10.8 `experiments/analysis/tables.py` -- LaTeX tables: main results, ablation results, signal validation
- [ ] 10.9 CLI: `chronoagent run-experiment --config <path> --output results/`

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
| **Findings** | _fill in_ |
| **Challenges** | _fill in_ |
| **Completed** | _fill in: date_ |

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
- [ ] 11.1 Paper structure: `main.tex` + sections (abstract, intro, related work, system design, methodology, experiments, results, discussion, conclusion)
- [ ] 11.2 Map claims to experiments:
  - C1: Behavioral signals shift under poisoning (signal validation)
  - C2: BOCPD+Chronos ensemble detects shifts with AWT>=0 (main experiment)
  - C3: Health-weighted allocation improves efficiency under attack (vs no-monitoring)
  - C4: Competitive AUROC vs reactive baselines with temporal context (vs SentinelAgent)
  - C5: Forecaster adds <1% computational overhead (latency measurement)
- [ ] 11.3 `bibliography.bib` -- all cited works
- [ ] 11.4 Makefile targets: `reproduce`, `reproduce-signal`, `reproduce-main`, `reproduce-ablations`, `reproduce-figures`
- [ ] 11.5 Pin all versions in Docker image; all seeds fixed; all configs in repo

**Key Files:** `paper/*.tex`, `paper/bibliography.bib`, `Makefile`

**README contribution:** Citation block, link to paper PDF/arXiv, reproducibility badge (`make reproduce`).

### Phase 11 Log
| | |
|--|--|
| **Findings** | _fill in_ |
| **Challenges** | _fill in_ |
| **Completed** | _fill in: date_ |

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
- [ ] 12.1 `.github/workflows/ci.yml` -- ruff, mypy, pytest unit + integration, codecov. Services: redis, postgres
- [ ] 12.2 `.github/workflows/experiments.yml` -- manual trigger, runs experiments on GPU runner (or CPU + mock for CI validation)
- [ ] 12.3 `.github/workflows/release.yml` -- on tag `v*`: build wheel, build Docker image, push to GHCR

**Key Files:** `.github/workflows/*.yml`

**README contribution:** CI/CD badges, contributing guide link.

### Phase 12 Log
| | |
|--|--|
| **Findings** | _fill in_ |
| **Challenges** | _fill in_ |
| **Completed** | _fill in: date_ |

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
pip install -e ".[forecaster]"                       # Chronos-2-Small — CPU-safe, ~2-5s/inference
cp .env.example .env                                 # add CHRONO_TOGETHER_API_KEY
docker compose up -d redis postgres chroma           # infra only, no GPU needed
alembic upgrade head                                 # DB migrations
make test                                            # verify setup (uses MockBackend, no API calls)
```

> **No GPU required.** Together.ai handles all LLM inference via API. MockBackend handles all tests and experiments — zero API calls, zero cost, deterministic. Chronos-2-Small runs on CPU (46M params). Ollama is completely optional — skip it.

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
CHRONO_FORECASTER_ENABLED=true                       # Chronos-2-Small, CPU-safe, ~2-5s/call
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
| No GPU / weak CPU | LLM inference speed | **Not a problem by design.** Together.ai handles LLM calls (API). MockBackend handles all experiments (zero compute). Chronos-2-Small (46M) runs on CPU. Ollama not required. |
| Chronos-2 not yet released / API change | Forecaster unavailable | BOCPD-only fallback built into architecture; Chronos is optional dep |
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
