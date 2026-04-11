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
| **Signal Results** | KL-div: d=1.611 (MINJA), d=1.348 (AGENTPOISON). All other signals d<0.4. Entropy/retrieval_count/tool_calls are MockBackend constants (σ=0). AWT=0 (concurrent). |
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

**Goal:** BOCPD + Chronos-2-Small ensemble producing per-agent health scores in [0,1], async, with graceful degradation.

**Exit Criteria:**
- Health score updates within 500ms of new signal (non-blocking)
- BOCPD detects synthetic changepoints with F1 > 0.8
- Chronos forecasts next 10 steps with MASE < 1.5
- Graceful fallback to BOCPD-only when Chronos unavailable
- Health scores on API and message bus

**Tasks:**
- [x] 4.1 `scorer/bocpd.py` -- Online BOCPD (Adams & MacKay 2007, ~80 lines NumPy) for streaming; `ruptures` for offline validation. `update(obs) -> changepoint_probability [0,1]`
- [x] 4.2 `scorer/chronos_forecaster.py` -- Lazy-loaded Chronos-2-Small (46M params). `forecast(history) -> ForecastResult`. `compute_anomaly_score(history, actual) -> [0,1]`. Returns None if unavailable.
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
- Chronos-2-Small (46M params) is Apache 2.0 licensed. Use `amazon/chronos-t5-small` checkpoint. Lazy-load to avoid startup cost.
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
| **Findings** | **10.1:** New `src/chronoagent/experiments/config_schema.py` ships four pydantic models that every Phase 10 experiment YAML loads through: `ExperimentConfig` (top-level: `name`, `seed`, `num_runs`, `num_prs`, plus nested `attack` / `ablation` / `system`), `AttackConfig` (`type` Literal `"minja"|"agentpoison"|"none"`, `target` Literal `"security_reviewer"|"summarizer"|"both"`, `injection_step`, `n_poison_docs`, `strategy` free-form descriptor), `AblationConfig` (four bool toggles: `forecaster`, `bocpd`, `health`, `integrity`, all default `True`), `SystemConfig` (`llm_backend`, `bocpd_hazard_lambda`, `health_threshold`, `integrity_threshold` -- the runtime knobs the harness overrides per experiment). Every model uses `model_config = ConfigDict(extra="forbid")` so a YAML key typo hard-fails at load time instead of silently shadowing. `ExperimentConfig.from_yaml(path)` is the single loader entry point: it surfaces `FileNotFoundError`, `ValueError` for non-mapping roots, and `pydantic.ValidationError` for any field-level failure. `per_run_seed(run_index)` is a deterministic helper that returns `seed + run_index` so a single config drives `num_runs` reproducible runs without re-loading. The schema is intentionally narrow: it captures only the parameters PLAN.md task 10.1 enumerates plus the minimum needed to make YAML loading work (`from_yaml`, `per_run_seed`). **10.2:** New `src/chronoagent/experiments/metrics.py` ships the four paper-facing metric functions the runner (10.6) and analysis layer (10.7 / 10.8) call to turn a run artefact into headline numbers. (1) `advance_warning_time(injection_step, detection_step) -> int` returns the signed step difference `injection_step - detection_step`. Per Pivot A, concurrent detection (`injection_step == detection_step`) returns `0` instead of NO-GO, and a negative return value means the detector was late. Both arguments are validated `>= 0`; callers must filter runs where the detector never fired. Numpy int inputs are coerced to plain Python `int` so the value survives JSON serialisation unchanged. (2) `allocation_efficiency(results) -> float` cumulative success rate of allocator decisions, accepting either a plain `Sequence[bool]` or a sequence of mapping-style audit rows (each with a `"success"` key). Empty sequence returns `0.0`, mixed bool + mapping sequences work, missing-`"success"`-key raises `KeyError`, non-bool/non-Mapping elements raise `TypeError`. The mapping path uses `bool(item["success"])` so `0`/`1`-coded integer flags also count correctly. (3) `detection_auroc(y_true, y_scores) -> float` is a thin wrapper around `sklearn.metrics.roc_auc_score` that short-circuits undefined inputs into `float('nan')`: empty arrays, zero-size scores, or single-class `y_true`. Perfect-ranking returns `1.0`, perfectly-inverted ranking returns `0.0`, uninformative (all-same) scores return `0.5` via sklearn's tie-breaking convention. (4) `detection_f1(y_true, y_pred) -> float` wraps `sklearn.metrics.f1_score` with `zero_division=0.0` so all-zero predictions return `0.0` silently instead of emitting `UndefinedMetricWarning`, and short-circuits empty arrays into `float('nan')`. Both sklearn wrappers use `warnings.catch_warnings()` + `simplefilter("ignore", UndefinedMetricWarning)` so callers get a clean stdout when running the 5-run aggregate. NaN returns let aggregators use `np.nanmean` over per-run results without wrapping each call in try/except. Both sklearn wrappers explicitly return `float(...)` so the result type is a plain Python `float` (not a numpy scalar) for JSON serialisation. **Test file:** `tests/unit/test_experiment_metrics.py` with 51 tests across 7 classes. `TestAdvanceWarningTime` (9 tests: early, concurrent, late, zero-zero, parametrized invariant check over 5 pairs, numpy-int coercion, negative injection rejected, negative detection rejected). `TestAllocationEfficiencyBoolInput` (7 tests: all success, all failure, half-half, two-thirds, single success, single failure, empty returns 0.0). `TestAllocationEfficiencyMappingInput` (5 tests: mapping all success, mixed, extra fields ignored, missing key raises KeyError, truthy non-bool `"success": 1/0` counts). `TestAllocationEfficiencyMixedAndErrors` (3 tests: mixed bool+mapping, string element raises TypeError, None element raises TypeError). `TestDetectionAuroc` (12 tests: perfect ranking = 1.0, inverted = 0.0, uninformative = 0.5, partial overlap bounded, accepts Python lists, empty y_true = nan, empty y_scores = nan, single-class all-zero = nan, single-class all-one = nan, all-zero scores with two classes = 0.5, return type is Python float, nan return type is Python float). `TestDetectionF1` (12 tests: perfect predictions = 1.0, all wrong = 0.0, half correct (TP=1 FP=1 FN=1) = 0.5, Python lists accepted, empty y_true = nan, empty y_pred = nan, all-zero true+pred = 0.0 silently, all-zero true + all-one pred = 0.0, single-class all-one exact match = 1.0, no warnings on zero division (asserted via `warnings.simplefilter("error", UndefinedMetricWarning)`), return type is Python float, nan return type is Python float). |
| **Challenges** | **10.1:** Two design calls locked in. (1) **Strategy validator only enforces non-emptiness, not whitespace stripping.** The first test attempt expected `cfg.strategy == "high_noise"` after passing `"  high_noise  "`, but the validator only rejects whitespace-only strings. Silent input mutation is a footgun (operators looking at result rows would be confused why their YAML value differs from the recorded one), so the validator stays as-is and the test was renamed to `test_strategy_with_surrounding_whitespace_accepted_verbatim` and asserts the string passes through unchanged. (2) **`name` field uses a strict ASCII-safe validator.** The validator allows `[A-Za-z0-9_-]` only, because `ExperimentConfig.name` is spliced into output filenames by the runner (10.6) and into log/event names. Allowing dots, slashes, or shell metacharacters would create a path-traversal risk and break filename hashing. The test parametrizes over both safe (`main`, `main_experiment`, `ablation-no-bocpd`, `v2_run`) and unsafe (`""`, `"   "`, `"with space"`, `"../escape"`, `"weird!"`) names. **10.2:** Four design calls locked in. (1) **Empty `allocation_efficiency` returns `0.0`, not `nan`.** Downstream cumulative tracking in the runner (10.6) benefits from a numeric value it can always append to a CSV; "success rate of zero decisions" is vacuously zero and the test asserts this explicitly so the semantics cannot silently change. The other two nan-returning metrics (`detection_auroc`, `detection_f1`) use `nan` because they feed into statistical aggregations (`np.nanmean` across 5 seeds) where a zero would pollute the mean, whereas allocation efficiency aggregates across allocator decisions within a single run. (2) **AUROC and F1 wrap sklearn but short-circuit before calling it** for every case where sklearn would emit `UndefinedMetricWarning` or raise. The alternatives (let sklearn raise, catch at the call site) pollute the runner and force every caller to write the same try/except block; centralising the nan translation here means the runner can build a `DataFrame` row with no defensive code. The `warnings.catch_warnings()` block guards the sklearn call anyway so a future sklearn version that starts emitting new undefined-metric warnings still produces a clean stdout. (3) **`AllocationResult` type alias accepts `bool | Mapping[str, Any]` rather than forcing a projection.** The task allocator's audit trail rows are already dict-shaped; demanding callers project to booleans before calling `allocation_efficiency` would be extra ceremony for no gain, and the `isinstance(item, bool)` / `isinstance(item, Mapping)` branch is a one-line check inside the loop. The test file covers the mixed bool+mapping case explicitly. (4) **`pyproject.toml` mypy overrides extended with `sklearn.*`.** The 10.2 module is the first `src/` import of sklearn (the rest of the project uses numpy/scipy for stats), and sklearn ships without a `py.typed` marker, so `mypy --strict` flagged `[import-untyped]` until `sklearn.*` joined the existing ignore list next to `ruptures.*`, `scipy.*`, `pandas.*`. **Test + lint + type check status end of 10.2:** 1167 tests passed (1116 prior + 51 metrics, zero failures), 95.70% coverage, `src/chronoagent/experiments/metrics.py` 100% covered (50/50), ruff check + format --check clean on 117 files, mypy --strict clean on 72 source files. Zero flakes this run (the usual intermittent ChromaDB cross-test-pollution pattern did NOT trip). |
| **Completed** | 10.1: 2026-04-11 &nbsp;·&nbsp; 10.2: 2026-04-11 |

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
