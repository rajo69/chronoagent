# ChronoAgent: Session Context for Claude Code

> This file is auto-read at every session start. Keep it current. Full plan is in PLAN.md.

---

## Current Status

| Field | Value |
|-------|-------|
| **Current phase** | Phase 6: Memory Integrity Module (Phase 5 closed) |
| **Next task** | 6.1, `memory/integrity.py` -- `MemoryIntegrityModule` with 4 detection signals (embedding outlier, freshness anomaly, retrieval frequency spike, content-embedding mismatch); `check_retrieval(query, docs) -> IntegrityResult` |
| **Blocked?** | No |
| **Last session** | 2026-04-10. Phase 5 task 5.7 done [x] -> **Phase 5 closed**. Branch `feat/phase5-task5.7-allocator-tests` cut off `main` after merging the 5.6 PR (#8) earlier in the same session. 5.7 is a *test sweep*, no product code change. Added `TestPhase57IntegratedSweep` (3 Hypothesis tests at the bus + cache + negotiation layer) and `TestPhase57DegradationWalk` (1 deterministic three-step walk) to `tests/unit/test_task_allocator.py`. The Hypothesis tests cover: (a) exactly-one-of-(assigned, escalated) invariant for any random valid snapshot published over the bus, plus the highest-bid invariant on the result ledger; (b) byte-equivalence (modulo task_id) between `allocator.allocate` and a direct `run_contract_net` call against the same snapshot, which pins the bus + cache + projection layer; (c) last-write-wins cache invariant under random publish streams. Strategies mirror `test_negotiation.py`: full snapshot of `_healthy_float = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)`, `task_type` from `TASK_TYPES`, `threshold` from the same float strategy. Hypothesis settings use `max_examples=200` and `suppress_health_check=[HealthCheck.function_scoped_fixture]` because each example creates its own `LocalBus` + `DecentralizedTaskAllocator` inside the test body. The degradation walk uses `security_review` (clear second-best is `style_reviewer` at 0.55 in the default matrix) and steps full-health -> degraded specialist -> all-degraded; asserts the three regimes (specialist wins; peer takes over with bid `0.55`; everyone falls below threshold and the round escalates with the full bid ledger preserved). Header docstring of `test_task_allocator.py` extended to call out the 5.7 sweep. PLAN.md ticked 5.7 and filled in the Phase 5 Log table with the four locked-in design decisions (pure negotiation, allocator owns state, shared round-robin cursor, audit-record hook deferred to a later task). README.md updated per the project's "README update rule": Phase 5 row ticked, Phase 6 marked Next, "Currently:" line replaced with the Phase 5 close (650 tests, ~93% coverage). 4 new tests: total 650 tests pass (646 + 4), full-suite coverage 93.43%. `task_allocator.py` still 100% covered. **Test-suite flake note:** the agent-retrieval tests (`TestStyleReviewerAgent::test_review_retrieves_docs`, `TestSummarizerAgentSynthesize::test_synthesize_retrieves_templates`) failed once intermittently in the first full run of this session and passed on rerun and in isolation; both are ChromaDB/embedding-related and unrelated to the allocator. Worth flagging if they recur on CI for #9. All 4 CI checks green locally on the second run (ruff check, ruff format, mypy src/ on 55 files, pytest --cov). PR #8 (5.6) merged earlier this session (`109901e`); 5.7 PR will be opened next. **Audit-hook reminder:** `task_allocator.py` still does not write to `AllocationAuditRecord`. The persistent surface exists from 5.5; whoever wires the hook later should serialize via `[asdict(b) for b in result.all_bids]` and pull a `Session` from `app.state.session_factory`. |

> Update this table at the end of every session before closing.

---

## How to Start a Session

```
1. Read this file (CLAUDE.md). You now have full context
2. Read only the current phase section from PLAN.md
3. Pick the next unchecked [ ] task
4. Implement → write test → make test green → [x]
5. Update "Current Status" table above before ending session
```

---

## Git Workflow

```
main          protected, never commit directly
feat/*        new features (one per phase task is fine)
fix/*         bug fixes
chore/*       config, CI, docs, refactor
```

**Branch per task:**
```bash
git checkout -b feat/phase0-pyproject    # start task
# ... implement + test ...
git add <specific files>                 # never git add .
cz commit                                # conventional commit prompt
git push origin feat/phase0-pyproject
gh pr create --base main                 # open PR
# CI must be green before merge
```

**Commit format (enforced by commitizen):**
```
feat(scope): short description
fix(scorer): handle zero-variance KL-div
chore(ci): add mypy strict step
docs(readme): add architecture diagram
test(allocator): add negotiation invariant tests
```

**Never:**
- `git add .` or `git add -A`, always add specific files
- Commit directly to `main`
- Skip pre-commit hooks (`--no-verify`)
- Commit `.env` files

---

## Dev Practices (enforced every task)

| Practice | Tool | When |
|----------|------|------|
| Lint + format | `ruff` | pre-commit + CI |
| Type checking | `mypy --strict` | pre-commit + CI |
| Commit message | `commitizen` | pre-commit |
| Test coverage | `pytest --cov` | CI (min 80%) |
| Dependency lock | `uv` + `requirements.lock` | CI installs from lock |
| Type hints | manual | every new function |
| Docstrings | manual | every public function |

**Test rule:** Write the test in the same session as the implementation, before moving to the next task. `make test` green = task done.

**Phase 1 exception:** Research experiment, no unit tests. Gate = measured signal results plus GO/NO-GO decision.

**Prose style rule (project-wide):** Never write em dashes (Unicode U+2014) in any file: README, docs, PLAN.md, CLAUDE.md additions, code comments, docstrings, commit messages. Use commas, colons, parentheses, semicolons, or two sentences instead. The hyphen `-` and en dash `–` are fine.

**README update rule:** Update the "Project status" section of `README.md` at the close of every phase. Tick the phase, update the test count and coverage, and add a one-line "Currently:" note. The README, `CLAUDE.md`, and `PLAN.md` stay in sync at session end.

---

## Stack Quick-Ref

| Component | Choice | Note |
|-----------|--------|------|
| LLM (real) | Together.ai | Default. Get key at api.together.ai |
| LLM (tests/experiments) | MockBackend | Zero cost, deterministic |
| LLM (optional) | Ollama | GPU only, skip |
| Forecaster | Chronos-2-Small + BOCPD | CPU-safe. Async. |
| Agents | LangGraph + LangChain | |
| Memory | ChromaDB | |
| API | FastAPI | |
| Messaging | Redis pub/sub | |
| DB | SQLite (dev) / PostgreSQL (prod) | |
| Dashboard | FastAPI + HTML/Chart.js | No npm build step |
| Container | Docker + Compose | |

---

## Key Commands

```bash
make dev              # start FastAPI with hot reload
make test             # pytest unit + integration (MockBackend, no API calls)
make lint             # ruff + mypy
make docker-up        # full stack
cz commit             # conventional commit
cz bump               # bump version + changelog
pre-commit run --all  # run all hooks manually
chronoagent run-experiment --config configs/experiments/signal_validation.yaml --output results/
```

---

## Phase Tracker (mirror of PLAN.md, update both)

| # | Phase | Status |
|---|-------|--------|
| 0 | Bootstrap | `[x]` |
| 1 | Signal Validation (GO/NO-GO) | `[x]` |
| 2 | Core Agent Pipeline | `[x]` |
| 3 | Behavioral Monitor | `[x]` |
| 4 | Temporal Health Scorer | `[x]` |
| 5 | Decentralized Task Allocator | `[x]` |
| 6 | Memory Integrity Module | `[ ]` |
| 7 | Human Escalation Layer | `[ ]` |
| 8 | Observability Dashboard | `[ ]` |
| 9 | Production Hardening | `[ ]` |
| 10 | Research Experiment Suite | `[ ]` |
| 11 | Paper Scaffold + Reproducibility | `[ ]` |
| 12 | CI/CD + Release | `[ ]` |

---

## Pivot Status

| Pivot | Triggered? | Action Taken |
|-------|-----------|--------------|
| A (AWT=0) | **Yes** | Concurrent detection reframe. No code changes. |
| B (no signal) | No | KL-div confirmed. 3/6 signals MockBackend constants. |
| C (Chronos underperforms) | No | n/a |

---

## Project Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | This file: session context, git workflow, quick-ref |
| `PLAN.md` | Full implementation plan: phases, tasks, logs, pivot protocol |
| `chrono_agent_research.md` | Research dossier: role alignments, literature gaps (for applications) |
| `docs/phase1_decision.md` | Phase 1 GO/NO-GO ruling, raw results, pivot analysis |
