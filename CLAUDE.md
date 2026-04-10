# ChronoAgent: Session Context for Claude Code

> This file is auto-read at every session start. Keep it current. Full plan is in PLAN.md.

---

## Current Status

| Field | Value |
|-------|-------|
| **Current phase** | Phase 5: Decentralized Task Allocator |
| **Next task** | 5.7, allocator test sweep: mock-health bid invariants, all-low-health escalation, timeout handling, Hypothesis "exactly one agent assigned or escalated" property test |
| **Blocked?** | No |
| **Last session** | 2026-04-10. Phase 5 task 5.6 done [x]. Branch `feat/phase5-task5.6-roundrobin-fallback` (already cut off `main` at session start). PLAN.md back-ticked 5.3 and 5.4 which were merged via PRs #5/#6 but never marked. Round-robin fallback added to `DecentralizedTaskAllocator.allocate` in `src/chronoagent/allocator/task_allocator.py`: wraps the call to `run_contract_net` in a `try/except Exception`, logs a `WARNING` ("task_allocator: negotiation failed for task_id=... task_type=... (<ExcType>: <msg>); falling back to round-robin"), and delegates to a new private `_round_robin_fallback(task_id, task_type, exc)` helper. Fallback advances a new `self._round_robin_cursor` (init 0) under `self._lock` and picks `AGENT_IDS[cursor % len(AGENT_IDS)]`, returning a synthesized `NegotiationResult(assigned_agent=<pick>, escalated=False, winning_bid=None, all_bids=(), rationale="round-robin fallback after negotiation error (<ExcType>: <msg>): assigned to <agent>", threshold=self._threshold)`. The cursor is shared across task types: deterministic given canonical `AGENT_IDS` order, simpler than per-type cursors. Pipeline routing already tolerates `winning_bid is None` (graph.py:243) and routes non-specialist picks through the existing escalation-placeholder branch, so no graph.py change needed. 7 new tests in `TestRoundRobinFallback` in `tests/unit/test_task_allocator.py`: generic `RuntimeError` triggers fallback + WARNING captured + result fields valid; `TimeoutError` (the "negotiation timeout" path called out in PLAN, even though `run_contract_net` is currently sync) triggers fallback identically; cursor advances across calls (two full passes through `AGENT_IDS`); task_id/task_type echoed faithfully; happy path (no monkeypatch) leaves the cursor at 0; concurrent fallback across 4 threads x 25 rounds yields exactly even distribution per agent (cursor atomic under lock); `InvalidHealthError` (negotiation's own ValueError subclass) is caught too. Tests use `monkeypatch.setattr(ta_mod, "run_contract_net", ...)` to swap the symbol on the `chronoagent.allocator.task_allocator` module, since `allocate` looks it up by name there. `task_allocator.py` 100% covered (73/73). Total 646 tests pass (639 + 7), full-suite coverage 93.43%. All 4 CI checks green locally (ruff check, ruff format, mypy src/ on 55 files, pytest --cov). Note: still no `task_allocator.py` audit-record hook (still deferred). |

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
| 5 | Decentralized Task Allocator | `[ ]` |
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
