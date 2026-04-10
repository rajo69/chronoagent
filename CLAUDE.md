# ChronoAgent: Session Context for Claude Code

> This file is auto-read at every session start. Keep it current. Full plan is in PLAN.md.

---

## Current Status

| Field | Value |
|-------|-------|
| **Current phase** | Phase 5: Decentralized Task Allocator |
| **Next task** | 5.6, fallback in `task_allocator.py`: negotiation timeout/error -> round-robin with warning log |
| **Blocked?** | No |
| **Last session** | 2026-04-10. Phase 5 task 5.5 done [x]. Branch `feat/phase5-task5.5-allocation-audit-record` cut off `main`. New `AllocationAuditRecord` ORM model in `src/chronoagent/db/models.py` (table `allocation_audit_records`): columns `id` PK, `task_id` (str, indexed), `task_type` (str), `assigned_agent` (str, nullable for escalation rows), `escalated` (bool), `all_bids` (JSON list of `{agent_id, capability, health, score}` dicts in canonical AGENT_IDS order), `health_snapshot` (JSON `{agent_id: health}`), `rationale` (str up to 1024), `threshold` (float), `timestamp` (UTC datetime). Composite index `ix_aar_task_ts` on `(task_id, timestamp)` for the "full audit trail for one task" query pattern. JSON columns use SQLAlchemy generic `JSON` type (maps to SQLite TEXT, PostgreSQL JSONB). New Alembic migration `alembic/versions/002_create_allocation_audit_records.py` (revision `002`, down_revision `001`) with full upgrade/downgrade pair. 10 new tests in `tests/unit/test_allocation_audit_record.py`: model construction (repr, nullable assigned_agent), ORM round-trip via in-memory SQLite (insert/query, JSON column round-trip preserving AGENT_IDS order, escalated row with NULL agent, UTC timestamp round-trip, query by task_id, composite index visible via Inspector), and Alembic migration upgrade/downgrade against tmp file-backed SQLite. Migration tests monkeypatch `chronoagent.config.load_settings` because `load_settings` reads `configs/base.yaml` and passes init kwargs to `Settings(...)` which outrank env vars in pydantic-settings, so `monkeypatch.setenv("CHRONO_DATABASE_URL", ...)` alone does not redirect the migration. `db/models.py` is 100% covered (36/36). Total 639 tests pass (629 + 10), full-suite coverage 93.40%. No allocator/pipeline code touched and no `task_allocator.py` audit hook yet (deferred to a later task; the model is just the persistent surface). All 4 CI checks green locally (ruff check, ruff format, mypy src/ on 55 files, pytest --cov). |

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
