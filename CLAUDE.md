# ChronoAgent: Session Context for Claude Code

> This file is auto-read at every session start. Keep it current. Full plan is in PLAN.md.

---

## Current Status

| Field | Value |
|-------|-------|
| **Current phase** | Phase 8: Observability Dashboard |
| **Next task** | 8.3 -- Prometheus metrics (optional prod path): `src/chronoagent/observability/metrics.py`, gauges/counters/histograms plus Grafana dashboard JSON. |
| **Blocked?** | No |
| **Last session** | 2026-04-11. Phase 8 task 8.2 complete (frontend). Single self-contained HTML at `src/chronoagent/dashboard/static/index.html` (~550 lines, inlined CSS + vanilla JS, Chart.js 4.4.1 from jsdelivr CDN). New `src/chronoagent/dashboard/__init__.py` exposes `STATIC_DIR` and `INDEX_HTML` path constants. **Route wiring:** added `GET /dashboard/` to the existing router in `dashboard.py` returning `FileResponse(INDEX_HTML, media_type="text/html")`, `include_in_schema=False`, 500 if the asset is missing. No `StaticFiles` mount, no pyproject changes: hatchling with `packages = ["src/chronoagent"]` bundles the HTML automatically. **Layout:** 3x2 CSS grid. Row 1: Agents (280px, left), Signal Explorer (center, flex), Allocation Log (360px, right). Row 2: Memory Inspector (spans 2 cols), Escalation Queue (right). Dark theme with accent `#5aa9ff`, good/warn/bad thresholds at 0.75/0.5. **Data flow:** WebSocket `/dashboard/ws/live` drives the header stats strip (system health %, agent count, pending escalations, quarantine count) plus the agents list every 2s, with auto-reconnect every 3s on close. REST polling fills the rest: allocations + escalations every 5s, memory every 10s, timeline every 10s (or on click / signal-select change). Initial `GET /dashboard/api/agents` fires before the WS so the panel is never blank. **Signal Explorer:** Chart.js line chart, dropdown picks one of the 6 signal fields (`total_latency_ms`, `retrieval_count`, `token_count`, `kl_divergence` (default), `tool_calls`, `memory_query_entropy`); selecting an agent in the left list rewires the fetch to `/dashboard/api/agents/{id}/timeline?limit=200`. First agent auto-selected on load. **WS merge quirk:** live frames omit `last_signal_at` for speed, so the client re-injects the last-known value from the prior render before passing to `renderAgents`. XSS-safe via a local `escapeHtml` helper used on every user-sourced string (agent_id, task_type, rationale, trigger). **Tests:** added `TestDashboardIndex` class to `tests/unit/test_dashboard_router.py` (2 cases: landmark-string check on `GET /dashboard/`, and bundled-asset presence check via the `chronoagent.dashboard` module constants). **CI (local, all 4 gates):** ruff check + format clean on `src/ tests/`, mypy --strict clean on 64 files, 889 tests pass, 94.58% coverage. 2 new tests added on top of the 887 from 8.1. PLAN.md ticked 8.2. |

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
| 6 | Memory Integrity Module | `[x]` |
| 7 | Human Escalation Layer | `[x]` |
| 8 | Observability Dashboard | `[ ]` (8.1, 8.2 done) |
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
