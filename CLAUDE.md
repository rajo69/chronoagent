# ChronoAgent: Session Context for Claude Code

> This file is auto-read at every session start. Keep it current. Full plan is in PLAN.md.

---

## Current Status

| Field | Value |
|-------|-------|
| **Current phase** | Phase 9: Production Hardening (Phase 8 closed) |
| **Next task** | 9.1 -- `api/middleware.py` rate limiting (slowapi or custom): POST 10/min, GET 60/min, WS 5 concurrent. |
| **Blocked?** | No |
| **Last session** | 2026-04-11. **Phase 8 task 8.3 completed locally (not yet merged).** Branch `feat/phase8-task8.3-prometheus-metrics` has 4 new product files and 1 new test file. New source: `src/chronoagent/observability/metrics.py` (`ChronoAgentMetrics` class with isolated `CollectorRegistry`, 8 gauges, 7 counters, 3 histograms, passive update methods plus `render()` returning the exposition payload), `src/chronoagent/observability/metrics_wiring.py` (`subscribe_metrics_to_bus`/`unsubscribe_metrics_from_bus` attach closures on `health_updates`, `escalations`, `memory.quarantine`; closures parse dict-or-dataclass payloads, tolerate malformed/non-dict payloads, drop silently with a warning), `src/chronoagent/api/routers/metrics.py` (`GET /metrics` with `include_in_schema=False`; calls `_refresh_poll_gauges(request, metrics)` to snapshot `system_health`, `pending_escalations`, `quarantine_size`, `memory_baseline_size`, `memory_pending_refit` from `app.state` before `metrics.render()`; 503 when sink missing; tolerates partial state). `src/chronoagent/main.py` wires the sink into the lifespan after the escalation handler (`app.state.metrics = ChronoAgentMetrics()` then `app.state.metrics_subscribers = subscribe_metrics_to_bus(bus, app.state.metrics)`; teardown mirrors). `create_app` includes the new `metrics_router`. **Metric taxonomy:** per-agent gauges `chronoagent_agent_{health,bocpd_score,chronos_score}` labelled by `agent_id`; aggregate gauges `chronoagent_system_health`, `chronoagent_escalation_queue_pending`, `chronoagent_memory_{quarantine_size,baseline_size,pending_refit}`; counters `chronoagent_health_updates` (label `agent_id`), `chronoagent_allocations` (labels `task_type`, `outcome` in `{assigned,escalated,fallback}`), `chronoagent_escalations` (label `trigger`), `chronoagent_quarantine_events`, `chronoagent_quarantined_docs`, `chronoagent_reviews` (label `risk`), `chronoagent_integrity_checks` (label `flagged`); histograms `chronoagent_allocation_bid_score` (label `task_type`, score buckets), `chronoagent_review_duration_seconds` (label `risk`, latency buckets), `chronoagent_integrity_aggregate_score` (score buckets). **Grafana:** `docs/grafana/chronoagent_dashboard.json`, 9 panels, templated `${DS_PROMETHEUS}` datasource, spans system health, queues, per-agent timeseries, allocation rates by task type/outcome, escalation rates by trigger, review-duration p50/p95 by risk, integrity-score p50/p95/p99. **Test file:** `tests/unit/test_metrics.py` with 38 tests covering sink isolation, each observer method, wiring fan-out, malformed-payload drops, router happy path, 503 when sink missing, partial-state tolerance, and the empty-scorer system-health default. **Key gotchas locked in:** (a) `prometheus_client` auto-appends `_total` to counter names so test assertions must query `chronoagent_foo_total`, and unset bare counters still emit a `0.0` sample (check `== 0.0`, not `is None`); unset labelled counters/histograms emit no sample (check `is None`). (b) `TestClient(create_app())` runs the real lifespan which uses the default SQLAlchemy pool; in-memory SQLite then gives each new connection a fresh empty database, so the `/metrics` handler's count query errors with "no such table: escalation_records". Fix: override `app.state.session_factory` with a `StaticPool`-backed engine after entering the `TestClient` context (same pattern as `test_dashboard_router.client_app`). (c) `TemporalHealthScorer` subscribes to `SIGNAL_CHANNEL`/`signal_updates` and publishes to `HEALTH_CHANNEL`/`health_updates`; publishing on the health channel does NOT populate the scorer's internal `_health_cache`. The metrics sink's per-agent gauges ARE updated via `metrics_wiring` subscribing on the health channel (direct sink update, not scorer cache), but `system_health` (poll-derived from the scorer cache) stays at its 1.0 default until something publishes on `signal_updates`. **Test + CI status:** 927 tests pass (889 pre-8.3 + 38 new), 94.81% coverage, ruff check + format --check clean on `src/ tests/`, mypy --strict clean on 67 files. Flake: `TestQuarantine` tripped once during the first coverage run with an unrelated "test_reason_recorded_when_provided" AssertionError but did NOT reproduce on the immediate rerun or with `--no-cov`; passed in isolation both times; not introduced by task 8.3. PLAN.md Phase 8 Log filled in (Findings / Challenges / Decisions / Completed 2026-04-11) and task 8.3 ticked. CLAUDE.md phase tracker flipped Phase 8 to `[x]`. README.md Phase 8 row + Currently line + test count/coverage still need the close-of-phase tick. Branch not yet pushed; no PR open. **Previous session carry-over:** Phase 8 tasks 8.1 + 8.2 merged to `main`. 8.1 (backend) landed as PR #18 squash commit `b1117a2`; 8.2 (frontend) landed as PR #20 squash commit `9071638`. Only 8.3 (Prometheus metrics + Grafana JSON) remains before the phase closes. Fresh branch `feat/phase8-task8.3-prometheus-metrics` already cut from main and checked out, ready for the next session. **PR workflow gotcha (lock this in):** stacked PRs against the original CLI flow *do not* auto-retarget when the base branch is deleted via `gh pr merge --delete-branch`. GitHub auto-closes them instead and refuses reopen because the base branch no longer exists. The first 8.2 PR was #19, stacked on `feat/phase8-task8.1-dashboard-backend`; when #18 was merged with `--delete-branch`, #19 was force-closed. Fix was to force-push the rebased 8.2 branch (dropped the two 8.1 commits via `git rebase --onto main ba00706`) and open a fresh PR #20 against main with the same description. Lesson: either do NOT pass `--delete-branch` on the base merge until stacked children are retargeted, or don't stack in the first place when the base is about to merge immediately. **What 8.2 shipped (now on main):** Single self-contained HTML at `src/chronoagent/dashboard/static/index.html` (~550 lines, inlined CSS + vanilla JS, Chart.js 4.4.1 from jsdelivr CDN, no build step, no npm). New `src/chronoagent/dashboard/__init__.py` exposes `STATIC_DIR` and `INDEX_HTML` path constants. `GET /dashboard/` added to `src/chronoagent/api/routers/dashboard.py` returning `FileResponse(INDEX_HTML, media_type="text/html")`, `include_in_schema=False`, 500 if the asset is missing. No `StaticFiles` mount; hatchling with `packages = ["src/chronoagent"]` bundles the HTML automatically, no pyproject changes needed. Layout is a 3x2 CSS grid (Agents left, Signal Explorer center, Allocation Log right; Memory Inspector spans 2 cols across the bottom, Escalation Queue bottom right). Dark theme with accent `#5aa9ff`, health thresholds at 0.75 / 0.5. WebSocket `/dashboard/ws/live` drives the header strip + agents list every 2s with 3s auto-reconnect; REST polls fill the rest (allocations + escalations 5s, memory 10s, timeline 10s or on click). Live frames omit `last_signal_at` for speed, so the client re-injects the last-known value from the prior render. XSS-safe via a local `escapeHtml` helper. `TestDashboardIndex` class added to `tests/unit/test_dashboard_router.py` (landmark-string + bundled-asset presence checks). **Test + CI status at merge:** 889 tests pass, 94.58% coverage, ruff check + format --check clean on `src/ tests/`, mypy --strict clean on 64 files, both PRs green on 3.11 + 3.12. PLAN.md ticked 8.2. CLAUDE.md phase tracker still shows `[ ]` for Phase 8 overall (8.3 outstanding) but 8.1 + 8.2 are done. README "Currently:" updated mid-phase; full phase-close README tick comes after 8.3. |

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
| 8 | Observability Dashboard | `[x]` |
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
