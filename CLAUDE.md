# ChronoAgent: Session Context for Claude Code

> This file is auto-read at every session start. Keep it current. Full plan is in PLAN.md.

---

## Current Status

| Field | Value |
|-------|-------|
| **Current phase** | Phase 11 (Paper Scaffold + Reproducibility) OPEN. Tasks 11.1-11.3 merged (PRs #39-#42). Task 11.4 shipped locally on branch `feat/phase11-task11.4-reproduce-makefile`, PR to open against `main`. |
| **Next task** | **Phase 11 task 11.5 (Docker pin + seed freeze).** After 11.4 merges, cut `feat/phase11-task11.5-docker-pin` FRESH from `main`. **11.5 scope:** Pin all Python dependency versions in the Docker image; verify all experiment seeds are fixed and configs are checked into the repo; add a smoke-test CI step that runs `make reproduce-signal` inside the Docker image. |
| **Blocked?** | No |
| **Last session** | 2026-04-12. **Phase 11 task 11.4 shipped on branch `feat/phase11-task11.4-reproduce-makefile`, PR to open against `main`.** Local pre-push run: ruff check clean, ruff format --check clean (136 files already formatted), mypy --strict clean (80 source files, zero issues), pytest 1507 passed 95.94% coverage. **What 11.4 shipped:** (a) **Latency microbenchmark wired end-to-end.** `RunResult` gained a `latency_ms: float` field (mean per-step wall-clock detector latency, measured via `time.perf_counter()` around `detector.run(signal_matrix)` in `_run_single`). `AggregateResult` gained `latency_ms: MetricAggregate`. Both `runs.csv` and `aggregate.json` now persist the new field. The `_METRIC_COLUMNS` tuple in `tables.py` gained a 5th entry `("latency_ms", "Latency (ms/step)", 2)`, so both the main results and ablation tables now render a latency column automatically. (b) **Dedicated `make_latency_table()`** produces `results/tables/table4_latency.tex`, a focused per-detector latency comparison (2-column: Configuration, Latency). Bolding is inverted (lower is better) unlike the other metrics. Wired into `generate_all_tables()` and the `compare-experiments` CLI command. (c) **Paper C5 section flipped.** The `\todo{replace this paragraph...}` in `06_results.tex` is replaced by a prose paragraph referencing `\cref{tab:latency}` plus a `\begin{table}[tbp]...\inputiffound{../results/tables/table4_latency.tex}...\end{table}` float. (d) **Makefile reproducibility targets.** Five new targets: `reproduce-signal` (Phase 1), `reproduce-main` (main + agentpoison + sentinel), `reproduce-ablations` (three ablation configs), `reproduce-figures` (the `compare-experiments` pass), and `reproduce` (umbrella). Each mirrors the protocol in `05_experiments.tex`. (e) **7 new tests** in `test_experiment_tables.py` for `TestMakeLatencyTable`: path defaults, override, structure, bolding (lowest is best), empty list rejection, missing experiment, display names, positive values. Existing tests updated for the new column (`lcccc` -> `lccccc`, metric keys set +1, `generate_all_tables` returns 3 or 4 paths instead of 2 or 3). CLI test patches extended to include `make_latency_table`. **Files changed (3 src + 3 test + 3 doc + 1 paper + 1 Makefile):** `src/chronoagent/experiments/experiment_runner.py` (import time, latency_ms on RunResult/AggregateResult, timing in _run_single, aggregation, CSV + JSON persistence), `src/chronoagent/experiments/analysis/tables.py` (LATENCY_TABLE_STEM, _LATENCY_MS_DECIMALS, latency in _METRIC_COLUMNS, make_latency_table(), generate_all_tables updated), `src/chronoagent/cli.py` (import + call make_latency_table in compare-experiments); `tests/unit/test_experiment_tables.py` (TestMakeLatencyTable + existing test updates), `tests/unit/test_experiment_runner.py` (latency_ms in RunResult/AggregateResult constructors, metric keys set), `tests/unit/test_cli.py` (latency_ms in stub, make_latency_table patches); `paper/sections/06_results.tex` (C5 \todo -> \inputiffound + prose); `Makefile` (5 reproduce targets); PLAN.md (11.4 ticked + Completed), CLAUDE.md (this table), README.md (Currently bumped to 11.4). |

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
| 9 | Production Hardening | `[x]` |
| 10 | Research Experiment Suite | `[x]` |
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
