# ChronoAgent: Session Context for Claude Code

> This file is auto-read at every session start. Keep it current. Full plan is in PLAN.md.

---

## Current Status

| Field | Value |
|-------|-------|
| **Current phase** | Phase 10 CLOSED end-to-end; next is Phase 11 (Paper Scaffold + Reproducibility). |
| **Next task** | **Phase 11 task 11.1 (first unchecked task in PLAN.md Phase 11 section): paper scaffold.** Task 10.9 is shipped locally on branch `feat/phase10-task10.9-cli-run-experiment` (pending commit at session end). Push and open a PR against `main` mirroring the 10.8 PR #37 format (cite 1500 passed / 95.98% coverage / ruff + ruff format --check clean on 130 files / mypy --strict --no-incremental clean on 80 src files; 1 edited src file (`src/chronoagent/cli.py`) + 1 edited test file (`tests/unit/test_cli.py`, 11 -> 28 tests)). After CI goes green on 3.11 + 3.12, merge with `gh pr merge --squash --delete-branch`, sync local `main`, THEN cut `feat/phase11-task11.X-...` from the updated main. **Phase 10 is now CLOSED end-to-end** -- all nine tasks (10.1 schema, 10.2 metrics, 10.3 sentinel, 10.4 no-monitoring, 10.5 six YAMLs, 10.6 runner + FullSystemDetector, 10.7 plots, 10.8 tables, 10.9 CLI) merged or in review. Next session opens Phase 11 (Paper Scaffold + Reproducibility): see the PLAN.md Phase 11 section for the task list. Typical first task is the LaTeX skeleton under `paper/` with a cross-linked `results/` directory, followed by a reproducibility checklist and the figure/table hookup to the 10.7 / 10.8 output files. |
| **Blocked?** | No |
| **Last session** | 2026-04-12. **Phase 10 task 10.9 shipped on branch `feat/phase10-task10.9-cli-run-experiment`, PR to open against `main`.** Local pre-push run for 10.9: ruff check + ruff format --check clean on 130 files, mypy --strict --no-incremental clean on 80 source files, pytest 1500 passed (1483 prior + 17 net-new CLI tests; `tests/unit/test_cli.py` grew from 11 tests to 28) 95.98% coverage, zero flakes this run. **What 10.9 shipped:** Restructured `src/chronoagent/cli.py`. Converted `chronoagent run-experiment` into a Typer sub-app with two subcommands: `phase1` (the existing `SignalValidationRunner` entry point, unchanged body apart from the command name and a handful of ASCII-only echo tweaks) and `phase10` (the new single-experiment Phase 10 driver). The `phase10` body loads an `ExperimentConfig` via `ExperimentConfig.from_yaml`, constructs `ExperimentRunner(cfg, collect_raw=True)`, calls `runner.run()` and `write_experiment_results(agg, output, raw_runs=runner.raw_runs)`, then optionally renders the single-experiment signal-drift figure via `plot_signal_drift` and the headline LaTeX main results table via `make_main_results_table`. Flags on `phase10`: `--plots/--no-plots` (default on), `--tables/--no-tables` (default on), `--quiet` (default off). Config-load failures (`FileNotFoundError`, pydantic `ValidationError`, generic `ValueError`) surface as exit 1 with an explicit ERROR line on stderr; plot/table failures log stderr WARNINGs but leave exit 0 since `runs.csv` + `aggregate.json` + `raw/` are already persisted by the time rendering starts and should not be invalidated by a rendering glitch. **New top-level `chronoagent compare-experiments` command:** takes `--output <results-dir>`, one or more `--experiment <name>`, optional `--ablation <name>` (repeatable), optional `--full-system <name>` (default: first `--experiment`), optional `--drift-experiment <name>` (default: first `--experiment`), plus its own `--plots/--no-plots` / `--tables/--no-tables` / `--quiet` switches. Dispatches to `generate_all_plots(output, experiments, drift_experiment=...)`, `make_main_results_table(output, experiments)`, and `make_ablation_table(output, full_system, ablations)`. Ablation table is skipped with an operator-visible note when no `--ablation` values are supplied (avoids the `ValueError` from `make_ablation_table` on an empty list). `compare-experiments` is stricter than `phase10`: plot/table generator failures return exit 1 because rendering is its sole job. **Key design decisions locked in:** (a) **Subcommand group, not `--phase` flag.** Keeps each command's arg surface honest: `phase1` takes the legacy `runner:`/`analysis:` nested YAML, `phase10` takes an `ExperimentConfig`. A shared `--config` would force one command body to dispatch on YAML shape and silently misinterpret typos. (b) **`run-experiment phase10` only renders per-experiment artefacts.** `plot_signal_drift` and `make_main_results_table` both work on a one-entry list; the other five figures and the ablation table would render misleading one-point comparisons, so they are deliberately skipped with an operator-visible note pointing at `compare-experiments`. Narrower than CLAUDE.md's `generate_all_plots(output, [cfg.name])` wording but correctness-preserving. (c) **`compare-experiments` is a separate top-level command, not a sub-option.** Matches the typical research workflow ("run every experiment first, then compare") and keeps each command's responsibility narrow. (d) **Deferred imports inside each command body.** Preserves `chronoagent serve` startup time by keeping Phase 10 imports out of the import graph unless the relevant subcommand runs. This means tests that want to stub `ExperimentRunner` / `plot_signal_drift` / `make_main_results_table` / `generate_all_plots` / `make_ablation_table` have to patch at the origin namespace (`chronoagent.experiments.experiment_runner.ExperimentRunner`, etc.), not at `chronoagent.cli.*`; the `from ... import` re-execution inside each call picks up the origin-side patch cleanly. (e) **Plot/table failures in `run-experiment phase10` log WARNINGs but do not fail the run.** `runs.csv` + `aggregate.json` + `raw/` have already been persisted by the time rendering starts. **Key gotchas locked in:** (1) **Typer sub-app wiring requires `app.add_typer(sub, name="run-experiment")`** to register the top-level command name. (2) **Patching deferred imports must target the origin namespace.** First test attempt patched `chronoagent.cli.ExperimentRunner` and got AttributeError because the class never enters the `chronoagent.cli` namespace (the `from ... import ExperimentRunner` sits inside the function body and re-executes each call). Fix: patch at `chronoagent.experiments.experiment_runner.ExperimentRunner`, and similarly at `chronoagent.experiments.analysis.plots.plot_signal_drift`, `chronoagent.experiments.analysis.tables.make_main_results_table`, `chronoagent.experiments.analysis.plots.generate_all_plots`, `chronoagent.experiments.analysis.tables.make_ablation_table`. (3) **`make_ablation_table` raises `ValueError` on an empty `ablation_names` list.** `compare-experiments` checks the list length and prints `Skipping ablation table (no --ablation values supplied).` before calling the generator. **Test file:** `tests/unit/test_cli.py` grew from 11 to 28 tests (+17). `TestHelp` x7 (root help, `serve --help`, `run-experiment --help` lists both subcommands, `phase1 --help`, `phase10 --help` documents `--plots` / `--tables` / `--quiet`, `compare-experiments --help` documents `--experiment`, `check-health --help`). `TestRunExperimentPhase1` x4 (valid config exits 0, echoes config path, echoes output dir, rejects unknown attack type). `TestRunExperimentPhase10` x8 (happy path writes `runs.csv` + `aggregate.json` via `_StubRunner`, `--no-plots` + `--no-tables` short-circuit, `--plots` calls `plot_signal_drift` with `run_index=0` and experiment name, `--tables` calls `make_main_results_table` with a single-entry list, `--quiet` suppresses progress echoes, missing config file exits 1 with `not found`, invalid config (negative seed) exits 1 with `invalid experiment config`, plot failure is logged as WARNING with exit 0). `TestCompareExperiments` x5 (missing output directory exits 1, happy path calls `generate_all_plots` + `make_main_results_table` + `make_ablation_table` with defaults threaded through correctly, skips ablation table when no `--ablation` flags, `--no-plots` + `--no-tables` short-circuit both generators, explicit `--full-system` and `--drift-experiment` override defaults). `TestCheckHealth` x2 and `TestServe` x2 (unchanged). **Test helpers:** `_phase10_yaml(name)` returns a validator-clean YAML for the Phase 10 schema; `_stub_aggregate(name, num_runs)` builds a deterministic `AggregateResult`; `_StubRunner` is a drop-in `ExperimentRunner` replacement that never touches ChromaDB and, when `collect_raw=True`, emits a `(num_prs, 6)` matrix with columns 3 (kl_divergence) and 5 (memory_query_entropy) bumped after `injection_step` plus a decision stream flagging every post-injection step so the persisted raw files are valid plot inputs. **Files changed (1 edited src + 1 edited test + 3 doc):** `src/chronoagent/cli.py` (restructured from ~136 lines to ~380 lines), `tests/unit/test_cli.py` (11 -> 28 tests), PLAN.md (10.9 Findings/Challenges/Completed added, 10.9 ticked, Phase 10 row flipped to `[x]`), CLAUDE.md (this table, Phase 10 tracker flipped to `[x]`), README.md (Phase 10 row ticked, badges + Currently line bumped). No new dependencies; `typer` and `pydantic` are already pinned. Phase 10 is now CLOSED end-to-end -- all nine tasks merged or in review. |

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
