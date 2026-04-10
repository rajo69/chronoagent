# ChronoAgent: Session Context for Claude Code

> This file is auto-read at every session start. Keep it current. Full plan is in PLAN.md.

---

## Current Status

| Field | Value |
|-------|-------|
| **Current phase** | Phase 6: Memory Integrity Module |
| **Next task** | 6.2, replace the centroid-distance baseline inside `MemoryIntegrityModule.fit_baseline` with a `sklearn.IsolationForest` fitted on clean embeddings; periodic refit every N new docs. Same `fit_baseline(embeddings)` entry point; the embedding-outlier signal in `_embedding_outlier_score` is the only consumer that needs to change. |
| **Blocked?** | No |
| **Last session** | 2026-04-10. Phase 6 task 6.1 done [x]. Branch `feat/phase6-task6.1-memory-integrity-module` cut off `main` (which now includes PR #9 from the 5.7 session, commit `a1ae14c`). New module `src/chronoagent/memory/integrity.py` (195 LOC) with `RetrievedDoc`, `DocSignal`, `IntegrityResult`, `MemoryIntegrityModule`. Four orthogonal signals: (1) **embedding outlier** -- cosine distance to a centroid baseline fitted via `fit_baseline(embeddings)`; saturates at 3x the in-sample 99th-percentile distance, radius floored at 1e-3; returns 0.0 if no baseline is fitted, 1.0 for zero-norm input. (2) **freshness anomaly** -- linear ramp over the leading 10% of `freshness_window_seconds` (default 30 days * 0.1 = 3 days); future-dated >60s saturates; missing or unparseable `created_at` metadata is silent. (3) **retrieval frequency** -- z-score of the doc's lifetime count against population mean+std across `_retrieval_counts`, mapped to [0,1] with `retrieval_spike_z` (default 3.0) as the saturation point; silent if <2 distinct docs, 0 variance, or count==0. (4) **content-embedding mismatch** -- re-embed `doc.text` via the injected backend in a single batched call, then `(1 - cos_sim) / 2`; this is the headline detector for MINJA / AgentPoison style attacks where the attacker injects an embedding that does not match the doc text. Aggregation: L1-normalised weighted sum (defaults `embedding_outlier=0.25 / freshness_anomaly=0.15 / retrieval_frequency=0.20 / content_embedding_mismatch=0.40`), flagged when `aggregate >= flag_threshold` (default 0.6). Module is self-contained: owns the baseline centroid+radius, owns a bounded `Counter[str]` retrieval history (LRU evicts least-frequent ids when `retrieval_history_max` exceeded), and does NOT touch ChromaDB or `MemoryStore`. The caller assembles `RetrievedDoc(doc_id, text, embedding, distance_to_query=0.0, metadata={})` records and passes them to `check_retrieval(query, docs) -> IntegrityResult`; this keeps the detection logic trivially testable and decoupled from the store. **Critical ordering inside `check_retrieval`:** retrieval-history bookkeeping happens *after* scoring, so a doc that just surfaced does not count against itself in the same call. `now_fn` is injectable for deterministic freshness tests. Constructor validates `flag_threshold in [0,1]`, weight key set, weight signs, and positivity of all numeric args. 41 unit tests in `tests/unit/test_memory_integrity.py` (TestConstruction, TestEmpty, TestContentEmbeddingMismatch, TestEmbeddingOutlier, TestFreshnessAnomaly, TestRetrievalFrequency, TestAggregation, TestEdgeCases, TestIntegrityResultContainer) pin every signal in isolation, the aggregation, the weight + threshold validation, the retrieval-history ordering, plus an explicit MINJA-centroid-injection scenario where the doc embedding equals the centroid of three target queries but the text is benign. **Important test gotcha:** MockBackend's hash embeddings are pseudo-random and nearly orthogonal, so text-based "tight cluster" baselines do NOT actually cluster in cosine space -- the outlier test builds a synthetic 384-d baseline anchored on `(1, 0, 0, ...)` instead. `integrity.py` 100% covered (195/195). PLAN.md ticked 6.1; Phase 6 Log left empty (filled at phase close). Total 691 tests pass (650 + 41), full-suite coverage 93.96%. Three flaky agent-retrieval tests (`test_agents.py::TestSecurityReviewerAgent::test_review_retrieves_docs`, `test_retrieve_memory_returns_result`, `TestPlannerAgent::test_decompose_retrieves_docs`) failed once on the full run and passed in isolation -- same ChromaDB/embedding flake the 5.7 session flagged, unrelated to integrity. All four CI checks green locally (ruff check, ruff format on 56 files, mypy src/ on 56 files, pytest --cov). Initial ruff hit a SIM108 ternary refactor on `fit_baseline` and ruff format reformatted both new files; both fixed before commit. Commit `c2b318b`; PR not yet opened. **Audit-hook reminder still applies:** `task_allocator.py` does not write to `AllocationAuditRecord`. **README update reminder:** Phase 6 is mid-phase, README "Currently:" line not yet touched -- update at phase close per the README rule. |

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
