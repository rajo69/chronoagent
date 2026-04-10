# ChronoAgent: Session Context for Claude Code

> This file is auto-read at every session start. Keep it current. Full plan is in PLAN.md.

---

## Current Status

| Field | Value |
|-------|-------|
| **Current phase** | Phase 7: Human Escalation Layer |
| **Next task** | Phase 7 complete -- all 4 tasks done [x]. Next: Phase 8 (Observability Dashboard), task 8.1. |
| **Blocked?** | No |
| **Last session** | 2026-04-11. Phase 7 complete: all 4 tasks done [x], PR #17 open. 845 tests pass, 94.29% coverage. PLAN.md Phase 7 log filled in. Branch `feat/phase7-task7.1-escalation-layer`. Implemented on `feat/phase6-task6.3-quarantine-store`, cut off the post-6.2 `main`. New `memory/quarantine.py` adds `QuarantineStore`, a thin wrapper around a *separate* ChromaDB collection that holds documents flagged by `MemoryIntegrityModule`. The active `MemoryStore` and the quarantine store live in two distinct collections; retrieval only ever queries the active store, so flagged docs are excluded from agent context until a human approves them. **API:** `count`, `list_ids() -> list[str]`, `get_doc(doc_id) -> StoredDoc | None`, `quarantine(active_store, ids, *, reason=None) -> list[str]`, `approve(active_store, ids) -> list[str]`. Both move methods are idempotent at the ID level: quarantining an already-quarantined or non-existent ID is a no-op, and approving an ID not currently in quarantine is a no-op. Repeated IDs in a single call are deduplicated while preserving order so the caller can replay an `IntegrityResult.flagged_ids` array verbatim. Quarantine adds `quarantined_at` (Unix epoch from injectable `now_fn`, defaults to `time.time`) and an optional `quarantine_reason` string to each moved record's metadata. Approve strips both bookkeeping keys so the restored record looks identical to its pre-quarantine state, then re-inserts via `MemoryStore.add` using the *original* embedding so the cosine vector space is unperturbed by the round trip. **MemoryStore additions to support 6.3:** new `StoredDoc` dataclass (doc_id/text/embedding/metadata) and new `MemoryStore.get_by_ids(ids)` method that fetches full records by ID (text + embedding + metadata) and silently drops missing IDs. Used by `QuarantineStore.quarantine` to pull records out of the active store. **Critical ChromaDB gotcha:** `collection.upsert` rejects empty metadata dicts with `Expected metadata to be a non-empty dict`. The end-to-end test exposed this: a doc added without metadata, quarantined (gains `quarantined_at` + `quarantine_reason`), then approved (both stripped) leaves an empty dict that fails the upsert. Fix: `approve` splits the restore into a "has surviving metadata" batch and a "no metadata after stripping" batch; the latter is added with `metadatas=None` so `MemoryStore.add` omits the field entirely. **Critical mypy gotcha:** ChromaDB's stub for `Collection.upsert` only accepts ndarray-flavoured embeddings and a narrow `Mapping` for metadatas, even though the runtime accepts plain `list[list[float]]` and `list[dict[str, Any]]`. Mirror MemoryStore.add's existing escape hatch: route through an `Any`-typed kwargs dict (`upsert_kwargs: dict[str, Any] = {...}; self._collection.upsert(**upsert_kwargs)`). **Coverage strategy:** trust ChromaDB's `include=[...]` contract on read paths (it always returns the requested fields when at least one ID matches) so the four-way `if raw_embeddings is not None and i < len(...)` defensive ladders become single `assert embeddings_field is not None` lines. Same simplification applied to `MemoryStore.get_by_ids`. Per-row metadata can still legitimately be `None` (when a doc was inserted without a metadata dict) so that one stays guarded. `quarantine.py` 105/105 lines (100%); `store.py` 81/82 (99%, sole missing line is the pre-existing `get_all_embeddings` empty-stored defensive path, untouched by this PR). **End-to-end test** (`TestIntegrityFlowEndToEnd.test_flagged_id_can_be_quarantined_and_restored`) wires the full chain: seed clean corpus + one MINJA-style poison doc (stored embedding from text A, visible text from unrelated text B), run `MemoryIntegrityModule.check_retrieval` with weights pinned to `content_embedding_mismatch=1.0` (the noisy embedding-outlier signal cannot be relied on without a fitted baseline, so the 4-signal default `flag_threshold=0.6` is unreachable from a single mismatch), assert `poison_1` lands in `flagged_ids`, hand the array straight to `quarantine_store.quarantine(active_store, result.flagged_ids, reason="integrity_module")`, assert the active query no longer surfaces it, then approve and assert the embedding round-trips byte-equal and no quarantine metadata leaks. **Test totals:** 28 new tests in `tests/unit/test_memory_quarantine.py` (`TestGetByIds` x5, `TestQuarantineIntrospection` x4, `TestQuarantine` x12, `TestApprove` x6, `TestIntegrityFlowEndToEnd` x1). 736 tests pass total (708 + 28), full-suite coverage 94.36%. All four CI checks green locally (ruff check, ruff format on 85 files, mypy src/ on 57 files, pytest --cov). **No flake on this run.** PLAN.md ticked 6.3. **README update reminder:** Phase 6 still mid-phase (6.4 + 6.5 outstanding); README "Currently:" line not yet touched, update at phase close per the README rule. **Audit-hook reminder still applies:** `task_allocator.py` does not write to `AllocationAuditRecord`. **Background context (6.1 + 6.2, both merged to `main` -- PR #10 -> `89d02e4`, PR #11 -> `d343151`):** 6.1 introduced `MemoryIntegrityModule` with 4 detection signals (embedding outlier, freshness anomaly, retrieval frequency spike, content-embedding mismatch). 6.2 replaced the centroid+radius outlier baseline with `sklearn.ensemble.IsolationForest` (already a core dep, never previously imported), added `record_new_docs(embeddings)` auto-refit hook gated on `refit_interval`, and added observability properties `baseline_fitted`, `pending_refit_count`, `baseline_size`. The IsolationForest needs `# type: ignore[import-untyped]` since scikit-learn ships no `py.typed` marker. |

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
