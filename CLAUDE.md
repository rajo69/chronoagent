# ChronoAgent: Session Context for Claude Code

> This file is auto-read at every session start. Keep it current. Full plan is in PLAN.md.

---

## Current Status

| Field | Value |
|-------|-------|
| **Current phase** | Phase 11 (Paper Scaffold + Reproducibility) OPEN. Task 11.1 merged via PR #39 (`b3b86f9`) + docs cleanup PR #40 (`47241a0`). Task 11.2 merged via PR #41 (`fb8a6fc`). Task 11.3 shipped locally on branch `feat/phase11-task11.3-bibliography`, PR to open against `main`. |
| **Next task** | **Phase 11 task 11.4 (Makefile `reproduce*` targets + latency harness).** After 11.3 merges, cut `feat/phase11-task11.4-reproduce-makefile` FRESH from `main`. **11.4 scope:** (a) Add top-level Makefile targets `reproduce`, `reproduce-signal`, `reproduce-main`, `reproduce-ablations`, `reproduce-figures` so a cold-checkout reader can regenerate the full results set with one command. `reproduce` is the umbrella; the others are the per-config subsets. Each target should shell out to `chronoagent run-experiment phase10 --config configs/experiments/<name>.yaml --output results/` for the runs, then `chronoagent compare-experiments ...` for the cross-experiment tables/figures, using the exact flag set the `05_experiments.tex` protocol block documents. (b) Wire the C5 latency microbenchmark that 11.2 downgraded and 11.3 did not touch: add a `latency_ms` (float) column to `RunResult` + `MetricAggregate`, persist it in `runs.csv` / `aggregate.json`, add a column to `make_main_results_table` (currently exactly AWT, AUROC, F1, alloc-eff per `_METRIC_COLUMNS` in `tables.py`), and either extend `plot_signal_drift` with a latency panel or add a new `plot_latency_bar` function that writes `results/tables/table4_latency.tex`. Then flip the `\todo` at the end of the C5 subsection in `paper/sections/06_results.tex` to a `\inputiffound{../results/tables/table4_latency.tex}` drop. Whichever plot/table function is chosen must get a unit test under `tests/unit/experiments/` since Phase 10 is fully-tested code, unlike the paper-only deltas of 11.1-11.3. (c) The "Sentinel-style baseline" citation placeholder left in the 11.3 Challenges block (cited to LlamaGuard + NeMo Guardrails for now, not to a specific Sentinel paper) is NOT 11.4 scope unless a specific published "Sentinel" paper is chosen during the writing pass; otherwise defer to a later writing task. Mirror the 10.9/11.1/11.2/11.3 PR format (1500 passed baseline, ruff/ruff format/mypy clean, list files changed). Expect the test count to climb above 1500 since 11.4 adds new source code and tests -- the first Phase 11 task to do so. Pre-existing Chroma test-pollution flake in `tests/unit/test_memory_router.py::TestQuarantineDocs::test_quarantine_without_reason` may re-trip; not a blocker. |
| **Blocked?** | No |
| **Last session** | 2026-04-12. **Phase 11 task 11.3 shipped on branch `feat/phase11-task11.3-bibliography`, PR to open against `main`.** Local pre-push run: ruff check clean, ruff format --check clean (136 files already formatted), mypy --strict clean (80 source files, zero issues), pytest 1500 passed 95.98% coverage (unchanged from 11.2: no source files touched, all deltas live under `paper/` + docs). **What 11.3 shipped:** (a) `paper/bibliography.bib` replaced end-to-end. The one-entry `@misc{chronoagent-stub,...}` placeholder from 11.1 is gone; the file now holds 16 real entries grouped by the section that introduces them. Attacks block (cited from `01_introduction.tex`, `02_related_work.tex` section 2.1, `05_experiments.tex` config enumeration): `minja` (Dong et al. 2025, arXiv:2503.03704), `agentpoison` (Chen et al. NeurIPS 2024, arXiv:2407.12784), `amemguard` (Wang et al. 2025, arXiv:2510.17968). Allocation block (cited from `02_related_work.tex` section 2.2, `03_system_design.tex` allocator paragraph, `04_methodology.tex` allocator subsection): `contractnet` (Smith 1980, IEEE Trans. Computers, DOI 10.1109/TC.1980.1675516), `decpomdp` (Oliehoek & Amato 2016, Springer textbook), `autogen` (Wu et al. 2023, arXiv:2308.08155), `metagpt` (Hong et al. ICLR 2024, arXiv:2308.00352), `crewai` (crewAI Inc., GitHub). Change-point + forecasting block (cited from `02_related_work.tex` section 2.3, `03_system_design.tex` scorer paragraph, `04_methodology.tex` detector subsection): `adams2007bocpd` (Adams & MacKay 2007, arXiv:0710.3742), `chronos` (Ansari et al. 2024, arXiv:2403.07815, with a `note` field documenting that ChronoAgent uses the Chronos-2 refresh of the same family). Runtime monitoring block (cited from `02_related_work.tex` section 2.4 and `05_experiments.tex` `baseline_sentinel` config): `nemoguardrails` (Rebedea et al. 2023, arXiv:2310.10501), `llamaguard` (Inan et al. 2023, arXiv:2312.06674), `constitutionalai` (Bai et al. 2022, arXiv:2212.08073). Framework stack block (cited from `01_introduction.tex` and `03_system_design.tex`): `langchain` (Chase, GitHub), `langgraph` (LangChain Inc., GitHub), `chromadb` (Chroma Inc., GitHub). (b) Inline `\cite` wiring across five sections. `01_introduction.tex`: framework-stack cite block added to the "Problem" paragraph (`\citep{langchain,langgraph,autogen,metagpt}`), attacks cited `\citep{minja,agentpoison,amemguard}`, contribution bullet C4 anchored on `\citep{minja}` and `\citep{agentpoison}`. `02_related_work.tex`: all four `\todo{cite ...}` markers removed, every subsection now has real `\citep` / `\citet` calls that resolve. Section 2.3 was rewritten to drop the literal "(Adams & MacKay, 2007)" and "Chronos-2 (2024)" parentheticals because `natbib` is in numeric mode (`[numbers,sort&compress]`) -- leaving the inline author-year would have produced duplicated attribution like "BOCPD (Adams & MacKay, 2007) [1]". `03_system_design.tex`: LangGraph/LangChain/Chroma cited at the architecture hook sentence, BOCPD + Chronos cited at the scorer paragraph, contract-net cited at the allocator paragraph. `04_methodology.tex`: BOCPD and Chronos cited at the detector subsection, plus a new one-sentence Dec-POMDP framing pointer (`\citet{decpomdp}`) at the top of the health-weighted allocation subsection. `05_experiments.tex`: `main_experiment` and `agentpoison_experiment` config lines anchored on `\citep{minja}` / `\citep{agentpoison}`; `baseline_sentinel` line cited to LlamaGuard + NeMo Guardrails with the sentence "in the spirit of" so the citation does not fabricate a "Sentinel" paper. Abstract (`00_abstract.tex`), results (`06_results.tex`), discussion (`07_discussion.tex`), and conclusion (`08_conclusion.tex`) were left cite-free on the convention that abstracts and discussion blocks avoid inline citations in numeric-style papers and that the results/conclusion sections reuse references already anchored in the methodology and experiments. (c) `main.tex` comment housekeeping: the task-11.3 line in the header comment was rewritten from "Task 11.3 populates bibliography.bib" (future tense) to "Task 11.3 populates bibliography.bib with real entries for every work named in the prose and wires \\cite{} calls inline" (past/done), and the pre-bibliography banner `% Bibliography (populated in task 11.3)` was trimmed to `% Bibliography`. No functional LaTeX change. (d) Local audit script: a 13-line Python script (`_bibcheck.py`, deleted after use) extracted `^@\\w+{key,` from `paper/bibliography.bib` and `\\cite[tp]?{...}` from every `.tex` file under `paper/`, then printed the set difference both ways. Output confirmed zero missing cite keys and zero unused bib entries before commit. This is the best verification available pre-push since no LaTeX toolchain is installed locally; a full `latexmk -pdf main.tex` compile is left for whoever runs the CI LaTeX step. **Key design decisions locked in for 11.3:** (1) **Real entries, not fabricated ones.** Every bib key corresponds to a known paper or software repo; no entry was made up to "satisfy" a name-drop. The one edge case (the "Sentinel-style baseline" from the CLAUDE.md scoping note) was handled by citing representative reactive filters (LlamaGuard + NeMo Guardrails) rather than fabricating a `sentinel` key. The scoping gap is documented in the 11.3 Challenges block in PLAN.md. (2) **Strip inline author-year parentheticals when going numeric.** `natbib` with `[numbers,sort&compress]` renders `\citep{adams2007bocpd}` as `[N]`, so literal "(Adams & MacKay, 2007)" + `\citep{adams2007bocpd}` would double-attribute. Every such parenthetical in the pre-11.3 prose was stripped and replaced by a bare cite. (3) **Cite each work once per section that names it**, not once per paper. With numeric citations there is no reader benefit to withholding a later cite, and the section-level coverage protects against the reader who skims straight to a single section. (4) **Abstract stays cite-free** on the standard convention that abstracts do not contain inline citations in numeric-style papers. **Files changed (0 src + 0 test + 3 doc + 6 paper):** `paper/bibliography.bib` (rewritten: 16 real entries replacing the 1-entry stub), `paper/main.tex` (2 header-comment tweaks; no functional LaTeX change), `paper/sections/01_introduction.tex` (framework stack cite + attacks cite + C4 contribution anchor), `paper/sections/02_related_work.tex` (all four `\todo{cite ...}` markers removed, real `\citep`/`\citet` calls across all four subsections, literal author-year parentheticals stripped), `paper/sections/03_system_design.tex` (LangGraph/LangChain/Chroma + BOCPD/Chronos + contract-net cites), `paper/sections/04_methodology.tex` (BOCPD + Chronos cites at the detector subsection, Dec-POMDP framing pointer at the allocator subsection), `paper/sections/05_experiments.tex` (MINJA + AgentPoison + LlamaGuard/NeMo Guardrails cites at the config enumeration); PLAN.md (11.3 ticked, 11.3 entries appended to Findings/Challenges/Completed), CLAUDE.md (this table), README.md (Currently: line bumped from 11.2 to 11.3). No new Python dependencies, no source code changes, no test changes. Test suite unchanged at 1500 passing / 95.98% coverage. **What 11.2 shipped (retained for context):** (a) New `\figureiffound` macro in `paper/main.tex` (right after `\inputiffound`): guards `\IfFileExists` around `\includegraphics[width=\columnwidth]{...}` so a missing Phase 10 figure becomes a visible red `\todo` in the PDF instead of a hard compile error. Takes an EXPLICIT path (`../results/figures/fig5_roc_curve.png`) rather than a bare stem because `\IfFileExists` does NOT respect `\graphicspath` (a documented gotcha in the `graphics` bundle's internals). (b) Claim-to-experiment map in `paper/sections/05_experiments.tex` rewritten from a raw `\begin{tabular}` inside `\begin{center}` into a proper `\begin{table}[tbp]\centering\caption{...}\label{tab:claim-map}\end{table}` booktabs float. This resolves the `\cref{tab:claim-map}` forward-reference that 11.1 dropped in, so the scaffold no longer emits an `undefined reference` warning on every compile. The table has 4 columns (Claim, Primary experiment, Comparator, Artefacts) and splits C2/C3 across two rows so the long artefact lists fit. (c) C5 softened, not instrumented. The C5 claim in `05_experiments.tex` went from "less than 1\% computational overhead" to "can be run inline per PR-handling step without stalling the control plane. C5 is a soft, qualitative claim in this draft; a quantitative wall-clock microbenchmark is deferred to task 11.4 and the reproducibility harness." The matching subsection in `06_results.tex` has a paragraph explaining the soft-claim framing plus a single `\todo` pointing at `results/tables/table4_latency.tex` for 11.4. **Rationale for downgrading rather than instrumenting:** wiring a real latency benchmark would require a new `latency_ms` column on `RunResult` + `MetricAggregate`, a schema change on `runs.csv` + `aggregate.json`, a new column on `make_main_results_table`, a new plot function, and a fresh test set for each of those -- well beyond the "map claims to experiments" scope of 11.2 and more naturally co-located with the `make reproduce` Makefile wiring that 11.4 already owns. The original CLAUDE.md claim that `make_main_results_table` "already has a column for" latency was incorrect (verified against `tables.py`: `_METRIC_COLUMNS` is exactly AWT, AUROC, F1, alloc-eff), which confirmed 11.2 would have had to add the column itself. (d) Four figure placeholders in `06_results.tex` flipped to `\figureiffound`: fig1_signal_drift for C2 (per-experiment path `../results/main_experiment/figures/fig1_signal_drift.png`), fig4_allocation_efficiency_over_time for C3 (multi-experiment path `../results/figures/...`), fig5_roc_curve for C4, fig6_ablation_bar_chart for Ablations. The C1 `signals_minja.png` placeholder was ALSO flipped to `\figureiffound` at `../results/signals_minja.png` even though it is a Phase 1 artefact, because the file already exists on disk and it costs nothing to guard it. fig2_health_score_comparison and fig3_awt_box_plot are not yet referenced anywhere in the scaffold; leaving them for a later section-surgery pass. (e) Paper-side captions/labels for tables `tab:main-results` and `tab:ablation`. The two `\inputiffound{../results/tables/tableN_*.tex}` calls are now wrapped in `\begin{table}[tbp]\caption{...}\label{tab:...}\end{table}` floats so `\cref` references from the result prose resolve cleanly. The tabular body is still owned by the 10.8 generator; only the caption + label live on the paper side, preserving the Phase 10 principle that the paper is coupled to whatever `chronoagent compare-experiments` most recently wrote. (f) Per-claim prose paragraphs dropped into every `06_results.tex` subsection (C1 through C5 plus Ablations) so the results section now reads as a real paper skeleton rather than a figure-wiring stub. Each paragraph is ~4 sentences, references the matching figure via `\cref{fig:...}`, and cross-links to the relevant table via `\cref{tab:...}`. **Key design decisions locked in for 11.2:** (1) Explicit paths in `\figureiffound`, not stems, because `\IfFileExists` ignores `\graphicspath`. This is the OPPOSITE convention from how plain `\includegraphics{fig1_signal_drift}` would work; the asymmetry is intentional since every current figure is Phase-10-generated and may be stale. (2) C5 deferral, not instrumentation, documented with full rationale in the 11.2 Challenges entry in PLAN.md so future sessions can see why the call was made. 11.4 owns the actual microbenchmark wiring. (3) `\texttt{}` around any identifier with underscores inside caption/fallback prose (`signals_minja` -> `\texttt{signals\_minja}`). **Files changed (0 src + 0 test + 3 doc + 3 paper):** `paper/main.tex` (added `\figureiffound`), `paper/sections/05_experiments.tex` (rewrote claim-map table, softened C5 claim text, dropped stale `\todo{expand per-experiment prose in task 11.2.}`), `paper/sections/06_results.tex` (flipped all figure placeholders, wrapped two `\inputiffound` calls in `\begin{table}\label{}\end{table}` floats, dropped per-claim prose paragraphs, softened C5 subsection, dropped stale `\todo{narrative prose per claim once 11.2 finalizes the wiring.}`); PLAN.md (11.2 ticked, 11.2 entries added to Findings/Challenges/Completed), CLAUDE.md (this table), README.md (Currently: line bumped from 11.1 to 11.2). No new Python dependencies, no source code changes. Test suite unchanged at 1500 passing / 95.98% coverage. **What 11.1 shipped (retained for context):** New `paper/` directory holding (a) `main.tex` with a `[11pt,a4paper]{article}` preamble (inputenc utf8, fontenc T1, lmodern, microtype, geometry 1in, parskip, amsmath/amssymb/amsthm, booktabs, multirow, graphicx, subcaption, xcolor, siunitx, natbib with `plainnat`, hyperref, cleveref), a `\graphicspath{}` that lists `../results/`, `../results/figures/`, and every per-experiment `../results/<exp>/figures/` so body sections can reference figures by bare stem (`fig1_signal_drift.png` etc.), a `\inputiffound` macro that guards every `\input{../results/tables/<stem>.tex}` call so a missing Phase 10 artefact becomes a visible red `\todo` in the PDF instead of a hard compile error, a `\sys`/`\awt`/`\kldiv`/`\todo` convenience macro block, title block, and nine `\input{sections/##_name}` lines ending in a stub `\bibliography{bibliography}`; and (b) nine section files under `paper/sections/` (`00_abstract.tex`, `01_introduction.tex` with C1-C5 enumerated contribution block and a roadmap of `\Cref` forwards, `02_related_work.tex` with four labeled subsections (attacks, allocation, BOCPD/Chronos, runtime monitoring), `03_system_design.tex` with three control-plane paragraphs (Monitor, Scorer, Allocator) plus Human Escalation, `04_methodology.tex` with formal signal definitions, BOCPD + Chronos ensemble, AWT definition, and health-weighted allocation cost function, `05_experiments.tex` with claims C1-C5, the six-config enumeration, a raw `tabular` claim-to-experiment map that 11.2 is supposed to rewrite into a proper `\label{tab:claim-map}`, and the full reproducibility protocol quoting the `chronoagent run-experiment phase10` + `chronoagent compare-experiments` command lines, `06_results.tex` with per-claim subsections C1-C5 plus ablations and `\inputiffound{../results/tables/table1_main_results.tex}` / `table2_ablation.tex` drop points, `07_discussion.tex` with Pivot-A documentation + limitations + threat-model boundary, `08_conclusion.tex` with a Reproducibility section heading); and (c) `paper/bibliography.bib` with one placeholder `@misc{chronoagent-stub, ...}` entry that keeps `bibtex` happy until 11.3 fills in real entries. **Key design decisions locked in:** (a) **Nine section files, not a single `main.tex` blob.** Keeps `main.tex` readable and gives 11.2 a small, unambiguous drop point per claim (edit `05_experiments.tex` and `06_results.tex` in place; no section-surgery required). (b) **`natbib` + `plainnat`, not `biblatex`.** Biblatex would force a `backend=biber` step in the CI LaTeX image; natbib works with plain `bibtex` and is the lower-cost dependency. (c) **`\inputiffound` on every table `\input`.** Tables live at `../results/tables/*.tex` (a Phase 10 output), and the scaffold must compile before 11.2/11.4 regenerate those artefacts. A bare `\input{../results/tables/table1_main_results.tex}` would hard-fail on a clean checkout; `\inputiffound` turns that into a visible red `\todo` line in the PDF so the gap is obvious to anyone reading the scaffold PDF. (d) **C5 is explicitly unbacked.** The scaffold leaves a `\todo{microbenchmark table; populated in 11.2 alongside the latency instrumentation.}` rather than a placeholder number. 11.2 must either wire a real latency microbenchmark into the runner (preferred) or downgrade the claim. Documented as a Challenge in the PLAN.md Phase 11 log so the decision isn't silently deferred. (e) **Graphics path covers both multi-experiment and per-experiment figure dirs.** `plot_signal_drift` (per-experiment) writes under `../results/<exp>/figures/`; `generate_all_plots` (multi-experiment comparisons) writes under `../results/figures/`. The scaffold lists both plus every Phase 10 experiment name explicitly so body text can reference either class of figure by bare stem. If a new experiment is added, `\graphicspath` in `main.tex` needs one more entry. **Key gotchas locked in:** (1) **`\IfFileExists` paths are resolved relative to the current working directory at compile time, NOT the `\input`-er directory.** Compile from `paper/` (i.e. `cd paper && latexmk -pdf main.tex`) or paths break. Documented in the main.tex header comment. (2) **`natbib` with zero `\cite` commands is fine**: bibtex will warn but not error. The stub `@misc` entry is there so `bibtex main` always produces a `main.bbl` even if no section cites anything. (3) **Em dashes (U+2014) are banned project-wide** (CLAUDE.md prose style rule). All nine section files audited clean before commit; the project uses `--` in source (renders as en dash) or commas/colons. **Files changed (0 src + 0 test + 3 doc + 11 new paper files):** `paper/main.tex`, `paper/sections/00_abstract.tex`, `paper/sections/01_introduction.tex`, `paper/sections/02_related_work.tex`, `paper/sections/03_system_design.tex`, `paper/sections/04_methodology.tex`, `paper/sections/05_experiments.tex`, `paper/sections/06_results.tex`, `paper/sections/07_discussion.tex`, `paper/sections/08_conclusion.tex`, `paper/bibliography.bib` (all new); PLAN.md (11.1 Findings/Challenges/Completed added, 11.1 ticked), CLAUDE.md (this table), README.md (Phase 11 row flipped to In progress, Currently line bumped to 11.1). No new Python dependencies, no source code changes. Test suite unchanged at 1500 passing / 95.98% coverage. |

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
