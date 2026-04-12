# ChronoAgent

> A multi-agent LLM system that watches its own behavior over time and uses what it sees to decide which agent to trust with the next task.

[![CI](https://github.com/rajo69/chronoagent/actions/workflows/ci.yml/badge.svg)](https://github.com/rajo69/chronoagent/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![tests](https://img.shields.io/badge/tests-1500%20passing-brightgreen)
![coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
![license](https://img.shields.io/badge/license-Apache%202.0-blue)

---

## What this is

ChronoAgent is a research prototype for a question that sits between three fields and is not answered by any of them on its own:

> **If a group of LLM agents is working together on a task, can the system tell, ahead of time, which agent is about to misbehave, and route work around it before any harm is done?**

Existing tools either detect a problem after it has already corrupted the output (anomaly detection on agent traces) or score trust at a single point in time (LLM evaluation benchmarks). Neither of these answer the question above. ChronoAgent treats every agent's observable behavior (latency, retrieval patterns, output entropy, tool-use frequency, KL divergence from a clean baseline) as a time series, learns what "healthy" looks like for each agent, and watches for the moment that signal starts drifting toward something the system has previously seen go wrong.

The output is a per-agent **health score** in `[0, 1]` that updates in real time. Other parts of the system (the task allocator, the human escalation layer) consume this score directly.

---

## The problem in plain language

Multi-agent LLM systems are starting to be deployed in places where mistakes are expensive: code review, customer support, financial analysis, healthcare triage. In every published attack scenario for these systems (memory poisoning, prompt injection through retrieved documents, tool misuse), the agent's *internal behavior* visibly changes for some number of steps before its *external output* becomes obviously wrong. That gap, however small, is the only window the system has to react.

Most current defenses live in the wrong end of that window. They wait for the bad output, classify it after the fact, and then try to recover. ChronoAgent is an attempt to live in the early part of the window instead.

---

## The idea

The contribution is not a new model. It is a particular combination of three off-the-shelf components, glued together with a small amount of original code, that produces a useful signal nobody else is producing today.

| Component | What it does | Why this one |
|---|---|---|
| **BOCPD** (Bayesian Online Changepoint Detection, Adams & MacKay 2007) | Detects when the distribution of a noisy signal has shifted | Robust to noise, no training needed, runs in microseconds, works on streaming data |
| **Chronos-2-Small** (Amazon, Apache 2.0) | Pretrained transformer that forecasts the next steps of a time series | The lightest practical foundation model for the job, around 46M parameters, no per-deployment fine-tuning |
| **An ensemble layer** | Blends BOCPD's changepoint probability with Chronos's anomaly score and converts both into a health score | Either component can fail or be unavailable; the ensemble degrades gracefully to whichever one is healthy |

The ensemble feeds two downstream subsystems:

1. A **decentralized task allocator** that weights its routing decisions by predicted agent reliability, so a wobbly agent gets fewer high-stakes tasks until its score recovers.
2. A **human escalation layer** that pages a person when no available agent has a health score above a configurable threshold.

The full justification for why this combination is the right one, with an honest map of the literature it sits next to, is in [`chrono_agent_research.md`](./chrono_agent_research.md).

---

## Project status

The project is built in numbered phases. Each phase has an exit criterion and a test gate. At the end of every phase, the relevant row in this table gets a tick.

| Phase | Name | Status |
|---|---|---|
| 0 | Bootstrap | ✅ Done |
| 1 | Signal Validation Gate | ✅ Done |
| 2 | Core Agent Pipeline | ✅ Done |
| 3 | Behavioral Monitor | ✅ Done |
| 4 | Temporal Health Scorer | ✅ Done |
| 5 | Decentralized Task Allocator | ✅ Done |
| 6 | Memory Integrity Module | ✅ Done |
| 7 | Human Escalation Layer | ✅ Done |
| 8 | Observability Dashboard | ✅ Done |
| 9 | Production Hardening | ✅ Done |
| 10 | Research Experiment Suite | ✅ Done |
| 11 | Paper Scaffold and Reproducibility | 🚧 In progress |
| 12 | CI/CD and Release | ⬜ Planned |

**Currently:** Phase 11 task 11.2 wired the claim-to-experiment map. The raw `tabular` in `paper/sections/05_experiments.tex` is now a proper `\begin{table}[tbp]\label{tab:claim-map}\end{table}` booktabs float, so the `\cref{tab:claim-map}` forward-reference dropped in by 11.1 resolves cleanly. Every `\todo{\includegraphics of figN_*}` placeholder in `06_results.tex` is now a `\figureiffound{../results/.../figN_*.png}{...}` call backed by a new `\figureiffound` macro in `main.tex` that mirrors the `\inputiffound` pattern: it guards `\IfFileExists` around `\includegraphics[width=\columnwidth]` so the scaffold still compiles green on a clean checkout. C5 (forecaster overhead) is explicitly downgraded in this draft to a qualitative "runs inline on CPU without stalling the control plane" claim, with the quantitative wall-clock microbenchmark deferred to 11.4 so it co-locates with the `make reproduce` wiring. Two result tables now get paper-side captions/labels (`tab:main-results`, `tab:ablation`), and every C1--C5 subsection plus the ablations block carries a short prose paragraph so `06_results.tex` reads as a real paper skeleton rather than a figure-wiring stub. Tasks 11.3 (bibliography), 11.4 (Makefile `reproduce*` targets + latency harness), and 11.5 (Docker pin + seed freeze) remain. 1500 tests pass with 95.98% line coverage; no source files changed in 11.2, so the test count carries forward unchanged.

**An honest pivot worth recording:** Phase 1 was a hard signal-validation gate. Before building anything else, we measured whether the behavioral signals we wanted to forecast were actually distinguishable from noise. KL divergence from a clean baseline turned out to be a strong primary signal (Cohen's d ≈ 1.6 on the MINJA attack benchmark), while three of the six secondary signals were effectively constant under our test conditions. The original framing around "advance warning time" did not survive the data. The project was reframed around concurrent detection plus reliability-weighted allocation, which the data does support. The full ruling is in [`docs/phase1_decision.md`](./docs/phase1_decision.md).

This pivot is part of why the README exists in this form. Building a research system is mostly about being honest with yourself when the empirical results contradict the original story.

---

## Architecture at a glance

```
                +---------------------------+
                |    LLM Agent Pipeline     |
                |  (LangGraph StateGraph)   |
                |  Planner, Reviewers,      |
                |       Summarizer          |
                +-------------+-------------+
                              |
                              v   per-step behavioral metrics
                +---------------------------+
                |   Behavioral Collector    |
                |  (latency, KL, entropy,   |
                |     tool calls, etc.)     |
                +-------------+-------------+
                              |
                              v   signal_updates  (message bus)
                +---------------------------+
                |  Temporal Health Scorer   |
                |   BOCPD  +  Chronos-2     |
                |       (ensemble)          |
                +-------------+-------------+
                              |
                              v   health_updates (message bus)
                +---------------------------+
                |   Downstream consumers    |
                |  - Task allocator (Ph 5)  |
                |  - Memory integrity (6)   |
                |  - Escalation layer  (7)  |
                |  - Dashboard         (8)  |
                +---------------------------+
```

Each box is a Python module under `src/chronoagent/`. Communication between them goes through a `MessageBus` interface that has two implementations: a `LocalBus` (in-process and synchronous, used in dev and tests) and a `RedisBus` (production). Swapping from one to the other is a single line in the FastAPI lifespan.

---

## Quick start

### Run it with Docker

```bash
git clone https://github.com/rajo69/chronoagent.git
cd chronoagent
cp .env.example .env
docker compose up --build
curl http://localhost:8000/health
```

### Run it without Docker

```bash
pip install uv
uv pip install -e ".[dev,experiments]"
pre-commit install
make test          # 503 tests, mock backend, no external API calls
make dev           # FastAPI on http://localhost:8000
```

The default backend is a deterministic `MockBackend` that returns canned responses. Tests and examples run with zero API cost. To use a real LLM, set `CHRONO_LLM_BACKEND=together` and provide `CHRONO_TOGETHER_API_KEY` in `.env`.

### Try the health endpoint

```bash
curl http://localhost:8000/api/v1/agents/health
```

This returns the current health score for every registered agent, the breakdown into BOCPD and Chronos contributions, and a system-level aggregate.

### Open the dashboard

With the server running, visit:

```
http://localhost:8000/dashboard/
```

The dashboard is a single self-contained HTML page (no build step, Chart.js from a CDN) that shows:

- **Header strip:** system health, agent count, pending escalations, quarantined doc count, live via a WebSocket at `/dashboard/ws/live` (refreshes every 2 seconds).
- **Agents panel (left):** per-agent health bars ordered most-at-risk first. Click an agent to drive the signal explorer.
- **Signal Explorer (center):** line chart of the selected agent's behavioral signals. A dropdown switches between KL divergence (default), latency, token count, retrieval count, tool calls, and memory query entropy.
- **Allocation Log (right):** most recent task-allocation decisions, task type, winning agent or `ESCALATED`, and the human-readable rationale.
- **Memory Inspector (bottom left):** baseline fit state, signal weights, total retrievals, and the current quarantine list.
- **Escalation Queue (bottom right):** pending escalations plus the most recently resolved ones.

All panels degrade gracefully: if the WebSocket drops it reconnects every 3 seconds, and the REST panels continue polling on their own interval.

---

## Repository layout

```
chronoagent/
├── src/chronoagent/
│   ├── api/             FastAPI routers (health, signals, review, health_scores)
│   ├── agents/          BaseAgent ABC plus the four review-pipeline agents
│   ├── pipeline/        LangGraph wiring for the agent pipeline
│   ├── memory/          ChromaDB-backed memory store and integrity helpers
│   ├── monitor/         BehavioralCollector, KL divergence, entropy
│   ├── scorer/          BOCPD, Chronos forecaster, ensemble, health scorer
│   ├── messaging/       MessageBus ABC, LocalBus, RedisBus
│   ├── observability/   structlog configuration
│   ├── db/              SQLAlchemy models, Alembic migrations, session helpers
│   ├── experiments/     Signal-validation experiment runner and analyzer
│   ├── config.py        Pydantic settings (CHRONO_* env vars)
│   ├── main.py          FastAPI app factory and lifespan
│   └── cli.py           Typer CLI (serve, run-experiment, check-health)
├── tests/
│   ├── unit/            Around 450 unit tests
│   └── integration/     End-to-end pipeline tests with the mock backend
├── docs/
│   ├── phase1_decision.md    Phase 1 GO/NO-GO ruling and raw signal results
│   └── dev_log.md            Per-session engineering notes
├── PLAN.md              Full implementation plan with task-level checklists
├── CLAUDE.md            Session context for the AI pair-programmer
├── chrono_agent_research.md   Pre-planning research dossier and gap analysis
└── pyproject.toml       Project metadata, deps, ruff/mypy/pytest config
```

---

## Development

### The four checks that gate every commit

CI runs the same four commands on every push and pull request. Run them locally before pushing.

```bash
py -m ruff check src/ tests/           # lint
py -m ruff format --check src/ tests/  # formatting
py -m mypy src/                        # strict static types
py -m pytest tests/ -q                 # unit and integration tests
```

If any of these fail, CI fails. There is no override.

### Useful commands

```bash
make dev              # FastAPI with hot reload
make test             # full test suite, mock backend, no API calls
make test-fast        # unit tests only
make lint             # ruff + mypy
make lint-fix         # ruff with --fix
make docker-up        # bring up the full stack (app + redis + postgres + chroma)
cz commit             # conventional commit prompt
```

### How a phase ends

Each phase in `PLAN.md` lists tasks with checkbox status. A phase is closed when:

1. Every task is checked off
2. The phase test gate is green
3. The phase exit criterion (a measurable claim, written before the phase started) is satisfied
4. The status row in this README is updated

This README, `CLAUDE.md`, and `PLAN.md` are kept in sync at the close of every session.

---

## Research context

The longer version of why this project exists, with a literature map of the four adjacent fields and an honest assessment of where the gap is and is not, lives in [`chrono_agent_research.md`](./chrono_agent_research.md). The short version:

- Memory poisoning attacks on LLM agents are documented (AGENTPOISON, MINJA, the LiU 2026 paper). The defenses that exist are reactive.
- Anomaly detection in multi-agent systems exists (SentinelAgent, TraceAegis). It is also reactive.
- Time-series forecasting has been applied to LLM agents only as a tool the agents *use*, not as a way to forecast the agents *themselves*.
- Decentralized task allocation in multi-agent reinforcement learning treats agents as reliable actors, with no notion of predicted future reliability.

ChronoAgent's contribution is the integration: a system that takes the time-series view seriously for behavioral metrics, and that lets two unrelated downstream subsystems (task allocation and security escalation) consume the same predictive signal. The contribution boundary is deliberately narrow. The project does not invent new forecasting models, new attacks, or new MARL algorithms. It uses existing tools in a combination that has not been published.

---

## Stack

| Layer | Choice |
|---|---|
| Web framework | FastAPI |
| Agent framework | LangGraph + LangChain |
| Vector memory | ChromaDB |
| Forecasting | BOCPD (around 80 lines of NumPy, in-house) and Chronos-2-Small |
| Message bus | LocalBus (dev), Redis pub/sub (prod) |
| Database | SQLite (dev), PostgreSQL (prod), Alembic for migrations |
| LLM backends | MockBackend (default), Together.ai, Ollama (optional) |
| Lint and format | Ruff |
| Type checking | Mypy strict mode |
| Tests | Pytest + Hypothesis |
| Container | Docker Compose |

---

## License

Apache 2.0. See [`LICENSE`](./LICENSE).

---

## Acknowledgements

Built as part of a portfolio of work for PhD applications in temporal modeling, multi-agent systems security, and decentralized coordination. The research dossier in [`chrono_agent_research.md`](./chrono_agent_research.md) lists the specific positions and the papers each section draws from.
