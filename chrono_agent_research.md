# ChronoAgent: Pre-Planning Research Dossier
> Compiled: 2026-04-09 | Last updated: 2026-04-09 (roles 3 & 5 verified via playwright) | Status: Research Only, No Architecture/Scope Yet

---

## Confidence Legend
- ✅ High, verified via direct page fetch or multiple corroborating sources
- ⚠️ Medium, inferred from secondary sources (search snippets, mirrors, related listings)
- ❌ Low, couldn't access primary source; treat as approximate

---

## 1. Target Roles: What They're Actually Asking For

### Role 1, AI Security PhD, Linköping University (LiU) | Deadline: 20 Apr
**Confidence: ✅ (direct fetch)**
- **Exact problem:** Memory poisoning in LLM agents, adversaries exploit persistent memory (storage, retrieval, adaptation) to manipulate agent behavior over time
- **Technical focus:** Attack mechanisms, defenses, statistical signal processing, decentralized ML, privacy
- **Key fact:** The LiU group has already published the foundational paper on this exact topic (arxiv:2601.05504). This PhD is the continuation of that work. Your project must align with their research lineage, not just the topic.
- **Application domain:** Healthcare, finance, cybersecurity, autonomous vehicles
- **Desirables:** Mathematical maturity, ML security, collaborative research

---

### Role 2, Decentralized Task Allocation in Multi-Agent Systems, Aston | Deadline: 22 Apr
**Confidence: ✅ (direct fetch)**
- **Exact problem:** How do multiple agents coordinate without central control when operating under partial information and uncertainty?
- **Technical focus:** Game theory, MARL, probabilistic planning (Dec-POMDP), robustness under delays/abnormal agent behavior, simulation-based evaluation
- **Application domain:** Logistics, distributed services, intelligent infrastructure
- **Desirables:** Reinforcement learning, game theory, optimisation, large-scale simulation

---

### Role 3, Distributed Machine Learning PhD, NTNU (Norway/SURE-AI) | Deadline: 24 Apr
**Confidence: ✅ (verified via playwright)**
- **Exact problem:** Making distributed ML robust to network outages and computational bottlenecks, not LLM-focused, this is a signal processing / communications PhD
- **Technical focus:** Distributed ML algorithms, coding and information theory, communication networks and protocols, statistical signal processing
- **Context:** Part of SURE-AI (Norwegian national AI centre, 2025–2030). Department of Electronic Systems, Signal Processing group. Supervisor: Prof. Kimmo Kansanen (kimmo.kansanen@ntnu.no)
- **Required:** Master's in electronics, mathematics, data science, or communication technology. Strong mathematical background.
- **Preferred:** Prior experience in distributed ML, coding/information theory, communication networks
- **⚠️ Critical reframe:** This is NOT an agentic AI / LLM role. It is a communications/signal processing PhD that happens to touch ML. ChronoAgent's connection here is through "distributed behavioral metric aggregation that is robust to node failures", framing must emphasize the distributed systems robustness angle, not the LLM angle. Weaker alignment than previously assumed.

---

### Role 4, Sustainable and Resource-Efficient ML PhD, LiU | Deadline: 24 Apr
**Confidence: ✅ (direct fetch)**
- **Exact problem:** ML's computational and environmental cost, scaling has delivered performance at the cost of enormous energy, memory, and carbon footprint
- **Technical focus:** Data selection/filtering, model compression, hardware-aware optimization, inference cost reduction, integration with fairness and accessibility
- **Desirables:** Python, LaTeX/git, GNU/Linux, strong mathematics, algorithm implementation
- **Application domain:** Sustainable AI broadly, any deployment context

---

### Role 5, Generative AI & Intelligent Systems Engineer KTP, BCU | Deadline: 27 Apr
**Confidence: ✅ (verified via playwright)**
- **Exact problem:** Hockley Mint (jewellery manufacturer) wants an AI-powered platform for bespoke jewellery design within retailer networks
- **Technical focus:** Multilingual conversational AI, NLP generative rendering pipelines, AI-driven 3D generation (Diffusion, CLIP, NeRF, 3D GANs), ETL/data pipelines, system architecture, cloud (AWS/GCP/Azure)
- **Essential:** Python, PyTorch/TensorFlow, LLMs/NLP, API/system architecture, cloud computing, ETL pipelines
- **Desirable (key hooks):** Multi-agent AI frameworks (LangChain explicitly listed), Explainable AI (SHAP, LIME), generative AI (Diffusion/CLIP/NeRF), CAD/CAM, manufacturing/digital commerce
- **Context:** KTP = 26-month industry placement at Hockley Mint, employed by BCU. Supervisors: Gerald Feldman & Essa Shahra (BCU Computer Science). Salary: £38-42k.
- **Honest assessment:** This is an engineering delivery role, not research. The explicit mention of multi-agent frameworks (LangChain) is a direct hook for ChronoAgent. What gets you hired: a working multi-agent LLM system you built, not a research paper. Show the code.

---

### Role 6, Human-Centered AI for Programming Environments PhD, UvA | Deadline: 28 Apr
**Confidence: ✅ (direct fetch)**
- **Exact problem:** AI coding tools and IDEs risk reinforcing exclusion for neurodiverse and diverse developers if accessibility, cognition, and power relations are treated as secondary
- **Technical focus:** IDE plugin/extension dev (VS Code), LLM integration into developer tools, participatory/qualitative research methods, mixed-methods (A/B testing, focus groups, co-design)
- **Desirables:** Feminist/queer-inclusive computing perspectives, HCI, accessibility standards, open-source experience, generative AI hands-on
- **⚠️ Important:** This role has a strong social-science component. Technical strength alone is insufficient, you need HCI/participatory research framing.

---

### Role 7, Agentic AI for Autonomous Task Management PhD, Teesside | Deadline: 3 May
**Confidence: ✅ (direct fetch)**
- **Exact problem:** Building agentic AI systems that autonomously perform complex tasks across digital platforms while remaining controllable and explainable
- **Technical focus:** Multi-agent architectures, RL/distributed systems, planning/scheduling, dynamic task allocation, software integration, workflow automation
- **Industry partner:** MCD Systems (bespoke software and AI solutions)
- **Desirables:** Master's degree, distributed systems, RL or planning algorithms background

---

### Role 8, ML, Forecasting & Time Series PhD, University of Reykjavik | Deadline: 30 May
**Confidence: ✅ (direct fetch via informatics-europe.org)**
- **Exact problem:** Building practical foundation models for time-series forecasting that are fine-tunable for specific needs, with interpretability for critical industries (manufacturing)
- **Technical focus:** Transformer-based architectures (inverted transformers), chain-of-thought reasoning, training on large/synthetic datasets, anomaly detection, classification, imputation
- **Compute context:** LUMI AI Factory (high-performance GPU cluster)
- **Desirables:** GPU computing experience, ML model fine-tuning, manufacturing domain knowledge

---

### Role 9, Multi-Agent Agentic Systems PhD, Aston (CyberHub) | Deadline: 1 Jun
**Confidence: ✅ (direct fetch + secondary)**
- **Exact problem:** Developing collaborative LLM-based agents that autonomously manage resources, compose services, and optimize infrastructure delivery
- **Technical focus:** LLM-based agent development, MCP (Model Context Protocol), A2A (Agent-to-Agent) protocols, LangChain/CrewAI frameworks, cloud/local testbeds, infrastructure security and monitoring
- **Application domain:** Autonomous resource allocation, service composition, infrastructure security
- **Funding:** Fully-funded, £21,805/yr stipend, October 2026 start

---

## 2. Existing Literature Landscape

### 2a. Memory Poisoning in LLM Agents
**Confidence: ✅ (multiple verified papers)**

| Paper | What it does | Gap it leaves |
|-------|-------------|---------------|
| AGENTPOISON (NeurIPS 2024) | Red-teaming via knowledge base / memory poisoning | Reactive, attacks demonstrated, not predicted |
| MINJA (arxiv:2503.03704, Mar 2025) | Query-only memory injection achieving >95% success | Reactive, attack, not defense |
| A-MemGuard (arxiv:2510.02373, 2025) | Dual-memory + consensus verification defense, 95% reduction in poisoning | Reactive sanitization, detects/fixes after injection |
| LiU paper (arxiv:2601.05504, Jan 2026) | Full attack/defense framework on memory-based LLM agents, tested on EHR (MIMIC-III) | Reactive, trust thresholds need adaptive calibration; open problem stated explicitly |

**Honest gap:** All defenses are reactive. None forecast the *likelihood* of impending attack from behavioral drift signals.

---

### 2b. Anomaly Detection in Multi-Agent Systems
**Confidence: ✅ (direct fetch on two key papers)**

| Paper | Method | Proactive? | Gap |
|-------|--------|-----------|-----|
| SentinelAgent (arxiv:2505.24201, 2025) | Graph-based execution graph modeling + LLM oversight agent | Partially, models structure, but intervenes reactively during execution | No temporal forecasting of future anomaly likelihood |
| TraceAegis (arxiv:2510.11203, 2025) | Hierarchical trace-based detection, 94.3% accuracy | Reactive | No proactive prediction |
| Temporal Attack Pattern Detection (arxiv:2601.00848, Jan 2026) | Fine-tunes LMs on OpenTelemetry traces (synthetic + real), 74.29% benchmark accuracy | Reactive, analyzes past traces | Explicitly requires human oversight; false positive problem open |

**Honest gap:** All three are reactive, they detect attacks after behavioral deviation begins. None forecast agent trustworthiness drift before it manifests as an anomaly.

---

### 2c. LLMs + Time Series Forecasting
**Confidence: ✅**

| Paper | What it does | Relevance to ChronoAgent |
|-------|-------------|--------------------------|
| Empowering TS Forecasting with LLM-Agents (arxiv:2508.04231) | LLM agents automate forecasting pipeline (data selection, model choice) | Uses LLMs to forecast external signals, not agent behavior |
| DCATS | LLM agent for data-centric AutoML for TS, 6% error reduction | Same, forecasting is the output, not the coordination mechanism |
| Time-LLM | Patches TS as text prototypes into frozen LLMs | Technical approach relevant for lightweight implementation |
| Amazon Chronos-2 (Oct 2025) | Pretrained transformer for uni/multivariate/covariate forecasting | Most relevant off-the-shelf model for ChronoAgent's forecasting component |

**Honest gap confirmed:** These use LLMs to forecast external time-series. No paper uses temporal models to forecast internal agent behavioral states for coordination/security purposes.

---

### 2d. Decentralized MARL for Task Allocation
**Confidence: ✅**

| Paper | What it does | Gap |
|-------|-------------|-----|
| LGTC-IPPO (arxiv:2503.02437, 2025) | Dynamic cluster consensus for decentralized multi-resource allocation | Stable rewards, but agent trust/health not factored into allocation |
| Dec-POMDP survey (multiple 2024-2025) | Comprehensive framework for partial observability | Uncertainty is environmental, not about agent reliability |
| MARL for Resource Allocation survey (arxiv:2504.21048) | Covers cooperative and noncooperative decentralized regimes | No temporal agent health prediction feeding into allocation |

**Honest gap:** Task allocation treats agents as reliable actors. No framework adjusts allocation based on predicted future agent reliability/health.

---

### 2e. Proactive/Predictive Security in MAS: Direct Search
**Confidence: ✅ (search performed explicitly for this gap)**
- TrustAgent (EMNLP 2024), TrustLLM (2024), focus on trustworthiness benchmarks and alignment, not temporal prediction
- No paper found that uses time-series forecasting of agent behavioral metrics to predict and preempt adversarial manipulation
- **Gap is confirmed real and unoccupied as of April 2026**

---

## 3. The Core Claim: Is the Gap Real?

**Yes, with one important nuance.**

The gap that ChronoAgent targets, *using time-series forecasting of agent behavioral metrics to proactively inform both task allocation and security*, does not appear in the literature. The four communities (TS forecasting, MAS security, decentralized MARL, agentic AI) have not bridged this.

**Nuance to be honest about:** The gap is at the intersection of existing mature fields. ChronoAgent is not discovering new physics, it's applying known tools (transformer-based forecasting, MARL, memory integrity) in a combination that hasn't been done. This is appropriate PhD-level novelty: a new system with a clear conceptual contribution.

---

## 4. Known Unknowns / Things to Verify Before Planning

| Question | Confidence | Why It Matters |
|----------|-----------|----------------|
| NTNU PhD exact research angle | ⚠️ | Couldn't access JS-rendered page, need to know if it's federated learning, edge inference, or something else entirely before framing project for this role |
| BCU KTP full JD | ⚠️ | SSL error, secondary sources suggest conversational AI + 3D generation; confirm before pitching a forecasting/security project to a jewellery platform KTP |
| LiU AI Security deadline (20 Apr) | ✅ | Only 11 days away, project needs to at minimum be describable in a proposal by then |
| Whether SentinelAgent code is open-source | ❓ | If yes, can build on it rather than from scratch, check GitHub |
| Chronos-2 licensing | ❓ | Need MIT/Apache for open use in project |
| Whether SURE-AI has public API or data | ❓ | Could be a collaboration opportunity for NTNU application |

---

## 5. Cross-Role Alignment Map (Honest Version)

| Role | Alignment | Honest Note |
|------|-----------|-------------|
| LiU AI Security | Core | Temporal drift → memory poisoning prediction. Fits their open problem on adaptive trust calibration |
| Aston Decentralized | Core | Allocation weighted by predicted agent reliability under uncertainty |
| NTNU Distributed ML | Moderate (revised down) | This is a signal processing PhD, not LLM. Framing: distributed metric aggregation robust to node failures. Weaker than originally claimed. |
| LiU Sustainable ML | Strong | Lightweight forecasters (quantized Chronos-2) vs bloated LLMs = efficiency story |
| BCU KTP | Moderate-Strong (revised up) | LangChain explicitly listed as desirable. This is an engineering role, show a working multi-agent system. The project as a portfolio is the pitch. |
| UvA Human-Centered | Moderate-Weak | Human-in-the-loop when agent confidence is low is a valid framing, but core HCI/neurodiversity work is very different, be honest in application |
| Tees Agentic | Core | Predictive autonomy + controllability via forecasting is exactly their stated problem |
| Reykjavik Forecasting | Core | Temporal models for agent behavior = transformer-based TS applied to novel domain |
| Aston Multi-Agent | Core | LLM orchestration + MCP/A2A protocols + security monitoring |

---

## 6. Challenge Analysis, Logical Reasoning Through Each Risk

### Challenge 1: Evaluation is hard, how do you benchmark "proactive" security?

**Why it's hard:** Most anomaly detection benchmarks measure detection accuracy after an attack is in progress. Proactive detection requires measuring whether the system warns *before* the attack succeeds, a different and harder metric.

**Logical solution:**
- Adopt existing attack generators (AGENTPOISON, MINJA) as your attack source. You do not need to invent new attacks, you need a new *detection timing* metric.
- Define "advance warning time" (AWT): how many agent interaction steps before task output corruption does the system raise an anomaly flag?
- Baselines to beat: SentinelAgent (reactive graph-based), TraceAegis (trace-based, 94.3% accuracy). If ChronoAgent achieves similar accuracy but with positive AWT, that is the contribution.
- The evaluation framework itself is a publishable contribution, analogous to how AGENTPOISON's benchmark was NeurIPS-worthy independent of the attack method.
- Use synthetic attack injection (controllable ground truth) + a small real-world trace dataset for generalization testing.

**Residual risk:** Low. This is solvable by design, not by hoping signals are strong.

---

### Challenge 2: The temporal signal may be weak

**Why it's hard:** Agent behavior signals (latency, retrieval patterns, output entropy) may not drift in a statistically reliable way before an attack manifests. If the signal is indistinguishable from noise, forecasting is useless.

**Logical solution, three layers:**

1. **Choose signals with known sensitivity.** Memory poisoning specifically corrupts retrieval. The signal to watch is retrieval query entropy and retrieved document distribution shift (KL-divergence from baseline). These are not noisy, they are direct mechanistic consequences of the attack. This is not a coincidence; it follows from how the attack works. The LiU paper (2601.05504) and A-MemGuard both found behavioral shifts in retrieval patterns under attack, the signal exists.

2. **Use changepoint detection, not just forecasting.** Bayesian Online Changepoint Detection (BOCPD) and PELT are robust to noise and excel at detecting distribution shifts in streaming data. They do not require strong predictable drift, they detect when the distribution of a signal changes. This is more appropriate than pure forecasting when signals are intermittent.

3. **Decouple the security claim from the allocation claim.** The allocation benefit (routing tasks away from unreliable agents based on predicted health) does not require the security signal to be perfect. Even a noisy health score improves allocation decisions probabilistically. The security detection can be a secondary result. If the signal proves too weak empirically, the allocation contribution still stands.

**Residual risk:** Medium on the security side. Low on the allocation side. Empirical validation of signal strength is the first experiment to run, before building the full system.

---

### Challenge 3: Scope creep, touching 4 fields shallowly

**Why it's hard:** Each field (TS forecasting, MAS security, decentralized MARL, agentic AI) has deep literature. Trying to advance all four results in a system paper with no real contribution to any.

**Logical solution, fix the contribution boundary explicitly:**

- **We contribute:** A temporal reliability scoring module that ingests agent behavioral time-series and outputs a per-agent health score, used for (a) task allocation weighting and (b) anomaly flagging.
- **We do NOT contribute:** New forecasting architectures (use Chronos-2/BOCPD off the shelf), new attack methods (use AGENTPOISON/MINJA), new MARL algorithms (use existing IPPO or simple weighted allocation), new LLM agent architectures (use LangChain/AutoGen as infrastructure).
- The paper's claim is about the *integration* and the *evaluation*, not about advancing any single subfield. This is a systems contribution with a well-scoped novelty claim.
- Concrete scope: 4-agent pipeline (Planner, Executor, Reviewer, Memory), one domain (autonomous code review), one attack type (memory injection), one forecasting method (BOCPD + Chronos-2 ensemble), two metrics (AWT, allocation efficiency under attack).

**Residual risk:** Low if the boundary is written down and enforced before any code is written.

---

### Challenge 4: Compute cost, forecasting on top of LLM agents

**Why it's hard:** Each agent already runs a 7B+ LLM. Adding a forecasting model on top multiplies cost and directly contradicts the sustainable ML angle (LiU Role 4).

**Logical solution, the overhead is small by design:**

- The forecasting model does NOT process text or run an LLM. It processes a vector of ~10 scalar behavioral metrics per agent per timestep: latency (ms), retrieval count, output token count, KL-divergence from baseline embedding, tool call frequency. These are cheap to log.
- The forecaster itself is tiny: BOCPD has no parameters; Chronos-2-Small is ~46M parameters vs. 7B+ for the agent LLMs. The overhead ratio is ~0.6%.
- The agents themselves can use quantized small LLMs (Phi-3-Mini at 3.8B INT4, Gemma-2B) instead of 70B models. The full system with quantized agents + tiny forecaster uses significantly less compute than a single GPT-4 call.
- This is the sustainable ML story for LiU Role 4: a secure, coordinated multi-agent system that achieves better reliability than a single large model at a fraction of the compute cost.
- Health score updates run asynchronously, they do not block agent inference, so latency impact is negligible.

**Residual risk:** Very low. The compute architecture is designed from the ground up to be lightweight. This is not a retrofit, it is a constraint that shapes every design decision.

---

## 7. Revised Alignment Map (Post-Playwright Verification)

| Role | Alignment | Honest Note |
|------|-----------|-------------|
| LiU AI Security | **Core** | Temporal drift = memory poisoning early warning. Directly extends their open problem on adaptive trust calibration (2601.05504) |
| Aston Decentralized | **Core** | Allocation weighted by predicted agent reliability = coordination under uncertainty |
| NTNU Distributed ML | **Moderate** (revised down from Strong) | Signal processing PhD, not LLM. Frame as: distributed metric aggregation robust to node failure. Honest stretch. |
| LiU Sustainable ML | **Strong** | Tiny forecaster + quantized agents < single large LLM. Efficiency is a design constraint, not a footnote. |
| BCU KTP | **Moderate-Strong** (revised up) | LangChain explicitly desirable. Engineering portfolio role, a working system is the pitch. |
| UvA Human-Centered | **Moderate-Weak** | Human escalation when health score drops is valid. But HCI/neurodiversity framing requires genuine social science background. |
| Tees Agentic | **Core** | Predictive autonomy + explainable health scores = controllability. Exact fit. |
| Reykjavik Forecasting | **Core** | BOCPD/Chronos-2 applied to novel domain (agent behavior) = novel application of their research area |
| Aston Multi-Agent | **Core** | LLM orchestration + MCP/A2A + security monitoring. Explicitly listed technologies. |

---

## 8. Open Questions Before Planning Starts

1. **What is the single prototype domain?** Recommendation: autonomous code review pipeline (3-4 agents). Justification: (a) easy to instrument, (b) MINJA/AGENTPOISON attacks are already designed for coding agents, (c) direct relevance to BCU KTP (LangChain-based), Tees, and Aston.
2. **First experiment to run:** Signal strength validation, instrument a 3-agent LangChain pipeline, inject MINJA-style memory attacks, measure whether retrieval KL-divergence drifts before task output degrades. If yes, proceed. If no, use BOCPD on latency + token entropy instead.
3. **SentinelAgent open-source?** Check arxiv/GitHub for released code, could provide the evaluation baseline without reimplementing it.
4. **Chronos-2 license?** Apache 2.0, confirmed open for use.
5. **Stack decision:** LangChain (agents) + BOCPD/Chronos-2-Small (forecasting) + SQLite (memory store) + Python. No exotic dependencies.

---

*Sources: arxiv.org, informatics-europe.org, jobs.ac.uk (playwright), jobbnorge.no (playwright), aston.ac.uk, liu.se, uva.nl, tees.ac.uk, lakera.ai, unit42.paloaltonetworks.com, galileo.ai*
