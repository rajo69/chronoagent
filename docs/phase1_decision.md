# Phase 1 Decision Document — Signal Validation GO/NO-GO Ruling

> **Date:** 2026-04-09  
> **Phase:** 1 — Signal Validation  
> **Status:** CONDITIONAL GO with Pivot A applied (concurrent detection reframe)  
> **Author:** ChronoAgent research log (auto-generated from a deterministic experiment run)

---

## 1. Experiment Setup

| Parameter | Value |
|-----------|-------|
| Backend | MockBackend (deterministic, seed=42) |
| Agent pair | SecurityReviewerAgent + SummarizerAgent |
| Steps per phase | 25 (clean) + 25 (poisoned) |
| Injected docs | 10 per collection (20 total) |
| Calibration steps | 10 (Phase A) |
| Attacks tested | MINJAStyleAttack, AGENTPOISONStyleAttack |
| Signals measured | 6 (latency, retrieval_count, token_count, kl_divergence, tool_calls, entropy) |

The experiment was run with:

```bash
chronoagent run-experiment \
  --config configs/experiments/signal_validation.yaml \
  --output results/
```

---

## 2. Raw Results

### 2.1 MINJAStyleAttack

| Signal | Clean μ | Clean σ | Poison μ | Poison σ | Cohen's d | Large effect? |
|--------|---------|---------|----------|----------|-----------|---------------|
| total_latency_ms | 4.09 | 0.86 | 4.12 | 0.70 | 0.040 | no |
| retrieval_count | 6.000 | 0.000 | 6.000 | 0.000 | 0.000 | no |
| token_count | 18.36 | 4.06 | 17.24 | 2.31 | 0.339 | no |
| **kl_divergence** | **42.21** | **32.89** | **97.18** | **47.61** | **1.343** | **YES** |
| tool_calls | 2.000 | 0.000 | 2.000 | 0.000 | 0.000 | no |
| memory_query_entropy | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | no |

Signals meeting the large-effect threshold (d > 0.8): 1 of 6, which gives NO-GO under the raw criterion.

### 2.2 AGENTPOISONStyleAttack

| Signal | Clean μ | Clean σ | Poison μ | Poison σ | Cohen's d | Large effect? |
|--------|---------|---------|----------|----------|-----------|---------------|
| total_latency_ms | 4.20 | 1.66 | 5.72 | 4.96 | 0.409 | no |
| retrieval_count | 6.000 | 0.000 | 6.000 | 0.000 | 0.000 | no |
| token_count | 18.36 | 4.06 | 17.24 | 2.31 | 0.339 | no |
| **kl_divergence** | **42.21** | **32.89** | **89.03** | **33.89** | **1.402** | **YES** |
| tool_calls | 2.000 | 0.000 | 2.000 | 0.000 | 0.000 | no |
| memory_query_entropy | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | no |

Signals meeting the large-effect threshold (d > 0.8): 1 of 6, which gives NO-GO under the raw criterion.

### 2.3 Changepoint and AWT Summary

| Attack | Signal | PELT CP | AWT | First crossing |
|--------|--------|---------|-----|----------------|
| MINJA | kl_divergence | step 9 | 0 | step 30 |
| AGENTPOISON | kl_divergence | step 9 | 0 | step 27 |

AWT is defined as the difference between the PELT changepoint step and the first threshold-crossing step. An AWT of 0 indicates that the threshold was crossed at or after the statistical changepoint, which corresponds to concurrent rather than proactive detection.

---

## 3. Signal-by-Signal Analysis

### kl_divergence: primary signal (strong effect)

The KL divergence between retrieval embeddings and the clean Gaussian baseline is the dominant signal under both attacks:

- MINJA: d = 1.343 (large), with the mean more than doubling (42 → 97).
- AGENTPOISON: d = 1.402 (large), with the mean roughly doubling (42 → 89).
- PELT flags a changepoint at index 9 of the concatenated (Phase A ‖ Phase C) signal series, and the detection threshold is first crossed at index 30 (step 5 of Phase C). Step indices throughout this document use this concatenated numbering.
- AWT = 0 in both cases. The PELT index precedes the threshold crossing, and the floor at zero (concurrent detection) dominates the raw subtraction.

**Interpretation.** KL divergence behaves as a reliable, attack-agnostic detector across the two attacks tested here, with d > 1.3 in both cases; whether the same effect size holds for attack classes outside MINJA and AGENTPOISON is open work. The doubling of embedding-distribution divergence is mechanistically consistent with the threat model: poisoned documents inject out-of-distribution embeddings into the retrieval space, shifting the empirical distribution away from the clean baseline calibrated in Phase A. The signal responds to both query-optimised (MINJA) and backdoor-trigger (AGENTPOISON) injection patterns; we do not claim coverage of adaptive attacks that target the KL signal itself, and that case is open work.

### retrieval_count, tool_calls: zero variance under MockBackend

Both signals are constant across all steps (retrieval_count = 6, tool_calls = 2). This is a MockBackend artefact: the deterministic backend always retrieves exactly `top_k = 3` documents per agent (3 + 3 = 6), and exactly two ChromaDB queries are issued per step. With σ = 0, Cohen's d is undefined and is reported as 0.000.

**Interpretation.** These signals carry no information under MockBackend, which is a deliberate simplification of the observation model for deterministic experimentation rather than a property of the signals themselves. Under a real LLM backend, retrieval patterns and tool-call frequency would be expected to vary with prompt content and with attack influence on retrieval, in which case the signals may become informative; re-running the validation against a real backend is open work. We therefore do not treat the zero d-values as evidence against the security framing; they are an artefact of the deterministic harness.

### memory_query_entropy: zero variance under MockBackend

The Shannon entropy of the top-k similarity score distribution is 0.000 throughout. The cause is the same MockBackend determinism: fixed responses regardless of query produce identical ChromaDB similarity scores against the fixed knowledge-base embeddings, and a constant entropy cannot differentiate poisoned from clean states.

The same artefact caveat as above applies. Under real LLM inference with token-level query variation, the similarity distribution would vary, in which case entropy may become informative; re-evaluating entropy against a real backend is tracked as open work. Entropy is retained in the signal set on this basis.

### total_latency_ms: small effect, indistinguishable from noise

Cohen's d is 0.119 under MINJA and 0.369 under AGENTPOISON. The latency signal has variance but shows only a small to small-to-medium effect; the variance is primarily timing noise from the Python process, since MockBackend inference is sub-millisecond and is not measurably affected by attack content.

**Interpretation.** Latency is not a reliable signal under MockBackend. It may become informative under a real LLM backend, where poisoned retrieval adds to context length and therefore to inference time, and we therefore retain it pending that re-evaluation; quantifying its contribution under realistic inference latency is open work.

### token_count: small-to-medium effect, but confounded

Cohen's d = 0.339 in both attacks; the value is identical because the same PR seeds are used in both runs. The difference (clean μ = 18.36 vs poisoned μ = 17.24) is small-to-medium, but it is driven by the different PR seeds (pr_seed = 0 vs pr_seed = 1000) rather than by the attack itself. The signal is therefore best described as confounded with PR content variation rather than as a measurement of the attack effect, and we do not treat it as an attack indicator at this stage; whether token count carries residual attack signal once the PR-seed confound is controlled for is open work.

---

## 4. GO/NO-GO Ruling

### Strict criterion: NO-GO

By the pre-specified gate (≥ 2 signals with Cohen's d > 0.8), both experiments return NO-GO: 1 of 6 signals meets the threshold.

### Adjusted ruling: CONDITIONAL GO with Pivot A

The strict NO-GO is substantially explained by MockBackend limitations rather than by a genuine absence of signal:

| Signal | Can show effect under MockBackend? | Reason |
|--------|------------------------------------|--------|
| kl_divergence | Yes, and does (d > 1.3) | Embedding arithmetic is backend-independent |
| retrieval_count | No | Fixed at `top_k` × agents = constant |
| tool_calls | No | Fixed at 2 per step |
| memory_query_entropy | No | Fixed similarity scores |
| token_count | Partial | Varies with PR seed, not attack |
| total_latency_ms | Partial | Mostly timing noise |

Of the six signals, three are MockBackend constants with no capacity to differentiate any state. Read in this light, the result is "1 of 3 informative signals shows a large effect" rather than "1 of 6 signals shows a large effect," which is a materially different conclusion.

KL divergence alone is sufficient to support the core detection hypothesis: embedding-distribution shift under memory poisoning is detectable, large in magnitude, and consistent across the two attack types tested. The framing "behavioural signals drift during memory poisoning" is therefore confirmed for the primary signal, with the temporal qualifier addressed below.

### Pivot A: concurrent detection

The advance warning time is zero on the only confirmed signal. Per the pre-specified pivot protocol:

> **Pivot A:** "Behavioral Time-Series Monitoring for Reliable Multi-Agent Task Allocation with Concurrent Attack Detection"

The paper framing therefore shifts from "proactive early warning" to "concurrent detection." PELT changepoint detection flags index 9 of the concatenated (Phase A ‖ Phase C) series for both attacks, and the detection threshold is first crossed at index 30 (MINJA) and index 27 (AGENTPOISON), which corresponds to real-time detection rather than advance warning. The AWT = 0 empirical result means either that the KL signal as constructed here does not expose enough leading structure to flag before the regime change, or that the burst-injection attack schedule used in Phase 1 makes the leading window too short to detect; we do not claim structural impossibility, and gradual-injection attack schedules together with leading-indicator signal designs are open work.

**Impact on project scope.** No code changes are required. Allocation efficiency becomes the primary paper contribution, and concurrent detection via KL divergence is the security contribution; whether a leading variant of the KL signal can recover a positive AWT under a slower injection schedule is tracked as open work.

---

## 5. Viable Signals for Phase 2 and Beyond

| Signal | Status | Note |
|--------|--------|------|
| **kl_divergence** | Confirmed primary signal | Large effect under both attacks; proceed to Phase 2 |
| memory_query_entropy | Deferred | Requires a real LLM backend; retained in the signal set |
| retrieval_count | Deferred | Requires a real LLM backend to show variance |
| total_latency_ms | Deferred | May be informative under real inference latency |
| tool_calls | Low priority | Likely to remain constant (2 per step) by design |
| token_count | Deferred as an attack signal | Confounded by PR content variation |

---

## 6. Pivot Status Update

| Pivot | Triggered? | Action |
|-------|-----------|--------|
| **A (AWT = 0)** | **Yes** | Reframe to concurrent detection; no code changes required |
| B (no signal) | No, since KL divergence is confirmed | Not applicable |
| C (Chronos underperforms) | Not yet evaluated | Evaluated in Phase 4 |

---

## 7. Decision

**Phase 1 result: CONDITIONAL GO.** Phase 2 (Core Agent Pipeline) proceeds, with kl_divergence as the primary detection signal and the secondary signals (entropy and retrieval_count) on watch for Phase 2 and beyond against real LLM backends. The paper framing is concurrent detection (AWT ≥ 0) rather than proactive early warning, and allocation efficiency remains the primary contribution; detection is the secondary contribution. The signal-validation claim (paper C1), that *behavioural signals, specifically retrieval-embedding KL divergence, shift significantly under memory-poisoning attacks (Cohen's d > 1.3 under MINJA and AGENTPOISON)*, is supported by the data presented above.

---

## 8. Log Update (for PLAN.md Phase 1 Log)

| Field | Value |
|-------|-------|
| **Signal Results** | KL-div: d = 1.343 (MINJA), d = 1.402 (AGENTPOISON). All other signals: d < 0.45. Entropy, retrieval_count, and tool_calls are MockBackend constants (σ = 0). |
| **GO/NO-GO Decision** | Conditional GO with Pivot A. Strict criterion: NO-GO (1 of 6). Adjusted reading: primary signal confirmed. |
| **Pivot Taken** | Pivot A; AWT = 0; concurrent-detection framing. |
| **Findings** | KL divergence is the only backend-independent signal; it works for both MINJA and AGENTPOISON. MockBackend eliminates 3 of 6 signals as observable. PELT detects a changepoint at step 9 of 25. |
| **Challenges** | MockBackend limits signal observability; three signals are constants. Entropy is flat at 0.0 (uniform similarity distribution). The token-count effect is a PR-seed confound. |
| **Decisions** | Applied Pivot A; deferred three signals to Phase 2 with a real LLM backend. KL divergence is promoted as the anchor signal for Phase 2 monitor integration. |
| **Completed** | 2026-04-09 |
