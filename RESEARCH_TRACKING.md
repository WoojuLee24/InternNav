# InternNav Research Tracking

_Last updated: 2026-04-27_
_Active branch: research/async-foundation (forked from true_async_background_thread @ 091350ac)_
_Original branch preserved: true_async_background_thread (DO NOT MODIFY)_
_Maintainer: kemal-mudie_

---

## Research Philosophy

**80% research innovation / 20% engineering implementation.**
Every code change must be motivated by a testable hypothesis. Experiments are the primary unit of progress.

---

## SYSTEM STATE (Verified 2026-04-27)

### Architecture (Correct Design)

```
TRUE Async Target (documented in docs/TRUE_ASYNC_DOCUMENTATION.md):
  HTTP → put(queue) [<1ms] → return cached output immediately
  Background Thread: continuous agent.step() → writes cache

Current code (research/async-foundation branch):
  HTTP → agent.step() [~300ms, BLOCKS] → then queues background → then checks cache → return
```

### Implementation Gap (One Structural Bug)

In `scripts/realworld/http_internvla_server_debug.py` at line 437:
```python
with agent_lock:
    dual_sys_output = agent.step(...)  # ← HTTP STILL BLOCKS HERE ~300ms
```
The cache check (lines 500–516) happens AFTER the synchronous step. The fix is to remove this block
from the HTTP handler and only use the cache + queue pattern.

### What the Measured Results Mean

All measurements in `optimization_tracking_async.md` used `ASYNC_BACKGROUND_INFERENCE=False`
(background thread disabled). The measured metrics (1.58–1.74 Hz, ~300ms) represent
**sync behavior on the async endpoint** — NOT true async decoupled performance.

### ASYNC_BACKGROUND_INFERENCE Flag (current: True in code)

- `True` (committed): Background thread exists but HTTP still blocks on agent.step() before cache check.
  Background and HTTP serialize on `agent_lock` — no actual parallelism.
- `False` (used in experiments): HTTP blocks on agent.step(), no background. Pure sync behavior.

### Enhanced Agent (Unconnected Research File)

`internnav/agent/internvla_n1_agent_enhanced.py` (651 lines, untracked) contains SOTA innovations:
- Action tokens (OpenVLA-style) — removes text parsing bottleneck
- Adaptive action chunking — generate N future waypoints at once
- Temporal caching (_s2_cache with TTL)
- Enhanced coordinate extraction with confidence scoring
- Entropy-based trajectory vs discrete selection
- Parallel decoding (PD-VLA style)
NOT yet connected to the server.

---

## Active Research Questions

| Question | Status | Priority |
|----------|--------|----------|
| Does removing sync step() from HTTP handler actually achieve <1ms response? | **CRITICAL — must verify first** | P0 |
| What is the true joint_req_hz when HTTP returns cache-first? | Open | P0 |
| Does trajectory_ratio stay ≥ 50% with true async (background-only) inference? | Open | P0 |
| Can enhanced agent (action tokens) replace text parsing to cut S2 latency? | Open | P1 |
| What is the optimal plan_step_gap for quality × speed? | Open | P1 |
| Can temporal caching (enhanced agent) reduce S2 invocations by 30-50%? | Open | P1 |
| Is trajectory_ratio a reliable proxy for SPL/SR on VLN benchmarks? | Open | P2 |
| Can S1 be trained to tolerate variable S2 update frequency? | Open | P2 |
| What is the minimum S1←S2 communication bandwidth needed? | Open | P2 |

---

## Proven Results (Verified by Experiment)

| Mode | joint_req_hz | trajectory_ratio | s2_latency_ms | Notes |
|------|-------------|-----------------|--------------|-------|
| SYNC (baseline, bag 073623) | 2.49 Hz | 63.3% | 313 ms | Ground truth |
| ASYNC endpoint, KV OFF (bag 073623) | 0.93 Hz | 74.0% | 605 ms | Sync behavior |
| ASYNC endpoint, KV ON (bag 073623) | 1.69 Hz | 56.2% | 300 ms | Sync behavior |
| ASYNC 3-bag aggregate | 1.63 Hz avg | 54.4% avg | 296 ms avg | Sync behavior |
| TRUE ASYNC (cache-first HTTP) | **NOT MEASURED** | — | — | Next experiment |

**KV Cache finding**: KV ON gives +81.7% throughput but -17.8pp trajectory_ratio vs KV OFF.
**Temperature finding**: temp=0.8 gives 63% trajectory_ratio vs 54% at temp=1.0 (same bag).

---

## Papers — Priority Queue

### Priority 1: Directly Applicable (Read This Sprint)

### [Speculative Decoding — Leviathan et al., 2023] — Relevance: 5/5
**Source**: arxiv:2211.17192
**Core idea**: Small draft model generates tokens speculatively, large model verifies. 2-3x speedup.
**Borrowable**: Apply to S2 VLM inference — small VLM drafts, InternVLA-N1 verifies.
**Target**: `http_internvla_server_debug.py` inference pipeline
**Status**: `[ ] queued`

### [A3C: Async Methods for Deep RL — Mnih et al., 2016] — Relevance: 4/5
**Source**: arxiv:1602.01783
**Core idea**: Proved async updates with shared model don't collapse quality if staleness is bounded.
**Borrowable**: Theoretical foundation for why async S2 updates are safe for S1. Staleness bound.
**Target**: Architecture justification + S1 tolerance to stale S2 outputs
**Status**: `[ ] queued`

### [NavGPT: Explicit Reasoning in VLN with LLMs] — Relevance: 4/5
**Source**: Search arxiv "NavGPT explicit reasoning VLN"
**Core idea**: LLM as explicit reasoning module — structured planning chain for VLN
**Borrowable**: Structured S2 output format → shorter token generation → lower s2_latency_ms
**Target**: `internvla_n1_agent.py` S2 output parsing, prompt engineering
**Status**: `[ ] queued`

### [OpenVLA: Open-Source Vision-Language-Action Model] — Relevance: 5/5
**Source**: arxiv:2406.09246
**Core idea**: Action tokens instead of text coordinates — eliminates regex parsing bottleneck
**Borrowable**: The enhanced agent already implements this concept. Connect to server.
**Target**: `internnav/agent/internvla_n1_agent_enhanced.py` → `http_internvla_server_debug.py`
**Status**: `[x] analyzed` (implemented in enhanced agent, not connected)

### [PD-VLA: Parallel Decoding for VLA] — Relevance: 4/5
**Source**: Search arxiv "parallel decoding vision-language-action"
**Core idea**: Decode multiple action tokens in parallel (bidirectional attention for actions)
**Borrowable**: Already sketched in enhanced agent — verify and connect
**Target**: `internvla_n1_agent_enhanced.py`
**Status**: `[ ] reading`

### Priority 2: Background Research

### [DistServe — Zhong et al., OSDI 2024] — Relevance: 3/5
**Source**: Referenced in TRUE_ASYNC_DOCUMENTATION.md
**Core idea**: Disaggregate prefill/decode phases for LLM serving
**Connection**: Motivates our S1/S2 decoupling approach
**Status**: `[ ] queued`

### [AAC: Adaptive Action Chunking — 2024] — Relevance: 4/5
**Source**: Search arxiv "adaptive action chunking entropy"
**Core idea**: Entropy-based selection of action chunk size
**Borrowable**: Implemented in enhanced agent — verify design is correct
**Status**: `[ ] reading`

---

## Research Ideas — Verified / Unverified

### Idea 1: True Async HTTP Handler Fix [ENGINEERING, ~2 hours]
**Status**: Designed, not implemented
**What**: Remove `agent.step()` from `/eval_dual_async`, use cache-first pattern
**Expected**: joint_latency → <1ms, joint_req_hz → client-rate limited (~20+ Hz)
**Risk**: Low — fix is isolated, background thread already correct
**Next**: Implement, run 3-bag protocol, compare to sync baseline

### Idea 2: S2 Temperature Optimization [VALIDATED RESEARCH]
**Status**: Partially validated (temp=0.8 → 63% traj_ratio vs 54% at temp=1.0)
**What**: Lower temperature → more deterministic → higher trajectory ratio
**Expected at temp=0.7**: >65%? (need to run)
**Next**: Sweep temp ∈ {0.7, 0.75, 0.8, 0.85} and measure trajectory_ratio

### Idea 3: Enhanced Agent Integration [RESEARCH, ~1 week]
**Status**: Enhanced agent built but unconnected
**What**: Connect `internvla_n1_agent_enhanced.py` to server instead of base agent
**Expected**: Faster coordinate parsing, possible trajectory_ratio improvement
**Files**: `internvla_n1_agent_enhanced.py` → `http_internvla_server_debug.py`

### Idea 4: Temporal S2 Caching [RESEARCH, 3-4 days]
**Status**: Designed in enhanced agent
**What**: If observation cosine similarity > threshold, reuse previous S2 output
**Expected**: 30-50% reduction in S2 invocations, maintained quality
**Risk**: Quality degrades if threshold too high

### Idea 5: Adaptive plan_step_gap [RESEARCH, 1 week]
**Status**: Designed
**What**: Dynamic S2 frequency based on navigation difficulty (uncertainty proxy)
**Expected**: 20-30% fewer S2 calls without trajectory_ratio drop

### Idea 6: Speculative S2 Pre-fetch [NOVEL, 2 weeks]
**Status**: Idea only
**What**: Extrapolate robot state 500ms forward, send to S2 as next query
**Expected**: Eliminate S2 wait from S1 perspective
**Risk**: Prediction errors reduce S2 output quality

---

## Literature Review (OpenCode Session, 2026-05-06)

### Key Papers by Category

#### Async Architecture & Decoupled Inference
1. **DistServe** (Zhong et al., OSDI 2024) — Disaggregate prefill/decode for LLMs. 2-3x throughput gain. Relevance: 5/5.
2. **EPD-Serve** (arXiv:2601.11590, 2025) — Encode/prefill/decode disaggregation for VLMs. 57-69% throughput gain. Relevance: 4/5.
3. **DynaServe** (arXiv:2504.09285, 2025) — Elastic tandem execution with dynamic request splitting. Relevance: 4/5.
4. **DuetServe** (arXiv:2511.04791, 2025) — SM-level partitioning for prefill/decode within single GPU. Relevance: 3/5.
5. **NVIDIA Triton Decoupled Models** — Bi-directional streaming RPC for async responses. Relevance: 4/5.

#### Speculative Decoding for VLAs
1. **Spec-VLA** (arXiv:2507.22424, 2025) — Speculative decoding for VLAs with relaxed acceptance. 1.42x speedup, 44% acceptance boost. Relevance: 5/5.
2. **HeiSD** (arXiv:2603.17573, 2026) — Hybrid speculative decoding with kinematic awareness. 2.45x speedup in sim, 2.06-2.41x real-world. Relevance: 5/5.
3. **KERV** (arXiv:2603.01581, 2026) — Kinematic-rectified speculative decoding. 27-37% acceleration with no SR loss. Relevance: 5/5.
4. **SpecVLM** (arXiv:2509.11815, 2025) — EAGLE-2-style for VLMs. 1.5-2.9x end-to-end speedup. Relevance: 5/5.
5. **ViSpec** (arXiv:2509.15235, 2025) — Vision-aware speculative decoding. First substantial VLM speedup. Relevance: 5/5.
6. **Spec-LLaVA** (arXiv:2509.11961) — Dynamic tree-based verification for VLMs. 3.28x speedup. Relevance: 4/5.

#### Parallel Decoding & Action Representation
1. **PD-VLA** (arXiv:2503.02310, 2025) — Parallel decoding for VLAs with action chunking. 2.52x execution frequency. Relevance: 5/5.
2. **OpenVLA** (arXiv:2406.09246, 2024) — Action tokens instead of text. Eliminates regex parsing. Relevance: 5/5.
3. **RD-VLA** (arXiv:2602.07845, 2026) — Recurrent-depth VLA with implicit test-time compute. 93% success on LIBERO. Relevance: 5/5.

#### KV Cache Compression
1. **VL-Cache** (arXiv:2410.23317, 2024) — Modality-aware KV compression. 10% cache retains accuracy, 2.33x speedup. Relevance: 5/5.
2. **AirCache** (arXiv:2503.23956, 2025) — Inter-modal relevance KV compression. 29-66% latency reduction. Relevance: 5/5.
3. **PrefixKV** (arXiv:2412.03409, 2025) — Adaptive prefix KV cache with binary search. Relevance: 4/5.
4. **PureKV** (arXiv:2510.25600, 2025) — Spatial-temporal sparse attention. 5x KV compression, 3.16x prefill speedup. Relevance: 4/5.
5. **VLN-Cache** (arXiv:2603.07080, 2026) — Visual/temporal dynamics-aware caching for VLN. 1.52x speedup. Relevance: 5/5.
6. **LightVLM** (arXiv:2509.00419, 2025) — Pyramid token merging + KV compression. 2.02x throughput, 3.65x prefill speedup. Relevance: 5/5.
7. **SparseVILA** — Decoupled visual sparsity with query-aware KV retrieval. Relevance: 4/5.

#### Early Exit & Adaptive Scheduling
1. **DART** (arXiv:2603.12269, 2026) — Input-difficulty-aware adaptive threshold. 3.3x speedup, 5.1x lower energy. Relevance: 4/5.
2. **DeeAD** (arXiv:2511.20720, 2025) — Dynamic early exit for VLA. Terminates when deviation < threshold. Relevance: 5/5.
3. **FREE** — Adversarial training for VLM early exit. 1.51x speedup. Relevance: 4/5.
4. **A-ViT** (arXiv:2112.07658, 2022) — Adaptive token computation for vision transformers. Relevance: 4/5.

#### Knowledge Distillation
1. **SpecVLM** (arXiv:2509.11815, 2025) — Online-logit distillation for draft model. No offline corpus needed. Relevance: 5/5.
2. **Hinton et al., 2015** — Classic knowledge distillation framework. Relevance: 3/5.
3. **FitNets** (Romero et al., 2015) — Hint-based distillation with intermediate features. Relevance: 3/5.

### Borrowable Techniques Mapped to InternNav

| Technique | Maps to | Expected Gain | Priority |
|-----------|---------|---------------|----------|
| Spec-VLA relaxed acceptance | `internvla_n1_agent.py:verify_draft()` | +25-44% acceptance | A |
| PD-VLA parallel decoding | `internvla_n1_policy.py:generate()` | 2.52x S2 speedup | A |
| VL-Cache KV compression | `internvla_n1_policy.py:forward()` | 30-50% latency reduction | A |
| HeiSD kinematic draft | `internvla_n1_agent.py:step()` | 27-37% acceleration | A |
| VLN-Cache temporal reuse | `http_internvla_server_debug.py` | 1.52x speedup | A |
| DistServe disaggregation | `internvla_n1_agent.py` (split S1/S2) | S1≥30Hz | B |
| DeeAD early exit | `internvla_n1_policy.py` (add exit heads) | 30-50% S2 reduction | B |
| RD-VLA recurrent state | `internvla_n1_policy.py` (add hidden state) | Adaptive compute | A |

### Next Actions
- [ ] Read PLAN.md Gate 0 fix: remove agent.step() from HTTP handler
- [ ] Implement I-001 (Gate 0 fix) → verify with 3-bag protocol
- [ ] After Gate 0 passes: implement I-021 (Spec-VLA) or I-031 (PD-VLA) for Gate 2

---

## Session Notes
_Updated by /daily and session-stop hook_
