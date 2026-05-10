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

## Bibliography (ArXiv Index)

**Speculative Decoding / VLA**
- Spec-VLA — arXiv:2507.22424
- HeiSD — arXiv:2603.17573
- KERV — arXiv:2603.01581
- SpecVLM — arXiv:2509.11815
- ViSpec — arXiv:2509.15235
- Spec-LLaVA — arXiv:2509.11961

**Parallel Decoding / Action Tokens**
- PD-VLA — arXiv:2503.02310
- OpenVLA — arXiv:2406.09246
- RD-VLA — arXiv:2602.07845

**KV Cache / Memory**
- VL-Cache — arXiv:2410.23317
- AirCache — arXiv:2503.23956
- PrefixKV — arXiv:2412.03409
- PureKV — arXiv:2510.25600
- VLN-Cache — arXiv:2603.07080
- LightVLM — arXiv:2509.00419

**Adaptive Scheduling / Early Exit**
- DART — arXiv:2603.12269
- DeeAD — arXiv:2511.20720
- A-ViT — arXiv:2112.07658

**Disaggregated Inference**
- DistServe — OSDI 2024
- EPD-Serve — arXiv:2601.11590

---

## BibTeX (Key Papers)

```
@article{specvla2025,
  title={Spec-VLA: Speculative Decoding for Vision-Language-Action Models},
  author={Wang, et al.},
  journal={arXiv preprint arXiv:2507.22424},
  year={2025}
}

@article{heisd2026,
  title={HeiSD: Hybrid Speculative Decoding for Embodied Agents},
  author={Zheng, et al.},
  journal={arXiv preprint arXiv:2603.17573},
  year={2026}
}

@article{kerv2026,
  title={KERV: Kinematic-Rectified Speculative Decoding for VLAs},
  author={Zheng, et al.},
  journal={arXiv preprint arXiv:2603.01581},
  year={2026}
}

@article{specvlm2025,
  title={SpecVLM: Speculative Decoding for Vision-Language Models},
  author={Liu, et al.},
  journal={arXiv preprint arXiv:2509.11815},
  year={2025}
}

@article{vispec2025,
  title={ViSpec: Vision-Aware Speculative Decoding for VLMs},
  author={Chen, et al.},
  journal={arXiv preprint arXiv:2509.15235},
  year={2025}
}

@article{specllava2025,
  title={Spec-LLaVA: Tree-Based Speculative Decoding for VLMs},
  author={Zhao, et al.},
  journal={arXiv preprint arXiv:2509.11961},
  year={2025}
}

@article{pdvla2025,
  title={PD-VLA: Parallel Decoding for Vision-Language-Action Models},
  author={Song, et al.},
  journal={arXiv preprint arXiv:2503.02310},
  year={2025}
}

@article{openvla2024,
  title={OpenVLA: Open-Source Vision-Language-Action Model},
  author={Kim, et al.},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}

@article{rdvla2026,
  title={RD-VLA: Recurrent-Depth Vision-Language-Action Model},
  author={Zhou, et al.},
  journal={arXiv preprint arXiv:2602.07845},
  year={2026}
}

@article{vlcache2024,
  title={VL-Cache: Modality-Aware KV Cache Compression for VLMs},
  author={Tu, et al.},
  journal={arXiv preprint arXiv:2410.23317},
  year={2024}
}

@article{aircache2025,
  title={AirCache: Inter-Modal Relevance KV Compression},
  author={Li, et al.},
  journal={arXiv preprint arXiv:2503.23956},
  year={2025}
}

@article{prefixkv2025,
  title={PrefixKV: Adaptive Prefix KV Cache Allocation},
  author={Zhang, et al.},
  journal={arXiv preprint arXiv:2412.03409},
  year={2025}
}

@article{purekv2025,
  title={PureKV: Spatial-Temporal Sparse Attention for KV Compression},
  author={Wang, et al.},
  journal={arXiv preprint arXiv:2510.25600},
  year={2025}
}

@article{vlncache2026,
  title={VLN-Cache: Visual Dynamics-Aware Caching for VLN},
  author={Liu, et al.},
  journal={arXiv preprint arXiv:2603.07080},
  year={2026}
}

@article{lightvlm2025,
  title={LightVLM: Pyramid Token Merging for VLMs},
  author={Xu, et al.},
  journal={arXiv preprint arXiv:2509.00419},
  year={2025}
}

@article{dart2026,
  title={DART: Difficulty-Aware Adaptive Thresholding},
  author={Chen, et al.},
  journal={arXiv preprint arXiv:2603.12269},
  year={2026}
}

@article{deead2025,
  title={DeeAD: Dynamic Early Exit for VLA},
  author={Sun, et al.},
  journal={arXiv preprint arXiv:2511.20720},
  year={2025}
}

@article{avit2022,
  title={A-ViT: Adaptive Token Computation for Vision Transformers},
  author={Yin, et al.},
  journal={arXiv preprint arXiv:2112.07658},
  year={2022}
}

@article{distserve2024,
  title={DistServe: Disaggregating Prefill and Decode for LLM Serving},
  author={Zhong, et al.},
  journal={Proceedings of OSDI},
  year={2024}
}

@article{epdserve2025,
  title={EPD-Serve: Encode/Prefill/Decode Disaggregation for VLMs},
  author={Zhang, et al.},
  journal={arXiv preprint arXiv:2601.11590},
  year={2025}
}

@article{temperature2017,
  title={On Calibration of Modern Neural Networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  journal={Proceedings of ICML},
  year={2017}
}

@article{mcdropout2016,
  title={Dropout as a Bayesian Approximation: Representing Model Uncertainty},
  author={Gal, Yarin and Ghahramani, Zoubin},
  journal={Proceedings of ICML},
  year={2016}
}

@article{a3c2016,
  title={Asynchronous Methods for Deep Reinforcement Learning},
  author={Mnih, Volodymyr and others},
  journal={Proceedings of ICML},
  year={2016}
}
```

---

## Session Notes
_Updated by /daily and session-stop hook_

---

## Post-Gate Research Plan (Async-First Architectures)

_Stage this plan for after Gate-complete experiments. Focus: new, unique async-first designs that exploit true async achievements and exceed current VLA baselines._

**New system repo (proposed):** `ChronoNav` — async-first navigation system with latency-aware semantics and decision-point safety.
**Benchmarks/repos to compare:** `InternNav`, `TIC-VLA` (https://github.com/ucla-mobility/TIC-VLA), DynaNav suite (from TIC-VLA).

### ChronoNav — Theory & Foundations (Synthesis)

**Core hypothesis**: Time-shifted semantic reasoning is inevitable in real-world VLN. If the control policy is explicitly conditioned on delay/staleness and decision points are protected, then asynchronous deployment maintains safety and quality with large compute savings.

**Mathematical pillars**
- **Delay-aware control**: with delay $\Delta$, action deviation is bounded by Lipschitz constants; derive safe delay budgets.
- **Event-triggered control**: trigger updates only on decision points; guarantees recall given detector TPR.
- **Bounded staleness**: cache reuse is safe when staleness error remains within tolerance; enforce max-hold time.
- **Calibration & uncertainty**: confidence gating guarantees safe fallback under uncertainty.

**ChronoNav modules (with theory hooks)**

1) **ChronoCore (Latency-Aware Control)**
   - Theory: Lipschitz delay bound; semantic drift bound.
   - Hypothesis: explicit delay embedding reduces action drift by >50% under 1–2s lag.

2) **DecisionPulse (Decision-Point Detection)**
   - Theory: event-triggered control with dwell time; recall bound.
   - Hypothesis: decision recall >= 95% with S2 usage <= 20%.

3) **StaleGuard (Cache-Conditional Confidence)**
   - Theory: bounded staleness + calibrated confidence.
   - Hypothesis: fresh action rate >= 50% while maintaining 80–90% S2 reduction.

4) **FlowFrame (Temporal Flow Planning)**
   - Theory: flow horizon error accumulation $H\epsilon$.
   - Hypothesis: flow reuse improves stability while bounding drift.

5) **Q-Gate MoE (Semantic Expert Routing)**
   - Theory: calibrated gating; entropy regularization prevents collapse.
   - Hypothesis: discrete-action recall improves under high delay.

**Key evidence anchors (to verify)**
- TIC-VLA (2026): latency-aware VLA baseline.
- Event-triggered inference/control: arXiv:2109.05601, arXiv:1901.07806.
- Decision-point detection VLN: arXiv:2007.00696.
- Bounded staleness caches: arXiv:1806.10254.
- Calibration: Guo et al. 2017.

### ChronoNav Module Formulations (Deep)

**1) ChronoCore (Latency-Aware Control)**
Let $o_t$ be observation, $s_{t-\Delta}$ delayed semantic state, $\tau$ staleness. Encode:
$$z_t = \text{ViT}(o_t), \quad h_t = \text{Qwen}(s_{t-\Delta}), \quad e_t = E(\Delta, \tau)$$
Policy:
$$a_t = \pi_\theta([z_t, h_t, e_t, u_{t-k:t-1}])$$
Loss:
$$\mathcal{L}_{cc} = \|a_t - a_t^*\|_2^2 + \lambda_1 \text{KL}(\pi_\theta \| \pi_{fresh}) + \lambda_2 \phi(\tau)$$
**Proof**: With $\|h_t - h_{t-\Delta}\| \le c\Delta$ and Lipschitz $L_h$, $\|a_t - a_t^{fresh}\| \le L_h c\Delta$.

**2) DecisionPulse (Decision-Point Detection)**
Decision score:
$$d_t = \sigma(g_\phi(z_t, u_{t-1}))$$
Gate:
$$\text{runS2}(t) = \mathbb{1}[d_t > \delta \lor \tau_t > \tau_{max}]$$
Loss:
$$\mathcal{L}_{dp} = \text{BCE}(d_t, y_t^{disc}) + \beta \text{KL}(p_{disc} \| p_{fresh,disc})$$
**Proof**: If detector TPR $\ge 1-\epsilon$, then decision recall $R_d \ge 1-\epsilon$ (misses only from detector error).

**3) StaleGuard (Cache-Conditional Confidence)**
Confidence head:
$$c_t = \sigma(r_\eta(z_t, h_t))$$
Cache decision:
$$\text{useCache}(t) = \mathbb{1}[c_t - \gamma\tau_t > \kappa]$$
Action-specific TTL:
$$\tau_{max} = \tau_{traj}\mathbb{1}[a_t\in traj] + \tau_{disc}\mathbb{1}[a_t\in disc]$$
**Proof**: Since $c_t$ bounded and $\tau_t$ increases, $c_t-\gamma\tau_t$ crosses $\kappa$ in finite time → forced refresh.

**4) FlowFrame (Temporal Flow Planning)**
Horizon flow:
$$F_t = [u_t,\dots,u_{t+H}] = f_\psi(z_t, h_t, e_t)$$
Loss with dynamics penalty:
$$\mathcal{L}_{ff} = \sum_{k=0}^H \|u_{t+k}-u^*_{t+k}\|^2 + \lambda \Psi(F_t)$$
**Proof**: Per-step error $\epsilon$ implies total deviation $\le H\epsilon$ (triangle inequality).

**5) Q-Gate MoE (Semantic Expert Routing)**
Experts $f_i$, gate $\alpha=\text{softmax}(g_\psi(x,\Delta,\tau))$:
$$a_t = \sum_i \alpha_i f_i(z_t,h_t)$$
Loss:
$$\mathcal{L}_{moe} = \mathcal{L}_{control} + \lambda H(\alpha) + \mu\sum_i\|f_i-\bar{f}\|^2$$
**Proof**: Temperature scaling yields calibrated gate; safe-stop triggers at bounded false-negative rate.

**6) AMSC (Action-Memory Safety Controller)**
Safety projection:
$$a_t = \arg\min_{a\in\mathcal{A}_{safe}} \|a-\hat{a}_t\|$$
**Proof**: Projection guarantees $a_t \in \mathcal{A}_{safe}$ by construction.

**7) L-FEED (Lag-Aware Feedback Distillation)**
Distill delayed S2:
$$\mathcal{L}_{distill} = \|a^{S1}_t - a^{S2}_{t-\Delta}\|^2 + \lambda \text{KL}(p^{S1} \| p^{S2})$$
**Proof**: If S2 is Lipschitz in delay, minimization yields bounded S1 deviation proportional to training error.

**8) LARA (Latency-Adaptive Reasoning Allocation)**
Depth selection:
$$d_t = \max\{d: \ell(d) \le B\}$$
**Proof**: Guarantees per-step latency $\ell(d_t) \le B$ with maximal depth under constraint.

---

### ChronoNav System Block Diagram (Text Spec)

```
Input: RGB_t, Depth_t, Odom_t, Instruction
  ├── Fast Perception: ViT(RGB_t, Depth_t) → z_t
  ├── Slow Semantics: Qwen(Instruction, Memory_{t-Δ}) → h_{t-Δ}
  ├── Latency Embedder: E(Δ, τ) → e_t
  ├── DecisionPulse: d_t = σ(g(z_t, Odom_t))
  ├── StaleGuard: cache gate uses c_t - γτ_t
  ├── Q-Gate MoE: route {Trajectory, Discrete, Safe-Stop} experts
  ├── FlowFrame: optional horizon flow predictor
  ├── AMSC: safety projection on final action
Output: action_t, confidence_t, staleness_penalty_t
```

**Async contract**: S2 updates are decoupled; S1 consumes cached S2 outputs with explicit staleness/latency metadata. DecisionPulse can override cache to request fresh S2.

**Design note (external reference)**: Conceptual inspiration for explicit semantic/control separation can be compared against TIC-VLA's framework diagram, but ChronoNav extends it with decision-point gating, confidence-staleness control, and safety projection to prevent action suppression.

---

## ChronoNav Architecture (Expanded Blocks + Signals)

**Block-level dataflow**
```
RGB_t, Depth_t ──► FastPerception (ViT) ──► z_t ─┐
Instruction ──► SlowSemantics (Qwen) ──► h_{t-Δ} ─┼─► ChronoCore ─► â_t
Delay/Staleness ──► LatencyEmbed E(Δ,τ) ─────────┘
z_t, u_{t-1} ──► DecisionPulse ──► d_t ──► StaleGuard (g_t)
z_t,h_{t-Δ} ──► Q-Gate MoE ──► â_t (traj/disc/safe)
z_t,h_{t-Δ},e_t ──► FlowFrame ──► F_t (optional constraint)
â_t,F_t ──► AMSC ──► a_t
```

**Signal definitions**
- $z_t$: fast visual embedding
- $h_{t-\Delta}$: delayed semantic embedding
- $e_t$: latency/staleness embedding
- $d_t$: decision probability
- $g_t$: cache gate decision
- $F_t$: horizon flow vector

**Key differences vs TIC-VLA**
- Adds **DecisionPulse** to guarantee discrete-action recall.
- Adds **StaleGuard** to bound replay bias and force refresh.
- Adds **AMSC** safety projection to prevent unsafe overrides.

---

## ChronoNav Module Parameters (Placeholder)

| Module | Params (M) | Notes |
|--------|------------|-------|
| FastPerception (ViT) | TBD | choose ViT-B/ViT-L |
| SlowSemantics (Qwen) | TBD | Qwen-1.8B/7B |
| ChronoCore | TBD | control transformer |
| DecisionPulse | TBD | small MLP |
| StaleGuard | TBD | calibration head |
| Q-Gate MoE | TBD | K experts |
| FlowFrame | TBD | flow head |
| AMSC | 0 | projection layer |

---

## ChronoNav Training Schedule (Draft)

| Stage | Epochs | Loss Weights | Notes |
|-------|--------|--------------|-------|
| S1 | 5–10 | L_cc=1.0, L_dp=1.0 | train ChronoCore + DecisionPulse |
| S2 | 5–10 | L_moe=0.5, L_ff=0.5 | add MoE + FlowFrame |
| S3 | 3–5 | L_distill=0.5 | lag-aware distillation |
| S4 | 2–3 | all | joint fine-tuning |

---

## ChronoNav Evaluation Protocol (Post-Gate)

_Goal: prove ChronoNav fixes action suppression and replay bias while keeping async compute savings._

### Offline Metrics (Primary)
1. **Decision-point recall**: action-triggered frames captured.
2. **Action KL divergence**: cached vs fresh action distribution.
3. **Trajectory divergence**: DTW/Fréchet on cmd_vel sequences.
4. **Replay-bias index**: served vs fresh trajectory ratios.
5. **Counterfactual delta**: mean L2 deviation to fresh-every-frame.

### Closed-Loop Metrics (Secondary)
1. **SR/SPL** on R2R/RxR/REVERIE/ObjectNav (sim).
2. **Real-world stability** under multi-second reasoning lag.

### Gate Criteria
- Decision recall >= 95% (no action suppression).
- Action KL <= 0.2 vs fresh baseline.
- Replay bias <= 10pp.
- S2 usage <= 25% while preserving safety.

---

## ChronoNav Architecture Spec (Full Node Graph)

_A clear node/edge specification for the full async-first system._

**Nodes**
1. **N1: FastPerception** — ViT(RGB, Depth) → $z_t$
2. **N2: SlowSemantics** — Qwen(Instruction, memory) → $h_{t-\Delta}$
3. **N3: LatencyEmbed** — $E(\Delta, \tau)$ → $e_t$
4. **N4: DecisionPulse** — $d_t = \sigma(g(z_t, u_{t-1}))$
5. **N5: StaleGuard** — gate $g_t = \mathbb{1}[c_t-\gamma\tau_t>\kappa]$
6. **N6: ChronoCore** — control transformer $\pi_\theta([z_t,h_{t-\Delta},e_t,u_{t-k:t-1}])$
7. **N7: Q-Gate MoE** — expert routing for action type
8. **N8: FlowFrame** — horizon flow predictor $F_t$
9. **N9: AMSC** — safety projection onto $\mathcal{A}_{safe}$
10. **N10: Output** — $(a_t, c_t, \tau_t)$

**Edges**
- N1 → N4, N6, N7, N8
- N2 → N6, N7, N8
- N3 → N6, N7
- N4 → N5 (decision override)
- N5 → N6 (cache vs fresh)
- N6 → N7 → N9 → N10
- N8 → N9 (optional flow constraints)

**Node equations**
- N6 (ChronoCore): $a_t = \pi_\theta([z_t, h_{t-\Delta}, e_t, u_{t-k:t-1}])$
- N7 (MoE): $a_t = \sum_i \alpha_i f_i(z_t, h_{t-\Delta})$
- N8 (FlowFrame): $F_t = f_\psi(z_t, h_{t-\Delta}, e_t)$
- N9 (AMSC): $a_t = \arg\min_{a\in\mathcal{A}_{safe}} \|a-\hat{a}_t\|$

---

## ChronoNav Training Objectives (Unified)

**Total loss**
$$\mathcal{L} = \mathcal{L}_{cc} + \lambda_{dp}\mathcal{L}_{dp} + \lambda_{moe}\mathcal{L}_{moe} + \lambda_{ff}\mathcal{L}_{ff} + \lambda_{distill}\mathcal{L}_{distill}$$

**Components**
- $\mathcal{L}_{cc}$: latency-aware control loss (ChronoCore)
- $\mathcal{L}_{dp}$: decision-point loss (DecisionPulse)
- $\mathcal{L}_{moe}$: expert diversity + routing (Q-Gate MoE)
- $\mathcal{L}_{ff}$: flow horizon loss (FlowFrame)
- $\mathcal{L}_{distill}$: lag-aware distillation (L-FEED)

**Optimization**
- Stage 1: train ChronoCore + DecisionPulse on fresh data
- Stage 2: inject delay distributions for latency-consistent training
- Stage 3: joint fine-tuning with StaleGuard + MoE + AMSC

---

## ChronoNav Ablation Grid (Expected Effects)

| Ablation | Removed Node | Expected Effect | Metrics Impact |
|----------|--------------|----------------|----------------|
| ABL-1 | DecisionPulse | decision recall drops | decision_recall ↓, action_KL ↑ |
| ABL-2 | StaleGuard | replay bias increases | replay_bias ↑ |
| ABL-3 | Q-Gate MoE | action diversity drops | action_diversity ↓ |
| ABL-4 | FlowFrame | stability under lag drops | DTW/Fréchet ↑ |
| ABL-5 | AMSC | safety violations rise | safety_violations ↑ |
| ABL-6 | LatencyEmbed | delay robustness drops | action drift ↑ |

---

## Mathematical Guarantees Summary (Node → Bound)

| Node | Guarantee | Bound |
|------|-----------|-------|
| ChronoCore | delay-stability | $\|a_t - a_t^{fresh}\| \le L_h c\Delta$ |
| DecisionPulse | decision recall | $R_d \ge 1-\epsilon$ |
| StaleGuard | forced refresh | $\exists T: c_t-\gamma\tau_t \le \kappa$ |
| FlowFrame | horizon drift | $\le H\epsilon$ |
| Q-Gate MoE | calibrated fallback | bounded FN under calibrated gate |
| AMSC | safety constraints | $a_t \in \mathcal{A}_{safe}$ |

---

## ChronoNav Inference Scheduling Policy

**Policy logic (pseudo)**
```
if DecisionPulse(d_t):
    run S2 now
elif StaleGuard(c_t, τ_t):
    use cached S2
else:
    use ChronoCore + Q-Gate MoE
if τ_t > τ_max: force S2
```

**Scheduling guarantees**
- Decision frames are always re-evaluated unless detector fails.
- Cache reuse has a bounded staleness due to τ_max.
- Compute use stays below target by S2 usage cap.

---

## ChronoNav Data + Logging Schema (Offline Metrics)

**Required fields per frame**
- timestamp, frame_id
- action_type (traj/discrete)
- action_vector (cmd_vel)
- decision_score d_t
- confidence c_t
- staleness τ_t
- cache_hit (bool)
- fresh_run (bool)
- latency Δ (ms)

**Derived metrics**
- decision_recall, replay_bias
- action_KL, DTW/Fréchet
- counterfactual_delta

---

## ChronoNav Deployment Checklist (Real-World Safety)

1. **Decision safety**: decision_recall >= 95% on offline logs.
2. **Replay bias**: replay_bias <= 10pp.
3. **Action drift**: DTW/Fréchet <= threshold.
4. **Latency budget**: SLO met >= 99%.
5. **Safety projection**: AMSC enabled; no violations in sim.
6. **Fallback**: safe-stop expert enabled and calibrated.

---

## ChronoNav Experiment Chronology Template

| Date | Milestone | Change | Offline Metrics | Closed-Loop Result | Notes |
|------|-----------|--------|-----------------|--------------------|-------|
| TBD | M1 | ChronoCore prototype | drift/recall | N/A | |
| TBD | M2 | DecisionPulse gate | decision_recall | N/A | |
| TBD | M3 | StaleGuard | replay_bias | N/A | |
| TBD | M4 | Q-Gate MoE | action_KL | N/A | |
| TBD | M5 | FlowFrame | DTW/Fréchet | N/A | |
| TBD | M6 | System integration | all metrics | SR/SPL | |

---

## ChronoNav Interface Contracts (Module I/O)

**ChronoCore**
- Input: $z_t, h_{t-\Delta}, e_t, u_{t-k:t-1}$
- Output: $a_t, c_t$

**DecisionPulse**
- Input: $z_t, u_{t-1}$
- Output: $d_t$ (decision probability)

**StaleGuard**
- Input: $c_t, \tau_t, d_t$
- Output: cache gate $g_t$

**Q-Gate MoE**
- Input: $z_t, h_{t-\Delta}, e_t$
- Output: mixture weights $\alpha$ and action $a_t$

**FlowFrame**
- Input: $z_t, h_{t-\Delta}, e_t$
- Output: horizon flow $F_t$

**AMSC**
- Input: $\hat{a}_t, \mathcal{A}_{safe}$
- Output: safe action $a_t$

---

## Offline Metric Equations (Explicit)

**Decision recall**
$$R_d = \frac{\#(\text{decision frames where S2 ran})}{\#(\text{decision frames total})}$$

**Replay bias**
$$\text{replay\_bias} = \text{served\_traj\_ratio} - \text{fresh\_traj\_ratio}$$

**Action KL**
$$\text{KL}(P\|Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$$

**DTW**
$$\text{DTW}(X,Y) = \min_w \sqrt{\sum_k \|x_{w_k} - y_{w_k}\|^2}$$

**Fréchet distance**
$$d_F(X,Y) = \inf_{\alpha,\beta} \max_{t\in[0,1]} \|X(\alpha(t)) - Y(\beta(t))\|$$

**Counterfactual delta**
$$\Delta = \frac{1}{T} \sum_{t=1}^T \|a^{cached}_t - a^{fresh}_t\|_2$$

---

## ChronoNav Data Pipeline (Training + Evaluation)

**Training data flow**
1. Collect async logs with full fields (action_type, confidence, staleness, delay).
2. Build decision-point labels from discrete action transitions.
3. Construct delayed semantic pairs $(o_t, s_{t-\Delta})$ for ChronoCore.
4. Train DecisionPulse (BCE), ChronoCore (control loss), StaleGuard (calibration).
5. Fine-tune MoE experts on action subsets.
6. Distill S2 to S1 with delay-labeled pairs.

**Evaluation data flow**
1. Run offline logs through ChronoNav with cache enabled.
2. Compute offline metrics (decision_recall, action_KL, DTW/Fréchet, replay_bias).
3. If offline gates pass, run closed-loop in sim (R2R/RxR/REVERIE/ObjectNav).
4. If sim passes, run controlled real-world tests with safety projection.

---

## ChronoNav Config Schema (Draft)

**Latency/Cache**
- `max_hold_frames` (int, default: 10)
- `staleness_penalty_gamma` (float, default: 0.1)
- `decision_threshold` (float, default: 0.6)
- `cache_confidence_threshold` (float, default: 0.5)

**DecisionPulse**
- `decision_model` (str, default: "decision_pulse_v1")
- `decision_recall_target` (float, default: 0.95)

**ChronoCore**
- `latency_embed_dim` (int, default: 16)
- `delay_distribution` (str, default: "uniform[0,2s]")

**MoE**
- `experts` (list, default: ["traj","disc","safe"])
- `moe_entropy_weight` (float, default: 0.01)

**FlowFrame**
- `horizon` (int, default: 10)
- `flow_penalty_lambda` (float, default: 0.1)

**Safety**
- `safe_action_set` (str, default: "stop,turn,slow")
- `safety_projection` (bool, default: true)

---

## ChronoNav Model Card (Draft)

**Intended Use**: Async-first navigation with delayed semantics and decision-point safety.
**Not for**: Unsupervised deployment in unstructured human environments without safety projection.
**Core Strengths**: Delay robustness, decision-point recall, replay bias control.
**Known Limitations**: Requires calibrated confidence; depends on decision-point detector quality.
**Safety Features**: AMSC safety projection, safe-stop fallback, decision-point overrides.

---

## Benchmarking Checklist (ChronoNav)

| Benchmark | What It Measures | Offline Proxy | Gate | Notes |
|-----------|------------------|---------------|------|-------|
| R2R | SR/SPL | action_KL + DTW | pass offline first | sim baseline |
| RxR | multilingual VLN | decision_recall | pass offline first | language stress |
| REVERIE | object grounding | action_KL | pass offline first | object cues |
| ObjectNav | object goals | decision_recall | pass offline first | goal cues |

---

## ChronoNav Failure Taxonomy

1. **Action Suppression** — no discrete actions in fresh-only outputs.
2. **Replay Bias** — inflated trajectory_ratio due to cache reuse.
3. **Delay Drift** — cmd_vel divergence grows with latency.
4. **Over-Triggering** — S2 usage spikes beyond budget.
5. **Calibration Failure** — confidence overestimates correctness.
6. **Safety Violation** — projected safe action not applied.

---

## Deployment Readiness Rubric

| Criterion | Threshold | Status |
|----------|-----------|--------|
| Decision recall | >= 95% | TBD |
| Action KL | <= 0.2 | TBD |
| Replay bias | <= 10pp | TBD |
| DTW/Fréchet | <= threshold | TBD |
| S2 usage | <= 25% | TBD |
| Safety violations | 0 | TBD |

---

## ChronoNav Research Questions (Post-Gate)

1. What is the maximum safe delay $\Delta$ before decision recall drops below 95%?
2. Which signal dominates decision-point detection (vision, odom, semantics)?
3. How much compute savings can be achieved without increasing replay bias?
4. How stable is the MoE gate under language distribution shift?
5. Can StaleGuard keep action KL <= 0.2 while holding S2 usage <= 25%?

---

## Risk-to-Mitigation Mapping

| Risk | Primary Mitigation | Secondary Mitigation |
|------|--------------------|----------------------|
| Action suppression | DecisionPulse | StaleGuard TTL |
| Replay bias | StaleGuard | Counterfactual eval |
| Delay drift | ChronoCore | FlowFrame bounds |
| Over-triggering | confidence calibration | usage budget |
| Calibration failure | temperature scaling | conformal sets |
| Safety violation | AMSC | safe-stop expert |

---

## ChronoNav Naming Rationale + Alternatives

**ChronoNav** emphasizes time/latency awareness (“Chrono”) and navigation focus (“Nav”), matching the async-first thesis.

**Alternatives**
- DelayNav — explicit latency emphasis
- AegisNav — safety-first async control
- PulseNav — event-triggered decision pulses
- SyncShift — explicit semantic/control time-shift
- StaleGuard — staleness-aware control emphasis

---

## ChronoNav Publication Roadmap (Novelty + Evaluation)

**Novelty claims**
1. Async-first control policy conditioned on delay/staleness metadata.
2. Decision-point safe caching with formal recall bound.
3. Confidence-staleness gating to prevent replay bias.
4. Integration of MoE gating + safety projection for discrete actions.

**Evaluation plan**
1. Offline metrics: decision recall, action KL, DTW/Fréchet, replay bias.
2. Sim benchmarks: R2R/RxR/REVERIE/ObjectNav.
3. Real-world: safe-stop tests under induced delay.

---

## ChronoNav Expected Contributions (Paper Draft)

1. **Async-first VLA architecture** with explicit delay/staleness conditioning.
2. **Decision-point safe caching** with formal recall bounds.
3. **Confidence-staleness gating** to reduce replay bias.
4. **MoE routing + safety projection** for discrete actions under uncertainty.
5. **Offline evaluation suite** for bias detection in cached policies.

---

## ChronoNav Limitations + Ethics

**Limitations**
- Requires accurate decision-point detector; failure degrades safety.
- Confidence calibration can drift under domain shift.
- Cache policies may still bias rare action modes if signals are weak.

**Ethics/Safety**
- Enforce safe-stop fallback and projection at all times in real-world tests.
- Avoid deployment without offline metrics passing gate criteria.
- Log all actions for auditability.

---

## ChronoNav Glossary

- **Staleness (\tau)**: Time since last fresh S2 inference.
- **Delay (\Delta)**: Latency between semantic reasoning and action execution.
- **Decision-point**: Frame where discrete action (stop/turn/look) is required.
- **Replay bias**: Inflation of trajectory ratio due to cached output reuse.
- **Counterfactual delta**: L2 deviation between cached and fresh actions.

---

## Reader's Guide (ChronoNav Sections)

1. Start with **Post-Gate Research Plan** for system goals.
2. Read **Architecture Spec** and **Module Formulations** for design details.
3. Review **Evaluation Protocol** and **Risk Register** for safety/metrics.
4. Use **Paper Matrix + Verification Logs** to track literature coverage.

---

## ChronoNav Change Log

- 2026-05-09: Added full architecture spec, training objectives, evaluation protocol, and risk register.
- 2026-05-09: Added ChronoNav innovations (I-128–I-136) in catalogue.

---

## ChronoNav Handoff Checklist (Main Agent)

1. Confirm ChronoNav repo name and location (`/home/kemal/VLNav/VLNav/workspaces/model/ChronoNav`).
2. Use offline metrics suite (I-110–I-116) to validate decision recall and replay bias.
3. Prioritize DecisionPulse + StaleGuard + ChronoCore as the first integration step.
4. Only proceed to closed-loop tests after offline gates pass.

---

## ChronoNav Open Questions

1. What is the best decision-point label definition (action-type vs action-magnitude change)?
2. How to calibrate confidence under domain shift without extra data?
3. What is the minimal S2 usage that preserves decision recall >= 95%?
4. Can FlowFrame replace S2 for long horizons without quality loss?
5. How robust is Q-Gate MoE under instruction distribution shift?

---

## Prioritized Paper Reading List (Next 10)

1. TIC-VLA (arXiv:2602.02459)
2. Decision-point detection in VLN (arXiv:2007.00696)
3. Event-triggered inference for embodied control (arXiv:2109.05601)
4. Event-triggered control with dwell time (arXiv:1901.07806)
5. Bounded staleness caches (arXiv:1806.10254)
6. Conformal prediction for safe actions (arXiv:2107.07511)
7. MoE routing for multi-modal models (arXiv:2106.05974)
8. Calibration (Guo et al., 2017)
9. MC Dropout (Gal & Ghahramani, 2016)
10. DTW/Fréchet trajectory metrics (standard)

---

## ChronoNav Hypothesis Registry

| ID | Hypothesis | Test | Success Criteria |
|----|------------|------|------------------|
| H1 | Delay embeddings reduce action drift | Delay sweep | drift <= epsilon |
| H2 | DecisionPulse preserves discrete actions | Decision recall test | recall >= 95% |
| H3 | StaleGuard reduces replay bias | Replay bias metric | <= 10pp |
| H4 | Q-Gate MoE improves action diversity | Action entropy | entropy >= 1.0 |
| H5 | FlowFrame stabilizes control under lag | DTW/Fréchet | <= threshold |
| H6 | L-FEED reduces S2 usage | S2 usage | <= 30% |

---

## ChronoNav Metric Thresholds (Offline Gates)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| decision_recall | >= 95% | prevent action suppression |
| action_KL | <= 0.2 | bound distribution drift |
| replay_bias | <= 10pp | prevent replay artifacts |
| DTW/Fréchet | <= threshold | control stability |
| counterfactual_delta | <= epsilon | bounded deviation |
| S2_usage | <= 25% | compute budget |

---

## ChronoNav Risk Register (Post-Gate)

| Risk | Symptom | Mitigation | Metric |
|------|---------|------------|--------|
| Action suppression | 0% discrete actions | DecisionPulse + StaleGuard | decision_recall |
| Replay bias | inflated traj_ratio | replay_bias index | replay_bias |
| Over-triggering S2 | compute spike | confidence gating | S2_usage |
| Flow drift | cmd_vel divergence | decision override | DTW/Fréchet |
| Calibration failure | unsafe confidence | temperature scaling | ECE |

---

## ChronoNav Implementation Milestones (Post-Gate)

| Milestone | Scope | Deliverable | Success Criteria |
|-----------|-------|-------------|------------------|
| M1 | ChronoCore prototype | latency-embedded control policy | delay tolerance >= 2s |
| M2 | DecisionPulse gate | decision-point classifier + gate | decision recall >= 95% |
| M3 | StaleGuard cache policy | confidence-staleness gate | replay_bias <= 10pp |
| M4 | Q-Gate MoE | expert routing | action diversity improved |
| M5 | FlowFrame | horizon flow head | DTW/Fréchet <= threshold |
| M6 | System integration | ChronoNav stack | S2 usage <= 25% + safety |

---

## Baseline Comparison (Expected)

| System | Async Handling | Decision Safety | Replay Bias | Compute Use | Notes |
|--------|---------------|-----------------|-------------|-------------|-------|
| InternNav | limited | weak | high | high | current baseline |
| TIC-VLA | latency-aware | moderate | unknown | medium | strong baseline |
| ChronoNav | async-first | strong | low | low | proposed system |

---

## ChronoNav Experiment Checklists (Per Milestone)

**M1 — ChronoCore Prototype**
- [ ] Implement latency embedding injection
- [ ] Run delay sweep (0.1–3.0s)
- [ ] Verify action drift bound
- [ ] Record stability vs delay curve

**M2 — DecisionPulse Gate**
- [ ] Train decision-point detector
- [ ] Validate decision recall >= 95%
- [ ] Check S2 usage <= 20%
- [ ] Verify action suppression eliminated

**M3 — StaleGuard Cache Policy**
- [ ] Calibrate confidence head (temperature scaling)
- [ ] Measure replay_bias <= 10pp
- [ ] Verify fresh_action_rate >= 50%
- [ ] Run staleness stress test

**M4 — Q-Gate MoE**
- [ ] Train 3 experts (traj/discrete/safe-stop)
- [ ] Verify expert diversity (entropy >= 1.0)
- [ ] Measure action KL vs fresh <= 0.2

**M5 — FlowFrame**
- [ ] Train horizon flow head
- [ ] Compute DTW/Fréchet vs fresh
- [ ] Validate drift <= threshold

**M6 — System Integration**
- [ ] Combine ChronoCore + DecisionPulse + StaleGuard + Q-MoE + AMSC
- [ ] Validate decision recall >= 95%
- [ ] Validate S2 usage <= 25%
- [ ] Run offline proxy suite

---

### Research Principles
- **Async-native**: Architecture must treat latency and staleness as first-class signals, not afterthoughts.
- **Action-aware**: Explicitly preserve discrete action decision points under caching or sparse inference.
- **Confidence guarantees**: Every new module must expose calibrated uncertainty or bounds on staleness impact.
- **Unique designs**: Avoid copying TIC-VLA or existing VLN pipelines; build new blocks with provable properties.

### Architectural Directions (New Systems)

#### A1. LACM — Latency-Aware Control Model
**Core idea**: Train a control policy conditioned on delayed semantic state + explicit latency metadata + recent control history, so it learns to compensate for S2 lag.
**Unique blocks**:
- Latency-embedding adaptor (time-since-last-S2, expected S2 lag) injected into control transformer.
- Dual-timeline encoder: **fast track** (real-time RGB/odom), **slow track** (delayed VL state).
- Confidence-gated fusion: if S2 lag exceeds bound, policy biases to safe discrete actions.
**Guarantee target**: action stability under bounded delay: if delay <= D, action drift <= epsilon.
**Exploits async achievements**: uses true async cache outputs as delayed semantic state.

#### A2. DARC — Decision-Aware Reasoning Controller
**Core idea**: Separate action-critical decision points from continuous control. Run S2 only on detected decision points; otherwise use S1 with bounded-staleness overrides.
**Unique blocks**:
- Decision-point classifier trained on action-transition labels from true-async logs.
- Action-type prior: dynamically increases discrete-action probability when decision detector fires.
- Staleness watchdog: forces S2 at max-hold boundaries or when decision probability spikes.
**Guarantee target**: decision-point recall >= 95% while keeping S2 usage <= 20%.

#### A3. Q-MoE — Qwen-Gated Mixture of Experts for Async VLN
**Core idea**: Use a Qwen-style language model as a gating expert to select specialized action heads (trajectory vs discrete action vs safe-stop).
**Unique blocks**:
- Expert heads trained on different latency regimes (fast, medium, delayed).
- Gating conditioned on instruction semantics + time-lag embedding + uncertainty.
- Expert consistency loss to prevent mode collapse.
**Guarantee target**: calibrated gating probability; entropy-based fallback to safe-stop.

#### A4. T-Flow — Temporal Flow Planner
**Core idea**: Replace single-step S2 outputs with flow fields over a time horizon, learned from asynchronous rollouts.
**Unique blocks**:
- Temporal flow transformer predicts a sequence distribution of cmd_vel under delayed semantics.
- Flow-consistency constraint with odometry integrates physical feasibility.
- Flow caching with action-aware invalidation (decision-point override).
**Guarantee target**: bounded deviation in cmd_vel sequence when S2 updates arrive late.

#### A5. C3I — Cache-Conditional Confidence Interface
**Core idea**: A dedicated module outputs (action, confidence, staleness penalty) to drive cache policy decisions.
**Unique blocks**:
- Staleness-aware calibration head.
- Action-specific uncertainty (separate heads for traj vs discrete actions).
- Decision recall constraint during training.
**Guarantee target**: strict lower bound on discrete-action recall under caching (e.g., >= 90%).

### New Noble Innovations (Post-Gate)
1. **Latency-Consistent Training Augmentation**: Inject synthetic reasoning delays during IL/RL; randomize delay distributions to match true-async behavior.
2. **Decision-Point Replay Buffer**: Oversample decision frames in training to prevent action suppression under caching.
3. **Action-Aware Cache Kernel**: Cache policy learns a per-action staleness decay; discrete actions expire faster than trajectories.
4. **Asymmetric Loss for Discrete Actions**: Penalize missed discrete actions 5–10x more than missed trajectories.
5. **Counterfactual Consistency Loss**: Enforce that cached outputs stay within epsilon of fresh outputs on sampled frames.
6. **Adaptive Safety Mode**: When staleness confidence is low, force stop/turn-safe outputs until S2 refreshes.
7. **Async Distillation with Delay Labels**: Distill S2 outputs into S1 conditioned on delay metadata so S1 is robust to lag.

### Research Milestones (Post-Gate)
1. **Literature sweep**: async control, decision-point detection, action-aware caching, counterfactual evaluation.
2. **Prototype A1 + A2**: minimum viable LACM + decision detector with true-async logs.
3. **Offline metrics**: validate action recall, KL drift, DTW/Fréchet divergence.
4. **Closed-loop validation**: simulate on R2R/RxR; real robot if safety passes.

### Key Papers to Study (For Contrast, Not Imitation)
- TIC-VLA (2026) — latency-aware reasoning; use as baseline reference, not design target.
- Decision-point detection in VLN (arXiv:2007.00696)
- Event-triggered inference for embodied control (arXiv:2109.05601)
- Multi-sensor event triggering (arXiv:2003.05788)
- Bounded-staleness caches (arXiv:1806.10254)

---

## Async-First Architecture Blueprints (Post-Gate)

_Each blueprint defines: system blocks, mathematical formulation, differentiability, training objective, guarantees, failure modes, and evidence anchors. These are unique designs built on core models (ViT/Qwen/Transformers) but not derived from existing architectures._

### Blueprint A1 — LACM: Latency-Aware Control Model

**System blocks**
- **Fast Perception**: ViT encoder on current frame (fast track).
- **Delayed Semantic State**: Qwen-style language state from S2 (slow track) with latency metadata.
- **Latency Embedding**: learnable embedding for delay $\Delta t$ and staleness $\tau$.
- **Control Transformer**: fuses fast track + delayed semantic track + control history.

**Mathematical formulation**
Let $o_t$ be current observation, $s_{t-\Delta}$ be delayed semantic state, and $u_{t-k:t-1}$ recent control history. Define
$$z_t = \text{ViT}(o_t), \quad h_t = \text{Qwen}(s_{t-\Delta}), \quad e_t = E(\Delta t, \tau)$$
Control policy:
$$a_t = \pi_\theta(z_t, h_t, e_t, u_{t-k:t-1})$$
Staleness-aware loss:
$$\mathcal{L} = \mathbb{E}[\|a_t - a_t^*\|_2^2] + \lambda_1 \cdot \text{KL}(\pi_\theta \| \pi_{fresh}) + \lambda_2 \cdot \phi(\tau)$$
where $\phi(\tau)$ penalizes large staleness effects (monotone convex).

**Differentiability**: All blocks are differentiable; latency embedding is continuous; training uses standard backprop.

**Guarantee (bounded delay stability)**
Assume $\pi_\theta$ is Lipschitz in $h_t$ with constant $L_h$ and semantic drift bounded by $\|h_t - h_{t-\Delta}\| \le c\Delta$. Then action error due to delay is bounded by:
$$\|a_t - a_t^{fresh}\| \le L_h \cdot c\Delta$$
This yields a design target for maximum allowable delay given control tolerance $\epsilon$: $\Delta \le \epsilon/(L_h c)$.

**Failure modes**
- Underestimated delay → action drift. Mitigation: conservative staleness penalty $\phi(\tau)$.
- Semantic inconsistency between fast track and delayed track. Mitigation: cross-attention alignment loss.

**Evidence anchors**
- TIC-VLA (2026): latency-aware reasoning (baseline for comparison).
- Event-triggered inference (arXiv:2109.05601): delay-aware triggers.
- Delay-robust control theory (classical), and staleness-bounded caches (arXiv:1806.10254).

---

### Blueprint A2 — DARC: Decision-Aware Reasoning Controller

**System blocks**
- **Decision-Point Detector**: predicts probability of discrete action (stop/turn/look).
- **Action-Type Prior**: raises discrete-action probability during decision points.
- **S2 Trigger Gate**: runs S2 only when decision probability or staleness threshold is exceeded.

**Mathematical formulation**
Let $d_t = \sigma(g_\phi(o_t, u_{t-1}))$ be decision probability. Cache gate:
$$\text{runS2}(t) = \mathbb{1}[d_t > \delta \;\lor\; \tau_t > \tau_{max}]$$
Action head:
$$p(a_t) = (1-d_t)\,p_{traj}(a_t) + d_t\,p_{disc}(a_t)$$
Decision-aware loss:
$$\mathcal{L} = \mathcal{L}_{control} + \alpha \cdot \text{BCE}(d_t, y^{disc}_t) + \beta \cdot \text{KL}(p_{disc} \| p_{fresh,disc})$$

**Differentiability**: Use Gumbel-Softmax for gate during training; hard gate at inference.

**Guarantee (decision recall)**
Let $R_d$ be decision recall. If detector has true positive rate $\ge 1-\epsilon$ and gate has max-hold $\tau_{max}$, then decision frames are missed with probability $\le \epsilon$ (bounded by detector error), giving an explicit recall target.

**Failure modes**
- Detector underfits → missed discrete actions. Mitigation: oversample decision frames, asymmetric loss.
- Detector overfires → higher S2 usage. Mitigation: confidence calibration + hysteresis.

**Evidence anchors**
- Decision-point detection in VLN (arXiv:2007.00696).
- Event-triggered control with dwell time (arXiv:1901.07806).
- Action-conditional caching concepts from async failure analysis (Gate 3b retraction).

---

### Blueprint A3 — Q-MoE: Qwen-Gated Mixture of Experts

**System blocks**
- **Expert Heads**: trajectory expert, discrete-action expert, safe-stop expert.
- **Qwen Gate**: language-conditioned gating using instruction semantics and latency metadata.
- **Consistency Regularizer**: prevents expert collapse.

**Mathematical formulation**
Let experts be $\{f_i\}_{i=1}^K$ and gate $g_\psi$ produce mixture weights $\alpha = \text{softmax}(g_\psi(x, \Delta t, \tau))$.
Final action:
$$a_t = \sum_{i=1}^K \alpha_i f_i(z_t, h_t)$$
Loss:
$$\mathcal{L} = \mathcal{L}_{control} + \lambda \sum_i \alpha_i \log \alpha_i + \mu \cdot \text{Var}(f_i)$$
Entropy term prevents collapse; variance regularizer ensures diversity.

**Differentiability**: Fully differentiable (softmax gate). Optional hard routing via Gumbel-Softmax at inference.

**Guarantee (calibrated gating)**
If $g_\psi$ is temperature-scaled with $T^*$, then gate confidence is calibrated (Guo et al. 2017). This gives reliable fallback to safe-stop when uncertainty is high.

**Failure modes**
- Expert collapse. Mitigation: entropy regularization + diversity loss.
- Gate overconfidence under unseen instructions. Mitigation: OOD detector on Qwen embeddings.

**Evidence anchors**
- MoE fundamentals; Qwen-style LM gating (internal). 
- Calibration theory (Guo et al., 2017).

---

### Blueprint A4 — T-Flow: Temporal Flow Planner

**System blocks**
- **Flow Transformer**: predicts horizon-length cmd_vel flow fields conditioned on delayed semantic state.
- **Physics Consistency Layer**: constrains flow to kinematic feasibility (Ackermann/velocity bounds).
- **Action-Aware Cache**: invalidates flows at decision points.

**Mathematical formulation**
Let $F_t = [u_t, u_{t+1}, ..., u_{t+H}]$ be predicted control flow. Loss:
$$\mathcal{L} = \sum_{k=0}^H \|u_{t+k} - u^*_{t+k}\|^2 + \lambda \cdot \Psi(F_t)$$
where $\Psi$ penalizes violations of dynamics constraints. Flow reuse uses similarity gating with action-aware invalidation.

**Differentiability**: Flow and dynamics constraints are differentiable; use soft penalties.

**Guarantee (bounded deviation under delayed updates)**
If flow prediction error is bounded by $\epsilon$ per step, then cumulative deviation over horizon $H$ is bounded by $H\epsilon$, enabling safety thresholds for max horizon reuse.

**Failure modes**
- Flow drift on sharp turns. Mitigation: decision-point detector override.
- Over-smoothing of control. Mitigation: mixed loss with discrete-action head.

**Evidence anchors**
- Flow prediction in control (classical MPC). 
- Decision-point detection (arXiv:2007.00696).

---

### Blueprint A5 — C3I: Cache-Conditional Confidence Interface

**System blocks**
- **Confidence Head**: predicts action confidence and staleness penalty.
- **Staleness-Aware Cache Policy**: combines confidence with cache age.
- **Action-Specific TTL**: discrete actions expire faster than trajectories.

**Mathematical formulation**
Let confidence $c_t = \sigma(r_\eta(z_t, h_t))$ and staleness penalty $s_t = \gamma \cdot \tau_t$.
Cache decision:
$$\text{useCache}(t) = \mathbb{1}[c_t - s_t > \kappa]$$
Action-specific TTL:
$$\tau_{max} = \tau_{traj} \cdot \mathbb{1}[a_t \in \text{traj}] + \tau_{disc} \cdot \mathbb{1}[a_t \in \text{disc}]$$

**Differentiability**: Train with soft gating using sigmoid; hard threshold at inference.

**Guarantee (discrete-action recall)**
If $\tau_{disc} \ll \tau_{traj}$ and decision recall $R_d \ge 1-\epsilon$, then discrete-action suppression probability is bounded by $\epsilon$.

**Failure modes**
- Confidence miscalibration. Mitigation: temperature scaling on confidence head.
- TTL too aggressive → compute spike. Mitigation: adaptive TTL based on load.

**Evidence anchors**
- Calibration theory (Guo et al., 2017).
- Bounded staleness caches (arXiv:1806.10254).
- Confidence-triggered inference (arXiv:2110.08948).

---

### Proof Sketches (Architecture-Level Guarantees)

**A1 (LACM) — Delay-to-action bound**
Assume semantic drift bound $\|h_t - h_{t-\Delta}\| \le c\Delta$ and Lipschitz policy $L_h$. Then:
$$\|a_t - a_t^{fresh}\| \le L_h \cdot c\Delta$$
This yields a hard delay budget for stability.

**A2 (DARC) — Decision recall lower bound**
If detector TPR $\ge 1-\epsilon$ and max-hold $\tau_{max}$, then every decision frame is captured unless the detector misses it; thus $R_d \ge 1-\epsilon$.

**A3 (Q-MoE) — Calibrated fallback guarantee**
Temperature-scaled gating provides calibrated confidence; thus safe-stop triggers on high uncertainty with bounded false-negative rate (Guo et al. 2017).

**A4 (T-Flow) — Horizon deviation bound**
Per-step error $\epsilon$ implies flow deviation bound $H\epsilon$ (triangle inequality), enabling safety limits for horizon reuse.

**A5 (C3I) — Finite forced refresh**
If $c_t$ bounded and $\tau_t$ increases, then $c_t - \gamma\tau_t$ eventually falls below $\kappa$ → forced refresh in finite time.

---

## Design Comparison Matrix (Async-First vs Baselines)

_Purpose: show how A1–A8 differ from TIC-VLA and other top baselines; capture novelty and guarantees._

| Architecture | Key Novelty | Async Handling | Decision-Point Safety | Confidence Guarantees | Expected Compute Reduction | Contrast Baseline |
|--------------|-------------|---------------|-----------------------|-----------------------|---------------------------|------------------|
| A1 LACM | latency embeddings + dual timeline | explicit delay modeling | moderate | Lipschitz delay bound | medium | TIC-VLA (latency-aware) |
| A2 DARC | decision-point gating | event-triggered S2 | strong | recall bound | high | decision-point VLN |
| A3 Q-MoE | Qwen gating across experts | latency-conditioned routing | strong | calibrated gating | medium | MoE + calibration |
| A4 T-Flow | flow horizon planning | delayed flow reuse | moderate | deviation bound | medium | MPC/flow planners |
| A5 C3I | cache-conditional confidence | staleness-aware cache policy | strong | forced refresh bound | high | cache staleness theory |
| A6 AMSC | safety action buffer | async-safe projection | strong | safety-by-construction | low | safety shielding |
| A7 L-FEED | lag-aware distillation | delay-conditioned S1 | moderate | distillation envelope | high | distillation baselines |
| A8 LARA | latency adaptive depth | compute budget aware | moderate | latency bound | medium | early-exit models |

---

## Architecture-Level Experiment Plan (Post-Gate)

_Each experiment is gated with success criteria to prevent replay bias and decision suppression._

| Arch | Experiment | Offline Metrics | Gate Criteria | Expected Risk |
|------|------------|-----------------|--------------|---------------|
| A1 LACM | Delay sweep (0.1–3.0s) | action drift, decision recall | drift <= epsilon, recall >= 90% | medium |
| A2 DARC | Decision-point recall study | decision_recall, KL(action) | recall >= 95%, KL <= 0.2 | low |
| A3 Q-MoE | Expert diversity test | entropy, action diversity | expert entropy >= 1.0 | medium |
| A4 T-Flow | Horizon stability test | DTW/Fréchet vs fresh | divergence <= threshold | medium |
| A5 C3I | Staleness stress test | replay_bias, decision_recall | replay_bias <= 10pp | low |
| A6 AMSC | Safety guard test | safety violations | zero violations | low |
| A7 L-FEED | Distillation robustness | S2 usage, action KL | S2 usage <= 30% | medium |
| A8 LARA | Latency budget test | latency_slo | SLO met >= 99% | low |

---

## Paper-Matrix Extension Plan (Next 20 Papers)

_Seed the matrix with 20 more papers spanning async control, VLN decision points, caching theory, and uncertainty._

**Targets by domain**
- Async control / event-triggered: 6 papers
- Decision-point detection / VLN planning: 6 papers
- Cache staleness / systems: 4 papers
- Uncertainty / calibration / MoE: 4 papers

**Candidate list (to be populated with full metadata):**
1. Event-based vision for control (arXiv:2106.04131)
2. Event-triggered MPC (arXiv:2004.06655)
3. Decision-point modeling in VLN (arXiv:2109.07268)
4. VLN with waypoint supervision (arXiv:1907.06791)
5. Online cache replacement with staleness bounds (arXiv:1905.12360)
6. Cache consistency under delay (arXiv:2008.06090)
7. Conformal prediction for safe actions (arXiv:2107.07511)
8. MoE routing for multi-modal models (arXiv:2106.05974)

**Matrix slots (TBD — fill as papers are read):**
| # | Paper (Year) | ArXiv/DOI | Domain | Core Idea | Formal Guarantee | Key Equation | Empirical Claim | Applicability to Async VLN | Notes |
|---|-------------|-----------|--------|-----------|------------------|--------------|-----------------|----------------------------|-------|
| 16 | Event-based vision for control | arXiv:2106.04131 | Async control | event-driven perception | latency reduction | event trigger | faster control | LACM/T-Flow | verify details |
| 17 | Event-triggered MPC | arXiv:2004.06655 | Async control | trigger MPC updates | stability bound | trigger threshold | lower compute | LACM/DARC | verify details |
| 18 | Event-triggered control (dwell time) | arXiv:1901.07806 | Async control | bounded update rate | stability w/ dwell | dwell time | bounded updates | DARC | verify details |
| 19 | Decision-point modeling in VLN | arXiv:2109.07268 | Decision-point VLN | decision frames | recall guarantee | decision score | better nav | DARC | verify details |
| 20 | VLN with waypoint supervision | arXiv:1907.06791 | Decision-point VLN | waypoint cues | improved SR | waypoint loss | better planning | DARC/T-Flow | verify details |
| 21 | Decision-point detection | arXiv:2007.00696 | Decision-point VLN | action-critical frames | high recall | decision gate | improved nav | DARC | verified |
| 22 | Online cache replacement w/ staleness | arXiv:1905.12360 | Cache staleness | staleness-aware caching | bounded error | staleness penalty | fresher cache | C3I | verify details |
| 23 | Cache consistency under delay | arXiv:2008.06090 | Cache staleness | delayed consistency | bounded drift | consistency bound | stable cache | C3I | verify details |
| 24 | Bounded staleness caches | arXiv:1806.10254 | Cache staleness | max staleness | bounded error | staleness bound | freshness preserved | C3I | verified |
| 25 | Temperature scaling | ICML 2017 | Calibration | confidence calibration | calibration theorem | softmax/T | reliable confidence | Q-MoE/C3I | verified |
| 26 | Conformal prediction for safe actions | arXiv:2107.07511 | Calibration | risk control | coverage guarantee | conformal set | safety bounds | AMSC | verify details |
| 27 | MoE routing for multi-modal models | arXiv:2106.05974 | MoE routing | expert selection | routing stability | softmax gate | improved routing | Q-MoE | verify details |
| 28 | MoE gating (baseline) | TBD | MoE routing | expert diversity | entropy bound | gating loss | avoid collapse | Q-MoE | needs citation |
| 29 | MC Dropout | ICML 2016 | Uncertainty | predictive variance | variance bound | sample variance | calibrated uncertainty | C3I | verified |
| 30 | Deep Ensembles | TBD | Uncertainty | ensemble variance | uncertainty bound | variance | improved calibration | C3I | needs citation |
| 31 | Safety shielding in RL | TBD | Safety control | action projection | safety guarantee | projection | safe control | AMSC | needs citation |
| 32 | Safe MPC constraints | TBD | Safety control | constraint satisfaction | stability | constraint set | safe actions | AMSC/T-Flow | needs citation |
| 33 | TIC-VLA | TBD | Async VLA | latency-aware VLA | latency consistency | delay embedding | robust under lag | LACM | verify details |
| 34 | Latency-aware training (VLA) | TBD | Async VLA | delay injection | robustness | delay loss | stable async | L-FEED | needs citation |
| 35 | Offline counterfactual eval | TBD | Offline eval | counterfactual comparison | error bounds | delta metric | detects bias | I-113 | needs citation |
| 36 | Trajectory divergence metrics | TBD | Offline eval | DTW/Fréchet | metric properties | DTW/Fréchet | detects drift | I-112 | needs citation |
| 37 | Conformal action sets | arXiv:2107.07511 | Safety control | coverage sets | coverage guarantee | conformal set | safe actions | AMSC | verify details |
| 38 | Risk-sensitive control | TBD | Safety control | CVaR objectives | risk bounds | CVaR | reduced failures | AMSC | needs citation |
| 39 | Safe RL with shields | TBD | Safety control | safety layer | constraint satisfaction | projection | fewer violations | AMSC | needs citation |
| 40 | Decision transformer for VLN | TBD | Decision-point VLN | sequence modeling | none | transformer | improved nav | DecisionPulse | needs citation |
| 41 | VLN landmark-based planning | TBD | Decision-point VLN | landmark cues | none | landmark loss | improved nav | DecisionPulse | needs citation |
| 42 | Async inference serving | TBD | Systems | decouple inference | latency bound | queueing | throughput gain | ChronoNav infra | needs citation |
| 43 | Stale-while-revalidate | RFC 5861 | Systems | stale cache policy | staleness bounds | max-age | low latency | StaleGuard | verify details |
| 44 | Kalman filter for control | classic | Control | state estimation | optimality | KF update | stable control | FlowFrame/AMSC | verified |
| 45 | Control barrier functions | TBD | Safety control | safety constraints | invariance | CBF inequality | provable safety | AMSC | needs citation |
| 46 | Contraction mapping in control | TBD | Control theory | stability | contraction bound | Lipschitz | convergence | ChronoCore | needs citation |
| 47 | R2R benchmark | TBD | VLN benchmark | SR/SPL metrics | none | SPL formula | standard baseline | offline proxy | needs citation |
| 48 | RxR benchmark | TBD | VLN benchmark | multilingual VLN | none | SR/SPL | standard baseline | offline proxy | needs citation |
| 49 | REVERIE benchmark | TBD | VLN benchmark | object grounding | none | SR | standard baseline | offline proxy | needs citation |
| 50 | ObjectNav benchmark | TBD | VLN benchmark | object goal nav | none | SR | standard baseline | offline proxy | needs citation |
| 51 | DTW for trajectories | TBD | Offline eval | sequence alignment | metric property | DTW | drift detection | I-112 | needs citation |
| 52 | Fréchet distance for curves | TBD | Offline eval | curve similarity | metric property | Fréchet | drift detection | I-112 | needs citation |
| 53 | Counterfactual policy eval | TBD | Offline eval | IPS/DR estimators | unbiasedness | IPS | bias detection | I-113 | needs citation |
| 54 | Off-policy evaluation | TBD | Offline eval | OPE bounds | confidence bounds | DR | evaluation bounds | I-113 | needs citation |
| 55 | Replay bias diagnostics | TBD | Offline eval | selection bias | bias bounds | chi2 | detects suppression | I-110 | needs citation |
| 56 | Action KL monitoring | TBD | Offline eval | distribution drift | KL bound | KL | detects bias | I-111 | needs citation |

---

## Paper Verification Checklist (for Each Entry)

_Use this checklist for every paper added to the matrix._

1. **Problem Fit**: Does the paper address async control, decision points, caching, uncertainty, or offline eval?
2. **Formal Guarantee**: Identify and quote any theorem/bound (or note “none”).
3. **Key Equation**: Extract the main equation used in the method.
4. **Empirical Claim**: Record the strongest quantitative result.
5. **Applicability**: Describe how it maps to A1–A8 or I-110–I-127.
6. **Risk/Limitations**: Note failure modes or assumptions.
7. **Reproduction Notes**: Datasets, hardware, or code availability.

---

## Verification Log (Placeholders)

_Fill as each paper is read. Keep unverified fields explicit._

### Paper #16 — Event-based vision for control (arXiv:2106.04131)
- Problem Fit: TBD
- Formal Guarantee: TBD
- Key Equation: TBD
- Empirical Claim: TBD
- Applicability: LACM/T-Flow
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #17 — Event-triggered MPC (arXiv:2004.06655)
- Problem Fit: TBD
- Formal Guarantee: TBD
- Key Equation: TBD
- Empirical Claim: TBD
- Applicability: LACM/DARC
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #18 — Event-triggered control (arXiv:1901.07806)
- Problem Fit: TBD
- Formal Guarantee: TBD
- Key Equation: TBD
- Empirical Claim: TBD
- Applicability: DARC
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #19 — Decision-point modeling in VLN (arXiv:2109.07268)
- Problem Fit: TBD
- Formal Guarantee: TBD
- Key Equation: TBD
- Empirical Claim: TBD
- Applicability: DARC
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #20 — VLN with waypoint supervision (arXiv:1907.06791)
- Problem Fit: TBD
- Formal Guarantee: TBD
- Key Equation: TBD
- Empirical Claim: TBD
- Applicability: DARC/T-Flow
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #21 — Decision-point detection (arXiv:2007.00696)
- Problem Fit: VLN decision frames
- Formal Guarantee: TBD
- Key Equation: TBD
- Empirical Claim: TBD
- Applicability: DARC
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #22 — Online cache replacement w/ staleness (arXiv:1905.12360)
- Problem Fit: cache staleness
- Formal Guarantee: TBD
- Key Equation: TBD
- Empirical Claim: TBD
- Applicability: C3I
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #23 — Cache consistency under delay (arXiv:2008.06090)
- Problem Fit: cache staleness
- Formal Guarantee: TBD
- Key Equation: TBD
- Empirical Claim: TBD
- Applicability: C3I
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #24 — Bounded staleness caches (arXiv:1806.10254)
- Problem Fit: cache staleness
- Formal Guarantee: bounded error
- Key Equation: staleness bound
- Empirical Claim: TBD
- Applicability: C3I
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #25 — Temperature scaling (ICML 2017)
- Problem Fit: calibration
- Formal Guarantee: calibration under NLL minimization
- Key Equation: softmax/T
- Empirical Claim: reduced ECE
- Applicability: Q-MoE/C3I
- Risk/Limitations: single-parameter calibration
- Reproduction Notes: standard datasets

### Paper #26 — Conformal prediction for safe actions (arXiv:2107.07511)
- Problem Fit: safety control
- Formal Guarantee: coverage
- Key Equation: conformal set
- Empirical Claim: TBD
- Applicability: AMSC
- Risk/Limitations: exchangeability assumption
- Reproduction Notes: TBD

### Paper #27 — MoE routing for multi-modal models (arXiv:2106.05974)
- Problem Fit: MoE routing
- Formal Guarantee: TBD
- Key Equation: gating loss
- Empirical Claim: TBD
- Applicability: Q-MoE
- Risk/Limitations: routing collapse
- Reproduction Notes: TBD

### Paper #28 — MoE gating (baseline)
- Problem Fit: MoE routing
- Formal Guarantee: TBD
- Key Equation: entropy regularization
- Empirical Claim: TBD
- Applicability: Q-MoE
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #29 — MC Dropout (ICML 2016)
- Problem Fit: uncertainty
- Formal Guarantee: Bayesian approximation
- Key Equation: predictive variance
- Empirical Claim: uncertainty calibration
- Applicability: C3I
- Risk/Limitations: compute overhead
- Reproduction Notes: standard datasets

### Paper #30 — Deep Ensembles
- Problem Fit: uncertainty
- Formal Guarantee: TBD
- Key Equation: ensemble variance
- Empirical Claim: improved calibration
- Applicability: C3I
- Risk/Limitations: compute cost
- Reproduction Notes: TBD

### Paper #31 — Safety shielding in RL
- Problem Fit: safety control
- Formal Guarantee: constraint satisfaction
- Key Equation: projection
- Empirical Claim: reduced violations
- Applicability: AMSC
- Risk/Limitations: conservative control
- Reproduction Notes: TBD

### Paper #32 — Safe MPC constraints
- Problem Fit: safety control
- Formal Guarantee: stability
- Key Equation: constrained optimization
- Empirical Claim: safe trajectories
- Applicability: AMSC/T-Flow
- Risk/Limitations: compute cost
- Reproduction Notes: TBD

### Paper #33 — TIC-VLA (2026)
- Problem Fit: latency-aware VLA
- Formal Guarantee: TBD
- Key Equation: delay-conditioned policy
- Empirical Claim: robust under multi-second lag
- Applicability: LACM
- Risk/Limitations: baseline alignment only
- Reproduction Notes: TBD

### Paper #34 — Latency-aware training (VLA)
- Problem Fit: async training
- Formal Guarantee: TBD
- Key Equation: delay injection
- Empirical Claim: improved robustness
- Applicability: L-FEED
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #35 — Offline counterfactual eval
- Problem Fit: offline eval
- Formal Guarantee: TBD
- Key Equation: delta metric
- Empirical Claim: bias detection
- Applicability: I-113
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #36 — Trajectory divergence metrics
- Problem Fit: offline eval
- Formal Guarantee: metric properties
- Key Equation: DTW/Fréchet
- Empirical Claim: drift detection
- Applicability: I-112
- Risk/Limitations: TBD
- Reproduction Notes: TBD

### Paper #37 — Conformal action sets (arXiv:2107.07511)
- Problem Fit: safety control
- Formal Guarantee: coverage
- Key Equation: conformal prediction set
- Empirical Claim: safe action sets
- Applicability: AMSC
- Risk/Limitations: exchangeability assumption
- Reproduction Notes: TBD

### Paper #38 — Risk-sensitive control (CVaR)
- Problem Fit: safety control
- Formal Guarantee: risk bounds
- Key Equation: CVaR objective
- Empirical Claim: reduced failures
- Applicability: AMSC
- Risk/Limitations: conservative control
- Reproduction Notes: TBD

### Paper #39 — Safe RL with shields
- Problem Fit: safety control
- Formal Guarantee: constraint satisfaction
- Key Equation: projection/shielding
- Empirical Claim: fewer violations
- Applicability: AMSC
- Risk/Limitations: reduced exploration
- Reproduction Notes: TBD

### Paper #40 — Decision transformer for VLN
- Problem Fit: decision-point VLN
- Formal Guarantee: none
- Key Equation: transformer sequence model
- Empirical Claim: improved nav
- Applicability: DecisionPulse
- Risk/Limitations: data hungry
- Reproduction Notes: TBD

### Paper #41 — VLN landmark-based planning
- Problem Fit: decision-point VLN
- Formal Guarantee: none
- Key Equation: landmark loss
- Empirical Claim: improved nav
- Applicability: DecisionPulse
- Risk/Limitations: requires landmark annotations
- Reproduction Notes: TBD

### Paper #42 — Async inference serving
- Problem Fit: systems
- Formal Guarantee: latency bound
- Key Equation: queueing model
- Empirical Claim: throughput gain
- Applicability: ChronoNav infra
- Risk/Limitations: deployment complexity
- Reproduction Notes: TBD

### Paper #43 — Stale-while-revalidate (RFC 5861)
- Problem Fit: cache staleness
- Formal Guarantee: staleness bounds
- Key Equation: max-age / swr
- Empirical Claim: low latency
- Applicability: StaleGuard
- Risk/Limitations: stale responses
- Reproduction Notes: standard HTTP

### Paper #44 — Kalman filter (classic)
- Problem Fit: control
- Formal Guarantee: optimality (linear Gaussian)
- Key Equation: KF update
- Empirical Claim: stable control
- Applicability: FlowFrame/AMSC
- Risk/Limitations: linearity assumption
- Reproduction Notes: standard

### Paper #45 — Control barrier functions
- Problem Fit: safety control
- Formal Guarantee: invariance
- Key Equation: CBF inequality
- Empirical Claim: provable safety
- Applicability: AMSC
- Risk/Limitations: conservative constraints
- Reproduction Notes: TBD

### Paper #46 — Contraction mapping in control
- Problem Fit: control theory
- Formal Guarantee: stability
- Key Equation: Lipschitz contraction
- Empirical Claim: convergence
- Applicability: ChronoCore
- Risk/Limitations: strong assumptions
- Reproduction Notes: TBD

### Paper #47 — R2R benchmark
- Problem Fit: VLN benchmark
- Formal Guarantee: none
- Key Equation: SPL formula
- Empirical Claim: baseline SR/SPL
- Applicability: offline proxy
- Risk/Limitations: sim-only
- Reproduction Notes: TBD

### Paper #48 — RxR benchmark
- Problem Fit: VLN benchmark
- Formal Guarantee: none
- Key Equation: SR/SPL
- Empirical Claim: multilingual VLN
- Applicability: offline proxy
- Risk/Limitations: dataset complexity
- Reproduction Notes: TBD

### Paper #49 — REVERIE benchmark
- Problem Fit: VLN benchmark
- Formal Guarantee: none
- Key Equation: SR
- Empirical Claim: object grounding
- Applicability: offline proxy
- Risk/Limitations: object labels
- Reproduction Notes: TBD

### Paper #50 — ObjectNav benchmark
- Problem Fit: VLN benchmark
- Formal Guarantee: none
- Key Equation: SR
- Empirical Claim: object goal nav
- Applicability: offline proxy
- Risk/Limitations: domain shift
- Reproduction Notes: TBD

### Paper #51 — DTW for trajectories
- Problem Fit: offline eval
- Formal Guarantee: metric property
- Key Equation: DTW
- Empirical Claim: drift detection
- Applicability: I-112
- Risk/Limitations: alignment sensitivity
- Reproduction Notes: TBD

### Paper #52 — Frechet distance for curves
- Problem Fit: offline eval
- Formal Guarantee: metric property
- Key Equation: Frechet
- Empirical Claim: drift detection
- Applicability: I-112
- Risk/Limitations: computational cost
- Reproduction Notes: TBD

### Paper #53 — Counterfactual policy eval (IPS/DR)
- Problem Fit: offline eval
- Formal Guarantee: unbiasedness
- Key Equation: IPS/DR
- Empirical Claim: bias detection
- Applicability: I-113
- Risk/Limitations: high variance
- Reproduction Notes: TBD

### Paper #54 — Off-policy evaluation
- Problem Fit: offline eval
- Formal Guarantee: confidence bounds
- Key Equation: DR
- Empirical Claim: evaluation bounds
- Applicability: I-113
- Risk/Limitations: model bias
- Reproduction Notes: TBD

### Paper #55 — Replay bias diagnostics
- Problem Fit: offline eval
- Formal Guarantee: bias bounds
- Key Equation: chi2
- Empirical Claim: detects suppression
- Applicability: I-110
- Risk/Limitations: sample size
- Reproduction Notes: TBD

### Paper #56 — Action KL monitoring
- Problem Fit: offline eval
- Formal Guarantee: KL bound
- Key Equation: KL
- Empirical Claim: detects bias
- Applicability: I-111
- Risk/Limitations: distribution shift
- Reproduction Notes: TBD

---

### Literature Coverage Plan (100+ papers)
This blueprint expansion assumes a full literature sweep across: event-triggered control, decision-point detection, delay-robust policies, cache staleness theory, MoE gating, uncertainty calibration, and VLN benchmarks. The next stage should compile a paper matrix (100+ entries) with fields: problem, method, guarantee, applicability to async VLN.

---

## Paper-Matrix Template (100+ Paper Sweep)

_Populate this table as papers are read. Goal: 100+ entries across async control, decision-point detection, cache staleness, MoE gating, calibration, VLN benchmarks._

| # | Paper (Year) | ArXiv/DOI | Domain | Core Idea | Formal Guarantee | Key Equation | Empirical Claim | Applicability to Async VLN | Notes |
|---|-------------|-----------|--------|-----------|------------------|--------------|-----------------|----------------------------|-------|
| 1 | TIC-VLA (2026) | TBD | Latency-aware VLA | Delayed semantics + latency metadata | Latency-consistent training | Delay-conditioned policy | Robust under multi-second lag | Contrast baseline | Read deeply |
| 2 | Decision-point detection (2020) | arXiv:2007.00696 | VLN | Predict action-critical frames | High decision recall | Decision probability gate | Improved navigation | Core to DARC | Read deeply |
| 3 | Event-triggered inference (2021) | arXiv:2109.05601 | Control | Trigger compute on events | Bounded updates | Trigger threshold | Lower compute | Gate policy | Read deeply |
| 4 | Multi-sensor event triggering (2020) | arXiv:2003.05788 | Control | Composite triggers | Stability w/ dwell time | Composite score | Lower latency | Multi-signal gate | Read deeply |
| 5 | Bounded staleness caches (2018) | arXiv:1806.10254 | Systems | Max staleness constraints | Bounded error | Staleness bound | Freshness preserved | Cache policy | Read deeply |
| 6 | Calibration (2017) | ICML 2017 | ML | Temperature scaling | Calibrated confidence | Softmax/T | Reliable uncertainty | Confidence gating | Read |
| 7 | DeeAD (2025) | arXiv:2511.20720 | VLA | Early exit | Speed/quality bound | Exit threshold | 1.5x speed | LARA | Read |
| 8 | A-ViT (2022) | arXiv:2112.07658 | Vision | Adaptive compute | Latency reduction | Halting score | Faster ViT | LARA | Read |
| 9 | Decision-point detection | arXiv:2007.00696 | VLN | Action-critical frames | High recall | Decision gate | Better navigation | DARC | Read |
| 10 | Event-triggered control | arXiv:1901.07806 | Control | Dwell time triggers | Stability | Trigger threshold | Bounded updates | DARC | Read |
| 11 | Bounded staleness caches | arXiv:1806.10254 | Systems | Max staleness | Bounded error | Staleness bound | Freshness preserved | C3I | Read |
| 12 | MC Dropout | ICML 2016 | ML | Uncertainty | Variance bound | Sample variance | Calibrated uncertainty | C3I | Read |
| 13 | Temperature scaling | ICML 2017 | ML | Calibration | Confidence alignment | Softmax/T | Reliable confidence | Q-MoE | Read |
| 14 | Waypoint prediction | arXiv:2002.01641 | VLN | Learnable waypoints | Improved SR | Waypoint loss | Better planning | DARC/T-Flow | Read |
| 15 | Spec-VLA | arXiv:2507.22424 | VLA | Speculative decoding | Speedup bound | Acceptance length | 1.42x speed | Async extensions | Read |

---

### Paper-Matrix Subtables (by Domain)

**A. Async Control + Event-Triggered Inference**
| # | Paper | ArXiv/DOI | Trigger Signal | Guarantee | Notes |
|---|-------|-----------|----------------|-----------|-------|
| A1 | Event-triggered inference | arXiv:2109.05601 | action/event | bounded updates | core baseline |
| A2 | Multi-sensor event trigger | arXiv:2003.05788 | composite | stability w/ dwell | core baseline |
| A3 | Event-triggered control | arXiv:1901.07806 | dwell time | stability | used in DARC |

**B. Decision-Point Detection / VLN**
| # | Paper | ArXiv/DOI | Signal | Guarantee | Notes |
|---|-------|-----------|--------|-----------|-------|
| B1 | Decision-point detection | arXiv:2007.00696 | action transitions | high recall | DARC | 
| B2 | Waypoint prediction | arXiv:2002.01641 | waypoint cues | improved VLN | A2/A4 |

**C. Cache Staleness / Systems**
| # | Paper | ArXiv/DOI | Constraint | Guarantee | Notes |
|---|-------|-----------|-----------|-----------|-------|
| C1 | Bounded staleness caches | arXiv:1806.10254 | max staleness | bounded error | C3I |

**D. Calibration + Uncertainty**
| # | Paper | ArXiv/DOI | Method | Guarantee | Notes |
|---|-------|-----------|--------|-----------|-------|
| D1 | Temperature scaling | ICML 2017 | scalar T | calibrated confidence | Q-MoE/C3I |
| D2 | MC Dropout | ICML 2016 | stochastic | uncertainty bound | uncertainty gate |

**E. Adaptive Compute / Early Exit**
| # | Paper | ArXiv/DOI | Method | Guarantee | Notes |
|---|-------|-----------|--------|-----------|-------|
| E1 | DeeAD | arXiv:2511.20720 | exit threshold | speed/quality | LARA |
| E2 | A-ViT | arXiv:2112.07658 | token halting | latency reduction | LARA |

---

### 2026-05-06 — Action-aware temporal cache + offline metrics
Focused on literature around event-triggered inference, decision-point detection in VLN, bounded-staleness caching, and offline counterfactual evaluation metrics. Main goal was to replace similarity-only caching with action-aware gates and to define offline metrics that detect action suppression and replay bias.

**Top 5 papers (action-aware + metrics):**
1. Decision-point detection for VLN (arXiv:2007.00696)
2. Event-triggered inference for embodied control (arXiv:2109.05601)
3. Multi-sensor event triggering (arXiv:2003.05788)
4. Bounded-staleness caches (arXiv:1806.10254)
5. Event-triggered control with dwell time (arXiv:1901.07806)
