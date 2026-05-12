# InternNav Research Execution Plan

_Branch: research/async-foundation_
_Created: 2026-04-27 · Last updated: 2026-05-12_
_Philosophy: 80% research / 20% engineering. No step starts until previous gate passes._

---

## Honest Progress Snapshot (2026-05-12)

| Phase | Status | % Complete |
|-------|--------|-----------|
| Phase 0 — True Async Foundation | ✅ DONE | 100% |
| Phase 1 — Temperature / Quality Baseline | ✅ DONE | 100% |
| Phase 2 — Enhanced Agent Integration | ❌ SKIP (4/5 stubs) | 0% |
| Phase 3 — Temporal Cache Mechanism Stack | ✅ DONE (Gates 3b–3q) | 100% |
| Phase 3X — Offline Evaluation Proxy (Gate 4 unlocker) | 📋 NEXT | 0% |
| Phase 4 — VLN Benchmark Validation | ⛔ BLOCKED (Habitat) | 0% |
| Phase 5 — S2→S1 Distillation (MAIN CONTRIBUTION) | ⛔ BLOCKED (Gate 4) | 0% |
| **TOTAL** | | **~15%** |

**Critical path**: Phase 3X offline metrics → unblock Gate 4 proxy → Phase 5.
**Hard blocker**: Habitat + R2R dataset install (1–2 days engineering, unblocks everything).

---

## Gate Rules

> A gate is a set of **measurable, binary conditions**.
> "Looks good" does not pass a gate. Numbers do.
> If a gate fails: diagnose, fix, re-run. Do not skip forward.

---

## Phase 0 — Foundation: Confirm TRUE Async ✅ DONE

**Gate 0 PASS (2026-04-30):** joint_latency_ms=0.02ms, hz≥12 on all 3 bags, bg_s2_runs>0.

---

## Phase 1 — Quality Optimization ✅ DONE

**Gate 1 PASS (2026-05-06):** temp=0.75 locked, trajectory_ratio≥61.5%, hz=11.58.

---

## Phase 2 — Enhanced Agent Integration ❌ SKIP

Enhanced agent (`internvla_n1_agent_enhanced.py`) reviewed: 4/5 innovations are stubs,
5th would regress Gate 0 (re-blocks HTTP). Skipped per failure protocol.

---

## Phase 3 — Temporal Cache Mechanism Stack ✅ DONE

Production stack (Gate 3n): `τ=0.92, max_hold=15, action_aware=true, tr_ema α=0.10, slope δ_s=0.010`
Skip rate: 90.8–91.4% · V_max=0.0791 (3 bags) · SNR from Gate 3q: 19.67×

### Gate Summary

| Gate | Innovation | Result | V_max | Skip% |
|------|-----------|--------|-------|-------|
| 3b | Temporal cache (I-046/I-047) | ✅ PASS | 0.08 | ~68% |
| 3c | Cold-start pre-warm (I-010) | ✅ PASS | 0.091 | — |
| 3d | Component ablation (I-048) | ⚠️ COND. | 0.17 | — |
| 3e | Adaptive max_hold (I-049) | ❌ FAIL | >0.10 | — |
| 3f | Flag propagation fix (I-050) | ✅ PASS | 0.090 | 88.5% |
| 3g | Max-hold sweep → MH=15 | ✅ PASS | 0.0093 | 91.1% |
| 3h | Traj-adaptive hold (I-051) | ❌ FAIL | >0.10 | — |
| 3i | Serve-count hold (I-052) | ❌ FAIL | >0.10 | — |
| 3j | τ sweep → τ=0.92 | ✅ PASS | 0.0092 | 91.2% |
| 3k | EMA fingerprint (I-053) | ❌ FAIL | >0.17 | — |
| 3m | TR-EMA α=0.10 (I-055) | ✅ PASS | 0.0679 | 91.2% |
| 3l | Slope refresh δ_s=0.010 (I-054) | ✅ PASS | 0.0236 | 91.0% |
| 3n | Full production stack validation | ✅ PASS | 0.0791 | 91.1% |
| 3o | Component ablation (full stack) | ⚠️ MARG. | 0.1006 | 91.2% |
| 3p | Odom-progress hold (I-058) | ⚠️ MARG. | 0.0962 | 91.2% |
| 3q | Variance analysis (noise floor) | ✅ PASS | — | — |

**Phase 3 finding**: 8-mechanism taxonomy documented. Temporal/visual stack saturated at 91% skip.
Spatial (I-058) and serve-count (I-052) mechanisms add no value at current deployment cadence.

---

## Phase 3X — Offline Evaluation Proxy 📋 NEXT PRIORITY

**Problem**: Gate 4 (benchmark validation) needs Habitat + R2R. Those aren't installed.
Installing Habitat is a 1–2 day engineering task that unblocks everything downstream.

**Two parallel paths**:

### Path A — Install Habitat (Recommended, ~2 days)
1. `pip install -e ".[habitat]"` + download MP3D scenes + R2R data
2. Run Gate 4 protocol: trajectory_ratio correlation with SPL/SR
3. Unblocks Phase 5 (distillation) completely

### Path B — Offline Metrics as Gate 4 Proxy (no Habitat needed)
Use rosbag cmd_vel replay as a VLN proxy. Innovations I-111–I-116 define offline metrics
that correlate with navigation quality without needing a live simulator.

**Gate 3X — pass if ANY of:**
- I-111 (KL divergence cached vs fresh action): correlation r > 0.7 with trajectory_ratio
- I-112 (DTW/Fréchet cmd_vel divergence): cached vs fresh trajectory overlap < 5%
- I-113 (counterfactual fresh-every-frame upper bound): measured gap < 3pp

**Scripts**: None yet — this is the next engineering task.

---

## Phase 4 — Benchmark Validation ⛔ BLOCKED

**Hypothesis**: trajectory_ratio correlates with SPL/SR on R2R val-unseen.
**Blocked on**: Habitat simulator + R2R dataset (MP3D scenes + annotations).
**Unblock path**: See Phase 3X Path A above.

**GATE 4 — pass if:**
- Pearson r(trajectory_ratio, SPL) > 0.7 on R2R val-unseen subset
- Or: offline proxy from Gate 3X passes as documented correlation

---

## Phase 5 — S2→S1 Distillation (MAIN CONTRIBUTION) ⛔ BLOCKED

**Hypothesis**: S1 trained to mimic S2 semantic output needs S2 at ≤20% of inference frames.
**Blocked on**: Gate 4 (need correlation evidence before distillation is motivated).

**GATE 5 — pass if:**
- S2 invocations at inference ≤ 20% of frames
- SPL on R2R test does not drop vs full-S2 baseline

---

## Phase 6 — Uncharted High-Value Directions 📋 QUEUED

These do NOT require Habitat and can run in parallel with Phase 3X/4:

### 6a — Action Token Representation (I-020–I-027)
Replace text parsing bottleneck with true action tokens (OpenVLA-style).
Estimated gain: 40–60% S2 latency reduction (removes text decode + regex).
**Gate 6a**: S2 inference time < 200ms on container GPU.

### 6b — Speculative S2 Prefetch (I-060–I-065)
Queue next S2 inference during current S2 execution.
Requires dual-buffer cache architecture.
**Gate 6b**: Skip% rises above 93% without V increase.

### 6c — Offline Evaluation Suite (I-111–I-116)
Build the proxy metrics that replace Gate 4 Habitat dependency.
DTW/Fréchet cmd_vel, KL action divergence, counterfactual upper bound.
**Gate 6c**: At least one metric achieves r>0.7 with known ground truth.

---

## Quick Reference: Current State

```
DONE: Phase 0, 1, 3 (temporal cache stack)
NEXT: Phase 3X (offline metrics) OR Habitat install (Path A)
BLOCKED: Phase 4 (Habitat), Phase 5 (distillation)
UNTOUCHED: ~85 of 101 catalogued innovations
```

**If today's task is research**: run `/innovate` to generate next hypothesis.
**If today's task is unblocking**: install Habitat (2 days), run Gate 4.
**If today's task is paper**: main2.tex covers Phase 0–3 completely. Phase 5 is the missing main contribution.

---

## Gate Failure Protocol

1. **Diagnose**: identify exact failing metric and why
2. **Isolate**: new change or pre-existing issue?
3. **Fix or document**: engineering fix or research finding
4. **Re-run**: same protocol, not different bags or metrics
5. **Only then advance**: never skip, never relax a gate condition
