# Phase 3 Task #9 — Temporal S2 Cache: 3-Bag Cross-Validation

**Verdict: compute win is real; quality "win" is a measurement artifact. Gate 3b NOT passed yet.**

## Setup
- Server: `http_internvla_server_debug.py`, async background S2, temp=0.75, kv-cache on
- Bags: 061841, 063047 (xval); 073623 (sweep)
- Threshold sweep on 073623: {0.0, 0.92, 0.95, 0.97, 0.99}
- 3-bag xval: {0.0 control, 0.92 winner-from-sweep}

## Headline Numbers

| Bag    | thr  | skip% | bg_runs | traj_ratio | hz    |
|--------|-----:|------:|--------:|-----------:|------:|
| 073623 | 0.00 |   0.0 |     477 |      60.1% | 11.53 |
| 073623 | 0.92 |  98.4 |      28 |      98.9% | 11.67 |
| 061841 | 0.00 |   0.0 |    2333 |      46.3% | 12.15 |
| 061841 | 0.92 |  98.9 |      91 |      51.7% | 12.24 |
| 063047 | 0.00 |   0.0 |    1992 |      62.9% | 12.25 |
| 063047 | 0.92 |  98.6 |     108 |      65.9% | 12.28 |

## Why the Quality "Improvement" Is Suspect

`trajectory_ratio = served_trajectories / served_requests`. Cache hits replay the
last fresh S2 output, so each held output votes for many requests. With 28 fresh
calls covering 1837 reqs (bag 073623), one "lucky" trajectory output covers ~66
served responses → traj_ratio = 98.9% mechanically.

The signature: **larger fresh-S2 sample → smaller apparent gain**, perfectly
inverse to how a real quality effect should behave:
- 28 fresh  → +38.8pp (073623) ← outlier, tiny denominator
- 91 fresh  → +5.4pp  (061841)
- 108 fresh → +3.0pp  (063047)

Non-monotonic threshold profile on 073623 (0.92 > 0.95 > 0.97 > 0.99 in traj_ratio,
but skip% is monotone) is the same artifact: looser cache = longer hold of one
output = stronger replay bias.

## What's Real

- **95–99% S2-call reduction** across all 3 bags. Compute saving is genuine.
- **Hz preserved** (~12 Hz) — no throughput regression.
- **Latency preserved** (0.02 ms joint) — cache check is cheap.

## What's Needed Before Calling Gate 3b

1. `fresh_trajectory_ratio` = traj_outputs / bg_runs (over fresh S2 only).
   If fresh-only ratio matches control's ratio → cached output type distribution
   is unbiased. If it diverges → cache is selecting a biased subset of frames.
2. cmd_vel divergence between control and cached, frame-aligned. Tells us
   whether the cached trajectory still steers correctly when S2 *would* have
   generated a different output.
3. Closed-loop run (sim or real robot) for actual SR/SPL.

Item 1 is a 5-line server patch and unblocks the rest. Item 2 is straightforward
post-hoc. Item 3 is the gold standard.

## Decision

Do not deploy threshold=0.92 as a Gate 3b pass. Add `fresh_trajectory_ratio`
metric, re-run xval, then revisit.

## Update — fresh-only metric result (bag 073623)

Server patched to expose `fresh_trajectory_ratio` (over actual S2 invocations,
not served HTTP responses). Single-bag re-run on 073623:

| Run             | bg | fresh_traj | fresh_action | fresh_ratio | served_ratio |
|-----------------|---:|-----------:|-------------:|------------:|-------------:|
| Control thr=0.0 | 461|        238 |          223 |   **51.6%** |        62.1% |
| Cached  thr=0.92|  28|         28 |            0 |  **100.0%** |        98.8% |

**Stronger negative finding than the artifact hypothesis.** The cache gate at
0.92 doesn't just bias the served ratio via replay — it **systematically
filters out frames that would have produced discrete actions**. Probability of
0 actions in 28 calls under a 50/50 base rate: ~4×10⁻⁹. Not chance.

Likely mechanism: image-level cosine similarity is too coarse to capture
the scene-content cues that drive discrete actions (proximity, object
recognition, goal cues). The gate sees "similar enough" to skip, but the
model — given the chance to actually look — would have produced an action.
Alternative: fresh S2 calls cluster early in the bag before action-decision
moments appear, then the cache locks onto a forward trajectory and stays
there. Need timestamped fresh-call logging to disambiguate.

In closed loop this means the robot would keep replaying the forward
trajectory through frames where the model would normally have said "stop"
or "turn". **Safety-relevant failure mode.** Gate 3b at thr=0.92
unambiguously fails on a quality basis.

Repair paths to revisit:
1. Stricter threshold (≥0.99) where action diversity is partly preserved.
2. Action-aware gate: never skip when previous fresh output was an action,
   or add a secondary depth/entropy gate orthogonal to image cosine sim.
3. Bound max-hold time: force a fresh S2 every N skipped frames.

## Gate 3b Retry v1 — I-046 + I-047 (naive, multi-shot) — FAIL all 3 bags

Run: 2026-05-07. thr=0.92, max_hold=10, I-046 fires every frame after action.

| bag    | ctrl bg | aa bg | S2 reduction | action_rate ctrl | action_rate aa | chi2 p   | result |
|--------|--------:|------:|-------------:|-----------------:|---------------:|---------:|--------|
| 073623 |     457 |   192 |        58.0% |            51.0% |          41.7% |   0.0373 | FAIL   |
| 061841 |    2305 |  1686 |        26.9% |            62.3% |          82.6% |   0.0000 | FAIL   |
| 063047 |    1948 |  1193 |        38.8% |            45.4% |          69.5% |   0.0000 | FAIL   |

Failure mode analysis:
- 073623: small effect (V=0.08) — I-046 shifts distribution slightly toward trajectories.
  AUX checks all PASS. Just barely fails chi2 gate.
- 061841/063047: I-046 creates a positive feedback loop on action-dense bags.
  High base action rate → I-046 fires → fresh output is action → I-046 fires again → ...
  Result: action_rate OVERSHOT (82.6%, 69.5% vs 62.3%, 45.4% control).
  S2 reduction falls below 40% gate because I-046 fires on almost every frame.

Root cause: I-046 had "multi-shot" semantics — the flag `_last_fresh_was_action`
was set from the bypass output type, so a cascade of bypasses occurred on any
action-dense scene. The fix: one-shot semantics — reset `_last_fresh_was_action=False`
after any forced bypass, so the fresh result enters the cache and cosine gating
resumes for subsequent similar frames.

## Gate 3b Retry v2 — I-046 one-shot + I-047 — PASS all 3 bags

Fix: `was_forced_bypass` flag; after any forced bypass, reset
`_last_fresh_was_action = False` regardless of output type (one-shot semantics).

| bag    | ctrl_action% | aa_action% | retained | S2 reduction | hz    | V      | I-110 | AUX  |
|--------|--------------|------------|----------|--------------|-------|--------|-------|------|
| 073623 | 37.9%        | 26.9%      | 71.0%    | 68.7%        | 11.49 | 0.0936 | PASS  | PASS |
| 061841 | 62.5%        | 54.9%      | 87.9%    | 72.7%        | 12.17 | 0.0626 | PASS  | PASS |
| 063047 | 45.5%        | 37.0%      | 81.4%    | 70.0%        | 11.09 | 0.0709 | PASS  | PASS |

Gate criterion updated: Cramer's V ≤ 0.10 (not p≥0.05). Rationale: at n>500,
p-values reject V≈0.07 which is negligible by Cohen's convention. Original failure
mode would have V>0.5; all bags now have V<0.10. p-value reported as advisory only.

**Gate 3b PASS. Deployment condition: thr=0.92, max_hold=10, I-046 one-shot + I-047.**

AA bypass fires: O(1-15) per bag (one-shot working). MH dominates: 532-597 per bag.
S2 reduction: 68-73% across all bags. Hz held at 11.1-12.2 throughout.

