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

