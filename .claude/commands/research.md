---
allowed-tools: Read, Bash(cat *), Bash(git log*), Bash(git branch*), Bash(grep*)
description: Research context — full research state, proven results, next experiments, open questions. Use at start of any research session.
---

## Research State
**Branch**: !`git branch --show-current`
**Recent commits**: !`git log --oneline -8`

**Gate status**:
!`grep "^## [✅⚠️📋🔄]" docs/GATE_STATUS.md 2>/dev/null`

**Experiment logs**:
!`ls stats/*/FINDINGS.md 2>/dev/null | xargs -I{} sh -c 'echo "--- {} ---"; tail -5 {}'`

---

## Research Philosophy

**80% research / 20% engineering.**
- Every code change needs a testable hypothesis.
- Gate = hypothesis + 3-bag experiment + PASS/FAIL decision.
- `trajectory_ratio` and `action_rate` are primary quality signals.
- True async (Gate 0) is the foundation — never regress on latency or Hz.

---

## System Architecture Summary

```
S1 (HTTP client, 12 Hz) ──► /eval_dual_async ──► return cache (0.02ms)
                                   │
                                   └──► queue for S2 background thread
                                              │
                                         Gate 3b cache gate
                                         (cosine sim, I-046, I-047)
                                              │ ~30% frames pass
                                         InternVLA-N1 S2 (~300ms)
                                              │
                                         async_cache update
```

Gate 3c adds: synthetic pre-warm at startup → cache ready before first real request.

---

## Proven Results (with commit references)

| Gate | What | Metric | Before | After | Commit |
|------|------|--------|--------|-------|--------|
| 0    | TRUE async | latency | 300ms | 0.02ms | `942059f1` |
| 0    | TRUE async | hz | 2.5 Hz | 12 Hz | `942059f1` |
| 1    | temp=0.75 | quality | baseline | locked | `9b82bae2` |
| 3b   | temporal cache | S2_runs | 2000-2300 | 130-620 | `6a09126b` |
| 3b   | temporal cache | action bias V | — | <0.10 | `6a09126b` |

---

## Open Research Questions (ranked by importance)

1. **Does trajectory_ratio correlate with SR/SPL?** (Gate 4, unresolved)
   - We assume trajectory_ratio is a quality proxy. Needs VLN benchmark validation.
   
2. **Is the 70% S2 reduction actually safe in closed-loop?**
   - Rosbag shows V<0.10 bias. But does the 30% of missed action frames matter?
   - Need closed-loop sim experiment (Habitat or Isaac).

3. **Can we push S2 reduction to 85%+ without quality loss?**
   - Larger max_hold (20-30)? Smarter gate (depth entropy)?
   - Gate 3b max_hold=10 is conservative.

4. **What is the minimum information S2 needs per fresh call?**
   - Could S2 use compressed/quantized vision tokens at cache-miss frames?
   - Connects to I-021 (speculative decoding) and I-022 (ViSpec).

5. **Can S1 be trained to be robust to variable-frequency S2 updates?**
   - With 70% S2 skip, S1 sometimes acts on stale S2 guidance for 10+ frames.
   - Training S1 with variable S2 dropout could improve robustness (Gate 5).

---

## Next Experiments (ranked by expected information gain)

### 1. Gate 3c — Cold-Start Pre-Fetch (ready to run)
```
Script: scripts/realworld/gate3c_prewarm_3bag.sh
Time:   ~60 min (3 bags × 2 conditions)
Outcome: Confirm waiting_responses=0 with --pre-warm-frames 3
```

### 2. Gate 3b max_hold sweep (2 hours)
```
Test max_hold ∈ {5, 10, 20, 50} while holding thr=0.92
Metric: S2 reduction, Cramer's V, hz
Goal: find knee of S2-reduction vs quality-bias curve
```

### 3. Gate 4 — VLN Benchmark (1-2 weeks)
```
Run Gate 0 and Gate 3b configs on R2R val-unseen (Habitat)
Metric: SR, SPL, NE
Goal: confirm trajectory_ratio → SPL correlation hypothesis
Blocked on: Habitat setup + R2R data
```

---

## If `$ARGUMENTS` is provided

Answer it in the research context above. Connect to:
- Relevant innovation in INNOVATIONS_CATALOGUE.md
- Related gate and commit
- Open question it addresses
- Experimental design (if new hypothesis)
