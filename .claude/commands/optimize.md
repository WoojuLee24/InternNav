---
allowed-tools: Read, Bash(cat *), Bash(grep*), Bash(git log*), Bash(git diff*), Bash(date*)
description: Optimization tracking — log results, analyze bottlenecks, suggest next optimization for InternNav async dual-system. Usage: /optimize [log|analyze|bottleneck|roadmap]
---

## Optimization State

**Current branch**: !`git branch --show-current`

**Key metrics history**:
!`grep -E "Hz|latency|trajectory|s2_req|joint_req|s1_req" OPTIMIZATION_TRACKING.md 2>/dev/null | head -40`

**Async-specific tracking**:
!`grep -E "Hz|latency|trajectory|ASYNC|TRUE ASYNC|background" optimization_tracking_async.md 2>/dev/null | head -50`

**Recent code changes to agent/server**:
!`git log --oneline --follow internnav/agent/internvla_n1_agent.py scripts/realworld/http_internvla_server*.py 2>/dev/null | head -10`

---

## Task

`$ARGUMENTS` — one of: `log`, `analyze`, `bottleneck`, `roadmap`, or paste raw metrics directly.

---

### Mode: `log` or raw metrics pasted

Parse the metrics and append to OPTIMIZATION_TRACKING.md in structured format:

```markdown
## [MODE_NAME] — [DATE]
**Commit**: [git hash]
**Config**: [server script + flags used]

| Metric | Value | vs SYNC Baseline | Status |
|--------|-------|-----------------|--------|
| joint_req_hz | X.XX Hz | +/-X% | ✅/⚠️/❌ |
| s2_req_hz | X.XX Hz | +/-X% | |
| s2_latency_ms | XXX ms | +/-X% | |
| trajectory_ratio | X.X% | +/-X pp | |
| s1_req_hz | X.XX Hz | +/-X% | |

**SYNC baseline reference**: joint_req_hz=2.81, s2_latency=495ms, trajectory_ratio=52.8%

**Key observation**: [What is the most important finding?]
**Quality gate**: trajectory_ratio ≥ 45%? [PASS/FAIL]
```

---

### Mode: `analyze`

Deep analysis of all logged results:

1. **Speed progress**: Plot the optimization journey (joint_req_hz over experiments)
2. **Quality stability**: Is trajectory_ratio holding above 50%?
3. **Latency breakdown**: Where is time actually spent?
   - S2 inference (VLM forward pass)
   - S2→S1 communication overhead
   - S1 trajectory generation
   - HTTP round-trip
4. **Correlation analysis**: Does lowering s2_latency_ms correlate with trajectory_ratio drop?
5. **Current ceiling**: What is the theoretical maximum Hz given model inference time?

---

### Mode: `bottleneck`

Identify the current #1 bottleneck:

**Known bottleneck hierarchy for this system:**
1. VLM forward pass (S2 inference) — ~233-495ms — hard wall without quantization
2. HTTP round-trip overhead — measurable with profiling
3. Thread lock contention — `s2_input_lock`, `s2_output_lock`, `s2_agent_lock`
4. S1 trajectory generation — currently ~0ms (passthrough?)
5. ROS2 message serialization — at high Hz

For each, estimate: Is this the current bottleneck? What would removing it gain?

---

### Mode: `roadmap`

Based on current results, generate a prioritized optimization roadmap:

**Tier 1 — No quality risk** (do immediately):
- Thread contention reduction
- Memory pre-allocation
- Batching strategies

**Tier 2 — Measured quality risk** (experiment first):
- Reducing S2 invocation frequency
- Speculative S2 pre-fetch
- S2 output caching / reuse

**Tier 3 — Architectural changes** (research contribution):
- Learned S2 scheduling policy
- Adaptive S1/S2 coupling
- Distillation of S2 into S1

For each tier, specify: expected Hz gain, expected trajectory_ratio impact, implementation complexity.
