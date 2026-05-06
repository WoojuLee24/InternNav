---
allowed-tools: Read, Bash(cat *), Bash(grep*), Bash(git log*), Bash(git branch*), Bash(curl*)
description: Current verified async system state — active innovations, gate results, deployment config. Run at start of any session.
---

## Branch & Commit State
**Current branch**: !`git branch --show-current`
**Recent commits**: !`git log --oneline -8`

## Gate Status Summary
!`grep -A2 "^## [✅⚠️📋🔄]" docs/GATE_STATUS.md 2>/dev/null | head -60`

## Live Server Metrics (if running)
!`curl -s http://localhost:5802/async_metrics 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'  hz={d[\"joint_req_hz\"]:.2f}  latency={d[\"joint_latency_ms\"]:.3f}ms  bg_runs={d[\"background_s2_runs\"]}')
    print(f'  thr={d[\"temporal_cache_threshold\"]}  max_hold={d[\"max_hold_frames\"]}  skip%={d[\"temporal_cache_skip_ratio\"]:.1f}')
    print(f'  AA_bypasses={d[\"action_aware_bypasses\"]}  MH_bypasses={d[\"max_hold_bypasses\"]}')
    print(f'  waiting_responses={d[\"waiting_responses\"]}  pre_warm_queued={d[\"pre_warm_frames_queued\"]}')
except: print('  [Server not running or metrics endpoint error]')
" 2>/dev/null || echo "  [Server not running on localhost:5802]"`

---

## Verified System State (2026-05-07)

### TRUE Async is IMPLEMENTED and VERIFIED (Gate 0 ✅)
The HTTP handler `/eval_dual_async` does NOT call `agent.step()`.
S2 runs in a background thread (`async_continuous_loop`).
HTTP latency: **0.02ms** (not 300ms). Throughput: **12 Hz**.

### Active Innovation Stack (Gate 3b ✅ + Gate 3c 🔄)

**Gate 3b — Temporal S2 cache (PASS)**
```
Background thread gate before each S2 call:
  1. I-046 (one-shot): force fresh if last output was action
  2. I-047 (max-hold): force fresh if skipped ≥ max_hold_frames consecutively
  3. cosine_sim(current, last) ≥ threshold → SKIP (serve cache)

Result: 70% S2 reduction, V<0.10, Hz maintained
Runtime control:
  curl http://localhost:5802/set_temporal_threshold?threshold=0.92
  curl http://localhost:5802/set_max_hold_frames?frames=10
```

**Gate 3c — Cold-start pre-fetch (🔄 in progress)**
```
At server startup (before first real request):
  --pre-warm-frames 3 → queues 3 synthetic S2 runs
  Cache pre-populated by t=0.9s
  Eliminates "waiting" responses at bag start

Metric: waiting_responses (in /async_metrics)
```

### Production Deployment Config
```bash
python3 scripts/realworld/http_internvla_server_debug.py \
    --mode async \
    --temperature 0.75 \
    --kv-cache \
    --calib scripts/realworld/calib/calib_scout.txt \
    --pre-warm-frames 3   # Gate 3c

# After server starts:
curl http://localhost:5802/set_temporal_threshold?threshold=0.92  # Gate 3b
curl http://localhost:5802/set_max_hold_frames?frames=10          # Gate 3b
```

### Disabling Cache (back to Gate 0 behavior):
```bash
curl http://localhost:5802/set_temporal_threshold?threshold=0.0
```

---

## Task
Given `$ARGUMENTS` (or no arguments for full report):

1. **GATE MAP**: Print current pass/fail for all gates
2. **ACTIVE CONFIG**: Exact flags/thresholds for production
3. **METRICS**: Interpret live /async_metrics dump
4. **NEXT EXPERIMENT**: What gate to run next and why

See `docs/GATE_STATUS.md` for full gate documentation.
See `docs/ARCHITECTURE.md` for before/after diagrams of each innovation.
