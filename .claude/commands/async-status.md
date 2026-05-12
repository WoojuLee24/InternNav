---
allowed-tools: Read, Bash(cat *), Bash(grep*), Bash(git log*), Bash(git branch*), Bash(curl*), Bash(docker ps*)
description: Current verified async system state — active innovations, gate results, deployment config. Run at start of any session.
---

## Branch & Commit State
**Current branch**: !`git branch --show-current`
**Recent commits**: !`git log --oneline -8`

## Overall Progress
!`head -12 PLAN.md 2>/dev/null`

## Gate Pipeline
!`head -12 docs/GATE_STATUS.md 2>/dev/null`

## Container & Live Metrics
!`docker ps --format "{{.Names}}: {{.Status}}" 2>/dev/null | grep vlnav || echo "[container not running]"`
!`curl -s http://localhost:5802/async_metrics 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'  hz={d[\"joint_req_hz\"]:.2f}  latency={d[\"joint_latency_ms\"]:.3f}ms  bg_runs={d[\"background_s2_runs\"]}')
    print(f'  thr={d[\"temporal_cache_threshold\"]}  MH={d[\"max_hold_frames\"]}  skip%={d[\"temporal_cache_skip_ratio\"]:.1f}')
    print(f'  AA={d[\"action_aware_bypasses\"]}  SP={d.get(\"slope_predict_bypasses\",0)}  EMA_alpha={d.get(\"ema_fingerprint_alpha\",\"off\")}')
except: print('  [Server not running]')
" 2>/dev/null || echo "  [Server not running on localhost:5802]"`

---

## Verified System State (Gate 3n Production Stack)

### TRUE Async — PASS (Gate 0)
HTTP handler: cache-only, 0.02ms latency, 12 Hz throughput.
Background thread: `async_continuous_loop` runs S2 at ~12 Hz, writes cache.

### Production Stack (Gate 3n PASS, 2026-05-11)
```
τ=0.92              → temporal similarity gate (Gate 3j)
max_hold=15         → forced refresh ceiling (Gate 3g)
action_aware=true   → I-046 one-shot post-action refresh (Gate 3f)
tr_ema α=0.10       → EMA fingerprint with bypass reset (Gate 3m)
slope δ_s=0.010     → predictive refresh on falling similarity (Gate 3l)

Result: 90.8–91.4% skip, V_max=0.0791, SNR=19.67×
NOT in production: I-058 (odom-progress, inert at useful thresholds)
```

### Production Deployment Commands
```bash
# In container:
python3 scripts/realworld/http_internvla_server_debug.py \
    --mode async --temperature 0.75 --kv-cache \
    --calib scripts/realworld/calib/calib_scout.txt \
    --pre-warm-frames 3

# Configure Gate 3n stack:
curl "http://localhost:5802/set_temporal_threshold?threshold=0.92"
curl "http://localhost:5802/set_max_hold_frames?frames=15"
curl "http://localhost:5802/set_action_aware?enabled=true"
curl "http://localhost:5802/set_ema_fingerprint?enabled=true&alpha=0.10&transition_reset=true"
curl "http://localhost:5802/set_slope_predict?enabled=true&threshold=0.010&window=3"

# Disable cache (Gate 0 / NOCACHE baseline):
curl "http://localhost:5802/set_temporal_threshold?threshold=0.0"
curl "http://localhost:5802/set_max_hold_frames?frames=0"
curl "http://localhost:5802/set_action_aware?enabled=false"
curl "http://localhost:5802/set_ema_fingerprint?enabled=false"
curl "http://localhost:5802/set_slope_predict?enabled=false"
```

---

## Task
Given `$ARGUMENTS` (or no arguments for full report):

1. **GATE MAP**: Print current pass/fail status for all gates
2. **ACTIVE CONFIG**: Exact flags/thresholds currently running
3. **NEXT PRIORITY**: What needs to happen to get from 15% → 25% (Phase 4 or Phase 3X)
4. **METRICS**: Interpret live /async_metrics if server running

See `docs/GATE_STATUS.md` for full gate documentation.
See `PLAN.md` for Phase 3X / Phase 4 / Phase 5 roadmap.
