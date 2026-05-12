#!/bin/bash
# UserPromptSubmit hook: injects relevant research context based on message keywords

input=$(cat 2>/dev/null)

PROMPT=$(echo "$input" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('message', '').lower())
except Exception:
    print('')
" 2>/dev/null)

ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")

# Gate/experiment keywords → inject current gate status
if echo "$PROMPT" | grep -qE "gate|experiment|run|bag|rosbag|container|docker|skip|v_max|cramers|metric"; then
    echo "=== Gate Pipeline Status ==="
    head -12 "$ROOT/docs/GATE_STATUS.md" 2>/dev/null
    echo "==========================="
fi

# Research/innovation keywords → inject roadmap state
if echo "$PROMPT" | grep -qE "next|plan|what.*do|research|innovate|hypothesis|phase|blocker|progress|stuck"; then
    echo "=== Research Plan (current state) ==="
    head -35 "$ROOT/PLAN.md" 2>/dev/null
    echo "====================================="
fi

# Paper/literature keywords → inject RESEARCH_TRACKING summary
if echo "$PROMPT" | grep -qE "paper|arxiv|literature|sota|baseline|speculative|distillation|kv.cache"; then
    echo "=== Literature (recent entries) ==="
    grep -A3 "Relevance: 5/5" "$ROOT/RESEARCH_TRACKING.md" 2>/dev/null | head -30
    echo "==================================="
fi

# Production stack keywords → inject current config
if echo "$PROMPT" | grep -qE "production|server|config|threshold|max.hold|ema|slope|stack"; then
    echo "=== Production Stack (Gate 3n) ==="
    echo "τ=0.92, MH=15, AA=true, TR-EMA α=0.10, slope δ_s=0.010"
    echo "V_max=0.0791, skip=91%, hz=11.8-12.2, V noise floor=0.007"
    echo "NOT in production: I-058 (odom-progress), I-052 (serve-count)"
    echo "=================================="
fi
