#!/bin/bash
# PostToolUse hook: detects performance metrics or gate results and surfaces context

input=$(cat 2>/dev/null)

python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    output = str(data.get('tool_response', {}).get('output', ''))

    # Gate pass/fail signals
    if 'PASS' in output or 'FAIL' in output or 'GATE3' in output or 'GATE4' in output:
        print('🚦 Gate result detected. Run /experiment log to record this result in EXPERIMENTS_LOG.md.')

    # Performance metrics
    perf_keys = ['req_hz', 'latency_ms', 'trajectory_ratio', 'joint_req', 'Hz', 'skip', 'V_max', 'cramers', 'bg=']
    if any(k in output for k in perf_keys) and len(output) > 50:
        print('📊 Metrics detected. Run /experiment log to persist results.')

    # Error signals in server/container output
    if 'Error' in output and ('NameError' in output or 'ImportError' in output or 'UnboundLocal' in output):
        print('⚠️  Python error in server output — likely missing import or global declaration. Check imports and global statements before re-running.')

    # Low bg_s2_runs signal
    import re
    bg_match = re.search(r'bg=(\d+)', output)
    if bg_match and int(bg_match.group(1)) < 10:
        print('⚠️  bg_s2_runs very low — server may have a NameError caught silently. Run: docker exec vlnav_internvla_server bash -c \"grep Error /tmp/gate*/server.log | head -5\"')
except Exception:
    pass
" 2>/dev/null
