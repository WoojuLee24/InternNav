#!/usr/bin/env bash
# Gate 0 — single-bag verification inside container
# Usage: docker exec vlnav_internvla_server bash /workspace/InternNav/scripts/realworld/gate0_run.sh <bag_name> <run_id>
set -e
source /opt/ros/jazzy/setup.bash

BAG_NAME="${1:-my_camera_bag_20260317_073623}"
RUN_ID="${2:-run1}"
BAG_PATH="/workspace/rosbag/${BAG_NAME}"
CALIB="/workspace/InternNav/scripts/realworld/calib/calib_scout.txt"
LOG_DIR="/tmp/gate0_${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "[Gate0] =========================================="
echo "[Gate0] Bag  : $BAG_NAME"
echo "[Gate0] RunID: $RUN_ID  Logs: $LOG_DIR"
echo "[Gate0] =========================================="

# 1. Reset metrics via dedicated endpoint (no image needed)
curl -sf http://localhost:5802/reset_metrics > "$LOG_DIR/reset.json"
echo "[Gate0] Metrics reset: $(cat $LOG_DIR/reset.json)"
sleep 1

# 2. Start client in background
python3.12 /workspace/InternNav/scripts/realworld/http_internvla_client_debug.py \
  --mode async \
  --kv-cache \
  --temperature 0.8 \
  --jpeg-quality 95 \
  --depth-png-compress 6 \
  --calib "$CALIB" \
  > "$LOG_DIR/client.log" 2>&1 &
CLIENT_PID=$!
echo "[Gate0] Client pid=$CLIENT_PID — waiting 3s..."
sleep 3

# 3. Play bag at rate=0.5 (gate protocol requires this)
echo "[Gate0] Playing bag at rate=0.5 ..."
ros2 bag play "$BAG_PATH" --rate 0.5 > "$LOG_DIR/bag.log" 2>&1
echo "[Gate0] Bag done. Collecting metrics..."

# 4. Final metrics
sleep 2
curl -sf http://localhost:5802/async_metrics > "$LOG_DIR/metrics_final.json"

# 5. Kill client
kill $CLIENT_PID 2>/dev/null || true

# 6. Gate check
cat "$LOG_DIR/metrics_final.json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
keys = ['total_requests','joint_req_hz','joint_latency_ms','trajectory_ratio',
        'trajectories','discrete_actions','background_s2_runs','elapsed_seconds']
print('='*56)
print('  BAG: ${BAG_NAME}')
for k in keys:
    if k in d:
        print(f'  {k:<32}: {d[k]}')
print('='*56)
traj = d.get('trajectory_ratio', 0)
lat  = d.get('joint_latency_ms', 9999)
hz   = d.get('joint_req_hz', 0)
bg   = d.get('background_s2_runs', 0)
results = [
    ('joint_latency_ms < 20   ', lat < 20,   f'{lat:.2f} ms'),
    ('trajectory_ratio >= 50% ', traj >= 50, f'{traj:.1f}%'),
    ('joint_req_hz >= 10      ', hz >= 10,   f'{hz:.2f} Hz'),
    ('background_s2_runs > 0  ', bg > 0,     str(bg)),
]
all_pass = all(r[1] for r in results)
print()
print('GATE 0 CONDITIONS:')
for label, ok, val in results:
    print(f'  {\"PASS\" if ok else \"FAIL\"}  {label}: {val}')
print()
print(f'OVERALL: {\"ALL PASS\" if all_pass else \"FAIL — see above\"}')
"
