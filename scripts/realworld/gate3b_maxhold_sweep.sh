#!/usr/bin/env bash
# Gate 3b Extension — max_hold parameter sweep (I-047 tune)
# Find the Cramer's V vs S2-reduction knee.
# Tests max_hold ∈ {3, 5, 10, 20, 30} on a single bag (073623) to map the curve.
# 3-bag cross-validation only for the winner.
#
# Pass criterion: V <= 0.10 AND S2_reduction >= 40% AND hz >= 11.0
# thr=0.92 fixed (Gate 3b winner). I-046 one-shot always active.
#
# Run on GPU 1, port 5803 (so Gate 3c can run concurrently on port 5802).
set -e
source /opt/ros/jazzy/setup.bash

CALIB="/workspace/InternNav/scripts/realworld/calib/calib_scout.txt"
SERVER="scripts/realworld/http_internvla_server_debug.py"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
LOG_BASE="/tmp/gate3b_mh_sweep"
PORT=5803   # separate port so Gate 3c (5802) can run in parallel
mkdir -p "$LOG_BASE"

# Single sweep bag + 3-bag cross-validation for winner
SWEEP_BAG="my_camera_bag_20260317_073623"
XVAL_BAGS=(
    "my_camera_bag_20260317_073623"
    "my_camera_bag_20260317_061841"
    "my_camera_bag_20260317_063047"
)

MAX_HOLDS=(3 5 10 20 30)

echo "=================================================="
echo " Gate 3b Extension — max_hold sweep (I-047)"
echo " thr=0.92 fixed, sweep max_hold on bag 073623"
echo " Port: $PORT (parallel with Gate 3c on 5802)"
echo "=================================================="

start_server() {
    local mh="$1"
    local tag="$2"
    pkill -f "http_internvla_server_debug.*5803\|http_internvla_server_debug.*port.*5803" 2>/dev/null || true
    # Kill anything on port 5803
    fuser -k 5803/tcp 2>/dev/null || true
    sleep 2

    python3 /workspace/InternNav/$SERVER \
        --mode async --temperature 0.75 --kv-cache \
        --calib "$CALIB" \
        --device cuda:1 \
        --port "$PORT" \
        > "$LOG_BASE/${tag}_server.log" 2>&1 &
    SERVER_PID=$!

    until curl -sf http://localhost:${PORT}/async_metrics > /dev/null 2>&1; do sleep 2; done
    echo "  server up on ${PORT} (pid=$SERVER_PID)"

    curl -sf "http://localhost:${PORT}/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:${PORT}/set_max_hold_frames?frames=${mh}" > /dev/null
    echo "  thr=0.92 max_hold=$mh"
}

run_one() {
    local bag="$1"
    local mh="$2"
    local tag="${bag##*_}_mh${mh}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=${bag##*_}  max_hold=$mh ---"

    start_server "$mh" "$tag"

    # Baseline control for this bag
    local ctrl_dir="$LOG_BASE/${bag##*_}_ctrl"
    if [ ! -f "$ctrl_dir/metrics.json" ]; then
        mkdir -p "$ctrl_dir"
        curl -sf "http://localhost:${PORT}/set_temporal_threshold?threshold=0.0" > /dev/null
        curl -sf http://localhost:${PORT}/reset_metrics > /dev/null
        sleep 1

        python3.12 /workspace/InternNav/$CLIENT \
            --mode async --kv-cache --temperature 0.75 \
            --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
            > "$ctrl_dir/client.log" 2>&1 &
        local cpid=$!; sleep 3
        ros2 bag play "/workspace/rosbag/$bag" --rate 0.5 > "$ctrl_dir/bag.log" 2>&1
        sleep 2
        curl -sf http://localhost:${PORT}/async_metrics > "$ctrl_dir/metrics.json"
        kill $cpid 2>/dev/null || true
        echo "  [ctrl] saved"
        sleep 3

        # Re-set to test config
        curl -sf "http://localhost:${PORT}/set_temporal_threshold?threshold=0.92" > /dev/null
        curl -sf "http://localhost:${PORT}/set_max_hold_frames?frames=${mh}" > /dev/null
    fi

    curl -sf http://localhost:${PORT}/reset_metrics > "$log_dir/reset.json"
    sleep 1

    python3.12 /workspace/InternNav/$CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
        > "$log_dir/client.log" 2>&1 &
    local client_pid=$!; sleep 3

    ros2 bag play "/workspace/rosbag/$bag" --rate 0.5 > "$log_dir/bag.log" 2>&1
    sleep 2
    curl -sf http://localhost:${PORT}/async_metrics > "$log_dir/metrics.json"
    kill $client_pid 2>/dev/null || true

    python3 - <<PYEOF
import json, math
with open("$log_dir/metrics.json") as f: t = json.load(f)
with open("$ctrl_dir/metrics.json") as f: c = json.load(f)
c_bg  = c.get("background_s2_runs",0)
t_bg  = t.get("background_s2_runs",0)
c_ft  = c.get("fresh_traj_outputs",0); c_fa = c.get("fresh_action_outputs",0); c_n = c_ft+c_fa
t_ft  = t.get("fresh_traj_outputs",0); t_fa = t.get("fresh_action_outputs",0); t_n = t_ft+t_fa
skip  = t.get("temporal_cache_skip_ratio",0)
hz    = t.get("joint_req_hz",0)
s2r   = (1-t_bg/max(1,c_bg))*100

# Cramer's V inline
def chi2_cramers(a,b,c,d):
    n=a+b+c+d
    if n==0: return 0.0,1.0,0.0
    r1,r2=a+b,c+d; c1,c2=a+c,b+d
    if min(r1,r2,c1,c2)==0: return 0.0,1.0,0.0
    ea=r1*c1/n; eb=r1*c2/n; ec=r2*c1/n; ed=r2*c2/n
    chi2=sum((max(0,abs(o-e)-0.5)**2)/e for o,e in ((a,ea),(b,eb),(c,ec),(d,ed)))
    import math; p=math.erfc(math.sqrt(chi2/2))
    V=math.sqrt(chi2/n)
    return chi2,p,V

chi2,p,V=chi2_cramers(c_ft,c_fa,t_ft,t_fa)
status='PASS' if V<=0.10 and t_bg<=0.6*c_bg and hz>=11.0 else 'FAIL'
print(f"  mh=$mh: S2-red={s2r:.1f}%  skip={skip:.1f}%  V={V:.4f}  hz={hz:.2f}  => {status}")
PYEOF

    pkill -f "$SERVER" 2>/dev/null || true
    sleep 5
}

# Phase 1: Single-bag sweep to find knee
echo ""
echo "=== Phase 1: Single-bag sweep (073623) ==="
for mh in "${MAX_HOLDS[@]}"; do
    run_one "$SWEEP_BAG" "$mh"
done

# Phase 2: Summary table and winner selection
echo ""
echo "=== Phase 2: Sweep Summary — Cramer's V vs S2-Reduction ==="
echo "max_hold | S2_reduction | skip% | V     | hz    | verdict"
echo "---------|-------------|-------|-------|-------|--------"
python3 - <<PYEOF
import json, math, os, glob

def chi2_cramers(a,b,c,d):
    n=a+b+c+d
    if n==0: return 0.0,1.0,0.0
    r1,r2=a+b,c+d; c1,c2=a+c,b+d
    if min(r1,r2,c1,c2)==0: return 0.0,1.0,0.0
    ea,eb,ec,ed=r1*c1/n,r1*c2/n,r2*c1/n,r2*c2/n
    chi2=sum((max(0,abs(o-e)-0.5)**2)/e for o,e in((a,ea),(b,eb),(c,ec),(d,ed)))
    p=math.erfc(math.sqrt(chi2/2)); V=math.sqrt(chi2/n)
    return chi2,p,V

bag_short="073623"
ctrl_f=f"/tmp/gate3b_mh_sweep/{bag_short}_ctrl/metrics.json"
try:
    with open(ctrl_f) as f: c=json.load(f)
except: print("control not found"); exit()
c_bg=c.get("background_s2_runs",0)
c_ft=c.get("fresh_traj_outputs",0); c_fa=c.get("fresh_action_outputs",0)

results=[]
for mh in [3,5,10,20,30]:
    mf=f"/tmp/gate3b_mh_sweep/{bag_short}_mh{mh}/metrics.json"
    if not os.path.exists(mf): continue
    with open(mf) as f: t=json.load(f)
    t_bg=t.get("background_s2_runs",0)
    t_ft=t.get("fresh_traj_outputs",0); t_fa=t.get("fresh_action_outputs",0)
    s2r=(1-t_bg/max(1,c_bg))*100
    skip=t.get("temporal_cache_skip_ratio",0)
    hz=t.get("joint_req_hz",0)
    _,_,V=chi2_cramers(c_ft,c_fa,t_ft,t_fa)
    status='PASS' if V<=0.10 and t_bg<=0.6*c_bg and hz>=11.0 else 'FAIL'
    results.append((mh,s2r,skip,V,hz,status))
    print(f"  {mh:8d} | {s2r:10.1f}% | {skip:5.1f}% | {V:.4f} | {hz:.2f}  | {status}")

# Pick winner: highest S2 reduction among PASS conditions
winners=[r for r in results if r[5]=='PASS']
if winners:
    winner=max(winners, key=lambda r: r[1])
    print(f"\nWinner: max_hold={winner[0]} (S2_red={winner[1]:.1f}%, V={winner[3]:.4f})")
    with open("/tmp/gate3b_mh_sweep/winner.txt","w") as f:
        f.write(str(winner[0]))
else:
    print("\nNo PASS conditions found — all max_hold values produce V>0.10")
    with open("/tmp/gate3b_mh_sweep/winner.txt","w") as f:
        f.write("10")
PYEOF

# Phase 3: 3-bag cross-validation with winner max_hold
WINNER=$(cat /tmp/gate3b_mh_sweep/winner.txt 2>/dev/null || echo "10")
echo ""
echo "=== Phase 3: 3-bag xval with max_hold=$WINNER ==="
for bag in "${XVAL_BAGS[@]}"; do
    run_one "$bag" "$WINNER"
done

echo ""
echo "=== Gate 3b Extension — Final Summary ==="
python3 /workspace/InternNav/scripts/viz/chi_squared_action_test.py \
    /tmp/gate3b_mh_sweep/073623_ctrl/metrics.json \
    /tmp/gate3b_mh_sweep/073623_mh${WINNER}/metrics.json || true
echo "DONE"
