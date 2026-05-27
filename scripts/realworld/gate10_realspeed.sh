#!/usr/bin/env bash
# Gate 10 — Production Stack at Real Speed (I-205)
# Hypothesis: the Gate 3n production stack (τ=0.92, MH=15, TR-EMA, slope predict)
# maintains V ≤ 0.10 and skip ≥ 80% at rate=1.0× (real robot speed).
# At 1.0×, inter-frame interval = 83ms, S2 latency ~380ms → S2 lag = 4-5 frames.
# The async architecture serves cache during S2 inference, so lag is expected but
# the question is whether more rapid scene change drives V above threshold.
#
# Design: run all 3 bags at rate=1.0× with Gate 3n stack.
# Compare: V vs NOCACHE-1.0× baseline (separate NOCACHE run at rate=1.0×).
# Pass criterion: V ≤ 0.10 AND skip ≥ 70% (relaxed from 91% due to faster scene change).
# Port 5804, cuda:1.
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

PORT=5804
DEVICE=cuda:1
CALIB="scripts/realworld/calib/calib_scout.txt"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
BAG_DIR="/workspace/rosbag"
LOG_BASE="/tmp/gate10_realspeed"
mkdir -p "$LOG_BASE"

BAGS=(073623 061841 063047)

echo "=================================================="
echo " Gate 10 — Production Stack at Real Speed (I-205)"
echo " rate=1.0×  PROD vs NOCACHE  Port: $PORT"
echo "=================================================="

start_server() {
    local mode=$1 tag=$2
    pkill -f "http_internvla_server_debug.*$PORT" 2>/dev/null || true
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 3

    python3 /workspace/InternNav/scripts/realworld/http_internvla_server_debug.py \
        --mode async --temperature 0.75 --kv-cache \
        --calib "$CALIB" --device "$DEVICE" --port "$PORT" \
        --pre-warm-frames 3 \
        > "$LOG_BASE/${tag}_server.log" 2>&1 &
    SERVER_PID=$!
    echo "  Waiting for server (pid=$SERVER_PID)..."
    until curl -sf "http://localhost:${PORT}/async_metrics" > /dev/null 2>&1; do sleep 3; done
    echo "  Server ready on :${PORT}"

    if [ "$mode" == "prod" ]; then
        curl -sf "http://localhost:${PORT}/set_temporal_threshold?threshold=0.92" > /dev/null
        curl -sf "http://localhost:${PORT}/set_max_hold_frames?frames=15" > /dev/null
        curl -sf "http://localhost:${PORT}/set_action_aware?enabled=true" > /dev/null
        curl -sf "http://localhost:${PORT}/set_ema_fingerprint?enabled=true&alpha=0.10&transition_reset=true" > /dev/null
        curl -sf "http://localhost:${PORT}/set_slope_predict?enabled=true&threshold=0.010&window=3" > /dev/null
        echo "  Config: Gate 3n PROD stack  rate=1.0×"
    else
        # NOCACHE: threshold=0 disables temporal cache entirely
        curl -sf "http://localhost:${PORT}/set_temporal_threshold?threshold=0.0" > /dev/null
        curl -sf "http://localhost:${PORT}/set_max_hold_frames?frames=0" > /dev/null
        curl -sf "http://localhost:${PORT}/set_action_aware?enabled=false" > /dev/null
        echo "  Config: NOCACHE  rate=1.0×"
    fi
}

run_one() {
    local bag=$1 mode=$2
    local tag="${bag}_${mode}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$bag  mode=$mode  rate=1.0× ---"
    start_server "$mode" "$tag"

    curl -sf "http://localhost:${PORT}/reset_metrics" > "$log_dir/reset.json"
    sleep 1

    python3.12 /workspace/InternNav/$CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
        --server-port "${PORT}" \
        > "$log_dir/client.log" 2>&1 &
    local cpid=$!; sleep 3

    # rate=1.0 (real speed)
    ros2 bag play "$BAG_DIR/my_camera_bag_20260317_$bag" --rate 1.0 \
        --topics /camera/camera/color/image_raw \
                 /camera/camera/aligned_depth_to_color/image_raw \
                 /gdq/msg/gdq_odom > "$log_dir/bag.log" 2>&1 || true
    sleep 3
    curl -sf "http://localhost:${PORT}/async_metrics" > "$log_dir/metrics.json"
    kill $cpid 2>/dev/null || true

    python3 - <<PYEOF
import json, math
with open("$log_dir/metrics.json") as f: t = json.load(f)
ft = t.get("fresh_traj_outputs", 0); fa = t.get("fresh_action_outputs", 0)
n  = ft + fa; tr = 100 * ft / n if n > 0 else 0
skip = t.get("temporal_cache_skip_ratio", 0)
hz   = t.get("joint_req_hz", 0)
s2l  = t.get("s2_latency_ms", 0)
print(f"  bag=$bag mode=$mode: n={n}  skip={skip:.1f}%  traj={tr:.1f}%  s2_lat={s2l:.0f}ms  hz={hz:.2f}")
PYEOF

    pkill -f "http_internvla_server_debug.*${PORT}" 2>/dev/null || true
    sleep 5
}

# Phase 1: NOCACHE baseline at rate=1.0×
echo ""
echo "=== Phase 1: NOCACHE baseline at rate=1.0× ==="
for bag in "${BAGS[@]}"; do
    run_one "$bag" "nocache"
done

# Phase 2: PROD at rate=1.0×
echo ""
echo "=== Phase 2: PROD (Gate 3n) at rate=1.0× ==="
for bag in "${BAGS[@]}"; do
    run_one "$bag" "prod"
done

# Phase 3: Summary + Cramér's V
echo ""
echo "=== Phase 3: Gate 10 Summary ==="
echo "bag    | mode    | skip%  | traj%  | V      | s2_lat | verdict"
echo "-------|---------|--------|--------|--------|--------|--------"
python3 - <<PYEOF
import json, math, os

def cramers(a,b,cc,d):
    nn=a+b+cc+d
    if nn==0: return 0.0
    r1,r2=a+b,cc+d; c1,c2=a+cc,b+d
    if min(r1,r2,c1,c2)==0: return 0.0
    ea,eb,ec,ed=r1*c1/nn,r1*c2/nn,r2*c1/nn,r2*c2/nn
    chi2=sum((max(0,abs(o-e)-0.5)**2)/e for o,e in((a,ea),(b,eb),(cc,ec),(d,ed)))
    return math.sqrt(chi2/nn)

base="/tmp/gate10_realspeed"
passes=0
for bag in ["073623","061841","063047"]:
    nf=f"{base}/{bag}_nocache/metrics.json"
    pf=f"{base}/{bag}_prod/metrics.json"
    if not os.path.exists(nf) or not os.path.exists(pf):
        print(f"  {bag}: missing data"); continue
    with open(nf) as f: nc=json.load(f)
    with open(pf) as f: pr=json.load(f)
    cft=nc.get("fresh_traj_outputs",0); cfa=nc.get("fresh_action_outputs",0)
    ft=pr.get("fresh_traj_outputs",0);  fa=pr.get("fresh_action_outputs",0)
    V=cramers(cft,cfa,ft,fa)
    skip=pr.get("temporal_cache_skip_ratio",0)
    s2l=pr.get("s2_latency_ms",0)
    nctr=100*cft/max(1,cft+cfa); tr=100*ft/max(1,ft+fa)
    ok = V<=0.10 and skip>=70.0
    if ok: passes+=1
    print(f"  {bag} | nocache | {nc.get('temporal_cache_skip_ratio',0):5.1f}% | {nctr:5.1f}% | —      | {nc.get('s2_latency_ms',0):6.0f}ms | BASELINE")
    print(f"  {bag} | prod    | {skip:5.1f}% | {tr:5.1f}% | {V:.4f} | {s2l:6.0f}ms | {'PASS' if ok else 'FAIL'}")

verdict="PASS" if passes==3 else ("PARTIAL" if passes>0 else "FAIL")
print(f"\n  Gate 10 FINAL: {verdict} ({passes}/3 bags)  criterion: V<=0.10 AND skip>=70%")
print(f"  Key question: does async cache still provide quality-neutral speedup at real speed?")
PYEOF

echo ""
echo "=== Gate 10 DONE ==="
