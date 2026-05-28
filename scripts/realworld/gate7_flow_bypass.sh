#!/usr/bin/env bash
# Gate 7 — Optical Flow Cache Invalidation (I-202)
# Hypothesis: optical flow magnitude > threshold forces S2 refresh at motion onset
# (turns, sudden stops) that visual embeddings miss, while maintaining V≤0.10
# and skip ratio within 2pp of Gate 3n baseline at steady state.
#
# Phase 1: threshold sweep (off=baseline, 5, 10, 20, 40 px) on bag 073623
# Phase 2: 3-bag xval for best threshold
#
# Pass criterion: V≤0.10 AND flow_bypasses>0 (system fires) AND |skip_delta|≤3pp
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

PORT=5804
DEVICE=cuda:1
CALIB="scripts/realworld/calib/calib_scout.txt"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
# BAG_DIR: rosbag played manually from gds container via FastDDS
LOG_BASE="/tmp/gate7_flow"
mkdir -p "$LOG_BASE"

SWEEP_BAG="073623"
XVAL_BAGS=(073623 061841 063047)
# off=Gate 3n baseline; numeric = flow magnitude threshold at 80x60 resolution
THRESHOLDS=(off 40 20 10 5)

echo "=================================================="
echo " Gate 7 — Optical Flow Cache Invalidation (I-202)"
echo " flow_threshold ∈ {off,40,20,10,5} on bag $SWEEP_BAG"
echo " Port: $PORT  Device: $DEVICE"
echo "=================================================="

start_server() {
    local tag=$1
    pkill -f "http_internvla_server_debug.*$PORT" 2>/dev/null || true
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 3
    cd "$REPO_DIR" && python3 scripts/realworld/http_internvla_server_debug.py \
        --mode async --temperature 0.75 --kv-cache \
        --calib "$CALIB" --device "$DEVICE" --port "$PORT" \
        --pre-warm-frames 3 \
        > "$LOG_BASE/${tag}_server.log" 2>&1 &
    SERVER_PID=$!
    echo "  Waiting for server (pid=$SERVER_PID)..."
    until curl -sf "http://localhost:${PORT}/async_metrics" > /dev/null 2>&1; do sleep 3; done
    echo "  Server ready on :${PORT}"
}

apply_gate3n() {
    curl -sf "http://localhost:${PORT}/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:${PORT}/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:${PORT}/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:${PORT}/set_ema_fingerprint?enabled=true&alpha=0.10&transition_reset=true" > /dev/null
    curl -sf "http://localhost:${PORT}/set_slope_predict?enabled=true&threshold=0.010&window=3" > /dev/null
}

run_one() {
    local bag=$1 thr=$2
    local tag="${bag}_flow${thr}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"
    echo ""
    echo "--- bag=$bag  flow_threshold=$thr ---"

    start_server "$tag"
    apply_gate3n

    if [ "$thr" == "off" ]; then
        curl -sf "http://localhost:${PORT}/set_flow_bypass?enabled=false" > /dev/null
        echo "  Config: Gate 3n stack + flow=OFF (baseline)"
    else
        curl -sf "http://localhost:${PORT}/set_flow_bypass?enabled=true&threshold=${thr}" > /dev/null
        echo "  Config: Gate 3n stack + flow=ON threshold=${thr}"
    fi

    curl -sf "http://localhost:${PORT}/reset_metrics" > "$log_dir/reset.json"
    sleep 1

    cd "$REPO_DIR" && python3.12 $CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
        --server-port "${PORT}" \
        > "$log_dir/client.log" 2>&1 &
    local cpid=$!; sleep 3
# 
#     ros2 bag play "$BAG_DIR/my_camera_bag_20260317_$bag" --rate 0.5 \  # played manually from gds container
#         --topics /camera/camera/color/image_raw \
#                  /camera/camera/aligned_depth_to_color/image_raw \
#                  /gdq/msg/gdq_odom > "$log_dir/bag.log" 2>&1 || true
    sleep 3
    curl -sf "http://localhost:${PORT}/async_metrics" > "$log_dir/metrics.json"
    kill $cpid 2>/dev/null || true

    python3 - <<PYEOF
import json, math, os
with open("$log_dir/metrics.json") as f: t = json.load(f)

bg   = t.get("background_s2_runs", 0)
ft   = t.get("fresh_traj_outputs", 0)
fa   = t.get("fresh_action_outputs", 0)
n    = ft + fa
tr   = 100 * ft / n if n > 0 else 0
skip = t.get("temporal_cache_skip_ratio", 0)
hz   = t.get("joint_req_hz", 0)
flow = t.get("flow_bypasses", 0)

ctrl_f = "$LOG_BASE/${bag}_flowoff/metrics.json"
V = 0.0; status = "BASELINE"

if os.path.exists(ctrl_f) and "$thr" != "off":
    with open(ctrl_f) as f: c = json.load(f)
    cft = c.get("fresh_traj_outputs", 0); cfa = c.get("fresh_action_outputs", 0)
    cskip = c.get("temporal_cache_skip_ratio", 0)
    def cramers(a,b,cc,d):
        nn = a+b+cc+d
        if nn==0: return 0.0
        r1,r2=a+b,cc+d; c1,c2=a+cc,b+d
        if min(r1,r2,c1,c2)==0: return 0.0
        ea,eb,ec,ed=r1*c1/nn,r1*c2/nn,r2*c1/nn,r2*c2/nn
        chi2=sum((max(0,abs(o-e)-0.5)**2)/e for o,e in ((a,ea),(b,eb),(cc,ec),(d,ed)))
        return math.sqrt(chi2/nn)
    V = cramers(cft, cfa, ft, fa)
    ctr = 100 * cft / max(1, cft + cfa)
    tr_delta = tr - ctr
    skip_delta = skip - cskip
    ok = V <= 0.10 and flow > 0 and abs(skip_delta) <= 3.0
    status = "PASS" if ok else "FAIL"
    print(f"  flow={float('$thr') if '$thr' != 'off' else 0:.0f}px  skip={skip:.1f}%  Δskip={skip_delta:+.1f}pp  traj={tr:.1f}%  Δtraj={tr_delta:+.1f}pp  V={V:.4f}  flow_triggers={flow}  hz={hz:.2f}  => {status}")
else:
    print(f"  flow=OFF  skip={skip:.1f}%  traj={tr:.1f}%  flow_triggers={flow}  hz={hz:.2f}  => {status}")
PYEOF

    pkill -f "http_internvla_server_debug.*${PORT}" 2>/dev/null || true
    sleep 5
}

# Phase 1: Sweep
echo ""
echo "=== Phase 1: Sweep on bag $SWEEP_BAG ==="
for thr in "${THRESHOLDS[@]}"; do
    run_one "$SWEEP_BAG" "$thr"
done

# Phase 2: Summary
echo ""
echo "=== Phase 2: Sweep Summary ==="
echo "  thr  | skip%  | Δskip | traj% | Δtraj | V      | flow_n | verdict"
echo "-------|--------|-------|-------|-------|--------|--------|--------"
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

bag="$SWEEP_BAG"
cf=f"/tmp/gate7_flow/{bag}_flowoff/metrics.json"
if not os.path.exists(cf): print("baseline not found"); exit()
with open(cf) as f: c=json.load(f)
cft=c.get("fresh_traj_outputs",0); cfa=c.get("fresh_action_outputs",0)
ctr=100*cft/max(1,cft+cfa); cskip=c.get("temporal_cache_skip_ratio",0)

results=[]
for thr_str in ["off","40","20","10","5"]:
    mf=f"/tmp/gate7_flow/{bag}_flow{thr_str}/metrics.json"
    if not os.path.exists(mf): continue
    with open(mf) as f: t=json.load(f)
    ft=t.get("fresh_traj_outputs",0); fa=t.get("fresh_action_outputs",0)
    tr=100*ft/max(1,ft+fa); skip=t.get("temporal_cache_skip_ratio",0)
    flow_n=t.get("flow_bypasses",0); hz=t.get("joint_req_hz",0)
    V=cramers(cft,cfa,ft,fa)
    tr_delta=tr-ctr; skip_delta=skip-cskip
    if thr_str=="off":
        status="BASELINE"
    else:
        ok = V<=0.10 and flow_n>0 and abs(skip_delta)<=3.0
        status="PASS" if ok else "FAIL"
    results.append((thr_str,skip,skip_delta,tr,tr_delta,V,flow_n,hz,status))
    print(f"  {thr_str:4s} | {skip:6.1f}% | {skip_delta:+5.1f}pp | {tr:5.1f}% | {tr_delta:+5.1f}pp | {V:.4f} | {flow_n:6d} | {status}")

winners=[r for r in results if r[8]=='PASS']
if winners:
    w=min(winners, key=lambda r: abs(r[2]))  # smallest skip change
    print(f"\nWinner: flow_threshold={w[0]}  skip_delta={w[2]:+.1f}pp  V={w[5]:.4f}  flow_n={w[6]}")
    with open("/tmp/gate7_flow/winner.txt","w") as f: f.write(w[0])
else:
    print("\nNo PASS — all flow configs fail criteria")
    with open("/tmp/gate7_flow/winner.txt","w") as f: f.write("off")
PYEOF

WINNER=$(cat /tmp/gate7_flow/winner.txt 2>/dev/null || echo "off")
echo ""
echo "=== Phase 3: 3-bag xval with flow_threshold=$WINNER ==="
if [ "$WINNER" == "off" ]; then
    echo "No improvement over baseline. Gate 7 FAIL."
else
    for bag in "${XVAL_BAGS[@]}"; do
        run_one "$bag" "$WINNER"
    done
    echo "=== Gate 7 DONE ==="
fi
