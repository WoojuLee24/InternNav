#!/usr/bin/env bash
# Gate 6b — temporal threshold sweep (I-201)
# Hypothesis: raising τ from 0.92→0.95/0.97/0.99 increases skip ratio
# while keeping V≤0.10 and fresh_traj_ratio within 3pp of τ=0.92 baseline.
#
# Gate 3n production stack is held constant except for τ.
# Phase 1: Single-bag sweep (073623) for τ ∈ {0.92, 0.95, 0.97, 0.99}
# Phase 2: 3-bag xval for winner
#
# Pass criterion: skip_ratio_winner > skip_ratio_baseline + 2pp
#                 AND V≤0.10 AND |fresh_traj - baseline_traj| ≤ 3pp
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

PORT=5804
DEVICE=cuda:1
CALIB="scripts/realworld/calib/calib_scout.txt"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
# BAG_DIR: rosbag played manually from gds container via FastDDS
LOG_BASE="/tmp/gate6b_threshold"
mkdir -p "$LOG_BASE"

SWEEP_BAG="073623"
XVAL_BAGS=(073623 061841 063047)
TAU_VALUES=(0.92 0.95 0.97 0.99)

echo "=================================================="
echo " Gate 6b — temporal threshold sweep (I-201)"
echo " τ ∈ {0.92,0.95,0.97,0.99} on bag $SWEEP_BAG"
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

configure_stack() {
    local tau=$1
    # Gate 3n stack with variable τ
    curl -sf "http://localhost:${PORT}/set_temporal_threshold?threshold=${tau}" > /dev/null
    curl -sf "http://localhost:${PORT}/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:${PORT}/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:${PORT}/set_ema_fingerprint?enabled=true&alpha=0.10&transition_reset=true" > /dev/null
    curl -sf "http://localhost:${PORT}/set_slope_predict?enabled=true&threshold=0.010&window=3" > /dev/null
    echo "  Config: Gate 3n stack + τ=${tau}"
}

run_one() {
    local bag=$1 tau=$2
    local tag="${bag}_tau${tau//./_}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"
    echo ""
    echo "--- bag=$bag  τ=$tau ---"

    start_server "$tag"
    configure_stack "$tau"

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

bg  = t.get("background_s2_runs", 0)
ft  = t.get("fresh_traj_outputs", 0)
fa  = t.get("fresh_action_outputs", 0)
n   = ft + fa
tr  = 100 * ft / n if n > 0 else 0
s2l = t.get("s2_latency_ms", 0)
skip= t.get("temporal_cache_skip_ratio", 0)
hz  = t.get("joint_req_hz", 0)

# Compare to τ=0.92 baseline
ctrl_f = "$LOG_BASE/${bag}_tau0_92/metrics.json"
V = 0.0
status = "BASELINE" if "$tau" == "0.92" else "N/A"

if os.path.exists(ctrl_f) and "$tau" != "0.92":
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
    skip_gain = skip - cskip
    ok = V <= 0.10 and abs(tr_delta) <= 3.0 and skip_gain > 2.0
    status = "PASS" if ok else "FAIL"
    print(f"  τ={float('$tau'):.2f}  skip={skip:.1f}%  Δskip={skip_gain:+.1f}pp  traj={tr:.1f}%  Δtraj={tr_delta:+.1f}pp  V={V:.4f}  hz={hz:.2f}  => {status}")
else:
    print(f"  τ={float('$tau'):.2f}  skip={skip:.1f}%  traj={tr:.1f}%  hz={hz:.2f}  => {status}")
PYEOF

    pkill -f "http_internvla_server_debug.*${PORT}" 2>/dev/null || true
    sleep 5
}

# Phase 1: sweep
echo ""
echo "=== Phase 1: Sweep on bag $SWEEP_BAG ==="
for tau in "${TAU_VALUES[@]}"; do
    run_one "$SWEEP_BAG" "$tau"
done

# Phase 2: Summary
echo ""
echo "=== Phase 2: Sweep Summary ==="
echo "  τ     | skip%  | Δskip | traj% | Δtraj | V      | verdict"
echo "--------|--------|-------|-------|-------|--------|--------"
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
cf=f"/tmp/gate6b_threshold/{bag}_tau0_92/metrics.json"
if not os.path.exists(cf):
    print("baseline τ=0.92 not found"); exit()
with open(cf) as f: c=json.load(f)
cft=c.get("fresh_traj_outputs",0); cfa=c.get("fresh_action_outputs",0)
ctr=100*cft/max(1,cft+cfa); cskip=c.get("temporal_cache_skip_ratio",0)

results=[]
for tau_str,tau_label in [("0_92","0.92"),("0_95","0.95"),("0_97","0.97"),("0_99","0.99")]:
    mf=f"/tmp/gate6b_threshold/{bag}_tau{tau_str}/metrics.json"
    if not os.path.exists(mf): continue
    with open(mf) as f: t=json.load(f)
    ft=t.get("fresh_traj_outputs",0); fa=t.get("fresh_action_outputs",0)
    tr=100*ft/max(1,ft+fa); skip=t.get("temporal_cache_skip_ratio",0); hz=t.get("joint_req_hz",0)
    V=cramers(cft,cfa,ft,fa)
    tr_delta=tr-ctr; skip_gain=skip-cskip
    if tau_label=="0.92":
        status="BASELINE"
    else:
        ok = V<=0.10 and abs(tr_delta)<=3.0 and skip_gain>2.0
        status="PASS" if ok else "FAIL"
    results.append((tau_label,skip,skip_gain,tr,tr_delta,V,hz,status))
    print(f"  {tau_label} | {skip:6.1f}% | {skip_gain:+5.1f}pp | {tr:5.1f}% | {tr_delta:+5.1f}pp | {V:.4f} | {status}")

winners=[r for r in results if r[7]=='PASS']
if winners:
    w=max(winners, key=lambda r: r[1])  # highest skip
    print(f"\nWinner: τ={w[0]}  skip={w[1]:.1f}%  V={w[5]:.4f}")
    with open("/tmp/gate6b_threshold/winner.txt","w") as f: f.write(w[0])
else:
    print("\nNo PASS — all higher-τ configs break quality criteria")
    with open("/tmp/gate6b_threshold/winner.txt","w") as f: f.write("0.92")
PYEOF

WINNER=$(cat /tmp/gate6b_threshold/winner.txt 2>/dev/null || echo "0.92")
echo ""
echo "=== Phase 3: 3-bag xval with τ=$WINNER ==="
if [ "$WINNER" == "0.92" ]; then
    echo "No improvement. Gate 6b FAIL."
else
    for bag in "${XVAL_BAGS[@]}"; do
        run_one "$bag" "$WINNER"
    done
    echo "=== Gate 6b DONE ==="
fi
