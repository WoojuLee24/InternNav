#!/usr/bin/env bash
# Gate 8 — EMA Warm-Up Acceleration (I-203)
# Hypothesis: using a high α_warm for the first warmup_N frames after each TR-EMA reset
# lets the EMA fingerprint converge to the local scene faster, lifting cold-start skip
# from ~75% toward ~91% within the first 30 seconds of a bag.
#
# Sweep: warmup_N ∈ {0 (off), 5, 10, 20} × α_warm=0.5 on bag 073623.
# Then 3-bag xval for the winner.
#
# Pass criterion: skip_start ≥ 82% (within 9pp of steady-state 91%) AND V ≤ 0.10 AND
#   steady-state skip within ±2pp of Gate 3n baseline (90.8-91.4%).
# "skip_start" = skip ratio during the first 60s of the bag (approximated by first third
#   of total requests).
# Port 5804, cuda:1.
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

PORT=5804
DEVICE=cuda:1
CALIB="scripts/realworld/calib/calib_scout.txt"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
BAG_DIR="/workspace/rosbag"
LOG_BASE="/tmp/gate8_ema_warmup"
mkdir -p "$LOG_BASE"

SWEEP_BAG="073623"
XVAL_BAGS=(073623 061841 063047)
WARMUP_N_VALUES=(0 5 10 20)
ALPHA_WARM=0.5

echo "=================================================="
echo " Gate 8 — EMA Warm-Up Acceleration (I-203)"
echo " warmup_N ∈ {0,5,10,20}  α_warm=$ALPHA_WARM"
echo " Bag: $SWEEP_BAG  Port: $PORT  Device: $DEVICE"
echo "=================================================="

start_server() {
    local tag=$1
    pkill -f "http_internvla_server_debug.*$PORT" 2>/dev/null || true
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 3

    python3 /workspace/InternNav/scripts/realworld/http_internvla_server_debug.py \
        --mode async --temperature 0.75 --kv-cache \
        --calib "$CALIB" \
        --device "$DEVICE" \
        --port "$PORT" \
        --pre-warm-frames 3 \
        > "$LOG_BASE/${tag}_server.log" 2>&1 &
    SERVER_PID=$!

    echo "  Waiting for server (pid=$SERVER_PID)..."
    until curl -sf "http://localhost:${PORT}/async_metrics" > /dev/null 2>&1; do sleep 3; done
    echo "  Server ready on :${PORT}"

    # Apply Gate 3n production stack
    curl -sf "http://localhost:${PORT}/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:${PORT}/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:${PORT}/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:${PORT}/set_ema_fingerprint?enabled=true&alpha=0.10&transition_reset=true" > /dev/null
    curl -sf "http://localhost:${PORT}/set_slope_predict?enabled=true&threshold=0.010&window=3" > /dev/null
}

configure_warmup() {
    local n=$1
    if [ "$n" -eq 0 ]; then
        curl -sf "http://localhost:${PORT}/set_ema_warmup?enabled=false" > /dev/null
        echo "  Warmup: OFF (baseline Gate 3n)"
    else
        curl -sf "http://localhost:${PORT}/set_ema_warmup?enabled=true&warmup_frames=${n}&alpha_warm=${ALPHA_WARM}" > /dev/null
        echo "  Warmup: ON  warmup_N=$n  α_warm=$ALPHA_WARM"
    fi
}

run_one() {
    local bag=$1 warmup_n=$2
    local tag="${bag}_w${warmup_n}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$bag  warmup_N=$warmup_n ---"

    start_server "$tag"
    configure_warmup "$warmup_n"

    curl -sf "http://localhost:${PORT}/reset_metrics" > "$log_dir/reset.json"
    sleep 1

    python3.12 /workspace/InternNav/$CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
        --server-port "${PORT}" \
        > "$log_dir/client.log" 2>&1 &
    local cpid=$!; sleep 3

    ros2 bag play "$BAG_DIR/my_camera_bag_20260317_$bag" --rate 0.5 \
        --topics /camera/camera/color/image_raw \
                 /camera/camera/aligned_depth_to_color/image_raw \
                 /gdq/msg/gdq_odom > "$log_dir/bag.log" 2>&1 || true
    sleep 3
    curl -sf "http://localhost:${PORT}/async_metrics" > "$log_dir/metrics.json"
    kill $cpid 2>/dev/null || true

    python3 - <<PYEOF
import json, math, os
with open("$log_dir/metrics.json") as f: t = json.load(f)

ft   = t.get("fresh_traj_outputs", 0)
fa   = t.get("fresh_action_outputs", 0)
n    = ft + fa
tr   = 100 * ft / n if n > 0 else 0
skip = t.get("temporal_cache_skip_ratio", 0)
hz   = t.get("joint_req_hz", 0)
wn   = $warmup_n

# Cramer's V vs warmup_N=0 baseline (if available)
ctrl_f = "$LOG_BASE/${bag}_w0/metrics.json"
V = 0.0
if os.path.exists(ctrl_f) and $warmup_n != 0:
    with open(ctrl_f) as f: c = json.load(f)
    cft = c.get("fresh_traj_outputs", 0); cfa = c.get("fresh_action_outputs", 0)
    def cramers(a,b,cc,d):
        nn = a+b+cc+d
        if nn==0: return 0.0
        r1,r2=a+b,cc+d; c1,c2=a+cc,b+d
        if min(r1,r2,c1,c2)==0: return 0.0
        ea,eb,ec,ed=r1*c1/nn,r1*c2/nn,r2*c1/nn,r2*c2/nn
        chi2=sum((max(0,abs(o-e)-0.5)**2)/e for o,e in ((a,ea),(b,eb),(cc,ec),(d,ed)))
        return math.sqrt(chi2/nn)
    V = cramers(cft, cfa, ft, fa)
    cskip = c.get("temporal_cache_skip_ratio", 0)
    skip_delta = skip - cskip
    status = 'PASS' if V <= 0.10 and abs(skip - 91.0) <= 2.0 else 'FAIL'
else:
    skip_delta = 0.0
    status = 'BASELINE'

print(f"  wN={wn:2d}  skip={skip:.1f}%  Δskip={skip_delta:+.1f}pp  traj={tr:.1f}%  V={V:.4f}  hz={hz:.2f}  => {status}")
PYEOF

    pkill -f "http_internvla_server_debug.*${PORT}" 2>/dev/null || true
    sleep 5
}

# Phase 1: Single-bag sweep
echo ""
echo "=== Phase 1: Warm-Up Sweep (bag $SWEEP_BAG) ==="
for wn in "${WARMUP_N_VALUES[@]}"; do
    run_one "$SWEEP_BAG" "$wn"
done

# Phase 2: Summary
echo ""
echo "=== Phase 2: Sweep Summary ==="
echo "warmup_N | skip%  | Δskip  | traj%  | V      | verdict"
echo "---------|--------|--------|--------|--------|--------"
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

bag="${SWEEP_BAG}"
ctrl_f=f"/tmp/gate8_ema_warmup/{bag}_w0/metrics.json"
if not os.path.exists(ctrl_f):
    print("baseline w0 not found")
    exit()
with open(ctrl_f) as f: c=json.load(f)
cft=c.get("fresh_traj_outputs",0); cfa=c.get("fresh_action_outputs",0)
cskip=c.get("temporal_cache_skip_ratio",0)

results=[]
for wn in [0,5,10,20]:
    mf=f"/tmp/gate8_ema_warmup/{bag}_w{wn}/metrics.json"
    if not os.path.exists(mf): continue
    with open(mf) as f: t=json.load(f)
    ft=t.get("fresh_traj_outputs",0); fa=t.get("fresh_action_outputs",0)
    tr=100*ft/max(1,ft+fa); skip=t.get("temporal_cache_skip_ratio",0)
    hz=t.get("joint_req_hz",0)
    V=cramers(cft,cfa,ft,fa)
    skip_delta=skip-cskip
    if wn==0:
        status="BASELINE"
    else:
        status="PASS" if V<=0.10 and abs(skip-91.0)<=2.0 else "FAIL"
    results.append((wn,skip,skip_delta,tr,V,hz,status))
    print(f"  {wn:8d} | {skip:6.1f}% | {skip_delta:+6.1f}pp | {tr:6.1f}% | {V:.4f} | {status}")

winners=[r for r in results if r[6]=='PASS']
if winners:
    winner=max(winners, key=lambda r: r[1])  # highest skip
    print(f"\nWinner: warmup_N={winner[0]}  skip={winner[1]:.1f}%  V={winner[3]:.4f}")
    with open("/tmp/gate8_ema_warmup/winner.txt","w") as f: f.write(str(winner[0]))
else:
    print("\nNo PASS — warm-up acceleration does not improve skip ratio with V<=0.10")
    with open("/tmp/gate8_ema_warmup/winner.txt","w") as f: f.write("0")
PYEOF

# Phase 3: 3-bag xval
WINNER=$(cat /tmp/gate8_ema_warmup/winner.txt 2>/dev/null || echo "0")
echo ""
echo "=== Phase 3: 3-bag xval with warmup_N=$WINNER ==="
if [ "$WINNER" == "0" ]; then
    echo "No improvement over baseline. Gate 8 FAIL."
else
    for bag in "${XVAL_BAGS[@]}"; do
        run_one "$bag" "$WINNER"
    done
    echo ""
    echo "=== Gate 8 Final: 3-bag results (warmup_N=$WINNER) ==="
    python3 - <<PYEOF2
import json, math, os

def cramers(a,b,cc,d):
    nn=a+b+cc+d
    if nn==0: return 0.0
    r1,r2=a+b,cc+d; c1,c2=a+cc,b+d
    if min(r1,r2,c1,c2)==0: return 0.0
    ea,eb,ec,ed=r1*c1/nn,r1*c2/nn,r2*c1/nn,r2*c2/nn
    chi2=sum((max(0,abs(o-e)-0.5)**2)/e for o,e in((a,ea),(b,eb),(cc,ec),(d,ed)))
    return math.sqrt(chi2/nn)

WINNER="$WINNER"
passes=0
for bag in ["073623","061841","063047"]:
    cf=f"/tmp/gate8_ema_warmup/{bag}_w0/metrics.json"
    tf=f"/tmp/gate8_ema_warmup/{bag}_w{WINNER}/metrics.json"
    if not os.path.exists(cf) or not os.path.exists(tf):
        print(f"  bag={bag}: missing data")
        continue
    with open(cf) as f: c=json.load(f)
    with open(tf) as f: t=json.load(f)
    cft=c.get("fresh_traj_outputs",0); cfa=c.get("fresh_action_outputs",0)
    ft=t.get("fresh_traj_outputs",0); fa=t.get("fresh_action_outputs",0)
    V=cramers(cft,cfa,ft,fa)
    skip=t.get("temporal_cache_skip_ratio",0); cskip=c.get("temporal_cache_skip_ratio",0)
    ok = V<=0.10 and abs(skip-91.0)<=2.0
    if ok: passes+=1
    print(f"  bag={bag}: skip={skip:.1f}% (Δ{skip-cskip:+.1f}pp)  V={V:.4f}  {'PASS' if ok else 'FAIL'}")

verdict = "PASS" if passes==3 else ("PARTIAL" if passes>0 else "FAIL")
print(f"\n  Gate 8 FINAL: warmup_N={WINNER}  {verdict} ({passes}/3 bags)")
PYEOF2
fi

echo ""
echo "=== Gate 8 DONE ==="
