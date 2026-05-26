#!/usr/bin/env bash
# Gate 6a — max_new_tokens sweep (I-200)
# Hypothesis: reducing max_new_tokens 80→32 cuts S2 latency ≥30% with V≤0.10.
#
# Phase 1: Single-bag sweep (073623) for tokens ∈ {80, 64, 48, 32, 16}
# Phase 2: 3-bag cross-validation for winner
#
# Pass criterion (winner): S2 latency < 200ms AND V≤0.10 AND traj_ratio within 8pp of 80-token baseline
# Run on port 5804, cuda:2 (so Gate 3X analysis can run concurrently on 5802)
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

PORT=5804
DEVICE=cuda:1
CALIB="scripts/realworld/calib/calib_scout.txt"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
BAG_DIR="/workspace/rosbag"
LOG_BASE="/tmp/gate6a_maxtokens"
mkdir -p "$LOG_BASE"

SWEEP_BAG="073623"
XVAL_BAGS=(073623 061841 063047)
TOKEN_VALUES=(80 64 48 32 16)

echo "=================================================="
echo " Gate 6a — max_new_tokens sweep (I-200)"
echo " tokens ∈ {80,64,48,32,16} on bag $SWEEP_BAG"
echo " Port: $PORT  Device: $DEVICE"
echo "=================================================="

start_server() {
    local tokens=$1 tag=$2
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

    # Apply Gate 3n production stack (so we measure latency WITH full cache active)
    curl -sf "http://localhost:${PORT}/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:${PORT}/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:${PORT}/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:${PORT}/set_ema_fingerprint?enabled=true&alpha=0.10&transition_reset=true" > /dev/null
    curl -sf "http://localhost:${PORT}/set_slope_predict?enabled=true&threshold=0.010&window=3" > /dev/null
    # Set the token count
    curl -sf "http://localhost:${PORT}/set_max_new_tokens?tokens=${tokens}" > /dev/null
    echo "  Config: Gate 3n stack + max_new_tokens=$tokens"
}

run_one() {
    local bag=$1 tokens=$2
    local tag="${bag}_t${tokens}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$bag  tokens=$tokens ---"

    # Baseline (tokens=80) reference for this bag
    local ctrl_dir="$LOG_BASE/${bag}_t80"
    if [ "$tokens" == "80" ]; then
        ctrl_dir="$log_dir"
    fi

    # Start fresh server only if port not already up with correct config
    start_server "$tokens" "$tag"

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

bg  = t.get("background_s2_runs", 0)
ft  = t.get("fresh_traj_outputs", 0)
fa  = t.get("fresh_action_outputs", 0)
n   = ft + fa
tr  = 100 * ft / n if n > 0 else 0
s2l = t.get("s2_latency_ms", 0)
skip= t.get("temporal_cache_skip_ratio", 0)
hz  = t.get("joint_req_hz", 0)
tok = t.get("max_new_tokens", $tokens)

# Cramer's V vs 80-token baseline (if available)
ctrl_f = "$LOG_BASE/${bag}_t80/metrics.json"
V = 0.0
if os.path.exists(ctrl_f) and "$tokens" != "80":
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
    ctr = 100 * cft / max(1, cft + cfa)
    tr_delta = tr - ctr
    status = 'PASS' if V <= 0.10 and s2l < 200 and abs(tr_delta) <= 8 else 'FAIL'
else:
    tr_delta = 0.0
    status = 'BASELINE'

print(f"  tok={tok:3d}  S2_lat={s2l:6.1f}ms  skip={skip:.1f}%  traj={tr:.1f}%  Δtraj={tr_delta:+.1f}pp  V={V:.4f}  hz={hz:.2f}  => {status}")
PYEOF

    pkill -f "http_internvla_server_debug.*${PORT}" 2>/dev/null || true
    sleep 5
}

# Phase 1: Single-bag sweep
echo ""
echo "=== Phase 1: Single-bag sweep (bag $SWEEP_BAG) ==="
for tok in "${TOKEN_VALUES[@]}"; do
    run_one "$SWEEP_BAG" "$tok"
done

# Phase 2: Summary and winner selection
echo ""
echo "=== Phase 2: Sweep Summary — S2 Latency vs Token Count ==="
echo "tokens | S2_lat_ms | skip% | traj% | Δtraj | V     | verdict"
echo "-------|-----------|-------|-------|-------|-------|--------"
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
ctrl_f=f"/tmp/gate6a_maxtokens/{bag}_t80/metrics.json"
if not os.path.exists(ctrl_f):
    print("baseline t80 not found")
    exit()
with open(ctrl_f) as f: c=json.load(f)
cft=c.get("fresh_traj_outputs",0); cfa=c.get("fresh_action_outputs",0)
ctr=100*cft/max(1,cft+cfa)

results=[]
for tok in [80,64,48,32,16]:
    mf=f"/tmp/gate6a_maxtokens/{bag}_t{tok}/metrics.json"
    if not os.path.exists(mf): continue
    with open(mf) as f: t=json.load(f)
    ft=t.get("fresh_traj_outputs",0); fa=t.get("fresh_action_outputs",0)
    tr=100*ft/max(1,ft+fa); s2l=t.get("s2_latency_ms",0)
    skip=t.get("temporal_cache_skip_ratio",0); hz=t.get("joint_req_hz",0)
    V=cramers(cft,cfa,ft,fa)
    tr_delta=tr-ctr
    if tok==80:
        status="BASELINE"
    else:
        status="PASS" if V<=0.10 and s2l<200 and abs(tr_delta)<=8 else "FAIL"
    results.append((tok,s2l,skip,tr,tr_delta,V,hz,status))
    print(f"  {tok:6d} | {s2l:9.1f} | {skip:5.1f}% | {tr:5.1f}% | {tr_delta:+5.1f}pp | {V:.4f} | {status}")

# Winner: highest latency reduction (lowest S2 lat) among PASS
winners=[r for r in results if r[7]=='PASS']
if winners:
    winner=min(winners, key=lambda r: r[1])
    print(f"\nWinner: max_new_tokens={winner[0]}  S2_lat={winner[1]:.1f}ms  V={winner[5]:.4f}")
    with open("/tmp/gate6a_maxtokens/winner.txt","w") as f: f.write(str(winner[0]))
else:
    print("\nNo PASS — all reduced-token configs produce V>0.10 or lat>200ms or traj shift>8pp")
    with open("/tmp/gate6a_maxtokens/winner.txt","w") as f: f.write("80")
PYEOF

# Phase 3: 3-bag xval with winner
WINNER=$(cat /tmp/gate6a_maxtokens/winner.txt 2>/dev/null || echo "80")
echo ""
echo "=== Phase 3: 3-bag xval with max_new_tokens=$WINNER ==="
if [ "$WINNER" == "80" ]; then
    echo "No improvement over baseline. Gate 6a FAIL."
else
    for bag in "${XVAL_BAGS[@]}"; do
        run_one "$bag" "$WINNER"
    done
    echo ""
    echo "=== Gate 6a Final: 3-bag results (tokens=$WINNER vs baseline=80) ==="
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
    cf=f"/tmp/gate6a_maxtokens/{bag}_t80/metrics.json"
    tf=f"/tmp/gate6a_maxtokens/{bag}_t{WINNER}/metrics.json"
    if not os.path.exists(cf) or not os.path.exists(tf):
        print(f"  bag={bag}: missing data")
        continue
    with open(cf) as f: c=json.load(f)
    with open(tf) as f: t=json.load(f)
    cft=c.get("fresh_traj_outputs",0); cfa=c.get("fresh_action_outputs",0)
    ft=t.get("fresh_traj_outputs",0); fa=t.get("fresh_action_outputs",0)
    V=cramers(cft,cfa,ft,fa)
    s2l=t.get("s2_latency_ms",0); skip=t.get("temporal_cache_skip_ratio",0)
    ctr=100*cft/max(1,cft+cfa); tr=100*ft/max(1,ft+fa)
    ok = V<=0.10 and s2l<200 and abs(tr-ctr)<=8
    if ok: passes+=1
    print(f"  bag={bag}: S2_lat={s2l:.1f}ms  skip={skip:.1f}%  traj_delta={tr-ctr:+.1f}pp  V={V:.4f}  {'PASS' if ok else 'FAIL'}")

verdict = "PASS" if passes==3 else ("PARTIAL" if passes>0 else "FAIL")
print(f"\n  Gate 6a FINAL: tokens={WINNER}  {verdict} ({passes}/3 bags)")
PYEOF2
fi

echo ""
echo "=== Gate 6a DONE ==="
