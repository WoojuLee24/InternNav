#!/usr/bin/env bash
# Gate 9 — Cosine Similarity Variance Gating (I-204)
# Hypothesis: std(sim_history[-W:]) > σ catches oscillatory scenes (doorways,
# left/right turns) where sim >= τ but the scene is unstable. Forcing S2 at
# high-variance moments improves cache validity without degrading overall skip rate.
#
# Sweep: σ ∈ {off, 0.05, 0.03, 0.01}, W=5 on bag 073623.
# Pass criterion: V ≤ 0.10 AND var_gate_bypasses > 0 AND |Δskip| ≤ 3pp.
# Port 5804, cuda:1.
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

# REPO_DIR: override via env or auto-detect from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PORT=5804
DEVICE=cuda:1
CALIB="$REPO_DIR/scripts/realworld/calib/calib_scout.txt"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
# BAG_DIR: rosbag played manually from gds container via FastDDS
LOG_BASE="/tmp/gate9_var_gate"
mkdir -p "$LOG_BASE"

SWEEP_BAG="073623"
XVAL_BAGS=(073623 061841 063047)
SIGMA_VALUES=("off" "0.05" "0.03" "0.01")
VAR_WINDOW=5

echo "=================================================="
echo " Gate 9 — Cosine Similarity Variance Gating (I-204)"
echo " σ ∈ {off,0.05,0.03,0.01}  W=$VAR_WINDOW"
echo " Bag: $SWEEP_BAG  Port: $PORT  Device: $DEVICE"
echo "=================================================="

start_server() {
    local tag=$1
    pkill -f "http_internvla_server_debug.*$PORT" 2>/dev/null || true
    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep 3

    cd "$REPO_DIR"
    python3 scripts/realworld/http_internvla_server_debug.py \
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

configure_var_gate() {
    local sigma=$1
    if [ "$sigma" == "off" ]; then
        curl -sf "http://localhost:${PORT}/set_var_gate?enabled=false" > /dev/null
        echo "  VarGate: OFF (baseline Gate 3n)"
    else
        curl -sf "http://localhost:${PORT}/set_var_gate?enabled=true&sigma=${sigma}&window=${VAR_WINDOW}" > /dev/null
        echo "  VarGate: ON  σ=$sigma  W=$VAR_WINDOW"
    fi
}

run_one() {
    local bag=$1 sigma=$2
    local safe_sigma="${sigma//./p}"  # replace . with p for filename
    local tag="${bag}_s${safe_sigma}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$bag  σ=$sigma ---"

    start_server "$tag"
    configure_var_gate "$sigma"

    curl -sf "http://localhost:${PORT}/reset_metrics" > "$log_dir/reset.json"
    sleep 1

    cd "$REPO_DIR"
    python3.12 $CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
        --server-port "${PORT}" \
        > "$log_dir/client.log" 2>&1 &
    local cpid=$!; sleep 3

    # rosbag played manually from gds container via FastDDS
    echo "  Waiting for bag play from gds container on topics: /camera/camera/color/image_raw ..."
    echo "  (play manually: ros2 bag play <bag> --rate 0.5 --topics ...)"
    # Old: ros2 bag play "$BAG_DIR/my_camera_bag_20260317_$bag" --rate 0.5 ...
    sleep 2  # give time for manual bag start
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
vgb  = t.get("var_gate_bypasses", 0)
sig  = "$sigma"

ctrl_f = "$LOG_BASE/${bag}_soff/metrics.json"
V = 0.0
skip_delta = 0.0
if os.path.exists(ctrl_f) and sig != "off":
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
    status = 'PASS' if V <= 0.10 and vgb > 0 and abs(skip_delta) <= 3.0 else 'FAIL'
else:
    status = 'BASELINE'

print(f"  σ={sig:5s}  skip={skip:.1f}%  Δskip={skip_delta:+.1f}pp  traj={tr:.1f}%  V={V:.4f}  var_bypasses={vgb:3d}  hz={hz:.2f}  => {status}")
PYEOF

    pkill -f "http_internvla_server_debug.*${PORT}" 2>/dev/null || true
    sleep 5
}

# Phase 1: Single-bag sweep
echo ""
echo "=== Phase 1: σ Sweep (bag $SWEEP_BAG) ==="
for sigma in "${SIGMA_VALUES[@]}"; do
    run_one "$SWEEP_BAG" "$sigma"
done

# Phase 2: Summary
echo ""
echo "=== Phase 2: Sweep Summary ==="
echo "σ      | skip%  | Δskip  | traj%  | V      | var_byp | verdict"
echo "-------|--------|--------|--------|--------|---------|--------"
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
ctrl_f=f"/tmp/gate9_var_gate/{bag}_soff/metrics.json"
if not os.path.exists(ctrl_f):
    print("baseline off not found"); exit()
with open(ctrl_f) as f: c=json.load(f)
cft=c.get("fresh_traj_outputs",0); cfa=c.get("fresh_action_outputs",0)
cskip=c.get("temporal_cache_skip_ratio",0)

results=[]
for sigma, tag in [("off","soff"),("0.05","s0p05"),("0.03","s0p03"),("0.01","s0p01")]:
    mf=f"/tmp/gate9_var_gate/{bag}_{tag}/metrics.json"
    if not os.path.exists(mf): continue
    with open(mf) as f: t=json.load(f)
    ft=t.get("fresh_traj_outputs",0); fa=t.get("fresh_action_outputs",0)
    tr=100*ft/max(1,ft+fa); skip=t.get("temporal_cache_skip_ratio",0)
    vgb=t.get("var_gate_bypasses",0); hz=t.get("joint_req_hz",0)
    V=cramers(cft,cfa,ft,fa)
    skip_delta=skip-cskip
    if sigma=="off":
        status="BASELINE"
    else:
        status="PASS" if V<=0.10 and vgb>0 and abs(skip_delta)<=3.0 else "FAIL"
    results.append((sigma,skip,skip_delta,tr,V,vgb,hz,status))
    print(f"  {sigma:5s} | {skip:6.1f}% | {skip_delta:+6.1f}pp | {tr:6.1f}% | {V:.4f} | {vgb:7d} | {status}")

winners=[r for r in results if r[7]=='PASS']
if winners:
    winner=min(winners, key=lambda r: r[4])  # lowest V (best quality)
    print(f"\nWinner: σ={winner[0]}  skip={winner[1]:.1f}%  V={winner[4]:.4f}  bypasses={winner[5]}")
    with open("/tmp/gate9_var_gate/winner.txt","w") as f: f.write(winner[0])
else:
    print("\nNo PASS — var gate either never fires or degrades V")
    with open("/tmp/gate9_var_gate/winner.txt","w") as f: f.write("off")
PYEOF

# Phase 3: 3-bag xval
WINNER=$(cat /tmp/gate9_var_gate/winner.txt 2>/dev/null || echo "off")
echo ""
echo "=== Phase 3: 3-bag xval with σ=$WINNER ==="
if [ "$WINNER" == "off" ]; then
    echo "No improvement over baseline. Gate 9 FAIL."
else
    for bag in "${XVAL_BAGS[@]}"; do
        run_one "$bag" "$WINNER"
    done
    echo ""
    echo "=== Gate 9 Final: 3-bag results (σ=$WINNER) ==="
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

SIGMA="$WINNER"
safe_sig=SIGMA.replace(".","p")
passes=0
for bag in ["073623","061841","063047"]:
    cf=f"/tmp/gate9_var_gate/{bag}_soff/metrics.json"
    tf=f"/tmp/gate9_var_gate/{bag}_s{safe_sig}/metrics.json"
    if not os.path.exists(cf) or not os.path.exists(tf):
        print(f"  bag={bag}: missing data"); continue
    with open(cf) as f: c=json.load(f)
    with open(tf) as f: t=json.load(f)
    cft=c.get("fresh_traj_outputs",0); cfa=c.get("fresh_action_outputs",0)
    ft=t.get("fresh_traj_outputs",0); fa=t.get("fresh_action_outputs",0)
    V=cramers(cft,cfa,ft,fa)
    skip=t.get("temporal_cache_skip_ratio",0); cskip=c.get("temporal_cache_skip_ratio",0)
    vgb=t.get("var_gate_bypasses",0)
    ok = V<=0.10 and vgb>0 and abs(skip-cskip)<=3.0
    if ok: passes+=1
    print(f"  bag={bag}: skip={skip:.1f}% (Δ{skip-cskip:+.1f}pp)  V={V:.4f}  var_byp={vgb}  {'PASS' if ok else 'FAIL'}")

verdict = "PASS" if passes==3 else ("PARTIAL" if passes>0 else "FAIL")
print(f"\n  Gate 9 FINAL: σ={SIGMA}  {verdict} ({passes}/3 bags)")
PYEOF2
fi

echo ""
echo "=== Gate 9 DONE ==="
