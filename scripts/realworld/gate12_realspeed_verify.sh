#!/usr/bin/env bash
# Gate 12 — Real-Speed Cache Quality Verification (I-207)
# Hypothesis: Gate 10 found PROD traj=84.8% vs NOCACHE traj=67.2% on bag 073623
# at rate=1.0× (n_prod=66 small sample). This +17.6pp gain may be:
#   (a) genuine: cache triggers preferentially on action frames → forced S2 runs
#       fall on trajectory-phase frames → higher fresh_traj_ratio
#   (b) small-n artifact: n=66 at sigma ≈ 5.8pp → one lucky draw
#
# Gate 12 design:
#   - Reuse Gate 10 NOCACHE data for 073623 (n=229, traj=67.2%)
#   - Run PROD 5× on bag 073623 at rate=1.0×  (n≈330 pooled)
#   - Pool all PROD runs, compute V against Gate 10 NOCACHE
#   - Pass criterion: fresh_traj_PROD > fresh_traj_NOCACHE + 5pp (genuine improvement)
#     AND V ≤ 0.20 (PROD distribution is better but not a quality DECREASE)
#   - Fail criterion: fresh_traj_PROD converges to NOCACHE ± 5pp (artifact)
# Port 5805, cuda:1.
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

PORT=5805
DEVICE=cuda:1
CALIB="scripts/realworld/calib/calib_scout.txt"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
BAG_DIR="/workspace/rosbag"
LOG_BASE="/tmp/gate12_realspeed_verify"
mkdir -p "$LOG_BASE"
N_RUNS=5
BAG=073623

echo "=================================================="
echo " Gate 12 — Real-Speed Cache Quality Verify (I-207)"
echo " bag=$BAG  rate=1.0×  PROD × $N_RUNS  Port: $PORT"
echo "=================================================="

start_server() {
    pkill -f "http_internvla_server_debug.*$PORT" 2>/dev/null || true
    sleep 3
    python3 /workspace/InternNav/scripts/realworld/http_internvla_server_debug.py \
        --mode async --temperature 0.75 --kv-cache \
        --calib "$CALIB" --device "$DEVICE" --port "$PORT" \
        --pre-warm-frames 3 \
        > "$LOG_BASE/server.log" 2>&1 &
    SERVER_PID=$!
    echo "  Waiting for server (pid=$SERVER_PID)..."
    until curl -sf "http://localhost:${PORT}/async_metrics" > /dev/null 2>&1; do sleep 3; done
    echo "  Server ready"
    # Gate 3n production stack
    curl -sf "http://localhost:${PORT}/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:${PORT}/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:${PORT}/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:${PORT}/set_ema_fingerprint?enabled=true&alpha=0.10&transition_reset=true" > /dev/null
    curl -sf "http://localhost:${PORT}/set_slope_predict?enabled=true&threshold=0.010&window=3" > /dev/null
    echo "  Config: Gate 3n PROD stack  rate=1.0×"
}

run_prod_once() {
    local run=$1
    local tag="${BAG}_prod_r${run}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"
    echo ""
    echo "--- bag=$BAG  PROD run $run/$N_RUNS  rate=1.0× ---"

    curl -sf "http://localhost:${PORT}/reset_metrics" > "$log_dir/reset.json"
    sleep 1

    python3.12 /workspace/InternNav/$CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
        --server-port "${PORT}" \
        > "$log_dir/client.log" 2>&1 &
    local cpid=$!; sleep 3

    ros2 bag play "$BAG_DIR/my_camera_bag_20260317_$BAG" --rate 1.0 \
        --topics /camera/camera/color/image_raw \
                 /camera/camera/aligned_depth_to_color/image_raw \
                 /gdq/msg/gdq_odom > "$log_dir/bag.log" 2>&1 || true
    sleep 3
    curl -sf "http://localhost:${PORT}/async_metrics" > "$log_dir/metrics.json"
    kill $cpid 2>/dev/null || true

    python3 -c "
import json
with open('$log_dir/metrics.json') as f: t = json.load(f)
ft = t.get('fresh_traj_outputs', 0); fa = t.get('fresh_action_outputs', 0)
n = ft + fa; tr = 100 * ft / n if n > 0 else 0
skip = t.get('temporal_cache_skip_ratio', 0)
s2l  = t.get('s2_latency_ms', 0)
print(f'  run=$run: n_fresh={n}  skip={skip:.1f}%  traj={tr:.1f}%  s2_lat={s2l:.0f}ms')
"
}

# Start the server once and reuse across all PROD runs
start_server

for i in $(seq 1 $N_RUNS); do
    run_prod_once "$i"
done

pkill -f "http_internvla_server_debug.*${PORT}" 2>/dev/null || true
sleep 3

echo ""
echo "=== Gate 12 Summary ==="
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

base="/tmp/gate12_realspeed_verify"
BAG="073623"
N_RUNS=5

# Gate 10 NOCACHE reference (n=229, traj=67.2%)
# From: /tmp/gate10_realspeed/073623_nocache/metrics.json
nc_path="/tmp/gate10_realspeed/${BAG}_nocache/metrics.json"
if os.path.exists(nc_path):
    with open(nc_path) as f: nc = json.load(f)
    nc_ft = nc.get("fresh_traj_outputs", 0)
    nc_fa = nc.get("fresh_action_outputs", 0)
    nc_tr = 100 * nc_ft / max(1, nc_ft + nc_fa)
    print(f"  NOCACHE ref (Gate 10): n={nc_ft+nc_fa}  traj={nc_tr:.1f}%")
else:
    # Fallback: use known values from Gate 10
    nc_ft, nc_fa = 154, 75  # 229 total, 67.2% traj
    nc_tr = 67.2
    print(f"  NOCACHE ref (Gate 10 fallback): n={nc_ft+nc_fa}  traj={nc_tr:.1f}%")

# Pool all PROD runs
pool_ft, pool_fa = 0, 0
per_run = []
for i in range(1, N_RUNS+1):
    pf = f"{base}/{BAG}_prod_r{i}/metrics.json"
    if not os.path.exists(pf):
        print(f"  run {i}: missing data")
        continue
    with open(pf) as f: pr = json.load(f)
    ft = pr.get("fresh_traj_outputs", 0)
    fa = pr.get("fresh_action_outputs", 0)
    skip = pr.get("temporal_cache_skip_ratio", 0)
    tr = 100 * ft / max(1, ft+fa)
    pool_ft += ft; pool_fa += fa
    per_run.append((i, ft+fa, tr, skip))
    print(f"  run {i}: n_fresh={ft+fa}  traj={tr:.1f}%  skip={skip:.1f}%")

n_pooled = pool_ft + pool_fa
pool_tr = 100 * pool_ft / max(1, n_pooled)
V = cramers(nc_ft, nc_fa, pool_ft, pool_fa)
delta = pool_tr - nc_tr

print(f"\n  === Pooled PROD (N={N_RUNS} runs) ===")
print(f"  n_pooled={n_pooled}  traj={pool_tr:.1f}%  V={V:.4f}  delta={delta:+.1f}pp vs NOCACHE {nc_tr:.1f}%")

# Verdict
# Pass: PROD traj significantly ABOVE NOCACHE (confirms positive cache bias)
# Fail: PROD traj converges to NOCACHE (small-n artifact)
THRESHOLD_POSITIVE = 5.0  # pp above NOCACHE to confirm genuine improvement
THRESHOLD_V = 0.20         # V <= 0.20 (PROD is better but different, not degraded)

if delta >= THRESHOLD_POSITIVE and V <= THRESHOLD_V:
    verdict = "CONFIRMED — genuine cache quality bias at rate=1.0×"
    result = "PASS"
elif delta >= THRESHOLD_POSITIVE and V > THRESHOLD_V:
    verdict = "AMBIGUOUS — large improvement but V too high (V={:.3f})".format(V)
    result = "PARTIAL"
elif abs(delta) < THRESHOLD_POSITIVE:
    verdict = "ARTIFACT — PROD converges to NOCACHE (small-n artifact confirmed)"
    result = "FAIL"
else:
    verdict = f"DEGRADED — PROD worse than NOCACHE by {-delta:.1f}pp"
    result = "FAIL"

print(f"\n  Gate 12: {result}")
print(f"  Finding: {verdict}")
print(f"  Implication for paper: {'Rate=1.0x cache stack raises fresh_traj beyond NOCACHE' if result=='PASS' else 'Gate 10 +17.6pp was small-n artifact; no correction needed'}")
PYEOF

echo ""
echo "=== Gate 12 DONE ==="
