#!/usr/bin/env bash
set -euo pipefail

# Real-robot A/B test launcher — compare any mechanism combination live.
# See docs/METHOD_IMPACT.md for quantitative impact of each mechanism.
#
# Usage:
#   ./async_check_tmux.sh [--profile <name>] [run_tag]
#   ./async_check_tmux.sh --list-profiles
#
# Profiles (set of mechanisms enabled):
#   production   Full stack: τ=0.92 + MH=15 + AA + TR-EMA + slope + warmup + var_gate  (DEFAULT)
#   gate3n      Gate 3n: τ=0.92 + MH=15 + AA + TR-EMA + slope (no warmup, no var_gate)
#   gate3g      Gate 3g: τ=0.92 + MH=15 + AA (no EMA, no slope, no warmup, no var_gate)
#   gate3f      Gate 3f: τ=0.92 + MH=10 + AA (pre-EMA, pre-MH-tuning)
#   gate0       NOCACHE: τ=0.0 (all caching disabled, pure async baseline)
#
# Per-mechanism overrides (env vars, override any profile):
#   DISABLE_CACHE=1     DISABLE_AA=1        DISABLE_EMA=1
#   DISABLE_SLOPE=1     DISABLE_WARMUP=1    DISABLE_VAR_GATE=1
#   MAX_HOLD=<int>      TAU=<float>         EMA_ALPHA=<float>
#   SLOPE_THRESHOLD=<float>  TEMPERATURE=<float>
#   INSTRUCTION="..."   (navigation goal, overrides server default)
#
# Examples:
#   ./async_check_tmux.sh                            # production stack
#   ./async_check_tmux.sh --profile gate3g           # Gate 3g only
#   DISABLE_VAR_GATE=1 ./async_check_tmux.sh         # production minus var_gate
#   DISABLE_EMA=1 DISABLE_SLOPE=1 ./async_check_tmux.sh --profile gate3n my_test
#   MAX_HOLD=20 TAU=0.95 ./async_check_tmux.sh       # custom threshold/hold
#   INSTRUCTION="Go to the kitchen and stop." ./async_check_tmux.sh  # custom goal

SESSION_NAME="async_check"
REPO_DIR="${REPO_DIR:-/home/gdr/gd_vln/workspace/src/InternNav}"
CALIB_PATH="${CALIB_PATH:-$REPO_DIR/scripts/realworld/calib/calib_scout.txt}"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/checkpoints/InternVLA-N1-w-NavDP}"
DEVICE="${DEVICE:-cuda:0}"
SERVER_PYTHON="${SERVER_PYTHON:-/opt/venv/bin/python3}"
ROS_PYTHON="${ROS_PYTHON:-python3.12}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
CLIENT_EXTRA_ARGS="${CLIENT_EXTRA_ARGS:-}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
DEPTH_PNG_COMPRESS="${DEPTH_PNG_COMPRESS:-6}"
SERVER_WARMUP_SEC="${SERVER_WARMUP_SEC:-15}"
SERVER_PORT="${SERVER_PORT:-5802}"

# Production overridable defaults
KV_CACHE="${KV_CACHE:-1}"
TEMPERATURE="${TEMPERATURE:-0.75}"
REPETITION_PENALTY="${REPETITION_PENALTY:-}"
PRE_WARM_FRAMES="${PRE_WARM_FRAMES:-3}"
INSTRUCTION="${INSTRUCTION:-}"  # empty = use server default

# Mechanism defaults — capture raw env overrides before profile overwrites them
PROFILE="${PROFILE:-production}"
_TAU_ENV="${TAU:-}"; _MH_ENV="${MAX_HOLD:-}"; _AA_ENV="${AA:-}"
_EMA_ENV="${EMA:-}"; _SLOPE_ENV="${SLOPE:-}"; _WARMUP_ENV="${WARMUP:-}"
_VG_ENV="${VAR_GATE:-}"; _EMA_ALPHA_ENV="${EMA_ALPHA:-}"; _SLOPE_THR_ENV="${SLOPE_THRESHOLD:-}"
_VG_SIGMA_ENV="${VAR_GATE_SIGMA:-}"

TAU="${TAU:-0.92}"
MAX_HOLD="${MAX_HOLD:-15}"
AA="${AA:-1}"
EMA="${EMA:-1}"
EMA_ALPHA="${EMA_ALPHA:-0.10}"
SLOPE="${SLOPE:-1}"
SLOPE_THRESHOLD="${SLOPE_THRESHOLD:-0.010}"
WARMUP="${WARMUP:-1}"
VAR_GATE="${VAR_GATE:-1}"
VAR_GATE_SIGMA="${VAR_GATE_SIGMA:-0.01}"

# ---------- Argument parsing ----------
LIST_PROFILES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --profile)
      shift
      PROFILE="$1"
      shift
      ;;
    --list-profiles)
      LIST_PROFILES=1
      shift
      ;;
    --help|-h)
      head -30 "$0" | grep -E "^# " | sed 's/^# //'
      exit 0
      ;;
    *)
      # First non-flag arg is run_tag
      RUN_TAG="$1"
      shift
      break
      ;;
  esac
done
RUN_TAG="${RUN_TAG:-${1:-async_test}}"

# Show profiles and exit
if [ "$LIST_PROFILES" = 1 ]; then
  echo "Available profiles:"
  echo "  production   Full stack (default)"
  echo "  gate3n       τ=0.92 + MH=15 + AA + TR-EMA + slope"
  echo "  gate3g       τ=0.92 + MH=15 + AA (no EMA/slope/warmup/var_gate)"
  echo "  gate3f       τ=0.92 + MH=10 + AA (Gate 3f baseline)"
  echo "  gate0        NOCACHE: τ=0.0 (pure async baseline)"
  echo ""
  echo "Override any mechanism via env vars (highest priority):"
  echo "  DISABLE_CACHE=1  DISABLE_AA=1  DISABLE_EMA=1"
  echo "  DISABLE_SLOPE=1  DISABLE_WARMUP=1  DISABLE_VAR_GATE=1"
  echo "  TAU=<float>  MAX_HOLD=<int>  EMA_ALPHA=<float>"
  echo "  SLOPE_THRESHOLD=<float>  TEMPERATURE=<float>"
  echo "  INSTRUCTION=<string>      (navigation goal, overrides server default)"
  echo ""
  echo "See docs/METHOD_IMPACT.md for quantitative impact of each mechanism."
  exit 0
fi

# ---------- Apply profile ----------
case "$PROFILE" in
  gate0)
    TAU=0.0; MAX_HOLD=0; AA=0; EMA=0; SLOPE=0; WARMUP=0; VAR_GATE=0
    ;;
  gate3f)
    TAU=0.92; MAX_HOLD=10; AA=1; EMA=0; SLOPE=0; WARMUP=0; VAR_GATE=0
    ;;
  gate3g)
    TAU=0.92; MAX_HOLD=15; AA=1; EMA=0; SLOPE=0; WARMUP=0; VAR_GATE=0
    ;;
  gate3n)
    TAU=0.92; MAX_HOLD=15; AA=1; EMA=1; SLOPE=1; WARMUP=0; VAR_GATE=0
    ;;
  production)
    TAU=0.92; MAX_HOLD=15; AA=1; EMA=1; SLOPE=1; WARMUP=1; VAR_GATE=1
    ;;
  *)
    echo "Unknown profile: $PROFILE"
    echo "Use --list-profiles to see available options."
    exit 1
    ;;
esac

# ---------- Env overrides (highest priority: apply after profile) ----------
[ -n "$_TAU_ENV" ] && TAU="$_TAU_ENV"
[ -n "$_MH_ENV" ] && MAX_HOLD="$_MH_ENV"
[ -n "$_AA_ENV" ] && AA="$_AA_ENV"
[ -n "$_EMA_ENV" ] && EMA="$_EMA_ENV"
[ -n "$_SLOPE_ENV" ] && SLOPE="$_SLOPE_ENV"
[ -n "$_WARMUP_ENV" ] && WARMUP="$_WARMUP_ENV"
[ -n "$_VG_ENV" ] && VAR_GATE="$_VG_ENV"
[ -n "$_EMA_ALPHA_ENV" ] && EMA_ALPHA="$_EMA_ALPHA_ENV"
[ -n "$_SLOPE_THR_ENV" ] && SLOPE_THRESHOLD="$_SLOPE_THR_ENV"
[ -n "$_VG_SIGMA_ENV" ] && VAR_GATE_SIGMA="$_VG_SIGMA_ENV"

# DISABLE_* flags (convenience, same priority)
[ "${DISABLE_CACHE:-0}" = 1 ] && TAU=0.0
[ "${DISABLE_AA:-0}" = 1 ] && AA=0
[ "${DISABLE_EMA:-0}" = 1 ] && EMA=0
[ "${DISABLE_SLOPE:-0}" = 1 ] && SLOPE=0
[ "${DISABLE_WARMUP:-0}" = 1 ] && WARMUP=0
[ "${DISABLE_VAR_GATE:-0}" = 1 ] && VAR_GATE=0

# ---------- Print config (before filesystem ops) ----------
echo "============================================="
echo " PROFILE:        $PROFILE"
echo " TAG:            $RUN_TAG"
echo " TEMPERATURE:    $TEMPERATURE"
echo " PRE_WARM:       ${PRE_WARM_FRAMES} frames"
echo " SERVER_PORT:    ${SERVER_PORT}"
echo " DEVICE:         ${DEVICE}"
echo " INSTRUCTION:    ${INSTRUCTION:-default (Exit door. Turn left...)}"
echo "---------------------------------------------"
echo " Mechanisms:"
echo "   Cache (τ=$TAU)     [$( [ "$TAU" != "0.0" ] && echo "ON" || echo "OFF")]"
echo "   MaxHold          [$( [ "$MAX_HOLD" -gt 0 ] 2>/dev/null && echo "ON MH=$MAX_HOLD" || echo "OFF")]"
echo "   ActionAware      [$( [ "$AA" = 1 ] && echo "ON" || echo "OFF")]"
echo "   TR-EMA α=$EMA_ALPHA    [$( [ "$EMA" = 1 ] && echo "ON" || echo "OFF")]"
echo "   Slope δ=$SLOPE_THRESHOLD [$( [ "$SLOPE" = 1 ] && echo "ON" || echo "OFF")]"
echo "   EMA-Warmup       [$( [ "$WARMUP" = 1 ] && echo "ON" || echo "OFF")]"
echo "   VarGate σ=$VAR_GATE_SIGMA  [$( [ "$VAR_GATE" = 1 ] && echo "ON" || echo "OFF")]"
echo "============================================="

# ---------- Build runtime config ----------
RUN_DIR="test_data/async_check_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
cd "$REPO_DIR"
mkdir -p "$RUN_DIR" 2>/dev/null || echo "  (log dir: $RUN_DIR)"

# Server extra args
SERVER_GEN_ARGS="--kv-cache --temperature $TEMPERATURE --pre-warm-frames $PRE_WARM_FRAMES"
[ -n "$REPETITION_PENALTY" ] && SERVER_GEN_ARGS="$SERVER_GEN_ARGS --repetition-penalty $REPETITION_PENALTY"
[ -n "$INSTRUCTION" ] && SERVER_GEN_ARGS="$SERVER_GEN_ARGS --instruction '$INSTRUCTION'"
SERVER_FULL_EXTRA="$SERVER_EXTRA_ARGS $SERVER_GEN_ARGS"

# Client extra args
CLIENT_GEN_ARGS="--kv-cache --temperature $TEMPERATURE"
[ -n "$REPETITION_PENALTY" ] && CLIENT_GEN_ARGS="$CLIENT_GEN_ARGS --repetition-penalty $REPETITION_PENALTY"
CLIENT_FULL_EXTRA="$CLIENT_EXTRA_ARGS $CLIENT_GEN_ARGS"

# Build the curl config string
CONFIG_CMDS=""
add_config() { CONFIG_CMDS="${CONFIG_CMDS}$1; "; }

if [ "$TAU" != "0.0" ]; then
  add_config "curl -sf 'http://localhost:${SERVER_PORT}/set_temporal_threshold?threshold=${TAU}' > /dev/null"
  add_config "curl -sf 'http://localhost:${SERVER_PORT}/set_max_hold_frames?frames=${MAX_HOLD}' > /dev/null"
fi
[ "$AA" = 1 ] && add_config "curl -sf 'http://localhost:${SERVER_PORT}/set_action_aware?enabled=true' > /dev/null"
[ "$EMA" = 1 ] && add_config "curl -sf 'http://localhost:${SERVER_PORT}/set_ema_fingerprint?enabled=true&alpha=${EMA_ALPHA}&transition_reset=true' > /dev/null"
[ "$SLOPE" = 1 ] && add_config "curl -sf 'http://localhost:${SERVER_PORT}/set_slope_predict?enabled=true&threshold=${SLOPE_THRESHOLD}&window=3' > /dev/null"
[ "$WARMUP" = 1 ] && add_config "curl -sf 'http://localhost:${SERVER_PORT}/set_ema_warmup?enabled=true&warmup_frames=10&alpha_warm=0.5' > /dev/null"
[ "$VAR_GATE" = 1 ] && add_config "curl -sf 'http://localhost:${SERVER_PORT}/set_var_gate?enabled=true&sigma=${VAR_GATE_SIGMA}&window=5' > /dev/null"

# ---------- Launch tmux ----------
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" -n async
# Layout: 0.0 server (top-left), 0.1 scout_bridge (right), 0.2 client (bottom-left)
tmux split-window -h -t "$SESSION_NAME":0
tmux split-window -v -t "$SESSION_NAME":0.0

COMMON_PREFIX="source $ROS_SETUP; cd $REPO_DIR;"

# Pane 0: server with ASYNC mode
tmux send-keys -t "$SESSION_NAME":0.0 "$COMMON_PREFIX $SERVER_PYTHON scripts/realworld/http_internvla_server_debug.py --mode async --max-new-tokens 80 --resize_w 256 --resize_h 256 --num_history 1 --plan_step_gap 12 --device $DEVICE --model_path $MODEL_PATH --calib $CALIB_PATH --port $SERVER_PORT $SERVER_FULL_EXTRA 2>&1 | tee $RUN_DIR/server_stdout.log" C-m

# Pane 1: scout bridge
tmux send-keys -t "$SESSION_NAME":0.1 "$COMMON_PREFIX $ROS_PYTHON scripts/realworld/scout_bridge.py 2>&1 | tee $RUN_DIR/scout_bridge.log" C-m

# Pane 2: client with production stack config applied after server ready
CLIENT_CMD="$COMMON_PREFIX sleep $SERVER_WARMUP_SEC; "
CLIENT_CMD+="${CONFIG_CMDS} "
CLIENT_CMD+="echo '=== CONFIG APPLIED ($PROFILE) ==='; "
CLIENT_CMD+="$ROS_PYTHON scripts/realworld/http_internvla_client_debug.py --mode async --jpeg-quality $JPEG_QUALITY --depth-png-compress $DEPTH_PNG_COMPRESS --calib $CALIB_PATH --server-port $SERVER_PORT $CLIENT_FULL_EXTRA 2>&1 | tee $RUN_DIR/client.log"
tmux send-keys -t "$SESSION_NAME":0.2 "$CLIENT_CMD" C-m

echo ""
echo "  tmux session: $SESSION_NAME"
echo "  logs: $REPO_DIR/$RUN_DIR"
echo "  attach:  tmux attach -t $SESSION_NAME"
echo "  stop:    tmux kill-session -t $SESSION_NAME"
echo ""
echo "  Play rosbag from gds container manually after config is applied."
echo "  e.g. (gds): ros2 bag play /path/to/bag --rate 0.5 --loop"
echo ""
echo "  Panes: Ctrl-b 0 (server)  1 (scout_bridge)  2 (client+config)"
echo ""

# Auto-attach if not already in tmux
if [ -z "${NO_ATTACH:-}" ] && [ -z "${TMUX:-}" ]; then
  tmux attach -t "$SESSION_NAME"
elif [ -n "${TMUX:-}" ]; then
  echo "  Already inside tmux. Switch with: tmux switch-client -t $SESSION_NAME"
fi
