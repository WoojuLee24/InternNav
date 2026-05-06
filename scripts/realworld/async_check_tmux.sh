#!/usr/bin/env bash
set -euo pipefail

# Async 4-pane check launcher.
# Usage:
#   ./async_check_tmux.sh [bag_path] [run_tag]
#
# Experiments:
#   - KV OFF: --mode async (no KV-cache flag)
#   - KV ON: --mode async --kv-cache
#   - KV ON + temp 0.8: --mode async --kv-cache --temperature 0.8
#   - KV ON + temp 0.8 + rep_pen 1.1: --mode async --kv-cache --temperature 0.8 --repetition-penalty 1.1

SESSION_NAME="async_check"
REPO_DIR="${REPO_DIR:-/workspace/InternNav}"
CALIB_PATH="${CALIB_PATH:-$REPO_DIR/scripts/realworld/calib/calib_scout.txt}"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/checkpoints/InternVLA-N1-w-NavDP}"
DEVICE="${DEVICE:-cuda:0}"
RATE="${RATE:-0.7}"
SERVER_PYTHON="${SERVER_PYTHON:-/opt/venv/bin/python3}"
ROS_PYTHON="${ROS_PYTHON:-python3.12}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
CLIENT_EXTRA_ARGS="${CLIENT_EXTRA_ARGS:-}"
BAG_PLAY_EXTRA_ARGS="${BAG_PLAY_EXTRA_ARGS:-}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
DEPTH_PNG_COMPRESS="${DEPTH_PNG_COMPRESS:-6}"
SERVER_WARMUP_SEC="${SERVER_WARMUP_SEC:-7}"
BAG_START_DELAY_SEC="${BAG_START_DELAY_SEC:-8}"

KV_CACHE="${KV_CACHE:-}"
TEMPERATURE="${TEMPERATURE:-}"
REPETITION_PENALTY="${REPETITION_PENALTY:-}"

BAG_PATH="${1:-../rosbag/my_camera_bag_20260317_073623}"
RUN_TAG="${2:-async_test}"
RUN_DIR="test_data/async_check_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"

cd "$REPO_DIR"
mkdir -p "$RUN_DIR"

# Build server extra args
SERVER_GEN_ARGS=""
if [ -n "$KV_CACHE" ]; then
  SERVER_GEN_ARGS="$SERVER_GEN_ARGS --kv-cache"
fi
if [ -n "$TEMPERATURE" ]; then
  SERVER_GEN_ARGS="$SERVER_GEN_ARGS --temperature $TEMPERATURE"
fi
if [ -n "$REPETITION_PENALTY" ]; then
  SERVER_GEN_ARGS="$SERVER_GEN_ARGS --repetition-penalty $REPETITION_PENALTY"
fi
SERVER_FULL_EXTRA="$SERVER_EXTRA_ARGS $SERVER_GEN_ARGS"

# Build client extra args  
CLIENT_GEN_ARGS=""
if [ -n "$KV_CACHE" ]; then
  CLIENT_GEN_ARGS="$CLIENT_GEN_ARGS --kv-cache"
fi
if [ -n "$TEMPERATURE" ]; then
  CLIENT_GEN_ARGS="$CLIENT_GEN_ARGS --temperature $TEMPERATURE"
fi
if [ -n "$REPETITION_PENALTY" ]; then
  CLIENT_GEN_ARGS="$CLIENT_GEN_ARGS --repetition-penalty $REPETITION_PENALTY"
fi
CLIENT_FULL_EXTRA="$CLIENT_EXTRA_ARGS $CLIENT_GEN_ARGS"

echo "===== EXPERIMENT CONFIG ====="
echo "KV_CACHE: ${KV_CACHE:-OFF (default)}"
echo "TEMPERATURE: ${TEMPERATURE:-1.0 (default)}"
echo "REPETITION_PENALTY: ${REPETITION_PENALTY:-1.0 (default)}"
echo "SERVER_EXTRA: $SERVER_FULL_EXTRA"
echo "CLIENT_EXTRA: $CLIENT_FULL_EXTRA"
echo "======================="

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" -n async
tmux split-window -h -t "$SESSION_NAME":0
tmux split-window -v -t "$SESSION_NAME":0.0
tmux split-window -v -t "$SESSION_NAME":0.1

COMMON_PREFIX="source $ROS_SETUP; cd $REPO_DIR;"

# Pane 0: server with ASYNC mode and experiment flags
tmux send-keys -t "$SESSION_NAME":0.0 "$COMMON_PREFIX $SERVER_PYTHON scripts/realworld/http_internvla_server_debug.py --mode async --max-new-tokens 80 --resize_w 256 --resize_h 256 --num_history 1 --plan_step_gap 12 --device $DEVICE --model_path $MODEL_PATH --calib $CALIB_PATH $SERVER_FULL_EXTRA 2>&1 | tee $RUN_DIR/server_stdout.log" C-m

# Pane 1: scout bridge
tmux send-keys -t "$SESSION_NAME":0.1 "$COMMON_PREFIX $ROS_PYTHON scripts/realworld/scout_bridge.py 2>&1 | tee $RUN_DIR/scout_bridge.log" C-m

# Pane 2: client with ASYNC mode and experiment flags
tmux send-keys -t "$SESSION_NAME":0.2 "$COMMON_PREFIX sleep $SERVER_WARMUP_SEC; $ROS_PYTHON scripts/realworld/http_internvla_client_debug.py --mode async --jpeg-quality $JPEG_QUALITY --depth-png-compress $DEPTH_PNG_COMPRESS --calib $CALIB_PATH $CLIENT_FULL_EXTRA 2>&1 | tee $RUN_DIR/client.log" C-m

# Pane 3: rosbag player
tmux send-keys -t "$SESSION_NAME":0.3 "$COMMON_PREFIX sleep $BAG_START_DELAY_SEC; ros2 bag play $BAG_PATH --rate $RATE $BAG_PLAY_EXTRA_ARGS 2>&1 | tee $RUN_DIR/rosbag_play.log" C-m

echo "tmux session: $SESSION_NAME"
echo "logs: $REPO_DIR/$RUN_DIR"
echo "attach with: tmux attach -t $SESSION_NAME"
echo "stop with: tmux kill-session -t $SESSION_NAME"
echo ""
echo "Usage:"
echo "  # Run with default bag"
echo "  ./async_check_tmux.sh"
echo ""
echo "  # Run with specific bag"
echo "  ./async_check_tmux.sh rosbag/my_camera_bag_20260317_073623 test1"
echo ""
echo "  # Run sync baseline comparison"
echo "  REPO_DIR=/path/to/InternNav ./sync_check_tmux.sh rosbag/my_camera_bag_20260317_073623 baseline"

# Auto-detect if already inside a tmux session — avoid nested attach
if [ -z "${NO_ATTACH:-}" ] && [ -z "${TMUX:-}" ]; then
  tmux attach -t "$SESSION_NAME"
elif [ -n "${TMUX:-}" ]; then
  echo "Already inside tmux. Switch to experiment session with:"
  echo "  tmux switch-client -t $SESSION_NAME"
fi
