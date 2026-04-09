#!/usr/bin/env bash
set -euo pipefail

# Simple 4-pane sync check launcher.
# Run this inside the container where ROS Jazzy is installed.

SESSION_NAME="sync_check"
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
SERVER_WARMUP_SEC="${SERVER_WARMUP_SEC:-70}"
BAG_START_DELAY_SEC="${BAG_START_DELAY_SEC:-85}"

BAG_PATH="${1:-rosbag/my_camera_bag_20260310_035208}"
RUN_TAG="${2:-manual}"
RUN_DIR="test_data/baseline_sync_check_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"

cd "$REPO_DIR"
mkdir -p "$RUN_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" -n sync
tmux split-window -h -t "$SESSION_NAME":0
tmux split-window -v -t "$SESSION_NAME":0.0
tmux split-window -v -t "$SESSION_NAME":0.1

COMMON_PREFIX="source $ROS_SETUP; cd $REPO_DIR;"

# Pane 0: server
tmux send-keys -t "$SESSION_NAME":0.0 "$COMMON_PREFIX $SERVER_PYTHON scripts/realworld/http_internvla_server_debug.py --mode sync --max-new-tokens 80 --resize_w 256 --resize_h 256 --num_history 1 --plan_step_gap 12 --device $DEVICE --model_path $MODEL_PATH --calib $CALIB_PATH $SERVER_EXTRA_ARGS 2>&1 | tee $RUN_DIR/server_stdout.log" C-m

# Pane 1: scout bridge
tmux send-keys -t "$SESSION_NAME":0.1 "$COMMON_PREFIX $ROS_PYTHON scripts/realworld/scout_bridge.py 2>&1 | tee $RUN_DIR/scout_bridge.log" C-m

# Pane 2: client (wait for model server warmup)
tmux send-keys -t "$SESSION_NAME":0.2 "$COMMON_PREFIX sleep $SERVER_WARMUP_SEC; $ROS_PYTHON scripts/realworld/http_internvla_client_debug.py --mode sync --jpeg-quality $JPEG_QUALITY --depth-png-compress $DEPTH_PNG_COMPRESS --calib $CALIB_PATH $CLIENT_EXTRA_ARGS 2>&1 | tee $RUN_DIR/client.log" C-m

# Pane 3: rosbag player
tmux send-keys -t "$SESSION_NAME":0.3 "$COMMON_PREFIX sleep $BAG_START_DELAY_SEC; ros2 bag play $BAG_PATH --rate $RATE $BAG_PLAY_EXTRA_ARGS 2>&1 | tee $RUN_DIR/rosbag_play.log" C-m

echo "tmux session: $SESSION_NAME"
echo "logs: $REPO_DIR/$RUN_DIR"
echo "attach with: tmux attach -t $SESSION_NAME"
echo "stop with: tmux kill-session -t $SESSION_NAME"
echo "env overrides: REPO_DIR MODEL_PATH CALIB_PATH DEVICE RATE SERVER_PYTHON ROS_PYTHON ROS_SETUP SERVER_EXTRA_ARGS CLIENT_EXTRA_ARGS BAG_PLAY_EXTRA_ARGS JPEG_QUALITY DEPTH_PNG_COMPRESS"

if [ -z "${NO_ATTACH:-}" ]; then
  tmux attach -t "$SESSION_NAME"
fi
