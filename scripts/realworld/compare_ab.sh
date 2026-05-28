#!/usr/bin/env bash
set -euo pipefail

# A/B comparison: sync vs async
# Usage:
#   ./compare_ab.sh [bag_path] [run_count]

SESSION_NAME_PREFIX="compare_ab"
# REPO_DIR="${REPO_DIR:-/workspace/InternNav}"  # old path from original Docker setup
REPO_DIR="${REPO_DIR:-/home/gdr/gd_vln/workspace/src/InternNav}"
CALIB_PATH="${CALIB_PATH:-$REPO_DIR/scripts/realworld/calib/calib_scout.txt}"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/checkpoints/InternVLA-N1-w-NavDP}"
DEVICE="${DEVICE:-cuda:0}"
RATE="${RATE:-0.7}"
SERVER_PYTHON="${SERVER_PYTHON:-/opt/venv/bin/python3}"
ROS_PYTHON="${ROS_PYTHON:-python3.12}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
SERVER_WARMUP_SEC="${SERVER_WARMUP_SEC:-70}"
BAG_START_DELAY_SEC="${BAG_START_DELAY_SEC:-85}"

# BAG_PATH: rosbag played manually from gds container via FastDDS
BAG_PATH="${1:-}"
RUN_COUNT="${2:-3}"
MODE="${3:-both}"  # both, sync, async

cd "$REPO_DIR"

run_sync() {
    local tag="$1"
    local run_dir="test_data/sync_${tag}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$run_dir"
    
    local SESSION_NAME="sync_ab_${tag}"
    
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux kill-session -t "$SESSION_NAME"
    fi
    
    tmux new-session -d -s "$SESSION_NAME" -n sync
    tmux split-window -h -t "$SESSION_NAME":0
    tmux split-window -v -t "$SESSION_NAME":0.0
    tmux split-window -v -t "$SESSION_NAME":0.1
    
    COMMON_PREFIX="source $ROS_SETUP; cd $REPO_DIR;"
    
    tmux send-keys -t "$SESSION_NAME":0.0 "$COMMON_PREFIX $SERVER_PYTHON scripts/realworld/http_internvla_server_debug.py --mode sync --device $DEVICE --model_path $MODEL_PATH --calib $CALIB_PATH 2>&1 | tee $run_dir/server.log" C-m
    tmux send-keys -t "$SESSION_NAME":0.1 "$COMMON_PREFIX $ROS_PYTHON scripts/realworld/scout_bridge.py 2>&1 | tee $run_dir/bridge.log" C-m
    tmux send-keys -t "$SESSION_NAME":0.2 "$COMMON_PREFIX sleep $SERVER_WARMUP_SEC; $ROS_PYTHON scripts/realworld/http_internvla_client_debug.py --mode sync --calib $CALIB_PATH 2>&1 | tee $run_dir/client.log" C-m
    tmux send-keys -t "$SESSION_NAME":0.3 "$COMMON_PREFIX sleep $BAG_START_DELAY_SEC; echo "[gds] play bag manually"" C-m
    
    echo "[SYNC] Started: $SESSION_NAME"
    sleep 2
}

run_async() {
    local tag="$1"
    local run_dir="test_data/async_${tag}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$run_dir"
    
    local SESSION_NAME="async_ab_${tag}"
    
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux kill-session -t "$SESSION_NAME"
    fi
    
    tmux new-session -d -s "$SESSION_NAME" -n async
    tmux split-window -h -t "$SESSION_NAME":0
    tmux split-window -v -t "$SESSION_NAME":0.0
    tmux split-window -v -t "$SESSION_NAME":0.1
    
    COMMON_PREFIX="source $ROS_SETUP; cd $REPO_DIR;"
    
    tmux send-keys -t "$SESSION_NAME":0.0 "$COMMON_PREFIX $SERVER_PYTHON scripts/realworld/http_internvla_server_debug.py --mode async --device $DEVICE --model_path $MODEL_PATH --calib $CALIB_PATH 2>&1 | tee $run_dir/server.log" C-m
    tmux send-keys -t "$SESSION_NAME":0.1 "$COMMON_PREFIX $ROS_PYTHON scripts/realworld/scout_bridge.py 2>&1 | tee $run_dir/bridge.log" C-m
    tmux send-keys -t "$SESSION_NAME":0.2 "$COMMON_PREFIX sleep $SERVER_WARMUP_SEC; $ROS_PYTHON scripts/realworld/http_internvla_client_debug.py --mode async --calib $CALIB_PATH 2>&1 | tee $run_dir/client.log" C-m
    tmux send-keys -t "$SESSION_NAME":0.3 "$COMMON_PREFIX sleep $BAG_START_DELAY_SEC; echo "[gds] play bag manually"" C-m
    
    echo "[ASYNC] Started: $SESSION_NAME"
    sleep 2
}

echo "=== A/B Comparison Test ==="
echo "Bag: $BAG_PATH"
echo "Mode: $MODE"
echo ""

if [ "$MODE" = "both" ] || [ "$MODE" = "sync" ]; then
    echo "Running SYNC baseline..."
    run_sync "baseline"
fi

if [ "$MODE" = "both" ] || [ "$MODE" = "async" ]; then
    echo "Running ASYNC test..."
    run_async "test"
fi

echo ""
echo "Done starting tests. Attach with:"
echo "  tmux attach -t sync_ab_baseline  # for sync"
echo "  tmux attach -t async_ab_test   # for async"
echo ""
echo "Extract metrics from client.log:"
echo "  grep 'Received Trajectory' client.log | wc -l"
echo "  grep 'HTTP.*Latency' client.log | awk '{sum+=\$NF; n++} END {print sum/n}'"