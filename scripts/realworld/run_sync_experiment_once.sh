#!/usr/bin/env bash
set -euo pipefail

# Runs one sync_check_tmux experiment in a controlled way:
# - cleans stale related processes
# - launches tmux workflow
# - waits fixed duration
# - force-stops session/processes

CONTAINER_NAME="${CONTAINER_NAME:-vlnav_internvla_server}"
REPO_DIR_IN_CONTAINER="${REPO_DIR_IN_CONTAINER:-/workspace/InternNav}"
RUN_TAG="${RUN_TAG:-manual}"
BAG_PATH="${BAG_PATH:-rosbag/my_camera_bag_20260310_035208}"
RUN_DURATION_SEC="${RUN_DURATION_SEC:-420}"
SERVER_WARMUP_SEC="${SERVER_WARMUP_SEC:-20}"
BAG_START_DELAY_SEC="${BAG_START_DELAY_SEC:-30}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
CLIENT_EXTRA_ARGS="${CLIENT_EXTRA_ARGS:-}"
BAG_PLAY_EXTRA_ARGS="${BAG_PLAY_EXTRA_ARGS:-}"

docker exec "$CONTAINER_NAME" bash -lc "
set -euo pipefail
tmux kill-session -t sync_check 2>/dev/null || true
pkill -9 -f 'http_internvla_server_debug.py' || true
pkill -9 -f 'http_internvla_client_debug.py' || true
pkill -9 -f 'scripts/realworld/scout_bridge.py' || true
pkill -9 -f 'ros2 bag play' || true

cd '$REPO_DIR_IN_CONTAINER'
NO_ATTACH=1 \
REPO_DIR='$REPO_DIR_IN_CONTAINER' \
SERVER_PYTHON=/opt/venv/bin/python3 \
ROS_PYTHON=python3.12 \
ROS_SETUP=/opt/ros/jazzy/setup.bash \
SERVER_WARMUP_SEC='$SERVER_WARMUP_SEC' \
BAG_START_DELAY_SEC='$BAG_START_DELAY_SEC' \
SERVER_EXTRA_ARGS=\"$SERVER_EXTRA_ARGS\" \
CLIENT_EXTRA_ARGS=\"$CLIENT_EXTRA_ARGS\" \
BAG_PLAY_EXTRA_ARGS=\"$BAG_PLAY_EXTRA_ARGS\" \
scripts/realworld/sync_check_tmux.sh '$BAG_PATH' '$RUN_TAG'

sleep '$RUN_DURATION_SEC'
tmux kill-session -t sync_check 2>/dev/null || true
pkill -9 -f 'http_internvla_server_debug.py' || true
pkill -9 -f 'http_internvla_client_debug.py' || true
pkill -9 -f 'scripts/realworld/scout_bridge.py' || true
pkill -9 -f 'ros2 bag play' || true

ls -d '$REPO_DIR_IN_CONTAINER'/test_data/baseline_sync_check_'$RUN_TAG'_* 2>/dev/null | sort | tail -n 1
"
