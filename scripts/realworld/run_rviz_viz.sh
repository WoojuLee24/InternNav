#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd "$SCRIPT_DIR"

cleanup() {
  [[ -n "${VIZ_PID:-}" ]] && kill "$VIZ_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

pkill -f "python .*viz_stack\.py" >/dev/null 2>&1 || true
python ./viz_stack.py --ros-args -p use_sim_time:=true &
VIZ_PID=$!

exec rviz2 --ros-args -p use_sim_time:=true
