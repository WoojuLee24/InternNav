#!/usr/bin/env bash
set -eo pipefail

# Run inside Docker container where ROS2 is installed
CONTAINER_NAME="${MODEL_SERVER_NAME:-vlnav_internvla_server}"

# Check if we're already in the container
if [ -f "/.dockerenv" ]; then
    source /opt/ros/jazzy/setup.bash
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
    cd "$SCRIPT_DIR"
    pkill -f "python3.12 .*scout_bridge\.py" >/dev/null 2>&1 || true
    exec python3.12 ./scout_bridge.py
else
    # Run inside container
    docker exec -w /workspace/InternNav/scripts/realworld "$CONTAINER_NAME" bash -c "source /opt/ros/jazzy/setup.bash && python3.12 ./scout_bridge.py"
fi
