#!/usr/bin/env bash
# Launch the camera module standalone in the foreground (V4L2 -> ZMQ PUB :5605).
#   ./run.sh --device 0 --size 336        # size must match the S2 module's --resize
# Managed alternative:  deploy/run_manager.sh start cam   (or the one-shot: ... camera)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"
exec "$PY" -u -m deploy.camera_module.service "$@"
