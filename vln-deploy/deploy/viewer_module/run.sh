#!/usr/bin/env bash
# Launch the viewer module standalone in the foreground (3-panel live visualizer).
#   ./run.sh --view mjpeg                 # browser view at http://<host>:8089/
#   ./run.sh --view tk                    # native window (needs DISPLAY)
# Managed alternative:  deploy/run_manager.sh start viz   (or the one-shot: ... camera)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"
exec "$PY" -u -m deploy.viewer_module.service "$@"
