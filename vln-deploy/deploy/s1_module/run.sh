#!/usr/bin/env bash
# Launch the S1 module (System-1 controller, ~30 Hz) standalone in the foreground.
# All flags pass through to the service:  ./run.sh --help
#   ./run.sh                       # TRT FP16, 10 diffusion steps (defaults)
#   ./run.sh --no-trt --s1-steps 8 # PyTorch fp32, more steps
# Managed (daemonized) alternative:  deploy/run_manager.sh start s1
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"
exec "$PY" -u -m deploy.s1_module.service "$@"
