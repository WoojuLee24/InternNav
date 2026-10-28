#!/usr/bin/env bash
# Launch the S2 module (System-2 reasoner, ~1 Hz) standalone in the foreground.
# All flags pass through to the service:  ./run.sh --help
#   ./run.sh                            # Q4_K_M-latent GGUF @ 336px (defaults)
#   ./run.sh --s2-model f16 --resize 448 --kv-reuse
#   ./run.sh --host 0.0.0.0             # serve a remote orchestrator/manager (S2 off-device)
# Managed (daemonized) alternative:  deploy/run_manager.sh start s2
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/env.sh"
exec "$PY" -u -m deploy.s2_module.service "$@"
