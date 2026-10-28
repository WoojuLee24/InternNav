#!/usr/bin/env bash
# The deployment manager (master):  lifecycle + integration tests + demo.
#   ./run_manager.sh start [s1|s2|all] [--wait]     spawn module daemons (logs/, run/)
#   ./run_manager.sh status                         health table (safe while live)
#   ./run_manager.sh test [--suite ...] [--keep-up] integration test suites (PASS/FAIL)
#   ./run_manager.sh demo [--seconds 8]             full async control loop on the modules
#   ./run_manager.sh logs <s1|s2> [-n 40]           tail a module log
#   ./run_manager.sh stop [s1|s2|all]               graceful shutdown
#   ./run_manager.sh clean                          force-free ports + kill leftovers
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common/env.sh"
exec "$PY" -u -m deploy.manager "$@"
