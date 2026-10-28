#!/usr/bin/env bash
# Self-contained InternVLA-N1 deployment — everything runs from inside this folder.
#   ./run.sh start --wait | status | test | demo | camera | prompt "<text>" | stop | clean
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy/run_manager.sh" "$@"
