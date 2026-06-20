#!/bin/bash
# Usage: bash kube/download.sh <pod> <src> <dst>
set -euo pipefail

POD="${1:?Usage: $0 <pod> <src> <dst>}"
SRC="${2:?Usage: $0 <pod> <src> <dst>}"
DST="${3:-.}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

rsync -avz --progress -e "$SCRIPT_DIR/kube-rsync.sh" "$POD:$SRC" "$DST"
