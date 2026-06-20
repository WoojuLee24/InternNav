#!/bin/bash
# Wrapper for rsync over kubectl exec.
# rsync -e CMD calls: CMD <host> rsync --server ...
# We shift off the host arg kubectl doesn't need, then pass the rest to exec.
POD="${1:-internnav-train1-1-0}"
shift  # drop the host arg rsync injects
KUBECONFIG="${KUBECONFIG:-$(cd "$(dirname "$0")" && pwd)/internnav-kubeconfig.yaml}" \
  kubectl exec -i "$POD" -- "$@"
