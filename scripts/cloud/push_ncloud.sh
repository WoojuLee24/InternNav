#!/usr/bin/env bash
# push_ncloud.sh — Build and push NexusNav training image to Naver Cloud Container Registry
#
# Usage:
#   ./scripts/cloud/push_ncloud.sh <registry_endpoint> [tag]
#
# Example:
#   ./scripts/cloud/push_ncloud.sh abc123.kr.ncr.ntruss.com v3-20261022
#
# Requirements:
#   - Docker installed and daemon running
#   - NCLOUD_ACCESS_KEY and NCLOUD_SECRET_KEY environment variables set
#     (or interactive login via --login-only flag)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── argument parsing ──────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 <registry_endpoint> [tag]"
    echo "  registry_endpoint: e.g. abc123.kr.ncr.ntruss.com"
    echo "  tag:               image tag (default: latest)"
    echo ""
    echo "Env vars:"
    echo "  NCLOUD_ACCESS_KEY  — NCloud API access key (for non-interactive login)"
    echo "  NCLOUD_SECRET_KEY  — NCloud API secret key"
    exit 1
}

[[ $# -lt 1 ]] && usage

REGISTRY="${1}"
TAG="${2:-latest}"
IMAGE_NAME="nexusnav"
FULL_TAG="${REGISTRY}/${IMAGE_NAME}:${TAG}"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.nexusnav"

echo "[push_ncloud] Registry:   ${REGISTRY}"
echo "[push_ncloud] Image:      ${FULL_TAG}"
echo "[push_ncloud] Dockerfile: ${DOCKERFILE}"
echo "[push_ncloud] Build root: ${REPO_ROOT}"
echo ""

# ── login ─────────────────────────────────────────────────────────────────────
if [[ -n "${NCLOUD_ACCESS_KEY:-}" && -n "${NCLOUD_SECRET_KEY:-}" ]]; then
    echo "[push_ncloud] Logging in with NCLOUD_ACCESS_KEY..."
    echo "${NCLOUD_SECRET_KEY}" | docker login "${REGISTRY}" \
        --username "${NCLOUD_ACCESS_KEY}" \
        --password-stdin
else
    echo "[push_ncloud] No NCLOUD_ACCESS_KEY found — attempting interactive login..."
    docker login "${REGISTRY}"
fi

# ── build ─────────────────────────────────────────────────────────────────────
echo ""
echo "[push_ncloud] Building image (this takes ~10–20 min for flash-attn compile)..."
docker build \
    --file "${DOCKERFILE}" \
    --tag "${FULL_TAG}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --cache-from "${FULL_TAG}" \
    "${REPO_ROOT}"

echo "[push_ncloud] Build complete: ${FULL_TAG}"

# ── also tag as latest if not already ────────────────────────────────────────
if [[ "${TAG}" != "latest" ]]; then
    LATEST_TAG="${REGISTRY}/${IMAGE_NAME}:latest"
    docker tag "${FULL_TAG}" "${LATEST_TAG}"
    echo "[push_ncloud] Also tagged as: ${LATEST_TAG}"
fi

# ── push ──────────────────────────────────────────────────────────────────────
echo ""
echo "[push_ncloud] Pushing ${FULL_TAG} ..."
docker push "${FULL_TAG}"

if [[ "${TAG}" != "latest" ]]; then
    echo "[push_ncloud] Pushing ${LATEST_TAG} ..."
    docker push "${LATEST_TAG}"
fi

echo ""
echo "[push_ncloud] Done. To update the PyTorchJob manifest:"
echo "  export REGISTRY=${REGISTRY}"
echo "  envsubst < scripts/cloud/nexusnav-pytorchjob.yaml | kubectl apply -f -"
