#!/usr/bin/env bash
# launch_train.sh — End-to-end NexusNav training launch on Naver Cloud ML Expert Platform
#
# Usage:
#   ./scripts/cloud/launch_train.sh [--registry <endpoint>] [--tag <tag>] [--dry-run] [--delete]
#
# Prerequisites:
#   - kubectl configured with ML Expert Platform kubeconfig
#   - Image already pushed: ./scripts/cloud/push_ncloud.sh <registry>
#   - NCR secret created: kubectl create secret docker-registry ncr-registry-secret ...
#   - DDN PVC created (this script creates it if absent)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_NAME="nexusnav-train"
NAMESPACE="default"
DRY_RUN=false
DELETE=false
REGISTRY="${REGISTRY:-}"  # can be set via env or --registry flag
TAG="${TAG:-latest}"

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY="$2"; shift 2 ;;
        --tag)      TAG="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --delete)   DELETE=true; shift ;;
        --namespace) NAMESPACE="$2"; shift 2 ;;
        -h|--help)
            grep "^#" "$0" | head -10 | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── sanity checks ─────────────────────────────────────────────────────────────
echo "[launch_train] Checking kubectl context..."
KUBE_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "NONE")
if [[ "${KUBE_CONTEXT}" == "NONE" ]]; then
    echo "ERROR: kubectl is not configured. Download kubeconfig from ML Expert Platform portal."
    exit 1
fi
echo "[launch_train] kubectl context: ${KUBE_CONTEXT}"
echo "[launch_train] namespace:       ${NAMESPACE}"

# Check PyTorchJob CRD available
if ! kubectl get crd pytorchjobs.kubeflow.org &>/dev/null; then
    echo "ERROR: PyTorchJob CRD not found. Ensure Kubeflow Training Operator is installed."
    exit 1
fi

# ── delete existing job ───────────────────────────────────────────────────────
if [[ "${DELETE}" == "true" ]]; then
    echo "[launch_train] Deleting existing job ${JOB_NAME}..."
    kubectl delete pytorchjob "${JOB_NAME}" -n "${NAMESPACE}" --ignore-not-found
    echo "[launch_train] Deleted."
    exit 0
fi

# ── create DDN PVC if not exists ──────────────────────────────────────────────
if ! kubectl get pvc nexusnav-ddn-pvc -n "${NAMESPACE}" &>/dev/null; then
    echo "[launch_train] Creating DDN PVC..."
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  [dry-run] kubectl apply -f ${SCRIPT_DIR}/nexusnav-pvc.yaml"
    else
        kubectl apply -f "${SCRIPT_DIR}/nexusnav-pvc.yaml" -n "${NAMESPACE}"
        echo "[launch_train] Waiting for PVC to bind (up to 120s)..."
        kubectl wait pvc/nexusnav-ddn-pvc \
            --for=condition=Bound \
            --timeout=120s \
            -n "${NAMESPACE}" || {
            echo "WARNING: PVC not yet bound. Check storage class availability."
        }
    fi
else
    echo "[launch_train] DDN PVC nexusnav-ddn-pvc already exists."
fi

# ── check NCR pull secret ─────────────────────────────────────────────────────
if ! kubectl get secret ncr-registry-secret -n "${NAMESPACE}" &>/dev/null; then
    echo ""
    echo "WARNING: Pull secret 'ncr-registry-secret' not found."
    echo "Create it with:"
    echo "  kubectl create secret docker-registry ncr-registry-secret \\"
    echo "    --docker-server=<registry>.kr.ncr.ntruss.com \\"
    echo "    --docker-username=\$NCLOUD_ACCESS_KEY \\"
    echo "    --docker-password=\$NCLOUD_SECRET_KEY \\"
    echo "    -n ${NAMESPACE}"
    echo ""
fi

# ── delete any existing job with same name ────────────────────────────────────
if kubectl get pytorchjob "${JOB_NAME}" -n "${NAMESPACE}" &>/dev/null; then
    echo "[launch_train] Found existing job '${JOB_NAME}'. Deleting before relaunch..."
    kubectl delete pytorchjob "${JOB_NAME}" -n "${NAMESPACE}"
    sleep 5
fi

# ── apply PyTorchJob ──────────────────────────────────────────────────────────
echo ""
echo "[launch_train] Applying PyTorchJob manifest..."
MANIFEST=$(REGISTRY="${REGISTRY}" TAG="${TAG}" envsubst < "${SCRIPT_DIR}/nexusnav-pytorchjob.yaml")

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] Would apply:"
    echo "${MANIFEST}"
    exit 0
fi

echo "${MANIFEST}" | kubectl apply -f - -n "${NAMESPACE}"

echo ""
echo "[launch_train] Job submitted. Waiting for pods to start (up to 5min)..."
sleep 10

# ── poll pod status ───────────────────────────────────────────────────────────
for i in $(seq 1 30); do
    PHASE=$(kubectl get pytorchjob "${JOB_NAME}" -n "${NAMESPACE}" \
        -o jsonpath='{.status.conditions[-1].type}' 2>/dev/null || echo "Pending")
    echo "[launch_train] Status: ${PHASE} (check ${i}/30)"
    if [[ "${PHASE}" == "Running" || "${PHASE}" == "Succeeded" ]]; then
        break
    elif [[ "${PHASE}" == "Failed" ]]; then
        echo "ERROR: Job failed. Check logs below."
        kubectl logs -l training.kubeflow.org/job-name="${JOB_NAME}" \
            -n "${NAMESPACE}" --tail=50 || true
        exit 1
    fi
    sleep 10
done

# ── stream master logs ────────────────────────────────────────────────────────
echo ""
echo "[launch_train] Streaming master logs (Ctrl-C to detach, job continues)..."
echo "  To re-attach: kubectl logs -f -l training.kubeflow.org/job-name=${JOB_NAME} -n ${NAMESPACE}"
echo ""
kubectl logs -f \
    -l "training.kubeflow.org/job-name=${JOB_NAME},training.kubeflow.org/replica-type=master" \
    -n "${NAMESPACE}" \
    --tail=100 || true
