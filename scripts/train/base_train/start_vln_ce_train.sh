#!/usr/bin/env bash
# Launch VLN-CE mini training for a single model in the Habitat container.
# Usage: ./start_vln_ce_train.sh --model <navdp|dualsystem|chrononav|nexusnav>

set -euo pipefail

MODEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 --model <navdp|dualsystem|chrononav|nexusnav>"
    exit 1
fi

case "$MODEL" in
    navdp)      CFG="scripts/train/base_train/configs/navdp_vln_ce_mini.py" ;;
    dualsystem) CFG="scripts/train/base_train/configs/dualsystem_vln_ce_mini.py" ;;
    chrononav)  CFG="scripts/train/base_train/configs/chrononav_vln_ce_mini.py" ;;
    nexusnav)   CFG="scripts/train/base_train/configs/nexusnav_vln_ce_mini.py" ;;
    *) echo "Unknown model: $MODEL. Choose navdp|dualsystem|chrononav|nexusnav"; exit 1 ;;
esac

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONTAINER="vln_ce_train_${MODEL}"
LOG_DIR="/mnt/nvme0/vln_habitat/logs"
LOG_FILE="${LOG_DIR}/vln_ce_train_${MODEL}_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR" || true

PYPATH="/workspace/InternNav-dev:/workspace/InternNav:/workspace/InternNav/src/diffusion-policy:/workspace/ChronoNav:/workspace/InternNav/third_party/habitat-sim/src_python:/workspace/InternNav/third_party/habitat-sim/build/cp311-cp311-linux_x86_64/RelWithDebInfo/lib:/workspace/InternNav/third_party/habitat-sim/build/cp311-cp311-linux_x86_64/deps/magnum-bindings/src/python"

echo "Launching VLN-CE training: model=$MODEL container=$CONTAINER log=$LOG_FILE"

docker run -d --name "$CONTAINER" --gpus device=0 -v /home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev:/workspace/InternNav-dev:rw -v /home/kemal/VLNav/VLNav/workspaces/model/ChronoNav:/workspace/ChronoNav:rw -v /home/kemal/VLNav/VLNav/checkpoints:/checkpoints:ro -v /mnt/nvme0/vln_habitat/habitat_data:/workspace/InternNav/data:rw -v /mnt/nvme0/vln_habitat/logs:/workspace/InternNav/logs:rw -v /home/kemal/VLNav/VLNav/workspaces/model/InternNav:/workspace/InternNav:ro vlnav/habitat-eval:rebuilt bash -c "cd /workspace/InternNav && PYTHONPATH=${PYPATH} python /workspace/InternNav-dev/scripts/train/base_train/train.py --config /workspace/InternNav-dev/${CFG} 2>&1 | tee /workspace/InternNav/logs/vln_ce_train_${MODEL}_${TIMESTAMP}.log"

echo "Container '$CONTAINER' started. Tail logs: docker logs -f $CONTAINER"
echo "Or: tail -f $LOG_FILE"
