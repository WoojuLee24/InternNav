#!/bin/bash
# Start the InternNav apptainer container as a long-running INSTANCE.
# Run this ONCE (like `docker run -d`). Then use `dexec` (or
# `apptainer exec instance://<name> bash`) to open as many shells as you want
# (like `docker exec -ti`).
#
#   dstart                 인스턴스 'internnav' 시작, 모든 GPU 노출
#   dstart internnav 2     물리 GPU 2번만 사용
#   dstart internnav 0,2   GPU 0, 2번 사용
#
# run-torch2.7.0-...-6000pro-apptainer.sh 와의 차이:
#   저 스크립트는 `apptainer exec` 1회용 셸이라 터미널마다 mount namespace 가 따로 생긴다.
#   이 스크립트는 `apptainer instance start` 로 namespace 를 하나만 만들고,
#   모든 dexec 세션이 거기 합류한다 → tmux/프로세스가 터미널 간에 자연스럽게 공유된다.

INSTANCE_NAME="${1:-internnav}"
# Optional physical GPU id(s), e.g. "2" or "0,2". Apptainer's --nv exposes ALL GPUs,
# so we restrict the instance via CUDA_VISIBLE_DEVICES (inherited by every dexec session).
GPU_ID="${2:-${CUDA_VISIBLE_DEVICES:-}}"

# =========================================================
# 설정 (run-...-6000pro-apptainer.sh 와 동일하게 유지할 것)
# =========================================================
SANDBOX="${NAV_SANDBOX:-${HOME}/workspace/sandbox/internnav}"
# Overlay is per-instance: 하나의 overlay 이미지를 두 개의 살아있는 인스턴스가
# 동시에 쓰면 손상된다. 별도 인스턴스를 띄우려면 NAV_OVERLAY=/path/to.img 로 분리할 것.
# (인스턴스 1개 + dexec 여러 개는 같은 namespace 라 안전하다.)
OVERLAY="${NAV_OVERLAY:-${HOME}/workspace/sandbox/internnav_260723.img}"
CACHE="${HOME}/docker/isaac-sim"
WORKSPACE="${HOME}/workspace"
DATASETS_HOST="${HOME}/workspace/data"
TRAIN_DATASET_HOST="${HOME}/workspace/data"
MP3D_HOST="${TRAIN_DATASET_HOST}/InternData-N1-v0.5-mini/scene_data/mp3d_pe"

# =========================================================
# 같은 이름의 인스턴스가 이미 있으면 중복 기동 금지
#   (같은 overlay 를 두 인스턴스가 물면 손상되므로)
# =========================================================
if apptainer instance list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "${INSTANCE_NAME}"; then
    echo "인스턴스 '${INSTANCE_NAME}' 가 이미 실행 중입니다."
    echo "  셸 열기:   dexec ${INSTANCE_NAME}"
    echo "  중지:      drm ${INSTANCE_NAME}"
    exit 0
fi

# =========================================================
# 캐시 디렉토리 생성
# =========================================================
mkdir -p \
    "${CACHE}/cache/kit" \
    "${CACHE}/cache/ov" \
    "${CACHE}/cache/pip" \
    "${CACHE}/cache/glcache" \
    "${CACHE}/cache/computecache" \
    "${CACHE}/logs" \
    "${CACHE}/data" \
    "${CACHE}/documents"

xhost +local:root 2>/dev/null || true

# =========================================================
# tmux 소켓: 인스턴스 전용
# =========================================================
# apptainer 는 호스트 /tmp 를 그대로 bind 하므로, 인스턴스 안에서 그냥 `tmux` 를 치면
# 기본 소켓 /tmp/tmux-<uid>/default 에 붙는다. 그 소켓의 서버가 예전 컨테이너 실행에서
# 남은 것이면, 그 서버가 띄우는 셸은 예전 mount namespace 를 상속해서 data/ 가 비어 보인다.
# → 인스턴스마다 전용 소켓 디렉토리를 주면, 이 인스턴스의 모든 dexec 세션이
#   같은 namespace 의 같은 tmux 서버를 공유하게 된다. (실제 소켓은
#   ${TMUX_TMPDIR}/tmux-<uid>/default 로 한 단계 더 들어간다 — 경로 길이 제한 주의)
TMUX_DIR_HOST="${WORKSPACE}/.tmux/instance-${INSTANCE_NAME}"
TMUX_DIR_CONT="/ws/.tmux/instance-${INSTANCE_NAME}"
# 인스턴스가 죽으면 그 안의 tmux 서버도 같이 죽으므로, 새로 뜰 때 잔여 소켓을 비운다.
rm -rf "${TMUX_DIR_HOST}"
mkdir -p "${TMUX_DIR_HOST}"

# =========================================================
# 선택적 바인드 (경로가 존재할 때만 추가)
# =========================================================
OPTIONAL_BINDS=""
[ -e "${DATASETS_HOST}" ]      && OPTIONAL_BINDS="${OPTIONAL_BINDS} -B ${DATASETS_HOST}:/datasets:ro"
[ -e "${TRAIN_DATASET_HOST}" ] && OPTIONAL_BINDS="${OPTIONAL_BINDS} -B ${TRAIN_DATASET_HOST}:/ws/src/InternNav/data:rw"
[ -e "${MP3D_HOST}" ]          && OPTIONAL_BINDS="${OPTIONAL_BINDS} -B ${MP3D_HOST}:/isaac-sim/Matterport3D/data/v1/scans:ro"
[ -e "${HOME}/.Xauthority" ]   && OPTIONAL_BINDS="${OPTIONAL_BINDS} -B ${HOME}/.Xauthority:/home/dnwn24/.Xauthority:ro"

# =========================================================
# overlay 선택
# =========================================================
if [ -f "${OVERLAY}" ]; then
    OVERLAY_OPT="--overlay ${OVERLAY}"
    echo "[overlay] 영구 overlay 사용: ${OVERLAY}"
    echo "[overlay] 주의: 이 overlay 를 다른 인스턴스/run 스크립트와 동시에 쓰지 말 것 (손상 위험)."
else
    OVERLAY_OPT="--writable-tmpfs"
    echo "[overlay] '${OVERLAY}' 없음 → 임시 tmpfs 사용 (인스턴스 중지 시 사라짐)."
    echo "          영구화하려면: apptainer overlay create --size 10240 \"${OVERLAY}\""
fi

# =========================================================
# GPU 선택 (선택적)
#   apptainer 에는 docker 의 --gpus device=N 에 해당하는 플래그가 없다.
#   --nv 는 모든 GPU 를 노출하므로 CUDA_VISIBLE_DEVICES 로 제한한다.
# =========================================================
GPU_ENV_OPT=""
if [ -n "${GPU_ID}" ]; then
    GPU_ENV_OPT="--env CUDA_VISIBLE_DEVICES=${GPU_ID}"
    echo "[gpu] CUDA_VISIBLE_DEVICES=${GPU_ID} (physical GPU id; nvidia-smi 기준)"
else
    echo "[gpu] GPU 미지정 → 모든 GPU 노출 (--nv)"
fi

# =========================================================
# 인스턴스 시작 (백그라운드)
# =========================================================
# shellcheck disable=SC2086
apptainer instance start \
    --nv \
    ${OVERLAY_OPT} \
    ${GPU_ENV_OPT} \
    --env ACCEPT_EULA=Y \
    --env PRIVACY_CONSENT=Y \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env DISPLAY="${DISPLAY}" \
    --env HOME=/home/dnwn24 \
    --env TMUX_TMPDIR="${TMUX_DIR_CONT}" \
    --env PS1='\[\033[01;32m\]\w\[\033[00m\]\$ ' \
    -B "${CACHE}/cache/kit:/isaac-sim/kit/cache:rw" \
    -B "${CACHE}/cache/ov:/home/dnwn24/.cache/ov:rw" \
    -B "${CACHE}/cache/pip:/home/dnwn24/.cache/pip:rw" \
    -B "${CACHE}/cache/glcache:/home/dnwn24/.cache/nvidia/GLCache:rw" \
    -B "${CACHE}/cache/computecache:/home/dnwn24/.nv/ComputeCache:rw" \
    -B "${CACHE}/logs:/home/dnwn24/.nvidia-omniverse/logs:rw" \
    -B "${CACHE}/data:/home/dnwn24/.local/share/ov/data:rw" \
    -B "${CACHE}/documents:/home/dnwn24/Documents:rw" \
    -B "${WORKSPACE}:/ws:rw" \
    ${OPTIONAL_BINDS} \
    "${SANDBOX}" \
    "${INSTANCE_NAME}" || exit 1

echo
echo "인스턴스 '${INSTANCE_NAME}' 시작됨. 현재 인스턴스 목록:"
apptainer instance list
echo
[ -n "${GPU_ID}" ] && echo "GPU 제한:                       CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "쉘 열기 (docker exec -ti 대응): dexec ${INSTANCE_NAME}"
echo "인스턴스 중지:                  drm ${INSTANCE_NAME}"
echo "tmux 소켓:                      ${TMUX_DIR_CONT}  → 안에서 그냥 'tmux new -s train'"
