#!/bin/bash
# 노은역 ABC·DAgger 용 apptainer 실행 래퍼 (RTX PRO 6000 / Blackwell).
#
#   noeun_apptainer.sh isaac   <명령...>    # Isaac 렌더 / 모델 서버
#   noeun_apptainer.sh habitat <명령...>    # 81 navmesh 서버 (CPU)
#
#   DRY=1 noeun_apptainer.sh isaac python3 x.py    # 실행하지 않고 명령만 출력
#   GPU=1 noeun_apptainer.sh isaac ...             # 사용할 GPU (기본 0)
#
# docker 와 다른 점:
#   - docker exec 가 없다. 서버(81, 모델 서버)는 이 스크립트를 nohup ... & 로 띄운다.
#   - 네트워크는 호스트와 같다 (포트 매핑 없음). 노드를 공유하면 포트가 겹칠 수 있으니
#     빈 포트를 골라 81 --port / 84·82 --expert_url / 82 --agent_port 에 같게 준다.
#   - 이미지가 읽기 전용이라 Isaac 캐시 자리를 호스트 폴더로 bind 한다.
set -euo pipefail

MODE="${1:-}"; shift || true
if [ -z "${MODE}" ] || { [ "${MODE}" != "isaac" ] && [ "${MODE}" != "habitat" ]; }; then
    echo "사용법: $(basename "$0") (isaac|habitat) <명령...>" >&2
    exit 2
fi
if [ "$#" -eq 0 ]; then
    echo "오류: 실행할 명령이 없습니다." >&2
    exit 2
fi

# =========================================================
# 경로 (환경변수로 덮어쓸 수 있다)
# =========================================================
NAV="${NAV:-/data/users/dnwn24/src/InternNav}"
UTOPIA="${UTOPIA:-/data/users/dnwn24/src/InternUtopia}"
DATA="${DATA:-/home/dnwn24/workspace/data}"
CKPT="${CKPT:-${NAV}/checkpoints}"
CACHE="${CACHE:-/home/dnwn24/workspace/apptainer_cache/noeun}"
SIF_DIR="${SIF_DIR:-/home/dnwn24/workspace/sif}"
# isaac 은 이 서버에서 **sandbox 디렉터리**다 (SIF 가 아니다). overlay 에 torch·InternNav 환경이 있다.
SIF_ISAAC="${SIF_ISAAC:-/home/dnwn24/workspace/sandbox/internnav}"
OVERLAY_ISAAC="${OVERLAY_ISAAC:-/home/dnwn24/workspace/sandbox/internnav_260723.img}"
SIF_HABITAT="${SIF_HABITAT:-${SIF_DIR}/internnav-habitat-6000pro.sif}"

GPU="${GPU:-0}"

# =========================================================
# TRITON_OVERRIDE_ARCH: 이미지 ENV 가 A5000 용 sm86 으로 고정돼 있어 덮어써야 한다.
#   compute_cap 에서 계산 (6000 Pro = 12.0 -> sm120)
# =========================================================
if [ -z "${TRITON_ARCH:-}" ]; then
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    if [ -n "${CC}" ]; then
        TRITON_ARCH="sm${CC//./}"
    else
        echo "경고: compute_cap 을 읽지 못했습니다. TRITON_ARCH 를 직접 주세요." >&2
        TRITON_ARCH="sm120"
    fi
fi

# =========================================================
# 중첩 bind 이므로 InternNav 안에 마운트 지점이 먼저 있어야 한다
# =========================================================
mkdir -p "${NAV}/data" "${NAV}/checkpoints"

case "${MODE}" in
    isaac)   SIF="${SIF_ISAAC}"   ;;
    habitat) SIF="${SIF_HABITAT}" ;;
esac

# isaac 은 **인스턴스**로 쓴다 (overlay 에 python 환경이 있고, 인스턴스가 그 overlay 를 쓰기로 잡는다).
#   docker/start-apptainer-instance.sh 로 띄운 'internnav' 인스턴스에 붙는다.
#   instance:// 는 이미 bind·env 가 잡혀 있으므로 아래 ARGS 를 쓰지 않고 바로 exec 한다.
if [ "${MODE}" = "isaac" ] && [ -n "${INST:-internnav}" ] && \
   apptainer instance list 2>/dev/null | awk 'NR>1{print $1}' | grep -qx "${INST:-internnav}"; then
    PRE="cd /ws/src/InternNav && export PYTHONPATH=/ws/src/InternNav CUDA_VISIBLE_DEVICES=${GPU} ISAAC_ACTIVE_GPU=${GPU}"
    PRE="${PRE} ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES TRITON_OVERRIDE_ARCH=${TRITON_ARCH}"
    if [ -n "${DRY:-}" ]; then
        printf 'apptainer exec instance://%s bash -c %q\n' "${INST:-internnav}" "${PRE} && $*"
        exit 0
    fi
    exec apptainer exec "instance://${INST:-internnav}" bash -c "${PRE} && $*"
fi

if [ ! -f "${SIF}" ] && [ ! -d "${SIF}" ]; then
    echo "✗ 이미지가 없습니다: ${SIF}" >&2
    echo "  docker-archive tar 를 받아 아래로 만드세요 (다시 빌드하지 말 것):" >&2
    echo "    apptainer build ${SIF} docker-archive://<tar>" >&2
    exit 1
fi

# =========================================================
# 공통 인자
# =========================================================
ARGS=(
    exec --nv --cleanenv --writable-tmpfs
    --pwd /ws/src/InternNav
    -B "${NAV}:/ws/src/InternNav"
    -B "${UTOPIA}:/ws/src/InternUtopia"
    -B "${DATA}:/ws/src/InternNav/data"
    -B "${CKPT}:/ws/src/InternNav/checkpoints"
    --env PYTHONPATH=/ws/src/InternNav
    --env CUDA_VISIBLE_DEVICES="${GPU}"
    --env ISAAC_ACTIVE_GPU="${GPU}"
    --env ACCEPT_EULA=Y
    --env PRIVACY_CONSENT=Y
    --env OMNI_KIT_ACCEPT_EULA=YES
    --env TRITON_OVERRIDE_ARCH="${TRITON_ARCH}"
)

# =========================================================
# isaac 전용: 읽기 전용 이미지라 Isaac 캐시 자리를 호스트 폴더로 돌린다
# =========================================================
if [ "${MODE}" = "isaac" ]; then
    mkdir -p "${CACHE}/home" "${CACHE}/kit_cache" "${CACHE}/kit_data" "${CACHE}/kit_logs"
    ARGS+=(
        --home "${CACHE}/home:/root"
        -B "${CACHE}/kit_cache:/isaac-sim/kit/cache"
        -B "${CACHE}/kit_data:/isaac-sim/kit/data"
        -B "${CACHE}/kit_logs:/isaac-sim/kit/logs"
    )

    # Vulkan ICD: 이미지의 VK_DRIVER_FILES 자리에 호스트 것을 넣는다. 없으면 Isaac 렌더 불가.
    ICD=""
    for c in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
        [ -f "$c" ] && { ICD="$c"; break; }
    done
    if [ -z "${ICD}" ]; then
        echo "✗ 호스트에 nvidia_icd.json 이 없습니다 → Isaac 렌더 불가." >&2
        exit 1
    fi
    ARGS+=(-B "${ICD}:/etc/vulkan/icd.d/nvidia_icd.json")

    EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
    [ -f "${EGL}" ] && ARGS+=(-B "${EGL}:${EGL}")
fi

ARGS+=("${SIF}" "$@")

if [ -n "${DRY:-}" ]; then
    printf 'apptainer'
    printf ' %q' "${ARGS[@]}"
    printf '\n'
    exit 0
fi

exec apptainer "${ARGS[@]}"
