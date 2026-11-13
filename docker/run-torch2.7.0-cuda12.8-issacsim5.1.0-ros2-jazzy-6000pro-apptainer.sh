#!/bin/bash
# Apptainer equivalent of run-torch2.7.0-cuda12.8-issacsim5.1.0-ros2-jazzy-5090.sh
#
# 주요 변환:
#   --gpus all        → --nv
#   --network=host    → Apptainer 기본값 (host networking)
#   --shm-size 32g    → /dev/shm 시스템 설정에 의존 (변경 불가, root 필요)
#   -v src:dst        → -B src:dst
#   -w /ws            → --pwd /ws
#   -e KEY=VAL        → --env KEY=VAL

# =========================================================
# 설정 — 환경에 맞게 수정
# =========================================================

# sandbox 경로 (apptainer build --sandbox 결과물)
SANDBOX="${HOME}/workspace/sandbox/internnav"

# 영구 overlay 이미지 (컨테이너 안에서 설치/변경한 내용을 여기에 영구 저장).
# 없으면 아래 로직이 임시 tmpfs로 폴백한다.
#   생성: apptainer overlay create --size 10240 "${OVERLAY}"
OVERLAY="${HOME}/workspace/sandbox/internnav_260723.img"

# Isaac Sim 캐시 (Docker run script와 동일 경로 유지)
CACHE="${HOME}/docker/isaac-sim"

# 워크스페이스 (Docker의 /home/universe/.../workspace → 로컬 workspace)
WORKSPACE="${HOME}/workspace"

# 데이터셋 경로 (없으면 바인드 생략)
DATASETS_HOST="${HOME}/workspace/data"
TRAIN_DATASET_HOST="${HOME}/workspace/data"
MP3D_HOST="${TRAIN_DATASET_HOST}/InternData-N1-v0.5-mini/scene_data/mp3d_pe"

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

# =========================================================
# X11 forwarding
# =========================================================
xhost +local:root 2>/dev/null || true

# =========================================================
# 선택적 바인드 구성 (경로가 존재할 때만 추가)
# =========================================================
OPTIONAL_BINDS=""

if [ -e "${DATASETS_HOST}" ]; then
    OPTIONAL_BINDS="${OPTIONAL_BINDS} -B ${DATASETS_HOST}:/datasets:ro"
fi

if [ -e "${TRAIN_DATASET_HOST}" ]; then
    OPTIONAL_BINDS="${OPTIONAL_BINDS} -B ${TRAIN_DATASET_HOST}:/ws/src/InternNav/data:rw"
fi

if [ -e "${MP3D_HOST}" ]; then
    OPTIONAL_BINDS="${OPTIONAL_BINDS} -B ${MP3D_HOST}:/isaac-sim/Matterport3D/data/v1/scans:ro"
fi

if [ -e "${HOME}/.Xauthority" ]; then
    OPTIONAL_BINDS="${OPTIONAL_BINDS} -B ${HOME}/.Xauthority:/home/dnwn24/.Xauthority:ro"
fi

# =========================================================
# tmux 소켓 격리
# =========================================================
# apptainer는 exec 마다 별도 mount namespace를 만들고, /ws/src/InternNav/data 같은
# bind mount는 그 namespace 안에서만 보인다. 반면 tmux 기본 소켓(/tmp/tmux-<uid>/default)은
# 호스트 /tmp에 있어서 컨테이너 종료 후에도 살아남는다.
# 그래서 새 컨테이너에서 그냥 `tmux`를 치면 새 서버가 뜨는 게 아니라 예전 실행의 서버에
# 붙어버리고, 그 서버가 띄우는 셸은 예전 namespace를 상속해서 data/ 가 비어 보인다.
# → 실행마다 별도 소켓 디렉토리를 주어 항상 현재 namespace에서 새 서버가 뜨도록 한다.
# 실제 소켓은 tmux가 ${TMUX_TMPDIR}/tmux-<uid>/default 로 한 단계 더 파고 만든다.
# (unix socket 경로 길이 제한이 있으므로 run id는 짧게 유지)
TMUX_RUN_ID="$(date +%y%m%d_%H%M%S)"
TMUX_DIR_HOST="${WORKSPACE}/.tmux/${TMUX_RUN_ID}"
TMUX_DIR_CONT="/ws/.tmux/${TMUX_RUN_ID}"
mkdir -p "${TMUX_DIR_HOST}"

# 서버가 죽어 소켓이 사라진 예전 실행 디렉토리 정리 + 살아있는 것만 안내
PREV_TMUX_DIRS=""
for d in "${WORKSPACE}"/.tmux/*/; do
    [ -d "${d}" ] || continue
    [ "$(basename "${d}")" = "${TMUX_RUN_ID}" ] && continue
    if [ -n "$(find "${d}" -type s 2>/dev/null)" ]; then
        PREV_TMUX_DIRS="${PREV_TMUX_DIRS} $(basename "${d}")"
    else
        rm -rf "${d}"
    fi
done

echo "[tmux] 이번 실행 소켓: ${TMUX_DIR_CONT}  → 컨테이너 안에서 그냥 'tmux new -s train'"
if [ -n "${PREV_TMUX_DIRS}" ]; then
    echo "[tmux] 이전 실행의 살아있는 세션에 붙으려면 (그 실행의 mount를 그대로 물려받음):"
    for d in ${PREV_TMUX_DIRS}; do
        echo "         TMUX_TMPDIR=/ws/.tmux/${d} tmux a"
    done
fi

# =========================================================
# overlay 선택: 영구 overlay 파일이 있으면 사용, 없으면 임시 tmpfs로 폴백
# =========================================================
if [ -f "${OVERLAY}" ]; then
    OVERLAY_OPT="--overlay ${OVERLAY}"
    echo "[overlay] 영구 overlay 사용: ${OVERLAY} (변경분 유지됨)"
else
    OVERLAY_OPT="--writable-tmpfs"
    echo "[overlay] '${OVERLAY}' 없음 → 임시 tmpfs 사용 (변경분 종료 시 사라짐)."
    echo "          영구화하려면 호스트에서: apptainer overlay create --size 10240 \"${OVERLAY}\""
fi

# =========================================================
# 실행
# =========================================================
# shellcheck disable=SC2086
apptainer exec \
    --nv \
    ${OVERLAY_OPT} \
    --pwd /ws \
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
    bash

#   아래 두 줄은 Docker run script에서도 주석처리된 바인드입니다.
#   -B "${WORKSPACE}/src/InternNav:/workspace/isaaclab/InternNav:rw" \
#   -B "${WORKSPACE}/src/InternUtopia:/workspace/InternUtopia:rw" \
