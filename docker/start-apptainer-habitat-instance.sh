#!/bin/bash
# Start a SEPARATE apptainer branch for Habitat.
#
# 왜 별도 분기인가:
#   기본 인스턴스(internnav)의 overlay 에는 Isaac Sim 5.1 + torch 2.7.0+cu128 +
#   InternNav(editable) 환경이 들어 있다. 여기에 habitat-sim/habitat-lab 을 설치하면
#   학습·평가 환경이 오염될 수 있다.
#   → sandbox base(37G, isaac-sim 포함)는 그대로 공유하고 **overlay 만 새로** 만들어서
#     python 패키지 집합만 분리한다. 37G 를 복사할 필요가 없다.
#
# 사용법 (호스트에서 실행):
#   bash docker/start-apptainer-habitat-instance.sh                    # internnav-habitat, GPU 0
#   bash docker/start-apptainer-habitat-instance.sh internnav-habitat 1
#   NAV_OVERLAY=/path/to.img bash docker/start-apptainer-habitat-instance.sh
#
# 셸 열기:  dexec internnav-habitat
# 중지:     drm internnav-habitat
#
# 주의: VLN-Challenge 의 `dstart` alias 는 인스턴스명을 `internnav|nav*` 로 dispatch 하므로
#       `dstart internnav-habitat` 은 거부된다. 기동은 반드시 이 스크립트로 한다.
#       (`dexec`/`dps`/`drm` 은 이름만 보므로 그대로 쓸 수 있다.)

set -e

INSTANCE_NAME="${1:-internnav-habitat}"
GPU_ID="${2:-0}"

# habitat 전용 overlay. 기본 인스턴스의 internnav_260723.img 와 반드시 달라야 한다.
OVERLAY="${NAV_OVERLAY:-${HOME}/workspace/sandbox/internnav_habitat_260820.img}"
OVERLAY_SIZE_MB="${NAV_OVERLAY_SIZE_MB:-20480}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/start-apptainer-instance.sh"

if [ ! -f "${BASE_SCRIPT}" ]; then
    echo "✗ 오류: ${BASE_SCRIPT} 가 없습니다."
    exit 1
fi

# =========================================================
# overlay 를 먼저 만든다 (중요)
#   start-apptainer-instance.sh 는 overlay 파일이 없으면 조용히 --writable-tmpfs 로
#   폴백한다. 그러면 habitat 설치 내용이 인스턴스 종료 시 전부 사라진다.
# =========================================================
if [ -f "${OVERLAY}" ]; then
    echo "[overlay] 기존 habitat overlay 사용: ${OVERLAY}"
else
    echo "[overlay] habitat overlay 생성 중: ${OVERLAY} (${OVERLAY_SIZE_MB} MB)"
    mkdir -p "$(dirname "${OVERLAY}")"
    apptainer overlay create --size "${OVERLAY_SIZE_MB}" "${OVERLAY}"
    echo "[overlay] ✓ 생성 완료"
fi

# 기본 인스턴스의 overlay 를 실수로 재사용하는 것을 막는다 (동시 사용 시 손상)
DEFAULT_OVERLAY="${HOME}/workspace/sandbox/internnav_260723.img"
if [ "$(readlink -f "${OVERLAY}")" = "$(readlink -f "${DEFAULT_OVERLAY}")" ]; then
    echo "✗ 오류: habitat 분기가 기본 overlay(${DEFAULT_OVERLAY})를 가리키고 있습니다."
    echo "        분기 목적이 사라지고 두 인스턴스가 같은 overlay 를 물면 손상됩니다."
    exit 1
fi

# =========================================================
# 인스턴스 안에서 "여기는 habitat 분기" 임을 알 수 있게 표시
#   apptainer 는 APPTAINERENV_<NAME> 을 컨테이너 안 <NAME> 으로 전달한다.
#   install_habitat_6000pro.sh 가 이 값을 보고 오설치를 막는다.
# =========================================================
export APPTAINERENV_NAV_BRANCH=habitat

echo "[branch] 인스턴스: ${INSTANCE_NAME} / overlay: ${OVERLAY} / GPU: ${GPU_ID}"
echo

NAV_OVERLAY="${OVERLAY}" exec bash "${BASE_SCRIPT}" "${INSTANCE_NAME}" "${GPU_ID}"
