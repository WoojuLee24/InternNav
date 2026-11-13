#!/bin/bash
# Habitat 설치 — RTX PRO 6000 Blackwell + Isaac Sim apptainer 분기 전용
#
# install_habitat_5090.sh 를 이 환경에 맞게 고친 것. 원본은 건드리지 않았다.
# 원본이 이 컨테이너에서 깨지는 지점:
#   1. `python` 이 PATH 에 없음                → PY 변수로 python.sh 사용
#   2. `pip` 이 /usr/bin/pip (system py3.12)   → PIP 변수로 python.sh -m pip 사용
#   3. INTERNNAV_DIR 이 한 단계 깊음           → BASE_DIR=/ws/src 로 조정
#   4. CMAKE_CUDA_ARCHITECTURES=90 (GPU 불일치) → torch 로 자동 감지 (Blackwell=120)
#   5. [0/3] 데이터 링크가 무조건 exit 1        → 이미 있으면 skip 하는 guard 로 변경
#   6. 빌드 에러를 2>/dev/null 로 삼킴          → 로그 파일로 tee
#
# 실행 (habitat 분기 인스턴스 안에서):
#   bash docker/start-apptainer-habitat-instance.sh   # ← 호스트에서 먼저
#   dexec internnav-habitat                           # ← 호스트에서 셸 열기
#   bash docker/install_habitat_6000pro.sh

set -e

# =========================================================
# 실행 환경 (전부 환경변수로 override 가능)
# =========================================================
PY="${PY:-/workspace/isaaclab/_isaac_sim/python.sh}"
PIP="${PIP:-$PY -m pip}"

BASE_DIR="${BASE_DIR:-/ws/src}"
INTERNNAV_DIR="${INTERNNAV_DIR:-/ws/src/InternNav}"
HABITATSIM_DIR="${HABITATSIM_DIR:-$BASE_DIR/habitat-sim}"
HABITATLAB_DIR="${HABITATLAB_DIR:-$BASE_DIR/habitat-lab}"

# ninja 는 PATH 에 없고 isaacsim python 의 bin 에만 있다 (pip 패키지 ninja)
export PATH="/workspace/isaaclab/_isaac_sim/kit/python/bin:$PATH"

# GPU arch 자동 감지. RTX PRO 6000 Blackwell → 120 (sm_120)
CUDA_ARCH="${CUDA_ARCH:-$($PY -c 'import torch;m,n=torch.cuda.get_device_capability(0);print(f"{m}{n}")' 2>/dev/null | tail -1)}"
CUDA_ARCH="${CUDA_ARCH:-120}"
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

# 128 core 지만 habitat-sim 빌드는 메모리를 많이 먹어 OOM 위험이 있다
export MAX_JOBS="${MAX_JOBS:-16}"

# =========================================================
# 안전장치: habitat 분기가 아닌 곳에 설치하는 것을 막는다
#   신호 2개로 판별한다.
#     1) NAV_BRANCH=habitat  — start-apptainer-habitat-instance.sh 가 APPTAINERENV_ 로 주입
#     2) TMUX_TMPDIR=/ws/.tmux/instance-<name> — 인스턴스마다 다르게 주입되므로
#        apptainer 버전에 따라 1)이 전달되지 않아도 인스턴스 이름을 알 수 있다
# =========================================================
DETECTED_INSTANCE="$(basename "${TMUX_TMPDIR:-}" 2>/dev/null | sed 's/^instance-//')"

IS_HABITAT_BRANCH=0
if [ "${NAV_BRANCH}" = "habitat" ]; then
    IS_HABITAT_BRANCH=1
elif case "${DETECTED_INSTANCE}" in *habitat*) true ;; *) false ;; esac; then
    IS_HABITAT_BRANCH=1
fi

if [ "${IS_HABITAT_BRANCH}" != "1" ] && [ "${FORCE}" != "1" ]; then
    echo "========================================="
    echo " ✗ 중단: 여기는 habitat 분기가 아닌 것 같습니다."
    echo "     NAV_BRANCH        = '${NAV_BRANCH:-<unset>}'"
    echo "     감지된 인스턴스   = '${DETECTED_INSTANCE:-<unknown>}'"
    echo ""
    echo "   기본 internnav 인스턴스에 habitat 을 설치하면 Isaac Sim / 학습 환경이"
    echo "   오염될 수 있습니다. 호스트에서 아래를 먼저 실행하세요:"
    echo ""
    echo "     bash docker/start-apptainer-habitat-instance.sh"
    echo "     dexec internnav-habitat"
    echo ""
    echo "   이미 분기 안이 확실하다면: FORCE=1 bash docker/install_habitat_6000pro.sh"
    echo "========================================="
    exit 1
fi

echo "========================================="
echo " Habitat 설치 시작"
echo " PY:            $PY"
echo " PIP:           $PIP"
echo " BASE_DIR:      $BASE_DIR"
echo " INTERNNAV_DIR: $INTERNNAV_DIR"
echo " CUDA_ARCH:     $CMAKE_CUDA_ARCHITECTURES (CMAKE_CUDA_ARCHITECTURES)"
echo " MAX_JOBS:      $MAX_JOBS"
echo " 분기:          NAV_BRANCH=${NAV_BRANCH:-<unset>} / instance=${DETECTED_INSTANCE:-<unknown>}${FORCE:+  (FORCE=$FORCE)}"
echo "========================================="

# ── 데이터 심볼릭 링크 ────────────────────────
echo ""
echo "[0/3] 데이터 경로 확인 중..."

DATA_LINK="$INTERNNAV_DIR/data"

if [ -e "$DATA_LINK" ]; then
    if [ -L "$DATA_LINK" ]; then
        echo "  → 이미 심볼릭 링크입니다: $DATA_LINK → $(readlink "$DATA_LINK")"
    else
        echo "  → 이미 실제 디렉토리입니다: $DATA_LINK (그대로 사용)"
    fi
elif [ -n "$DATA_SRC" ]; then
    if [ ! -d "$DATA_SRC" ]; then
        echo "  ✗ 오류: DATA_SRC 가 존재하지 않습니다: $DATA_SRC"
        exit 1
    fi
    ln -s "$DATA_SRC" "$DATA_LINK"
    echo "  ✓ 심볼릭 링크 생성: $DATA_LINK → $DATA_SRC"
else
    echo "  ⚠ $DATA_LINK 가 없습니다. 필요하면 DATA_SRC=/path/to/data 로 재실행하세요."
fi

# ── habitat-sim ──────────────────────────────
echo ""
echo "[1/3] habitat-sim 설치 중..."

if [ ! -d "$HABITATSIM_DIR" ]; then
    git clone https://github.com/facebookresearch/habitat-sim.git "$HABITATSIM_DIR"
else
    echo "  → habitat-sim 디렉토리가 이미 존재합니다. 클론을 건너뜁니다."
fi

cd "$HABITATSIM_DIR"
git checkout v0.3.3
git submodule update --init --recursive

BUILD_LOG="$HABITATSIM_DIR/build_install.log"
echo "  → 빌드 로그: $BUILD_LOG"

if ! $PIP install . \
        --no-deps \
        --no-build-isolation \
        --config-settings="cmake.args=-DHABITATPHYSICS_BUILD_EXTRA=ON" \
        2>&1 | tee "$BUILD_LOG"; then
    echo "  → pip install 실패. setup.py 경로로 재시도합니다."
    $PY setup.py install --headless --with-bullet 2>&1 | tee -a "$BUILD_LOG"
fi

$PIP uninstall llvmlite -y || true
$PIP install \
    simplejson \
    omegaconf \
    numba \
    "llvmlite>=0.47.0" \
    hydra-core \
    antlr4-python3-runtime==4.9.3 \
    "gym==0.23.1" \
    ifcfg \
    "webdataset==0.1.40" \
    "faster-fifo==1.5.2" \
    fastdtw \
    --no-deps

echo "  ✓ habitat-sim 설치 완료"

# ── habitat-lab ──────────────────────────────
echo ""
echo "[2/3] habitat-lab 설치 중..."

if [ ! -d "$HABITATLAB_DIR" ]; then
    git clone --branch main https://github.com/facebookresearch/habitat-lab.git "$HABITATLAB_DIR"
else
    echo "  → habitat-lab 디렉토리가 이미 존재합니다. 클론을 건너뜁니다."
fi

cd "$HABITATLAB_DIR"
$PIP install -e habitat-lab --no-deps
$PIP install -e habitat-baselines --no-deps

echo "  ✓ habitat-lab 설치 완료"

# ── InternNav ────────────────────────────────
echo ""
echo "[3/3] InternNav[habitat] 설치 중..."

if [ ! -d "$INTERNNAV_DIR" ]; then
    echo "  ✗ 오류: InternNav 디렉토리가 존재하지 않습니다: $INTERNNAV_DIR"
    exit 1
fi

cd "$INTERNNAV_DIR"
# --no-deps 필수: habitat_requirements.txt 는 accelerate==1.4.0 / flash_attn==2.7.4.post1 /
# diffusers==0.32.2 를 핀하는데, 이 환경은 accelerate 1.13.0 / flash_attn 2.8.3+cu128torch2.7 이다.
$PIP install -e ".[habitat]" --no-deps --no-build-isolation

echo "  ✓ InternNav 설치 완료"

# ── 설치 검증 ─────────────────────────────────
echo ""
echo "========================================="
echo " 설치 검증 시작"
echo "========================================="

PASS=0
FAIL=0

check() {
    local label="$1"
    local cmd="$2"
    printf "  %-45s" "$label"
    if output=$(eval "$cmd" 2>&1); then
        echo "✓  $output"
        PASS=$((PASS + 1))
    else
        echo "✗  FAILED"
        echo "     → $output"
        FAIL=$((FAIL + 1))
    fi
}

# 데이터 경로 확인
printf "  %-45s" "data 경로"
if [ -e "$DATA_LINK" ]; then
    echo "✓  $DATA_LINK"
    PASS=$((PASS + 1))
else
    echo "✗  FAILED (경로가 존재하지 않음)"
    FAIL=$((FAIL + 1))
fi

check "habitat_sim import" \
    "$PY -c \"import habitat_sim; print('v' + habitat_sim.__version__)\" 2>&1 | tail -1"

check "habitat import" \
    "$PY -c \"import habitat; print('v' + habitat.__version__)\" 2>&1 | tail -1"

check "hydra import" \
    "$PY -c \"import hydra; print('OK')\" 2>&1 | tail -1"

check "internnav.habitat_extensions.vln import" \
    "$PY -c \"import internnav.habitat_extensions.vln; print('OK')\" 2>&1 | tail -1"

check "Evaluator habitat_vln 등록" \
    "$PY -c \"
from internnav.evaluator import Evaluator
keys = list(Evaluator.evaluators.keys())
assert 'habitat_vln' in Evaluator.evaluators, 'habitat_vln not found in: ' + str(keys)
print('OK  등록된 evaluators: ' + str(keys))
\" 2>&1 | tail -1"

# 기존 환경 회귀 확인 (base 를 건드리지 않았는지)
check "기존 패키지 버전 유지" \
    "$PY -c \"
import torch, numpy, accelerate, transformers
print(f'torch={torch.__version__} numpy={numpy.__version__} accelerate={accelerate.__version__} transformers={transformers.__version__}')
\" 2>&1 | tail -1"

# overlay 잔여 용량
printf "  %-45s" "overlay 잔여 용량"
echo "✓  $(df -h / | tail -1 | awk '{print $4 " free / " $2}')"

# ── 최종 결과 ─────────────────────────────────
echo ""
echo "========================================="
echo " 검증 결과: ${PASS} 통과 / $((PASS + FAIL)) 항목"
if [ "$FAIL" -eq 0 ]; then
    echo " 모든 설치가 성공적으로 완료되었습니다!"
    echo ""
    echo " 다음: 실제 headless 렌더 확인"
    echo "   $PY docker/smoke_habitat_render.py"
else
    echo " ${FAIL}개 항목이 실패했습니다. 위 오류를 확인하세요."
    echo " habitat-sim 빌드 로그: $BUILD_LOG"
    exit 1
fi
echo "========================================="
