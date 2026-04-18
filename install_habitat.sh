#!/bin/bash
set -e

BASE_DIR="/home/irteam/git"
INTERNNAV_DIR="$BASE_DIR/InternNav"
HABITATSIM_DIR="$BASE_DIR/habitat-sim"
HABITATLAB_DIR="$BASE_DIR/habitat-lab"

echo "========================================="
echo " Habitat 설치 시작"
echo " BASE_DIR: $BASE_DIR"
echo "========================================="

# ── 데이터 심볼릭 링크 ────────────────────────
echo ""
echo "[0/3] 데이터 심볼릭 링크 설정 중..."

DATA_SRC="/home/irteam/data-vol1"
DATA_LINK="$INTERNNAV_DIR/data"

if [ ! -d "$DATA_SRC" ]; then
    echo "  ✗ 오류: 원본 데이터 경로가 존재하지 않습니다: $DATA_SRC"
    exit 1
fi

if [ -L "$DATA_LINK" ]; then
    echo "  → 심볼릭 링크가 이미 존재합니다: $DATA_LINK → $(readlink "$DATA_LINK")"
elif [ -e "$DATA_LINK" ]; then
    echo "  ✗ 오류: $DATA_LINK 가 이미 존재하지만 심볼릭 링크가 아닙니다."
    exit 1
else
    ln -s "$DATA_SRC" "$DATA_LINK"
    echo "  ✓ 심볼릭 링크 생성: $DATA_LINK → $DATA_SRC"
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

export CMAKE_CUDA_ARCHITECTURES="90"  # RTX 5090: "90" / H200: "100"

pip install . \
    --no-deps \
    --no-build-isolation \
    --config-settings="cmake.args=-DHABITATPHYSICS_BUILD_EXTRA=ON" \
    2>/dev/null || \
    python setup.py install --headless --with-bullet

pip uninstall llvmlite -y
pip install \
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
pip install -e habitat-lab --no-deps
pip install -e habitat-baselines --no-deps

echo "  ✓ habitat-lab 설치 완료"

# ── InternNav ────────────────────────────────
echo ""
echo "[3/3] InternNav[habitat] 설치 중..."

if [ ! -d "$INTERNNAV_DIR" ]; then
    echo "  ✗ 오류: InternNav 디렉토리가 존재하지 않습니다: $INTERNNAV_DIR"
    exit 1
fi

cd "$INTERNNAV_DIR"
pip install -e ".[habitat]" --no-deps --no-build-isolation

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

# 심볼릭 링크 확인
printf "  %-45s" "data 심볼릭 링크"
if [ -L "$DATA_LINK" ] && [ -e "$DATA_LINK" ]; then
    echo "✓  $DATA_LINK → $(readlink "$DATA_LINK")"
    PASS=$((PASS + 1))
else
    echo "✗  FAILED (링크가 없거나 대상이 존재하지 않음)"
    FAIL=$((FAIL + 1))
fi

# habitat_sim 확인
check "habitat_sim import" \
    "python -c \"import habitat_sim; print('v' + habitat_sim.__version__)\""

# habitat 확인
check "habitat import" \
    "python -c \"import habitat; print('v' + habitat.__version__)\""

# hydra 확인
check "hydra import" \
    "python -c \"import hydra; print('OK')\""

# internnav habitat_extensions 확인
check "internnav.habitat_extensions.vln import" \
    "python -c \"import internnav.habitat_extensions.vln; print('OK')\""

# 평가자 등록 확인
check "Evaluator habitat_vln 등록" \
    "python -c \"
from internnav.evaluator import Evaluator
keys = list(Evaluator.evaluators.keys())
assert 'habitat_vln' in Evaluator.evaluators, 'habitat_vln not found in: ' + str(keys)
print('OK  등록된 evaluators: ' + str(keys))
\""

# ── 최종 결과 ─────────────────────────────────
echo ""
echo "========================================="
echo " 검증 결과: ${PASS} 통과 / $((PASS + FAIL)) 항목"
if [ "$FAIL" -eq 0 ]; then
    echo " 모든 설치가 성공적으로 완료되었습니다!"
else
    echo " ${FAIL}개 항목이 실패했습니다. 위 오류를 확인하세요."
    exit 1
fi
echo "========================================="
