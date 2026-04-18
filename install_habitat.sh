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

export CMAKE_CUDA_ARCHITECTURES="90"  # RTX 5090: "110" / H200: "90"

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

echo ""
echo "========================================="
echo " 모든 설치가 완료되었습니다!"
echo "========================================="
