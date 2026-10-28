# InternVLA-N1 Deployment Workflow (5090)

> **outdated (08-12 Q4 경로, 사용 안 함)** — 현재 재현 절차는 [command.md](../command.md).

RTX 5090 (x86, sm_120) 환경에서 InternVLA-N1-lite-DualVLN 배포판을 실행/검증하는 표준 가이드.
Thor(aarch64) 원본 자산은 그대로 보존, 5090 자산은 별도 이름공간에 두고 스크립트가 `uname -m`으로 자동 분기.

---

## 최종 상태 (5090 검증 결과)

| 항목 | 결과 |
|---|---|
| S1 체크포인트 재현 vs 원본 | ✅ bit-exact 일치 |
| S1 uncontended | ✅ 9.1 ms/infer (110 Hz) |
| S1 async in-episode | ✅ 28.1 Hz (TRT sm_120 FP16) |
| S2 async in-episode | ✅ 5.57 Hz (Q4_K_M-latent GGUF) |
| Dual e2e accuracy (배포 vs baseline S1) | ✅ 15/15 action match, 5.98× speedup |
| 배포 매니저 test suite | ✅ 18/18 PASS |
| Lifecycle 검증 | ✅ 24/24 PASS |

성능/정확도 상세 비교는 [COMPARISON.md](COMPARISON_q4_0812.md) 참조.

> **속도 항목 주의**: 위 수치는 원래 venv 스택(torch 2.13.0+cu130)에서 측정된 값이다. Step 1이 Isaac Sim과 동일한 `torch 2.7.0+cu128`로 고정되도록 바뀌었으므로(사유는 Step 1 참고), 재현 시 속도는 **TRT 경로 기준 약 5~10% 낮게** 나온다 (동일 조건 실측: S1 TRT 18.8 ms → 19.6~20.7 ms, PyTorch fp32 95.7 ms → 110.3 ms). 정확도 항목(bit-exact 재현 / 15/15 / 18/18 / 24/24)은 영향이 없다 — 두 torch에서 S1 TRT 궤적은 `max|Δ| = 0.0`으로 비트 동일하다.

---

## 환경 요구사항

검증 환경의 **실측값**이다.

| 항목 | 실측 | 요구 |
|---|---|---|
| GPU | RTX 5090 · compute cap 12.0 (sm_120) · 32,607 MiB · driver 590.48.01 | 다른 sm_XX도 가능 (엔진이 실행 시 자동 빌드) |
| CPU | Intel Core Ultra 9 285K · 24 스레드 · avx/avx2/f16c/fma 있음, **avx512f 없음** | AVX2 x86_64. Step 2가 `GGML_AVX512=OFF`로 빌드 |
| RAM | 125 GiB | ≥ 16 GB |
| OS | Ubuntu 24.04.2 LTS | — |
| Python | 시스템 3.12.3 / **Isaac Sim 3.11.13** | venv는 Isaac 3.11로 만든다 (Step 1 참고) |
| CUDA | nvcc 12.8 (V12.8.93) | 12.x 이상 |
| VRAM | S2 GGUF 러너 **7,186 MiB** + Isaac Sim headless **6,184 MiB** + S1 lite/TRT **약 800 MiB** ≈ **14.2 GB** | ≥ 16 GB (Step 11 eval 기준). Step 5~8만 하면 S2+S1로 약 8 GB |

**Disk** — 실측 (`du -sh`):

| | 크기 | 비고 |
|---|---|---|
| 사전 준비물 | | 재현 시작 전에 이미 있어야 함 |
| `checkpoints/InternVLA-N1-DualVLN` | 16 GB | Step 4의 입력 (원본 4 shards) |
| `quantization_kit/gguf` | 5.7 GB | S2 배포 GGUF 2개 (재생성 불가 — 아래 참고) |
| `quantization_kit/onnx` + `data` | 0.5 GB | TRT 재빌드 소스 + vln-ce 샘플 336개 |
| 재현이 만들어내는 것 | | |
| Step 1 `convenv_x86` | **12 GB** | 이 중 4.3 GB가 `tensorrt_libs` |
| Step 2 `llama.cpp/build_x86` | 199 MB | |
| Step 3 `trt_engines` | 218 MB | sm120 엔진 5개 |
| Step 4 `...-lite-DualVLN-reproduce` | 180 MB | 검증용, 사후 삭제 가능 |
| Step 10 eval 번들 | **5.9 GB** | GGUF 실복사본 |
| **재현에 필요한 순증** | **≈ 19 GB** | 사전 준비물 22 GB는 별도 |

Step 11 eval을 반복하면 `logs/lite/*/fps_*.mp4`(`vis_debug`)가 계속 쌓인다 — 3 에피소드 실행 몇 번에 3.5 GB가 찼다. 반복할 거면 config에서 `vis_debug: False`로 끄는 편이 낫다. Isaac Sim 셰이더 캐시(`/root/.cache/ov`)도 4.6 GB까지 자란다.

---

## 1. 환경 사전 빌드

각 스텝은 **목적 / 명령 / 기대 결과 / 참고** 4-필드 통일 format.

**명령** 블록은 맨 앞의 `cd`를 포함해 통째로 복사해야 하고, **기대 결과·검증** 블록은 어느 디렉토리에서든 그대로 붙여넣을 수 있도록 절대경로로 적었다.

### Step 1. 시스템/Python 환경 준비

**목적**: 5090 실행에 필요한 시스템 라이브러리 + Python venv 구성.

**명령**:
```bash
# 시스템 패키지 (root 권한 필요)
sudo apt update
sudo apt install -y libzmq3-dev

# Python venv (x86 전용) — 반드시 Isaac Sim 인터프리터로 생성할 것 (아래 참고)
cd /ws/src/InternNav/vln-deploy
/isaac-sim/kit/python/bin/python3 -m venv quantization_kit/convenv_x86
quantization_kit/convenv_x86/bin/pip install --upgrade pip

# torch/torchvision은 Isaac Sim과 동일 버전으로 고정. cu128 인덱스 필수 (아래 참고)
quantization_kit/convenv_x86/bin/pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.0' 'torchvision==0.22.0'
quantization_kit/convenv_x86/bin/pip install safetensors diffusers transformers einops accelerate 'numpy==1.26.4' pillow timm opencv-python-headless onnx onnxscript matplotlib pyzmq msgpack 'tensorrt==10.16.1.11'
```

**기대 결과**:
```bash
cd /ws/src/InternNav/vln-deploy && quantization_kit/convenv_x86/bin/python -c "import torch, tensorrt, diffusers, transformers; print('OK', torch.cuda.is_available())"
# → OK True
```

**참고**:
- `libzmq3-dev`: Step 2 llama.cpp `--zmq` 옵션에 필요.
- **venv는 시스템 `python3`가 아니라 Isaac Sim 인터프리터(`/isaac-sim/kit/python/bin/python3`, 3.11.13)로 만든다.** 시스템 python3.12로 만들면 cp312 확장모듈이 Isaac의 3.11에서 import되지 않아 **Step 9(Isaac eval 연동)가 성립하지 않는다.** `pyvenv.cfg`가 `executable = /isaac-sim/kit/python/bin/python3.11`인지로 확인할 수 있다. (`python3.12-venv`가 설치돼 있지 않으면 시스템 `python3 -m venv`는 `ensurepip` 부재로 아예 실패한다 — 그 경우 apt로 우회하지 말고 위 명령을 쓸 것.)
- `tensorrt==10.16.1.11`: 정확한 버전 고정이 필요하다. `.engine` plan 파일은 빌드한 TRT 버전에 락되므로, Step 3에서 만든 엔진을 Step 9의 Isaac eval이 그대로 재사용하려면 두 곳의 TRT 버전이 같아야 한다. (상한만 거는 `tensorrt<11`은 설치 시점에 따라 patch 버전이 달라진다. TRT 11부터는 `BuilderFlag.FP16`이 제거되어 `trt_s1.py` 자체가 호환 불가.)
- **`torch==2.7.0` / `torchvision==0.22.0` / `numpy==1.26.4`는 Isaac Sim 번들과 일부러 맞춘 값이다.** 최종 목표가 Step 9의 Isaac eval인데 거기서는 Isaac의 torch 2.7이 쓰이므로, Step 5~8의 배포 검증도 같은 스택을 재야 의미가 있다. 정확도 대가는 없다 — 동일 입력으로 S1 TRT 궤적을 비교하면 torch 2.13 ↔ 2.7이 `max|Δ| = 0.0`(비트 동일)이고, PyTorch fp32 경로도 상대오차 1.3e-06으로 fp32→fp16 양자화 오차(3.8e-03)보다 3000배 작다. 속도만 5~10% 낮아진다 (최종 상태 표의 주의 참조).
- **cu128 인덱스가 필수다.** PyPI 기본 `torch==2.7.0` 휠은 cu126이라 `arch_list`에 `sm_120`이 없어 5090에서 동작하지 않는다. Isaac 번들도 `torch 2.7.0+cu128`이다.
- 원래 문서 수치를 낸 스택은 `torch 2.13.0+cu130`, `transformers 5.15.0`, `diffusers 0.39.0`, `numpy 2.4.6`, `safetensors 0.8.0`이었다. 속도 수치까지 문서 그대로 재현하는 게 목적이라면 이 조합으로 되돌린다 (단 그 경우 Step 9의 Isaac eval과는 다른 스택이 된다).
- Isaac의 torch를 venv에서 재사용하는 방법(`.pth`로 `exts/omni.isaac.ml_archive/pip_prebundle` 추가)은 쓰지 않는다. torch 자체는 import되지만 Isaac이 의존성을 7개 `pip_prebundle` 디렉토리에 흩어놔서(`packaging`은 `core_archive`, `PIL`은 `omni.kit.pip_archive`) venv 안에서 `setup_python_env.sh`를 복제해야 한다.
- Thor(aarch64)는 별도 `convenv_aarch64/` 사용. 스크립트가 `uname -m`으로 자동 선택.

---

### Step 2. llama.cpp 5090용 빌드

**목적**: 5090(sm_120) native + Arrow Lake CPU (AVX-512 없음) + libzmq 링크된 `llama-s2-runner` 생성.

**명령**:
```bash
cd /ws/src/InternNav/vln-deploy/quantization_kit/llama.cpp
cmake -B build_x86 -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_F16C=ON -DGGML_FMA=ON \
    -DGGML_AVX512=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build_x86 --target llama-s2-runner llama-quantize -j$(nproc)
```

**기대 결과**:
```bash
cd /ws/src/InternNav/vln-deploy/quantization_kit/llama.cpp && ldd build_x86/bin/llama-s2-runner | grep libzmq   # → libzmq.so.5 => /usr/lib/x86_64-linux-gnu/libzmq.so.5
cd /ws/src/InternNav/vln-deploy/quantization_kit/llama.cpp && cuobjdump --list-elf build_x86/bin/libggml-cuda.so | grep sm_120  # → sm_120a.cubin
```

**참고**:
- `GGML_NATIVE=OFF`: 빌드 머신 CPU 자동감지 비활성화 (SIGILL 방지).
- `GGML_AVX512=OFF`: Arrow Lake는 AVX-512 미지원.
- `sm_120`: RTX 5090 native. 다른 GPU는 해당 compute capability로 변경.
- Thor 빌드(`build/`, sm_110)는 그대로 보존됨.

---

### Step 3. TRT 엔진 사전 빌드 (옵션)

**목적**: S1 파이프라인을 native TRT FP16으로 가속. 첫 실행 시 자동 빌드되지만 미리 만들어두면 async 첫 실행 지연 제거.

**명령**:
```bash
cd /ws/src/InternNav/vln-deploy
quantization_kit/convenv_x86/bin/python -c "
import sys; sys.path.insert(0, '.')
from quantization_kit.trt_s1 import (_engine_for, _sm_tag,
    DENOISE_SHAPES, RGB_SHAPES, RGB_SHAPES_B1, MEMENC_SHAPES, RESAMPLER_SHAPES)
sm = _sm_tag()
for tag, onnx, shp in [
    (f's1_denoise_native_fp16_{sm}', 's1_denoise_step.onnx', DENOISE_SHAPES),
    (f'rgb_dinov2_native_fp16_{sm}', 'rgb_model_dinov2.onnx', RGB_SHAPES),
    (f'rgb_dinov2_b1_native_fp16_{sm}', 'rgb_model_dinov2.onnx', RGB_SHAPES_B1),
    (f'memenc_native_fp16_{sm}', 'memory_encoder.onnx', MEMENC_SHAPES),
    (f'resampler_native_fp16_{sm}', 'rgb_resampler.onnx', RESAMPLER_SHAPES),
]:
    _engine_for(tag, onnx, shp, 'quantization_kit/onnx', 'quantization_kit/trt_engines')
"
```

**기대 결과**:
```bash
ls /ws/src/InternNav/vln-deploy/quantization_kit/trt_engines/*_sm120.engine | wc -l   # → 5
```

**참고**:
- `_sm_tag()`가 `torch.cuda.get_device_capability()`로 감지 → 5090은 `sm120`, 4090은 `sm89` 등.
- Thor 원본 `*_sm110.engine` 파일은 그대로 유지 (legacy 이름).
- 이 스텝을 skip해도 async/dual 첫 실행 시 자동 빌드됨 (수십 초 지연 발생).

---

## 2. 모델 추출 및 검증
### Step 1. S1 체크포인트 추출
**목적**: 원본 `InternVLA-N1-DualVLN`(16 GB)에서 S1-only subset (`InternVLA-N1-lite-DualVLN`)을 재추출해 원본과 bit-exact 일치 확인.
**명령**:
```bash
cd /ws/src/InternNav/vln-deploy
bash deploy/tools/copy_sidecar.sh checkpoints/InternVLA-N1-DualVLN <DST>
quantization_kit/convenv_x86/bin/python deploy/tools/extract_s1_weights.py checkpoints/InternVLA-N1-DualVLN <DST>/model-s1only.safetensors
```

```
  sidecar는 가중치를 뺀 나머지 10여 개 파일(약 5 MB)이고, 그중 `config.json`은 **필수**다 — `S1LiteModel.__init__`이 읽는다. `*.safetensors`와 `model.safetensors.index.json`은 반드시 빠져야 한다 (인덱스가 남으면 없는 shard를 가리켜 로더가 깨진다).
```

---


### Step 1-1. S1 체크포인트 재현 + 검증

**목적**: 원본 `InternVLA-N1-DualVLN`(16 GB)에서 S1-only subset (`InternVLA-N1-lite-DualVLN`)을 재추출해 원본과 bit-exact 일치 확인.

**명령**:
```bash
cd /ws/src/InternNav/vln-deploy
bash verify/reproduce_lite.sh
```

**기대 결과**:
```
sidecar diffs : 0
metadata equal: True
tensor count  : orig=608  repro=608
key diff      : 0
value diff    : 0
OVERALL: PASS ✅
```

**참고**:
- 원본 checkpoint(`checkpoints/InternVLA-N1-DualVLN/`) 존재 필요.
- 결과물은 `checkpoints/InternVLA-N1-lite-DualVLN-reproduce/`에 저장 (재현 검증용).
- 실제 배포는 원본 `checkpoints/InternVLA-N1-lite-DualVLN/`를 사용.
- S1 체크포인트 만들기는 **sidecar 복사 + 텐서 추출** 두 단계다. 임의 경로에 만들려면 두 도구를 직접 부르면 된다 (`reproduce_lite.sh`는 여기에 원본 대조를 붙인 것):
```bash
cd /ws/src/InternNav/vln-deploy && bash deploy/tools/copy_sidecar.sh checkpoints/InternVLA-N1-DualVLN <DST> && quantization_kit/convenv_x86/bin/python deploy/tools/extract_s1_weights.py checkpoints/InternVLA-N1-DualVLN <DST>/model-s1only.safetensors
```
  sidecar는 가중치를 뺀 나머지 10여 개 파일(약 5 MB)이고, 그중 `config.json`은 **필수**다 — `S1LiteModel.__init__`이 읽는다. `*.safetensors`와 `model.safetensors.index.json`은 반드시 빠져야 한다 (인덱스가 남으면 없는 shard를 가리켜 로더가 깨진다).

---

### Step 2. S2 단독 실행 검증

**목적**: S2 GGUF (Q4_K_M-latent)가 5090에서 정상 로드·추론.

**명령**:
```bash
cd /ws/src/InternNav/vln-deploy
./quantization_kit/run.sh s2
```

**기대 결과**: `OK`로 시작하는 3줄 + `exit=0`. 3번째 줄은 **76,460자짜리 base64 한 줄**이라 터미널을 뒤덮는다 (정상 동작).

```
OK                     ← RESET 응답
OK	←←←←               ← TEXT 응답: discrete action (공백 없이 붙어서 출력됨)
OK	Lwq8vjlmGT9Q...  ← LATENT 응답: base64 76,460자 (약 75 KB)
```

블롭 없이 앞부분만 보려면:
```bash
cd /ws/src/InternNav/vln-deploy && ./quantization_kit/run.sh s2 2>&1 | cut -c1-60
```

**검증**: 3번째 줄을 디코드하면 정확히 57,344 바이트 = `(1, 4, 3584)` float32여야 한다.
```bash
cd /ws/src/InternNav/vln-deploy && ./quantization_kit/run.sh s2 2>&1 | tail -1 | cut -f2 | base64 -d | wc -c    # → 57344
```

**참고**:
- 같은 이미지·프롬프트면 latent가 결정론적으로 재현된다. 두 번 돌려 마지막 수십 자가 같으면 정상.
- 이 번들에 `text-f16-latent.gguf`가 없어도 `Q4_K_M-latent.gguf`로 자동 fallback (`cmd_s2` [fix]).
- 기본 이미지: `quantization_kit/dbg/test448.png` (448×448 RGB smoke test).
- `cmd_s2`는 `2>/dev/null`로 stderr를 통째로 버린다. 출력이 3줄이 아니거나 비어 있으면 아래로 실제 에러를 확인할 것:
```bash
printf 'RESET\nTEXT\tquantization_kit/dbg/test448.png\t<__media__>Where should you go next? Please output the next waypoint coordinates in the image.\nLATENT\nQUIT\n' | quantization_kit/llama.cpp/build_x86/bin/llama-s2-runner -m quantization_kit/gguf/InternVLA-S2-text-Q4_K_M-latent.gguf --mmproj quantization_kit/gguf/InternVLA-S2-mmproj-f16.gguf 2>&1 | tail -40
```

---

### Step 3. Dual (S2+S1 동기) 검증 — 실 데이터

**목적**: 실제 vln-ce 데이터로 S2 → S1 파이프라인이 의미있는 navigation 결정을 반환.

**명령**:
```bash
cd /ws/src/InternNav/vln-deploy
# 한 프레임 dual
./quantization_kit/run.sh dual --sample quantization_kit/data/vln-ce/samples/sample_00000

# 여러 에피소드 replay
./quantization_kit/run.sh replay --n 2
```

**기대 결과**:
- `[S2 stepN] text='...' pixel=[y, x] latent=(1, 4, 3584)`
- `[S1] traj -> discrete actions idx[:4] = [1, 1, 1, 1]` (또는 유사한 action 시퀀스)
- `replay`: 각 step마다 pixel 목표 + action 출력, 정상 종료

**참고**:
- vln-ce sample 336개는 `quantization_kit/data/vln-ce/samples/`에 위치 (외부 데이터 폴더로 별도 준비).
- S1 성능(28 Hz)은 TRT 엔진(Step 3) 필요. TRT 없으면 `--no_trt`로 fallback (~4 Hz).

---

### Step 4. Async 파이프라인 검증

**목적**: S1/S2를 별개 프로세스(ZMQ)로 병렬 실행. Overlap으로 S1이 S2 추론 중에도 계속 tick.

**명령**:
```bash
cd /ws/src/InternNav/vln-deploy
./quantization_kit/run.sh async --seconds 8 --s2-mps-pct 0
```

**기대 결과**:
```
=== async control summary ===
  S1: ~230 infers @ ~28 Hz  (~18 ms/infer)
  S2: ~30 fired / ~29 done @ ~5.6 Hz  (~123 ms/infer)
  exit=0
```

**참고**:
- 실 데이터 사용: `--frames-dir quantization_kit/data/vln-ce/samples/sample_00000` 로 이미지 폴더 지정.
- `--s2-mps-pct 0`: MPS 데몬 없음 경고 억제 (5090에서는 MPS 불필요).
- 포트 5601–5610 사용. 크래시 시: `./quantization_kit/run.sh async-clean`.
- libzmq 없는 러너는 자동으로 pipe mode fallback (`cmd_async` [fix]).

---

### Step 5. 배포 매니저 lifecycle + test suite

**목적**: 배포 매니저(`./run.sh`)가 start/stop/status/test/demo 전 lifecycle을 정상 관리.

**명령**:
```bash
cd /ws/src/InternNav/vln-deploy

# 전체 test suite (protocol + s1 + s2 + e2e)
./run.sh test --keep-up

# lifecycle 자동 검증 (10 시나리오)
./verify/verify_lifecycle.sh

# Dual e2e 정확도 (배포 vs baseline S1)
./quantization_kit/convenv_x86/bin/python verify/verify_dual_accuracy.py --n 15
```

**기대 결과**:
- `test --keep-up` → `[manager] integration test: ALL PASS (suites: protocol, s1, s2, e2e)` (18/18)
- `verify_lifecycle.sh` → `24/24 PASS ✅`
- `verify_dual_accuracy.py` → `ACTION AGREEMENT: 15/15 (100.0%)`, `SPEEDUP: ~6×`

**참고**:
- `test --keep-up`은 첫 실행 시 S2 데몬 spawn까지 수십 초 대기.
- 각 검증 스크립트 개별 실행 가능 (`verify/` 아래 자족).
- `run.sh clean`은 시스템 `fuser` 명령 필요 (없으면 부분 실패, `stop`으로 대체 가능).

---

### Step 6. Isaac Sim eval 연동 (공유 프로세스)

**목적**: Isaac Sim eval을 기존처럼 `python.sh`로 그대로 돌리면서 S1 TRT 가속만 얹는다. Isaac 인터프리터·패키지는 전혀 건드리지 않고, venv에서 `tensorrt`만 빌려 쓴다.

**명령**:
```bash
cd /ws/src/InternNav/vln-deploy
SP=$PWD/quantization_kit/convenv_x86/lib/python3.11/site-packages
mkdir -p quantization_kit/trt_py311
for p in tensorrt tensorrt_bindings tensorrt_libs; do ln -sf $SP/$p quantization_kit/trt_py311/; done
export PYTHONPATH=$PWD/quantization_kit/trt_py311:$PYTHONPATH

/isaac-sim/python.sh -c "import torch, numpy, tensorrt, internutopia, isaacsim, pinocchio; print(torch.__version__, numpy.__version__, tensorrt.__version__)"
```

**기대 결과**:
```
2.7.0+cu128 1.26.4 10.16.1.11
```
- `internutopia` / `isaaclab` / `pinocchio` / `isaacsim` import 정상 (Isaac 스택 무손상)
- S1 추론: 110.3 ms (PyTorch fp32) → 19.6 ms (TRT fp16), 5.6×
- 궤적 정확도: venv 스택 TRT 결과와 `max|Δ| = 0.0` (비트 동일). fp32→fp16 양자화 오차 rel 3.81e-03 대비 torch 2.13 vs 2.7 차이는 0

**참고**:
- **venv python을 직접 실행하거나 `PYTHONEXE`로 넘기면 안 된다.** venv는 `include-system-site-packages = false`라 Isaac site-packages가 site 디렉토리로 등록되지 않고, 그 안의 `.pth` 10개가 실행되지 않는다. `__editable__.internutopia-*.pth` / `__editable__.isaaclab-*.pth` / `cmeel.pth`가 주입하던 `internutopia`·`isaaclab`·`pinocchio` import가 전부 깨진다.
- **venv site-packages 전체를 PYTHONPATH에 얹어도 안 된다.** torch 2.13 / numpy 2.4.6이 Isaac 것(2.7 / 1.26.4)을 가려 sim 확장이 깨진다. 위처럼 tensorrt 3개 디렉토리만 심링크할 것.
- Isaac 순정에 없는 패키지는 `timm`과 `tensorrt` 둘뿐이고, S1 lite 경로는 timm을 쓰지 않는다 (`s1_inference.py:68`의 DepthAnything 전용). Isaac에서는 sim의 GT depth를 쓰면 추가 설치 0건.
- 엔진은 Step 3에서 만든 `*_sm120.engine`을 그대로 재사용 (양쪽 다 TRT 10.16.1.11이라 재빌드 불필요). `.engine` plan은 TRT 버전에 락되므로 Step 1의 tensorrt 버전과 반드시 일치해야 한다.
- `runner.py` 연동: `machine == "h1"` 경로가 이미 `H1_PYTHON = /workspace/isaaclab/_isaac_sim/python.sh`를 쓰므로 인터프리터는 그대로 두고, subprocess env에 `env["PYTHONPATH"] = "<repo>/quantization_kit/trt_py311:" + env.get("PYTHONPATH", "")` 한 줄만 추가한다.
- SimulationApp을 띄운 상태에서도 궤적 `max|Δ| = 0.0`, 추가 VRAM 약 0.8 GB (sim 포함 총 6.9 GB), 경합 페널티 약 1%.

---

### Step 10. eval 번들 생성 (`--model-path` 하나로 S1+S2)

**목적**: S1 lite·S2 GGUF·runner를 체크포인트 디렉토리 하나에 모아, Isaac Sim eval을 `--model-path` 하나만 주고 돌릴 수 있게 한다. 세 자산은 타입이 달라 `quantization_kit` 파이프라인에서 서로 다른 곳에 떨어진다 — S1은 `checkpoints/<name>/`(HF 디렉토리), S2는 `quantization_kit/gguf/`(GGUF), runner는 `build_x86/bin/`(바이너리+`libggml*.so`).

**명령**:
```bash
cd /ws/src/InternNav/vln-deploy
./deploy/tools/make_eval_bundle.sh                     # 기본: checkpoints/InternVLA-N1-DualVLN-lite
./deploy/tools/make_eval_bundle.sh --verify-only       # 복사 없이 기존 번들만 점검
```
```
deploy/tools/make_eval_bundle.sh [DST] [--s1 DIR] [--quick] [--verify-only]
  DST            번들 디렉토리 (기본 checkpoints/InternVLA-N1-DualVLN-lite)
  --s1 DIR       System-1 lite 체크포인트 소스 (기본 checkpoints/InternVLA-N1-lite-DualVLN)
  --quick        복사한 6 GB의 md5 재확인 생략 (크기만 비교)
  --verify-only  아무것도 복사하지 않고 기존 번들만 검증
```

**기대 결과**:
```
==> [1/4] System-1 lite checkpoint
==> [2/4] System-2 GGUF (about 5.7 GB)
==> [3/4] llama.cpp runner + shared libs from build_x86
==> [4/4] verify
  OK    System-1: 1 safetensors
  OK    config.json (S1LiteModel reads it)
  OK    S2 InternVLA-S2-text-Q4_K_M-latent.gguf
  OK    S2 InternVLA-S2-mmproj-f16.gguf
  OK    llama-s2-runner
  OK    libggml*.so (12)
  OK    md5 ... (3개)
  OK    no symlink escapes the bundle
  OK    runner resolves all libs from bin/
  OK    BUNDLE_MANIFEST.txt written

bundle OK: .../checkpoints/InternVLA-N1-DualVLN-lite  (5.9G)
```
빈 디렉토리에서 약 8초, 재실행 시 전송 0바이트(멱등).

**결과 구조** (5.9 GB):
```
checkpoints/InternVLA-N1-DualVLN-lite/
├── model-s1only.safetensors        180 MB   S1 lite
├── config.json, tokenizer ...               S1LiteModel이 config.json을 읽는다
├── gguf/                           5.7 GB   S2 양자화 (Q4_K_M-latent + mmproj-f16)
├── bin/                             71 MB   llama-s2-runner + libggml/libllama/libmtmd
└── BUNDLE_MANIFEST.txt                      빌드 시각·호스트·소스 경로·md5
```

**참고**:
- **전부 실복사이고 심링크를 쓰지 않는다.** 심링크로 공유하면 나중에 다른 양자화 세트를 받을 때 기존 번들을 덮어쓸 경로가 생긴다. 번들마다 자기 파일을 소유하면 여러 세트가 공존해도 충돌하지 않는다. (하드링크도 같은 이유로 부적합 — `cp`가 제자리 덮어쓰기를 하면 공유 inode라 다른 번들까지 바뀐다.)
- 검증은 소스 대비 md5 대조 + **번들 밖을 가리키는 심링크 0개** + `LD_LIBRARY_PATH=<번들>/bin`만으로 runner의 라이브러리가 전부 해결되는지를 본다. `bin/` 내부의 `lib*.so -> lib*.so.0 -> lib*.so.0.N` 버전 링크는 디렉토리 안에서 닫히므로 자족성을 깨지 않는다 (`rsync -a`로 보존).
- `bin/`은 **아키텍처 종속**이다 (`uname -m`으로 `build`/`build_x86` 자동 선택). Thor로 번들을 옮기면 그쪽에서 이 스크립트를 다시 돌려 `bin/`을 교체해야 한다. `gguf/`와 S1 safetensors는 아키텍처 독립이라 그대로 유효하다.
- 기본 S1 소스(`InternVLA-N1-lite-DualVLN`)와 기본 대상(`InternVLA-N1-DualVLN-lite`)은 이름이 비슷한 별개 디렉토리다(내용은 동일한 중복본). 혼동을 피하려면 `--s1`으로 명시할 것.
- Step 2(runner 빌드)와 Step 4(S1 lite 추출)가 선행되어야 한다. 없으면 어느 단계를 먼저 돌려야 하는지 알려주고 종료한다.

---

### Step 11. Isaac Sim eval 실행 (양자화 스택)

**목적**: 번들을 InternNav의 Isaac Sim eval 파이프라인에 물려 실제로 돌린다.

**InternNav 쪽 배선** (이미 적용됨 — 새 환경에서 재현할 때 필요한 목록):

| 파일 | 변경 |
|---|---|
| `internnav/agent/internvla_n1_agent_llamacpp.py` | `_resolve_bundle_assets()` 추가. `runner_bin`/`model_gguf`/`mmproj_gguf`/`lib_dir`가 config에 없으면 `<model_path>/gguf/`·`bin/`에서 유도 (명시 키 우선 = 하위 호환) |
| `scripts/eval/configs/h1_internvla_n1_async_lite_cfg.py` | 신규. `model_name='internvla_n1_llamacpp'`, `resize 448`, `use_trt_s1=True`, `s1_steps=5`. env/task/dataset은 원본 config와 동일하게 두어 A/B 가능 |
| `internnav/configs/evaluator/vln_default_config.py` | `get_config()`의 model_name 분기에 `'internvla_n1_llamacpp'` 추가. 없으면 `model_settings` 미할당으로 `UnboundLocalError`. 미등록 이름용 `else: raise ValueError` 도 추가 |
| `scripts/train_eval/qwenvl_train/runner.py` | `TRT_PY311` 상수 + `run_eval()` env에 PYTHONPATH 주입 (Step 9). 디렉토리 없으면 건너뛰므로 비양자화 실행 무영향 |
| 두 eval config | `EVAL_MAX_EPISODES` / `EVAL_MAX_NEW_TOKENS` / `VLN_TIMING` 환경변수 릴레이. 미설정 시 기존 동작 그대로 |

**명령**:
```bash
cd /ws/src/InternNav && python scripts/train_eval/qwenvl_train/runner.py --config scripts/eval/configs/h1_internvla_n1_async_lite_cfg.py --machine h1 --no-train --model-path /ws/src/InternNav/checkpoints/InternVLA-N1-DualVLN-lite 2>&1 | tee run_lite.log
```

**기대 결과**: 로그에 에피소드 완주 기록과 결과 JSON.
```
[lite] loaded 608 S1 tensors  | missing=0 unexpected=0
[3/3][step_index:196] finish: [trajectory_id:7192_1807][duration:34.75 s][step_count:196][fps:5.64][result:success]
-> <model-path>/logs_stop/result_h1.json
```

**참고**:
- **결과는 `logs_none`이 아니라 `logs_stop`에 떨어진다.** `runner.py:384`가 `logs_<flash_collision|none>`로 이름을 짓는데, 이 값은 eval_cfg가 아니라 **config의 `PARAMS`**에서 온다 (우리 config의 `flash_collision=None`과 무관).
- **재실행 전 반드시 lmdb를 지운다**: `rm -rf <model-path>/logs_stop/lmdb`. 평가기는 lmdb에 이미 있는 에피소드를 resume-skip하고 `result_h1.json`은 lmdb 전체를 집계하므로, 안 지우면 두 번째 실행이 아무것도 돌지 않고 이전 숫자가 그대로 나온다.
- `--debugpy eval`을 붙이면 `vln_distributed_evaluator.py:333`에서 **2 에피소드 후 강제 종료**된다. 전체 실행에는 빼야 한다.
- 사전 점검 (Isaac Sim 부팅 없이):
```bash
cd /ws/src/InternNav && PYTHONPATH=/ws/src/InternNav/vln-deploy/quantization_kit/trt_py311 /isaac-sim/python.sh -c "import importlib; importlib.import_module('scripts.eval.configs.h1_internvla_n1_async_lite_cfg'); from internnav.agent import Agent; import tensorrt, quantization_kit.s1_inference; print('agents:', [k for k in Agent.agents if 'llamacpp' in k], '| tensorrt', tensorrt.__version__)"
```

---

### Step 12. 원본 대비 정확도·속도 비교

**목적**: 같은 에피소드에서 양자화 스택과 원본(7B in-process)의 SR/SPL과 속도를 비교한다.

**명령** — 한 번에:
```bash
cd /ws/src/InternNav && ./vln-deploy/verify/compare_eval.sh -n 10
```
```
verify/compare_eval.sh [-n N] [-t TOKENS] [-v lite|orig|both] [--native-tokens]
                       [--no-timing] [--keep-lmdb] [--out DIR] [--python P]
                       [--lite-config P] [--lite-model P] [--orig-config P] [--orig-model P]
                       [--extra "ARGS"] [--dry-run]
```
lmdb 정리 → orig 실행 → lite 실행 → 리포트까지 순차로 한다 (병렬 금지: GPU 경합).

**명령** — `runner.py`를 직접 돌린 경우:
```bash
rm -rf /ws/src/InternNav/checkpoints/InternVLA-N1-DualVLN/logs_stop/lmdb
```
```bash
cd /ws/src/InternNav && EVAL_MAX_EPISODES=10 EVAL_MAX_NEW_TOKENS=128 VLN_TIMING=/tmp/orig_timing.csv python scripts/train_eval/qwenvl_train/runner.py --config scripts/eval/configs/h1_internvla_n1_async_cfg.py --machine h1 --no-train --model-path /ws/src/InternNav/checkpoints/InternVLA-N1-DualVLN 2>&1 | tee /tmp/orig_run.log
```
(lite도 동일하게, config/model-path/VLN_TIMING 경로만 바꿔서) 그 다음:
```bash
cd /ws/src/InternNav && ./vln-deploy/verify/compare_report.py --out /tmp --orig-log /tmp/orig_run.log --lite-log /tmp/lite_run.log --orig-logdir logs_stop --lite-logdir logs_stop
```

**기대 결과**: 정확도표 + 속도 3종.
```
    metric         orig       lite   delta (lite - orig)
    Count             3          3   +0.0000 same
    SR           0.3333     0.3333   +0.0000 same
    SPL          0.3333     0.3260   -0.0073 worse
    NE           6.3630     4.8949   -1.4681 better

  per-episode (from the run log; excludes Isaac boot + scene load)
    trajectory     orig steps   orig s  orig fps  lite steps   lite s  lite fps
    7192_1809             277    64.42      4.30         246    49.55      4.96
    orig  total: 3 ep, 658 steps / 129.3s  ->  5.09 steps/s
    lite  total: 3 ep, 662 steps / 131.2s  ->  5.05 steps/s
    -> lite runs the sim loop at 0.992x the orig step rate
```

**공정성 — 세 가지가 맞아야 한다**:
1. **같은 에피소드**: `EVAL_MAX_EPISODES`를 두 실행에 동일하게. 리포트가 `trajectory` 열이 어긋나거나 `Count`가 다르면 경고한다.
2. **같은 생성 길이**: `EVAL_MAX_NEW_TOKENS=128`. 안 맞추면 orig 1024 / lite 128이라 양자화 쪽에 유리하다. 리포트가 `max_tok` 열로 보여주고 다르면 경고한다.
3. **깨끗한 lmdb**: 위 Step 11 참고. `compare_eval.sh`가 자동으로 지운다 (`--keep-lmdb`로 끌 수 있음).

**산출물**:

| 경로 | 내용 |
|---|---|
| `docs/COMPARISON.md` 섹션 5 | 사람이 읽는 리포트. `<!-- BEGIN eval-comparison -->` ~ `<!-- END -->` 사이만 갱신되고 바깥은 건드리지 않는다. `--md PATH`로 대상 변경, `--no-md`로 끄기 |
| `verify/compare_out/history.jsonl` | 실행마다 1줄 append. 섹션 5-3의 이력 표가 여기서 렌더된다 |
| `verify/compare_out/{orig,lite}_run.log` | runner.py stdout. **`finish:` 줄(에피소드별 duration/steps/fps)이 여기에만 있다** — 평가기의 파일 로그에는 없다 |
| `verify/compare_out/{orig,lite}_timing.csv` | `VLN_TIMING` per-call 계측 |
| `verify/compare_out/runs.tsv` | variant/episodes/max_tok/wall_s/exit |
| `<model-path>/logs_stop/result_h1.json` | 평가기가 쓰는 정확도 원본 |
| `logs/lite/{original,llamacpp}/fps_*.mp4` | `vis_debug` 영상. **실행한 디렉토리 기준 상대경로**다 |

**참고**:
- 이전 실행의 `result_h1.json` / `*_run.log` / `*_timing.csv`는 **지우지 않고 `.prev`로 밀어둔다.** 지우도록 했더니 실행을 Ctrl-C로 중단했을 때 새 결과도 안 생겨 비교 대상이 통째로 사라지는 사고가 있었다.
- 중단된 실행은 로그가 잘린 채 남는다. 리포트가 한쪽에 에피소드가 0건이면 "재실행하라"고 경고한다.

**참고**:
- **`python`을 스크립트 안에서 쓰면 안 된다.** `.bashrc` alias라 비대화형 셸에서 `command not found`(exit 127)가 된다. `compare_eval.sh`는 `/workspace/isaaclab/_isaac_sim/python.sh`를 기본으로 쓰고 `--python`으로 교체 가능하다.
- **속도는 세 층위로 봐야 한다.** wall clock(부팅·씬 로딩 포함), per-episode `steps/s`(로그의 `fps` — 시뮬 루프만), `VLN_TIMING`의 per-call(모델만). 3 에피소드 예비 측정에서는 `steps/s`가 5.09 vs 5.05로 **차이가 거의 없었다** — Isaac Sim의 렌더링·물리가 병목이라 S1이 110 ms → 20 ms가 되어도 스텝 시간이 거의 안 줄어든다. 양자화의 실익은 eval 속도가 아니라 VRAM(7B in-process 제거)과 온디바이스 배포다.
- `VLN_TIMING` 훅(`verify/eval_timing.py`)은 `runner.py`와 eval 서브프로세스 **양쪽**에 설치된다 (`runner.py:569`가 PARAMS를 읽으려 config를 자기 프로세스에서도 import하기 때문). 나중에 끝나는 부모가 빈 파일로 덮어쓰지 않도록 이벤트가 0건이면 쓰지 않는다.
- 에피소드는 10개 이상. 3개면 SR이 0/3·1/3·2/3·3/3 네 값뿐이라 정확도 비교가 성립하지 않는다.

---

## Thor / 5090 아키텍처 자동 분기

모든 스크립트가 `uname -m` (bash) 또는 `platform.machine()` (Python)으로 아키텍처를 감지:

| 자산 | Thor (aarch64) | 5090 (x86) |
|---|---|---|
| Python venv | `quantization_kit/convenv_aarch64/` | `quantization_kit/convenv_x86/` |
| llama.cpp | `llama.cpp/build/` (sm_110) | `llama.cpp/build_x86/` (sm_120a) |
| S1 TRT 엔진 | `trt_engines/*_sm110.engine` (원본) | `trt_engines/*_sm{cc}.engine` (실행 시 자동 생성) |
| S1 lite ckpt | 동일 (아키텍처 독립) | 동일 |
| S2 GGUF | 동일 (아키텍처 독립) | 동일 |
| TensorRT | apt `python3-libnvinfer` | pip `'tensorrt==10.16.1.11'` |

**분기 지점** (모두 `# [fix]` 주석):
- `quantization_kit/run.sh:21` — PY
- `quantization_kit/run.sh:56-63` — cmd_s2 GGUF fallback
- `quantization_kit/run.sh:68-74` — cmd_async pipe-mode auto-detect
- `deploy/common/env.sh:7` — PY
- `deploy/common/config.py:20` — CONVENV_PY
- `quantization_kit/trt_s1.py:84-113` — `_sm_tag()` + 엔진 태그 동적화
- `quantization_kit/async_zmq/run_async_demo.py:25` — PY
- `quantization_kit/run_dual_demo.py:37` — f16 → Q4 fallback
- `quantization_kit/run_episode_replay.py:101` — 동일
- `quantization_kit/verify_rgb_cache.py:48-52` — 아키텍처 + f16 → Q4 fallback

---

## 파일 구조

```
vln-deploy/
├── run.sh                              # 최상위 배포 매니저 wrapper
├── checkpoints/
│   ├── InternVLA-N1-DualVLN/           # 원본 16 GB (Qwen backbone 포함, 4 shards)
│   ├── InternVLA-N1-lite-DualVLN/      # 배포용 S1-only 0.18 GB
│   ├── InternVLA-N1-lite-DualVLN-reproduce/  # Step 4 재현본
│   ├── InternVLA-N1-DualVLN-lite/      # Step 10 eval 번들 5.9 GB (S1 + gguf/ + bin/)
│   └── depth_anything_v2_metric_hypersim_vits.pth
├── quantization_kit/
│   ├── run.sh                          # s2 / dual / replay / async / camera / prompt
│   ├── convenv_aarch64/                # Thor용 venv
│   ├── convenv_x86/                    # 5090용 venv
│   ├── gguf/                           # S2 GGUF (mmproj-f16 + Q4_K_M-latent)
│   ├── trt_engines/                    # S1 TRT 엔진 (*_sm110 = Thor, *_sm120 = 5090)
│   ├── trt_py311/                      # Step 9: Isaac python용 tensorrt 심링크 오버레이
│   ├── onnx/                           # S1 ONNX (TRT 재빌드 소스)
│   ├── data/vln-ce/samples/            # 336 실제 샘플 (외부 데이터)
│   ├── trt_s1.py                       # _sm_tag() 자동 감지
│   ├── s1_inference.py                 # S1LiteModel
│   └── llama.cpp/
│       ├── build/                      # aarch64 (sm_110, Thor)
│       └── build_x86/                  # x86 (sm_120a, 5090, libzmq 링크)
├── deploy/
│   ├── manager.py                      # 배포 매니저 (start/stop/status/test/demo/prompt)
│   ├── common/
│   │   ├── config.py                   # CONVENV_PY 자동 분기
│   │   └── env.sh                      # PY/BUILD 자동 분기
│   ├── tools/copy_sidecar.sh           # Step 4: 가중치 뺀 sidecar만 복사
│   ├── tools/extract_s1_weights.py     # S1 lite 추출 (Step 4)
│   ├── tools/make_eval_bundle.sh       # Step 10: S1+S2+runner를 한 디렉토리로
│   ├── s1_module/                      # S1 서비스 (async)
│   ├── s2_module/                      # S2 서비스 (async)
│   ├── camera_module/                  # V4L2 카메라 (V4L2 없어서 미검증)
│   └── viewer_module/                  # 시각화
├── verify/
│   ├── reproduce_lite.sh               # Step 4
│   ├── verify_lifecycle.sh             # Step 8 (24 서브테스트)
│   ├── verify_dual_accuracy.py         # Step 8 (배포 vs baseline S1)
│   ├── compare_eval.sh                 # Step 12: orig vs lite Isaac eval A/B
│   ├── eval_timing.py                  # Step 12: VLN_TIMING per-call 계측 (기본 off)
│   └── compare_report.py               # Step 12: 정확도·속도 표
└── docs/
    ├── WORKFLOW.md                     # 이 문서
    └── COMPARISON.md                   # 원본 vs 배포 비교 (크기/속도/정확도)
```

---

## 알려진 제약

| 이슈 | 상태 |
|---|---|
| `run.sh setup` (S2 GGUF 재양자화) | ❌ 실행 불가 — `quantization_kit/tools/{strip_to_qwen,gguf_add_latent_tokens}.py` 이 번들에 없음. 원본 개발 트리에만. 배포용 GGUF 2개는 이미 포함되어 있어 실사용 문제 없음 |
| `run.sh bench` | ❌ `quantization_kit/bench/` 폴더 없음 |
| `run.sh present` | ❌ `presentation/` 폴더 없음 |
| `run.sh eval` (Habitat SR/SPL) | ❌ `script/` 폴더 + habitat 스택 필요 |
| 실시간 카메라 (`run.sh camera`) | ⚠ V4L2 장치(`/dev/video*`) 필요 (이 환경에서 미검증) |
| `run.sh clean` | ⚠ 시스템 `fuser` 필요. 없으면 부분 실패, `run.sh stop`으로 대체 |
| S2 원본 safetensors vs GGUF Q4 정확도 재측정 | ⚠ transformers 5.15와 원본 코드(4.51 기준) API 불일치. 원본 트리 검증값(97.9% action match)은 [COMPARISON.md](COMPARISON_q4_0812.md) 참조 |

---

## 관련 문서

- [COMPARISON.md](COMPARISON_q4_0812.md) — 원본 vs 배포판 크기/속도/정확도 비교
- `BUNDLE.md` — 배포 번들 개요, 원본 개발 트리 경로
- `deploy/README.md` — 3-모듈 아키텍처 (S2/S1/manager) 및 ZMQ 프로토콜
- `quantization_kit/RUN.md` — 원본(Thor 기준) 양자화 파이프라인 가이드
- `quantization_kit/trt_engines/README.md` — TRT 엔진 파일명 규칙
