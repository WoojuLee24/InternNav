# InternVLA-N1 배포판 명령 모음 (5090)

> **outdated (08-12 Q4 경로, 사용 안 함)** — 현재 재현 절차는 [command.md](../command.md).

[WORKFLOW.md](WORKFLOW_q4_0812.md)에서 **실행 가능한 명령만** 순서대로 추출한 문서. 목적·기대결과·설계 배경은 WORKFLOW.md 참조.

`[완료]` = 2026-08-12 기준 이 환경에 이미 있음 (재실행 불필요). 섹션 1은 환경당 한 번, 섹션 2·3은 체크포인트마다 매번 돌린다.

---

## 0. 전제

- 원본 체크포인트 `checkpoints/InternVLA-N1-DualVLN` (16 GB, 4 shards) 이 있어야 한다.
- **맨몸 `python` / `pip` 금지** — `.bashrc` alias라 스크립트·비대화형 셸에서 `command not found`(exit 127). 항상 절대경로를 쓴다.
  - 배포 스택 (venv) → `/ws/src/InternNav/vln-deploy/quantization_kit/convenv_x86/bin/python`
  - Isaac Sim → `/isaac-sim/python.sh` 또는 `/workspace/isaaclab/_isaac_sim/python.sh`
- 모든 명령은 맨 앞 `cd`를 포함해 통째로 복사한다.

### 파일 구성
- 받은 파일: S2 양자화 GGUF (vln-deploy/quantization_kit/gguf/) , S1 ONNX (vln-deploy/quantization_kit/onnx/), S1 lite (checkpoints/InternVLA-N1-lite-DualVLN)
- 빌드할 파일: S1 TRT (vln-deploy/quantization_kit/trt_engines/)
- 최종 파일: checkpoints/InternVLA-N1-lite2-DualVLN

→ **다른 원본 체크포인트로 바꾸면** TRT 엔진이 옛 가중치로 돈다 (에러 없이 조용히). 그때는 eval config에서 `use_trt_s1: False`로 끌 것 (S1 20 ms → 110 ms, 정확도는 fp32라 오히려 높음). 아래 절차는 `$SRC`가 이 GGUF/ONNX를 만든 것과 **같은 원본**임을 전제한다.

---

## 1. 환경 빌드

### 1-1. Isaac-sim 이미지 기반으로 docker 이미지 생성

- Isaac-sim base 이미지 생성
```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
vim ./docker/.env.base # change Isaac-Sim version to 5.1.0, ubuntu24.04
./docker/container.py start # issac-lab-base:latest image 및 container 생성
./docker/container.py enter # issac-lab-base:latest container 접속 확인
```

- InternNav 이미지 생성
```bash
docker build -f docker/Dockerfile-torch2.7.0-cuda12.8-issacsim5.1.0-ros2-jazzy-5090 -t {image_name} . # 도커 이미지 빌드
bash docker/run-torch2.7.0-cuda12.8-issacsim5.1.0-ros2-jazzy-5090.sh # container 생성 및 접속
```

### 1-2. vln-deploy용 시스템 패키지 + Python venv `[완료]`

```bash
sudo apt update && sudo apt install -y libzmq3-dev   # llama.cpp --zmq 링크에 필요
cd /ws/src/InternNav/vln-deploy && /isaac-sim/kit/python/bin/python3 -m venv quantization_kit/convenv_x86   # 반드시 Isaac 인터프리터(3.11)로. 시스템 3.12로 만들면 3-2/3-3이 성립 안 함
cd /ws/src/InternNav/vln-deploy && quantization_kit/convenv_x86/bin/pip install --upgrade pip
cd /ws/src/InternNav/vln-deploy && quantization_kit/convenv_x86/bin/pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.0' 'torchvision==0.22.0'   # cu128 필수 (PyPI 기본은 cu126이라 sm_120 없음)
cd /ws/src/InternNav/vln-deploy && quantization_kit/convenv_x86/bin/pip install safetensors diffusers transformers einops accelerate 'numpy==1.26.4' pillow timm opencv-python-headless onnx onnxscript matplotlib pyzmq msgpack 'tensorrt==10.16.1.11'   # tensorrt 버전 고정 필수 (.engine 이 TRT 버전에 락됨)
```

확인:
```bash
cd /ws/src/InternNav/vln-deploy && quantization_kit/convenv_x86/bin/python -c "import torch, tensorrt, diffusers, transformers; print('OK', torch.cuda.is_available())"   # -> OK True
```

### 1-2. llama.cpp 빌드 (sm_120 + AVX2, AVX-512 없음 + libzmq) `[완료]`
- gguf을 실행하려면 필요함

```bash
cd /ws/src/InternNav/vln-deploy/quantization_kit/llama.cpp && cmake -B build_x86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_F16C=ON -DGGML_FMA=ON -DGGML_AVX512=OFF -DCMAKE_CUDA_ARCHITECTURES=120   # 다른 GPU면 120을 해당 compute capability로
```
```bash
cd /ws/src/InternNav/vln-deploy/quantization_kit/llama.cpp && cmake --build build_x86 --target llama-s2-runner llama-quantize -j$(nproc)
```

확인:
```bash
cd /ws/src/InternNav/vln-deploy/quantization_kit/llama.cpp && ldd build_x86/bin/llama-s2-runner | grep libzmq   # -> libzmq.so.5 => /usr/lib/x86_64-linux-gnu/libzmq.so.5
cd /ws/src/InternNav/vln-deploy/quantization_kit/llama.cpp && cuobjdump --list-elf build_x86/bin/libggml-cuda.so | grep sm_120   # -> sm_120a.cubin
```

### 1-3. S1 TRT 엔진 5개 사전 빌드 `[완료]`

skip 해도 첫 실행 시 자동 빌드되나 수십 초 지연이 붙는다.

```bash
cd /ws/src/InternNav/vln-deploy && quantization_kit/convenv_x86/bin/python -c "
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

확인:
```bash
ls /ws/src/InternNav/vln-deploy/quantization_kit/trt_engines/*_sm120.engine | wc -l   # -> 5
```

### 1-4. Isaac python용 tensorrt 심링크 오버레이 `[완료]`

Isaac 인터프리터·패키지를 건드리지 않고 venv에서 `tensorrt`만 빌려 쓴다. **venv site-packages 전체를 PYTHONPATH에 얹으면 안 된다** (torch/numpy가 Isaac 것을 가림).

```bash
cd /ws/src/InternNav/vln-deploy && mkdir -p quantization_kit/trt_py311 && SP=$PWD/quantization_kit/convenv_x86/lib/python3.11/site-packages && for p in tensorrt tensorrt_bindings tensorrt_libs; do ln -sf $SP/$p quantization_kit/trt_py311/; done
```

확인:
```bash
cd /ws/src/InternNav/vln-deploy && PYTHONPATH=$PWD/quantization_kit/trt_py311 /isaac-sim/python.sh -c "import torch, numpy, tensorrt, internutopia, isaacsim, pinocchio; print(torch.__version__, numpy.__version__, tensorrt.__version__)"   # -> 2.7.0+cu128 1.26.4 10.16.1.11
```

---

## 2. 모델 추출

원본 16 GB에서 S1-only subset(180 MB)을 뽑고, S2 GGUF·runner와 합쳐 `--model-path` 하나로 돌아가는 eval 번들을 만든다.

아래 3개 변수만 바꾸면 다음 체크포인트에도 그대로 쓴다. **같은 터미널에서 한 번만** 실행 (절대경로 — cwd에 안 휘둘리게).

```bash
SRC=/ws/src/InternNav/checkpoints/InternVLA-N1-DualVLN && DST=/ws/src/InternNav/checkpoints/InternVLA-N1-lite2-DualVLN && RECV=/ws/src/InternNav/checkpoints/InternVLA-N1-lite-DualVLN
```

| 변수 | 뜻 | 크기 |
|---|---|---|
| `SRC` | 원본 (S1+S2 합본) | 16 GB |
| `DST` | **이번에 만들 것.** S1 추출본 + eval 번들이 여기 한 곳에 들어간다 | 5.9 GB |
| `RECV` | 배포처에서 받은 S1 lite. 2-2의 대조 기준 — **절대 쓰기 대상으로 삼지 말 것** | 180 MB |

### 2-1. S1 lite 체크포인트 추출
sidecar 복사 + 텐서 추출 2단계. sidecar는 가중치를 뺀 나머지 파일(약 5 MB)이고 그중 `config.json`이 **필수**다 (`S1LiteModel.__init__`이 읽음). `*.safetensors`와 `model.safetensors.index.json`은 빠져야 한다.

```bash
cd /ws/src/InternNav/vln-deploy && bash deploy/tools/copy_sidecar.sh $SRC $DST
cd /ws/src/InternNav/vln-deploy && quantization_kit/convenv_x86/bin/python deploy/tools/extract_s1_weights.py $SRC $DST/model-s1only.safetensors
```

확인:
```bash
ls -la $DST/   # -> model-s1only.safetensors (약 180 MB) + config.json + tokenizer 등
```

### 2-2. eval 번들 생성 (S1 + S2 GGUF + runner 한 디렉토리)

```bash
cd /ws/src/InternNav/vln-deploy && ./deploy/tools/make_eval_bundle.sh $DST --s1 $DST
```

`--s1`과 DST가 같으면 S1 복사를 건너뛰고(`make_eval_bundle.sh:61`) `gguf/`·`bin/`만 붙는다. 2-1이 이미 `$DST`에 S1을 넣었으므로 **디렉토리 하나로 끝난다.** (`--s1=DIR` 등호 형식은 파서가 거부한다 — 공백으로 띄울 것)

옵션:
```
deploy/tools/make_eval_bundle.sh [DST] [--s1 DIR] [--quick] [--verify-only]
  DST            번들 디렉토리 (기본 checkpoints/InternVLA-N1-DualVLN-lite)
  --s1 DIR       S1 lite 소스 (기본 checkpoints/InternVLA-N1-lite-DualVLN)
  --quick        복사한 6 GB의 md5 재확인 생략 (크기만 비교)
  --verify-only  복사 없이 기존 번들만 검증
```
> 기본값 두 개는 **이름만 비슷한 별개 디렉토리**다 (`DualVLN-lite` vs `lite-DualVLN`). 게다가 기본 `--s1`은 받은 파일(`$RECV`)을 가리킨다. 위처럼 `$DST`를 두 자리에 명시하면 둘 다 안 건드린다.

빈 디렉토리에서 약 8초, 재실행 시 전송 0바이트(멱등). 결과 5.9 GB:
```
$DST/
├── model-s1only.safetensors   180 MB   (2-1이 넣음)
├── config.json, tokenizer ...          (2-1이 넣음)
├── gguf/                      5.7 GB   quantization_kit/gguf/ 에서 복사
├── bin/                        71 MB   llama.cpp/build_x86/bin/ 에서 복사
└── BUNDLE_MANIFEST.txt                 빌드 시각·소스 경로·md5
```
`gguf/`와 `bin/`의 출처는 `make_eval_bundle.sh:26,28,52,54`에 고정이라 인자로 못 바꾼다. S2는 체크포인트가 바뀌어도 그대로이기 때문.


---

## 3. 검증

### 3-1. 배포 스택 단독 (Isaac Sim 없이)

**S2 GGUF 로드·추론**
```bash
cd /ws/src/InternNav/vln-deploy && ./quantization_kit/run.sh s2 2>&1 | cut -c1-60   # OK 3줄. cut 없이 돌리면 base64 76,460자가 터미널을 덮음
```
```bash
cd /ws/src/InternNav/vln-deploy && ./quantization_kit/run.sh s2 2>&1 | tail -1 | cut -f2 | base64 -d | wc -c   # -> 57344 = (1,4,3584) float32
```
```bash
cd /ws/src/InternNav/vln-deploy && printf 'RESET\nTEXT\tquantization_kit/dbg/test448.png\t<__media__>Where should you go next? Please output the next waypoint coordinates in the image.\nLATENT\nQUIT\n' | quantization_kit/llama.cpp/build_x86/bin/llama-s2-runner -m quantization_kit/gguf/InternVLA-S2-text-Q4_K_M-latent.gguf --mmproj quantization_kit/gguf/InternVLA-S2-mmproj-f16.gguf 2>&1 | tail -40   # 위가 3줄이 아닐 때 실제 에러 확인 (run.sh는 stderr를 버림)
```

**Dual (S2→S1 동기) — 실 vln-ce 데이터**
```bash
cd /ws/src/InternNav/vln-deploy && ./quantization_kit/run.sh dual --sample quantization_kit/data/vln-ce/samples/sample_00000   # 한 프레임
```
```bash
cd /ws/src/InternNav/vln-deploy && ./quantization_kit/run.sh replay --n 2   # 에피소드 replay. TRT 없으면 --no_trt 로 fallback (~4 Hz)
```

**Async (ZMQ 별개 프로세스, S1/S2 overlap)**
```bash
cd /ws/src/InternNav/vln-deploy && ./quantization_kit/run.sh async --seconds 8 --s2-mps-pct 0   # 기대: S1 ~28 Hz / S2 ~5.6 Hz. --frames-dir 로 실 이미지 폴더 지정 가능
```
```bash
cd /ws/src/InternNav/vln-deploy && ./quantization_kit/run.sh async-clean   # 크래시 시 포트 5601-5610 정리
```

**배포 매니저 lifecycle + test suite**
```bash
cd /ws/src/InternNav/vln-deploy && ./run.sh test --keep-up   # 기대: integration test ALL PASS (18/18). 첫 실행은 S2 데몬 spawn까지 수십 초
```
```bash
cd /ws/src/InternNav/vln-deploy && ./verify/verify_lifecycle.sh   # 기대: 24/24 PASS
```
```bash
cd /ws/src/InternNav/vln-deploy && ./quantization_kit/convenv_x86/bin/python verify/verify_dual_accuracy.py --n 15   # 기대: ACTION AGREEMENT 15/15, SPEEDUP ~6x
```
```bash
cd /ws/src/InternNav/vln-deploy && quantization_kit/convenv_x86/bin/python verify/verify_lookdown_resolution.py --n 30   # 기대: native 30/30(100%), fixed 384 ~12/30(40%) — look_down 프레임 해상도가 낮으면 S2가 pixel goal을 못 내고 discrete action("↓" 등)으로 새는 것을 실측
```
```bash
cd /ws/src/InternNav/vln-deploy && ./run.sh stop   # 데몬 정리 (clean 은 시스템 fuser 필요)
```

### 3-2. Isaac Sim eval 실행 (lite 단독)

**Isaac Sim eval 사전 점검 (Isaac 부팅 없이)**
```bash
cd /ws/src/InternNav && PYTHONPATH=/ws/src/InternNav/vln-deploy/quantization_kit/trt_py311 /isaac-sim/python.sh -c "import importlib; importlib.import_module('scripts.eval.configs.h1_internvla_n1_async_lite_cfg'); from internnav.agent import Agent; import tensorrt, quantization_kit.s1_inference; print('agents:', [k for k in Agent.agents if 'llamacpp' in k], '| tensorrt', tensorrt.__version__)"
```

**원본 대비 정확도·속도 비교** (lmdb 자동 정리 → orig 실행 → lite 실행 → 리포트, 순차. GPU 경합 때문에 병렬 금지)
```bash
cd /ws/src/InternNav && ./vln-deploy/verify/compare_eval.sh -n 10 --lite-model $DST   # 에피소드 10개 이상 권장 (3개면 SR이 0/3~3/3 네 값뿐이라 비교 무의미)
```

`--lite-model`을 안 주면 기본값 `checkpoints/InternVLA-N1-DualVLN-lite`(`compare_eval.sh:42`)를 본다 — 우리 번들은 `$DST`에 있으므로 반드시 명시할 것. `--orig-model`·양쪽 config는 기본값이 맞으니 생략.

옵션:
```
verify/compare_eval.sh [-n N] [-t TOKENS] [-v lite|orig|both] [--native-tokens]
                       [--no-timing] [--keep-lmdb] [--out DIR] [--python P]
                       [--lite-config P] [--lite-model P] [--orig-config P] [--orig-model P]
                       [--extra "ARGS"] [--dry-run]
  -n  에피소드 수 (기본 5) — EVAL_MAX_EPISODES 로 양쪽에 동일 적용
  -t  S2 생성 길이 (기본 128) — 안 맞추면 orig 1024 / lite 128 이라 양자화 쪽에 유리
```
```bash
cd /ws/src/InternNav && ./vln-deploy/verify/compare_eval.sh -n 10 --lite-model $DST --dry-run   # 실행할 명령만 출력
```


**공정성 3조건**: ① 같은 에피소드(`EVAL_MAX_EPISODES` 동일) ② 같은 생성 길이(`EVAL_MAX_NEW_TOKENS=128`) ③ 깨끗한 lmdb. 셋 다 리포트가 어긋나면 경고한다.

**산출물**

| 경로 | 내용 |
|---|---|
| `docs/COMPARISON.md` 섹션 5 | 사람이 읽는 리포트 (`<!-- BEGIN eval-comparison -->` 사이만 갱신) |
| `verify/compare_out/history.jsonl` | 실행마다 1줄 append |
| `verify/compare_out/{orig,lite}_run.log` | `finish:` 줄(에피소드별 duration/steps/fps)이 **여기에만** 있다 |
| `verify/compare_out/{orig,lite}_timing.csv` | `VLN_TIMING` per-call 계측 |
| `verify/compare_out/runs.tsv` | variant/episodes/max_tok/wall_s/exit |
| `<model-path>/logs_stop/result_h1.json` | 평가기가 쓰는 정확도 원본 |

---

## 관련 문서

- [WORKFLOW.md](WORKFLOW_q4_0812.md) — 각 스텝의 목적·기대결과·설계 배경
- [COMPARISON.md](COMPARISON_q4_0812.md) — 원본 vs 배포판 크기/속도/정확도
- `../BUNDLE.md`, `../deploy/README.md`, `../quantization_kit/RUN.md`
