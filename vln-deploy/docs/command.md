# InternVLA-N1 경량화 재현 명령 (5090 · Jetson Thor)

원본 체크포인트 `InternVLA-N1-DualVLN`에서 **경량 기본 구성**(S2 = GGUF Q8_0, S1 = TensorRT FP16)을 직접 다시 만들고 검증하는 순서다.

| 단계 | 내용 | 빈도 |
|---|---|---|
| [1. 환경 빌드](#1-환경-빌드) | docker 이미지 · venv · llama.cpp | 환경당 1회 |
| [2. 경량화](#2-경량화) | S1 ONNX → TRT FP16, S2 GGUF f16 → Q8_0 | 체크포인트마다 |
| [3. 검증](#3-검증) | 재현 동일성 · S1 · S2 · S2+S1 (open-loop) · Isaac Sim 주행 (closed-loop) · RGB-D 프레임 | 경량화마다 |
| [4. Jetson Thor](#4-jetson-thor) | 같은 스크립트를 Thor에서 (기준값은 5090에서 복사) | 장치마다 |
| [5. git](#5-git에-올리는-것--올리지-않는-것) | 올리는 것 / 올리지 않는 것과 다시 얻는 방법 | — |

- 설계 배경과 측정 결과: [`scripts/sim/report.md`](../../scripts/sim/report.md) (§0 결론, §5 dataset 검증). sim 실행·검증 명령: [`scripts/sim/command.md`](../../scripts/sim/command.md) (§4 빌드, §5 dataset 검증)
- 최근 재현 결과: [`Report.html`](Report.html)
- 08월 Q4 경로(옛 문서, 사용 안 함): [command_q4_0812.md](q4_0812/command_q4_0812.md), [WORKFLOW_q4_0812.md](q4_0812/WORKFLOW_q4_0812.md), [COMPARISON_q4_0812.md](q4_0812/COMPARISON_q4_0812.md)

### 한 번에 실행

1장(환경)이 되어 있으면 2장·3장을 명령 하나로 돌린다. **호스트**에서 실행하면 [isaacsim]/[gd_vln] 단계를 알아서 `docker run`으로 띄운다. Thor에서는 장치에서 그대로 실행한다([4장](#4-jetson-thor)).

```bash
cd ~/gd_project/modules/gd_vln/workspace/src/InternNav
./vln-deploy/verify/build_and_verify.sh                                   # 경량화 + 데이터셋 검증 + 합격 판정 (약 8분)
./vln-deploy/verify/build_and_verify.sh --stages build,dataset,sim        # + Isaac Sim 주행 20회 (sim·gdm 먼저 띄움, 3-5)
./vln-deploy/verify/build_and_verify.sh --stages rgbd --frames verify/rgbd_frames/17DRP5sb8fy   # RGB-D 프레임 검증 (3-6)
```
- `--stages`: `build`(2장) · `dataset`(3-3, 3-4 + 합격 판정) · `rgbd`(3-6) · `sim`(3-5). 기본 `build,dataset`.
- 이미 있는 결과물은 재사용한다. 처음부터 다시 만들려면 2-0으로 먼저 치운다. 3-1(md5 재현 동일성)과 3-2(S1 단독)는 러너에 넣지 않았다. 필요하면 아래 단계별 명령으로 돌린다.
- 로그·요약: `vln-deploy/verify/build_out/<시각>-<arch>/run.log`, 끝에 단계별 PASS/FAIL 표.

---

## 0. 전제

### 0-1. 두 개의 실행 환경

원본 HF 모델(기준값)과 경량화 도구가 서로 다른 transformers 버전을 요구해서 이미지를 둘로 나눈다.

| 표기 | 이미지 | 하는 일 | 핵심 버전 |
|---|---|---|---|
| **[isaacsim]** | `dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090` | 경량화 빌드, GGUF/TRT 실행 (`convenv_x86`) | transformers 5.15, TensorRT 10.16.1.11, torch 2.7.0+cu128 |
| **[gd_vln]** | `dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-5090` | 원본 HF 모델 실행 (기준값), 비교 | transformers 4.51, torch 2.9.0+cu130 |

호스트에서 아래 한 줄로 각 셸에 들어간다. 체크포인트(`/media/LTDataset/checkpoints`)와 데이터셋(`/media/TrainDataset`)을 `run_vln_stack.sh`와 같은 위치에 마운트한다. `vln-deploy/`는 root 소유라서 `-u 0`으로 들어간다.

```bash
# [isaacsim] 셸
docker run --rm -it -u 0 --entrypoint bash --gpus all --runtime=nvidia --network=host --ipc=host -v ~/gd_project/modules/gd_vln/workspace:/ws -v /media/LTDataset/checkpoints:/ws/src/InternNav/checkpoints:ro -v /media/TrainDataset:/ws/src/InternNav/data:ro -w /ws/src/InternNav/vln-deploy dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090
```
```bash
# [gd_vln] 셸
docker run --rm -it -u 0 --entrypoint bash --gpus all --runtime=nvidia --network=host --ipc=host -v ~/gd_project/modules/gd_vln/workspace:/ws -v /media/LTDataset/checkpoints:/ws/src/InternNav/checkpoints:ro -v /media/TrainDataset:/ws/src/InternNav/data:ro -w /ws/src/InternNav/vln-deploy dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-5090
```

- 두 셸을 동시에 열어 두어도 되지만 **GPU 작업은 한 번에 하나만** 돌린다 (7B HF 모델과 TRT 빌드가 겹치면 VRAM·RAM이 부족해진다).
- 다른 프로세스가 GPU를 쓰고 있으면(sim 스택 등) 먼저 끈다: `nvidia-smi`.

### 0-2. 변수 블록 — 셸에 들어갈 때마다 먼저 붙여넣기

아래 명령은 전부 이 변수만 쓴다. **아키텍처마다 다른 값은 이 블록에만 있다** (Thor 이식 시 여기만 바꾼다).

```bash
cd /ws/src/InternNav/vln-deploy
QK=quantization_kit
PY=$QK/convenv_x86/bin/python          # [isaacsim] 경량화 venv          Thor: convenv_aarch64
HFPY=/usr/bin/python3                  # [gd_vln]   원본 HF 실행 python
BIN=$QK/llama.cpp/build_x86/bin        # llama.cpp 빌드 결과             Thor: build
CUDA_ARCH=120                          # RTX 5090 = sm_120               Thor: 110
CKPT=checkpoints/InternVLA-N1-DualVLN  # 원본 (16 GB, shard 4개)
R=$QK/out/gguf                         # S2 GGUF 출력 (생성물은 전부 $QK/out/)
Q8=$R/InternVLA-S2-text-Q8_0-latent-embf16.gguf
MM=$R/InternVLA-S2-mmproj-f16.gguf
export PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

- **맨몸 `python`/`pip` 금지** — `.bashrc` alias라 스크립트에서는 없다. 항상 `$PY` / `$HFPY`.
- `checkpoints`는 `vln-deploy/checkpoints -> ../checkpoints` 심링크라 위 마운트로 원본이 보인다.

### 0-3. 데이터셋과 기준값

검증에 쓰는 데이터는 **새로 만들지 않는다.** 공개 데이터셋을 받아 쓰고, 기준값(원본 모델 출력)만 이 PC에서 한 번 만든다. 아래는 전부 이미 있으므로 처음 환경을 꾸릴 때만 한다.

| 무엇 | 위치 (컨테이너 기준) | 어떻게 생겼나 | 쓰는 곳 |
|---|---|---|---|
| 원본 체크포인트 | `checkpoints/InternVLA-N1-DualVLN` | HF 모델 `InternRobotics/InternVLA-N1-DualVLN` (공개, commit `a698a9e`)를 받는다 | 2장 전체 |
| VLN-CE R2R 궤적 | `data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/<scene>/` | HF 데이터셋 `InternRobotics/InternData-N1` (gated, commit `cd04ff6`)의 `vln_ce/traj_data/r2r/<scene>.tar.gz`를 받아 압축 해제 | 3-2 ~ 3-4 |
| S1 기준 입력 | `scripts/sim/verify/out/quant/InternVLA-N1-DualVLN_lookdown30/bf16_outputs.npy` | [gd_vln] `verify_quant.py` (HF bf16 S2+S1 replay) | 3-2의 `ref` |
| S1 기준값 | `vln-deploy/quantization_kit/verify_trt_repro_out/ref.npz` | [gd_vln] `verify_trt_repro.py ref` | 3-2 |
| S2 기준값 | `vln-deploy/quantization_kit/verify_s2_gguf_out/hf_bf16_384.npy` | [gd_vln] `verify_s2_gguf.py hf` | 3-3, 3-4 |
| sim 에피소드 | `scripts/sim/verify/e2e_episodes_etri_indoor_clear.json` | 손으로 정한 5개 (시작 자세·목표 좌표·성공 반경) | 3-5 |

원본 체크포인트 받기 (16 GB, 공개라 로그인 불필요, commit `a698a9e` 고정). [isaacsim] 셸에서 받으면 `/media/LTDataset/checkpoints`에 그대로 들어간다 (마운트를 `:ro` 대신 쓰기 가능하게 연다):
```bash
bash $QK/tools/fetch.sh ckpt checkpoints/InternVLA-N1-DualVLN    # huggingface_hub 는 convenv 에 들어 있다
ls checkpoints/InternVLA-N1-DualVLN/*.safetensors | wc -l          # -> 4
```
- git에 올리지 않는 것과 다시 얻는 방법은 [5장](#5-git에-올리는-것--올리지-않는-것)에 모아 두었다.

**open-loop에서 실제로 쓰는 데이터**는 `verify_s2_gguf.py: episodes()`가 고른다. r2r 아래 장면 폴더를 이름순으로 정렬한 **앞 6개 장면**(17DRP5sb8fy, 1LXtFkjw3qL, 1pXnuDYAj8r, 29hnd4uzFmX, 2n8kARJN3HM, 5LpN3gDmAk7)의 **첫 에피소드**, 앞 **40 step**이다. 그래서 받을 때도 이 6개 장면만 있으면 된다. 이 에피소드들은 R2R **train** split이다(지시문이 `raw_data/r2r/train`과 일치). 경량화 전후가 같은지 보는 시험이라 문제없지만, 처음 보는 장면에서의 성능을 잰 것은 아니다.

데이터 6 장면 받기 (gated: HF에서 `InternRobotics/InternData-N1` 접근 승인을 받고 토큰을 준비한다, commit `cd04ff6` 고정):
```bash
HF_TOKEN=hf_xxx bash $QK/tools/fetch.sh data ../data/InternData-N1-v0.5-mini   # 6 장면 tar.gz 받기 + 압축 해제
ls ../data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/17DRP5sb8fy             # -> data meta videos
```
- 전체를 풀 때는 `InternNav/scripts/dataset_converters/extract_dataset.sh <DIR> [JOBS]`.
- 한 장면 폴더 구성: `meta/episodes.jsonl`(지시문·길이), `data/chunk-000/episode_XXXXXX.parquet`(자세·정답 goal pixel), `videos/chunk-000/observation.images.rgb.125cm_0deg/`(정면 카메라 jpg), `…125cm_30deg/`(30° 아래 카메라).

기준값 만들기 (원본 체크포인트나 데이터를 바꿨을 때만, [gd_vln], 순서대로):
```bash
cd /ws/src/InternNav && $HFPY scripts/sim/verify/verify_quant.py --model_path checkpoints/InternVLA-N1-DualVLN --quants bf16 --episodes 6 --max_steps 40   # -> bf16_outputs.npy
cd /ws/src/InternNav/vln-deploy && $HFPY $QK/verify_trt_repro.py ref                                  # -> verify_trt_repro_out/ref.npz (64 case)
cd /ws/src/InternNav/vln-deploy && $HFPY $QK/verify_s2_gguf.py hf --resize 384 --tag hf_bf16_384      # -> verify_s2_gguf_out/hf_bf16_384.npy (S2 48회)
```

---

## 1. 환경 빌드

환경당 한 번. 이미 되어 있으면 각 "확인"만 돌린다.

### 1-1. docker 이미지

```bash
# 호스트. 이미지가 없을 때만
docker image inspect dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090 dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-5090 >/dev/null && echo OK
```
- gd_vln 이미지가 없으면 `InternNav/docker/run-torch2.9.0-cuda13.0-habitat-jazzy-5090.sh`이 Dockerfile로 빌드한다.
- isaacsim 이미지 실행 스크립트: `InternNav/docker/run-torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090.sh`

### 1-2. 경량화 venv `convenv_x86` — [isaacsim]

```bash
apt update && apt install -y libzmq3-dev          # llama.cpp --zmq 링크용
bash $QK/tools/setup_venv.sh                      # Isaac python 3.11 로 venv + requirements_x86.txt (검증된 고정 버전 78 개)
```
- 설치 내용: torch 2.7.0 / torchvision 0.22.0 (cu128 인덱스, sm_120 지원), 나머지는 `quantization_kit/tools/requirements_x86.txt` (2026-10-27 `pip freeze`, transformers 5.15 · tensorrt 10.16.1.11 · numpy 1.26.4 등).
확인:
```bash
$PY -c "import torch, tensorrt, transformers; print(torch.__version__, tensorrt.__version__, transformers.__version__, torch.cuda.is_available())"
# -> 2.7.0+cu128 10.16.1.11 5.x True
```
- `tensorrt` 버전 고정 이유: `.engine` 파일은 빌드한 TRT 버전에서만 열린다. TRT 11부터는 `BuilderFlag.FP16`이 없어 `trt_s1.py`가 동작하지 않는다.
- Thor: torch는 Jetson SBSA 인덱스, TensorRT는 pip 휠이 Tegra를 거부하므로 `apt install tensorrt python3-libnvinfer` ([quantization_kit/RUN.md](../quantization_kit/RUN.md) §0).

### 1-3. llama.cpp 빌드 (`llama-s2-runner`, `llama-quantize`) — [isaacsim]

```bash
bash $QK/tools/fetch.sh llama                     # upstream ggml-org/llama.cpp @ 389ff61d7 + tools/llama.cpp-vln.patch
cmake -S $QK/llama.cpp -B $QK/llama.cpp/build_x86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_F16C=ON -DGGML_FMA=ON -DGGML_AVX512=OFF -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH
cmake --build $QK/llama.cpp/build_x86 --target llama-s2-runner llama-quantize -j$(nproc)
```
확인:
```bash
ldd $BIN/llama-s2-runner | grep libzmq          # -> libzmq.so.5
$BIN/llama-quantize --help 2>&1 | grep -c Q8_0  # -> 1 이상
```
- `GGML_AVX512=OFF`: 이 PC의 CPU(Arrow Lake)는 AVX-512가 없다. Thor(ARM)에서는 AVX 옵션 4개를 빼고 `-B build`.
- `fetch.sh llama`가 하는 일을 손으로 하면 (원본 받기 → patch 적용):
  ```bash
  git init -q $QK/llama.cpp && git -C $QK/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git
  git -C $QK/llama.cpp fetch --depth 1 origin 389ff61d77b5c71cec0cf92fe4e5d01ace80b797 && git -C $QK/llama.cpp checkout FETCH_HEAD   # 원본 (다운로드 33 MB)
  git -C $QK/llama.cpp apply $PWD/$QK/tools/llama.cpp-vln.patch                                                                     # 우리 수정 덮어쓰기
  git -C $QK/llama.cpp status --short   # -> M 6 개 + ?? 2 개 (tools/mtmd/ 아래)
  ```
- **우리 수정 = `tools/llama.cpp-vln.patch`** (tools/mtmd 8 파일): `s2-runner.cpp`·`guidedog-cli.cpp`(새 파일), HF parity(`MTMD_QWEN25VL_HF_ALIGN`, `S2_HF_CHAT`), latent EOS, vision 임베딩 캐시(`S2_EMBD_CACHE`). 기존 vendored 소스의 출발점이 upstream `389ff61d7`(2026-05-10)이고, 그 위에 patch를 얹으면 vendored 소스와 8 파일 모두 바이트 단위로 같다(2026-10-27 확인). 3-3의 `--hf_parity`는 이 patch가 있어야 동작한다.

---

## 2. 경량화

결과물은 전부 `quantization_kit/out/` 한 폴더에 생긴다. 실행 예시 소요 시간은 RTX 5090 실측 ([Report.html](Report.html)).

| 결과물 | 경로 | 크기 |
|---|---|---|
| S1 ONNX 7개 | `out/onnx/*.onnx` | 0.35 GB |
| S1 TRT FP16 엔진 5개 | `out/trt_engines/*_native_fp16_sm120.engine` | 0.22 GB |
| S2 decoder **Q8_0** (기본) | `out/gguf/InternVLA-S2-text-Q8_0-latent-embf16.gguf` | 9.1 GB |
| S2 vision projector | `out/gguf/InternVLA-S2-mmproj-f16.gguf` | 1.35 GB |
| (중간) Qwen 분리본·text f16·f16-latent | `out/gguf_build/` — 빌드 중에만 있고 끝나면 지운다 | 최대 약 45 GB |

### 2-0. (다시 만들 때) 기존 결과물 치우기 — [isaacsim]

빌드 스크립트들은 **파일이 있으면 건너뛴다**. 다시 만들거나 체크포인트를 바꿀 때는 `out/` 을 통째로 옮긴다 (지우지 않고 rename — 3-1 비교에 쓴다).

```bash
mv $QK/out $QK/out.old-$(date +%Y%m%d) && ls -d $QK/out.old-*
```

### 2-1. S1 → ONNX → TensorRT FP16 — [isaacsim]

```bash
bash $QK/tools/build_s1_trt.sh $CKPT          # [ckpt] [onnx_dir] [engine_dir], 기본 out/onnx / out/trt_engines
```
1. `export_s1_onnx.py`: 원본 S1(diffusion policy)을 TRT로 옮길 수 있는 부분 7개로 나눠 ONNX로 내보낸다 (1-step denoise 네트워크, DINOv2, memory encoder, resampler, head들). 반복(diffusion loop)은 그래프로 못 만들어서 1 step만 ONNX가 되고 반복은 Python이 돈다.
2. `trt_s1._engine_for`: ONNX → TensorRT FP16 엔진 5개 (`out/trt_engines/*_native_fp16_sm120.engine`).

- 엔진 이름의 `sm120`은 GPU 세대에서 자동으로 붙는다(Thor는 `sm110`). 엔진은 GPU 세대·TRT 버전에 묶이므로 다른 GPU에서는 이 단계만 다시 한다.
- 처음 빌드 약 1분(5090). **Isaac Sim이 떠 있으면 엔진 빌드(`build_serialized_network`)가 멈춘다** — 2026-10-27에 sim 켜진 상태 3회 모두 멈춤(`docker rm -f`도 안 먹는 좀비 컨테이너로 남음), sim을 끄면 61초에 성공. 빌드 전에 `docker stop isaac-sim`. 이미 만든 엔진을 불러오는 것은 sim과 같이 돌아도 된다(3-5).

### 2-2. S2 → GGUF f16 (+ latent 토큰) → Q8_0 — [isaacsim]

```bash
bash $QK/tools/build_s2_gguf.sh $CKPT $R     # 약 2분 (5090)
ls -la $Q8 $MM
```
스크립트가 하는 일 (5단계, 이미 Q8_0·mmproj가 있으면 건너뛴다):
1. `strip_to_qwen.py`: DualVLN에서 S1을 떼고 순수 Qwen2.5-VL 형식으로 (`out/gguf_build/qwen`)
2. text decoder → GGUF f16
3. vision projector(mmproj) → GGUF f16
4. `gguf_add_latent_tokens.py`: S2가 S1에 넘기는 latent query 4행을 GGUF에 그대로 주입 → `text-f16-latent.gguf`
5. **decoder Q8_0** (배포 기본): decoder 블록만 8 bit, 토큰 임베딩·출력층은 f16 (`embf16`). 끝나면 `out/gguf_build/`(중간 파일)를 지워 `out/gguf/`에는 배포 파일 2개만 남는다

- Q8_0를 기본으로 쓰는 이유: FP16과 정확도가 같으면서 더 빠르고 6 GB 작다. Q4는 로봇 명령 일치가 95.4%로 기준(99%)에 못 미쳐 더 이상 만들지 않는다.
- 두 빌드 스크립트 모두 `uname -m`으로 venv(`convenv_x86|aarch64`)와 llama.cpp 빌드 폴더(`build_x86|build`)를 고른다. `PY=…`, `BIN=…` 환경변수로 덮어쓸 수 있다.

---

## 3. 검증

모든 비교의 기준선은 **원본 HF bf16 S2 + PyTorch bf16 S1**이다. 3-1~3-4는 **open-loop**(녹화된 VLN-CE R2R 6 에피소드 × 40 step을 정해진 순서로 재생, 모델 출력이 다음 입력을 바꾸지 않음), 3-5만 **closed-loop**(Isaac Sim에서 모델 명령대로 로봇이 움직임).

**open-loop 데이터**: 0-3 참고 (R2R train 6 장면 × 첫 에피소드 × 40 step, S2 48회).

| 단계 | 무엇을 보나 | 합격 기준 (기존 실측) |
|---|---|---|
| 3-1 재현 동일성 | 새로 만든 파일이 이전 결과물과 같은가 | ONNX·GGUF md5 일치 |
| 3-2 S1 단독 | TRT FP16 vs PyTorch 경로 차 | 경로 차 ≈ 1 mm |
| 3-3 S2 단독 | Q8_0 vs HF: 행동·텍스트·pixel, latent cos | 텍스트 ≈ 96%, latent cos ≈ 0.9988 |
| **3-4 S2+S1** | 최종 로봇 명령·경로 | **명령 일치 ≥ 0.99, ADE<5cm ≥ 0.99** (0.992 / 1.000) |
| 3-5 sim (closed-loop) | 실제 주행 성공률 | 기준선 이상, c3·c4 전부 성공 (12/20) |
| 3-6 RGB-D 프레임 | 라벨 없는 프레임에서 이상 출력이 없는가, 두 장치 결과가 같은가 | checks 전부 통과, S1 명령 일치 ≥ 0.99 |

### 3-1. 재현 동일성 — [isaacsim]

2-0에서 옮겨 둔 이전 결과물과 비교한다 (처음 만드는 경우엔 건너뛴다).

```bash
O=$(ls -d $QK/out.old-* | tail -1); echo "비교 대상: $O"
for f in $QK/out/onnx/*.onnx $R/*.gguf; do o=$O/${f#$QK/out/}; [ -f $o ] || { echo "NEW   $f"; continue; }; [ "$(md5sum <$f)" = "$(md5sum <$o)" ] && echo "SAME  $f" || echo "DIFF  $f"; done
```
- TRT 엔진은 빌드할 때마다 바이트가 달라지므로(커널 자동 선택) 비교하지 않는다. 대신 3-2에서 수치로 본다.

### 3-2. S1 TRT FP16 vs PyTorch — [isaacsim]

```bash
$PY $QK/verify_trt_repro.py trt
```
- 같은 입력(실제 VLN-CE 프레임 + HF S2 latent)과 같은 초기 noise로 원본 PyTorch S1(bf16/fp32)과 재현 TRT 엔진을 돌려 경로를 비교한다. 결과: `quantization_kit/verify_trt_repro_out/report.json`
- 기준 `verify_trt_repro_out/ref.npz`가 없을 때만 먼저 [gd_vln] `$HFPY $QK/verify_trt_repro.py ref`.

### 3-3. S2 Q8_0 vs HF bf16 — [isaacsim] → [gd_vln]

```bash
# [isaacsim] Q8_0 replay (48회 S2 호출)
$PY $QK/verify_s2_gguf.py gguf --hf_parity --resize 384 --embd_cache 512 --tag hfp_gguf_q8 --text $Q8 --mmproj $MM
```
```bash
# [gd_vln] HF 기준과 비교
$HFPY $QK/verify_s2_gguf.py cmp --ref hf_bf16_384
```
- `--hf_parity`: llama.cpp 쪽 입력을 HF와 똑같이 맞춘다 (해상도 392 / look-down 644×476, chat 형식, latent 앞 EOS, 반복 패널티 범위). 끄면 HF와 비교 자체가 공정하지 않다.
- `--embd_cache 512`: history 프레임의 vision 임베딩을 캐시. 결과는 비트 단위로 같고 약 100 ms 빠르다.
- HF 기준 `verify_s2_gguf_out/hf_bf16_384.npy`는 원본 체크포인트에만 의존하므로 재사용한다. 체크포인트를 바꿨거나 없으면 먼저 [gd_vln] `$HFPY $QK/verify_s2_gguf.py hf --resize 384 --tag hf_bf16_384`.
- 결과: `verify_s2_gguf_out/report_vs_hf_bf16_384.json`

### 3-4. S2+S1 전체 (최종 로봇 명령) — [gd_vln] → [isaacsim] → [gd_vln]

3-3의 S2 출력을 S1까지 흘려 최종 명령을 비교한다. S2 한 번당 그 프레임과 다음 4 프레임에 S1을 돌린다(240 슬롯, 경로 165개). 모든 조합이 슬롯마다 같은 초기 noise를 쓰므로 차이는 모델에서만 나온다.

```bash
# [gd_vln] S1 원본 PyTorch bf16 (HF S2 latent / Q8 S2 latent 각각)
$HFPY $QK/verify_s1s2_full.py ref --test_tags hfp_gguf_q8
```
```bash
# [isaacsim] S1 TRT FP16 (재현 엔진 + DINOv2/cond TRT, 배포와 같은 구성)
$PY $QK/verify_s1s2_full.py trt --test_tags hfp_gguf_q8
```
```bash
# [gd_vln] 비교 + 합격 판정
$HFPY $QK/verify_s1s2_full.py cmp --test_tags hfp_gguf_q8
$HFPY -c "
import json; r = json.load(open('$QK/verify_s2_gguf_out/report_s1s2_full.json'))
for k in ('hfp_gguf_q8+TRT', 'HF+TRT'):
    c, t = r[k]['command_agree(first action)'], r[k]['traj_agree(ADE<5cm)']
    print(f'{k:<18} 명령 일치 {c:.3f}  ADE<5cm {t:.3f}  ', 'PASS' if c >= 0.99 and t >= 0.99 else 'FAIL')"
```
기대:
```
hfp_gguf_q8+TRT    명령 일치 0.992  ADE<5cm 1.000   PASS
HF+TRT             명령 일치 0.996  ADE<5cm 1.000   PASS
```
- `HF+TRT`: S2는 원본 그대로, S1만 TRT → S1 경량화만의 영향
- `hfp_gguf_q8+TRT`: 배포 구성 전체 (S2 Q8_0 + S1 TRT)
- 결과: `verify_s2_gguf_out/report_s1s2_full.json`

### 3-5. sim closed-loop (Isaac Sim) — 호스트

2장 결과물을 Isaac Sim 로봇에 붙여 실제로 주행한다. ETRI 실내 장면, **Mode B**(S1 경로를 MPC가 그대로 추종, planner 없음), 장애물 없는 구간 5개(`e2e_episodes_etri_indoor_clear.json`) × 4회. 성공 = 모델이 STOP을 내고 목표에서 에피소드별 반경 이내(c1·c2 1.0 m, c3~c5 1.5 m). 호스트에 X 디스플레이(`DISPLAY=:0`)가 있어야 한다. 다른 GPU 작업은 끈다.

```bash
# 1) sim (터미널 1, 포어그라운드로 계속 떠 있음). 장면 로딩 약 2분
cd ~/gd_project/modules/sim_docker && ./run_rbq10_etri_sim.sh indoor
```
```bash
# 2) gdm Mode B (tmux 세션 GDM-LPP 를 백그라운드로 만든다) + gd_vln 컨테이너
docker exec gdm_docker-gdm-1 bash -lc 'cd /home/gdr/gdm/workspace && ./lpp-sim-vln' ; ~/gd_project/modules/gd_vln/workspace/src/InternNav/docker/run-torch2.9.0-cuda13.0-habitat-jazzy-5090.sh -d
```
```bash
# 3) 준비 확인: 카메라 토픽이 보일 때까지 기다린다
docker exec internnav-habitat bash -c 'cd /ws/src/InternNav && source scripts/sim/setup_ros_ws.sh >/dev/null; ros2 topic list | grep -E "^/clock|camera/camera/color/image_raw|gdq_odom"'   # -> 3줄
```
```bash
# 4) 주행 20회: 스택(S2 Q8_0 + S1 TRT, async :5801)을 띄우고 5 ep × 4회 실행 후 스택 정지 (약 15분)
cd ~/gd_project/modules/gd_vln/workspace/src/InternNav && bash scripts/sim/verify/run_s2_sr.sh 4 gguf_q8
```
```bash
# 5) 요약 표
docker exec internnav-habitat bash -c 'cd /ws/src/InternNav && python3 scripts/sim/verify/summarize_e2e.py scripts/sim/verify/out/e2e --match s2sr_gguf_q8_rep'
```
기대 (2026-10-27 실측, 이전 기록은 [scripts/sim/report.md §6-1](../../scripts/sim/report.md#6-1-구성별-sr-mode-b)):

| 구간 | c1 프린터 | c2 마커 | c3 교차로 | c4 프린터 지나 교차로 | c5 중간 출발 | SR |
|---|---|---|---|---|---|---|
| 이번 (Q8_0 + TRT) | 0/4 | 0/4 | 4/4 | 4/4 | 4/4 | **12/20** |
| 10-06 (Q8_0 + TRT) | 0/4 | 0/4 | 4/4 | 4/4 | 1/4 | 9/20 |
| 기준선 (HF + PyTorch) | 0/2 | 0/2 | 2/2 | 2/2 | 1/2 | 5/10 |

- **합격 판단**: c1·c2는 기준선도 항상 실패하는 구간이고 c3·c4는 항상 성공한다. 기준선보다 SR이 낮지 않고 c3·c4가 전부 성공이면 통과. c5는 반복마다 흔들리는 구간이다.
- sim 루프 지연 기대: S2 p50 ≈ 500 ms, S1 p50 ≈ 70 ms (`scripts/sim/verify/out/live/s2sr_gguf_q8_dual.log`의 `s1=`, `s2=`).
- 결과: `scripts/sim/verify/out/e2e/s2sr_gguf_q8_rep{1..4}/`. dual_server 로그의 reset 횟수가 에피소드 수(20)와 같아야 측정이 오염되지 않은 것이다.
- 스택이 쓰는 파일이 2장의 결과물이다: `out/gguf/…Q8_0…gguf`, `out/onnx/`, `out/trt_engines/`.
- 끄기: `docker stop isaac-sim`, gdm 컨테이너 안에서 `./lpp-kill`. 데모로 직접 지시를 주려면 `scripts/sim/run_vln_sim.sh -b gguf`.

### 3-6. RGB-D 프레임 검증 (S2 · S1 · S2+S1) — [isaacsim]

정답 라벨이 없는 RGB-D 프레임(로봇 RealSense 녹화, 또는 데이터셋에서 뽑은 프레임)으로 세 가지를 확인한다. 같은 프레임을 두 장치(5090, Thor)에서 돌려 결과를 비교하는 것이 주 용도다. 스크립트: `verify/verify_rgbd.py`.

프레임 폴더 형식:
```
<DIR>/rgb/0000.jpg ...        정면 카메라 (필수)
<DIR>/depth/0000.png ...      uint16 mm (RealSense·데이터셋과 같은 단위, 없으면 0)
<DIR>/lookdown/0000.jpg ...   30° 아래 카메라 (S2가 "↓"일 때, 선택)
<DIR>/instruction.txt         지시문 (또는 --instruction)
```
```bash
# 데이터셋 에피소드 -> 프레임 폴더 (데이터셋이 있는 쪽에서)
$PY verify/verify_rgbd.py export --scene 17DRP5sb8fy --episode 0 --steps 40 --out verify/rgbd_frames/17DRP5sb8fy
# S2+S1 실행 (S2는 5 프레임마다, S1은 매 프레임) -> verify/rgbd_out/<이름>/{s2.npy,s1.npz,summary.json}
$PY verify/verify_rgbd.py run --frames verify/rgbd_frames/17DRP5sb8fy --mode dual --out verify/rgbd_out/17DRP5sb8fy-x86
# S1 단독: 다른 실행의 S2 출력을 그대로 넣어 S1만 비교
$PY verify/verify_rgbd.py run --frames verify/rgbd_frames/17DRP5sb8fy --mode s1 --latents verify/rgbd_out/17DRP5sb8fy-x86 --out verify/rgbd_out/17DRP5sb8fy-s1
# 두 실행 비교 (S2 판단·텍스트·pixel·latent cos, S1 첫 명령·ADE). S1 명령 일치·ADE<5cm >= 0.99 면 PASS
$PY verify/verify_rgbd.py cmp verify/rgbd_out/17DRP5sb8fy-x86 verify/rgbd_out/17DRP5sb8fy-aarch64
```
- `--mode s2|s1|dual`. `s1`에서 `--latents`가 없으면 S2를 먼저 한 번 돌려(시간 측정 제외) latent를 얻는다.
- `summary.json`의 `checks`: latent 모양 (4×3584)·유한값, 경로 유한값·10 m 이내, 명령이 전진/좌/우뿐인지. 하나라도 어기면 종료코드 1.
- S1 초기 noise는 프레임마다 고정(`verify_s1s2_full.py`와 같은 방식)이라, 같은 프레임이면 장치 간 차이는 모델·커널 차이에서만 나온다.
- DualVLN S1은 depth를 쓰지 않는다(RGB만). depth는 그대로 넘기므로 depth를 쓰는 S1(NavDP 계열)을 붙여도 같은 폴더로 검증할 수 있다.
- 옵션: `--no-trt`(S1 PyTorch fp32), `--s1-steps`, `--gap`, `--text/--mmproj`(다른 GGUF), `--engine-dir`.
- 5090 실측 (17DRP5sb8fy 20 프레임): S2 4회 p50 378 ms, S1 15회 p50 56 ms (18 Hz), checks 전부 통과. 같은 장치에서 `s1 --latents`로 다시 돌린 결과와 비교하면 ADE 0 cm(비트 동일).

---

## 4. Jetson Thor

같은 스크립트를 Thor(aarch64, sm_110)에서 그대로 돌린다. 경로·venv·빌드 폴더는 `uname -m`으로 자동 선택된다. Thor에는 HF 원본(transformers 4.51) 환경이 없으므로 **기준값은 5090에서 만든 파일을 복사**해 쓴다.

| 항목 | 5090 (x86) | Thor (aarch64) |
|---|---|---|
| 실행 | 호스트 → docker | 장치에서 바로 |
| venv | `quantization_kit/convenv_x86` | `quantization_kit/convenv_aarch64` |
| torch / TensorRT | cu128 wheel / pip `tensorrt==10.16.1.11` | Jetson SBSA cu130 wheel / `apt install tensorrt python3-libnvinfer` |
| llama.cpp | `build_x86`, `CUDA_ARCH=120`, AVX 옵션 | `build`, `CUDA_ARCH=110`, AVX 옵션 없음 |
| TRT 엔진 | `*_sm120.engine` | `*_sm110.engine` (Thor에서 새로 빌드) |
| GGUF | 5090에서 빌드 | 아키텍처 무관 → **5090 것을 복사**하거나 Thor에서 빌드 |
| HF 기준값 | gd_vln에서 생성 | 5090 결과 복사 |

### 4-1. Thor 준비 (1회)

```bash
# 5090 호스트 -> Thor: 코드 + 원본 체크포인트 + 데이터 6 장면 + 기준값 (THOR=user@thor-ip)
cd ~/gd_project/modules/gd_vln/workspace/src
rsync -a --exclude 'quantization_kit/convenv_*' --exclude 'llama.cpp/build*' --exclude 'gguf_build' --exclude '*.old-*' InternNav/ $THOR:~/InternNav/
rsync -a /media/LTDataset/checkpoints/InternVLA-N1-DualVLN $THOR:~/InternNav/checkpoints/
for s in 17DRP5sb8fy 1LXtFkjw3qL 1pXnuDYAj8r 29hnd4uzFmX 2n8kARJN3HM 5LpN3gDmAk7; do rsync -aR /media/TrainDataset/./InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/$s $THOR:~/InternNav/data/; done
```
- `InternNav/vln-deploy/internnav`과 `checkpoints`는 상위 폴더를 가리키는 심링크라 InternNav 전체를 옮긴다. 옮기기 전에 `du -sh InternNav` 로 크기를 본다 (`out/gguf/` 10.5 GB 포함, Thor에서 GGUF를 다시 만들 거면 `--exclude out`).
- 기준값(`verify_s2_gguf_out/hf_bf16_384.{npy,json}`, `s1s2_full_pt__hf_bf16_384.npz`)은 위 rsync에 같이 들어간다.

```bash
# Thor 에서 (vln-deploy/). rsync 대신 git clone 했다면 먼저: fetch.sh ckpt / fetch.sh data (0-3)
sudo apt install -y tensorrt python3-libnvinfer libzmq3-dev
bash quantization_kit/tools/setup_venv.sh          # convenv_aarch64 (Jetson SBSA cu130 torch)
bash quantization_kit/tools/fetch.sh llama          # llama.cpp 를 옮기지 않았을 때
cmake -S quantization_kit/llama.cpp -B quantization_kit/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=110
cmake --build quantization_kit/llama.cpp/build --target llama-s2-runner llama-quantize -j$(nproc)
```
- 원본 Thor 설치 안내: [quantization_kit/RUN.md](../quantization_kit/RUN.md) §0. TensorRT는 pip 휠이 Tegra를 거부하므로 apt 패키지를 쓴다(스크립트가 `/usr/lib/python3.12/dist-packages`를 자동으로 붙인다).
- Thor의 TensorRT가 11 이상이면 `trt_s1.py`가 쓰는 `BuilderFlag.FP16`이 없어 엔진 빌드가 실패한다. `dpkg -l | grep nvinfer`로 10.x인지 먼저 확인한다.

### 4-2. Thor 실행

```bash
cd ~/InternNav
./vln-deploy/verify/build_and_verify.sh                          # 빌드(TRT sm110, GGUF) + 데이터셋 검증 + 합격 판정
./vln-deploy/verify/build_and_verify.sh --stages dataset         # GGUF·엔진을 이미 만들었거나 5090 GGUF를 복사했으면
```
Thor에서 데이터셋 검증이 하는 일:
- 3-3: Thor S2(Q8_0) replay → HF 기준과 비교 (`--no_s1`: HF S1이 필요한 경로 ADE 열은 빠진다)
- 3-4: Thor S1 TRT로 `HF+TRT`(S1만 Thor), `thor_q8+TRT`(S2·S1 모두 Thor)를 5090 HF+PyTorch 기준과 비교 → 합격 판정 (명령 일치·ADE<5cm ≥ 0.99)

RGB-D (5090 결과와 장치 간 비교):
```bash
# 5090: 프레임 만들고 기준 실행
./vln-deploy/verify/build_and_verify.sh --stages rgbd --frames verify/rgbd_frames/17DRP5sb8fy        # -> verify/rgbd_out/17DRP5sb8fy-x86
# Thor: 프레임·기준 실행 폴더를 복사한 뒤
./vln-deploy/verify/build_and_verify.sh --stages rgbd --frames verify/rgbd_frames/17DRP5sb8fy --rgbd-ref verify/rgbd_out/17DRP5sb8fy-x86
```
- R-1 Thor S2+S1 실행, R-2 5090 S2 출력으로 Thor S1만 실행, R-3·R-4 5090 실행과 비교.
- 로봇 RealSense 녹화도 같은 폴더 형식(rgb/, depth/ uint16 mm)으로 넣으면 같은 명령으로 돈다.
- **Thor 단계는 아직 Thor에서 돌려보지 않았다.** 경로 분기는 코드만 맞춰 두었다.

## 5. git에 올리는 것 / 올리지 않는 것

**올리는 것** = 코드와 문서만 (`vln-deploy/.gitignore`, `scripts/sim/.gitignore`):
`vln-deploy/` 의 `docs/` · `deploy/` · `verify/*.py,*.sh` · `quantization_kit/*.py,*.sh,RUN.md` · `quantization_kit/async_zmq/` · `quantization_kit/tools/`(빌드·다운로드 스크립트, `llama.cpp-vln.patch`, `requirements_x86.txt`), `scripts/sim/` 코드, `internnav/.../internvla_n1_policy_llamacpp*.py`.

**올리지 않는 것과 다시 얻는 방법:**

| 항목 | 크기 | 다시 얻기 |
|---|---|---|
| 원본 체크포인트 `InternVLA-N1-DualVLN` | 16 GB | 다운로드: `fetch.sh ckpt` (0-3) |
| 검증 데이터 InternData-N1 6 장면 | 0.1 GB | 다운로드: `fetch.sh data` (0-3, gated) |
| llama.cpp 소스 | 0.15 GB | 다운로드 + patch: `fetch.sh llama` (1-3) |
| llama.cpp 빌드 `build_x86/` · `build/` | 0.2 GB | 1-3 `cmake` |
| venv `convenv_x86/` · `convenv_aarch64/` | 12 GB | 다운로드: `setup_venv.sh` (1-2) |
| 생성 weight `out/` (S1 ONNX·TRT 0.6 GB + S2 GGUF 10.5 GB) | 11 GB | 2-1 `build_s1_trt.sh`, 2-2 `build_s2_gguf.sh` |
| 기준값 `verify_s2_gguf_out/` · `verify_trt_repro_out/` · `scripts/sim/verify/out/quant/` | 0.4 GB | 0-3 "기준값 만들기" + 3장 |
| RGB-D 프레임 `verify/rgbd_frames/` | 작음 | `verify_rgbd.py export` (3-6) |
| 실행 결과·로그 (`build_out/`, `rgbd_out/`, `scripts/sim/verify/out/`, `*.log`) | 1.1 GB | 다시 실행하면 생김 |
| `scripts/sim/_staging/` (옛 패치 스크립트·백업) | 0.6 MB | 내용은 `llama.cpp-vln.patch` 와 vln-deploy 코드에 이미 반영됨 |

**다시 만들 수 없는 것 (원본 데이터로 취급, push 안 함):** 08월 원 개발자 번들 `quantization_kit/onnx/` · `trt_engines/` · `gguf/`(Q4) 와 그 번들의 샘플 `quantization_kit/data/vln-ce/samples/`. 지금 경로(1~4장)에서 이것을 쓰는 곳은 3-2 `verify_trt_repro.py`(원본 엔진과 비교하는 부분)뿐이다. 옛 Q4 경로 스크립트(`quantization_kit/run.sh dual`, `deploy` 기본값, `verify_dual_accuracy.py` 등)도 이 번들을 쓴다.

---

## 관련 문서

- [`scripts/sim/report.md`](../../scripts/sim/report.md) — 경량화 설계·실측 전체 (명령은 [`command.md`](../../scripts/sim/command.md))
- [`Report.html`](Report.html) — 이 문서대로 재현한 최근 결과
- [`../quantization_kit/RUN.md`](../quantization_kit/RUN.md) — Thor 원본 실행 가이드 (Q4 번들)
- 옛 Q4 경로 (08-12, 사용 안 함): [command_q4_0812.md](q4_0812/command_q4_0812.md), [WORKFLOW_q4_0812.md](q4_0812/WORKFLOW_q4_0812.md), [COMPARISON_q4_0812.md](q4_0812/COMPARISON_q4_0812.md)
