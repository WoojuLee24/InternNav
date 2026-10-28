# scripts/sim — 실행·검증 명령

InternVLA-N1(gd_vln) ↔ Isaac Sim RBQ10(sim_docker) ↔ gdm(gdm_docker)을 띄우고 검증하는 명령을 모았다. 파일과 토픽, 파라미터 설명은 [README.md](README.md), 측정 결과는 [report.md](report.md)에 있다.

- **절 번호는 report.md와 같다.** 이 문서의 §5를 실행한 결과는 report.md §5에 있다.
- 실행 위치 표기:
  - [호스트]
  - [gd_vln]: `internnav-habitat` 컨테이너 (`docker/run-torch2.9.0-cuda13.0-habitat-jazzy-5090.sh`)
  - [gdm]: `gdm_docker-gdm-1` 컨테이너
  - [isaacsim]: isaacsim 이미지 + `vln-deploy/quantization_kit/convenv_x86`
- 경로는 `gd_vln/workspace/src/InternNav` 기준이다.

| 절 | 내용 | 결과 (report.md) |
|---|---|---|
| [0. 한 번에 실행](#0-한-번에-실행) | sim + gdm + 모델 스택 + bridge + 웹을 한 줄로 | — |
| [1. 따로 띄우기](#1-따로-띄우기) | sim, gdm, gd_vln 컨테이너, isaacsim 셸 | — |
| [2. 모델 스택](#2-모델-스택) | `run_vln_stack.sh` 구성별 | — |
| [3. 로봇 연결과 시각화](#3-로봇-연결과-시각화) | bridge, 웹, rviz, 지시 변경 | — |
| [4. 모델 빌드](#4-모델-빌드) | S1 TRT, S2 GGUF (최초 1회) | — |
| [5. dataset 검증](#5-dataset-검증) | S1, S2, S2+S1, 분리 구조 (open-loop) | [§5](report.md#5-dataset-검증-결과) |
| [6. sim 주행 검증](#6-sim-주행-검증) | 구성별 SR, Mode A/B (closed-loop) | [§6](report.md#6-sim-주행-결과) |
| [7. 연동 기능 검증](#7-연동-기능-검증) | 토픽, 좌표계, 투영, PID, gdm 연결 | [§7](report.md#7-연동-기능-검증-결과) |
| [8. vln-deploy 서비스 옵션](#8-vln-deploy-서비스-옵션) | S1/S2 서비스 인자 | — |
| [9. 주의](#9-주의) | | — |

---

## 0. 한 번에 실행

전제: gd_vln 컨테이너가 떠 있어야 한다. 없으면 [1-3](#1-3-gd_vln-컨테이너)을 먼저 실행한다.

```bash
# [호스트] Mode A, 노은역, 경량 기본 구성(S2 GGUF Q8_0 + S1 TRT, async)
scripts/sim/run_vln_sim.sh -m a --map noeun --gdm "Walk straight and stop at the ticket gate."
# [호스트] Mode B, ETRI 실내 (기본값: -m b, --map etri_indoor → sim indoor)
scripts/sim/run_vln_sim.sh --gdm
```

tmux 세션 `VLN-SIM` 하나를 6칸으로 나눠 띄운다. 칸을 닫아도 sim과 모델 스택은 남는다.

| 동작 | 명령 |
|---|---|
| 끄기 | `scripts/sim/run_vln_sim.sh stop --gdm` (stop 칸에 입력되어 있어 Enter만 누르면 된다) |
| 지시 변경 | `scripts/sim/run_vln_sim.sh instr "turn left and go to the sofa"` (`""`이면 정지). 웹 입력과 같다 |
| 도움말 | `scripts/sim/run_vln_sim.sh -h` |

### 0-1. 이 한 줄이 실행하는 명령

`scripts/sim/run_vln_sim.sh --gdm "<지시>"`(Mode B, ETRI 실내, 기본 구성) 기준이다. `[gd_vln]` 칸은 호스트에서 `docker exec -it -w /ws/src/InternNav internnav-habitat bash -c '…'`로 감싸서 실행한다.

| 칸 | 위치 | 실행 명령 |
|---|---|---|
| sim | 호스트 | `cd ~/gd_project/modules/sim_docker && GUIDEDOG_HUNAV=0 ./run_rbq10_etri_sim.sh indoor` (isaac-sim에서 `run_rbq10_sim.py`가 이미 돌고 있으면 건너뛴다) |
| server | 호스트 | `S1=trt S2=gguf S2_QUANT=bf16 scripts/sim/run_vln_stack.sh start /checkpoints/InternVLA-N1-DualVLN async && tail -F scripts/sim/verify/out/live/stack_{s2,s1,dual}.log` |
| bridge | gd_vln | `. scripts/sim/setup_ros_ws.sh && until curl -sf localhost:5801/health; do sleep 5; done; python3 scripts/sim/vln_bridge_node.py --ros-args -p use_sim_time:=true -p mode:=trajectory -p dry_run:=false -p log_dir:=$LOG -p instruction:="<지시>"` |
| web | gd_vln | `. scripts/sim/setup_ros_ws.sh && python3 scripts/sim/web_viz.py --ros-args -p use_sim_time:=true -p port:=8090 -p map:=scripts/sim/maps/etri_indoor` |
| gpu | gd_vln | `nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 5 \| tee $LOG/gpu.csv` |
| stop | 호스트 | `scripts/sim/run_vln_sim.sh stop --gdm` (입력만 해 두고 실행하지 않는다) |
| gdm (별도 터미널 창, DISPLAY가 없으면 tmux 창 `VLN-SIM:gdm`) | 호스트 → gdm | `docker exec -it gdm_docker-gdm-1 bash -lc 'tmux kill-session -t GDM-LPP; until timeout 3 ros2 topic echo --once /clock; do sleep 2; done; ./lpp-sim-vln'` (Mode A는 `./lpp-sim-travmap`) |

- `$LOG` = `scripts/sim/verify/out/<yymmdd_HHMMSS>_<mode>_<태그>/`. bridge, web, gpu 로그가 여기에 쌓인다.
- 시작 전에 이전 실행의 bridge, web, server 프로세스를 정리한다. 남아 있으면 goal과 cmd_vel이 두 번 발행되기 때문이다.
- `stop --gdm`은 sim 프로세스(`pkill -INT -f run_rbq10_sim.py`), 모델 스택(`run_vln_stack.sh stop`), bridge/web, gdm tmux 세션 `GDM-LPP`, `VLN-SIM` 세션을 차례로 끈다.

### 0-2. 옵션

| 옵션 | 값 (기본) | 바뀌는 것 |
|---|---|---|
| `-m`, `--mode` | `a` \| `b` (b) | a = `pixel_goal`: S2 pixel goal → gdm planner (`./lpp-sim-travmap`). b = `trajectory`: S1 경로 → MPC (`./lpp-sim-vln`) |
| `-b`, `--backend` | `single` \| `hf` \| `gguf` (gguf) | single = `vln_server.py` 단일 서버(HF, 동기). hf = S2 HF + S1 분리. gguf = S2 GGUF Q8_0 + S1 분리 |
| `--s1` | `trt` \| `hf` (trt) | hf/gguf의 S1. trt와 gguf는 dualvln 전용이다 |
| `--async` / `--sync` | (async) | dual_server 실행 방식 |
| `--model` | `dualvln` \| `navdp` \| `wodagger` (dualvln) | 체크포인트 |
| `-q`, `--quant` | `bf16` \| `nf4` \| `int8` (bf16) | single/hf의 S2 양자화 |
| `--map` | `noeun` \| `etri` \| `<maps 이름>` \| `<경로 prefix>` \| `none` (etri_indoor) | 웹 배경 지도. `--sim`을 따로 주지 않으면 sim 장면도 이 값으로 정한다(noeun → noeun, etri_indoor → indoor, 그 밖 → none) |
| `--sim` | `indoor` \| `1f` \| `outdoor` \| `noeun` \| `worldcup` \| `none` | sim 장면. `none`이면 sim은 따로 띄운다 |
| `--hunav` | | HuNavSim 보행자 (`GUIDEDOG_HUNAV=1`) |
| `--lookdown` | | 30° look-down 카메라를 S2 "↓" 입력으로 쓴다. sim에 `GUIDEDOG_LOOKDOWN_CAM=1`, bridge에 `-p lookdown_ns:=/camera_lookdown/camera` |
| `--dry` | | goal/cmd_vel을 내지 않고 계산과 시각화만 한다 |
| `--gdm` | (off) | gdm도 띄운다 |

- 예전 형식 `run_vln_sim.sh pixel_goal dualvln bf16 "<지시>" [dry]`은 `-b single`로 동작한다.
- 컨테이너 안에서도 실행할 수 있다. 다만 docker가 없어 sim과 S1 TRT·S2 GGUF 스택은 띄우지 못한다. `single`과 `-b hf --s1 hf`(`run_dual.sh`)만 직접 띄우고, 그 밖의 구성은 `:5801`에 이미 떠 있는 스택을 쓴다.

---

## 1. 따로 띄우기

[0](#0-한-번에-실행)을 쓰지 않을 때, 또는 §6처럼 bridge 없이 sim과 gdm만 필요할 때 쓴다.

### 1-1. sim
```bash
# [호스트] 포어그라운드로 계속 떠 있다. 장면 로딩에 약 2분 걸린다. 인자 indoor|1f|outdoor|noeun
cd ~/gd_project/modules/sim_docker && ./run_rbq10_etri_sim.sh indoor
```
- 노은역: `GUIDEDOG_BACKGROUND_USD=/isaac-sim/workspace/NoeunStation/noeun_station_collision.usdz ./run_rbq10_etri_sim.sh noeun`. `NoeunStation/`에는 `noeun_station_nurec.usdz`·`noeun_station_mesh.obj`가 없어서 이 값을 주지 않으면 "배경 파일이 없습니다"로 멈춘다(`run_vln_sim.sh`는 자동으로 넣는다)
- 보행자를 넣으려면 `./run_rbq10_*_dynamic_sim.sh`
- 30° look-down 카메라: `GUIDEDOG_LOOKDOWN_CAM=1 ./run_rbq10_etri_sim.sh indoor`

### 1-2. gdm
```bash
# [호스트] 컨테이너 셸
cd ~/gd_project/modules/gdm_docker && ./run.sh
# [gdm] Mode A: planner가 trav_map의 elevation 레이어를 쓴다
./lpp-sim-travmap
# [gdm] Mode B: planner 없음
./lpp-sim-vln
# [gdm] 끄기 (모드를 바꿀 때도 먼저 끈다)
./lpp-kill
```
- 셸에 들어가지 않고 Mode B를 띄울 때: `docker exec gdm_docker-gdm-1 bash -lc 'cd /home/gdr/gdm/workspace && ./lpp-sim-vln'` (tmux 세션 `GDM-LPP`를 백그라운드로 만든다)
- `./lpp-sim`은 쓰지 않는다. clustering3d 맵에는 elevation이 없어 `Layer elevation not found`가 반복된다.

### 1-3. gd_vln 컨테이너
```bash
# [호스트] 컨테이너가 있으면
docker start internnav-habitat
# [호스트] 셸
docker exec -it -w /ws/src/InternNav internnav-habitat bash
```
컨테이너가 없을 때 (이미지가 없으면 빌드한다):
```bash
docker/run-torch2.9.0-cuda13.0-habitat-jazzy-5090.sh -d
```
- ROS 메시지(gd_ifc_pkg) 빌드와 환경 설정: [gd_vln] `source scripts/sim/setup_ros_ws.sh`

### 1-4. isaacsim 셸 (빌드, 검증용)
```bash
# [호스트] vln-deploy는 root 소유라 -u 0으로 들어간다
docker run --rm -it -u 0 --entrypoint bash --gpus all --runtime=nvidia --network=host --ipc=host -v ~/gd_project/modules/gd_vln/workspace:/ws -v /media/LTDataset/checkpoints:/ws/src/InternNav/checkpoints:ro -v /media/TrainDataset:/ws/src/InternNav/data:ro -w /ws/src/InternNav/vln-deploy dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090
```

### 1-5. 준비 확인
```bash
# [호스트] 3줄이 나오면 sim이 준비된 것이다
docker exec internnav-habitat bash -c 'cd /ws/src/InternNav && source scripts/sim/setup_ros_ws.sh >/dev/null; ros2 topic list | grep -E "^/clock|camera/camera/color/image_raw|gdq_odom"'
```

---

## 2. 모델 스택

`scripts/sim/run_vln_stack.sh`. 모두 [호스트]에서 실행한다.

| 명령 | 구성 |
|---|---|
| `S2=gguf scripts/sim/run_vln_stack.sh start` | **경량 기본 구성**: S2 GGUF Q8_0 + S1 TRT 10 step, async |
| `S2=gguf GGUF_TEXT=quantization_kit/out/gguf/InternVLA-S2-text-f16-latent.gguf scripts/sim/run_vln_stack.sh start` | S2 GGUF FP16 |
| `S2=gguf GGUF_TEXT=quantization_kit/out/gguf/InternVLA-S2-text-Q4_K_M-latent-embf16.gguf scripts/sim/run_vln_stack.sh start` | S2 GGUF Q4 (메모리 최소, 로봇 명령 일치 95.4%) |
| `S1=hf S2=gguf scripts/sim/run_vln_stack.sh start` | S2 GGUF Q8_0 + S1 PyTorch bf16 |
| `S1=hf scripts/sim/run_vln_stack.sh start` | **기준선**: S2 HF bf16 + S1 PyTorch bf16 |
| `scripts/sim/run_vln_stack.sh start` | S2 HF bf16 (env 기본 `S2=hf`) + S1 TRT |
| `scripts/sim/run_vln_stack.sh start /checkpoints/InternVLA-N1-DualVLN sync` | sync 모드 (단일 서버와 같은 실행 순서) |
| `scripts/sim/run_vln_stack.sh stop` | 전체 정지 |

- 인자: `<start|stop> [model_path] [async|sync]`
  - `model_path` 기본값은 `/checkpoints/InternVLA-N1-DualVLN`이다.
  - w-NavDP와 wo-dagger는 `S1=hf`로 실행한다. TRT 엔진은 DualVLN 전용이다.
- 환경변수:
  - `S2=hf|gguf`
  - `S2_QUANT=bf16|nf4|int8` (hf)
  - `GGUF_TEXT` / `GGUF_MMPROJ` (gguf, 기본값은 `quantization_kit/out/gguf/`의 Q8_0 decoder + mmproj F16)
  - `S2_RESIZE=384`
  - `S2_HF_PARITY=1`: HF와 같은 입력과 생성 방식을 쓴다. 0이면 기존 112 정렬 경로다. 맞추는 항목은 [report.md §5-3](report.md#5-3-s2s1-전체)에 있다.
  - `S2_EMBD_CACHE=512`: vision 임베딩 캐시. 출력은 bit 단위로 같고 S2가 약 100 ms 빨라진다. 0이면 끈다.
  - `S1=trt|hf`, `S1_STEPS=10`
  - `MPS_S2` / `MPS_S1`: SM %, 0이면 끈다. `S1=hf`일 때만 쓸 수 있고, bit 단위 재현이 깨진다
  - `S1_CTR` / `S1_HEALTH_PORT`: 남아 있는 S1 컨테이너와 이름·포트가 충돌할 때만 바꾼다
- 로그: `scripts/sim/verify/out/live/stack_{s2,s1,dual}.log`

---

## 3. 로봇 연결과 시각화

| 위치 | 명령 | 설명 |
|---|---|---|
| gd_vln | `source scripts/sim/setup_ros_ws.sh && python3 scripts/sim/vln_bridge_node.py --ros-args -p use_sim_time:=true -p mode:=trajectory -p log_dir:=scripts/sim/verify/out/live/web_bridge` | **Mode B** bridge (gdm `./lpp-sim-vln`). 지시문은 웹이나 `/vln/instruction`으로 받는다 |
| gd_vln | `source scripts/sim/setup_ros_ws.sh && python3 scripts/sim/vln_bridge_node.py --ros-args -p use_sim_time:=true -p mode:=pixel_goal -p instruction:="Walk straight down the hallway and stop at the intersection."` | **Mode A** bridge (gdm `./lpp-sim-travmap`) |
| gd_vln | `python3 scripts/sim/web_viz.py --ros-args -p use_sim_time:=true -p port:=8090 -p map:=scripts/sim/maps/etri_indoor` | 웹 모니터 `http://<호스트 IP>:8090`. `?snap=1`은 저대역폭 |
| 호스트 | `scripts/sim/rviz/open_rviz.sh` | rviz2 (gdm 컨테이너에서 띄운다) |
| gd_vln | `ros2 topic pub --once /vln/instruction std_msgs/String "{data: 'Walk forward and stop in front of the printer on your right.'}"` | 지시문 변경. `''`이면 정지 |
| gd_vln | `python3 scripts/sim/maps/make_topdown_map.py <mesh.obj> scripts/sim/maps/<name> --floor_z <바닥 z>` | 웹 top-down 지도 만들기 |

- bridge 파라미터: `mode`(trajectory|pixel_goal), `server_url`(기본 `http://127.0.0.1:5801`), `dry_run`, `log_dir`, `lookdown_ns`. 나머지 기본값은 [README.md](README.md#bridge-주요-파라미터-2026-10-05-live-검증으로-정한-기본값)

---

## 4. 모델 빌드

최초 1회만 한다. 결과물은 이미 있다. 위치는 모두 [isaacsim], `cd /ws/src/InternNav/vln-deploy` 후 실행한다. 처음부터 재현하는 전체 절차는 [vln-deploy/docs/command.md](../../vln-deploy/docs/command.md)에 있다.

| 대상 | 명령 | 설명 |
|---|---|---|
| S1 | `quantization_kit/convenv_x86/bin/python deploy/tools/export_s1_onnx.py --ckpt checkpoints/InternVLA-N1-DualVLN --out quantization_kit/out/onnx` | ONNX export: 1 step denoise 네트워크, DINOv2, memory encoder, resampler, head. `--only`로 일부만 export한다 |
| S1 | `quantization_kit/convenv_x86/bin/python quantization_kit/verify_trt_repro.py trt` | TRT FP16 엔진 빌드(`out/trt_engines/`, 이미 있으면 재사용)와 비교 |
| S2 | `bash quantization_kit/tools/build_s2_gguf.sh [ckpt] [out_dir]` | GGUF 빌드: strip → text f16 → mmproj f16 → latent 4행 주입 → decoder Q4_K_M(token_embd/output F16). 기본 출력 `quantization_kit/out/gguf/` |
| S2 | `LD_LIBRARY_PATH=quantization_kit/llama.cpp/build_x86/bin quantization_kit/llama.cpp/build_x86/bin/llama-quantize --token-embedding-type f16 --output-tensor-type f16 quantization_kit/out/gguf/InternVLA-S2-text-f16-latent.gguf quantization_kit/out/gguf/InternVLA-S2-text-Q8_0-latent-embf16.gguf Q8_0` | decoder Q8_0 (약 10 s) |

- S2 산출물 (`quantization_kit/out/gguf/`):
  - `InternVLA-S2-text-Q8_0-latent-embf16.gguf` (9.1 GB, 기본)
  - `InternVLA-S2-text-f16-latent.gguf` (15.2 GB)
  - `InternVLA-S2-text-Q4_K_M-latent-embf16.gguf` (6.1 GB)
  - `InternVLA-S2-mmproj-f16.gguf` (1.35 GB)
- llama.cpp 빌드: [isaacsim] `cmake --build build_x86 --target llama-s2-runner -j 16`. 패치 스크립트는 `scripts/sim/_staging/patch_*.py`

---

## 5. dataset 검증

VLN-CE r2r val 녹화 6 ep × 40 step(`/media/TrainDataset`)을 재생해 경량 구성과 원본을 비교한다.
- [gd_vln] (`vln-deploy/`) 표기는 gd_vln 컨테이너에서 `cd vln-deploy` 후 실행한다는 뜻이다.
- [isaacsim] 명령은 모두 `cd /ws/src/InternNav/vln-deploy` 후 실행한다.

### 5-1. S1 단독
→ 결과: [report.md §5-1](report.md#5-1-s1-단독)

| 위치 | 명령 | 설명 |
|---|---|---|
| gd_vln | `/usr/bin/python3 vln-deploy/quantization_kit/verify_trt_repro.py ref` | 원본 PyTorch S1(bf16/fp32, 10/5 step)을 64 case에 고정 noise로 실행해 기준을 만든다 |
| isaacsim | `quantization_kit/convenv_x86/bin/python quantization_kit/verify_trt_repro.py trt` | TRT와 원본·PyTorch 비교. 결과는 `quantization_kit/verify_trt_repro_out/report.json` |

### 5-2. S2 단독
→ 결과: [report.md §5-2](report.md#5-2-s2-단독)

| 위치 | 명령 | 설명 |
|---|---|---|
| gd_vln | `python3 vln-deploy/quantization_kit/verify_s2_gguf.py hf --resize 384 --tag hf_bf16_384` | HF bf16 기준(5 step마다 S2, 48회). `--quant fp16`이면 HF fp16 참고값이다. `VLN_HF_ATTN=sdpa`이면 attention 커널만 바꾼다 |
| isaacsim | `bash quantization_kit/run_verify_s2_gguf.sh` | GGUF FP16·Q8_0·Q4 replay(HF parity, 캐시 512). 태그 `hfp_gguf_{f16,q8,q4}` |
| isaacsim | `quantization_kit/convenv_x86/bin/python quantization_kit/verify_s2_gguf.py gguf --hf_parity --resize 384 --embd_cache 512 --tag <이름> --text <text.gguf> --mmproj <mmproj.gguf>` | 단일 모델 replay. `--embd_cache 0`이면 캐시를 끈다 |
| isaacsim | `bash quantization_kit/run_verify_s2_embdcache.sh` | 캐시 on/off 출력 동일성 |
| gd_vln | `python3 vln-deploy/quantization_kit/verify_s2_gguf.py cmp --ref hf_bf16_384` | 기준 대비 행동·텍스트·pixel, latent cos, S1 경로 차(각 latent를 HF S1에 넣음), 정답 근접률. 결과는 `verify_s2_gguf_out/report_vs_hf_bf16_384.json` |
| isaacsim → gd_vln (`vln-deploy/`) | `quantization_kit/convenv_x86/bin/python quantization_kit/verify_vision_embd.py gguf` → `/usr/bin/python3 quantization_kit/verify_vision_embd.py hf` → `python3 quantization_kit/verify_vision_embd.py cmp` | 같은 입력에서 vision encoder 출력 HF vs llama.cpp |
| — | `S2_DEBUG=1` | runner 디버그: 프롬프트 토큰 ID, 생성 토큰, top-3 logit을 stderr로 출력 |

### 5-3. S2+S1 전체
→ 결과: [report.md §5-3](report.md#5-3-s2s1-전체)

| 위치 | 명령 | 설명 |
|---|---|---|
| gd_vln (`vln-deploy/`) | `/usr/bin/python3 quantization_kit/verify_s1s2_full.py ref` | 각 S2 replay의 latent마다 S1 PyTorch bf16 실행(S2 프레임 + 다음 4 프레임 = 165회, case별 고정 noise). `--noise_salt 1`이면 다른 noise(기준선) |
| isaacsim | `quantization_kit/convenv_x86/bin/python quantization_kit/verify_s1s2_full.py trt` | 같은 입력으로 S1 TRT 실행(sim 배포 구성: 재현 엔진 + DINOv2/cond TRT + CPU PIL 전처리) |
| 아무 곳 (`vln-deploy/`) | `python3 quantization_kit/verify_s1s2_full.py cmp` | {HF, GGUF} × {PyTorch, TRT}를 HF+PyTorch와 비교. `--test_tags`(기본 `hfp_gguf_q8 hfp_gguf_f16 hfp_gguf_q4`). 결과는 `verify_s2_gguf_out/report_s1s2_full.json` |

### 5-4. S1/S2 분리 구조
→ 결과: [report.md §5-4](report.md#5-4-s1s2-분리-구조)

| 위치 | 명령 | 설명 |
|---|---|---|
| gd_vln | `scripts/sim/verify/run_dual_bench.sh /checkpoints/InternVLA-N1-DualVLN` | 단일 서버와 분리 sync/async/MPS 비교: 동일성과 지연(HF backend, 240 요청) |
| gd_vln | `python3 scripts/sim/verify/bench_dual.py run --url http://127.0.0.1:5801 --tag <이름> --period 0.3` / `... cmp <A>.npy <B>.npy` | 떠 있는 서버에 재생하고 두 실행을 비교한다. `--period 0`은 연속 요청 |

---

## 6. sim 주행 검증

ETRI 실내, 장애물 없는 구간 5개(`scripts/sim/verify/e2e_episodes_etri_indoor_clear.json`):
- c1 프린터
- c2 마커
- c3 교차로
- c4 프린터 지나 교차로
- c5 중간 출발

성공 = 모델이 STOP을 내고 목표에서 반경 이내(c1·c2 1.0 m, c3~c5 1.5 m).

준비:
1. sim과 gdm은 [1-1](#1-1-sim), [1-2](#1-2-gdm)로 띄운다. Mode B는 `./lpp-sim-vln`, Mode A는 `./lpp-sim-travmap`이다.
2. gd_vln 컨테이너를 띄우고 [1-5](#1-5-준비-확인)로 확인한다.
3. bridge는 띄우지 않는다. `run_e2e.py`가 직접 지시를 보낸다.

### 6-1. 구성별 SR (Mode B)
→ 결과: [report.md §6-1](report.md#6-1-구성별-sr-mode-b)

| 대상 | 위치 | 명령 | 설명 |
|---|---|---|---|
| S2 비교 | 호스트 | `bash scripts/sim/verify/run_s2_sr.sh 4 gguf_q8 gguf_f16 gguf_f16_pt gguf_repro` | 1번째 인자는 반복 횟수, 이후는 구성 목록: `hf_bf16`, `hf_bf16_pt`(기준선), `gguf_q8`, `gguf_f16`, `gguf_f16_pt`(S1 PyTorch), `gguf_repro`(Q4), `gguf_orig`(원본 vln-deploy GGUF). S1은 `_pt`를 빼면 TRT. 구성마다 스택을 띄우고 끝나면 끈다 |
| S1 비교 | 호스트 | `bash scripts/sim/verify/run_trt_sr.sh 2` | S2 HF 고정. S1은 PyTorch bf16 / TRT 10 / TRT 5 step |
| 요약 | gd_vln | `python3 scripts/sim/verify/summarize_e2e.py scripts/sim/verify/out/e2e --match <태그>` | 에피소드 결과 표. 예: `--match s2sr_gguf_q8_rep` |
| 속도 | gd_vln | `scripts/sim/verify/run_live_speed.sh` | 40 s 직진: 로봇 속도, RTF, 응답 간격 |

- 결과 폴더: `scripts/sim/verify/out/e2e/<태그>/`, 지연 로그 `scripts/sim/verify/out/live/s2sr_<구성>_dual.log`

### 6-2. Mode A / Mode B 직접 실행
→ 결과: [report.md §6-2](report.md#6-2-mode-a와-mode-b)

```bash
# [호스트] 스택 (경량 기본 구성)
S2=gguf scripts/sim/run_vln_stack.sh start
# [gd_vln] Mode A (gdm: ./lpp-kill; ./lpp-sim-travmap)
python3 scripts/sim/verify/run_e2e.py --mode pixel_goal --tag <태그> --episodes scripts/sim/verify/e2e_episodes_etri_indoor_clear.json
# [gd_vln] Mode B (gdm: ./lpp-kill; ./lpp-sim-vln)
python3 scripts/sim/verify/run_e2e.py --mode trajectory --tag <태그> --episodes scripts/sim/verify/e2e_episodes_etri_indoor_clear.json
```
- 결과 폴더: `scripts/sim/verify/out/e2e/<태그>/<mode>/`
- `--inspect`를 붙이면 에피소드마다 `inspect.json`과 `inspect.png`를 만든다([7](#7-연동-기능-검증) `live_inspect.py`)

---

## 7. 연동 기능 검증

→ 결과: [report.md §7](report.md#7-연동-기능-검증-결과)

| 스크립트 | 확인하는 것 |
|---|---|
| `verify/verify_quant.py` | S2/S1/dual의 VRAM, 지연, 정확도(vs bf16, vs VLN-CE GT) |
| `verify/verify_sensors.py check:={rates,geom,twist,sync}` | V0 토픽 / V2·V4·V6 카메라·depth 기하 / V7 twist / V8 동기 |
| `verify/verify_pid.py` | discrete PID 정확도 |
| `verify/run_e2e.py --mode {pixel_goal,trajectory} --tag <모델_quant> --episodes verify/e2e_episodes_etri_indoor.json` | 주행 평가 |
| `verify/plumbing_test.sh` | Isaac Sim 없이 fake_sim으로 배선 확인 |
| `verify/frame_test.sh` | map≠odom 좌표계 일관성 (fake_sim `map_odom` + `stub_server.py` + `frame_check.py`) |
| `verify/overlay_test.sh` | 고정 pixel 5곳의 초록 원 위치와 기준 프레임, 정면/30° (`overlay_check.py`) |
| `verify/proj_check.py` + `proj_summary.py` | 주행 중 S2 pixel goal 3D 점의 depth vs 같은 광선의 라이다 depth(평면 맞춤), 정면/30° |
| `verify/gdm_test.sh [초] [lookdown 0\|1]` | 실제 sim+gdm에서 pixel 1개 → gp 해석, gdm 계획, 도착, cmd_vel 발행자 (`gdm_check.py`) |
| `verify/live_inspect.py` | 주행 중 S1 경로/gdm 계획 → cmd_vel → 로봇을 기록·판정: cross-track, v/w 제한, w≈v·κ, 실제 속도(pose 차분), 발행자 수, 정지 위치. `run_e2e.py --inspect`로 에피소드마다 실행 |
| `verify/dataset_test.sh [ep수] [tag]` | 배포 스택에 sync dual_server(:5894)를 붙여 VLN-CE 에피소드 재생 → S2 pixel·동작, S1 경로를 학습 GT와 비교(`dataset_check.py`). :5801 dual_server를 먼저 내려야 한다(ZMQ PUSH/PULL 공유) |
| `python3 -m pytest scripts/sim/tests` | 단위 테스트 (projection, control_utils) |

- 경로는 `scripts/sim/` 기준이다. 결과는 `scripts/sim/verify/out/`에 쌓인다.

---

## 8. vln-deploy 서비스 옵션

기존 기본값을 그대로 둔다.

| 서비스 | 옵션 | 설명 |
|---|---|---|
| S2 `deploy/s2_module/run.sh` | `--s2-backend gguf\|hf` | gguf(기본) = llama-s2-runner, hf = transformers |
| | `--gguf-text`, `--gguf-mmproj`, `--resize 384 --hf-parity`, `--embd-cache 512` | [gguf] 모델 파일, HF 동일 입력·생성, 임베딩 캐시. 캐시는 기본 0(off)이고 sim 스택은 512 |
| | `--hf-model-path`, `--hf-quant`, `--hf-resize` | [hf] |
| S1 `deploy/s1_module/run.sh` | `--s1-backend lite\|hf`, `--s1-steps 10` | lite = S1LiteModel + TRT |
| | `--trt-onnx-dir`, `--trt-engine-dir` | 원본 `onnx`/`trt_engines`, 재현본 `out/onnx`/`out/trt_engines` |
| | `--no-trt`, `--no-gpu-prep`, `--rgb-cache` | PyTorch fp32 / CPU PIL 전처리 / DINOv2 캐시 |
| interpreter | `VLN_PY=/usr/bin/python3` | HF backend는 gd_vln 이미지의 python을 쓴다 |

---

## 9. 주의
- S1 TRT 컨테이너가 시작 직후 CUDA 초기화 전에 멈춘 적이 있다. kill이 안 되고, 다시 시도하면 정상으로 뜬다. 이름이나 포트가 충돌하면 `S1_CTR=vln-s1-trt2 S1_HEALTH_PORT=5624`로 우회한다.
- 측정 중에는 bridge(지시문 송신 클라이언트)가 하나만 떠 있어야 한다. dual_server 로그의 reset 횟수가 에피소드 수와 같아야 한다.
- MPS는 결과 재현성을 깨므로 평가에서는 끈다.
- `S2_HF_PARITY=0`(기존 112 정렬 경로)과 원본 vln-deploy GGUF(`quantization_kit/gguf/`)는 HF와 결과가 같지 않다. 평가에 쓰지 않는다.
- DDS: gd_vln 컨테이너(root)와 sim/gdm(uid 1000) 사이에서는 Fast DDS SHM으로 데이터가 오지 않는다. 그래서 `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`를 쓴다(`setup_ros_ws.sh`에 설정됨).
