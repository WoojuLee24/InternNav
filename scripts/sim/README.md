# scripts/sim — InternVLA-N1 ↔ Isaac Sim(RBQ10) ↔ gdm 연동

계획: `~/.claude/plans/toasty-humming-quilt.md`. realworld 코드(`scripts/realworld/`)는 수정하지 않고 파생했다.

| 파일 | 역할 |
|---|---|
| `vln_server.py` | Flask 모델 서버 (`/eval_dual`, `/health`). `--quant bf16\|nf4\|int8`, instruction은 요청마다 받는다 |
| `vln_agent.py` | `InternVLAN1AsyncAgent` 상속. quant 로딩, 프레임별 debug 저장 제거, S1 depth clip(5 m) |
| `quant.py` | bitsandbytes 설정. LLM decoder(`model.layers`)만 양자화하고 vision과 S1은 bf16 유지 |
| `projection.py` | crop/resize와 K 변환, depth median, ground-ray, 궤적 투영 (순수 numpy) |
| `vln_bridge_node.py` | ROS2 bridge. Mode A `pixel_goal`, Mode B `trajectory` |
| `run_vln_sim.sh` | 한 번에 실행하는 tmux 런처 (sim / server / bridge / web / gpu, `--gdm`이면 gdm도) |
| `run_vln_stack.sh` | 모델 스택 (S2 + S1 + `dual_server`) 시작/정지 |
| `dual_server.py` | HTTP ↔ ZMQ 어댑터 (:5801) |
| `web_viz.py`, `rviz/` | 웹 모니터, rviz2 설정 |
| `maps/` | 웹 top-down 지도와 생성 스크립트 |
| `_staging/` | vln-deploy·llama.cpp 패치 스크립트, 원본 백업 |
| `setup_ros_ws.sh` | gd_ifc_pkg(gdm 메시지) 빌드와 source |
| `tests/` | 단위 테스트 (`python3 -m pytest scripts/sim/tests`) |
| `verify/` | 검증 스크립트 V0~V8, 결과는 `verify/out/` |

## 문서
| 문서 | 내용 |
|---|---|
| README.md (이 문서) | 파일, 토픽, bridge 파라미터, 시각화, 모델 설명 |
| [command.md](command.md) | **실행·검증 명령.** 맨 위 "한 번에 실행"이 sim + gdm + 모델 스택 + bridge를 한 줄로 띄우고, 그 한 줄이 실행하는 명령을 칸별로 적었다 |
| [report.md](report.md) | 측정 결과. 절 번호가 command.md와 같다 (명령 §5 → 결과 §5) |

한 번에 실행 (gd_vln 컨테이너가 떠 있어야 한다. [command.md §1-3](command.md#1-3-gd_vln-컨테이너)):
```bash
scripts/sim/run_vln_sim.sh -m a --map noeun --gdm "walk forward to the door and stop"
```
옵션과 이 한 줄이 실행하는 명령: [command.md §0](command.md#0-한-번에-실행)

## 토픽
- 입력: `/camera/camera/color/image_raw`, `/camera/camera/aligned_depth_to_color/image_raw`, `/camera/camera/color/camera_info`, `/gdq/msg/gdq_odom`(pose만 사용, 로봇 위치), `/tf`(base_link←카메라만)
- 출력 좌표: `/vln/trajectory`, `/vln/goal_marker`는 `odom` frame
- Mode A 출력: `/gdg/data/gp` (GDGGuidancePoint, VIA_GP. STOP이면 FORCE_TO_STOP_GP). 회전·전진 discrete 동작은 PID로 `/cmd_vel`에 낸다(planner보다 우선)
- Mode B 출력: `/gdq/msg/cmd_vel` (MPC, v,w≤0.4. 서버 응답이 3 s 이상 없으면 정지 — sim 에서 S2 1회가 약 2 s)
- 시각화: `/vln/pixel_goal_image`(초록 원 = pixel goal(640x480 입력 프레임 좌표), 주황 = S1 궤적), `/vln/goal_marker`, `/vln/trajectory`, `/vln/s1s2_agreement`, `/vln/status`

## 주의
- odom twist는 쓰지 않는다. sim은 world frame으로, 실기는 0으로 나온다. PID의 Kd를 0보다 크게 주면 WARN을 띄운다.
- Mode B에는 장애물·주행가능 영역 검사가 없다. 추후 trav_map 기반으로 교체할 예정이다.

## bridge 주요 파라미터 (2026-10-05 live 검증으로 정한 기본값)
| 파라미터 | 기본값 | 이유 |
|---|---|---|
| `pose_source` | `odom` | 로봇 pose = `/gdq/msg/gdq_odom`을 이미지 stamp에서 보간. 실기 클라이언트·gdm `/gdg/data/gp`와 같은 odom 좌표계라 map≠odom이어도 목표가 맞다(`verify/frame_test.sh`). `tf`는 이전 방식(TF map←base_link) |
| `pixel_space` | `input` | S2 pixel goal은 서버에 보낸 640x480 프레임의 [row, col]이다. 384 공간이 아니다(V5) |
| `ground_z_auto` | true | 바닥 높이를 depth로 추정한다. ETRI indoor 실측 -0.476 m로, 고정값 -0.55와 다르다 |
| `goal_max_dist` / `goal_resend_s` | 6.0 m / 5 s | mesh 구멍 때문에 48 m goal이 생겼고, 첫 GP가 유실되는 문제가 있었다 |
| `discrete_block_max_s` | 12 s | discrete 회전·전진을 실행하는 동안 모델 질의를 보류한다 |
| `pid_v_min` / `pid_w_min` / `pid_lead_s` | 0.3 / 0.4 / 0.45 | 보행 dead zone과 관성 보상 (`verify/verify_pid.py`) |
| `server_timeout` | 3.0 s | sim에서 S2 1회가 약 2 s 걸린다 |
- DDS: 이 컨테이너(root)와 sim/gdm(uid 1000) 사이는 Fast DDS SHM으로 데이터가 오지 않아 `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`를 쓴다(이미지 ENV와 `setup_ros_ws.sh`에 설정됨).

## 검증과 결과
- 검증 스크립트 목록과 실행 명령: [command.md §5~7](command.md#5-dataset-검증)
- 측정 결과: [report.md](report.md)

## 시각화
### 웹 모니터 (`web_viz.py`, 포트 8090)
- 실행: `run_vln_sim.sh`의 web 칸에서 자동으로 뜬다. 따로 띄우는 명령은 [command.md §3](command.md#3-로봇-연결과-시각화)
- 접속: `http://<호스트 IP>:8090`. 저대역폭이나 스크린샷용으로는 `http://<호스트 IP>:8090/?snap=1`(MJPEG 대신 0.5 s마다 JPEG)
- 화면 구성:
  - **최신 S2 판단**: pixel goal(초록)과 S1 경로(파랑). "LD"가 붙으면 30° look-down 프레임 기준이다.
  - 실시간 overlay, FPV(14.8°), look-down(30°) 카메라
  - top-down 지도(`maps/<name>.png/.json`): 로봇, 주행 경로, S1 경로, gdm 계획, VLN goal
  - S2 답변 표: 텍스트, action/pixel, 프레임, goal, 지연
  - 지시문 입력(Enter/Send), STOP(빈 지시문 → bridge 정지), Reset(정지 + 시작 위치 0,0,-90° 로 teleport, `-p start_x/start_y/start_yaw_deg`), 좌표 teleport
- 지도 생성: `python3 scripts/sim/maps/make_topdown_map.py <mesh.obj> scripts/sim/maps/<name> --floor_z <바닥 z>`

### rviz2 (`rviz/vln.rviz`)
- 호스트에서 `scripts/sim/rviz/open_rviz.sh`를 실행한다. gd_vln 이미지에는 rviz2가 없어 gdm 컨테이너에 설정 파일을 복사해서 띄운다(gdm repo는 수정하지 않음).
- 표시: RobotModel, Ouster 점군, `/vln/robot_path`(web_viz가 발행), `/vln/trajectory`, `/gdm/local_plan_map`, `/vln/goal_marker`, 이미지 3개(VLN overlay, FPV, look-down)

## 30° look-down 카메라 (sim)
- sim: `GUIDEDOG_LOOKDOWN_CAM=1 ./run_rbq10_etri_sim.sh indoor`. 설정은 `sim_docker/workspace/config/lookdown_camera.yaml`(기존 camera_link를 기반으로 pitch만 30°로 변경), 코드는 `sim_docker/workspace/scripts/lookdown_camera.py`
- bridge: `-p lookdown_ns:=/camera_lookdown/camera` (`LOOKDOWN=1 scripts/sim/run_vln_sim.sh ...`). S2가 "↓"라고 답하면 서버가 30° 프레임으로 다시 질의하고, pixel goal은 30° 카메라 K/TF/depth로 투영한다(`pixel_frame=lookdown`).

## 모델
- `dualvln` = InternVLA-N1-DualVLN, `navdp` = InternVLA-N1-w-NavDP, `wodagger` = InternVLA-N1-wo-dagger. wodagger는 예전 "InternVLA-N1" 릴리스로, config에 `system1`이 없어 `quant.legacy_config`가 `navdp_async`로 지정한다. NavDP 텐서의 키와 shape가 w-NavDP와 동일하다.

## 경량화 관련 코드 위치
| 위치 | 파일 |
|---|---|
| `scripts/sim/` | `run_vln_stack.sh`(스택), `dual_server.py`(HTTP↔ZMQ 어댑터), `vln_bridge_node.py`, `web_viz.py`, `rviz/`, `verify/`(sim·dataset 검증), `quant.py`(HF 로더, `fp16`·`VLN_HF_ATTN` 옵션), `_staging/`(vln-deploy와 llama.cpp 패치 스크립트, 원본 백업) |
| `vln-deploy/` (root 소유) | `quantization_kit/async_zmq/s2_service.py`, `s2_policy_hf.py`, `deploy/s1_module/service{,_hf}.py`, `deploy/tools/export_s1_onnx.py`, `quantization_kit/tools/`(GGUF 빌드), `quantization_kit/verify_*.py` |
| `vln-deploy/quantization_kit/llama.cpp/tools/mtmd/` | opt-in 패치: `MTMD_QWEN25VL_HF_ALIGN`(clip.cpp, mtmd-image.cpp, models/qwen2vl.cpp), `S2_LATENT_WITH_EOS`·`S2_EMBD_CACHE`·`S2_HF_CHAT`·`S2_DEBUG`·`EMBD` 명령(s2-runner.cpp). 패치 스크립트는 `scripts/sim/_staging/patch_*.py`, 빌드는 isaacsim에서 `cmake --build build_x86 --target llama-s2-runner -j 16` |
| `internnav/model/basemodel/internvla_n1/` | `internvla_n1_policy_llamacpp.py`(GGUF S2, 캐시, BMP 정리, `_format_prompt` 훅, `repeat_last_n`), `internvla_n1_policy_llamacpp_hfparity.py`(HF 동일 입력·생성) |
