# bridge 좌표계를 odom 기준으로 변경 (2026-10-07)

## 무엇이 달라졌나
- `scripts/sim/vln_bridge_node.py`
  - 새 param `pose_source`(기본 `odom`), `odom_frame`(기본 `odom`), `odom_max_dt`(0.1 s)
  - `odom`: `/gdq/msg/gdq_odom` pose를 이미지 stamp에서 보간(앞뒤 두 odom, 위치 선형·회전 slerp)해 로봇 pose로 쓴다. 이미지보다 새 odom이 올 때까지 최대 0.2 s 기다린다. 실기 클라이언트(`scripts/realworld/http_internvla_client.py`)와 gdm `/gdg/data/gp`(odom pose 기준 상대 좌표)와 같은 좌표계.
  - `tf`: 이전 방식(TF `map_frame`←`base_link`). `-p pose_source:=tf`로 되돌릴 수 있다.
  - `/vln/trajectory`, `/vln/goal_marker`, `/goal_pose`의 frame_id = `odom_frame`(odom 모드). 코드 변수명의 "map"은 이 world 좌표계를 뜻한다.
  - CSV `tf_dt` 열 = odom 모드에서 보간에 쓴 odom과 이미지 stamp의 차이(s).
  - `publish_vis`: 자홍 십자(384 가설) 제거, 초록 원은 `pixel_to_proc` 좌표.
- `scripts/sim/verify/fake_sim.py`: param `map_odom:=[x, y, yaw_deg]` (map→odom 오프셋, 기본 0 = 기존과 동일)
- 새 파일: `verify/stub_server.py`(고정 응답 서버), `verify/frame_check.py`(판정), `verify/frame_test.sh`(실행)

## 명령
`docker exec internnav-gdsim bash -c 'cd /ws/src/InternNav && scripts/sim/verify/frame_test.sh 15'`
- 인자: case당 측정 시간(s). ROS_DOMAIN_ID 77(실제 sim과 분리), stub 포트 5891.
- 판정: goal = `/vln/goal_marker`를 실제 TF로 base에 옮긴 점 vs `/gdg/data/gp` 상대 위치. path = `/vln/trajectory`를 stamp 시점 base로 옮긴 점 vs stub 경로(x=0.1i). 기준 ≤1 cm.

## 결과 (verify/out/frame_261007_060315)
| pose_source | map→odom | goal | path |
|---|---|---|---|
| odom | 0 | 0.0 cm PASS | 0.1 cm PASS |
| odom | (2, -1, 30°) | 0.0 cm PASS | 21개 중 20개 ≤0.1 cm, 1개 1.1 cm |
| tf | 0 | 0.0 cm PASS | 0.0 cm PASS |
| tf | (2, -1, 30°) | **167.6 cm FAIL** | 0.0 cm PASS |

- 이전 방식(tf)은 map≠odom이면 gdm 목표가 167.6 cm 어긋남 → odom 방식에서 0 cm.
- path는 bridge 안에서 같은 좌표계로 계산·추종하므로 두 방식 모두 맞다.
- odom 방식 path 1.1 cm 1건: 0.2 s 안에 새 odom이 오지 않아 가장 가까운 odom을 쓴 경우로 추정(fake_sim 단일 스레드 부하). 보간 전(nearest)에는 최대 6.5 cm였다.

## 실제 sim 재확인 (2026-10-07, ETRI indoor clear 5 ep × 1회, S2 GGUF Q8_0 + S1 TRT async)
명령(gd_vln): `python3 scripts/sim/verify/run_e2e.py --mode <trajectory|pixel_goal> --tag odomframe_q8trt --episodes scripts/sim/verify/e2e_episodes_etri_indoor_clear.json` (이전 방식은 `--tag tfframe_q8trt --bridge_args "pose_source:=tf"`). gdm: Mode B `./lpp-sim-vln`, Mode A `./lpp-sim`.

| 방식 | mode | c1 | c2 | c3 | c4 | c5 | SR | pose 시각 일치 |
|---|---|---|---|---|---|---|---|---|
| odom (새) | trajectory | ✗ 2.9 | ✗ 4.8 | ✅ 0.7 | ✅ 1.0 | ✅ 0.8 | 3/5 | 704/704 |
| tf (이전) | trajectory | ✗ 3.4 | ✗ 5.3 | ✅ 0.7 | ✅ 1.0 | ✗ 11.7 | 2/5 | 635/695 |
| odom (새) | pixel_goal | ✗ 3.5 | ✗ 3.4 | ✅ 1.4 | ✅ 0.6 | ✗ 2.2 | 2/5 | 683/683 |
| tf (이전) | pixel_goal | ✗ 1.2 | ✗ 5.0 | ✅ 0.3 | ✅ 1.2 | ✗ 10.8 | 2/5 | 599/647 |

- 숫자는 멈춘 위치와 목표의 거리(m). 이전 기록(Q8_0+TRT Mode B 4회: c1·c2 항상 실패, c3·c4 항상 성공, c5 1/4)과 같은 패턴 → 새 bridge로 바꿔도 동작이 나빠지지 않았다. c5는 원래 결과가 흔들리는 에피소드라 1회 차이는 의미 없다.
- pose 시각 일치: odom 방식은 모든 질의에서 이미지 시각의 pose를 보간으로 구했다(보간 간격 p50 15 ms, 최대 65 ms). 이전 tf 방식은 7~9%에서 그 시각의 TF가 없어 최신 TF로 대신했다.
- Mode A goal 투영은 두 방식 모두 전부 depth(odom 112회, tf 105회).

## look-down 켬 경로 확인 (2026-10-07)
- 발견: 기본 스택 `dual_server.py`가 bridge가 보낸 30° 이미지를 버리고 있었다. S2 "↓" 재질의도 정면 프레임으로 했고 reply는 항상 `pixel_frame=fpv` → look-down 켬/끔이 같은 동작이었다(첫 실행 `odomframe_q8trt_ldon`: S2 119회 모두 fpv).
- 수정(`dual_server.py`): 요청의 `lookdown_image/lookdown_depth`를 받아 `S2Client.fire(lookdown_rgb=...)`로 넘긴다. S2가 "↓ > x y"로 답하면(s2_service가 재질의 시 "  >  "로 이어 붙임) goal 프레임을 30° rgb/depth로 두고 `pixel_frame=lookdown`으로 답한다. S1 기억 프레임도 30° 프레임(원본 agent·vln_server.py와 같음). 30° 이미지가 없으면 기존과 동일.
- 결과(`odomframe_q8trt_ldon2`, Mode A, clear 5 ep): ✗3.8 ✗4.9 ✅0.5 ✅1.0 ✗11.5 = 2/5. S2 답 125회 중 117회가 "↓" 후 30° 프레임 답, goal 117개 모두 `lookdown`/depth 투영, pose 시각 일치 718/718.
- 확인할 점(③에서): 30° 프레임 기준일 때 S1 경로 끝과 pixel goal 차이 중앙값 106 px(정면 프레임 21 px, 표본 5~6개). overlay에서도 S1 경로가 화면 아래쪽에 짧게만 보인다. S1이 30° 기억 프레임에서 짧은 경로를 내는지, 30° 카메라 투영이 틀린지 가려야 한다. 이전 단일 서버 결과도 look-down 켬에서 Mode B가 1/5(끔 2/5)였다.

## 30° 프레임에서 S1 경로가 짧은 원인 (2026-10-07)
- 30° 카메라 기하는 정상: `verify_sensors.py check:=geom camera_ns:=/camera_lookdown/camera` depth vs lidar p90 1.5 cm(정면 2.1 cm).
- bridge CSV에 `traj_end_x/y`(S1 경로 끝, 요청 시점 base) 열 추가 후 Mode A clear 5 ep × look-down 켬/끔(`s1len_ldon`/`s1len_ldoff`):

| | S2 목표 거리(바닥 모델) p50 | 같은 응답의 S1 끝 거리 p50 | pixel row p50 |
|---|---|---|---|
| 정면(끔) | 2.95 m | 2.88 m | 203 |
| 30°(켬) | 3.21 m | 0.72 m (p90 1.65) | 78 |

- → 투영 문제가 아니라 **S1이 30° 기억 프레임 + 정면 현재 프레임 조합에서 목표까지 가지 않는 짧은 경로를 낸다.**
- 학습(`internvla_n1_lerobot_dataset.py:1102-1153`): S1 `traj_images/traj_depths`는 기억·현재 프레임 모두 30°(pitch_2) 이미지·depth.
- 공식 habitat 평가(`habitat_extensions/vln/habitat_vln_evaluator.py:596-610, 701-750`): 매 스텝 LOOKDOWN×2로 30° 이미지를 찍어 S1의 기억·현재 프레임 둘 다 30° 이미지·depth로 준다.
- 현재 배포(`dual_server.py` → S1 `act(latent, g_rgb, rgb, ...)`, 원본 realworld agent도 같음): 현재 프레임은 항상 정면 이미지. look-down 끔이면 기억·현재 모두 정면(학습과 다르지만 일관), 켬이면 30°/정면이 섞임 → 경로가 짧아짐.

## 도구 1 확장 결과 (2026-10-07)
### ② 시각화: `verify/overlay_test.sh` (fake_sim + stub_server, 모델·Isaac 불필요)
- stub이 고정 pixel 5곳(중앙·네 모서리)을 차례로, 2프레임 늦은 프레임 기준(`--lag 2`)으로 답한다. 받은 이미지를 저장하고 `llm="fid=<id> x y"`로 기준 프레임을 알려준다. fake_sim `lookdown:=true`는 dataset 30° 영상을 30° 카메라로 발행.
- 판정(`overlay_check.py`): 초록 원 중심 vs pixel, overlay 배경 vs 기준 프레임(그림 픽셀 제외 평균 차).
- 결과(out/overlay_261007_115219): 정면 n=60, 30° n=56 모두 위치 오차 **0.00 px**, 기준 프레임 차 0.80(가장 최근 프레임과는 16.5) → PASS.
### ④ 전달: `verify/gdm_test.sh [초] [lookdown 0|1]` (실제 sim + gdm `./lpp-sim`, 모델 불필요)
- 매 경우 FORCE_TO_STOP → (0,0,−90°) teleport → stub `--once`가 pixel 1개만 답 → bridge Mode A → gdm. `gdm_check.py`가 잰다.
- 발견·수정: bridge가 gp 상대 좌표를 로봇 3D 자세(기울기 포함)로 계산해 gdm(평면: odom xy + R(yaw)·rel) 해석과 2.0 cm 차이 → 평면 계산으로 변경 후 0.0 cm.
- 결과(out/gdm_261007_121018, gdm_261007_121247): 6/6 PASS

| pixel (row,col) | 카메라 | gp rel (x 앞, y 왼쪽) | gp vs marker | 첫 계획 지연 | 도착 거리 |
|---|---|---|---|---|---|
| 400,320 | 정면 | (1.04, −0.01) | 0.0 cm | 0.90 s | 0.26 m |
| 330,200 | 정면 | (1.29, +0.29) | 0.0 cm | 0.72 s | 0.26 m |
| 330,440 | 정면 | (1.32, −0.31) | 0.0 cm | 0.55 s | 0.23 m |
| 400,320 | 30° | (0.76, −0.01) | 0.0 cm | 0.97 s | 0.21 m |
| 330,200 | 30° | (0.90, +0.21) | 0.0 cm | 0.79 s | 0.28 m |
| 330,440 | 30° | (0.91, −0.22) | 0.0 cm | 1.00 s | 0.20 m |

- 좌우 정상(왼쪽 pixel → y+, 오른쪽 → y−), cmd_vel 발행자 1개(gdm). 첫 계획 끝점은 목표에서 0.2~0.7 m(gdm 국소 계획), 최종 도착은 0.2~0.3 m.

## S1 입력을 학습·공식 평가와 맞춤 (2026-10-07)
- 변경(`dual_server.py`): 요청에 30° 이미지가 있으면 S1 기억 프레임 = S2 질의 시점의 30° rgb/depth, 현재 프레임 = 현재 30° rgb/depth(habitat_vln_evaluator.py `images_dp/depths_dp`와 같음). S2 history는 그대로 정면. pixel 투영 프레임(`pixel_frame`)은 "↓" 재질의 여부로 정한다. 30° 이미지가 없으면 이전과 동일(기억·현재 모두 정면).
- 결과(clear 5 ep × 1회, Q8_0 + TRT, look-down 켬, tag `s1ld_ldon`):

| | S2 목표 거리 p50 | S1 끝 거리 p50 (p10/p90) | Mode A | Mode B |
|---|---|---|---|---|
| 변경 전 (기억 30° + 현재 정면, `s1len_ldon`) | 3.21 m | 0.74 m (0.11/1.61) | 2/5 | (이전 단일 서버 1/5) |
| **변경 후 (기억·현재 모두 30°)** | 2.70 m | **2.47 m** (1.18/2.75) | **3/5** | **3/5** |
| 참고: look-down 끔 (모두 정면) | 2.95 m | 2.71 m | 2/5 | 3/5 (`odomframe_q8trt`) |

- S1 경로가 다시 S2 목표까지 이어진다. Mode B c3·c4·c5 성공(✗3.2 ✗5.5 ✅0.7 ✅0.9 ✅0.8), Mode A ✗2.4 ✗4.0 ✅0.2 ✅1.1 ✅0.7.
- 남은 불일치: look-down 끔(실기처럼 카메라 1대)에서는 S1이 정면 이미지를 받는다(학습은 30°). 단일 서버 경로(`vln_server.py`/`vln_agent.py`)도 아직 현재 프레임이 정면이다(기본 스택 아님).

## ③ 좌표 변환 검증 (2026-10-07)
### 방법: `verify/proj_check.py` (실제 sim 주행과 함께 실행) + `verify/proj_summary.py`
- S2 답마다(`/vln/s2` JSON에 `pixel_stamp`, `goal_xyz` 추가) 그 시각에 가장 가까운 라이다 점군을 world(TF, 점군 시각) → 그 카메라(TF, pixel 시각)로 옮겨 640x480 이미지에 투영한다(로봇 이동 보정).
- 목표 pixel 15 px 안의 라이다 점으로 평면 1/z = a·u + b·v + c를 맞추고(잔차 큰 점 2회 제거) **목표 pixel에서의 라이다 depth**를 읽어 bridge 3D 점의 depth와 비교한다. 라이다 점 3개 미만 또는 한 줄뿐이면 판정하지 않는다(no_lidar/few).
- 처음 방식(주변에서 가장 가까운 라이다 줄)은 바닥을 비스듬히 볼 때 라이다 depth를 짧게 잡아 4~6 m에서 +0.3 m 편향이 났다(out/proj_261007). 평면 맞춤으로 교체.
- 실행: `python3 scripts/sim/verify/proj_check.py --ros-args -p use_sim_time:=true -p out:=<dir>`를 띄운 채 `run_e2e.py`, 끝나면 `proj_summary.py <dir>`. 샘플마다 png(라이다 점 = 거리색, 사용한 점 = 자홍).

### 결과 (out/proj_261007b, Mode B clear 5 ep × look-down 켬/끔, 264 샘플, 모두 depth 투영)
| 카메라 | 거리 | n | 판정 불가 | \|오차\| p50 / p90 | 오차 중앙값 | ≤10 cm |
|---|---|---|---|---|---|---|
| 정면 | 0-2 m | 10 | 5 | 2.3 / 3.4 cm | +2.3 cm | 100% |
| 정면 | 2-4 m | 116 | 0 | 5.1 / 22.9 cm | +3.8 cm | 70% |
| 정면 | 4-6 m | 6 | 0 | 9.9 / 15.5 cm | +8.1 cm | 50% |
| 정면 | 6 m 이상 | 8 | 5 | 46 m | +46 m | 0% |
| 30° | 0-2 m | 27 | 18 | 1.3 / 4.7 cm | +1.0 cm | 100% |
| 30° | 2-4 m | 85 | 0 | 4.4 / 12.9 cm | +2.6 cm | 84% |
| 30° | 4-6 m | 12 | 0 | 5.4 / 17.6 cm | +5.4 cm | 75% |

- bridge 3D 점을 다시 이미지로 투영하면 pixel과 0.1 px 이내 → 투영 계산 자체는 정확하다.
- 오차 중앙값은 2~5 cm로 작지만 p90은 기준(10 cm)을 넘는다. 큰 오차는 정면 카메라의 바닥, 라이다 점이 적은 곳에서 나온다.
- **6 m 이상 8건**: 4 m 앞 바닥을 찍었는데 depth가 약 50 m(바닥 mesh 구멍으로 depth 광선이 빠짐). 라이다는 같은 자리 바닥을 3.5~4.2 m로 본다. gdm 목표는 `goal_max_dist`(6 m)로 잘리지만 방향만 맞고 거리는 틀린다.
- 판정 불가는 주로 0~2 m(라이다 가장 아래 줄이 약 1.3 m부터 바닥에 닿음).

### S1 경로 (같은 실행, bridge CSV `traj_start_x/y`, `traj_head_deg`)
| | n | 첫 점의 로봇 위치와 거리 | 첫 방향 \|각\| p50 / p90 / 최대 |
|---|---|---|---|
| 30° 켬 | 752 | 0 cm (전부) | 0.5° / 1.4° / 8.0° |
| 끔 | 849 | 0 cm (전부) | 0.7° / 1.7° / 11.0° |
- 기준(≤5 cm, ≤5°) 통과(5° 초과는 1% 미만).

### 결정 (2026-10-08): depth 구멍 대책
- 그대로 두고 기록만 한다. 6 m 이상 depth 구멍(8/264건)은 sim 재구성 mesh 문제로 보고 bridge는 수정하지 않음.

## 도구 2: 실행 중 검사 노드 (2026-10-08)
- 파일: `scripts/sim/verify/live_inspect.py` (신규), `run_e2e.py --inspect` (기본 꺼짐, 켜면 에피소드마다 함께 실행)
- 기록: odom pose, cmd_vel(/gdq/msg/cmd_vel, /cmd_vel), /vln/trajectory, /gdm/local_plan_map, /vln/s2, /gdg/data/gp, /vln/status, 발행자 수 → `<ep>/raw/`, 분석 `<ep>/inspect.json`, 그림 `<ep>/inspect.png`
- 실제 속도는 odom twist가 아니라 pose 차분으로 계산(twist: sim world frame, 실기 0)
- 재분석: `python3 scripts/sim/verify/live_inspect.py --analyze <ep dir> --mode trajectory --goal x,y`
- 합성 데이터 확인: 0.10 m 평행 이동 → cross-track 0.100, 반경 2.1 m 원 → |w − vκ| 0.007
- 실행: `python3 scripts/sim/verify/run_e2e.py --mode trajectory --tag inspect_ldon --episodes scripts/sim/verify/e2e_episodes_etri_indoor_clear.json --bridge_args "lookdown_ns:=/camera_lookdown/camera" --inspect`

### 결과 (Mode B, look-down ON, clear 5 ep, out/e2e/inspect_ldon) — SR 2/5 (c3, c4)
| ep | cross-track RMSE / max | 제한 위반 | 명령 0.4 → 실제 v | w RMSE | 발행자 | 정지 vs 마지막 S2 목표 | 정지 vs 정답 |
|---|---|---|---|---|---|---|---|
| c1 | 1.2 / 2.8 cm | 0 | 0.28 m/s | 0.029 | 1 | 0.84 m | 3.5 m ✗ |
| c2 | 1.2 / 2.7 cm | 0 | 0.28 | 0.032 | 1 | 0.97 m | 5.1 m ✗ |
| c3 | 1.3 / 3.1 cm | 0 | 0.28 | 0.032 | 1 | 1.61 m | 0.8 m ✅ |
| c4 | 1.3 / 3.2 cm | 0 | 0.28 | 0.031 | 1 | 1.19 m | 1.0 m ✅ |
| c5 | 1.3 / 3.6 cm | 0 | 0.27 | 0.048 | 1 | 0.68 m | 11.5 m ✗ |
- S1 경로 → MPC → 로봇: 통과. 단, 경로가 응답마다(0.3~1 s) 로봇 위치에서 새로 시작하므로 cross-track이 작게 나오는 면이 있다.
- MPC는 v를 항상 0.4(상한)로 명령하는데 sim 로봇은 0.27~0.28 m/s(약 70%)로 걷고, 지연은 약 0.7 s. w는 잘 따라간다.
- |w − v·κ| p50 0.03~0.04 rad/s. 직선 복도라 w가 거의 0이어서 상관계수(0.1~0.45)는 의미가 적다.
- /cmd_vel 발행자 2 = bridge(선언만) + run_e2e(정지 명령). 정상.
- 실패 원인은 모두 S2 판단이다(경로/제어 아님):
  - c1/c2: S2 목표가 계속 앞쪽으로 찍혀 프린터/마커(y=−4.0/−2.4)를 지나 y≈−7.5에서 STOP. 로봇은 마지막 S2 목표에서 0.8~1.0 m 안에 정지.
  - c5: 시작 직후 S2가 "←←←←"를 3번 연속 출력(180° 회전) → 반대 방향으로 주행. 10/07 13:40 실행에서도 같은 시작 출력.

## 도구 3: dataset 정답 대조 (2026-10-08)
- 파일: `scripts/sim/verify/dataset_check.py`, `dataset_test.sh` (신규). 배포 스택(S2 GGUF Q8_0 + S1 TRT)에 sync dual_server(:5894)를 붙여 VLN-CE r2r 에피소드를 bridge와 같은 HTTP API로 재생(정면 0° + 30° look-down, teacher forcing)
- GT는 학습 코드와 동일: S2 = goal.125cm_30deg [x,y] / relative_goal_frame_id / actions[i+1], S1 = poses[i : s+k_s+1]을 `get_trajectory_relative_to_frame(camera_deg=30)` + `interpolate_and_resample_trajectory`로(데이터셋 소스에서 ast로 로드, 복사 아님)
- 주의: S1/S2 module은 ZMQ PUSH/PULL이라 dual_server가 2개 붙으면 응답이 나뉜다 → :5801 async 서버를 내리고 실행 후 다시 띄움(스크립트가 검사)
- 실행: `scripts/sim/verify/dataset_test.sh 61 q8trt` (61 scene × 1 ep, 3779 프레임, S2 1256회, 약 15분)
- 출력: out/dataset_q8trt_261007_234739/ `summary.json`, `summary.png`, `steps.jsonl`, `png/<ep>_<frame>.jpg`(30° 프레임 위 GT 빨강 / 예측 초록 pixel, 우하단 S1 예측(초록) vs GT(빨강) 경로)

### 결과
| 항목 | 값 |
|---|---|
| S2 출력 종류 일치(pixel/←/→/STOP) | 94.4% |
| pixel L2 p50 / p90 | 11.7 / 41.8 px |
| pixel hit@30 | 83.1% (n=421). ①과 같은 부분집합(6 ep, 40 step 미만) 83.3% = bf16 HF 83% (261002 결과) → 양자화·분리 배포로 인한 저하 없음 |
| pixel 기준 프레임 | 100% lookdown (GT와 같은 30° 프레임) |
| S1 ADE p50 / p90 | 10.5 / 27.0 cm (n=2487) |
| S1 FDE p50 / p90 | 17.4 / 46.6 cm |
| S1 끝점 좌우 부호 일치(|y_gt|>0.3 m) | 98.9% (n=1305) → y 부호(왼쪽 +) 맞음 |
- GT 행(row) 별 hit@30: 먼 곳(0~160) 73%, 중간(160~320) 88%, 가까운 곳(320~480) 53%
- 큰 오차(>100 px)는 대부분 그럴듯한 다른 목표(예: 두 문 중 다른 문)
- STOP: GT STOP 40/40 맞힘. 이른 STOP은 대부분 에피소드 끝 몇 프레임 전
- 시작 프레임: dataset 61개 중 47개가 회전으로 시작(GT ← 29, → 18)하고 모델도 43/47 같은 방향으로 회전. sim c5의 시작 "←←←←"는 이 학습 분포(시작 시 회전)의 영향일 수 있음(가설)
