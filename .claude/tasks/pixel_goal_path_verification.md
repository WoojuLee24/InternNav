# pixel goal / path 검증 (다음 할 일, 2026-10-07 작성)

사용자가 "할 일이 뭐지?"라고 물으면 **"1. pixel_goal 검증"**이라고 답하고 아래 계획을 실행한다.
배경과 결과 정리는 `scripts/sim/S1S2_COMPRESSION.md`에 있다. 현재 sim 기본 구성은 S2 GGUF Q8_0 + S1 TRT, HF parity다.

## 모델 출력과 전달 경로 (코드 기준)
- **S2 pixel goal**
  - 모델 텍스트 `"x y"` → `[row, col]`로 바꾼다. 좌표계는 look-down 호출에 넣은 프레임(640x480 입력)이다.
  - bridge는 `pixel_space=input`이라 `(u, v) = (col, row)`로 쓴다.
  - Mode A 흐름은 다음과 같다(`vln_bridge_node.py:send_goal`):
    - `pixel_to_base`(depth window median, 없으면 ground ray)로 3D 점을 구한다.
    - map 좌표로 바꿔 `/vln/goal_marker`에 표시한다.
    - odom 기준 상대 pose로 변환해 `/gdg/data/gp`(VIA_GP)로 gdm에 보낸다.
- **S1 path**
  - 33×2 waypoint이고, 로봇 base 기준(x 전방, y 왼쪽, m, 0에서 시작)이다.
  - `T_map_base`로 map 좌표로 바꿔 `/vln/trajectory`에 발행한다.
  - Mode B에서는 `traj_skip=3`을 적용한 뒤 MPC 기준 경로로 넘기고, MPC가 `/gdq/msg/cmd_vel`을 낸다.
  - 이미지 위 경로는 `base_traj_to_pixels`(ground_z_auto)로 그린다.
- **시각화**
  - `/vln/pixel_goal_image`, `/vln/s2_image`, `log_dir`에 `s2_*.jpg`로 남긴다.
  - overlay는 `pixel_frame_id`로 찾은 캐시 프레임(S2가 본 프레임) 위에 그린다.

## 먼저 처리하거나 결정받을 것
1. **look-down 이미지 → 결정됨(2026-10-07): 두 구성 모두 검증한다.**
   - (a) look-down 끔(기본): 정면 프레임을 look-down 자리에 넣음, `pixel_frame=forward`
   - (b) look-down 켬: `GUIDEDOG_LOOKDOWN_CAM=1 ./run_rbq10_etri_sim.sh indoor` + `LOOKDOWN=1 scripts/sim/run_vln_sim.sh`. "↓"이면 30° 프레임으로 다시 질의하고, 30° 카메라의 K/TF/depth로 투영(`pixel_frame=lookdown`)
   - ②~④ 도구는 두 구성에서 모두 돌리고, 결과표에 구성별 열을 둔다.
2. **이중 마커 → 처리됨(2026-10-07)**: 자홍 십자(384 가설)를 지우고, 초록 원은 `pixel_to_proc` 좌표에 그린다(`publish_vis`).
3. **gp 프레임 가정 → 처리됨(2026-10-07)**: bridge를 odom 기준(`pose_source=odom`)으로 바꿨다. map≠odom 테스트는 `verify/frame_test.sh`(결과 `.claude/memory/261007_bridge_odom_frame_result.md`).

## 진행 기록
- 2026-10-07: bridge odom 기준 변경 후 실제 sim 재확인 완료(Mode A/B 각 5 ep, 이전 패턴과 동일). 결과 `.claude/memory/261007_bridge_odom_frame_result.md`. look-down 켬 경로 확인 완료: dual_server가 30° 이미지를 버리던 문제 수정. 30° 프레임에서 S1 끝-pixel 차 106 px(정면 21 px) → ③에서 원인 확인 필요.
- 2026-10-07: 106 px 원인 = S1 입력 불일치(학습·공식 평가는 S1 기억·현재 프레임 모두 30°, 배포는 현재 프레임이 정면). 도구 1 확장 완료: ② `overlay_test.sh` PASS, ④ `gdm_test.sh` 6/6 PASS(gp 3D→평면 계산 수정).
- 2026-10-07: S1 기억·현재 프레임 모두 30°로 변경(학습·공식 평가와 동일, 사용자 결정). S1 끝 0.74→2.47 m, look-down 켬 Mode A 3/5, Mode B 3/5.
- 2026-10-07: ③ 완료(`proj_check.py`). 투영 계산 정확(재투영 0.1 px), 오차 중앙값 2~5 cm, p90 13~23 cm(2~4 m), depth 구멍으로 6 m 이상 46 m 오차 8건. S1 첫 점 0 cm, 첫 방향 p90 1.7°.
- depth 구멍 대책: **그대로 두고 기록만** (사용자 결정, 2026-10-08). sim mesh 구멍이라 실기와 다를 수 있음.
- 2026-10-08: 도구 2 완료(`live_inspect.py`, `run_e2e.py --inspect`). S1→MPC→로봇 통과(cross-track RMSE 1.3 cm, 제한 위반 0, 발행자 1). 실패 c1/c2/c5는 S2 판단(지나침, 시작 180° 회전).
- 2026-10-08: 도구 3 완료(`dataset_check.py`, `dataset_test.sh`). Q8+TRT 배포 스택 hit@30 83.1% (bf16과 동일), S1 ADE p50 10.5 cm, y 부호 98.9%. 검증 도구 1~3 완료.

## 검증 단계 (pixel goal / path 공통 4단계)
| 단계 | pixel goal | path | 합격 기준 |
|---|---|---|---|
| ① 출력 의미 | dataset GT pixel 대조. hit@30px 0.93~0.97로 **완료**. GT·예측 overlay 이미지 저장은 미완 | dataset GT 궤적(에피소드 pose로 계산한 향후 경로)과 ADE, y 부호 | HF와 같은 수준 |
| ② 시각화 | 고정 pixel(중앙·모서리) 주입 → 그려진 위치 대조. 보낸 프레임과 overlay 프레임 해시 일치 | 이미지 경로 vs rviz `/vln/trajectory`. 라이다 바닥점 투영으로 ground_z·외부 파라미터 확인 | 0 px, 해시 일치 |
| ③ 좌표 변환 | 투영점 vs `/ouster/points`(1~6 m). depth/ground 비율(`verify/analyze_bridge_log.py`). 단위 테스트는 `tests/test_projection.py` | 첫 점이 로봇 위치 ≤5 cm, 첫 구간 방향이 heading ≤5° | 바닥 5 m 이내 ≤10 cm |
| ④ 전달 | `/gdg/data/gp`를 odom × rel로 되돌려 `/vln/goal_marker`와 ≤1 cm. gdm 목표·plan 끝점과 일치. 지연 | MPC 기준 vs 실제 odom cross-track RMSE ≤15 cm, v/w 제한 위반 0, w ≈ v·곡률, cmd_vel 발행자 1개 | |

## 만들 도구 (순서)
1. **가짜 서버 주입 테스트**
   - 정해진 pixel/path를 돌려주는 fake server를 쓴다. `verify/fake_sim.py`와 `plumbing_test.sh`(ROS_DOMAIN_ID 77)를 확장한다.
   - ②~④를 모델 없이 결정적으로 합격/불합격 판정한다.
2. **실행 중 검사 노드**
   - 위 토픽들을 동기화해 에피소드별 CSV, 요약 이미지, 수치를 남긴다.
   - 기준선 sim 0/5 이상(2026-10-06, STOP이 목표에서 2.5~16 m, sim GPU 4.2→8.0 GB로 sim 상태가 변함)의 원인 분석에도 쓴다.
3. **dataset 정답 대조**: GT/예측 pixel overlay와 S1 path vs GT 궤적 ADE를 만든다.

## 참고
- 측정 중에는 bridge가 하나만 떠 있어야 한다(dual_server reset 횟수 = 에피소드 수).
- S1 TRT 컨테이너가 시작할 때 멈추면 `S1_CTR=vln-s1-trt2 S1_HEALTH_PORT=5624`로 우회한다.
