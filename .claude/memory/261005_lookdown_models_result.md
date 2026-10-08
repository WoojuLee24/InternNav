# 261005 30° look-down 카메라, 선명 구간 시나리오, 3개 모델 평가 + 시각화

## 1. 30° look-down 카메라 (sim)
- 기존 설정 `sensor_configuration.yaml`(camera_link, D455, 14.76°)은 **수정하지 않았다**.
- 새 설정 `sim_docker/workspace/config/lookdown_camera.yaml`: camera_link 기반으로 위치·내부파라미터·해상도·depth를 복사하고 pitch만 30°로 바꿨다.
- 코드: `sim_docker/workspace/scripts/lookdown_camera.py`(신규). `run_rbq10_sim.py`에는 env guard가 붙은 hook 4곳, +23줄만 추가했다.
- 실행: `GUIDEDOG_LOOKDOWN_CAM=1 ./run_rbq10_etri_sim.sh indoor`
- 검증:
  - TF `base_link→camera_lookdown_color_optical_frame`의 roll이 -120°(= -90 - 30)로 나온다.
  - depth vs lidar \|err\| p50 1.1 cm, p90 3.7 cm. overlay 정렬 OK(`out/sensors_geom_camera_lookdown_*`)
  - 발행률: look-down 3.6 Hz, FPV 6.9 → 5.1 Hz(렌더 부하 증가)
- 서버/bridge: S2가 "↓"라고 답하면 30° 프레임으로 다시 질의하고, pixel goal은 30° 카메라의 K/TF/depth로 투영한다(`pixel_frame=lookdown`). 30° 프레임 기준 S1 끝점과의 Δpx 중앙값은 12 px다.

## 2. 모델
- DualVLN(`nextdit_async`), w-NavDP(`navdp_async`), **wo-dagger**(예전 "InternVLA-N1" 릴리스)
- wo-dagger는 config에 `system1=None`이 있어 그대로는 로드되지 않는다. NavDP 텐서의 키와 shape가 w-NavDP와 100% 같아서(`latent_queries`만 16 vs 4) `quant.legacy_config`로 `navdp_async`를 지정해 로드했다.

## 3. 선명 구간 한정 평가 (`e2e_episodes_etri_indoor_clear.json`, 전방 복도만, 에피소드당 1회, bf16)
`scripts/sim/verify/run_matrix.sh <mode> scripts/sim/verify/e2e_episodes_etri_indoor_clear.json clear dualvln:off dualvln:on navdp:on wodagger:on`
- 인자: 모드(`pixel_goal`=Mode A / `trajectory`=Mode B), 에피소드 파일, 결과 태그 접미사, `모델:lookdown(on|off)` 목록

| tag | mode | c1_printer | c2_marker | c3_intersection | c4_past_printer | c5_mid_start | SR | mean dist [m] |
|---|---|---|---|---|---|---|---|---|
| dualvln_bf16_ldoff_clear | pixel_goal | ✗ 3.0 | ✗ 2.7 | ✅ 1.3 | ✅ 0.3 | ✗ 11.9 | 0.40 | 3.9 |
| dualvln_bf16_ldoff_clear | trajectory | ✗ 3.2 | ✗ 4.9 | ✅ 0.5 | ✅ 1.2 | ✗ 11.3 | 0.40 | 4.2 |
| dualvln_bf16_ldon_clear | pixel_goal | ✗ 3.4 | ✗ 5.1 | ✅ 0.7 | ✅ 0.8 | ✅ 0.7 | 0.60 | 2.1 |
| dualvln_bf16_ldon_clear | trajectory | ✗ 1.9 | ✗ 4.6 | ✗ 2.4 | ✅ 1.3 | ✗ 11.9 | 0.20 | 4.4 |
| navdp_bf16_ldon_clear | pixel_goal | ✗ 3.4 | ✗ 5.0 | ✗ 2.1 | ✅ 0.6 | ✗ 10.8 | 0.20 | 4.4 |
| navdp_bf16_ldon_clear | trajectory | ✗ 3.5 | ✗ 5.0 | ✅ 0.5 | ✅ 0.7 | ✗ 12.6 | 0.40 | 4.4 |
| wodagger_bf16_ldon_clear | pixel_goal | ✗ 3.3 | ✗ 5.8 | ✅ 1.4 | ✗ 1.8 | ✅ 0.5 | 0.40 | 2.6 |
| wodagger_bf16_ldon_clear | trajectory | ✗ 3.4 | ✗ 5.1 | ✅ 1.0 | ✅ 1.3 | ✅ 1.0 | 0.60 | 2.4 |

✅ success (stopped within radius)  ✗ stopped elsewhere  ⏱ timeout; number = final distance to goal

모델별 합계(10회 = 2 모드 × 5):

| 구성 | 성공 |
|---|---|
| DualVLN, look-down off | 4/10 |
| DualVLN, look-down on | 4/10 (Mode A는 2→3으로 늘고 Mode B는 2→1로 줄음) |
| w-NavDP, look-down on | 3/10 |
| wo-dagger, look-down on | **5/10** |

해석:
- **짧은 거리 랜드마크 정지(c1 프린터, c2 마커)는 16회 모두 실패**했다. 모든 모델이 목표를 3~5 m 지나쳐 교차로 근처까지 간다. 모델이 "가까운 목표에서 멈추기"를 못 한다.
- 교차로 정지(c3, c4)는 16회 중 12회 성공했다.
- c5(중간 출발)는 여러 run에서 모델이 처음에 180° 돌아 반대로 갔다(모델 판단이고 파이프라인 문제는 아님).
- look-down 30°의 효과와 모델 간 차이는 n=1로는 판단할 수 없다(±1~2회 차이). 결론을 내려면 반복(예: 에피소드당 3~5회)이 필요하다.

## 4. 시각화
- 웹 모니터 `scripts/sim/web_viz.py` → `http://<호스트>:8090`(`?snap=1`은 저대역폭)
  - 화면: 최신 S2 판단(pixel goal + S1 경로), 실시간 overlay, FPV, look-down, top-down 지도(로봇, 주행 경로, S1 경로, gdm 계획, goal), S2 답변 표
  - 조작: 지시문 입력, STOP, teleport
- rviz2 `scripts/sim/rviz/vln.rviz` → 호스트에서 `scripts/sim/rviz/open_rviz.sh`(gdm 컨테이너의 rviz2 사용)
- bridge 추가 토픽: `/vln/s2`(S2 답 JSON), `/vln/s2_image`(S2 판단 프레임). 빈 instruction을 받으면 정지한다.
