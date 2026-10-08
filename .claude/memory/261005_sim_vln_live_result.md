# 261005 sim VLN 연동: 실제 sim+gdm 검증 결과

환경: sim_docker `run_rbq10_etri_sim.sh indoor` + gdm `lpp-sim`(Mode A) / `lpp-sim-vln`(Mode B) + gd_vln 컨테이너(`internnav-gdsim`, DualVLN bf16 서버). GPU는 RTX 5090 32GB 1장이다.

## 검증 항목별 결과
| ID | 항목 | 결과 | 판정 |
|---|---|---|---|
| V0 | 토픽과 TF | 전 토픽 수신, TF map←base/camera/os_sensor 정상. 수신율: odom 34 Hz, color 6.9 Hz, depth 15 Hz (sim RTF 약 0.7) | 통과 |
| V1c | VRAM (sim+gdm+bf16) | 대기 23.5 GB / 32.6 GB(sim 4.3 GB, 서버 15.7 GiB), 추론 피크 +1 GiB → 여유 약 7 GB | 통과(bf16 사용 가능) |
| V2 | 카메라 crop/K와 lidar overlay | 벽·문틀·복도 끝 정렬 OK (`verify/out/sensors_geom_*/overlay_lidar_on_rgb.png`) | 통과 |
| V4 | physx depth vs lidar | \|err\| p50 1.5 cm, p90 7.4 cm (DOWNSCALE 4) | 통과. DOWNSCALE 2는 불필요 |
| V5 | pixel goal 좌표 공간 | live 21회에서 S1 끝점과 Δpx 중앙값: 입력공간 11.9 px vs 384공간 220 px | 통과(입력 640x480 [row,col]) |
| V6 | 투영 | 2~4 m depth 투영 오차 1.5 cm. ground-ray는 floor 높이 실측 -0.476 m(설정값 -0.55 아님)로 0.20 m에서 0.075 m로 개선. 0~2 m는 lidar 바닥점이 없어 측정할 수 없음 | 통과. floor 자동 추정 추가 |
| V7 | odom twist frame | yaw -90°에서 전진: twist가 world pose-diff와 RMSE 0.019, body와 0.289 → **world 확정**. cmd_vel은 body로 해석(이동 방향 오차 1.1°) | 통과 |
| V8 | 이미지↔TF 동기 | stamp 시점 TF lookup 100%(124 프레임), color-depth Δ ≤50 ms | 통과 |
| V10⑤ | discrete PID | 보행 dead zone과 관성 때문에 원래 PID로는 15° 회전이 4~9° 부족하거나 과회전했다. dead zone 보상(v≥0.3, w≥0.4)과 실측 속도 기반 lead 0.45 s를 적용한 결과: 전진 오차 ≤7 cm, 회전 오차 ≤9° | 기준(3°/5 cm) 미달. 실용상 허용으로 판단(다음 S2가 재계획) |
| V11 | Mode B | v,w 제한 위반 0건, `/gdq/msg/cmd_vel` publisher는 bridge 1개, 서버를 멈추면 약 2.4 s 뒤 정지하고 재개되면 다시 주행 | 통과 |
| V12 | cmd_vel 중재 | Mode A에서 bridge가 `/gdq/msg/cmd_vel` publisher로도 잡혀 2개였다. 수정 후 Mode A는 `/cmd_vel`만 쓴다 | 수정 완료 |

## 주행 결과 (V13, DualVLN bf16, 에피소드당 1회)
에피소드는 `verify/e2e_episodes_etri_indoor.json`에 있다. 지도는 메인 복도(x≈0, y +5~-20)와 y≈-9.5 교차 복도로 되어 있다.

| episode | Mode A (v3) | Mode B (v2) |
|---|---|---|
| e1 교차로에서 정지 | 실패(0.2 m 차, 1.7 m) | **성공**(0.9 m) |
| e2 교차로 좌회전 후 정지 | 실패(교차로 정지) | 실패(좌회전 후 사무실 쪽 캐비닛 앞 정지) |
| e3 교차로 우회전 후 정지 | 실패(교차로 정지) | 실패 |
| e4 뒤로 돌아 복도 끝 | 실패. 직전 v2 run에서는 뒤로 돌아 y=6.5까지 가서 2.3 m 차였다 | 실패(300° 회전 후 원래 방향) |
| e5 교차로 지나 끝까지 | 실패(교차로 정지) | 실패(교차로 정지) |

- 배선과 제어는 정상이다. 목표 전달, 주행, STOP 처리 모두 동작한다.
- 탐색 성공률은 낮다. 대부분 교차로(약 10 m)에서 멈춘다.
- 원인 분석:
  1. **splat 품질**: nuRec 촬영 경로 밖 방향(뒤쪽, 측면)은 렌더가 뭉개진다(`e4` 프레임 참고). 모델이 뒤쪽 복도를 인식하지 못하고 계속 회전했다.
  2. **look-down 불일치**: 학습은 30° 내려다본 영상, sim 카메라는 14.8° 고정이다.
  3. n=1이라 편차가 크다. 같은 지시도 run마다 결과가 달랐다(e4).

## 검증 중 고친 버그 (bridge/스크립트)
- root 컨테이너 ↔ uid 1000 sim/gdm 사이 Fast DDS SHM으로 데이터가 오지 않았다 → `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`
- Mode A에서 bridge가 `/gdq/msg/cmd_vel` publisher를 중복으로 만들었다 → mode별 publisher 1개
- Mode B watchdog 1 s가 S2(약 2 s)마다 걸려 stop-go가 생겼다 → `server_timeout` 3 s
- PID 실행 중에도 질의해 회전이 잘렸다 → discrete 실행 중 질의 보류(`discrete_block_max_s`)
- 새 publisher의 첫 GP가 유실되고 dedup 때문에 재전송되지 않았다 → 구독 확인 후 전송, 같은 goal을 5 s마다 재전송
- mesh 구멍 depth로 48 m 거리 goal이 생겼다 → `goal_max_dist` 6 m로 방향 유지 clamp
- PID dead zone과 관성 → `control_utils.discrete_step`
- ground 높이 고정값 오차 → `projection.estimate_ground_z` 자동 추정

## 남은 것
- V1d (bf16 vs int8 closed-loop)와 w-NavDP 비교. 현재 baseline 성공률이 낮고 편차가 커서, 에피소드 반복(예: 각 3회) 없이는 비교 의미가 작다.
- 탐색 성능 개선 방안(별도 결정 필요): sim에 30° look-down 카메라 추가, splat 품질이 좋은 구간으로 시나리오 한정, 지시문을 시야 안 랜드마크 중심으로 변경.
