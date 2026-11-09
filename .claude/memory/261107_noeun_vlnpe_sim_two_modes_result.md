# 노은역 평가 두 모드 (VLN-PE H1 flash · sim RBQ10 물리 보행) — 2026-11-07, A5000

## 용어
- **VLN-PE 모드** = Isaac(InternUtopia) + H1, **flash** 이동(매 걸음 위치를 직접 놓는다. 막힘은 raycast, 높이는 바닥 raycast + 1.05)
- **sim 모드** = sim_docker RBQ10, **물리 보행**(보행 정책이 /cmd_vel PID 로 이산 동작을 실제로 걷는다)
- **이산 동작** = 0.25 m 전진 · 15 deg 회전 · 제자리 (두 모드 같음)
- **GT 재생** = habitat navmesh 로 만든 reference_path 를 같은 규칙(방향 오차 > 7.5 deg 면 회전, 아니면 전진,
  waypoint 0.15 m 안이면 다음)으로 따라가기 → "그 시뮬레이터 이동으로 GT 경로를 갈 수 있나" 검증
- **경로 이탈(dev)** = 매 걸음 위치에서 reference_path 선분까지 최소 xy 거리
- **같은 층 성공** = STOP + xy ≤ 3 m + 목표와 높이 차 ≤ 1.5 m

## 이전과 달라진 것
1. VLN-PE: `robot_physics` 옵션 추가 (`NoeunVLNEvalTaskCfg`, 기본 'on' = 이전과 같음). 노은 config 는 'off'
   (H1 collision · 중력 끔). 이전에는 ep38 개찰구 지난 뒤 PhysX world.step 이 24 h 멈췄다 (몸이 씬 mesh 와 겹침).
   ep39 비교: on NE 0.09 · 158 걸음 / off NE 0.04 · 130 걸음.
2. sim 모드 새로 만듦 (아래 파일). sim_docker 최소 수정 1 개(커밋 안 함): `GUIDEDOG_INIT_FLOOR_Z`
   — 주면 스폰 높이 하한 = 바닥 + 0.65 (이전: 절대 0.65 → 지하층 z≈-5 에 못 세움). 안 주면 이전과 같음.
3. sim 동작 보정: 걸은 뒤 오차 > 0.05 m / 3 deg 면 같은 목표로 다시 (최대 2 번). 보정 전 15 deg 명령 → 2.5~19 deg,
   0.25 m → 0.19 m 로 GT 가 waypoint 근처를 맴돌았다(ep39 600 걸음 중 회전 576). 보정 후 전진 0.26 m · 방향 ±3 deg,
   회전 중앙값 14~16 deg, ep39 성공(NE 0.067 m).

## VLN-PE GT 재생 (v6 val_seen 40 ep, robot_physics off)
| 항목 | 값 |
|---|---|
| 성공 | **37/40** — 실패 3 개(2 · 13 · 18)는 전부 걸음 상한 600 초과 (waypoint 223~249 개, 8 m 넘게 올라가는 긴 경로) |
| 막힘 | 0 |
| 경로 이탈 | 최대 0.099 m · 평균 0.019 m |
| 높이 차 (성공) | 최대 0.038 m |
→ Isaac flash 이동으로 habitat GT 경로를 계단 포함 그대로 간다. 600 걸음 상한은 긴 다층 경로 3 개에 부족하다
  (GT 추종이 waypoint 마다 회전을 넣어 모델보다 걸음이 많을 수 있음 — 모델 평가 상한은 VLN-PE 와 같게 600 유지).

## VLN-PE 모델 평가 — v01 both-b1 wait, v6_b1 val_seen 40 ep (2026-11-08)
- follower = 모델이 낸 목표점까지 가는 방식. straight(직선) → navmesh(habitat 길찾기, `scripts/eval/noeun_navmesh_server.py` :8040).
| follower | 끝난 ep | SR | OS | SPL | NE | 결과 |
|---|---|---|---|---|---|---|
| straight | 14 (중단) | 0/14 | — | — | — | 개찰구 · 바닥 없음에서 막힘 |
| **navmesh** | **40** | **0.275** | 0.50 | 0.266 | 13.6 m | `logs/noeun_v01_bothb1wait_v6b1_navmesh/result.json` |
- 같은 14 ep 비교: straight 0 성공 → navmesh 2 성공.
- 실패 이유 (progress 로그): 목표 못 감 24 · 걸음 상한 초과 6 (ep 2 · 5 · 9 · 10 · 13 · 18 — 긴 다층 경로).
- 같은 ep 도 실행마다 다를 수 있다 (ep35: 멈췄던 실행에서 성공, 다른 실행에서 실패) → 차이 1~2 ep 은 잡음으로 본다.
- progress 로그 성공 10 vs result.json 11 — 멈춘 실행들의 lmdb 기록이 섞인 듯 (미확인). result.json 을 기준으로.

## Isaac 렌더 멈춤 (VLN-PE)
- `rep.orchestrator.step` 이 시작 54~61 분 뒤 프레임을 안 내고 멈춘다 (3 번, 위치 무관). GPU 를 나눠도 (ISAAC_ACTIVE_GPU) 다시 생김. 원인 미확인.
- 우회: `scripts/eval/run_noeun_vlnpe_eval.sh` — eval.log 가 10 분 안 늘면 eval + agent server 를 죽이고 다시 띄운다 (끝난 ep 은 건너뜀).
- 렌더 GPU 는 CUDA_VISIBLE_DEVICES 를 안 따른다 → InternUtopia `core/runner.py` 에 `ISAAC_ACTIVE_GPU` (커밋 안 함).

## sim GT 보행 (진행 중, 상한 1500)
| ep | 걸음 | 걸린 시간 | NE | SPL | 결과 |
|---|---|---|---|---|---|
| 0 | 721 | 52 분 | 0.11 m | 0.63 | 성공 (계단 2 번) |
| 1 | 347 | 28 분 | 0.08 m | 0.68 | 성공 (계단) |
| 2 | 1094 | 127 분 | 0.04 m | 0.74 | 성공 (계단 3 번, 110 m) |
| 3 | 754 | 55 분 | 0.17 m | — | 성공 |
| 4 | 716 | 58 분 | 0.07 m | — | 성공 |
| 5 | 922 | 76 분 | 0.12 m | — | 성공 |
| 6 | 341 | 31 분 | 0.13 m | — | 성공 |
| 7 | — | 4 시간+ | — | — | **개찰구에서 막힘** (아래) |

### ep7 막힘 분석 (2026-11-08)
- 경로: 개찰구 통로를 지나 (14.58, -2.07) 쪽으로 왼쪽으로 꺾는다. navmesh 로 보면 통로 폭이 ~0.3 m (y -1.7 ~ -1.9) 로 좁다.
- 일어난 순서 (sim.log trunk 위치):
  1. 통로 안 (15.3, -1.7) 에서 개찰구에 부딪힘 (가속도 8.7, 평소 2~4) → 밀려 나가 waypoint 59 (15.18, -1.76) 를 0.3 m 지나침 (도착 판정 0.15 m).
  2. 지나친 waypoint 로 돌아가려고 제자리 회전 → 회전 중 개찰구 옆면을 긁으며 북쪽으로 0.5 m 미끄러짐 (14.67, -1.08).
  3. 거기서 waypoint 59 로 가는 직선 (남동쪽) 이 개찰구 몸체를 지난다 → 전진 명령을 받아도 못 나감, **약 2 시간** (t 1190→8090 s).
  4. 이후 빠져나와 통로로 다시 들어가 왔다 갔다 함.
- 원인: GT 따라가기는 다음 waypoint 로 **직선**으로 간다 (장애물을 모름). 좁은 개찰구에서 몸이 부딪혀 경로에서 밀려나면 그 직선이 개찰구를 지난다.
  VLN-PE GT 재생은 같은 ep7 을 성공했다 — 거기는 robot_physics off (몸 충돌 없음) 라 개찰구에 안 부딪힌다.
- 대응: `--blocked_steps 20` (전진이 20 번 못 나가면 blocked 로 ep 끝) — GT · 모델 평가 모두. `steps.jsonl` 은 이제 걸음마다 기록.
- 상한 600 이전 실행: ep0 이 721 걸음 필요해 실패 → 검증은 1500.
- 느리다: 걸음당 ~4~7 초 (보정 포함) → 40 ep 다 돌리면 하루 이상.

## 명령
- VLN-PE GT 재생 [internnav-isaacsim]: `CUDA_VISIBLE_DEVICES=1 NOEUN_AGENT_SERVER=0 NOEUN_DATA=/ws/src/InternNav/data/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1/raw_data/r2r /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/noeun_gt_replay_check.py --out logs/noeun_gt_replay_v6`
- VLN-PE 모델 [호스트]: `PORT=8032 SG=1 EG=0 NOEUN_FOLLOWER=navmesh NOEUN_DATA=/ws/src/InternNav/data/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1/raw_data/r2r scripts/eval/run_noeun_vlnpe_eval.sh <task> <ckpt 컨테이너 경로>` (navmesh 서버 먼저: internnav-habitat 에서 `python3 scripts/eval/noeun_navmesh_server.py`)
- sim GT 보행 [호스트]: `SIM_GPU=3 EVAL_ARGS="--max_step 1500" scripts/sim/run_noeun_sim_eval.sh gt gt_v6 /media/TrainDataset/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1/raw_data/r2r/val_seen/val_seen.json.gz`
  - 인자: `<policy gt|model> <tag> <episodes 호스트 경로> [ep id 목록]`. env `SIM_GPU`(1) · `SIM_READY_S`(600) · `EVAL_ARGS`
- sim 모델 [호스트]: agent server(GGUF) 를 먼저 띄우고 `NOEUN_AGENT=internvla_n1_s2_gguf scripts/sim/run_noeun_sim_eval.sh model <tag> <episodes>` (스크립트 머리 주석 참고)

## 파일
- `internnav/env/utils/internutopia_extension/{configs/tasks,tasks}/noeun_vln_eval_task.py` — `robot_physics`
- `scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py` — `robot_physics: NOEUN_ROBOT_PHYSICS (off)`
- `scripts/sim/verify/run_noeun_s2_sim_eval.py` — sim 러너 (gt / model, hfov 68.7 crop, PID + 보정, 지표)
- `scripts/sim/run_noeun_sim_eval.sh` — ep 마다 sim 을 시작 자세로 다시 띄움
- `scripts/sim/sim_docker_a5000.override.yml` — 이 서버 docker 가 compose GPU 예약을 못 받아 runtime: nvidia 로
- sim_docker `workspace/scripts/run_rbq10_sim.py` — `GUIDEDOG_INIT_FLOOR_Z` (커밋 안 함)

## 주의
- sim 컨테이너는 privileged 라 `SIM_GPU` 가 실제로 GPU 를 제한하는지 미확인 (4 장 다 보인다).
- sim 장면 = VLN-PE 와 같은 `noeun_station_collision.usdz` (이 서버엔 nurec/obj 없음) → `sim_docker/workspace/NoeunStation/` 에 복사.
- 사람(hunav) ep 은 sim 에서 아직 못 돌린다 (녹화 재생 기능이 이 서버 sim_docker 에 없음).
- `pkill -f <패턴>` 은 같은 패턴을 담은 자기 셸까지 죽인다 → `ps | grep "[x]"` 로.

## GGUF 대조 실험 (2026-11-08) — sim 성적이 낮은 게 경량화 탓인지 sim 탓인지 가르기
| 평가 | 모델 | 로봇·이동 | SR | OS | 성공 ep |
|---|---|---|---|---|---|
| VLN-PE | bf16 | H1 flash | 0.275 (progress 로그 10) | 0.50 | 3 7 8 11 14 16 22 23 32 38 |
| VLN-PE | GGUF Q8 | H1 flash | **0.20 (8/40)** | 0.50 | 5 11 14 16 22 23 33 38 |
| sim | GGUF Q8 | RBQ10 보행 | 3/13 (진행 중) | — | 10 11 12 |
- GGUF vs bf16 (VLN-PE): 성공 6 개 겹침, 차이 2~3 ep — 실행마다 1~2 ep 흔들리므로 경량화 손실은 작다 (확정하려면 반복 실행 필요).
- sim vs VLN-PE (GGUF, 같은 ep 0~12): sim 3/13, VLN-PE 2/13 → sim 이 더 나쁘지 않다. 처음 0/10 은 어려운 ep 이 앞에 몰린 탓.
- 결과: logs/noeun_gguf_v01_bothb1wait_v6b1_navmesh/ (렌더 멈춤 1 번 자동 재시작), scripts/sim/verify/out/noeun_s2_sim/gguf_v01_b1wait/

## sim GGUF 최종 (40 ep, 2026-11-09) + 실패 원인
| 평가 | 모델 | SR | OS | SPL | NE | 막힘 · 넘어짐 |
|---|---|---|---|---|---|---|
| VLN-PE | bf16 | 0.275 | 0.50 | 0.266 | 13.6 | — |
| VLN-PE | GGUF | 0.20 | 0.50 | 0.195 | 16.7 | — |
| **sim** | GGUF | **0.225 (9/40)** | 0.375 | 0.198 | 17.4 | 0 · 0 |
- sim 성공 ep: 10 11 12 17 19 20 23 38 39. VLN-PE GGUF 와 겹치는 성공은 11 23 38 뿐 → ep 단위 결과는 흔들림이 크다.
- sim 실패 31 개 분류 (진행 = 1 - 끝 위치 navmesh 남은 거리 / 처음 거리, 다른 층 = |dz| > 1.5 m):
| 유형 | 개수 | ep |
|---|---|---|
| 같은 층, 일찍 멈춤 / 방향 틀림 (진행 < 0.5) | 9 | 0 27 28 31 32 33 34 35 36 |
| 다른 층, 길 틀림 (진행 < 0.5) | 7 | 1 5 6 7 9 14 15 |
| 같은 층, 목표 3 m 안을 지나치고 딴 데서 멈춤 | 6 | 21 24 26 29 30 37 |
| 다른 층, 덜 올라감 (진행 ≥ 0.5) | 3 | 3 8 16 |
| 같은 층, 가다가 못 미쳐 멈춤 (진행 ≥ 0.5) | 3 | 4 22 25 |
| 걸음 상한 600 초과 (긴 다층 경로) | 3 | 2 13 18 |
| 막힘 · 넘어짐 | 0 | — |
- 결론: 실패는 전부 모델 판단(어디로 갈지 · 언제 멈출지)이다. 보행 · 충돌 문제로 실패한 ep 은 없다.
  - STOP 이 너무 이르다: 4 ep 은 60 걸음 안에 STOP (27 28 31 33).
  - STOP 이 안 나온다: 6 ep 은 목표 3 m 안을 지나갔는데 멈추지 않음.
  - 층 판단: 10 ep 이 목표와 다른 층에서 끝남 (계단을 안 오르거나 덜 오름).
- 모델 출력 모양 (agent/epXXXX/steps.jsonl): 매번 "↓" 뒤 픽셀 목표, 대부분 화면 가운데 (≈320, 240) = "앞으로". 끝에 STOP 한 번.
- 분석 스크립트: scratchpad sim_fail.py (navmesh 서버 :8040 로 남은 거리 계산).

## 실패 로그 추가 확인 (2026-11-09, sim GGUF agent 로그)
- 대화 위치: STOP 은 언제나 새 질의의 첫 답에서만 나온다. "↓" 다음 답은 1264/1264 (bf16 VLN-PE) · 866/866 (sim) 전부 픽셀 목표.
  → 멈춤 판단은 "첫 답: STOP / ↓ / 화살표" 한 곳에서 일어난다.
- 지나침 6 ep (21 24 26 29 30 37): 목표 3 m 안에서 이미 2~8 번 질의됐는데 매번 "↓ → 픽셀 목표(앞으로)". 질의가 적어서가 아니라 목표를 못 알아본 것.
  (추종 최대 step = 8, `MAX_STEPS` — 2 m 마다 다시 묻는다)
- 층 실패 10 ep: navmesh snap 어긋남 0 · 길 없음 0 → 평가 쪽 문제 아님.
  - 1 6 9 14 15: 위쪽 픽셀 목표 0 개 — 계단을 한 번도 고르지 않음.
  - 5 7: 지하(-5) → 대합실(-0.5) 까지 오르고 다음 계단을 못 감. 3 8 16: z≈2.4 (중간 층) 에서 멈춤.

## VLN-PE bf16 ablation — 추종 step · STOP 확인 (2026-11-09 시작, 600 step 그대로)
- 설정 (config 값): `NOEUN_MAX_FOLLOW` (agent `max_follow_steps`, 기본 8) · `NOEUN_STOP_CONFIRM` (`stop_confirm`, 기본 1 = STOP 이 연달아 N 번 나와야 멈춤; 아니면 같은 관측으로 대화를 새로 열어 다시 묻는다, agent 로그 `stop_rejected`).
- 기준: logs/noeun_v01_bothb1wait_v6b1_navmesh (SR 0.275).
- A: logs/noeun_v01_b1wait_v6b1_nav_follow4 (NOEUN_MAX_FOLLOW=4) · B: logs/noeun_v01_b1wait_v6b1_nav_stop2 (NOEUN_STOP_CONFIRM=2).
- GPU: internnav-isaacsim 에서 EG=2 · 3 은 렌더 장치를 못 만든다 ("No device could be created") → A · B 둘 다 Isaac 은 GPU 0, 서버 GPU 1 · 2.
- 결과 (40 ep, 2026-11-09 12:48, 렌더 멈춤 각 1 번 자동 재시작):
| | SR | OS | SPL | NE | 성공 ep | 지나침 (OS 인데 실패) | S2 질의 |
|---|---|---|---|---|---|---|---|
| 기준 | 0.275 | 0.50 | 0.266 | 13.6 | 3 5 7 8 11 14 16 22 23 32 38 | 9 | 2665 |
| A follow 4 | 0.275 | 0.525 | 0.256 | 15.8 | 기준 −22 +19 | 10 | 4843 |
| B STOP 2 번 | 0.25 | 0.525 | 0.244 | 14.0 | 기준 −3 −23 −32 +19 +25 | 11 | — |
- B: 첫 STOP 38 번 중 다시 물어도 STOP 33 · 바뀜 5. 같은 관측이라 거의 같은 답.
- 결론: 둘 다 실행 흔들림(1~3 ep) 안 — 효과 없음. 질의를 1.8 배 늘려도 지나침이 안 줄어 "목표를 못 알아봄" 해석과 맞다.
  → 멈춤 · 층 판단은 테스트 시 설정으로 안 풀린다. 학습 데이터 쪽 (목표 근처 STOP 샘플 · 계단 장면) 으로 간다.

## 학습 데이터 구조 점검 (2026-11-09, `NavPixelGoalDataset` + noeun_base.stage1)
- 실제 값: pixel_goal_only False · stop_weight 5 (trainer 기본) · sample_step 4 · num_future_steps 4 · 1 epoch.
- STOP 샘플 = (경로 x 문장 3) 마다 **마지막 프레임 1 장** x 5. 끝 1~2 프레임 전은 라벨 없음 (goal_len < 3 · len-1 건너뜀). 목표를 지나친 상태 샘플은 없음 (GT 만).
- 턴 샘플은 전진(1) 에서 끊는다 → "↑" 는 학습에 없다 (출력도 0).
- 경로 길이: train v6 중앙 29.6 m · hunav 31.5 m · val 30.9 m — 분포 같음. 층 바뀜 train 194/387 · val 19/40 — 계단 경로 비율도 같음.
- val 길이별 성공 (bf16 기준/A/B · sim GGUF): 0-10 m 2/3/1/2 (9 개) · 10-30 m 2/1/3/3 (10) · 30-60 m 4/4/4/3 (13) · 60 m+ 3/3/2/1 (8).
  짧은 경로가 가장 약하다 (지나침 6 개 중 4 개가 4.5~6.6 m). train 0-10 m 59/387 (15%) vs val 22%.

## 열린 루프 픽셀 · STOP 점검 (2026-11-09, hunav val_b1 37 ep, bf16)
- 스크립트 `scripts/eval/noeun_openloop_pixel.py` (GT 자세에서 첫 답 + '↓' 면 둘째 답 픽셀). 결과 `$E/eval_v01/<ckpt>/pixel/rows.jsonl` (1927 행).
- 계단: 오름 프레임(정답 지점 0.3 m+ 위) 370 개 — 오차 중앙 28 px · <100 px 91% (평지 23 px · 80%). 치우침 없음 (+3/-1 px).
  층 바뀜 ep 18 개 모두 오름 프레임 대부분 <100 px. → **GT 자세에서는 계단을 본다. VLN-PE 계단 실패는 인식이 아니라 닫힌 루프(자세 벗어남 · 실행)** — 계단 샘플 늘리기보다 DAgger/실행 쪽.
  (val_b1 에 내림 계단 프레임 없음.)
- STOP: 마지막(정답) 프레임 STOP 14/37. 길이별 0-10 m 1/7 · 10-30 m 4/11 · 30-60 m 6/11 · 60 m+ 3/8.
  마지막 프레임 비 STOP 23 개 중 10 개가 화면 가운데 픽셀(321 240 ≈ 앞 2.2 m) — "계속 앞으로" 기본값.
  끝 12 프레임 밖 일찍 STOP 4/1483 (거의 없음). → **자세 어긋남이 아니라 STOP 자체를 약하게 배움 (특히 짧은 경로)**, 모델은 멈추기보다 계속 가는 쪽으로 치우침.
  처방 후보: 끝 반경(~1 m) 안 프레임도 STOP 라벨 · 짧은 경로 STOP 가중.
