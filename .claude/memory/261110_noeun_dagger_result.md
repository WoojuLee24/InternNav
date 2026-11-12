# 노은역 S2 DAgger (A5000 에서 롤아웃 · 렌더, H200 에서 학습)

## 1단계 설계 확인 (2026-11-10)
- 렌더러: `51_render_rbq.py` 와 같은 `gs_vlnpe/apply_real/04_render_obs_isaac.py` (`build_renderer` · `render_along(cas, c2w)`),
  카메라 `camera_profiles['60cm_15deg']`, 씬 `data/GS_USDZ/noeun_station_collision.usdz`, light ambient_only · rtx_ambient 10 · iso 70.
  - val hunav ep0 의 학습 자세(`pose.60cm_15deg` 를 raw start 로 월드 복원)를 다시 렌더 → 학습 이미지와 거의 같음 (프레임 0 · 20 · 40, 시작 위치 복원 오차로 몇 cm 어긋남).
  - 속도: 렌더러 준비 34 s, **0.66 s/프레임** (GPU 0).
  - 평평한 trunk(`frames_to_poses`, 기울기 0) 는 학습 프레임 0 보다 덜 숙인다 — 학습은 RBQ 물리 걷기 trunk 자세(흔들림 · 숙임 포함). 롤아웃은 학습 자세의 trunk pitch 중앙값을 쓰는 것을 검토.
- 라벨: `gs_vlnpe/vlnce_pixel_goal.select_goal(poses, actions, i, rig, depth)` — 프레임 i 의 pixel goal 은 **그 뒤 궤적(자세 + 행동)** 과 프레임 i 깊이만으로 정해진다.
  → 롤아웃 상태마다 전문가 연속 궤적을 렌더 없이 운동학으로 굴려 같은 함수로 라벨 (train 과 같은 규칙).
- 경로 · 문장: raw train `vln_pe_noeun_nav3d_rbq_{v6,hunav_v1}/raw_data/r2r/train/train.json.gz` (200 + 187 = 387, 층 바뀜 100 + 94).
  `raw episode_id == traj_data episode_index` (63_build_labeled assert). 성공 반경 goals.radius = **3.0 m**.
  문장 라벨(train)은 A5000 에 없음 — H200 `/home/irteam/data-vol1/nav3d/noeun/instruction_labels/<b1 이름>/{v6,hunav_v1}_train/*.json.gz` 필요.
- 전문가 행동: `scripts/eval/noeun_navmesh_server.py` (habitat_sim PathFinder, 회전 15° · 전진 0.25 m 규칙). 목표 = reference path 위 앞쪽 점 (문장과 같은 길로 복귀).
- 학습 로더: 롤아웃 에피소드의 행동은 모델 행동이라 기존 로더(턴 = 에피소드 행동열, STOP = 마지막 프레임)로는 라벨이 틀린다
  → 프레임별 전문가 라벨 열을 읽는 새 데이터셋 클래스가 필요 (H200).

## 2단계 코드 · 시험 (2026-11-10)
- `scripts/dataset_converters/nav3d/noeun/81_dagger_navmesh_server.py` (habitat 컨테이너, :8041): /next(기존 Planner) · /step(try_step 전진) · /expert(정답 경로 복귀 궤적).
  - 전문가: ref 위 진행 번호(되돌아가지 않음) -> lookahead 2 m 점을 가까워질(0.5 m) 때까지 유지 -> navmesh 경로 다음 점으로 15° 회전 / 전진(남은 각 흡수).
    처음 판(매 step 목표 바꿈 · 각 흡수 없음)은 좌우 흔들림(LFRFL) — GT 250 step 경로가 332 step. 고친 뒤 245 step, val 8 ep 모두 끝 STOP · 최대 이탈 < 0.45 m.
- `82_dagger_rollout.py` (isaac 컨테이너): 렌더(51 과 같음, trunk pitch = -0.75 x 경사 — 학습 RBQ 자세 실측) -> agent server(평가와 같은 internvla_n1_s2, navmesh follower, base 0.55)
  -> 전문가 라벨(select_goal) -> beta 로 전문가/모델 실행. 저장: vln_ce 형식 + dagger.* 열 (전문가 행동 16 개 · STOP · 실행 주체 · 이탈 · world 자세).
- `83_dagger_viz.py`: 위에서 본 경로 · 프레임에 전문가 정답 그림 · 라벨 종류 요약.
- 시험 (val hunav ep 1 · 27, beta 0.5, 데이터로 쓰지 않음): 240 + 47 프레임, 둘 다 전문가 STOP 으로 끝, 이탈 중앙 0.10 · 최대 0.59 m, **1.0 s/step**.
  라벨: pixel 241 · turn 26 · STOP 2 · 정답 없음(전진인데 보이는 목표 없음) 10 · pixel 짧음(<3) 8.
  agent 의 pixel -> world 변환 확인 (goal z = 바닥 + 0.06 m, 2~2.6 m 앞). 그림 logs/dagger_smoke/viz/.
- 렌더 씬(GS)에 사람이 구워져 있다 (벤치에 앉은 사람 등) — 전문가는 WAIT 를 하지 않으므로 b1_wait 의 WAIT 라벨과 겹칠 수 있다.

## 3단계 train 롤아웃 (2026-11-10 16:22 시작)
- 문장: H200 b1 train 라벨을 rclone 으로 `/media/TrainDataset/data/noeun_eval/labels/{v6_train,hunav_v1_train}/` 에 복사 (200 x 3 · 187 x 3, 빠진 경로 없음).
- 설정: 387 경로 전체 (v6 200 + hunav_v1 187), beta 0.3, 문장 형제 = 경로 번호 % 3, 모델 v01 both-b1_wait-wait.
- 출력: `/media/TrainDataset/data/noeun_dagger/v01_b1wait_beta03/traj_data/r2r/dagger_{v6,hunav_v1}_s{0,1,2}` (컨테이너 `data/data/...`).
- 병렬: GPU 0 의 Isaac 렌더러는 하나에 약 8.5 GB → **2 개까지** (3 번째는 OOM 으로 죽음). 작업자 w0 · w1 (agent server GPU 1 · 2, :8032 · :8033),
  경로 번호 % 3 == 2 몫은 w0(v6) · w1(hunav) 끝난 뒤 이어서 (`logs/dagger_train_w2_on{0,1}.log`). 82 에 `--agent_log_dir` 추가 (같은 out_root 에서 작업자별 agent 로그).
- 속도: 작업자당 약 1 프레임/s.
- 66/387 시점 (19:20) 끝난 이유: model_stop_early 40 중 37 이 도착 3 m 안 (중앙 1.0 m) — 전문가 STOP 반경 0.35 m 가 엄격해서 붙은 이름, 실제 일찍 멈춤은 3 개.
  deviated 8 (경로 중앙 61 m, 진행 49 %). train 경로라 모델이 끝을 잘 알아본다 → val 과 같은 조건 비교 필요.

## 다음: train 경로 같은 조건 평가 (DAgger 끝나면 자동)
- 평가 셋: `/media/TrainDataset/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1_train40/raw_data/r2r/val_seen/val_seen.json.gz`
  v6 train 40 경로 (multifloor 19 + flat 21, seed 0 — val_seen 구성과 같음), 문장 = 형제 0 (val b1 과 같은 규칙). 폴더 이름 val_seen 은 config 가 그 split 을 읽어서.
- 대기 스크립트(호스트 nohup, 로그 `logs/after_dagger_train40.log`): 82 프로세스 0 개 → :8032 · :8033 서버 내림 →
  `SG=1 EG=0 NOEUN_FOLLOWER=navmesh NOEUN_DATA=.../vln_pe_noeun_nav3d_rbq_v6_b1_train40/raw_data/r2r scripts/eval/run_noeun_vlnpe_eval.sh noeun_v01_bothb1wait_v6b1train40_navmesh <v01 ckpt>`
- 비교 기준: val_seen 40 (같은 평가기) SR 0.275 · OS 0.50.

## train 40 평가 결과 (2026-11-11 00:09, 같은 평가기 · 모델 · follower)
| 경로 | SR | OS | SPL | NE | 평지 성공 | 층 이동 성공 |
|---|---|---|---|---|---|---|
| val_seen 40 | 0.275 | 0.50 | 0.266 | 13.6 | 4/21 | 6/19 |
| **train 40** | **0.425** | 0.675 | 0.391 | 11.2 | 9/21 | 7/19 |
(성공 개수는 progress 로그 기준 — 재시작 때문에 result.json 과 1 개 차이)
- 해석: 평지는 train 이 2 배 (외운 효과) · 층 이동은 같음 (실행 문제 — 계단 · 걸음 상한 7) · 지나침(OS-SR)은 둘 다 0.23~0.25.
  → 경로 늘리기(평지 일반화) · DAgger(계단 실행) · 반경 STOP(지나침) 셋 다 필요.

## DAgger 중단 · 재시작 (2026-11-10 21:55 → 11-11 04:51)
- 원인: 작업자가 v6 끝나고 hunav 로 넘어가며 같은 agent server 에 /agent/init 을 다시 보냄 → 모델 두 번째 로드 OOM (500). v6 134/200 저장된 채 멈춤.
- 재시작: 일감마다 서버를 새로 띄우는 runner (scratchpad dagger_worker.sh, 로그 logs/dagger_runner_w{0,1}.log · dagger_train_r{0,1}.log).
  w0: v6 나머지 66 + hunav 0~59, w1: hunav 60~186. 남은 253 ep, 약 10~11 h.

## train 40 실패 원인 (2026-11-11, lmdb + agent 로그, 분석 logs/noeun_v01_bothb1wait_v6b1train40_navmesh/t40_over.py · cats.json)
| 유형 | train 평지/층 | val 평지/층 |
|---|---|---|
| 성공 | 9 / 8 | 4 / 7 |
| 지나침 (3 m 안 지나감) | 8 / 2 | 8 / 1 |
| 걸음 상한 | 0 / 6 | 0 / 5 |
| 같은 층 못 감 · 다른 층 · 넘어짐 | 4 / 3 | 9 / 6 |
- 지나침 10: 목표 0.1~2.5 m 까지 감, 3 m 안 질의 2~7 번 모두 화면 가운데 픽셀(앞으로 ~2 m). 한 번 지나치면 돌아오지 않고 NE 3~33 m.
  성공 ep 은 3 m 안 질의 1~3 번 뒤 STOP. 학습 STOP = 끝 프레임 1 장 (끝 1~2 프레임 전 라벨 없음, 3 프레임+ 전은 '앞으로') · 지나친 상태 샘플 없음.
  → 외우기와 무관 (train · val 같은 수) — STOP 라벨 설계 문제. 반경 STOP + DAgger(지나친 뒤 돌아오기) 가 처방.
- 걸음 상한 6: 경로 51~119 m (5 개 85 m+), agent 600 step 중 전진 307~389 · 회전 144~225 · 제자리 67~70. 5 개는 진행 60~93 % 에서 끊김 (맞게 가는 중).
  0.25 m 전진만으로도 100 m = 400 step → 상한 600 은 이 경로에 빠듯 (GT 재생도 val 3 개 600 초과). 평가 예산 문제가 큼.
- 진짜 길 판단 실패: 같은 층 못 감 5 (일찍 STOP · 방향 틀림) · 다른 층 1 · 넘어짐 1 = 7/40 (val 15/40 — 이 부분이 외우기 차이).

## hunav 중단 (2026-11-11 07:45, 사용자 결정) — 작업 후보로 남김
- v6 만 끝까지 (남은 37 을 w0 · w1 에 나눔, 약 1.5 h). hunav_v1 은 약 50/187 저장된 채 멈춤 — 나중에 같은 out_root 로 다시 돌리면 done 파일로 이어 간다.
- GPU 3 은 반경 STOP 열린 루프 평가에 씀.
