# 노은역 S2 학습 방식 ↔ VLN-PE 실제 주행 차이 분석 (2026-11-12)

## 용어
- 학습 장면 = 정답 경로(habitat navmesh 최단 경로) 위 프레임. 위치 · 방향 모두 정답 그대로.
- 실제 주행 = VLN-PE (Isaac H1 flash, navmesh follower, 600 걸음, v6_b1 val_seen 40 ep). 기록 = `logs/<task>/agent/ep*/steps.jsonl`.
- 이탈(dev) = 로봇 위치에서 정답 경로(같은 층 선분)까지 xy 거리.
- 2 m 앞 각도 = 정답 경로에서 2 m 앞 점이 카메라 정면에서 벗어난 각도. 화면 반폭 34.35° (hfov 68.7).
- 질의 = 모델에 물은 걸음 ('↓' 다음 재질의는 하나로 셈).
- 방향(yaw) 은 로그에 없어 복원: 시작 방향 + 15° x (좌 - 우). 전진 이동 방향과 비교한 오차 p90 0.2° → 믿을 만함.

## 명령
- 실제 주행 쪽: `docker exec -w /ws/src/InternNav internnav-isaacsim /workspace/isaaclab/_isaac_sim/python.sh logs/stopr_eval/vlnpe_gap.py --tasks 기준=noeun_v01_bothb1wait_v6b1_navmesh_1111 r1.0=noeun_v01_stopr1.0_v6b1_navmesh --fail logs/stopr_eval/vlnpe_fail.json --out logs/stopr_eval/vlnpe_gap`
- 학습 쪽 (hunav val_b1 37 ep, 학습과 같은 형식): `logs/stopr_eval/train_side.py <traj_data/r2r> <rows.jsonl>` (habitat 이미지 docker run)
- r1.0 STOP 장면 목표 각도: `logs/stopr_eval/r1_view_check.py <traj_data/r2r> 1.0`
- ABC 시험 데이터 각도별 전문가 답: `logs/stopr_eval/abc_view_check.py <out_root> 1.0`
- 결과: `logs/stopr_eval/vlnpe_gap.{txt,json}`
- 주의: 학습 쪽 기준값은 hunav val, 실제 주행은 v6 val (같은 역 · 같은 형식, ep 집합은 다름).

## 1. STOP
| 항목 | 학습 | 실제 주행 (r1.0) |
|---|---|---|
| 목표 정의 | 정답 경로 끝점 (= raw goals 위치, 40/40 거리 0) | 같음. 성공 = 3 m 안 STOP |
| 목표 고르는 법 | 경로 생성 때 뽑은 점. 문장은 그 뒤 LLM 이 근처 랜드마크로 묘사 ("stop by the white pillar") | 같은 문장 |
| STOP 장면 | 경로 위 · 끝 1 m 안 · 남은 행동 전진뿐 → 목표 각도 최대 10.5° | 3 m 안 질의 66 개 중 35 개가 경로에서 0.5 m 넘게 벗어남 |
| 3 m 안 질의 답 | — | pixel 53 · STOP 13 |
- 성공한 STOP 거리 0.7~2.9 m → 모델은 정확한 점이 아니라 "랜드마크 근처" 에서 멈춘다.
- 지나침 6 ep (1 6 21 24 35 38): 목표 0.08~0.89 m 까지 가고 STOP 안 함 → 계속 가다 8~23 m 떨어진 곳에서 STOP.

## 2. 경로
| 항목 | 학습 | 실제 주행 기준 | r1.0 |
|---|---|---|---|
| 질의 때 이탈 > 0.5 / 1 / 2 m | 0 | 40 / 36 / 30 % | 25 / 21 / 16 % |
| 1 m 넘게 벗어난 ep · 그중 경로로 돌아온 ep | — | 34 · 7 | 31 · **4** |
| 경로 위(≤0.5 m) 질의 중 2 m 앞이 화면 밖 | 9.5 % | 14.5 % | 18.2 % |
- 학습에는 벗어난 장면이 0 → 한 번 벗어나면 거의 못 돌아온다.
- 실패 ep 이 2 m 넘게 벗어난 곳 (진행률): 처음(0.05~0.12) 4 · 중간 계단/회전(0.5~0.9) 12 · 끝 근처(1.0, 지나침) 6.
- 벗어남 ≠ 실패: 성공 ep 도 최대 2.9 m 벗어남 (follower 가 모서리를 자름).

## 3. Instruction
- 경로마다 b1 형제 문장 3 개 (같은 생성기, seed 만 다름, 39/464 경로는 같은 문장). 평가 = val 경로의 형제 0. 문체 차이 없음.
- 같은 역 안 split (seen) → 모델이 문장 대신 그림(위치) 으로 길을 외울 수 있음. **문장을 따르는지 미검증** (문장 바꿔 넣기 시험 필요).
- 끝 묘사가 모호한 문장 많음: "stop on the stairs", "stop at the top", "stop near the sign", "white pillar"(여러 개).
- 첫 동작 "Turn around / left / right" 에서 바로 벗어난 ep 4 (15 31 33 39).

## 4. 걸음 수 (600 걸음 상한)
| | 학습(정답 경로) | 실제 주행 성공 ep |
|---|---|---|
| 걸음 / 경로 m | 4.2 | 6.6~6.9 |
- 전진 거리는 경로 길이와 거의 같음 (낭비 아님). 늘어난 건 회전 (약 2 번/m, 학습 약 0.2 번/m) 과 '↓' 제자리 (질의마다 1 걸음, 전체 11 %).
- → 600 걸음으로 갈 수 있는 경로 ≈ 88 m. 86~123 m 경로 6 개(걸음상한)는 완벽해도 못 끝냄.
- history = linspace(0, step-1, 8) 를 걸음 기준으로 고름 → 실제 주행 history 에는 제자리 회전 · 같은 그림 반복(재질의 때 같은 관측을 다시 넣음) 이 학습보다 많다.

## 5. 사진 비교 (2026-11-12) — 차이 없음
- 같은 c2w 를 VLN-PE 평가 카메라(InternUtopia, subframe 2) 와 학습 렌더러(build_renderer, 10 step) 로 찍음. v6 val_seen ep 1 8 20 27 34, 55 장.
- 명령: `CUDA_VISIBLE_DEVICES=0 NOEUN_AGENT_SERVER=0 NOEUN_TASK=render_compare_snap NOEUN_DATA=<v6_b1 raw_data/r2r> /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/noeun_vlnpe_snap.py --episodes 1,8,20,27,34 --out logs/render_compare/vlnpe`
  → `/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/86_render_compare.py --snap logs/render_compare/vlnpe --out logs/render_compare` (끝날 때 segfault 나지만 결과는 다 저장됨)
| 지표 | 평가 | 학습 렌더러 |
|---|---|---|
| 밝기 | 182.2 | 182.0 |
| 선명도(라플라시안 분산) | 251 | 209 |
| 차 평균 · PSNR 중앙 | 1.55 / 255 · 35.5 dB (최저 25.2, ep8 한 장) | |
- 그림 `logs/render_compare/pair_ep*.jpg` — 눈으로 구별 안 됨. 사진 차이는 실패 원인이 아니다.
- 카메라 자세: 학습은 RBQ 몸에 단 카메라(로봇 중심 0.3 m 앞, 계단에서 숙임 -8~18°), 평가는 로봇 중심 · 0.6 m · 15° 고정. 높이 차는 재는 법에 따라 0.1 m 안으로 달라 확정 못 함.

## 6. 문장 바꿔 넣기 (2026-11-12) — 모델은 문장을 쓴다 (특히 STOP)
- hunav val_b1 37 ep 의 문장을 18 칸 밀어 다른 ep 문장으로 바꿈 (`/media/TrainDataset/data/noeun_val_extract/val_b1_swap`, 사진 · 정답은 원본 링크). r1.0, 같은 열린 루프 스크립트 · 인자.
- 비교: `python3 logs/stopr_eval/compare.py --rows 원래문장=<pixel_1111/rows.jsonl> 바꾼문장=<swap_1112/rows.jsonl> --out logs/swap_eval`
| 지표 | 원래 문장 | 바꾼 문장 |
|---|---|---|
| 도착 프레임 STOP | 26/37 | 13/37 |
| 조기 STOP (끝 12 프레임 밖) | 0.81 % | 3.10 % |
| 평지 · 계단 픽셀 오차 중앙 | 19 · 27 px | 28 · 30 px |
| 답 종류 같음 · 두 답 픽셀 10 px 안 | 87 % · 58 % | |
- 언제 멈출지는 문장에 크게 기대고, 어디로 갈지는 사진(위치) 에 많이 기댄다 (같은 역이라 앞부분 경로가 겹치는 영향 섞임).
