# 노은역 S2 반경 STOP 셀 열린 루프 평가 (2026-11-11)

## 용어
- 열린 루프 점검 = 로봇을 움직이지 않고 정답 경로 위 프레임마다 모델 답(STOP / 픽셀 / 턴)만 채점.
- 반경 STOP 셀 = 도착점 r m 안 프레임도 STOP 정답으로 학습한 ckpt (기준 r = 0 → 끝 프레임만 STOP).
- 도착 프레임 = 에피소드 마지막 프레임 (n-1). 정답은 STOP.
- 조기 STOP = 끝 12 프레임 밖(평소 구간, every 4 표본)에서 STOP 이라고 답한 행 — 부작용 지표.
- 첫 STOP = 끝 12 프레임(1 프레임 ≈ 0.25 m) 안에서 처음 STOP 이 나온 위치 (끝에서 몇 프레임 전).
- 화면 가운데 = 예측 픽셀이 (320, 240) ±30 px 안 (11-09 메모의 10 개는 기준이 달라 12 로 다시 셈).

## 바뀐 것
- ckpt 만 다름. 스크립트 · 인자 · 에피소드(hunav val 37 ep) · 도커 이미지 모두 11-09 와 같음.
- 기준 ckpt 도 이번에 다시 돌림 → 11-09 rows 와 **1927/1927 행 예측 · 픽셀 완전히 같음** (GPU 번호가 달라도 같은 자 확인).
- 셀별 GPU: 기준 A5000 #3, r0.5 #1, r1.0 #2.

| 셀 | stop_radius_m | ckpt |
|---|---|---|
| 기준 | 0 | relabel_half_noeun-v01/s1.noeun-both-b1_wait-wait_internvla-n1-system2_20261029_104936 |
| r0.5 | 0.5 | relabel_half_noeun-v01/s1.noeun-both-b1_wait-wait_stopr0.5_internvla-n1-system2_20261110_130157 |
| r1.0 | 1.0 | relabel_half_noeun-v01/s1.noeun-both-b1_wait-wait_stopr1.0_internvla-n1-system2_20261110_112953 |

## 명령 (셀마다 `<ckpt>` 만 바꿈, 호스트에서 한 줄)
`docker run --rm --entrypoint bash --gpus device=<GPU> --runtime=nvidia --ipc=host -v /home/dnwn24/gd_project/modules/gd_vln/workspace:/ws -v /media/TrainDataset/data/noeun_val_extract:/media/TrainDataset/data/noeun_val_extract -v /media/LTDataset/checkpoints:/media/LTDataset/checkpoints:ro -w /ws/src/InternNav dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-a5000 -c "/usr/bin/python3 scripts/eval/noeun_openloop_pixel.py --model_path /media/LTDataset/checkpoints/relabel_half_noeun-v01/<ckpt> --data_root /media/TrainDataset/data/noeun_val_extract/val_b1 --every 4 --tail 12 --out /media/TrainDataset/data/noeun_val_extract/eval_v01/<ckpt>/pixel_1111"`

| 인자 | 뜻 | 왜 이 값 |
|---|---|---|
| --data_root val_b1 | 평가 셋 (hunav_s0 val 37 ep, b1 문장) | 11-09 와 같음 |
| --every 4 | 평소 구간은 4 프레임마다 채점 | 11-09 와 같음 (시간) |
| --tail 12 | 끝 12 프레임은 전부 채점 | 반경 1 m ≈ 4 프레임보다 넉넉히 |
| (기본) --rig 60cm_15deg · --num_history 8 · --resize 384 | 카메라 · 과거 프레임 수 · 입력 크기 | 학습과 같음 |

비교: `python3 logs/stopr_eval/compare.py --rows 기준=<rows> r0.5=<rows> r1.0=<rows> --img_root /media/TrainDataset/data/noeun_val_extract/val_b1/traj_data/r2r/hunav_s0 --viz_eps 29,24,17 --out logs/stopr_eval`

## 결과
| 지표 | 기준 | r0.5 | r1.0 |
|---|---|---|---|
| 도착 프레임 STOP | 14/37 | 18/37 | **26/37** |
|  └ 0-10 m | 1/7 | 2/7 | 3/7 |
|  └ 10-30 m | 4/11 | 5/11 | 8/11 |
|  └ 30-60 m | 6/11 | 6/11 | 9/11 |
|  └ 60+ m | 3/8 | 5/8 | 6/8 |
| 도착 프레임에 픽셀 · 그중 화면 가운데 | 23 · 12 | 19 · 10 | 10 · 8 (+턴 1) |
| 조기 STOP (끝 12 프레임 밖) | 4/1483 (0.27%) | 17/1483 (1.15%) | 12/1483 (0.81%) |
|  └ 그중 끝까지 10 m 넘음 | 1 | 4 | 1 |
| 첫 STOP 중앙 (프레임 · m) | 4 · 0.76 | 8 · 1.50 | 7 · 1.56 |
| 첫 STOP 없음 (끝 12 프레임 내내 STOP 안 함) | 17 | 14 | 10 |
| 계단 픽셀 오차 중앙 (세 셀 공통 362 행) | 27.1 px | 30.8 px | 27.2 px |
| 평지 픽셀 오차 중앙 (세 셀 공통 1131 행) | 20.0 px | 19.2 px | 18.1 px |

끝에서 k 프레임 전 STOP 개수 (37 ep 중):

| k (끝까지 m) | 0 (0) | 1 (0.25) | 2 (0.5) | 3 (0.75) | 4 (1.0) | 6 (1.5) | 8 (2.0) | 11 (2.55) |
|---|---|---|---|---|---|---|---|---|
| 기준 | 14 | 15 | 14 | 11 | 6 | 5 | 2 | 1 |
| r0.5 | 18 | 18 | 15 | 14 | 12 | 5 | 7 | 7 |
| r1.0 | 26 | 22 | 22 | 20 | 21 | 16 | 11 | 8 |

- 첫 STOP 위치는 학습 반경보다 멀다 (r0.5 중앙 1.5 m, r1.0 1.56 m). 다만 성공 반경 3 m 안이라 sim 성공에는 해가 안 된다.
- 조기 STOP 대부분은 끝 2~7 m (끝 근처). 진짜 먼 곳(10 m+) 은 r0.5 에서 3 개 늘고(ep21 22.8 m · ep23 17.2 m · ep25 19.0 m), r1.0 은 기준과 같은 1 개(ep23 12.8 m).
- 계단 픽셀은 r1.0 그대로, r0.5 는 +3.7 px. 평지는 둘 다 조금 좋아짐.

## 시각화 (`logs/stopr_eval/`)
행 = 기준 / r0.5 / r1.0, 열 = 끝 12 프레임, 초록 = 정답 픽셀, 빨강 = 예측 픽셀.
- `tail_ep29.jpg` — 9.4 m 짧은 경로, 기준 · r0.5 는 끝까지 STOP 못 함, r1.0 만 끝 프레임 STOP.
- `tail_ep24.jpg` — 26 m, 기준 STOP 없음 → r0.5 9 프레임 전 · r1.0 끝 프레임 STOP (둘 다 2~2.5 m 에서 조기 STOP 도 있음).
- `tail_ep17.jpg` — 124 m 긴 경로, 기준 1 프레임 전 → r0.5 · r1.0 은 2 m 쯤부터 STOP.
- 표 · 수치: `table.md` · `metrics.json` · `compare_out.txt` (ep 별 첫 STOP 목록 포함).

## 결론
- **r1.0 이 낫다.** 도착 STOP 14 → 26/37 (모든 길이 구간에서 늘어남), 먼 곳 조기 STOP 은 늘지 않음, 계단 픽셀 그대로.
- r0.5 는 이득이 작고(18/37) 먼 곳 조기 STOP 이 3 개 늘었다.
- 남은 과제: 짧은 0-10 m 경로는 r1.0 도 3/7. 끝 12 프레임 내내 STOP 안 하는 ep 10 개.
- 다음 (사용자 확인 후): 세 셀 sim / VLN-PE 평가 (`run_noeun_sim_eval.sh` · `run_noeun_vlnpe_eval.sh`).

## VLN-PE 닫힌 루프 (2026-11-12, v6_b1 val_seen 40, navmesh follower, 600 step)
- 명령: `PORT=<p> SG=<g> EG=0 NOEUN_FOLLOWER=navmesh NOEUN_DATA=/ws/src/InternNav/data/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1/raw_data/r2r scripts/eval/run_noeun_vlnpe_eval.sh <task> /ws/src/InternNav/checkpoints/relabel_half_noeun-v01/<ckpt>`
  task: 기준 `noeun_v01_bothb1wait_v6b1_navmesh_1111` · r1.0 `noeun_v01_stopr1.0_v6b1_navmesh` · r0.5 `noeun_v01_stopr0.5_v6b1_navmesh` (11-12 04:54 끝, 서버 GPU 1).
- 시작 전 internnav-isaacsim 이 GPU 를 못 봄 (NVML Unknown Error) → 컨테이너 재시작 후 처음부터. 렌더 멈춤 자동 재시작 셀당 2~3 번.
| 셀 | SR | OS | SPL | NE | 성공 | 지나침 | 같은층 못감 | 걸음상한 | 다른층 |
|---|---|---|---|---|---|---|---|---|---|
| 기준 (재실행) | 0.275 | 0.60 | 0.262 | 16.0 | 11 | 13 | 8 | 5 | 3 |
| r0.5 | 0.225 | 0.45 | 0.219 | 15.8 | 9 | 9 | 13 | 5 | 4 |
| r1.0 | **0.325** | 0.525 | 0.312 | 12.9 | 13 | **8** | 11 | 6 | 2 |
- 기준 재실행 SR 0.275 = 11-08 과 같음. r1.0 만 성공 4 19 20 25 27 34 · 기준만 성공 5 22 32 38 (실행 흔들림 1~3 ep 수준과 겹침).
- 지나침 13 → 8 (반경 STOP 효과). 대신 3~5 m 앞 STOP 2 → 4 (17 22 32 36) — 3 m 를 아깝게 못 채움.
- 남은 지나침 8: 목표 0.1~2.6 m 까지 가고 3 m 안 질의 2~5 번 모두 픽셀(앞으로 ~2 m), STOP 0.
  원인: 열린 루프는 GT 자세(정답 길 위 · 정답 방향) → 26/37 STOP 이지만, 닫힌 루프는 길에서 비켜선 자세 · 다른 방향으로 접근,
  질의 간격 ~2 m 라 1 m 반경 안에서 질의를 못 받고 지나침, 지나친 장면은 학습에 없음.
- 분석: `logs/stopr_eval/vlnpe_fail.py` (isaac python) → `vlnpe_fail.json` · `vlnpe_fail.txt`.
- 다음 후보: 목표 근처 비켜선 · 지나친 자세 렌더 + STOP (반경 ~2 m, 같은 층, history 포함), DAgger 학습, 1200 step 참고 평가.
- r0.5: 지나침은 13 → 9 로 줄었지만 같은층 못감 13 (일찍 · 엉뚱한 곳 STOP) 으로 SR 0.225 — 열린 루프의 먼 곳 조기 STOP 증가와 맞음. r1.0 이 세 셀 중 가장 낫다.
