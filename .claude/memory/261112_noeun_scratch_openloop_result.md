# 노은역 S2 scratch 셀 열린 루프 (2026-11-12)

## 용어
- scratch = R2R 학습 모델이 아니라 Qwen2.5-VL-7B-Instruct 에서 바로 노은 데이터 학습 (`relabel_half_noeun-v01-scratch`, 10-30 학습). ep1 / ep3 = 1 · 3 epoch.
- 기준 = 같은 데이터 · 셀(both-b1_wait-wait) 을 R2R 학습 모델에서 출발 (`relabel_half_noeun-v01`).
- 지표 뜻은 `261111_noeun_stop_radius_eval_result.md` 와 같음 (도착 프레임 STOP · 조기 STOP · 픽셀 오차).

## 바뀐 것
- ckpt 만 다름. 스크립트 · 인자 · ep(hunav val_b1 37) 는 반경 STOP 비교와 같음 (기준 · r1.0 행은 그때 결과 재사용 — 같은 날 같은 명령, 기준은 11-09 와 1927/1927 일치).
- 명령: 반경 STOP 문서와 같고 `--model_path /media/LTDataset/checkpoints/relabel_half_noeun-v01-scratch/<ckpt>` (GPU 3).
  ep1 `s1.noeun-both-b1_wait-wait_ep1_internvla-n1-system2_20261030_053555` · ep3 `s1.noeun-both-b1_wait-wait_ep3_internvla-n1-system2_20261030_101153`.

## 결과 (`logs/scratch_eval/table.md`)
| 지표 | 기준 | r1.0 | scratch ep1 | scratch ep3 |
|---|---|---|---|---|
| 도착 프레임 STOP | 14/37 | 26/37 | 24/37 | 15/37 |
| 조기 STOP (끝 12 프레임 밖) | 4 | 12 | 6 | 7 |
| 계단 픽셀 오차 중앙 | 28 px | 27 px | 35 px | 32 px |
| 평지 픽셀 오차 중앙 | 23 px | 19 px | 23 px | 24 px |

## 결론
- scratch ep1 은 STOP 을 잘 한다 (24/37, r1.0 과 비슷) — 그러나 ep3 은 15/37 로 기준 수준. epoch 를 늘리면 "끝 프레임 1 장만 STOP" 라벨에 맞춰져 STOP 이 줄어드는 듯 (추정).
- 계단 픽셀은 scratch 둘 다 나쁨 (+4~7 px) — R2R 출발이 계단 · 길 찍기에 도움.
- → R2R 출발 + 반경 STOP(r1.0) 이 가장 낫다. scratch 로 바꿀 이유는 약함.
