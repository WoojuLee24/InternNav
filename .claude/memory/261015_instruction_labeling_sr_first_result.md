# instruction_labeling — 첫 SR 결과 + 원인 분석 (2026-10-15 18시 KST)

설정: relabel_half (mini, s1 1 epoch, seed 1회), val_unseen 1,839, ld30. progress.json 을 (scene, episode) 로 dedupe (최신 run_stamp).
SR = 3 m 안에서 멈춘 비율, OS = 한 번이라도 3 m 안에 들어간 비율, over = OS=1 & SR=0 (목표 근처까지 갔다가 실패).

## 2x2 (학습 문장 × 평가 문장)
| 학습 \ 평가 | GT 문장 | 자기 label 문장 |
|---|---|---|
| GT | **42.5** | — |
| v278fix | 31.9 (over 8.1) | 31.9 |
| **b0** | 32.1 (over **17.2**) | **56.3** (over 11.6) |
| **C** (b0_slight-omit_fewshot-3) | 32.7 (over **17.9**) | **49.4** (over 13.6) |

## GT 학습 모델에 label 문장으로 평가 (= follower 평가: label 문장에 길 정보가 얼마나 남아 있나)
| 평가 문장 | SR | over |
|---|---|---|
| GT | 42.5 (463 ep 부분: 41.9) | 9.3 |
| **b0** | **40.5** | 11.7 |
| C | 39.5 | 11.0 |
| C+SS | 38.9 | 13.3 |
| v278fix 계열 (13개) | 27.8 ~ 30.1 | 13~15 |

## 해석
1. **label 문장의 정보량은 GT 에 거의 가깝다**: GT 로 학습한 모델이 b0 문장만 보고도 40.5 (GT 문장 42.5, v278fix 30.0).
   v278fix 계열이 잃던 12 %p 중 10 %p 를 되찾았다.
2. **b0 로 학습 + b0 로 평가 = 56.3** — GT 2x2 (42.5) 보다 높다. label 문장이 경로와 더 일관되게(기하에서 나온 회전) 쓰여 있어서 배우기 쉽다.
3. **문제는 학습 문장과 평가 문장의 문체 차이**: b0 로 학습한 모델에 사람 문장을 주면 32.1. 이때 목표를 지나치는 비율이 17 % (v278fix 8 %),
   지나친 경우 이동 거리 22.7 m (v278fix 9.2 m) — 제자리 회전이 아니라 계속 걸어간다. 정지 표현(stop/wait/없음)·끝 명사(방/물체)별로 나눠도 고르게 높다 → 특정 끝맺음이 아니라 전반적인 문체 차이.
4. 문장 지표(CIDEr) 순위와 follower SR 순위가 반대: CIDEr C+SS > C > b0, follower SR b0 > C > C+SS.
   → dev 에서 고를 때 CIDEr 만 보면 안 된다. follower SR 이 더 직접적인 지표.

## 정지 단서 A (축 1: 구현 검증) — 2026-10-16
- `route.stop_cue=A-grounded-v1` (label/stop_cues.py): MP3D `.house` 물체(O/C 줄) 중 정지 지점 수평 2 m 안, 높이 -0.2~2.2 m,
  마지막 프레임(정면·30°)에 투영되고 depth 로 안 가려진 물체 하나 → "You stop next to the bed, on your left (from scene annotation)."
- 학습 없는 검증 (dev 101 traj, eval/check_stop_cues.py): 후보 있음 90 %, 형제 중 하나라도 GT 마지막 문장과 명사 일치 57 %,
  top1 이 GT 마지막 문장에 나오는 비율 24.5 %, gemma yes/no 판정 정밀도 75 %. 후보 표시 그림 logs/instruction_labeling/stop_cues/A-grounded-v1_dev/viz.
- v2 (중심 보임 필수 + 문은 정면만 + 큰 가구 우선): 후보 59 %로 줄고 정확도 개선 없음 → v1 채택.
- 생성 (b0 + A-grounded-v1): 생성 문장의 92 % 가 받은 물체를 마지막 문장에 씀. 문장 지표는 b0 와 비슷 (CIDEr .239 vs .235, 명사 정확도 .347 vs .323).
- SR 검증: relabel_half 셀 → node2 queue (C+방v2 다음).
