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

## 정지 단서 P·R (축 1 학습 없는 검증, dev 101 traj) — 2026-10-16
- P-v1 (label/stop_cues.py place_candidates): 계단 위/아래(마지막 3 m 높이 변화 0.4 m), 문간(door 물체 0.8 m 안, 방금 들어온 방이면 방 이름),
  방 입구 바로 안쪽(방이 바뀌고 경계 1 m 안), 복도 끝. R-v1: 정지 지점의 방.
- 결과: 후보 있음 P 85 % / R 96 %. **GT 3문장 중 하나라도 같은 종류 단서를 쓰는 비율** (정밀도 대용):
  doorway 71 % (전체 기본 비율 51 %), stairs_bottom 83 %, stairs_top 100 %, entrance 58 %, room 37 %.
- gemma yes/no 판정은 장소 단서에 쓸 수 없음: 계단 아래 0 % 인데 GT 는 83 % 동의 — 정면 이미지 한 장으로 "내가 문간/계단 아래에 서 있나" 를 VLM 이 판단 못 함.
  → P·R 정밀도는 GT 동의율로 본다. A 는 VLM 판정(75 %) 과 GT 동의 둘 다.
- 계단 region(s) 은 방 이름에서 뺌 ('just inside the stairs' 방지).

## A·P·R 후보 mini 전체 계산 (2026-10-16) — 캐시 `instruction_labels/_stop_cues/A-grounded-v1+P-v1+R-v1/{split}.jsonl`
`label/build_stop_cues.py` (trajectory 단위, 32 프로세스). GT 동의율 = GT 3문장 중 하나라도 같은 종류 단서로 끝나는 비율, 기본 비율 = 전체 traj 에서 GT 가 그 종류를 쓰는 비율.

| 단서 | val_unseen 후보 | val GT 동의 | train 후보 | train GT 동의 | GT 기본 비율 (val/train) | 판정 |
|---|---|---|---|---|---|---|
| A 물체 | 89 % | 54 % | 81 % | 55 % | — | 사용 |
| P 문간 | 62 % | 73 % | 67 % | 69 % | 58 / 58 % | 사용 (+11~15 %p) |
| P 계단 위 | 5 % | 94 % | 5 % | 89 % | 14 / 12 % | 사용 (매우 정확) |
| P 계단 아래 | 6 % | 58 % | 3 % | 57 % | 8 / 5 % | 사용 |
| P 방 입구 "just inside" | 12 % | 53 % | 8 % | 41 % | 61 / 62 % | **제외** (기본 비율보다 낮음) |
| P 복도 끝 | 0 % | — | 0.6 % | 10 % | 7 / 5 % | **제외** (드물고 부정확) |
| R 방 | 89 % | 46 % | 86 % | 50 % | — | 사용 |
| 어느 단서든 있음 | 99 % | | 98 % | | | |

## 축 2: 정지 단서 고르는 방법 (2026-10-16 저녁) — 후보 캐시 A-grounded-v1 + P-v2(문간·계단) + R-v1 고정
- `route.stop_select=rule-v1` (계단 > 문간 > 물체 > 방 중 하나) / `vlm-v1` (후보 전부 주고 VLM 이 고름) / `rotate-v1` (형제마다 다른 후보).
- dev 정지 단서 종류 (마지막 문장, 정규식 근사, %): GT P35 A23 R7 O11 X18 / b0 P36 A53 R5 / rule P64 A32 / vlm P39 A21 R38 / rotate P48 A46 R3.
  단서 준수 rule 94 %, rotate 89 %; vlm 이 고른 것 문간 164 > 물체 70 > 방 49.
- 문장 지표는 모두 b0 수준 (CIDEr .233~.239). rotate 만 다양성 크게 ↑ (형제 유사도 .86 → .72, 고유 82 → 95 %).
- 전체 생성 (train 10,684 / val 1,839, 예외 0) → relabel_half 셀. 배치: node2 rule / node3 rotate / node4 vlm (각 노드 현재 job 뒤).

## 추가 결과 (2026-10-19 KST) — label→GT SR / OS
| label | SR / OS |
|---|---|
| b0_slight-omit_fewshot-3_keyframe_deg-0_rooms-house-v2 (C+WP+방v2) | **37.4** / 50.1 (현재 최고, label→label·GT→label 평가 중) |
| b0_slight-omit_fewshot-3_rooms-house-v2 (C+방v2) | 35.9 / 48.9 |
| C+SS+방v2 | 35.6 / 48.0 |
| b0_stop_select-vlm-v1 / rotate-v1 / rule-v1 | 35.0 / 34.7 / 33.0 |
| b0_stop_cue-A-grounded-v1 | 34.6 |
- 방 이동(house-v2) 을 넣은 3개가 모두 상위. 정지 단서 계열은 b0(32.1) 보다 +1~3.
- 대기: slight-omit+WP+방v2 (node2), Qwen3-VL-32B / 235B 로 만든 C+방v2 (node3 / node4). 전체 표는 `scripts/dataset_converters/instruction_labeling/README.md`.

## 전체 2x2 + 실패 유형 (2026-10-20 KST, scripts/eval_dashboard/data.py build_task, 1,839 ep 확인)
칸 = SR (지나침 / 못 감) %. 지나침 = OS=1 & SR=0, 못 감 = 목표 3 m 안에 한 번도 못 감. GT→GT = 42.5 (약 10.0 / 47.5, result_h200 의 OS−SR).

| label | label→GT | label→label | GT→label |
|---|---|---|---|
| C+WP+방v2 | 37.4 (12.7/49.6) | 52.9 (12.4/33.3) | 44.0 (12.8/43.1) |
| C+방v2 | 35.9 (12.7/51.0) | 51.9 (11.8/35.4) | 41.4 (12.2/46.4) |
| C+SS+방v2 | 35.6 (12.3/51.8) | 51.2 (12.5/35.5) | 42.2 (12.6/45.1) |
| slight-omit+WP+방v2 | 35.3 (13.4/50.2) | 평가 중 | 대기 |
| stop_select-vlm | 35.0 (12.6/52.1) | 51.7 (14.5/33.6) | 40.3 (12.3/47.4) |
| stop_select-rotate | 34.7 (11.1/53.9) | 51.5 (11.4/36.8) | 40.1 (12.1/47.8) |
| stop_cue-A-grounded | 34.6 (12.2/52.1) | 51.9 (12.6/35.2) | 41.4 (11.3/47.3) |
| delivery-2stage | 33.8 (17.6/45.9) | 54.7 (12.6/30.8) | 38.1 (14.8/47.1) |
| C+방v2 Qwen3-VL-32B | 33.8 (14.0/51.1) | 47.8 (12.7/38.3) | 37.4 (14.7/47.9) |
| slight-omit | 33.8 (18.0/44.3) | 50.5 (15.0/33.7) | 39.4 (10.8/49.8) |
| C+WP | 33.7 (17.7/46.2) | 52.5 (13.5/32.4) | 43.6 (11.3/45.1) |
| C+SS | 33.7 (16.0/49.4) | 50.8 (12.8/35.1) | 38.9 (13.3/47.7) |
| stop_select-rule | 33.0 (11.9/54.9) | 48.6 (14.2/36.8) | 39.5 (12.7/47.8) |
| C | 32.7 (17.0/47.5) | 49.4 (13.2/36.0) | 39.5 (11.0/49.4) |
| b0 | 32.1 (15.8/49.5) | 56.3 (11.3/31.2) | 40.5 (11.7/47.7) |
| v278fix | 31.9 (8.1/60.0) | 31.9 (10.0/58.1) | 30.0 (13.4/56.5) |
| C+방v2 Qwen3-VL-235B | 30.6 (19.0/48.2) | 48.3 (14.6/34.9) | 38.8 (12.1/48.9) |

- 정지 단서(b0 위): label→GT +0.9~+2.9, 지나침 15.8→11~13 로 줄지만 못 감 49.5→52~55 로 늘어 일부 상쇄. GT→label 은 b0 와 ±1 (정보량 그대로). label→label 은 56.3→48.6~51.9.
- 방v2: 지나침을 줄이면서 못 감은 거의 그대로 → 정지 단서보다 순이득이 크다. 모든 waypoint(WP) 는 방v2 와 함께일 때 GT→label 44.0 (정보량 최고).
- b0 의 label→label 56.3 이 가장 높다 = 자기 문체(형제 3문장이 거의 같음)에 맞춰진 효과. 개선 label 은 label→label 이 내려가고 label→GT 가 오른다.
