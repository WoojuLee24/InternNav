# 작성자 3명(w3-v1) dev 결과 — 2026-10-23 18:40 KST

목적: 형제 3문장을 서로 다른 "작성자"가 쓰게 해 GT 처럼 다양하게 (GT 통계 사용 없음). 이동 정보는 숨기지 않고 그림으로 준다.

| 작성자 | 이름 | 입력 |
|---|---|---|
| A | `b1` | 사진 + 번호 목록 + 방 순서 (기존) |
| B | `b1_overlay-arrow-v1_motion-none` | waypoint 도착 사진(돌기 전) 위에 진행 방향 화살표(아래 흰 원) + 바닥 초록선, 번호 목록 없음, 방 순서 줄 유지 |
| C | `b1_select-topdown-v1_motion-none` | .house 방 다각형 평면도 + 경로(출발 방향이 위) + 출발·도착 사진, 번호 목록 없음, 방 순서 줄 유지 |
| 섞음 | `w3-v1` (아직 파일 없음) | 경로마다 A·B·C 를 형제 3개에 무작위 배정 (seed `0:<traj>:w3`) |

코드: `label/visual.py` (e2e.build_request 가 guard clause 로 위임), `label/viz_request.py` (요청 이미지·프롬프트 저장),
config `frames.overlay`, `frames.select=topdown-v1`, `route.motion` (기본값이면 cfg None → 기존 gen_hash 불변 확인: b0 aba44a796514, b1 a69461a93ffb).

명령:
- `/usr/bin/python scripts/dataset_converters/instruction_labeling/label/annotate.py --method b1 --set frames.overlay=arrow-v1 --set route.motion=none --split val_unseen --subset dev`
- `/usr/bin/python scripts/dataset_converters/instruction_labeling/label/annotate.py --method b1 --set frames.select=topdown-v1 --set route.motion=none --split val_unseen --subset dev`
- 시각화: `label/viz_request.py --method b1 --set <위와 같음> --n 4` → `logs/instruction_labeling/viz_request/<이름>/ep_*.jpg`

dev 303 episode (101 trajectory):

| 세트 | 첫머리 이동/방나감/회전 % | 끝 stop/wait/없음 % | 단어 | 회전 덜 말함 | 도착방 | 형제 첫3 같음 | 형제 내용어 겹침 | 어휘 |
|---|---|---|---|---|---|---|---|---|
| GT | 56/17/17 | 53/29/16 | 27.2±11.3 | 31 % | 34 % | 5.9 % | 0.20 | 491 |
| A b1 | 30/6/64 | 98/0/2 | 26.4±3.0 | 0 % | 94 % | 77.6 % | 0.70 | 233 |
| B 화살표 | 36/45/17 | 97/0/3 | 26.4±3.2 | 49 % | 92 % | 48.5 % | 0.64 | 244 |
| C 지도 | 49/23/27 | 95/0/5 | 23.9±3.2 | 47 % | 99 % | 42.6 % | 0.64 | 185 |
| **w3 섞음** | 39/24/36 | 96/0/4 | 25.6±3.4 | 31 % | 96 % | **17.5 %** | **0.45** | 253 |

- 작성자끼리 같은 episode 내용어 겹침 A-B 0.49, A-C 0.49, B-C 0.46 (모델만 바꾼 Qwen 0.46 수준).
- 출발 회전 ≥90° 에서 회전으로 시작: GT 24 %, A 100 %, B 17 %, C 44 %.
- 안 바뀐 것: 끝맺음 stop 96 %, 도착 방 92~99 %, 길이 편차 3 (다음 단계 과제).
- 남은 확인: B·C 가 회전을 덜 말하는 것이 정보 손실인지 → val_unseen 전체 생성 + 정보량(GT→label) 검사.

## 진행 (2026-10-23 KST)
- C 의 평면도를 처음엔 pyplot 으로 그려 64 스레드에서 그림이 섞일 위험 → Figure API + lock 으로 고치고 C 전체 재생성 (위 표는 재생성 기준).
- 전체 생성: B·C 각각 train 10,684 / val_unseen 1,839, 오류 0, 품질검사 실패 B 14·33, C 27·109 (val·train).
- commit 3d106fc. node2 큐 (`sleep 1800` 앞, 백업 node2.txt.bak_1023_w3info): 준비 → 정보량 B → 정보량 C. 예상 완료 약 23:50 KST.

## 정보량 검사 결과 (node2, 2026-10-24 00:40 KST 확인)
정보량 = GT 문장으로 학습한 모델(학습 없음)에 label 문장을 넣은 SR. val_unseen 1,839 ep, 같은 episode 집합.

| 문장 | SR |
|---|---|
| GT | 42.5 (result row, 1839) |
| b1 (작성자 A, cell 이름 b0_slight-omit_fewshot-3_keyframe_deg-0_rooms-house-v2) | 44.0 |
| b1 + 출발 좌우뒤(실험 2) | 43.6 |
| **B 화살표** | **36.6** (b1 대비 −7.4 %p) |
| **C 지도** | **31.8** (b1 대비 −12.2 %p) |
| v278fix | 30.0 |

- B·C 모두 정보 손실. dev 에서 본 "회전 덜 말함 49 % / 47 %" 가 실제 손실로 나타남 (C 는 v278fix 수준).
- 결과: `/home/irteam/data-vol2/checkpoints/relabel_half/dualvln_mini_half_s1.train.gt_eval.gt_internvla-n1-system2_20261008_063318/logs/relabel_half_..._eval.{b1_overlay-arrow-v1_motion-none,b1_select-topdown-v1_motion-none}_internvla-n1-system2/`

## B·C 손실 분해 (2026-10-24 KST)
명령: `/usr/bin/python scripts/dataset_converters/instruction_labeling/eval/analyze_info_loss.py --labels b1 b1_overlay-arrow-v1_motion-none b1_select-topdown-v1_motion-none --short A B C --out logs/instruction_labeling/info_loss/w3`
(출력 `summary.md`, `path_*.png`, `features.json`. eval 결과는 `scripts/eval_dashboard/data.py` reader 로 읽음, 공통 1839 ep)

| 경로 | n | A | B | C |
|---|---|---|---|---|
| 곧은 경로 (출발 <30°, 중간 회전 0, 계단 X) | 84 | 52.4 | 59.5 | 51.2 |
| 출발 <30°, 중간 회전 있음, 계단 X | 153 | 54.9 | 43.1 | 41.8 |
| 중간 회전 2개 | 390 | 44.1 | 30.8 | 23.8 |
| 출발 회전 ≥135° | 573 | 41.5 | 36.3 | 26.4 |
| 계단 있음 | 492 | 32.5 | 21.3 | 21.5 |

- **손실은 회전·계단에서만 생긴다.** 곧은 경로는 손실 없음 (B 는 오히려 +7).
- 원인은 **방향을 틀리는 것이 아니라 빠뜨리는 것**: 출발 ≥45° 에서 방향을 틀린 문장은 B 1 %·C 2 %. 회전 단어가 경로보다 적은 문장 A 14 % / B 59 % / C 52 %, 그 문장 SR 35.1 / 32.3 / 28.3.
- 출발 ≥90° 에서 회전으로 시작: A 99 % / B 25 % / C 37 %. B 는 "Leave the X" 로 시작(51 %)하면서 방향을 안 말함.
- 계단: 오르기/내려가기를 맞게 말한 비율 A 442/492, B 147, C 62 (나머지는 계단·방향 생략 또는 반대).
- 실패 유형: 못 감(never_reached) A 43.1 → B 52.4 / C 55.7. 지나침은 비슷 (12.8 / 10.9 / 12.2).

## B·C 문장은 b1(A) 과 얼마나 다른가 (val_unseen 1,839, 2026-10-24 KST)
내용어 = text_metrics.STOP(방향·이동 동사 포함) 제외, 3글자 이상 단어. 겹침 = Jaccard.

| 비교 | 첫 3단어 같음 % | 내용어 겹침 | 2단어 묶음 겹침 |
|---|---|---|---|
| A vs B (같은 episode) | 20.3 | 0.49 | 0.29 |
| A vs C | 21.2 | 0.44 | 0.26 |
| B vs C | 26.6 | 0.42 | 0.25 |
| A/B/C vs GT | 4~6 | 0.13~0.14 | 0.07~0.08 |
| GT 형제끼리 (사람끼리) | 4.7 | 0.15 | 0.08 |
| A / B / C 형제끼리 | 75.5 / 50.4 / 41.5 | 0.65 / 0.63 / 0.64 | 0.52 / 0.43 / 0.43 |

- B·C 는 A 와 첫머리·표현은 꽤 다르지만(첫 3단어 같음 20 %), 쓰는 내용어는 절반쯤 같다 (0.44~0.49). 사람끼리(0.15) 보다 훨씬 비슷.
- 작성자 안에서는 내용어 겹침이 그대로 0.63~0.64 → 그림이 바꾼 건 첫머리(문장 구조)이지 고르는 물체·방이 아니다.
