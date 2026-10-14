# instruction_labeling P2/P3 — b0 baseline (dev subset) 결과 (2026-10-13 23시 KST)

코드: `scripts/dataset_converters/instruction_labeling/label/` (annotate, e2e, frames, route_words, sources),
`eval/text_metrics.py`, `eval/dev_subset.json`. 설정 `configs/labels/b0.yaml`.

## 용어
| 용어 | 뜻 |
|---|---|
| b0 | vln-s2-pipeline 의 e2e baseline 프롬프트 그대로 (이미지 + 숫자 없는 경로 단어 -> VLM 1회 호출) |
| b0_ours | 이번 이식본: 입력 이미지 = InternNav 조건 렌더(val_unseen, P1), 프레임 = 실제 궤적에서 start/회전점/goal |
| b0_s2 | vln-s2-pipeline 의 원래 b0 결과: 같은 프롬프트·모델, 프레임은 waypoint 에 카메라를 놓고 렌더 (heading 거울상 버그 포함) |
| dev subset | val_unseen 101 trajectory / 303 episode (scene 비율 유지, seed 0) |
| GT 행 | 사람 지시문 1개를 같은 경로의 나머지 2개로 잰 값 = 사람 상한. label 행도 **같은 2개**를 참조로 잰다 |
| sibling_BLEU-4 | 같은 경로의 생성 문장끼리 BLEU-4. 높을수록 3문장이 서로 똑같다 (GT 0.13) |
| goal_mismatch | 마지막 문장이 GT 마지막 문장들과 명사를 하나도 공유하지 않는 비율 |

## 실행
- 모델: 기존 gemma-4-31B-it vLLM 서버 (:8100, DP8). 303 episode 16 초. 예외 0, 품질 실패 3 (no_landmark).
- 명령:
  - `/usr/bin/python scripts/dataset_converters/instruction_labeling/label/annotate.py --label b0 --split val_unseen --subset dev`
  - `PYTHONPATH= /home/irteam/venvs/navcomposer/bin/python scripts/dataset_converters/instruction_labeling/eval/text_metrics.py --split val_unseen --subset dev --label b0=<json.gz> --out logs/instruction_labeling/p3/b0_dev`

## 결과 (dev 303 episode, 같은 참조)
| 지표 | GT(상한) | **b0_ours** | b0_s2 | v278fix (SR 31.9) | v278fix_fewshot-3 (SR 33.6) |
|---|---|---|---|---|---|
| BLEU-4 | 0.133 | **0.113** | 0.107 | 0.102 | 0.106 |
| METEOR | 0.202 | **0.201** | 0.192 | 0.180 | 0.188 |
| ROUGE-L | 0.383 | **0.368** | 0.357 | 0.343 | 0.353 |
| CIDEr | 0.328 | **0.237** | 0.199 | 0.174 | 0.164 |
| noun_precision | 0.428 | **0.322** | 0.290 | 0.246 | 0.245 |
| goal_mismatch % | 35.3 | **50.5** | 56.8 | 61.4 | 62.0 |
| first_turn_vs_gt % | 7.6 | 6.9 | 7.6 | 10.2 | 10.6 |
| turn_around % | 7.6 | 37.6 | 37.0 | 0.0 | 0.0 |
| stop/wait/stand/none % | 55/29/3/13 | 100/0/0/0 | 100/0/0/0 | 100/0/0/0 | 100/0/0/0 |
| unique_ratio | 1.00 | 0.82 | 0.83 | 0.97 | 0.99 |
| sibling_BLEU-4 | 0.13 | **0.86** | 0.84 | 0.73 | 0.67 |
| content_vocab | 462 | 205 | 213 | 256 | 265 |
| 첫 단어 | walk 36 / go 17 / turn 16 | turn 68 / walk 16 / bear 13 | 같음 | go 96 | go 55 / walk 41 |

- **b0_ours 가 문장 지표 전부에서 기존 label 보다 GT 에 가깝다.** b0_s2 와는 프롬프트·모델이 같고 입력 프레임만 다르다
  → InternNav 조건 렌더 + 실제 궤적 프레임(+ 올바른 방향)이 CIDEr 0.199→0.237, goal_mismatch 56.8→50.5.
  (단 b0_s2 는 거울상 방향 프레임이라 "방향 버그 수정"과 "렌더 조건"이 섞여 있다 — 순수 분리 안 됨.)
- CIDEr 는 사람 상한의 72 %.
- **b0 의 약점 (다음 ablation 후보)**:
  1. 같은 경로 3문장이 거의 같다 (sibling BLEU-4 0.86 vs GT 0.13, 고유 82 %). 프롬프트가 3 episode 에 동일, temperature 0.35.
  2. 시작 회전을 항상 말한다 (turn_around 37.6 % vs GT 7.6 %, 첫 단어 turn 68 %). route words 가 시작 회전을 항상 넣기 때문.
  3. 정지 표현이 100 % "stop" (GT stop 55 / wait 29 / none 13).
  4. 계단 방향 오류 1.7 %, 계단 없는데 언급 1.0 % (GT 0).
- SPICE 는 실패 (CalledProcessError: Java 버전과 SPICE 의 FST 직렬화 라이브러리 충돌). 다른 지표로 진행, TODO.
- 문장 지표는 1차 필터일 뿐이다 (LESSONS §4: 지표 1위가 SR +0 이었던 적 있음). 판정은 P5/P6.

## ablation: b0_2stage-v1 (장면 정보 전달 방식만 v278fix 로) — 추가 (2026-10-13 23:40 KST)
- `configs/labels/b0_2stage-v1.yaml` = `base: b0` + `scene_delivery: 2stage-v1` 두 줄. 방식 정의는 `configs/scene_delivery/2stage-v1.yaml`(버전 고정),
  생성기 `label/two_stage.py`, 실행은 같은 `annotate.py --label b0_2stage-v1`. (처음엔 `b0_2stage` 이름이었고 같은 내용)
- 바꾼 것 하나: writer 가 이미지를 직접 보는 대신, stage 1 이 프레임마다 장면을 글로 요약(v278fix gate3 프롬프트·temperature 0.4·150 토큰)
  + 도착 랜드마크 1문장 → writer 는 그 글만 본다. 프레임·캡션·route words·규칙·생성 파라미터·모델·seed 는 b0 와 동일.
- b0 회귀 확인: 리팩터 후 b0 6 episode 재생성 결과가 이전과 문장까지 6/6 동일.
- dev 303 ep, 28 초, 예외 0, 품질 실패 0, 장면 요약 파싱 실패 0.

| 지표 | GT(상한) | b0 | b0_2stage | v278fix |
|---|---|---|---|---|
| BLEU-4 | 0.133 | 0.113 | 0.113 | 0.102 |
| METEOR | 0.202 | 0.201 | 0.192 | 0.180 |
| CIDEr | 0.328 | **0.237** | 0.196 | 0.174 |
| noun_precision | 0.428 | **0.322** | 0.229 | 0.246 |
| extra_nouns /문장 | 2.16 | **2.69** | 3.94 | 3.66 |
| goal_mismatch % | 35.3 | **50.5** | 63.4 | 61.4 |
| stairs 오류 (방향/없는데) % | 0/0 | 1.7/1.0 | **0/0** | 0/0 |
| unique_ratio | 1.00 | 0.82 | **0.94** | 0.97 |
| sibling_BLEU-4 | 0.13 | 0.86 | **0.79** | 0.73 |
| content_vocab | 462 | 205 | **289** | 256 |

- **랜드마크 정확도는 이미지를 직접 보는 b0 가 확실히 낫다**: 글 요약을 거치면 CIDEr −17 %, 명사 정확도 0.32→0.23,
  GT 에 없는 명사 +1.25 개, 도착 지점 불일치 +13 %p. → LESSONS §4 "랜드마크 구체성 부족의 원인은 gate3" 를 같은 조건 대조로 확인.
- 2stage 가 나은 점: 다양성(고유 94 %, 형제 유사도 0.79)과 계단 오류 0. 다양성은 형제 episode 마다 장면 요약이 seed 로 달라지는 효과로 보인다.
- b0_2stage ≈ v278fix (CIDEr 0.196 vs 0.174): 남은 차이는 route words(시작 회전 포함)·모델 양자화·입력 프레임.
- 결론(문장 지표 기준): baseline 은 b0 유지. 다양성은 다른 방식(형제 문장 다르게 쓰기 등)으로 따로 ablation.

## 구조 변경: 장면 정보 전달 방식 = yaml 인자 `scene_delivery` (2026-10-13 24시 KST)
- label yaml 의 `scene_delivery: <name>` 이 `configs/scene_delivery/<name>.yaml` 을 가리킨다: `image`(b0 원래 방식), `2stage-v1`.
  정의 내용이 gen_hash 에 들어가므로 정의를 고치면 결과와 섞이지 않는다. 바꿀 때는 `2stage-v2.yaml` 을 새로 만든다.
- 다른 base 에도 두 줄로 적용: `label: b1_2stage-v1 / base: b1 / scene_delivery: 2stage-v1`.
- template 에 `{scene_intro}` / `{scene_context}` 자리를 두고 방식 정의가 채운다. 렌더된 프롬프트는 이전과 문자 단위로 동일함을 확인.
- 템플릿 텍스트가 바뀌어 gen_hash 가 바뀌었으므로 b0 / b0_2stage-v1 dev 를 재생성했다. 이전 산출물은 `instruction_labels/_superseded/`.

## 재실행 잡음 (같은 설정·같은 seed 를 두 번)
- 프롬프트·장면 요약은 같았는데 최종 문장이 b0 279/303, 2stage 227/303 만 같았다 → **vLLM 동시 배치 때문에 seed 를 고정해도 출력이 완전히 재현되지 않는다**.
- 지표 흔들림: CIDEr ±0.005, goal_mismatch ±1.4 %p, noun_precision ±0.001. (b0 vs 2stage 차 CIDEr 0.04 는 잡음의 약 8 배 → 유의)
- ablation 은 이 잡음 폭보다 작은 차이로 결론 내지 않는다.

| 지표 | b0 run1 | b0 run2 | 2stage run1 | 2stage-v1 run2 |
|---|---|---|---|---|
| CIDEr | 0.237 | 0.235 | 0.196 | 0.191 |
| METEOR | 0.201 | 0.201 | 0.192 | 0.191 |
| noun_precision | 0.322 | 0.323 | 0.229 | 0.229 |
| goal_mismatch % | 50.5 | 49.8 | 63.4 | 62.0 |
