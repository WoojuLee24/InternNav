# gepa_labeling 1~4단계: 손잡이 + GEPA 프롬프트 최적화 — 2026-10-24 KST

## 용어
- 손잡이 = 사람이 고르는 말투 축 (끝맺음 stop/wait/none · 설명 방식 turn/mixed/landmark · 길이 short/medium/long). 형제 3문장에 서로 다른 조합 (train GT 비율, (seed, trajectory) 고정)
- 사실 메모 = 번호 명령문 목록 대신 같은 경로 사실을 서술로 (route.facts=memo-v1)
- 대조군 사다리: 1 b1 → 2 b1_facts-memo-v1 (사실 표현) → 3 b1_facts-memo-v1_knobs-v1 (+손잡이) → 4 + writer-gepa-v1 (GEPA 로 고친 글)

## 코드
- instruction_labeling (기본값이면 cfg None, 기존 gen_hash 불변 확인: b0 aba44a796514, b1 a69461a93ffb):
  `label/facts.py` (route.facts), `label/knobs.py` (style.knobs), `config.py` (route.facts / style.knobs / style.writer), `e2e.py`·`style.py` guard clause
- gepa_labeling: `configs/gepa.yaml`, `prompt_opt/scorer.py` (채점), `check_scorer.py` (검증·비교 표), `adapter.py` (GEPAAdapter), `run_gepa.py`

## 채점기 (val_unseen dev 303)
- 사실 = 규칙만 (회전 방향·순서 부분열, 계단 위/아래, "stop in the <방>" 의 방, 최소 정보). 생략은 감점 안 함
- gemma 사실 검사는 껐다: 사람(GT) 문장 49 % 를 "모순" 이라 함 (지나는 계단을 "계단 없음" 등 거짓 경보)
- 규칙 사실 오류: 사람 8.6 %, b1 3.3 %, b1 left↔right 뒤집음 58.1 % (잡음)
- 점수 = 사실 × (0.5 손잡이 + 0.5 다양성)

## 사다리 1~4 (생성만, val_unseen dev 303, 규칙 채점기)
| 세트 | 사실 오류 | 손잡이 지킴 | 형제 첫3 같음 | 형제 내용어 겹침 | 점수 | 사람과 분포 차이 끝맺음/설명/길이 (TV×100) |
|---|---|---|---|---|---|---|
| 사람 | 8.6 % | — | 5.9 % | 0.20 | — | 0 / 0 / 0 |
| 1 b1 | 3.3 % | — | 77.6 % | 0.70 | — | 45 / 23 / 30 |
| 2 사실 메모 | 3.0 % | — | 72.6 % | 0.72 | — | 44 / 29 / 29 |
| 3 + 손잡이 | 3.6 % | 78.8 % | 57.1 % | 0.53 | 0.598 | 4 / 27 / 8 |
| 4 + GEPA (`b1_facts-memo-v1_knobs-v1_writer-gepa-v1`) | **2.3 %** | 79.5 % | **46.2 %** | **0.49** | **0.642** | 4 / 24 / 5 |
- 효과 나누기: 사실 표현 → 첫3 −5.0 %p / 손잡이 → 첫3 −15.5 %p, 끝맺음·길이 분포를 사람에 맞춤 / GEPA → 첫3 −10.9 %p, 사실 오류 3.6 → 2.3 %
- 설명 방식(turn 위주) 손잡이는 어느 단계도 따르지 않음 (turn 위주 1~5 % vs 사람 18 %). 형제 첫머리는 여전히 같은 경우가 많음 ("Exit the bathroom ...") — 첫머리는 경로가 정하는 축(0단계)
- 점수 3/4 (손잡이 없는 1·2 는 손잡이 점수 0 이라 점수 비교 안 함)

## GEPA v1 (완료 14:32 KST)
- 데이터: train 61 scene 중 8 scene 을 고르기용 (경로 100), 나머지에서 고치기용 경로 100. 예산 1,500 경로 평가
- 실행: `/home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/prompt_opt/run_gepa.py --run v1` → `logs/gepa_labeling/gepa/v1/`
- 결과: 고르기용 경로 100 에서 seed 0.604 → best 0.654, 후보 14 (`logs/gepa_labeling/gepa/v1/best.json`, `candidates.json`)
- best 글은 `instruction_labeling/configs/style.yaml` `writer.gepa-v1` 에 그대로 저장 (yaml 왕복 일치 확인)
  - style_instruction: "Strictly follow the path facts. Write the instruction in this style: {granularity}. If 'turn', focus on directional changes and avoid naming objects. ... Length must be {length}. Ending must be {ending}. To ensure diversity from other versions, vary your sentence structure, choose different landmarks, and start with a unique opening word. ..."
  - facts_header: "PATH FACTS (Use these as a strict ground-truth for accuracy, but do not mimic their structure, phrasing, or sequence. ...)"
  - 주의: "If 'turn'" 은 손잡이 이름을 가리키지만 실제 빈칸에는 설명 문장이 들어간다 (모델이 손잡이 이름을 모름)
- 채점기 개선은 다른 세션에서 진행 (2026-10-24). 채점기를 고치면 GEPA 를 다시 돌린다

## 다음 (승인 대기)
- 5단계: 대조군 2·3·4 val_unseen 전체 생성 → 정보량 검사 (각 1.8 h) → 통과한 것 2x2

## 추가 시험 (2026-10-24 오후 KST): 형제 함께 쓰기 + turn 위주 시 랜드마크 규칙 빼기
- `style.knobs=v2`: v1 + 설명 방식 turn 이면 detail 줄("Reference specific visible landmarks") 을 뺀다
- `delivery=joint3-v1` (`label/joint.py`, `configs/delivery/joint3-v1.yaml`): 같은 경로의 형제 문장을 한 요청으로 "서로 다른 n 명" 이 쓰게 (사람마다 손잡이 한 줄). trajectory 별 lock 으로 첫 형제 호출이 전부 만들고 나머지는 가져감
- 기존 이름 hash 불변 (b0 aba44a796514, b1 a69461a93ffb, 2 b1c8aa85b8ba, 3 39b2d1cc68a6)

| 세트 (dev 303) | 사실 오류 | 손잡이 지킴 | 형제 첫3 같음 | 형제 내용어 겹침 | 점수 | 분포 차이 끝맺음/설명/길이 | 첫 단어 상위 |
|---|---|---|---|---|---|---|---|
| 사람 | 8.6 % | — | 5.9 % | 0.20 | — | 0/0/0 | walk 36, go 17, turn 16, exit 11 % |
| 3 knobs-v1 | 3.6 % | 78.8 % | 57.1 % | 0.53 | 0.598 | 4/27/8 | |
| 4 knobs-v1 + GEPA | 2.3 % | 79.5 % | 46.2 % | 0.49 | 0.642 | 4/24/5 | turn 57, walk 18, begin 10 % |
| knobs-v2 | 4.0 % | 79.3 % | 54.8 % | 0.53 | 0.604 | 4/25/7 | turn 62, walk 30 % |
| **knobs-v2 + joint3** | 3.6 % | **82.6 %** | **14.5 %** | **0.32** | **0.768** | 4/20/7 | turn 47, walk 18, start 13, go 9 % |
- 랜드마크 규칙 빼기(v2) 는 효과 거의 없음 (turn 위주 3 → 5 %) → 원인은 그 규칙이 아님
- **함께 쓰기가 다양성을 크게 올림** (첫3 같음 57 → 14.5 %, 겹침 0.53 → 0.32), 사실 오류는 그대로
- 남은 문제: 첫 단어 turn 47 % (사람 16 %) — 사실 메모의 출발 회전이 "Turn right from the dining room ..." 으로 이어짐. turn 위주 설명 방식은 여전히 안 따름 (7 % vs 사람 18 %)

## GEPA v2-joint (2026-10-24 15:21~15:33 KST)
- 출발: route.facts=memo-v1 + style.knobs=v2 + delivery=joint3-v1 (seed 글은 v1 과 같음). 같은 train scene 분할·예산 1,500. 채점기는 오전에 검증한 규칙판 그대로 (다른 세션이 채점기 개선 중 — check_mutations.py 추가만, scorer.py 미변경)
- 고르기용 경로 100: seed 0.760 → best 0.821 (후보 14). `logs/gepa_labeling/gepa/v2-joint/best.json` → style.yaml `writer.gepa-v2-joint`

| 세트 (dev 303) | 사실 오류 | 손잡이 지킴 | 형제 첫3 같음 | 형제 내용어 겹침 | 점수 | 분포 차이 끝맺음/설명/길이 | 첫 단어 상위 |
|---|---|---|---|---|---|---|---|
| 사람 | 8.6 % | — | 5.9 % | 0.20 | — | 0/0/0 | walk 36, go 17, turn 16, exit 11 % |
| knobs-v2 + joint3 | 3.6 % | 82.6 % | 14.5 % | 0.32 | 0.768 | 4/20/7 | turn 47, walk 18, start 13 % |
| **+ GEPA v2-joint** | **2.0 %** | 81.2 % | **1.7 %** | **0.27** | **0.818** | 4/16/7 | turn 31, **start 20**, walk 11, go 9 % |
- 형제 다양성은 사람 수준 (첫3 같음 1.7 % < 사람 5.9 %, 겹침 0.27 vs 0.20), 사실 오류 최저
- 새 버릇: "Start by ..." 로 시작 20 % (GEPA 글의 "entirely different opening words" 지시 부작용, 사람 문장엔 드묾). 첫 단어 분포는 아직 사람과 다름 (walk 11 vs 36 %)
- 설명 방식 turn 위주 8 % (사람 18 %) — 여전히 부족

## GEPA v3 — 새 채점기 (모델 사실 검사, 다른 세션 작성) 로 다시 (2026-10-24 17:18~17:59 KST)
- 같은 joint 설정·데이터·예산. 사실 검사기만 gemma(1+2단계) / Sonnet 5.5(1+2단계). 채점기 검증은 `261024_scorer_claude_judge_result.md`
- 고르기용 경로 100: gemma 채점 0.698 → 0.778 (`v3-joint-gemma`), Sonnet 채점 0.764 → 0.806 (`v3-joint-sonnet`). 둘 다 style.yaml `writer` 에 저장
- dev 303 비교 (사실 오류는 채점기별; 형제 첫3 같음·분포는 채점기와 무관)

| 세트 | 사실 오류 (gemma 채점 / Sonnet 채점) | 손잡이 지킴 | 형제 첫3 같음 | 겹침 | 분포 차이 끝맺음/설명/길이 | 첫 단어 상위 |
|---|---|---|---|---|---|---|
| 사람 | 21.1 / 6.6 % | — | 5.9 % | 0.20 | 0/0/0 | walk 36, go 17, turn 16 % |
| '전' (joint, GEPA 전) | 13.2 / 7.9 % | 82.6 % | 14.5 % | 0.32 | 4/20/7 | turn 47, walk 18, start 13 % |
| v2-joint (규칙 채점 GEPA) | 11.6 / 6.9 % | 81.2 % | 1.7 % | 0.27 | 4/16/7 | turn 31, start 20 % |
| **v3-joint-gemma** | **8.9 / 4.0 %** | 78.8 % | 3.3 % | 0.28 | 4/**11**/12 | turn 34, start 19, walk 12 % |
| v3-joint-sonnet | 9.6 / 5.9 % | 79.5 % | 9.2 % | 0.31 | 4/15/12 | turn 43, start 19 % |
- gemma 채점으로 고친 글이 두 채점기 모두에서 사실 오류 최저 (Sonnet 채점 4.0 % < 사람 6.6 %), 설명 방식도 사람에 가장 가까움 (turn 위주 12 % vs 사람 18 %)
- Sonnet 채점으로 고친 글은 다양성이 덜 오름 (9.2 %). 이유 미확인 (Sonnet 채점이 엄격해 사실 쪽으로 기울었을 가능성)
- 공통 문제: "Start by" 시작 약 19 %, 길이 분포 차이가 7 → 12 로 커짐 (GEPA 가 length 풀이를 "short = 한 문장, medium = 두 문장" 으로 바꿈)
