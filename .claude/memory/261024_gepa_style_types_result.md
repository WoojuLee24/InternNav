# gepa_labeling 0단계: 사람(GT) 문장 말투 유형 (1차) — 2026-10-24 KST

코드: `scripts/dataset_converters/gepa_labeling/` (venv `/home/irteam/venvs/gepa`), 설정 `configs/style_types.yaml`
- `style_types/extract.py --split {train,val_seen,val_unseen} --stage {rules,judge}`: 규칙 특징 / gemma 판정(설명 방식 step·landmark·goal, 프롬프트 2개)
- `style_types/cluster.py`: train 으로 GMM(diag) k=4~8 BIC → val 에 유형만 붙임, `model.pkl` 저장
- `style_types/human_sheet.py sheet|agree`: 사람 판정 200개 시트(`_style_types/human_200.csv`) / 일치도
- 데이터: `data/InternData-N1-v0.5-mini/instruction_labels/_style_types/`, 보고서 `logs/gepa_labeling/style_types/types_report.md`

## gemma 판정 안정성 (같은 문장, 프롬프트 2개)
- 처음 3축(설명 방식·정지 위치·자세함): 40개 중 60~68 % 일치 → 정지 위치는 규칙(마지막 문장의 물체/방 단어)으로, 자세함은 뺌
- 설명 방식만, 기준을 구체화: 150개 77~79 %, train 전체 77.4 % (어긋남 대부분 step↔landmark)

## 1차 묶기 결과 (train 10,819)
- BIC 가 k=8(범위 끝)까지 계속 내려감 → 최적이 아니라 상한. prob≥0.9 가 99 % → 숫자·then 같은 0/1 특징 하나로 딱 갈림
- 형제 같은 유형 22.9 % vs 무작위 17.3 % (1.3 배) → 경로 영향이 조금 섞임. 유형 3 은 "출발 회전 말함" 100 % (출발 때 돌아야 하는 경로에 몰림)
- 뚜렷한 유형: "Go to the X" 짧은 목적지형 (6.6 %), 회전으로 시작형 (17 %), "With/Once…" 상황 설명형 (2.7 %, 대문자 포함)
- 나머지는 숫자 사용·then 사용 여부로 갈린 것이라 "사람 버릇" 으로 보기 어려움

## 다음 (승인 대기)
- 경로가 정하는 특징(출발 회전 말함 여부) 빼기, 0/1 글 습관(숫자·then·대문자) 비중 낮추기
- 유형은 사람이 읽을 수 있는 소수 축의 조합으로 직접 정의하는 안 (설명 방식 × 첫머리 × 끝맺음 × 길이) 과 비교

## 2차 (B 방식) — 2026-10-24 KST
- 설명 방식 = 규칙 비율 step_share = 회전 명령 수 / (회전 명령 + 지나는 것 표현) (train: ≤0.4 58 %, 중간 24 %, ≥0.6 17 %)
- gemma 는 "목적지만 말하나" 만: 두 프롬프트 일치 96 % (150), 전체 train yes 1 % / 애매 3 % (대부분 "Go to the X" 를 여러 줄 나열한 문장)
- GMM (k 4~10, 실루엣으로 선택): 실루엣 0.02~0.04 → **자연스러운 묶음이 없다** (사람 말투는 연속적). k=7 은 첫머리·끝맺음으로 잘린 것. 형제 같은 유형 35 % vs 무작위 24 %
- 축별 경로 영향 (형제 같은 값 비율 / 무작위 기대, 1 = 사람 버릇):
  끝맺음 0.99, then 1.01, 목적지만 1.01, 문장 수 1.05, 설명 방식 1.07, 길이 1.09, 숫자 1.11 → **사람이 고르는 축**
  정지 위치 표현 1.21, 첫머리 1.36, 회전 말한 비율 1.52, 출발 회전 말함 1.85 → **경로에 따라 정해지는 축**
- 결론: 유형을 "묶음" 으로 정하지 말고, 사람이 고르는 축(끝맺음·설명 방식·길이)을 손잡이로 쓰고, 경로가 정하는 축(첫머리 등)은 경로 사실에 맡긴다
