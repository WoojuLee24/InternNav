# gepa_labeling — 말투 유형별 자연어 프롬프트 최적화 (GEPA)

instruction_labeling 의 생성 코드(경로 사실·요청 만들기·설정)를 그대로 불러 쓰고,
말투 지시 문단만 [GEPA](https://github.com/gepa-ai/gepa) 로 유형별로 고친다. 결과 프롬프트는 영어 문장 그대로 남는다.

## 환경
- venv: `/home/irteam/venvs/gepa` (`--system-site-packages`, `gepa==0.1.4` 만 추가)
- 실행: `/home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/<script>.py ...`
- 문장 생성·프롬프트 수정 모두 node1 gemma-4-31B vLLM 서버 (:8100)

## 단계 (예정)
0. 말투 유형 나누기: train GT 문장 → 규칙 + gemma 판정, 사람 판정 200개 비교, 형제 검사
1. 채점기 + GEPA adapter: 유형 일치 / 사실 정확성 / 사람 문장 닮음 (점수 + 이유, 사례 사진 1장)
2. 유형별 GEPA: 고치기 train 경로, 고르기 val_seen 경로 (val_unseen 은 최종 정보량 검사·2x2 에만)
3. 형제 3개에 서로 다른 유형 → val_unseen 정보량 검사 → 2x2
