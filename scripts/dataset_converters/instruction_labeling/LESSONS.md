# LESSONS — vln-s2-pipeline 에서 배운 것

이 파일은 `/home/irteam/git/vln-s2-pipeline`(R2R 지시문 자동 생성기)에서 얻은 교훈을 정리한다.
instruction_labeling 을 설계하고 구현할 때 같은 실수를 반복하지 않기 위한 체크리스트다.
2026-10-13 (KST) 에 작성했다.

## 용어
- **label**: 같은 R2R 경로에 자동 생성기로 새로 쓴 지시문.
- **gate3**: 프레임을 VLM 에 넣어 장면을 묘사하게 하는 단계 (방 이름, 랜드마크).
- **gate4**: 장면 묘사와 경로를 받아 문장을 쓰는 단계.
- **e2e**: 이미지 여러 장을 VLM 한 번 호출에 바로 넣어 문장을 쓰는 방식.
- **SR**: 성공률. 멈춘 위치가 목표 3 m 안이면 성공. val_unseen 1,839 episode 로 잰다.
- **2x2**: {학습 문장 GT / label} × {평가 문장 GT / label}. 경로와 이미지는 같고 문장만 바꾼다.

## 기준 수치
조건: mini, s1, 1 epoch, seed 1회, ld30. 출처는 `relabel_vlnce/report_relabel_2x2_20261013.md`.

| 학습 문장 → 평가 문장 | SR |
|---|---|
| GT → GT | 42.5 |
| 최고 label (v278fix_fewshot-3) → GT | 33.6 |
| v278fix → GT | 31.9 |
| GT → label | 27.8 ~ 30.9 (label 이 경로 정보를 12~15%p 잃음) |
| 템플릿 label (v218fix) → 자기 문장 / GT | 40.5 / 27.3 (자기 문장에 과적합) |

## 1. 조용한 실패가 가장 비쌌다
- gate3 캐시가 없으면 템플릿으로 조용히 떨어졌다. 그 결과 train 문장의 내용어가 27종뿐이었다 (GT 는 2,228종).
- VLM 출력 파서가 형식(`ROOM:|LANDMARKS:`)을 놓치면 `room=unknown` 이 에러 없이 들어갔다.
- 경로 분석이 예외를 삼키고 "Walk forward and stop." 으로 대체했다.

→ fallback 은 금지한다. 쓸 수밖에 없다면 **반드시 카운트하고, 기준치를 넘으면 중단**한다.
→ 생성이 끝나면 내용어 수, 고유 문장 비율, 첫 단어 분포를 자동으로 검사한다.
   v6 은 고유 문장이 17% 라서 학습이 붕괴했다 (SR 8.7).

## 2. 좌표와 인덱스 버그는 그림으로만 잡힌다
- waypoint 인덱스를 렌더 프레임 인덱스로 잘못 써서 회전 프레임 대부분이 빠졌다.
- `heading_to_quat` 의 부호가 반대여서 회전 프레임과 goal 프레임이 **거울상 방향**을 봤다.
  v278fix 계열 전체가 이 상태로 만들어졌다 (2026-10-13 발견).

→ 렌더한 프레임은 top-down 경로와 카메라 방향을 함께 그림으로 저장해 눈으로 확인한다.
→ 렌더러 자체는 **InternData 사전 렌더 이미지와 픽셀 단위로 비교**해 검증한다.

## 3. 여러 변경을 한꺼번에 넣지 않는다
- HFOV 90→79, 검은 프레임 제거, key frame 매핑 수정을 한 번에 넣었다. 그래서 각각의 효과를 모른다.
- 실제로 SR 로 검증한 것은 gate4 프롬프트 변형뿐이고, 그것도 전부 ±2%p 안이다 (seed 1회).
- 다음 항목들은 SR 로 검증한 적이 없다:
  - e2e vs gate3
  - waypoint 프레임과 goal 측면 시점 (v3/v4)
  - AWQ vs bf16
  - 품질 게이트 G1~G7

→ ablation 은 **변수 하나씩** 바꾼다. 대조군은 그 자리에서 다시 돌린다.
→ 기준선은 seed 를 여러 번 돌린다. ±2%p 로 결론 내리지 않는다.

## 4. 문장 지표와 SR 은 따로 논다
- fewshot-5 는 문장 지표가 1위였지만 SR 은 +0.0 이었다.
- 랜드마크 구체성(CIDEr 0.13, GT 0.29)은 어떤 프롬프트로도 움직이지 않았다. 원인은 생성 단계가 아니라 장면 인식 단계(gate3)로 보인다.
- LLM-judge 를 생성 모델과 같은 계열(Gemma)로 돌렸다. 편향 가능성이 있다.

→ 문장 지표는 **1차 필터**로만 쓴다. 최종 판정은 follower 평가(GT 학습 모델에 label 문장으로 평가)와 학습 2x2 로 한다.
→ judge 모델은 생성 모델과 **다른 계열**을 쓴다.

## 5. 출력 규약 (어기면 eval 이 죽거나 조용히 틀린다)
- episode 키는 GT 키의 **부분집합**이어야 한다. 키가 더 있으면 habitat 의 `VLNEpisode(**ep)` 가 죽는다.
  학습은 통과하기 때문에 늦게 발견된다.
- 문장은 `". "` 로 끝나야 한다. evaluator 가 마지막 글자를 `[:-1]` 로 자른다.
- `instruction_vocab` 은 공식본(val_unseen_patched)에서 넣는다.
- lerobot `episode_index` 순서는 R2R `episode_id` 순서와 다르다. scene 안에서 GT 문장 완전일치로 매핑한다.

## 6. 재현성과 데이터 위생
- temperature 가 0 보다 크면 seed 를 `(seed, ep, stage, attempt)` 로 고정하고 기록한다.
  설정 hash 가 다르면 기존 출력을 덮어쓰지 않는다.
- 캐시는 split 별 디렉터리로 나눈다. episode_id 가 겹쳐서 train 이 val 캐시를 덮어쓴 적이 있다.
- 캐시를 gitignore 된 `outputs/` 에만 두었다가 잃었다. 캐시는 data-vol2 에 백업한다.
- GT 디렉터리는 읽기 전용으로 다룬다. 생성 스크립트가 GT 위치에 직접 쓴 적이 있다.
- 경로를 하드코딩하지 않는다. vln-s2-pipeline 에는 `/mnt/nvme0` 가 1,235곳에 있었다.
  노드마다 data 경로가 다르므로 env var 와 config 로만 받는다.

## 7. GT 누출 금지 (재현 실험이므로 특히 중요)
- vln-annotator 는 시작 동사를 GT 문장의 첫 단어에서 가져왔고, few-shot 예시도 같은 건물의 GT 문장이었다.
  이러면 "GT 재현" 점수가 부풀려진다.

→ 평가 split 의 GT 문장은 생성 코드에서 접근할 수 없게 한다.
→ few-shot 예시는 train scene 에서만 가져온다.

## 8. 모델 교체 시 주의
- 실제로 써 본 모델은 Gemma-4-31B (AWQ, bf16) 뿐이다.
- thinking 계열(Qwen3.5)은 `<think>` 출력이 섞이고 max_tokens 에서 잘린다. chat template 에서 thinking 을 끈다.
- 이미지를 여러 장 넣으려면 `--max-model-len` 과 `--limit-mm-per-prompt` 를 늘린다. 4096 으로는 16장이 안 들어간다.
- InternVL3 는 이미지 타일링 때문에 이미지 토큰이 크다.
- 캐시 키에 (모델, 프롬프트 hash, 렌더 설정 hash) 를 넣는다.
