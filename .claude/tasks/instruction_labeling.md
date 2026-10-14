# instruction_labeling — 계획

작성: 2026-10-13 (KST). 코드 위치: `scripts/dataset_converters/instruction_labeling/`.
교훈 체크리스트: `scripts/dataset_converters/instruction_labeling/LESSONS.md` (단계마다 확인).

## 목표
`vln_ce`(InternData-N1-v0.5-mini)의 GT 지시문을 **InternNav 입력과 같은 조건의 이미지 + GT waypoint**로
VLM(`data-vol2/checkpoints/VLM`)이 재현하게 한다. 재현 품질은 2x2 학습/평가 SR 로 판정한다.

## 용어
| 용어 | 뜻 |
|---|---|
| trajectory | R2R 경로 1개. 지시문 보통 3개 공유 |
| episode | 경로 1 + 지시문 1. LeRobot 저장 단위 |
| rig | 카메라 높이·pitch 조합. `125cm_0deg`(FPV), `125cm_30deg`(룩다운) 등 5종 |
| e2e | 이미지 여러 장 + 경로 단어를 VLM 한 번에 넣어 문장을 받는 방식 (baseline) |
| route words | `reference_path` 기하에서 규칙으로 만든 경로 설명 ("1. Turn left. 2. Walk forward a short way.") |
| dev subset | 빠른 반복용 고정 부분집합 (val_unseen 100 trajectory = 300 episode) |
| follower 평가 | GT 문장으로 학습한 고정 ckpt 에 label 문장으로 평가 → SR. 학습 불필요 |
| 2x2 (A) | {학습 문장 GT / label} × {평가 문장 GT / label} |

## 확정된 결정
- baseline = vln-s2-pipeline 의 **e2e(b0)** 를 InternNav 로 옮김 (`annotate.py` 의 `route_words`, `e2e_episode`, 품질검사,
  seed, 출력 규약 + `b0.yaml` 프롬프트). 이미지만 새 렌더러 산출물로 교체.
- 렌더는 **trajectory 단위 1회**, 출력은 train 과 동일하게 **episode 단위**(같은 jpg 를 episode 마다 복사).
  근거: train 은 같은 trajectory 의 episode 끼리 jpg 가 바이트 동일한 별도 파일 (5 scene 19,617 쌍 실측, 링크 아님).
- 서빙: `/home/irteam/venvs/vllm_gemma` (vllm 0.31.0) 로 vLLM OpenAI 호환 서버. 클라이언트는 habitat 파이썬(`/usr/bin/python`).
- 렌더러는 **train 재렌더 → InternData 이미지와 픽셀 비교** 를 통과해야 val 에 쓴다.
- VLM 입력 이미지 rig(FPV only vs FPV+LD30), 모델, 이미지 유무는 **ablation**.
- NavComposer(모듈형)는 baseline 확정 후 비교군으로 추가.

## 디렉터리 구조 (안)
```
scripts/dataset_converters/instruction_labeling/
  LESSONS.md
  README.md                 # 명령·인자 표
  configs/
    models.yaml             # 모델별 base_url, served_name, thinking off, mm limit, max_tokens, TP
    labels/b0.yaml          # baseline (b0 프롬프트 그대로) ; ablation 은 base: 상속
    serve/<model>.sh        # vllm serve 명령 (venv 경로·TP·max-model-len)
  render/
    render_vlnce.py         # reference_path -> 이산 action -> habitat 렌더 (5 rig) -> trajectory 캐시
    export_vlnce.py         # trajectory 캐시 -> episode 단위 LeRobot (train 포맷)
    verify_render.py        # train 재렌더 vs InternData: action / pose / pixel 비교 + 시각화
  label/
    route_words.py          # b0 이식
    e2e.py                  # b0 이식 (VLM 1회 호출, 재시도, 품질검사)
    annotate.py             # split 단위 실행, 캐시, 출력 json.gz + meta
  eval/
    text_metrics.py         # 다중참조 BLEU/METEOR/ROUGE/CIDEr/SPICE + 방향 정합성·어휘·고유율
    dev_subset.json         # 고정 dev subset
```
연결: 학습/평가 데이터 빌드는 기존 `relabel_vlnce/build_relabel_dataset.py` 를 재사용 (수정 필요하면 옵션 인자로만).

---

## P0. 준비 / 모델 서빙 확인
1. 디렉터리·`models.yaml`·`serve/*.sh` 작성.
2. 각 모델을 띄워 이미지 1장 + 16장 smoke 요청:
   gemma-4-31B-it, Qwen3-VL-32B-Instruct, Qwen3-VL-235B-A22B-Instruct-FP8, (선택) Qwen3.5-397B-A17B-FP8, InternVL3-78B.
   - 확인: vllm 0.31 로드 가능 여부, 필요한 TP/GPU 수, 16장 입력 시 토큰 수, thinking 끄기, `seed` 지원, 처리량.
   - 실패 모델은 표에 사유와 함께 기록하고 제외 (venv 추가는 사용자와 상의).
- 산출: `.claude/memory/<날짜>_instruction_labeling_p0_serve_result.md`

## P1. 렌더러 + 검증 (필수 관문)
렌더 조건 (학습·평가와 동일): 640x480, HFOV 79, **125cm rig 2종(`125cm_0deg`, `125cm_30deg`)만** (2026-10-13 결정), 이산 action 0.25 m / 15°.

1. **P1a action 일치**: train 의 `reference_path` → `vlnce_discretize.discretize` → action 시퀀스가 parquet `action` 과 같은지
   (dev: 3 scene 전체 episode). 다르면 원인 분석 후 진행 (InternData 가 다른 follower 를 썼을 가능성).
2. **P1b pose 일치**: 렌더 위치·방향 vs parquet `pose.<rig>` (위치 오차 mm, yaw 오차 °).
3. **P1c 픽셀 비교**: 같은 프레임을 렌더해 InternData jpg 와 비교.
   - 지표: md5 일치율, PSNR, SSIM, 최대 절대차. depth 는 mm 차.
   - 시각화: [InternData | 우리 | |diff|] 3단 이미지 + 히스토그램 → `logs/instruction_labeling/render_verify/`.
   - 통과 기준: md5 일치가 이상적. 아니면 사용자와 기준(PSNR/SSIM 임계) 합의 후 진행.
     (train 내부 jpg 가 바이트 동일 → InternData 렌더는 결정적이므로 같은 설정이면 일치 기대 가능.)
4. **P1d pixel goal / relative_goal_frame_id 재현**: `vlnce_pixel_goal.py` 로 계산해 parquet 과 비교.
5. **P1e val 렌더**: val_unseen(613 traj) 렌더 (val_seen 은 학습·평가에 안 쓰이므로 제외, 2026-10-13 결정) → episode 단위 export →
   `gs_vlnpe/05b_verify_vlnce.py` 로 스키마 검사. 출력은 GT 디렉터리가 아닌 별도 위치:
   `data/InternData-N1-v0.5-mini/vln_ce_render/traj_data/r2r/<split>/...` (위치는 P1 시작 전 사용자 확인).
6. top-down 경로 + 카메라 방향 그림을 episode 샘플마다 저장 (heading 부호 버그 재발 방지).
- 재사용: `gs_vlnpe/vlnce_discretize.py`, `vlnce_rig.py`, `vlnce_pixel_goal.py`, `vlnce_writer.py`, `05b_verify_vlnce.py`.
- 산출: `.claude/memory/<날짜>_instruction_labeling_p1_render_result.md`

## P2. baseline labeling (e2e/b0 이식)
1. `route_words`, `e2e_episode`, 품질검사(`_quality_check`), seed(`call_seed`), 출력 규약 이식.
   - 출력: GT 키만 + `instruction{instruction_text(". " 끝), instruction_tokens}`, top-level `instruction_vocab`(공식본), `_generation_meta`(config hash, model, git sha).
   - episode 별 meta: 입력 프레임 목록, route words, 프롬프트, 시도 횟수, 품질 이슈.
2. 프레임 선택: 렌더된 이산 프레임에서 start / 회전 지점 / goal 을 골라 최대 16장 (b0 의 key frame 규칙과 같게).
   기본 rig = `125cm_0deg`.
3. GT 누출 차단: 생성 코드는 평가 split 의 `instruction` 필드를 읽지 않는다 (입력 episode 에서 필드 제거 후 전달). few-shot 없음(b0).
4. 조용한 실패 차단: 파싱 실패·품질 실패·예외 카운트, 기준(예: 2%) 초과 시 중단.
5. 생성 후 자동 검사: 내용어 수, 고유 문장율, 첫 단어 분포, 단어 수 분포 (GT 와 나란히).
6. 실행 범위: dev subset → (통과 시) val_unseen 전체 → train. (val_seen 은 쓰지 않음)
- 산출: `.claude/memory/<날짜>_instruction_labeling_p2_baseline_result.md`

## P3. 문장 지표 (1차 필터)
- 다중참조(GT 3문장) BLEU-4 / METEOR / ROUGE-L / CIDEr / SPICE (pycocoevalcap; SPICE 는 Java 필요 → 확인).
- 방향 정합성(turn/stairs/goal 오류율), 명사 precision — vln-s2-pipeline `tools/compare_gt.py` 기준 이식.
- LLM-judge 는 생성 모델과 **다른 계열**.
- **GT 자체의 상한**: GT 3문장 중 1개를 나머지 2개로 평가한 leave-one-out 점수를 같이 표기 (지표의 천장).

## P4. ablation (dev subset, 문장 지표; 변수 하나씩)
| ID | 축 | 값 | 비고 |
|---|---|---|---|
| B | 모델 | gemma-4-31B-it / Qwen3-VL-235B-FP8 / Qwen3-VL-32B / (가능하면) Qwen3.5, InternVL3 | 기준: gemma bf16 |
| V | VLM 입력 rig | FPV(125cm_0deg) / FPV + LD30(125cm_30deg) | |
| C | 이미지 유무 | 이미지 + route words / route words 만 | 이미지 기여도 |
| N | 생성 구조 | e2e / NavComposer 모듈형 | baseline 확정 후 |
- 기준선은 seed 3회로 분산 측정 → 그보다 작은 차이는 결론 내지 않는다.

## P5. follower 평가 (학습 없음)
- GT 학습 ckpt (2x2 의 GT→GT 셀, SR 42.5 ckpt) 에 label 문장 평가 → SR. val_unseen 1,839, ld30.
- 상위 label 선별.

## P6. 2x2 (A)
- 상위 1~2 label × {학습 GT/label} × {평가 GT/label}, s1 기준 (기존 보고서와 같은 설정), seed 2회 이상.
- 비교 기준: GT 42.5 / v278fix 31.9 / v278fix_fewshot-3 33.6 (기존 보고서, 같은 설정).
- 빌드: `relabel_vlnce/build_relabel_dataset.py --labels <L> --emit data,yaml`.

---

## 확인 필요 (해당 단계 시작 전에 사용자에게 질문)
1. ~~rig 범위~~ → 125cm 2종 (확정). 출력 위치는 P1e 전에 확인.
2. P1c: 픽셀이 바이트 단위로 같지 않을 때 통과 기준 (P1c 결과를 보고 합의).
3. ~~val_seen 용도~~ → 안 쓰임 (확정). 렌더·labeling 제외.
4. P5/P6: 다른 노드에서 실행. P1 결과를 사용자가 확인한 뒤 진행 (P1 이 관문).

## 진행 상황
- [x] 분석 (vln-s2-pipeline, wiki/GitHub, InternNav 포맷), LESSONS.md, 중복 저장 확인
- [~] P0 서빙 — gemma-4-31B-it 은 기존 서버(:8100, DP8, 이미지 16장) 사용 (2026-10-13 결정). `--backend managed` 추가(2026-10-14): 실패 경로·정리만 검증, **GPU 가 비었을 때 성공 경로 검증 TODO**. TODO: Qwen 등 다른 모델은 GPU 확보 후
- [x] P1 렌더 + 검증 — 렌더러 릴리스와 기하 동일·RGB 차 < JPEG 잡음, val action = greedy_wp_r0.25 (VLN-CE GT 궤적과 평균 0.07 m), val_unseen 1,839 ep 렌더·검증 통과. train 은 릴리스 이미지 그대로 사용(재렌더 안 함, 2026-10-13 결정). 결과: `.claude/memory/261013_instruction_labeling_p1_render_result.md`
- [~] P2 baseline — b0 이식 완료, dev 303 ep 생성 (gemma). 결과 `.claude/memory/261013_instruction_labeling_p2_baseline_result.md`. 남은 것: val_unseen 전체 + train 생성
- [~] P3 지표 — BLEU/METEOR/ROUGE/CIDEr + 규칙지표(compare_gt 재사용) + 다양성. TODO: SPICE (Java 충돌)
- [ ] P4 ablation
- [ ] P5 follower 평가
- [~] P6 2x2 — 2026-10-14 node2/3 queue 투입 (relabel_half s1: train.L_eval.gt / train.L_eval.L / train.gt_eval.L), 4 label
