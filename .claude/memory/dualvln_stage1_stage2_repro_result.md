# DualVLN stage1 → stage2 → eval 전체 reproduce 준비 결과 (2026-08-18)

## 이전과 달라진 점

지금까지는 stage2만 reproduce했고 stage1은 공개 `InternVLA-N1-System2` ckpt를 init으로 썼다.
이번 작업으로 stage1(System-2 pretraining, `main:scripts/train/qwenvl_train/train_system2.sh`)부터
runner.py 체계로 학습할 수 있게 되었다.

### 수정 파일 (1개)
- `scripts/train_eval/qwenvl_train/default_config.py`
  - `Params.vision_tower_lr: Optional[float] = None` 추가 — None이면 `--vision_tower_lr` 미emit (기존 동작 그대로)
  - `lr_scheduler_min_lr`을 `Optional[float]`로 — None이면 `--lr_scheduler_kwargs` 미emit
    (plain `cosine`은 min_lr kwarg를 거부하므로 stage1에 필요)
  - **stage2 argv byte-identical 검증 완료**: 수정 전/후 `--print-train-argv` diff 없음, `verify_params.py` 전부 green

### 신규 파일 (3개)
- `scripts/train_eval/qwenvl_train/baseline/dualvln_stage1_full.py` — train_system2.sh 미러.
  init `Qwen/Qwen2.5-VL-7B-Instruct`, lr 2e-5 + vision_tower_lr 5e-6, plain cosine, epochs 2.0,
  tune_mm_* 전부 True(전체 VLM finetune), `pixel_goal_only=False`, `system1="none"`,
  8개 dataset 100% 샘플링. **EXP_NAME에 `internvla-n1-system2` substring 포함** —
  `internvla_n1_trainer.py:156`이 이 substring으로 로더를 분기하므로 stage2가 로컬 stage1
  output dir을 trainer 무수정으로 로드하는 핵심 장치.
- `scripts/train_eval/qwenvl_train/baseline/dualvln_stage2_from_stage1.py` —
  `dualvln_stage2_full.PARAMS`를 그대로 import(레시피 드리프트 없음), import 시점에
  `STAGE1_CKPT` env var → 없으면 최신 `dualvln_stage1_internvla-n1-system2_*` glob으로 stage1
  ckpt를 resolve. chat_template.json이 없으면 공개 ckpt에서 backfill.
- `scripts/train_eval/qwenvl_train/baseline/verify_params_stage1.py` — verify_params.py 헬퍼
  재사용. **결과: 36개 flag identical** (model_name_or_path, vision_tower_lr, lr_scheduler_type,
  vln_dataset_use 포함), 11개는 의도된 diff(effective batch 128 동일, inert flag들),
  8개 dataset 전부 디스크 존재 확인(r2r 122 scenes, rxr 118 scenes, 45deg pose 포함).

## 실행 명령 (전 과정) — 2026-08-19 갱신: auto-eval fallback 수정으로 단축

```bash
# 1) stage1 학습 + 자동 평가(S2 + ShortestPathFollower, R2R val_unseen)
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline/dualvln_stage1_full.py --machine h200

# 2) stage2 학습 + 자동 평가 — config가 최신 stage1 output dir 자동 선택 (STAGE1_CKPT=<dir>로 고정 가능)
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline/dualvln_stage2_from_stage1.py --machine h200
```
학습만 하려면 `--no-eval`, 기존 ckpt 재평가는 `--no-train --model-path <dir>`.

### self-contained checkpoint + eval 선택 로직 최종형 (2026-08-19, 이 섹션이 최신)

**저장 시점** (신규 `internnav/trainer/checkpoint_aux_callback.py` + trainer hookup 3줄):
`SaveAuxFilesCallback`이 checkpoint-<step> 저장 때마다 preprocessor_config.json(학습에 실제
사용된 뮤테이트값 — dataset이 max_pixels=313600으로 설정) + chat_template.json(base ckpt에서
resolve, 로컬/HF id 모두 동작)을 함께 기록 → **모든 checkpoint가 self-contained**
(AutoProcessor.from_pretrained 직접 가능, runner 없이 eval.py로도 평가 가능). 용량 +1.6KB/ckpt.
final save에도 chat_template.json 기록 추가 (기존엔 preprocessor만).

**평가 시점** (`runner._pick_eval_ckpt`, released-ckpt fallback **완전 제거**):
```
1. --model-path 명시 → 2. best_metric ckpt(validation run) → 3. output_dir 최상위 final model
→ 4. 최신 checkpoint-<step> (final save 실패 시) → 5. 없으면 ERROR 후 skip
```
best/latest는 자동: validation run이면 best, val_ratio=0이면 (best 부재로) latest.
guideline.md의 eval-only 예시에 `--model-path` 명시 추가됨.

검증: callback 유닛 테스트(HF-id chat_template resolve, 뮤테이트 max_pixels 보존, rank guard,
AutoProcessor 로드+apply_chat_template), _pick_eval_ckpt 6케이스 + 실데이터
(죽은 stage2 run → checkpoint-5000 자동 선택), verify_params 2종 green,
train argv md5 수정 전후 동일.

legacy 처리: `dualvln_stage2_full_20260818_101948/checkpoint-5000`에 aux 2개 수동 백필 완료
(preprocessor는 학습값 max_pixels=313600으로 패치) → 지금 바로
`--no-train --model-path .../checkpoint-5000` 평가 가능.

### (구버전 기록) auto-eval fallback 수정 (runner.py, 2026-08-19)
기존: `best = model_path or find_best_checkpoint(output_dir) or EVAL_MACHINE["model_path"]`
— val_ratio=0이면 best_metric이 없어 **w-NavDP로 silent fallback**했음 (문서화만 하고 미수정 상태였음).
수정 (2단계, `runner._pick_eval_ckpt()`로 분리):
1. best가 없고 `output_dir/config.json`이 존재하면 `output_dir`(최종 모델) 평가 — main의 "last checkpoint" 의미.
2. released-ckpt fallback은 **스모크(--max-steps)/eval-only 전용**으로 제한 + WARNING 출력.
   진짜 학습 후 아무것도 저장 안 됐으면 ERROR 출력 후 eval skip — silent 오평가 원천 차단.
   (fallback 완전 제거는 불가: guideline.md의 `--no-train --debug-dir` 디버그 워크플로우가
   released ckpt 자동 사용을 의도적으로 활용함.)
유닛 테스트 6개 케이스 통과 (final-model / real-train-빈dir→skip / best_metric 우선 /
smoke-fallback+경고 / --model-path 최우선 / 아무것도 없음→skip).

각 argument: `--config` 실험 정의(.py, PARAMS/EXP_NAME/TRAIN_MACHINE), `--machine h200` Habitat
eval + h200 경로 프리셋, `--no-eval` 학습만, `--no-train --model-path` 평가만.
무인 연속 실행: 위 3줄을 `queue.txt`에 넣고 `./run_queue.sh`.

## 검증 결과 요약

- stage2 argv byte-identity: PASS (수정 전후 diff 없음)
- `verify_params.py` (stage2 vs main train_dual_system.sh): ALL PARAMETERS MATCH
- `verify_params_stage1.py` (stage1 vs main train_system2.sh): ALL PARAMETERS MATCH
  (36 identical / 11 intended / 0 unexplained, effective batch 128==128, 8 datasets OK)
- 브랜치 코드가 main과 다른 부분 중 stage1 경로 영향 검증 완료:
  `stop_list*5`→`stop_weight`(기본 5, 동일), qwen2.5 로더 분기 main과 동일,
  `system1="none"`→S1 모듈 없음, vision_tower_lr은 `qwenvl_base.py` `create_optimizer`
  몽키패치에서 `"visual" in name` 그룹 5e-6으로 소비.

## stage1 smoke run 결과 (2026-08-19, PASS)

- 1차 시도 (`batch_size=16` 상속): step 1은 완주했으나 step 2에서 **CUDA OOM** (per-GPU ~117GB
  사용 + 23GB 할당 실패). stage2는 VLM frozen이라 batch 16이 들어가지만 stage1은 7B 전체
  finetune이라 불가. main도 per-device batch 2였음.
- 수정: `dualvln_stage1_full.py`에 `batch_size=4, grad_accum_steps=4` — effective batch는
  main과 동일한 128 유지 (8 GPU × 4 × 4 = 64 GPU × 2 × 1). `verify_params_stage1.py`의
  EXPECTED_DIFFS 갱신 → 35 flags identical, ALL PARAMETERS MATCH.
- 2차 시도: **10/10 step 완주, exit 0, 에러 0건**. loss 4.07 → 1.10로 정상 하강,
  lr warmup→cosine decay 확인(peak 2e-05). trainable 로그: vision 32개 블록 전부 True,
  merger True, LLM 28개 레이어 + embed 전부 True — `tune_mm_* True` 의도대로.
  vision_tower_lr(5e-6) param group 분기도 크래시 없이 통과.
- 속도: ~19.5s/optimizer step (eff 128). GPU 메모리 81–124GB로 샘플별 편차 큼
  (GPU당 헤드룸 최소 ~15GB) — 본 학습에서 간헐 OOM이 나면 `batch_size=2, grad_accum_steps=8`
  (main per-device 완전 동일)로 낮추고, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`도 옵션.
- 로그: `logs/260819_stage1_smoke_runner2.log` (1차 OOM: `..._runner.log`)

## stage1 평가: S2 + ShortestPathFollower (2026-08-19 추가, runner 통합 완료)

stage1(System-2 only) ckpt는 `_run_eval_system2`(`habitat_vln_evaluator.py:879`)로 평가한다:
S2가 discrete action 또는 pixel goal을 텍스트 출력 → pixel goal이면 GT depth unprojection →
navmesh snap → ShortestPathFollower(0.25m)가 추종. README 공식 프로토콜
("InternVLA-N1 (S2) + ShortestPathFollower", R2R val_unseen: **NE 4.25 / OS 68.3 / SR 60.9 / SPL 55.2**)
과 동일하며 이것이 stage1의 reproduce 타겟.

runner 통합을 위해 추가 수정 3건:
- `default_config.py` `build_habitat_eval_cfg`: `"mode": "system2" if p.system1 == "none" else
  "dual_system"` — stage1 config(system1="none")만 system2 모드, 기존 config 전부 무변경 확인.
- `dualvln_stage1_full.py`: `eval_config_path="scripts/eval/configs/vln_r2r_full_ld30.yaml"` 추가.
  ld30(2×15=30°)이어야 `_run_eval_system2`의 unprojection 하드코딩(pitch 30°, l.966)과 일치.
  ⚠️ 기존 `habitat_s2_cfg.py`가 가리키는 `vln_r2r.yaml`은 tilt 30(→60°)이라 unprojection과 불일치.
- `runner.py` `_backfill_chat_template`: system2_ckpt가 HF id(stage1은 "Qwen/...")면
  `hf_hub_download`로 chat_template.json을 가져오는 fallback 추가. **필수 수정** —
  AutoProcessor.apply_chat_template은 chat_template.json이 없으면 tokenizer template로
  fallback하지 않고 즉시 실패함을 실측 확인. 수정 후 end-to-end 테스트 통과(offline hub cache).

평가 명령 (stage1 학습 완료 후):
```bash
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline/dualvln_stage1_full.py --machine h200 --no-train --model-path <checkpoints_root>/baseline/dualvln_stage1_internvla-n1-system2_<TIMESTAMP>
```
sanity-gate (학습 전, released ckpt로 SR≈60.9 재현 확인):
```bash
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline/dualvln_stage1_full.py --machine h200 --no-train --model-path /home/irteam/data-vol2/checkpoints/InternVLA-N1-System2
```

## habitat eval SIGABRT 디버깅 (2026-08-19) — 드라이버 회귀 + 워크어라운드

### 증상
- system2/dual **양쪽** eval이 `env.step()` 렌더링 중 SIGABRT (python traceback 없음).
- eval 중 S2 generate가 매번 ~20초씩 스톨.
- 같은-GPU 실행에서 crash 전에도 **관측 이미지가 조용히 손상** (ep1 NE 14.14 vs 정상 3.90).

### 원인 규명 과정
1. faulthandler → `habitat_sim get_observation`(GL readback)에서 abort 확정.
2. 순수 sim으로 같은 action 재생 → 정상. 모델+sim harness(가벼운 generate) → 정상.
3. gdb backtrace → **`libnvidia-eglcore.so.580.126.16` 내부에서 abort()** 호출
   (`Magnum readImplementationRobustness`/glReadPixels 경로). `MAGNUM_DISABLE_EXTENSIONS=GL_ARB_robustness`로
   robustness를 꺼도 재발 → 드라이버 자체 버그.
4. 드라이버 라이브러리 설치일 **7/31** — 7월 초 eval은 정상이었으므로 회귀 지점 확정.
5. **렌더 GPU 분리 테스트 (모델 cuda:0, habitat gpu_device_id=1) → 2/2 에피소드 완주**
   (ep2 SR 1.0 / SPL 0.92) + 스톨/손상도 사라짐.

### 조치 (구현)
- `Params.eval_render_gpu_offset: int = 0` (default 무변경) → env_settings `render_gpu_offset`.
- `HabitatVLNEvaluator.__init__`: offset>0이면 `gpu_device_id = (LOCAL_RANK + offset) % device_count`.
- stage1/stage2 config에 `eval_render_gpu_offset=1` 설정.
- 근본 해결은 드라이버 업/다운그레이드 (ops) — 그 전까지 offset=1 유지.
- **8-rank 검증 PASS**: rank i 모델 / (i+1)%8 렌더 크로스 배치로 8/8 rank가 2/2 에피소드
  완주, abort 0건. 부수 효과: eval 속도 **~16배** (에피소드당 ~590s → 36–150s — 같은-GPU
  경합의 20초 스톨이 사라짐). same-GPU 시절의 낮은 성능(ep1 NE 14.14)도 프레임 손상 때문이었음
  → 이전에 같은-GPU로 측정된 habitat eval 수치가 있다면 재측정 필요.

## 남은 배관 테스트 (선택)

```bash
# stage2 handoff 배관 테스트 (공개 ckpt로 대행 — dir 이름이 substring 매칭됨)
STAGE1_CKPT=/home/irteam/data-vol2/checkpoints/InternVLA-N1-System2 python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline/dualvln_stage2_from_stage1.py --machine h200 --no-eval --max-steps 10
```

## 주의 (기존 run 관련)

`dualvln_stage2_full_20260818_101948`은 **2026-08-19 00:52, step 4993/18321에서 사망**
(로그가 progress bar 중간에서 잘림, traceback/OOM 기록 없음 → 외부 kill 추정).
save_steps=5000 직전이라 **checkpoint 0개, 14시간 학습 유실, resume 불가** — stage2 full은
처음부터 재실행 필요. (원래 지적했던 `--no-eval` 누락 문제는 사망으로 무의미해짐.)

## 기타 참고

- stage1 wall-clock: main은 64 GPU × 2 epochs → 8×H200 단일 노드로는 약 8배 (1주+ 예상).
  effective batch 128은 동일해 수렴 레시피는 유지.
- stage1 output dir 최상위에 최종 모델 저장(`safe_save_model_for_hf_trainer`) — 중간
  `checkpoint-*/`는 stage2 init으로 직접 사용 불가(top-level config.json 체크로 방어).

## stage1 학습속도 벤치마크 (2026-08-20)

10-step 스모크 실측 (eff batch 128 고정, 8×H200):

| 설정 | s/step | peak VRAM | 개선 |
|---|---|---|---|
| baseline: ckpt ON, b4×ga4 | 19.4 | ~124GB | — |
| A: expandable_segments + b8×ga2 | 19.1 | 89.9GB | ~2% |
| B: grad ckpt OFF + b2×ga8 | 18.1 | 113GB | ~7% |

- **GPU util 92%** (step 구간 실측) → 연산 포화. 데이터로더 병목 아님.
- 결론: 단일 노드 레버는 사실상 없음. per-step 연산(128 samples × 7B full finetune)이 지배.
  의미 있는 단축은 **멀티노드**뿐 (main도 64 GPU였음).
- 채택: 레시피 원형 유지(ckpt ON, b4×ga4) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`만
  추가(성능 무관, 파편화 감소로 OOM 안전 마진 확보). ETA ~10.8일 (48,974 steps × 19.4s).

### run_queue 운영 메모
- stage1을 죽이면 run_queue가 다음 줄로 넘어감. `debug/repeat_train.sh`는 큐 마지막의
  GPU-필러(무한 루프, 로그 없음)로 preempt 가능.
- 2026-08-20 03:13 큐로 재시작: stage1(_20260820_031307) → stage2_from_stage1 → filler 순.
  이전 run(_20260820_010835)은 step 126에서 벤치마크를 위해 중단 (유실 ~40분, ckpt 없음).

## dataloader 병목 수정 + resume (2026-08-28)

### 문제와 수정
- stage1이 ~48s/step(ETA 26일)으로 느렸던 원인: `NavPixelGoalDataset.__getitem__`이
  에피소드 0번째~현재 프레임을 **전부** 로드(프레임당 파일 3개 open + depth 리사이즈)하고
  history 8장/현재/미래 traj 외에는 버림 — 에피소드 후반 샘플은 수백 파일 dead load.
- 수정: history도 traj도 아닌 프레임을 skip하는 guard 2줄 (`internvla_n1_lerobot_dataset.py`).
- 검증: 12개 샘플(뒤쪽 start_frame_id=300 극단 포함) **출력 tensor 완전 동일** + 샘플당 3.4×
  고속화 (1.41s → 0.42s). 검증 방식: git index 원본을 별도 모듈로 로드해 동일 seed 비교.

### 중복 run 사건과 정리
- run_queue가 2개 살아 있어(8/20 03:13, 05:27) 같은 stage1을 2벌 학습하는 중복 발견.
- 정리(데이터 보존, 이름만 변경 — stage2 glob에서 제외): `duplicate-run_...052742`,
  `dead-run_...010835/154630/160947` (checkpoint 없는 것만).
- **본선 run: `dualvln_stage1_internvla-n1-system2_20260820_031307`** (glob이 이것만 잡음).

### resume
- checkpoint-15000(optimizer 포함 109GB)에서 재개 — 유실 43 step.
- runner는 새 dir을 만들어 자동 resume이 안 되므로, 원본 torchrun argv를 /proc에서 캡처해
  `scripts/train/qwenvl_train/dual_full/resume_dualvln_stage1_20260820_031307.sh`로 저장.
  같은 --output_dir → trainer가 최신 checkpoint 자동 resume.
- resume 직후 HF Trainer가 15,000 step 분량 데이터 순서 fast-forward (수정 덕에 ~3-4h 예상).
- wandb는 resume 구간이 새 run으로 기록됨 (기존 run id 미보존).
- 완료 후 평가: `runner --no-train --model-path <_031307 dir>` 수동 실행 필요 (runner 미경유 학습이므로).

### resume 결과 (2026-08-28 확정)
- fast-forward는 sampler 인덱스 스킵으로 **29초 만에 완료** (우려했던 수 시간 아님).
- loss 0.40에서 자연스럽게 연속, grad_norm ~1.5 (optimizer/scheduler 복원된 warm resume 확인).
- **20.0 s/step, GPU 평균 665W** (수정 전 48s/step, 190–340W) — dataloader 병목 해소,
  연산 한계 도달. 남은 34k steps ≈ **7.9일** (기존 페이스 ~19일).
