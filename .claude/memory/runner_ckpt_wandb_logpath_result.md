# runner.py: 체크포인트 해석 / wandb run 이름 / 로그 경로 정리 — 결과

작업일: 2026-08-18 · 브랜치 `feature/input_v0.1`

## 무엇이 달라졌나

### 1. 학습 후 평가가 방금 학습한 체크포인트를 쓴다

**이전**: `val_ratio=0` → `--eval_strategy no` → `trainer_state.json`에 `best_metric` 없음 →
`find_best_checkpoint()`가 `None` → `train_and_eval`이 **조용히**
`EVAL_MACHINE["h200"]["model_path"]`(= `checkpoints/InternVLA-N1-w-NavDP`)로 폴백.
로그는 정상 평가처럼 보이지만 실제로는 릴리즈 체크포인트를 평가.

**이후**: 해석 순서가 `--model-path` → `best_metric` → **`find_last_checkpoint()`** 이고,
전부 실패하면 `SystemExit`. EVAL_MACHINE 폴백은 삭제.

- `find_last_checkpoint(output_dir)` (runner.py): `output_dir` 최상위에 `config.json` +
  `model*.safetensors`가 있으면 그 자체를 반환(`internvla_n1_trainer.py`의
  `safe_save_model_for_hf_trainer()`가 학습 완료 시 여기에 최종 모델을 쓴다) → 없으면
  `checkpoint-N` 중 N 최대값.
- `EVAL_MACHINE[*]["model_path"]`는 h200/5090/h1 모두 `None` (키는 유지 — dict 스키마 불변).
- `scripts/eval/eval.py`에 guard: `--model_path`도 없고 config의 `model_path`도 `None`이면
  즉시 `SystemExit`. runner를 거치지 않는 직접 실행도 기본 체크포인트로 돌지 않는다.
- 선택 근거를 항상 출력: `[eval] machine=h200 checkpoint=... (explicit --model-path | lowest eval_loss | last checkpoint (no eval_loss))`

### 2. resume된 wandb run 이름이 유지된다 + 프로젝트 통일

**이전**: `run_eval`이 항상 `--wandb_run_name`을 넘겼고 `distributed_base.py`의
`wandb.init(name=...)`은 **resume된 run을 개명**한다. 그래서 `baseline/dualvln_stage2_full_<ts>`로
학습된 run이 평가되는 순간 체크포인트 이름으로 바뀌었다. 또 학습 run은 `huggingface`(기본 entity),
runner가 만드는 run은 `kaist-url-ai28/InternNav`로 가서 프로젝트가 갈라졌다.

**이후**:
- `wandb_run_id.txt`가 평가 시작 시점에 **이미 있으면** `--wandb_run_name`을 넘기지 않는다
  → `wandb.init(name=None)` → 원래 이름 유지. `eval.py`도 `WANDB_RUN_ID` env가 있으면
  `wandb_run_name` setdefault를 건너뛴다.
- id 파일이 **없을 때의 run 생성 경로는 그대로** (`original/<ckpt>_<ts>`로 생성 → 이후
  `--wandb_run_name`(= 체크포인트 경로 이름)으로 명명). 요구사항과 일치.
- `run_train`/`run_eval` 서브프로세스 env에 `WANDB_PROJECT`/`WANDB_ENTITY` export
  (`_wandb_env()`). HF Trainer의 `WandbCallback`이 이 env를 읽으므로 학습 run도 InternNav로.
- `EVAL_MACHINE`의 `wandb_project` 하드코딩(`"huggingface"`, `"internnav"`) → 모듈 상수
  `WANDB_PROJECT`. Habitat `eval_settings`에 `wandb_entity` 추가, `distributed_base.py`의
  `wandb.init()`에 `entity=` 전달.

**결과**: 학습(`train/loss`)과 평가(`test/sucs_all|spls_all|oss_all|nes_all|length`)가
**이름이 바뀌지 않은 같은 run** 하나(`kaist-url-ai28/InternNav`)에 쌓인다.

### 3. 로그가 `logs/{config}/` 아래로 분리된다

**이전**: `<ckpt>/logs/` 평면 구조 → 같은 체크포인트를 여러 config로 평가하면
`progress.json`·`result_h200.json`이 한 파일에 섞였다. `test_h200.log`는 매 실행 덮어써졌다.

**이후**:
```
<ckpt>/logs/
  train.log                                # 학습 (그대로)
  baseline_dualvln_stage2_full/            # exp_name의 '/'를 '_'로
    progress.json                          # resume 소스 (누적)
    result_h200.json                       # JSON-lines, 실행마다 append
    test_h200_20260818_084225.log          # 실행마다 새 파일
    eval_20260818_084234.log
    check_sim_0/ , vis_0/
  bev_s1.bev_s2.fpv_rgb_gt/
    ...
```
- h1은 collision 모드 분리를 유지: `logs/<exp_slug>_<flash_collision>/`
- **디렉터리 이름에 타임스탬프는 넣지 않았다**: Habitat은 `<log_dir>/progress.json`으로
  (`resume_from_output_path`), h1은 `<log_dir>/lmdb`로 이어서 평가하고 watchdog 재시작도
  여기에 의존한다. 매 실행 새 디렉터리면 항상 처음부터 다시 돈다. 대신 덮어써지면 곤란한
  `test_<machine>.log`에만 타임스탬프를 붙였다.
- `--debug-dir`가 `output_dir`을 가로채는 동작은 **설계 의도대로 유지**.
- 체크포인트 밖으로 새던 경로를 `EVAL_OUTPUT_DIR` 우선으로 변경 (`result_logger.finalize_all_results`의
  기존 패턴): `progress_log_multi_util.py`(h1 진행 로그), `visualize_util.py`(h1 vis_output),
  `result_logger.py:228`(eval_result.log), `default_config.py`의 `vis_debug_path`.

### 4. 재현용 기록 (`repro_<ts>.sh`)

학습·평가를 시작할 때마다 로그 디렉터리에 실행 가능한 재현 스크립트를 남긴다.

```
<ckpt>/logs/repro_20260818_085538.sh                            # 학습
<ckpt>/logs/<config>/repro_20260818_085522.sh                   # 평가
<ckpt>/logs/<config>/repro_20260818_085522.diff                 # 더티였을 때만
```
`<ts>`는 같은 실행의 `test_<machine>_<ts>.log`와 **동일**하므로 결과와 레시피가 짝지어진다.

내용: 실행 시각·호스트·브랜치·(machine/ckpt/config 또는 output/run) + `git checkout <commit>` +
**runner.py를 실제로 실행한 커맨드 라인 그대로**. 워킹트리가 더러웠으면 경고와 함께
`git diff HEAD`를 `repro_<ts>.diff`로 저장한다(커밋만으로는 더티 트리를 설명할 수 없고,
학습은 3일씩 걸린다). 기록 실패가 실행을 죽이지 않도록 전체를 try/except로 감쌌다.

왜 필요한가: 끝난 프로세스의 argv는 `ps`로 되읽을 수 없고, wandb run 이름으로도 알 수 없다
(`--debug-dir`가 있으면 `--model-path` 유무와 무관하게 `run_name`이 `exp_name_<ts>`로 유지된다).
런치 시점에 적어두는 수밖에 없다.

### 5. 시각화는 `--vis`를 줄 때만

`runner.py --vis` (store_true) 신설. 주면 `EVAL_VIS=1` env relay로
`vis_debug=True`(Habitat) / `vis_output=True`(h1)가 되고 출력이 `<log_dir>/vis_debug|vis_output`으로
간다. **안 주면 이전과 100% 동일** (둘 다 False, 파일 하나도 안 생김).

---

## 커맨드

### 학습 + 평가 (한 번에)
`python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline/dualvln_stage2_full.py --machine h200`

이제 **정상 동작한다**. 학습이 끝나면 `output_dir` 최상위의 최종 모델을 `find_last_checkpoint()`가
찾아 평가한다. (이전에는 여기서 릴리즈 체크포인트로 폴백됐다.)

### 학습만
`python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline/dualvln_stage2_full.py --machine h200 --no-eval`

### 평가만
`python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline/dualvln_stage2_full.py --machine h200 --no-train --model-path /home/irteam/data-vol2/checkpoints/InternVLA-N1-DualVLN`

| argument | 의미 |
|---|---|
| `--config` | 실험 config `.py` (PARAMS + eval_cfg). 학습·평가가 같은 파일을 쓴다 |
| `--machine h200` | 학습/평가 인프라 프리셋 (h200·5090 = Habitat, h1 = Isaac Sim) |
| `--no-train` / `--no-eval` | 한쪽만 실행 |
| `--model-path` | 평가할 체크포인트. `--no-train`일 때 `output_dir`도 이 경로가 된다 |
| `--vis` | (신규) per-episode 시각화 저장. 안 주면 아무것도 안 생김 |
| `--nproc N` | torchrun GPU 수 |
| `--max-steps N` | 스모크: 학습 N 스텝 + 평가 N 에피소드, wandb 비활성 |

---

## 검증 결과

### 1. 회귀 없음 — 학습 argv 완전 동일
`--print-train-argv` 출력을 변경 전/후로 diff → **차이 없음**. 학습 경로는 손대지 않았다.

### 2. 폴백 제거 (기존 동작이 실제로 막히는지)
```
train_and_eval(..., do_train=False, model_path=None)
  -> SystemExit: [eval] no checkpoint found under <output_dir>. Pass --model-path <ckpt> ...
EVAL_MACHINE model_paths: {'h200': None, '5090': None, 'h1': None}
```
runner를 안 거치는 직접 실행도 막힌다:
```
$ TRAIN_EVAL_TARGET=h200 /usr/bin/python scripts/eval/eval.py --config .../dualvln_stage2_full.py
[eval] no checkpoint to evaluate: pass --model_path <ckpt>. ...
```

### 3. `find_last_checkpoint` 선택 규칙
가짜 output_dir(`checkpoint-2000/5000/10000` + 가중치 없는 `checkpoint-15000`)로 확인:
```
checkpoint-* only          -> checkpoint-10000   # 가중치 없는 15000은 건너뜀
final model at top level   -> output_dir itself  # 학습 완료 후 최종 모델 우선
```

### 4. 로그 경로 + resume (실제 3 에피소드 평가 2회)
`InternVLA-N1-DualVLN`을 `--nproc 1 --max-steps 3`으로 두 번 실행:
```
<ckpt>/logs/baseline_dualvln_stage2_full/
  progress.json                   # 1회차 ep 10,11,12 -> 2회차는 이를 건너뛰고 ep 16,17,18
  result_h200.json                # {"...", "length": 3} / {"...", "length": 6}  (2줄)
  test_h200_20260818_084225.log   # 실행마다 별도 파일 (덮어쓰기 없음)
  test_h200_20260818_084526.log
  eval_20260818_084234.log , eval_20260818_084535.log
  check_sim_0/
```
- config별 분리: `exp_name`을 바꾸면 `EVAL_OUTPUT_DIR`이 다른 디렉터리로 간다
  (`bev/s1.bev_s2.fpv_rgb_gt` → `logs/bev_s1.bev_s2.fpv_rgb_gt/`).
- h1: `logs/baseline_dualvln_stage2_full_stop/` (collision 모드 분리 유지).
- `exp_name` 없이 `run_eval()`을 직접 부르면 기존 레이아웃(`logs/`, `logs_none/`)으로 폴백 — 하위 호환.
- **레포 루트 `logs/` 와 CWD에 새 파일 0개** (`find` diff로 확인).
- `--vis` 미지정 → 시각화 파일 0개.

### 5. wandb (argv/env 단위 검증)
```
id 파일 있음(resuming) : --wandb_run_name 미전달 ✅  WANDB_RUN_ID=abc123xy  WANDB_RESUME=allow
                          WANDB_PROJECT=InternNav  WANDB_ENTITY=kaist-url-ai28
id 파일 없음           : --wandb_run_name my_run_name  (기존 동작 유지)
eval_settings (h200)   : use_wandb=True  wandb_entity=kaist-url-ai28  wandb_project=InternNav
eval_settings (h1)     : wandb_entity=kaist-url-ai28  wandb_project=InternNav
```

### 6. repro 기록
학습·평가 양쪽에서 생성 확인. 더티 트리(15개 변경)를 감지해 경고 + `repro_<ts>.diff`(27 KB) 저장:
```
#!/usr/bin/env bash
# Reproduce this eval run.
#   when   : 2026-08-18 08:55:22
#   host   : internnav-train1-1-0
#   branch : feature/input_v0.1
#   machine: h200
#   ckpt   : /home/irteam/data-vol2/checkpoints/InternVLA-N1-DualVLN
#   config : scripts/train_eval/qwenvl_train/baseline/dualvln_stage2_full.py
#
#   WARNING: 15 uncommitted change(s) at launch time.
#   The commit below does NOT describe them -- apply repro_20260818_085522.diff too:
#     git apply repro_20260818_085522.diff

set -euo pipefail
cd /home/irteam/git/InternNav
git checkout 7463c7ee00db7e6b61f5bd609891b5eba6035511

python3 scripts/train_eval/qwenvl_train/runner.py --config .../dualvln_stage2_full.py --machine h200 --no-train --model-path /home/irteam/data-vol2/checkpoints/InternVLA-N1-DualVLN
```

### 7. `--vis` 매트릭스 (기본값 불변 확인)
| EVAL_VIS | EVAL_OUTPUT_DIR | vis_debug | vis_debug_path |
|---|---|---|---|
| (없음) | (없음) | False | `./logs/habitat/vis_debug` ← 변경 전과 동일 |
| (없음) | `/x/logs/exp` | False | `/x/logs/exp/vis_debug` |
| `1` | `/x/logs/exp` | True | `/x/logs/exp/vis_debug` |

h1도 동일: `EVAL_VIS=1`일 때만 `vis_output=True`.

---

## 변경 파일

| 파일 | 내용 |
|---|---|
| `scripts/train_eval/qwenvl_train/runner.py` | `find_last_checkpoint`/`_has_model_weights`/`_wandb_env`/`write_repro` 추가, 폴백 삭제, `resuming` 가드, `logs/<exp_slug>/`, 타임스탬프 test log, `--vis` |
| `scripts/train_eval/qwenvl_train/default_config.py` | `EVAL_MACHINE[*].model_path=None`, `wandb_project→WANDB_PROJECT`, `wandb_entity` 추가, `_vis_enabled()`/`_vis_dir()` |
| `scripts/eval/eval.py` | model_path guard, `WANDB_RUN_ID` 있을 때 `wandb_run_name` setdefault 스킵 |
| `internnav/evaluator/distributed_base.py` | `wandb.init(entity=...)` 1줄 |
| `internnav/utils/progress_log_multi_util.py` | `EVAL_OUTPUT_DIR` 우선 (h1 진행 로그) |
| `internnav/evaluator/utils/visualize_util.py` | `EVAL_OUTPUT_DIR` 우선 (h1 vis_output) |
| `internnav/evaluator/utils/result_logger.py` | `EVAL_OUTPUT_DIR` 우선 (`eval_result.log`) |

`internnav/habitat_extensions/vln/habitat_vln_evaluator*.py` 3개는 **수정하지 않았다.**

## 이번에 안 한 것

per-episode raw 결과 LMDB 저장은 별도 작업으로 분리. (Habitat 경로는 여전히 rank 0만
`progress.json`을 쓰므로 8-GPU 평가에서 에피소드의 7/8이 디스크에 안 남는다.)
참고 자료: `/home/irteam/git/VLN-Challenge`의
`third_party/vlnverse_emr/vlnverse/evaluator/utils/data_collector.py:127-149`(writer),
`training/extract_episode_table.py:45-104`(reader).

## 실행 중이던 평가에 대한 메모 (버그 아님)

PID 1218758 (2026-08-18 06:40 시작, 8-GPU)의 runner 커맨드는
```
python3 scripts/train_eval/qwenvl_train/runner.py --config .../baseline/dualvln_stage2_full.py \
  --machine h200 --no-train --model-path /home/irteam/git/InternNav/checkpoints/InternVLA-N1-w-NavDP \
  --debug-dir logs/baseline/InternVLA-N1-w-NavDP_ld30
```
`--model-path`가 **명시되어 있으므로 폴백과 무관**하고, 이번 변경 이후에도 그대로 동작한다.

교훈(진단 절차): 자식 프로세스의 `--wandb_run_name`이 `baseline/dualvln_stage2_full_<ts>`(체크포인트
이름이 아니라 exp_name)여서 "`--model-path`가 없었다"고 잘못 추론했다. `--debug-dir` 분기가
`train_and_eval`의 `elif not do_train and model_path` 분기보다 **먼저** 걸려 `run_name`이
`exp_name_<ts>`로 유지되기 때문에, wandb run 이름으로는 `--model-path` 유무를 구분할 수 없다.
runner 프로세스의 실제 argv(`ps -o cmd -p <PPID>`)를 봐야 한다. → 이 때문에 아래 repro 기록을 추가했다.

