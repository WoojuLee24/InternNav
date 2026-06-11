---
name: project_train_eval_qwenvl_python
description: scripts/train_eval/qwenvl_train — shell 대신 Python으로 train+eval을 한 프로세스로 구동하는 신규 구조(batch_size 구현 완료)
metadata:
  type: project
---

기존 `scripts/train/qwenvl_train/*/*.sh`는 `train.sh → eval_dual_system_mini_8gpu.sh
→ torchrun`로 **셸이 셸을 호출**해 디버깅 불가했고, train/eval 파라미터가 어긋났다
(게다가 eval 기본 config `habitat_dual_system_mini_cfg.py`는 **디스크에 없어서** batch_size
eval은 사실상 깨져 있었음). 이를 Python으로 대체한 신규 구조를 `feature/bev_v0.1`에 추가.

**위치**: `scripts/train_eval/qwenvl_train/` (기존 `scripts/train/qwenvl_train/`는 그대로 둠)

**구조 = `runner.py --config <config.py>`** (eval.py `--config` 관례와 대칭). **runner.py = 순수 엔진**(config 정의 없음),
config(=Params 정의+기본값+eval 빌더)는 **별도 파일 `default_config.py`로 분리**. config 파일은 순수 선언적(PARAMS+eval_cfg).
같은 config 파일을 runner(train)와 eval.py(eval)가 둘 다 `--config`로 읽음.

**파일 구조 (batch_size 기준)**:
- `runner.py` — **순수 엔진/실행 진입점**(config 정의 안 들어있음). `run_train`/`run_eval`/`find_best_checkpoint`/
  `train_and_eval`/`main_cli`(`--config` 기반)+`__main__`. `default_config`에서 `Params`만 import(타입힌트용).
  `_load_config(path)`로 config 파일 import해 `PARAMS`/`EXP_NAME` 읽음(EXP_NAME 없으면 `parent/stem`으로 유추).
  torchrun을 subprocess로 **딱 2번**(train, eval) 호출, 나머지는 순수 Python(pdb 가능). `_run_and_tee`가 `2>&1 | tee` 대체.
  - `--config` 기본값 = **최상위 `default_config.py`**(= b4 baseline).
- `default_config.py`(qwenvl_train **최상위**) — **runner에서 분리한 default config = 단일 소스**.
  `Params`(dataclass+`train_argv()`+`_fmt_num`) + `HABITAT_MACHINE` + `build_habitat_eval_cfg`/`build_h1_eval_cfg`/
  `build_eval_cfg(p,target)`/`make_eval_cfg(p)` 정의. 끝에 `EXP_NAME`+`PARAMS=Params()`(baseline)+`eval_cfg` 노출 →
  config 파일 겸 schema 모듈. `TRAIN_EVAL_TARGET` env(**h200|5090|h1**, 기본 h200)로 eval 대상 선택. **runner를 import 안 함**(의존 DAG: runner→default_config, 변형 config→default_config).
  - eval config는 기존 `habitat_dual_system_mini_{h200,5090}_cfg.py`를 기준으로 재현:
    `HABITAT_MACHINE`에 머신별 infra(yaml/wandb/max_new_tokens/output_path/model_path) 정의,
    num_history/resize/predict_step_num은 학습 Params에서 주입. **h200=batch_size 학습값(8/384)과 일치**,
    5090은 다른 모델(4/256)이라 infra만 동일하고 모델 파라미터는 학습값이 들어감(의도된 동작).
- `batch_size/b4_eff128_base.py`·`b2_eff128.py`·`b8_eff128.py` — **선언적 변형 config**(A 방식, shim 아님).
  부모 dir(qwenvl_train) sys.path 추가 후 `import default_config as base`, `PARAMS = replace(base.PARAMS, batch_size=, grad_accum_steps=)`(b4는 override 없음)
  + `eval_cfg = base.make_eval_cfg(PARAMS)` + `EXP_NAME`. (구 `batch_size/config.py`·`batch_size/default_config.py` wrapper는 삭제.)
- `batch_size/verify_params.py` — 4단계 검증, 전부 `ALL PARAMETERS MATCH ✅`:
  (1) `.sh` torchrun 인자 ↔ 신규 train_argv(b2/b4/b8, 숫자는 float 비교),
  (2) eval-infra: 생성 eval_cfg infra ↔ 실제 h200/5090 파일,
  (3) eval-full: 생성 h200 eval_cfg == h200 파일 전 키(model_path/output_path는 런타임 override라 제외,
      predict_step_num=32는 evaluator 기본값과 동일),
  (4) eval-sync: 학습 PARAMS == eval model_settings(h200/5090/h1).
  검증은 실제 config 파일(b4/b2/b8)을 `runner._load_config`로 로드해 PARAMS를 읽음(파일 자체를 테스트).

**단일 소스 보장 메커니즘**: train argv와 eval `model_settings`를 **같은 Params 한 객체**에서 생성 →
num_history/resize_w/resize_h/predict_step_num/num_future_steps가 train==eval로 자동 일치.
(기존 ABLATE_* env var 간접 전달 방식 대체. [[project_qwenvl_ablation_train_eval_sync]])

**실행**:
- `python scripts/train_eval/qwenvl_train/runner.py` (기본=b4 baseline, train→habitat eval 자동)
- `python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/batch_size/b2_eff128.py`
- 플래그: `--eval-target h1` / `--no-train --model-path <ckpt>`(eval만) / `--print-train-argv` / `--data-root` / `--checkpoints-root`
- 검증: `python scripts/train_eval/qwenvl_train/batch_size/verify_params.py`

**핵심 사실/함정**:
- 분산학습은 torchrun 런처 필요 → Python 오케스트레이터도 내부에서 `torchrun` subprocess 호출(셸 1겹뿐).
- eval.py `load_eval_cfg`와 runner `_load_config` 둘 다 `spec_from_file_location`라 config 폴더를 sys.path에 안 넣음
  → runner/변형 config 모두 `import default_config` 전에 qwenvl_train dir를 `sys.path.insert`(구현됨). default_config는 runner를 import 안 함(순환 없음).
- habitat eval은 `train_and_eval`이 학습 후 자동 실행. **h1 eval은 Isaac Sim 환경 필요** → 같은 config로 선택만
  가능, 실행은 수동.
- 이 환경엔 guideline의 `/workspace/isaaclab/_isaac_sim/python.sh`가 없음 → 검증/import는 `/usr/bin/python`으로 됨.

**남은 작업(사용자 지시)**: ① runner.py의 Params 필드가 많아 "복잡" → knob/고정 분리 단순화 제안했으나 사용자가
보류시킴(미적용). ② **train·eval config 통일 완료** — 한 config 파일(default_config + 변형)을 runner와 eval.py가 둘 다 `--config`로 읽음.
③ 신규 ablation은 이 batch_size 패턴 복제(최상위 default_config.py 재사용 + 폴더당 변형 config + verify). [[feedback_new_files_only]]
