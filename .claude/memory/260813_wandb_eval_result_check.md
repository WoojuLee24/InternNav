# h1 eval 결과의 wandb 기록 확인 (2026-08-13)

## 결론
- 4개 eval(2 checkpoint × flash-collision none/stop) **모두 정상 완료**, `result_h1.json` 4개 존재 (각 Count=1347).
- eval 프로세스 자체는 wandb에 기록 **실패**했음. 원인: 학습 run을 resume 하면서 step이 이미 1577 → eval의 per-episode `wandb.log(..., step=count)`가 1..1347이라 전부 무시됨
  (`wandb: WARNING Tried to log to step N that is less than the current step 1577`). 또한 마지막 `test_*` summary log(`vln_distributed_evaluator.py:415`)는 실행 전에 sim이 shutdown 되어 호출되지 않음.
- **2026-08-11 21:32에 `backfill_wandb_results.py`로 backfill 완료** → 현재 wandb에 `test_val_unseen_{none,stop}/*` 키로 정상 기록됨.

## 위치
| 항목 | 경로 |
|---|---|
| GPU1 (원본 ckpt) 결과 | `checkpoints/InternVLA-N1-DualVLN/logs_{none,stop}/result_h1.json` |
| GPU2 (학습 ckpt) 결과 | `checkpoints/image_base/base_s1.fpv_s2.fpv_20260713_073103/logs_{none,stop}/result_h1.json` |
| eval raw log | 같은 dir의 `test_h1.log`, `eval_<timestamp>.log` |
| wandb run id | `<checkpoint>/wandb_run_id.txt` |
| wandb (원본) | run name `InternVLA-N1-DualVLN` — https://wandb.ai/kaist-url-ai28/InternNav/runs/608d7apr |
| wandb (학습) | run name `image_base/base_s1.fpv_s2.fpv_20260727_073310` — https://wandb.ai/kaist-url-ai28/InternNav/runs/yzfbhlkr |

주의: yzfbhlkr의 wandb 표시 이름(`..._20260727_073310`)은 checkpoint 디렉토리 이름(`..._20260713_073103`)과 다름.
eval을 resume 할 때마다 `--wandb_run_name`이 새 timestamp로 들어가 run 이름이 덮어써졌기 때문. 로컬 config.yaml의 `run_name`은 원래 학습 이름 `image_base/base_s1.fpv_s2.fpv_20260713_073103` 그대로 남아 있음.
| backfill local run | `wandb/run-20260811_2132*-{608d7apr,yzfbhlkr}/files/wandb-summary.json` |

## 성능 (val_unseen, Count=1347)
| ckpt / flash-collision | SR | SPL | OS | NE | TL | CR | CFSR |
|---|---|---|---|---|---|---|---|
| InternVLA-N1-DualVLN / none | 0.5984 | 0.5293 | 0.6592 | 4.163 | 9.9355 | 0.0 | 0.5984 |
| InternVLA-N1-DualVLN / stop | 0.4276 | 0.3843 | 0.4967 | 5.089 | 8.7184 | 0.3477 | 0.1455 |
| base_s1.fpv_s2.fpv_20260713 / none | 0.3756 | 0.3337 | 0.4469 | 5.2338 | 8.5907 | 0.0 | 0.3756 |
| base_s1.fpv_s2.fpv_20260713 / stop | 0.2390 | 0.2096 | 0.2955 | 5.8855 | 7.7956 | 0.3300 | 0.0097 |

## 주의
- 실행 커맨드가 none/stop 둘 다 `tee run_gpu1.log`(gpu2도 동일)라 **첫 번째(none) stdout 로그가 두 번째(stop)에 덮어써짐**. 다만 per-run 로그는 `logs_none/`, `logs_stop/`에 따로 남아 손실 없음.
- 재발 방지: eval 시 `--wandb-new-run`을 쓰거나 backfill 스크립트를 eval 후 자동 실행하면 step 충돌 문제를 피할 수 있음.

## 신규 eval (checkpoints/InternVLA-N1-w-NavDP, wandb_run_id.txt 없음) 예상 동작
1. runner가 새 run 생성: name `original/InternVLA-N1-w-NavDP_<ts>`, **project `huggingface`** (runner.py:416-425) → id를 `checkpoints/InternVLA-N1-w-NavDP/wandb_run_id.txt`에 저장
2. eval subprocess가 그 id를 resume 하면서 `wandb.init(name=run_name)` → run 이름이 **`image_base/base_s1.fpv_s2.fpv_<ts>`** 로 덮어써짐
   (`run_name = _derive_exp_name(config) + launch timestamp`, runner.py:487 / evaluator `_wandb_init`)
   - 근거: huggingface project의 608d7apr 이름이 `image_base/base_s1.fpv_s2.fpv_20260726_210632`(eval 실행 시각)로 바뀌어 있음
3. per-episode `eval_val_unseen_none/samples`(step=count)는 **새 run이라 step 0부터라 이번엔 정상 기록됨**
4. 그러나 최종 `test_val_unseen_*` 은 **여전히 기록 안 됨** → backfill 필요.
   이유: h1 eval은 `self.env.close()`(Isaac Sim shutdown)에서 프로세스가 죽어 `vln_distributed_evaluator.py:410-416`에 도달하지 못함.
   전체 h1 eval 로그 어디에도 `--- VlnMultiEvaluator end ---` 가 없음 (마지막 줄이 항상 `Vln Env close`), run state도 `crashed`.

### 프로젝트 주의
- runner/evaluator eval run → **`kaist-url-ai28/huggingface`** (h1 eval_settings에 `wandb_project` 없음 → default "huggingface")
- 기존 `kaist-url-ai28/InternNav`의 동일 id run들은 2026-08-04에 별도로 sync/생성된 것. backfill 기본값(`--project InternNav`)이 여기를 가리킴.
- 신규 NavDP eval backfill 시: `--roots checkpoints/InternVLA-N1-w-NavDP --project huggingface`

## 2026-08-13 변경 (project 고정 + 시작 시 즉시 기록)
1. `default_config.py`: 모듈 상수 `WANDB_ENTITY = env WANDB_ENTITY or "kaist-url-ai28"`, `WANDB_PROJECT = env WANDB_PROJECT or "InternNav"` 추가.
   h1 `eval_settings`에 `wandb_entity` / `wandb_project` 추가 (h200/5090 habitat 경로는 기존 `EVAL_MACHINE[...]["wandb_project"]` 그대로, 미변경).
2. `runner.py`: `_wandb_target()` 헬퍼 추가, placeholder run 생성 `wandb.init(project="huggingface")` → `wandb.init(**_wandb_target())` 2곳 교체.
   → runner가 만드는 run과 eval subprocess가 resume 하는 run이 **같은 entity/project(InternNav)** 로 일치.
3. `vln_distributed_evaluator._wandb_init()`: `entity=` 인자 추가 + **init 직후 `wandb.log({f"eval_{tag}/started": 1})`** 및
   `[wandb] eval logging -> <url> (project=, name=, step=)` 출력 → 첫 episode(수 분) 전에 기록 여부 즉시 확인 가능.

### 검증 (throwaway run 생성 → 확인 → 삭제 완료)
- 2-process 테스트(runner 생성 → env resume): 같은 id 유지, `kaist-url-ai28/InternNav`에 생성 확인
- eval 쪽 `wandb.init(name=...)`가 run 이름을 **덮어씀** 확인 (`ZZZ_smoke_delete_me` → `image_base/base_s1.fpv_s2.fpv_TEST`)
- startup marker step 0에 기록 → 이후 `step=1` per-episode 로그 정상 수락
- 설정 로드 확인: `_wandb_target() = {'entity':'kaist-url-ai28','project':'InternNav'}`, h1 eval_settings에 동일 값

## 2026-08-13 변경 (2): run 이름 = checkpoint, 최종 log를 env.close() 앞으로
4. `runner.py` `_ckpt_run_name(model_path)` 추가 + eval-only 분기(`not do_train and model_path`)에서 `run_name` 을 이것으로 교체.
   - `checkpoints/image_base/s1.bev.rgb.concat_s2.fpv_20260714_024944` → `image_base/s1.bev.rgb.concat_s2.fpv_20260714_024944` (= 그 ckpt를 학습한 run 이름과 동일 → 더 이상 rename churn 없음)
   - `checkpoints/InternVLA-N1-w-NavDP/` → `InternVLA-N1-w-NavDP`
   - `checkpoints` 하위가 아니면 dir 이름으로 fallback. 학습 경로(do_train=True)는 기존 `<exp_name>_<ts>` 그대로.
5. `vln_distributed_evaluator.py`: 최종 `test_*` wandb.log + `wandb.finish()` 블록을 **`self.env.close()` 앞으로 이동**.
   → Isaac Sim shutdown 전에 업로드가 끝나므로 **backfill 불필요**. 값은 `result_logger.last_result`(= result_h1.json과 동일 내용, 매 episode 갱신)이라 backfill과 동일.

## 2026-08-13 변경 (3): wandb 키를 `test_<split>_<tag>/` 하나로 통일
- 기존 per-episode 로그 `eval_{split}_{tag}/samples` (+ `step=count`) → **`test_{split}_{tag}/{모든 metric}`, step 인자 제거**.
  - 섹션이 `eval_*`/`test_*`로 갈리지 않고 하나로 합쳐짐
  - step 충돌로 drop 되던 문제 소멸 (학습 run resume 시에도 기록됨)
  - 매 episode 최신 SR/SPL/...이 wandb에 올라가므로 중간에 죽어도 마지막 값이 남음 (backfill 이중 안전장치)
- startup marker도 `eval_{tag}/started` → **`test_{split}_{tag}/Count = 0`** (새 키 안 만듦). watchdog 재시작 시 차트에 0으로 한 번 dip 생기는 건 감수.
- 최종 블록(`test_*` + `wandb.finish()`)은 그대로, 위치만 `env.close()` 앞.
- `checkpoints/InternVLA-N1-w-NavDP/wandb_run_id.txt` = `sxji9aal` (08-13 10:55 eval이 생성, huggingface에 있음). InternNav에는 없으므로 다음 실행 시 InternNav에 같은 id로 새 run 생성됨(step 0).

## 2026-08-13 고아 eval 프로세스 사고 + 재발 방지
### 증상
GPU1 70.6GB / GPU2 63.1GB 점유. GPU당 eval.py가 **3개**씩 동시 실행 중이었음 (10:55, 11:32, 11:41 런치분 전부 생존, 각 21~24GB).
셋 다 같은 `EVAL_OUTPUT_DIR`(=`<ckpt>/logs_none`)의 lmdb·result_h1.json에 동시에 쓰고 있었음 → 결과 오염(NavDP result_h1.json이 Count=100 섞인 값).

### 원인
`_run_with_watchdog`가 eval을 `start_new_session=True`로 띄움(행 감지 시 트리째 kill 하려고). 부작용으로 터미널 Ctrl+C(SIGINT는 foreground process group에만 전달)가 자식에 도달하지 못하고,
`finally:`도 로그 파일만 닫아서 runner만 죽고 eval.py는 PPID=1 고아로 계속 실행됨.

### 조치
- runner 먼저 kill → 그 다음 `kill -TERM -<session leader>`로 6개 프로세스 그룹 정리. GPU1/2 각각 22MiB/4MiB로 회수 확인.
- 오염된 lmdb/result_h1.json 삭제 (권한 문제로 사용자가 직접 실행)
- `runner.py`: `_on_signal`(SIGINT/SIGTERM) 핸들러 + `except KeyboardInterrupt: return 130` + `finally`에서 `_kill_group(proc)` 추가.
  검증: 가짜 eval(bash 루프 + 손자 `sleep 100000`)로 SIGTERM/SIGINT 둘 다 테스트 → 자식·손자 모두 정리되고 `watchdog returned 130`.
  (주의: `kill -9`로 runner를 죽이면 여전히 고아가 됨)

## 2026-08-13 GPU2(bev) eval 중단 원인 = chat_template.json 누락
- 증상: 13:18 / 13:28 / 13:39 세 번 launch, 각 eval 로그 17KB에서 멈춤(10분 = idle_limit) → watchdog 3 strike 후 give up.
- 로그: `s2 infer error: No chat template is set for this processor.` — 에피소드가 하나도 안 끝나니 출력이 없어 hang으로 판정됨.
- 근본 원인: `TRAIN_MACHINE`에 `h1` 항목이 없어 `--machine h1`에서도 `p.system2_ckpt`가 h200 경로(`/home/irteam/data-vol2/...`)로 남음
  → `_backfill_chat_template`가 복사 실패, 경고 한 줄만 찍고 진행.
- 2차 문제: `train_and_eval`이 `run_eval`의 rc를 버려서 runner가 항상 exit 0 → `cmd1 && cmd2`가 실패에도 다음 명령을 실행함
  (그래서 오타 경로 `checkpoints/image_base/image_base/...`인 두 번째 명령이 돌아 `undp32td` 빈 wandb run과 `checkpoints/image_base/image_base/` 디렉토리가 생김).

### 조치
1. `checkpoints/image_base/s1.bev.rgb.concat_s2.fpv_20260714_024944/`에 chat_template.json 복사 (`checkpoints/InternVLA-N1-System2/`에서)
2. `EVAL_MACHINE["h1"]["system2_ckpt"] = /ws/src/InternNav/checkpoints/InternVLA-N1-System2` 추가
3. `_backfill_chat_template(ckpt, system2_ckpt, fallbacks=())` — fallback 인자 추가(기본값 있어 하위 호환), 전부 실패하면 `WARNING ... S2 will fail on every step` 명시 출력
4. `train_and_eval`이 eval rc를 return → `main_cli`에서 `sys.exit(rc)` → `&&` 체인이 실패에서 멈춤
- 검증: fallback 복사 성공/전부 실패 시 경고 출력, run_eval=1 모킹 시 `train_and_eval` → 1 리턴 확인

### chat_template.json 누락 ckpt 목록 (평가 전 확인 필요)
`base_s1.fpv_s2.fpv_20260709_130547`, `s1.bev.rgb.concat_s2.fpv_20260710_083435`, `s1.bev.rgb.replace_s2.fpv_20260711_040340`,
`s1.bev.rgb.replace_s2.fpv_20260714_223037`, `s1.depth.colormap.replace_s2.fpv_20260709_131534`, `s1.depth.normalized.replace_s2.fpv_20260710_060356`
(이제 runner가 자동 backfill 하므로 그대로 돌려도 됨)

## backfill 재실행 (필요 시)
`/workspace/isaaclab/_isaac_sim/python.sh scripts/train_eval/qwenvl_train/backfill_wandb_results.py --dry-run`
