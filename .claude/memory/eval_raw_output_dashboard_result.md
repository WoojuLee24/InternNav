# Eval raw output + dashboard — result (2026-09-30)

## What changed compared with before

| Area | Before | After |
|---|---|---|
| `progress.json` | Written by rank 0 only, so it held about 1/4 of the episodes (463/1839). The `if self.rank == 0` guard was a leftover from debug commit de74ff8. | Every rank writes it again, as on main (`_write_progress`, 4 places including bev/unified). With save_raw, each row also carries `ckpt`, `run_stamp`, `tl`, `ndtw_ref`. Data-vol2 is Lustre, so concurrent `O_APPEND` writes are safe. |
| Raw output | none | `internnav/habitat_extensions/vln/eval_recorder.py` writes JSON under `<log_dir>/raw/<run_stamp>/{run.json, episodes/, topdown/}`. It wraps `env.reset/step` and `model.generate`, so the episode loops are not modified. `Params.eval_save_raw=True` by default; if the eval_settings key is missing, the evaluator behaves as before. |
| `result_<machine>.json` row | metrics + timestamp | adds `ckpt`, `ckpt_step`, `config`, `machine`, `git_commit` (from the `EVAL_RUN_META` env), plus `tls_all` and `ndtw_refs_all` |
| wandb | Every checkpoint-N eval created its own `original/...` run. | A checkpoint-N eval is logged into the parent **training run**, with `test/*` plotted against `test/ckpt_step` (`define_metric`). **Only after training has finished** (final weights exist in the run dir). While training is still running, the eval keeps its own separate run as before, because a second process writing into the live run would make wandb drop the training's step logs. |
| Concurrent evals of the same ckpt+config | Allowed, so progress got mixed (the 9/29 30000 duplicate). | Blocked by `flock` on `<log_dir>/.eval.lock`. Data-vol2 is mounted with `flock`, so this also works across nodes. The second eval prints SKIP. |
| Evaluating intermediate checkpoints | Only the last checkpoint was evaluated automatically. | New runner option `--eval-ckpts all` or `--eval-ckpts 30000,40000,final`. `all` evaluates each checkpoint-N once, and skips the final model when it equals the last checkpoint-N. |
| Dashboard | none | `scripts/eval_dashboard/` (Flask `app.py`, plus `build_static.py` for a static export), ported from the VLN-Challenge `dev/jay` dashboard. See the README. |

## Commands (one line each)

Evaluate every checkpoint of one run:
```
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline_ablation/dualvln_stage1_full_gemm.py --machine h200 --no-train --model-path /home/irteam/data-vol2/checkpoints/baseline_ablation/dualvln_stage1_full_gemm_20260825_041953 --eval-ckpts all
```
- `--model-path` is the **run dir** (the parent of checkpoint-N).
- `--eval-ckpts` takes `all` or a list such as `35000,45000`. Omitting it keeps the previous single-checkpoint behavior.

Dashboard (on the workstation):
```
python scripts/eval_dashboard/app.py --root /home/irteam/data-vol2/checkpoints --port 8088
```

Static export (on the k8s pod, opened through the Jupyter `/files/` link):
```
python scripts/eval_dashboard/build_static.py --out logs/eval_dashboard
```

## Verification

- **Train argv unchanged:** `--print-train-argv` has the same md5 before and after (`f9013cfa…`).
- **Eval config:** the only new key in `eval_settings` is `save_raw: True`.
- **Self-checks:** `eval_recorder.py` (RLE / TL / nDTW / densify / dedupe) and `data.py` (failure case / split / SR@r / SPL@r) both pass.
- **End-to-end recorder check (`check_recorder.py`):** real HabitatEnv without sensors/renderer, GPU-free, 12 episodes in 3 scenes.
  - The tilt actions are excluded from frames, and `gen` is captured.
  - The last frame's `dist_to_goal` equals habitat NE.
  - Per-episode SPL@3 matches habitat SPL exactly, and SR@3 matches `sucs_all`.
- **Map geometry:** R2R reference points land on navigable cells 88% of the time with the (row=z, col=x) mapping, versus 27% with the axes swapped. The top-down map is built at the **median agent height**, because R2R episodes often start on a stair and then walk on another floor.
  - Checked visually by rendering 3 episodes: `/tmp/.../viewer_geometry_check2.png`.
- **nDTW:** the first version compared against the sparse reference (about 6 nodes), giving about 0.1 even for successful episodes. After resampling the reference every 0.25 m and dropping repeated positions, successful episodes score 0.87–0.96.
- **Flask:** curl on every read/write API passes. A DOM-shim smoke test in Node renders the leaderboard, task and 12 episode views (map, paths and agent marker drawn, no JS errors).
- **Real root (read-only):** 38 existing evals are indexed as legacy (metrics only) in 2.5 s.
- **Static export:** Jupyter serves the HTML under `sandbox allow-scripts` and the `.js` files as `application/javascript`. It is therefore implemented with `<script>`-based data chunks instead of `fetch`.

## Not yet verified

- A **real model eval with save_raw=True** was not run, because all GPUs are used by the sv_ep2 training (OOM risk; that run cannot resume). Run it on the next free node, e.g. `--max-steps 3` (3 episodes, smoke test, no wandb).
- The Flask UI in an actual browser; the user will check it on the workstation.
- The wandb `test/ckpt_step` curve: the code is in place but has not been checked on a real run.
