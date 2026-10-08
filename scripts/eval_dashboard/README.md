# Eval dashboard

Browses habitat R2R val_unseen eval results under the checkpoints root: a leaderboard per experiment group, a task page per (checkpoint × config), and an episode page with the trajectory on the navmesh map, a frame player, and the S2 text at each frame. It is ported from the VLN-Challenge `dev/jay` `dashboard/`.

## Data it reads (written by the eval, nothing extra to run)

```
<checkpoints_root>/<group>/<run>[/checkpoint-N]/logs/<exp_slug>/
  result_<machine>.json      one row per finished eval (+ ckpt, ckpt_step, config, git_commit)
  progress.json              one row per episode, every rank (+ ckpt, run_stamp, tl, ndtw_ref)
  raw/<run_stamp>/           eval_recorder.py, on by default (Params.eval_save_raw)
    run.json
    episodes/<scene>_<ep>.json   meta + per-frame pose/action/dist_to_goal/S2 text + metrics
    topdown/<scene>_y<h>.json    navmesh top-down grid (RLE)
```

- Evals from before `raw/` existed show up as metrics-only rows, with no episode pages.
- `raw/` sits inside the checkpoint dir, so `kube/upload_checkpoints_rclone.sh` uploads it together with the checkpoint.

## Run

Flask server:
```bash
python scripts/eval_dashboard/app.py --root /home/irteam/data-vol2/checkpoints --port 8088
```
- `--root` sets the checkpoints root to scan. The default comes from `config.json`.
- `--port` / `--host` set the bind address. The default is `0.0.0.0:8088`.

Static export, for the k8s pod where no port can be exposed. Open it through the Jupyter `/files/` link:
```bash
python scripts/eval_dashboard/build_static.py --out logs/eval_dashboard --filter baseline_ablation
```
- The result is at `https://<kubeflow>/notebook/p-internnav/<pod>/files/git/InternNav/logs/eval_dashboard/index.html`.
- `--filter` keeps only tasks whose path contains the given string.
- The export is read-only: annotate with the Flask app. Rebuild it to pick up new evals.

## Metrics

- **SR@r**: STOP was called and the final distance is ≤ r. With r=3 this reproduces habitat Success.
- **SPL@r**: uses the geodesic start→goal distance. It matches habitat SPL per episode.
- **OSR@r**: the minimum distance along the path is ≤ r.
- **NDTW**: `ndtw_ref`, nDTW against the R2R reference path resampled every 0.25 m. It is not the official VLN-CE nDTW, which needs dense GT that is only shipped for RxR.
- **TL**: the agent's path length.

Failure cases:

| Case | Meaning |
|---|---|
| `passed_goal` | was within 3 m at some point but ended outside |
| `never_reached` | never got within 3 m |
| `no_stop` | hit the step limit without calling STOP |

Manual tags are stored per task in `fail_analysis.json`. Error points are stored in `error_points.json`. The shared tag list is `<root>/_eval_dashboard/fail_reasons.json`.

## Checks

```bash
python internnav/habitat_extensions/vln/eval_recorder.py
python scripts/eval_dashboard/data.py
```
- `eval_recorder.py` runs the recorder helpers' self-check.
- `data.py` runs the metric/split self-check.
- The GPU-free end-to-end recorder check is described in the docstring of `check_recorder.py`.
