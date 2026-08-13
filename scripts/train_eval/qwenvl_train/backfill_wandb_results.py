#!/usr/bin/env python
"""Backfill final h1 eval metrics from result_h1.json into wandb.

Why: h1 evals log the final `test_{split}_{tag}/*` metrics only at the very end of
eval() (vln_distributed_evaluator.py). With the hang-watchdog the eval process is
often killed mid-run (or a fully-resumed relaunch exits early via "No episodes
found"), so that final wandb.log never runs and the run has no `test_*` summary.
But result_h1.json is written per-episode from the lmdb, so it is always complete.

This scans checkpoint dirs, reads each `logs_<tag>/result_h1.json`, and logs its
metrics to the checkpoint's wandb run (id from `<checkpoint>/wandb_run_id.txt`)
using the SAME key format the evaluator uses: `test_{split}_{tag}/{metric}`.

Run with the Isaac python (has wandb):
  /workspace/isaaclab/_isaac_sim/python.sh scripts/train_eval/qwenvl_train/backfill_wandb_results.py --dry-run
  /workspace/isaaclab/_isaac_sim/python.sh scripts/train_eval/qwenvl_train/backfill_wandb_results.py
"""
import argparse
import glob
import json
import os


def find_checkpoints(roots):
    """A checkpoint dir = a dir containing wandb_run_id.txt. If a root is itself a
    checkpoint use it directly, else scan its immediate subdirs."""
    ckpts = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"[skip] not a dir: {root}")
            continue
        if os.path.exists(os.path.join(root, "wandb_run_id.txt")):
            ckpts.append(root)
            continue
        for name in sorted(os.listdir(root)):
            sub = os.path.join(root, name)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "wandb_run_id.txt")):
                ckpts.append(sub)
    return ckpts


def collect_metrics(ckpt):
    """Return {wandb_key: value} for every logs_<tag>/result_h1.json under ckpt.

    Key format mirrors vln_distributed_evaluator.py exactly:
    test_{split}_{tag}/{metric}, tag derived from the 'logs_<tag>' dir name.
    """
    metrics = {}
    for f in sorted(glob.glob(os.path.join(ckpt, "logs_*", "result_h1.json"))):
        logs_dir = os.path.basename(os.path.dirname(f))          # e.g. logs_none
        tag = logs_dir[len("logs_"):] or "none"                  # -> none / stop / reset
        try:
            data = json.load(open(f))
        except Exception as e:
            print(f"    [warn] cannot read {f}: {e}")
            continue
        for split, m in data.items():                            # e.g. val_unseen
            if not isinstance(m, dict):
                continue
            for k, v in m.items():                               # SR, SPL, OS, NE, Count, ...
                metrics[f"test_{split}_{tag}/{k}"] = v
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+",
                    default=["checkpoints/InternVLA-N1-DualVLN", "checkpoints/image_base"],
                    help="checkpoint dirs (or parents of checkpoint dirs) to scan")
    ap.add_argument("--entity", default="kaist-url-ai28")
    ap.add_argument("--project", default="InternNav")
    ap.add_argument("--dry-run", action="store_true", help="print what would be logged, no wandb calls")
    args = ap.parse_args()

    ckpts = find_checkpoints(args.roots)
    if not ckpts:
        print("No checkpoints with wandb_run_id.txt found.")
        return

    logged, skipped = 0, 0
    for ckpt in ckpts:
        run_id = open(os.path.join(ckpt, "wandb_run_id.txt")).read().strip()
        metrics = collect_metrics(ckpt)
        if not metrics:
            skipped += 1
            continue
        print(f"\n=== {ckpt}")
        print(f"    run_id={run_id}  ->  {args.entity}/{args.project}")
        for k in sorted(metrics):
            print(f"      {k} = {metrics[k]}")
        if args.dry_run:
            logged += 1
            continue
        import wandb
        run = wandb.init(entity=args.entity, project=args.project, id=run_id,
                         resume="allow")
        wandb.log(metrics)
        wandb.finish()
        print(f"    -> logged {len(metrics)} keys to {run.url}")
        logged += 1

    print(f"\nDone. {'(dry-run) ' if args.dry_run else ''}checkpoints logged={logged}, "
          f"skipped(no result_h1.json)={skipped}.")


if __name__ == "__main__":
    main()
