#!/usr/bin/env python3
"""Validate an eval log dir's raw output against progress.json and result_<machine>.json (no GPU).

    python scripts/eval_dashboard/validate_raw.py <log_dir> [--expected 1839] [--png 5]

Checks (PASS/FAIL/WARN table, JSON written to <log_dir>/validate_raw.json):
  completeness   raw episodes == unique progress episodes == result `length` (== --expected)
  per-episode    raw metrics == progress row (exact); last frame dist_to_goal == ne (frames are
                 stored rounded to 1e-4); TL recomputed from frame positions == tl (exact)
  aggregate      sucs/spls/oss/nes_all recomputed with metrics_schema in the evaluator's gather
                 order (resumed rows first, then rank 0..N of the final run) == result row, bit-exact
  structure      frame actions in {None, 0..3}; success => last action STOP; collisions == collision frames;
                 run.json meta == result row
  map            each episode's top-down file exists; share of trajectory points on navigable cells;
                 PNG of a few episodes (map + GT + trajectory) for a visual check
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from internnav.evaluator import metrics_schema  # noqa: E402
from internnav.habitat_extensions.vln.eval_recorder import DECISION_KEYS, decision_counts, path_length  # noqa: E402

METRIC_KEYS = ("success", "spl", "os", "ne", "steps", "tl", "ndtw_ref")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def decode(g):
    import numpy as np

    return np.repeat(g["rle"][0::2], g["rle"][1::2]).reshape(g["rows"], g["cols"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log_dir")
    ap.add_argument("--expected", type=int, default=None, help="dataset episode count (1839 for R2R val_unseen)")
    ap.add_argument("--png", type=int, default=5, help="episodes to render for a visual check (0 = none)")
    args = ap.parse_args()
    d = os.path.abspath(args.log_dir)
    checks = []

    def check(name, ok, detail="", warn=False):
        checks.append({"check": name, "status": "PASS" if ok else ("WARN" if warn else "FAIL"), "detail": detail})

    # ---- load
    raw = {}
    for stamp_dir in sorted(glob.glob(os.path.join(d, "raw", "*"))):  # latest stamp wins
        for f in glob.glob(os.path.join(stamp_dir, "episodes", "*.json")):
            raw[os.path.splitext(os.path.basename(f))[0]] = (stamp_dir, f)
    progress = read_jsonl(os.path.join(d, "progress.json"))
    prog = {}
    for r in progress:
        prog[f"{r['scene_id']}_{int(r['episode_id']):04d}"] = r
    results = sorted((r for f in glob.glob(os.path.join(d, "result_*.json")) for r in read_jsonl(f)),
                     key=lambda r: str(r.get("timestamp")))
    last = results[-1] if results else {}

    # ---- completeness
    counts = {"raw": len(raw), "progress_unique": len(prog), "progress_lines": len(progress),
              "result_length": last.get("length"), "expected": args.expected}
    ok = len(raw) == len(prog) and (last.get("length") in (None, len(prog)))
    if args.expected is not None:
        ok = ok and len(raw) == args.expected
    check("completeness", ok, json.dumps(counts))
    missing, extra = sorted(set(prog) - set(raw)), sorted(set(raw) - set(prog))
    check("raw/progress key sets", not missing and not extra,
          f"missing raw: {missing[:10]}{'...' if len(missing) > 10 else ''}  raw only: {extra[:10]}")
    check("no duplicate progress rows", len(progress) == len(prog),
          f"{len(progress) - len(prog)} duplicate lines (resume/rerun)", warn=True)

    # ---- per-episode
    bad_metric, bad_d2g, bad_tl, bad_act, bad_stop, gen_frames, n_frames = [], [], [], [], [], 0, 0
    eps = {}
    for k, (stamp_dir, f) in raw.items():
        e = json.load(open(f))
        eps[k] = (stamp_dir, e)
        fr, m = e["frames"], e["metrics"]
        n_frames += len(fr)
        gen_frames += sum("gen" in x for x in fr)
        if k in prog and any(not same(m.get(x), prog[k].get(x)) for x in METRIC_KEYS if x in prog[k]):
            bad_metric.append(k)
        last_d = fr[-1].get("dist_to_goal") if fr else None
        if not (last_d is None and not math.isfinite(m["ne"]) or last_d is not None and abs(last_d - m["ne"]) <= 5e-5):
            bad_d2g.append(k)
        if round(path_length([x["pose"]["position"] for x in fr]), 4) != m.get("tl"):  # stored rounded
            bad_tl.append(k)
        if fr and (fr[0]["action"] is not None or any(x["action"] not in (0, 1, 2, 3) for x in fr[1:])):
            bad_act.append(k)
        if m.get("success") and (not fr or fr[-1]["action"] != 0):
            bad_stop.append(k)
    check("raw metrics == progress row", not bad_metric, f"{len(bad_metric)} mismatched: {bad_metric[:5]}")
    check("last frame dist_to_goal == ne", not bad_d2g, f"{len(bad_d2g)} mismatched: {bad_d2g[:5]}")
    check("TL from frames == tl", not bad_tl, f"{len(bad_tl)} mismatched: {bad_tl[:5]}")
    check("frame actions valid", not bad_act, f"{len(bad_act)} bad: {bad_act[:5]}")
    check("success => STOP called", not bad_stop, f"{len(bad_stop)} bad: {bad_stop[:5]}")
    coll_eps = [k for k, (_, e) in eps.items() if "collisions" in e["metrics"]]
    if coll_eps:
        bad_coll = [k for k in coll_eps if eps[k][1]["metrics"]["collisions"] != sum(1 for f in eps[k][1]["frames"] if f.get("collision"))]
        n_coll = sum(eps[k][1]["metrics"]["collisions"] for k in coll_eps)
        check("collisions == collision frames", not bad_coll,
              f"{n_coll} collisions in {sum(eps[k][1]['metrics']['collisions'] > 0 for k in coll_eps)} episodes; mismatched: {bad_coll[:5]}")
    check("S2 text captured", gen_frames > 0, f"{gen_frames}/{n_frames} frames carry gen", warn=True)
    # decision metrics (eval_decision_metrics): dec_* counters == recount from the raw decision records
    dec_eps = [k for k, (_, e) in eps.items() if "dec_n" in e["metrics"]]
    if dec_eps:
        bad_dec = [k for k in dec_eps if decision_counts([d for f in eps[k][1]["frames"] for d in f.get("decision", [])])
                   != {x: eps[k][1]["metrics"][x] for x in DECISION_KEYS}]
        n_dec = sum(eps[k][1]["metrics"]["dec_n"] for k in dec_eps)
        check("decision counters == raw decision records", not bad_dec, f"{len(bad_dec)} mismatched: {bad_dec[:5]} ({n_dec} decisions)")

    # ---- aggregate (bit-exact, evaluator gather order)
    if last and progress and all("rank" in r and "run_stamp" in r for r in progress):
        import torch

        stamp = str(last.get("timestamp"))
        # resume: rank 0 prepends every progress line written before this run (file order)
        order = [r for r in progress if r["run_stamp"] != stamp]
        for rank in sorted({r["rank"] for r in progress}):
            order += [r for r in progress if r["run_stamp"] == stamp and r["rank"] == rank]
        per_ep = {c: torch.tensor([float(r[k]) for r in order])
                  for c, k in (("SR", "success"), ("SPL", "spl"), ("OS", "os"), ("NE", "ne"))}
        schema = metrics_schema.habitat_aggregate(per_ep)
        schema.update(metrics_schema.progress_aggregate(progress))
        diff = {c: (schema[c], last[k]) for c, k in metrics_schema.HABITAT_KEYS.items()
                if c in schema and k in last and schema[c] != last[k]}
        check("aggregate == result row (bit-exact)", not diff, json.dumps(diff) if diff else
              f"SR={schema['SR']:.6f} SPL={schema['SPL']:.6f} OS={schema['OS']:.6f} NE={schema['NE']:.4f}")
    else:
        check("aggregate == result row (bit-exact)", False,
              "skipped: no result row yet, or progress rows lack rank/run_stamp (eval before save_raw)", warn=True)

    # ---- front-view frames: exactly the episodes the run's policy keeps, one JPEG per front-view frame
    policy = {}
    runs_meta = sorted(glob.glob(os.path.join(d, "raw", "*", "run.json")))
    if runs_meta:
        policy = json.load(open(runs_meta[-1]))
    if policy.get("raw_frames") in ("fail", "all", "none"):
        bad_img = []
        for k, (stamp_dir, e) in eps.items():
            want = {"fail": not e["metrics"].get("success"), "all": True, "none": False}[policy["raw_frames"]]
            fd = e["meta"].get("frames_dir")
            jpgs = sorted(glob.glob(os.path.join(stamp_dir, fd, "*.jpg"))) if fd else []
            refs = [f["image"] for f in e["frames"] if "image" in f]
            ok = (bool(fd) == want) and len(jpgs) == len(refs) and all(
                os.path.exists(os.path.join(stamp_dir, fd, r)) for r in refs) and (not want or len(refs) > 0)
            if not ok:
                bad_img.append(k)
        n_img = sum(1 for _, (sd, e) in eps.items() if e["meta"].get("frames_dir"))
        check(f"front-view frames match policy '{policy['raw_frames']}'", not bad_img,
              f"{n_img} episodes with frames; mismatched: {bad_img[:5]}")

    # ---- run meta
    runs = sorted(glob.glob(os.path.join(d, "raw", "*", "run.json")))
    if runs and last:
        meta = json.load(open(runs[-1]))
        diff = {k: (meta.get(k), last.get(k)) for k in ("ckpt", "ckpt_step", "config", "git_commit", "machine")
                if k in last and meta.get(k) != last.get(k)}
        check("run.json == result row meta", not diff, json.dumps(diff))

    # ---- map
    # navigability only for points on the map's own floor (within 0.5 m), 1-cell tolerance: checks the
    # map geometry. Points on another floor (stair / multi-floor episodes) are counted separately.
    maps, on_nav, n_pts, no_map, multi_floor = {}, 0, 0, [], []
    for k, (stamp_dir, e) in eps.items():
        key = e["meta"].get("topdown")
        p = os.path.join(stamp_dir, "topdown", f"{key}.json")
        if not key or not os.path.exists(p):
            no_map.append(k)
            continue
        if p not in maps:
            g = json.load(open(p))
            maps[p] = (g, decode(g))
        g, grid = maps[p]
        other = 0
        for fr in e["frames"]:
            x, y, z = fr["pose"]["position"]
            if abs(y - g["height"]) > 0.5:
                other += 1
                continue
            r, c = int((z - g["origin_z"]) / g["meters_per_pixel"]), int((x - g["origin_x"]) / g["meters_per_pixel"])
            n_pts += 1
            on_nav += bool((grid[max(0, r - 1):r + 2, max(0, c - 1):c + 2] > 0).any())
        if other:
            multi_floor.append(k)
    check("top-down file per episode", not no_map, f"{len(no_map)} without map: {no_map[:5]}")
    ratio = on_nav / n_pts if n_pts else 0.0
    check("same-floor trajectory on navigable cells", ratio >= 0.99, f"{ratio:.4f} of {n_pts} points")
    check("single-floor episodes", not multi_floor,
          f"{len(multi_floor)} episodes also visit another floor (expected for stair paths): {multi_floor[:5]}", warn=True)

    if args.png and eps:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = sorted(eps)[:: max(1, len(eps) // args.png)][: args.png]
        fig, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5), squeeze=False)
        for ax, k in zip(axs[0], keys):
            stamp_dir, e = eps[k]
            p = os.path.join(stamp_dir, "topdown", f"{e['meta']['topdown']}.json")
            if p in maps:
                g, grid = maps[p]
                m = g["meters_per_pixel"]
                ax.imshow(grid, cmap="gray", origin="upper",
                          extent=(g["origin_x"], g["origin_x"] + g["cols"] * m,
                                  -(g["origin_z"] + g["rows"] * m), -g["origin_z"]))
            gt = [(p[0], -p[2]) for p in e["meta"]["reference_path"]]
            tr = [(f["pose"]["position"][0], -f["pose"]["position"][2]) for f in e["frames"]]
            ax.plot(*zip(*gt), "g-s", ms=3, label="GT")
            ax.plot(*zip(*tr), "r-", lw=1.2, label="agent")
            xs, ys = zip(*(gt + tr))
            ax.set_xlim(min(xs) - 2, max(xs) + 2)
            ax.set_ylim(min(ys) - 2, max(ys) + 2)
            ax.set_title(f"{k} succ={e['metrics'].get('success')} ne={e['metrics'].get('ne'):.2f}", fontsize=9)
            ax.legend(fontsize=7)
        plt.tight_layout()
        png = os.path.join(d, "validate_raw.png")
        plt.savefig(png, dpi=70)
        check("visual check PNG", True, png)

    width = max(len(c["check"]) for c in checks)
    for c in checks:
        print(f"{c['status']:4}  {c['check']:<{width}}  {c['detail']}")
    out = os.path.join(d, "validate_raw.json")
    with open(out, "w") as f:
        json.dump({"log_dir": d, "checks": checks}, f, indent=1)
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    print(f"\n{'ALL PASS' if not n_fail else f'{n_fail} FAIL'}  -> {out}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
