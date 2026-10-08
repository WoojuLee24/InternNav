#!/usr/bin/env python3
"""Fit the v3 pixel-goal label selector (label_rules.reference_pixel_goal_v3) and evaluate it held-out.

Per camera setting: features of every feasible candidate (label_rules.v3_candidates) on the r2r
episodes [train_eps) of every scene -> gradient-boosted classifier "is this the label frame"; the
threshold is chosen on those same training frames; v2 and v3 are then compared on episodes
[test_eps) (never seen in fitting), split by the trainer's scene split (val = last 10% of scenes).
No GPU, no habitat: parquet poses/actions + depth PNGs only.

    python scripts/eval_dashboard/fit_label_rule_v3.py --out logs/label_validation_v3
Outputs <out>/v3_model_<setting>.pkl ({"model", "threshold", "features"}) and <out>/summary.json.
"""
import argparse
import json
import math
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from internnav.habitat_extensions.vln import label_rules as LR  # noqa: E402

TRAJ = os.path.join(REPO, "data/InternData-N1/vln_ce/traj_data/r2r")


def scene_frames(args):
    """[(scene, episode, t, label_k, label_uv, candidates)] for one scene / setting / episode range."""
    import pyarrow.parquet as pq
    from PIL import Image

    scene, setting, e0, e1 = args
    cam = LR.camera(int(setting.split("cm")[0]))
    meta = [json.loads(line) for line in open(f"{TRAJ}/{scene}/meta/episodes.jsonl")][e0:e1]
    out = []
    for m in meta:
        idx = m["episode_index"]
        df = pq.read_table(f"{TRAJ}/{scene}/data/chunk-{idx // 1000:03d}/episode_{idx:06d}.parquet").to_pandas()
        poses = [np.array(p.tolist(), dtype=np.float64) for p in df[f"pose.{setting}"]]
        actions = df["action"].tolist()
        ks = df[f"relative_goal_frame_id.{setting}"].tolist()
        goals = [g.tolist() for g in df[f"goal.{setting}"]]
        for t in range(len(poses)):
            depth = np.array(Image.open(f"{TRAJ}/{scene}/videos/chunk-000/observation.images.depth.{setting}/"
                                        f"episode_{idx:06d}_{t}.png")).astype(np.float32) / 1000.0
            v2 = LR.reference_pixel_goal_v2(poses, t, cam, depth)
            out.append({"scene": scene, "ep": idx, "t": t, "k": None if ks[t] == -1 else ks[t], "goal": goals[t],
                        "cands": LR.v3_candidates(poses, actions, t, cam, depth),
                        "v2": None if v2 is None else (v2[0], list(v2[1]))})
    return out


def frame_metrics(frames, pick):
    match = both = px10 = px2 = 0
    for f in frames:
        b = pick(f)
        if f["k"] is None and b is None:
            match += 1
        elif f["k"] is not None and b is not None:
            both += 1
            match += b[0] == f["k"]
            d = math.dist(b[1], f["goal"])
            px10 += d <= 10
            px2 += d <= 2
    return {"frames": len(frames), "frame_match": round(match / max(len(frames), 1), 4),
            "px_le10": round(px10 / max(both, 1), 4), "px_le2": round(px2 / max(both, 1), 4)}


def v3_pick(model, thr):
    def pick(f):
        if not f["cands"]:
            return None
        p = model.predict_proba(np.array([c[2] for c in f["cands"]], dtype=float))[:, 1]
        i = int(np.argmax(p))
        return (f["cands"][i][0], f["cands"][i][1]) if p[i] >= thr else None
    return pick


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings", default="125cm_30deg,60cm_30deg")
    ap.add_argument("--train-eps", default="0:2", help="episode index range per scene used for fitting")
    ap.add_argument("--test-eps", default="2:4", help="held-out episode index range per scene")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    os.makedirs(args.out, exist_ok=True)
    scenes = sorted(os.listdir(TRAJ))
    val_scenes = set(scenes[-max(1, math.ceil(len(scenes) * 0.1)):])
    tr0, tr1 = map(int, args.train_eps.split(":"))
    te0, te1 = map(int, args.test_eps.split(":"))
    summary = {"_meta": {"train_eps": args.train_eps, "test_eps": args.test_eps, "val_scenes": sorted(val_scenes),
                         "features": list(LR.V3_FEATURES)}}
    for setting in args.settings.split(","):
        with Pool(args.jobs) as pool:
            train = [f for fs in pool.map(scene_frames, [(s, setting, tr0, tr1) for s in scenes]) for f in fs]
            test = [f for fs in pool.map(scene_frames, [(s, setting, te0, te1) for s in scenes]) for f in fs]
        X = np.array([c[2] for f in train for c in f["cands"]], dtype=float)
        y = np.array([int(f["k"] is not None and c[0] == f["k"]) for f in train for c in f["cands"]])
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               random_state=0).fit(X, y)
        thr = max((0.05, 0.1, 0.15, 0.2, 0.3), key=lambda th: frame_metrics(train, v3_pick(model, th))["frame_match"])
        with open(os.path.join(args.out, f"v3_model_{setting}.pkl"), "wb") as fp:
            pickle.dump({"model": model, "threshold": thr, "features": list(LR.V3_FEATURES), "setting": setting}, fp)
        res = {"threshold": thr, "train_candidates": int(len(y)), "train_positives": int(y.sum())}
        for split in ("train", "val"):
            T = [f for f in test if (f["scene"] in val_scenes) == (split == "val")]
            res[f"heldout_{split}"] = {"v2": frame_metrics(T, lambda f: f["v2"]),
                                       "v3": frame_metrics(T, v3_pick(model, thr))}
        summary[setting] = res
        print(setting, json.dumps(res), flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=1)


if __name__ == "__main__":
    main()
