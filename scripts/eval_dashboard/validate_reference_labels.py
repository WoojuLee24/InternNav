#!/usr/bin/env python3
"""Validate the reference ("GT-like") decisions against the InternData-N1 r2r labels (no GPU).

For sampled r2r episodes of the train and val splits (the trainer's scene split, val_ratio 0.1):
  (a) dataset rule   label_rules.reference_pixel_goal on the GT trajectory vs the pixel-goal label
  (b) habitat        agent placed at each GT frame in a renderer-less habitat sim:
                     - ShortestPathFollower next action vs the dataset's next action
                     - reference pixel from the navmesh geodesic path to the goal vs the label
Outputs <out>/{frames.jsonl, summary.json, *.png}. One worker process per scene (habitat-sim
without a renderer segfaults on a scene switch).

    CUDA_VISIBLE_DEVICES="" python scripts/eval_dashboard/validate_reference_labels.py --out logs/label_validation --eps-per-scene 2
"""
import argparse
import glob
import gzip
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from internnav.habitat_extensions.vln import label_rules as LR  # noqa: E402

TRAJ = os.path.join(REPO, "data/InternData-N1/vln_ce/traj_data/r2r")
RAW = os.path.join(REPO, "data/InternData-N1/vln_ce/raw_data/r2r/train/train.json.gz")
SETTINGS = ["125cm_30deg", "60cm_30deg"]  # 125cm_30deg == the habitat eval camera (1.25 m, 2x15 deg ld30)
ACT = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}


# ----------------------------------------------------------------------------- worker (one scene)
def worker(scene, eps_per_scene, out_path, ep_offset=0, v3_dir=None):
    import habitat
    import pyarrow.parquet as pq
    import quaternion
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from habitat_baselines.config.default import get_config
    from PIL import Image

    from internnav.configs.evaluator import EnvCfg
    from internnav.env.habitat_env import HabitatEnv

    cfg = get_config(os.path.join(REPO, "scripts/eval/configs/vln_r2r_full_ld30.yaml"))
    with habitat.config.read_write(cfg):
        cfg.habitat.simulator.agents.main_agent.sim_sensors = {}
        cfg.habitat.simulator.create_renderer = False
        cfg.habitat.dataset.split = "train"
        cfg.habitat.dataset.content_scenes = [scene]
    env = HabitatEnv(EnvCfg(env_type="habitat", env_settings={"habitat_config": cfg, "rank": 0, "world_size": 1,
                                                             "output_path": "/nonexistent"}))
    sim = env._env.sim
    by_instr = {e.instruction.instruction_text.strip(): e for e in env._env.episodes}
    env._env.current_episode = next(iter(env._env.episodes))
    env._env.reset()
    follower = ShortestPathFollower(sim, goal_radius=0.25, return_one_hot=False)
    meta = [json.loads(line) for line in open(f"{TRAJ}/{scene}/meta/episodes.jsonl")]
    rows = []
    for m in meta[ep_offset:ep_offset + eps_per_scene]:
        instr = m["tasks"][0].split("<INSTRUCTION_SEP>")[0].strip()
        ep = by_instr.get(instr)
        if ep is None:
            continue
        idx = m["episode_index"]
        df = pq.read_table(f"{TRAJ}/{scene}/data/chunk-{idx // 1000:03d}/episode_{idx:06d}.parquet").to_pandas()
        raw_actions = df["action"].tolist()
        actions = raw_actions[1:] + [0]  # loader alignment: action taken after frame t
        goal = np.array(ep.goals[0].position)
        sp, sr = ep.start_position, ep.start_rotation
        q0 = np.quaternion(sr[3], *sr[:3])
        for setting in SETTINGS:
            cam = LR.camera(int(setting.split("cm")[0]))
            v3 = None
            if v3_dir and os.path.exists(os.path.join(v3_dir, f"v3_model_{setting}.pkl")):
                import pickle

                with open(os.path.join(v3_dir, f"v3_model_{setting}.pkl"), "rb") as fp:
                    v3 = pickle.load(fp)
            poses = [np.array(p.tolist(), dtype=np.float64) for p in df[f"pose.{setting}"]]
            ks = df[f"relative_goal_frame_id.{setting}"].tolist()
            goals = [g.tolist() for g in df[f"goal.{setting}"]]
            for t in range(len(poses)):
                dp = f"{TRAJ}/{scene}/videos/chunk-000/observation.images.depth.{setting}/episode_{idx:06d}_{t}.png"
                depth = np.array(Image.open(dp)).astype(np.float32) / 1000.0 if os.path.exists(dp) else None
                label = None if ks[t] == -1 else (ks[t], goals[t])
                rule = LR.reference_pixel_goal(poses, t, cam, depth)
                rule2 = LR.reference_pixel_goal_v2(poses, t, cam, depth)
                rule3 = (LR.reference_pixel_goal_v3(poses, raw_actions, t, cam, depth, v3["model"], v3["threshold"])
                         if v3 else None)
                # habitat: agent at the GT floor point, GT heading
                floor = LR.floor_point(poses[t], cam)
                pos = LR.to_habitat(floor, sp, sr)
                rot = q0 * quaternion.from_rotation_vector([0.0, LR.local_yaw(poses[t]), 0.0])
                sim.set_agent_state(pos.astype(np.float32), rot)
                spf = follower.get_next_action(goal)
                path = habitat_path(sim, pos, goal)
                geo_pts = [LR.to_local(p, sp, sr) for p in densify(path)[1:]]
                geo = LR.reference_pixel_goal(poses, t, cam, depth, points=geo_pts) if geo_pts else None
                rows.append({
                    "scene": scene, "episode": idx, "t": t, "n": len(poses), "setting": setting,
                    "label_k": label[0] if label else None, "label_uv": label[1] if label else None,
                    "rule_k": rule[0] if rule else None, "rule_uv": list(rule[1]) if rule else None,
                    "rule2_k": rule2[0] if rule2 else None, "rule2_uv": list(rule2[1]) if rule2 else None,
                    "rule3_k": rule3[0] if rule3 else None, "rule3_uv": list(rule3[1]) if rule3 else None,
                    "geo_uv": list(geo[1]) if geo else None,
                    "gt_action": int(actions[t]), "spf_action": None if spf is None else int(spf),
                    "d2goal": float(np.linalg.norm((pos - goal)[[0, 2]])),
                })
    env.close()
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def habitat_path(sim, a, b):
    import habitat_sim

    sp = habitat_sim.ShortestPath()
    sp.requested_start, sp.requested_end = a, b
    return [np.array(p) for p in sp.points] if sim.pathfinder.find_path(sp) else []


def densify(path, step=0.25):
    out = list(path[:1])
    for a, b in zip(path, path[1:]):
        n = max(1, int(np.linalg.norm(b - a) // step))
        out += [a + (b - a) * i / n for i in range(1, n + 1)]
    return out


# ----------------------------------------------------------------------------- summary + figures
def summarize(rows, val_scenes, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def split(r):
        return "val" if r["scene"] in val_scenes else "train"

    summary = {}
    for setting in SETTINGS:
        for sp in ("train", "val"):
            R = [r for r in rows if r["setting"] == setting and split(r) == sp]
            if not R:
                continue
            lab = [r for r in R if r["label_k"] is not None]
            gpx = [math.dist(r["geo_uv"], r["label_uv"]) for r in lab if r["geo_uv"] is not None]
            summary[f"{setting}/{sp}"] = {"frames": len(R), "labelled": len(lab)}
            for tag, kk, uu in (("rule", "rule_k", "rule_uv"), ("rule2", "rule2_k", "rule2_uv"), ("rule3", "rule3_k", "rule3_uv")):
                both = [r for r in lab if r.get(kk) is not None]
                px = [math.dist(r[uu], r["label_uv"]) for r in both]
                match = sum((r["label_k"] is None and r.get(kk) is None) or (r["label_k"] == r.get(kk)) for r in R)
                summary[f"{setting}/{sp}"].update({
                    f"{tag}_frame_match": round(match / len(R), 4),
                    f"{tag}_found_where_label": round(len(both) / max(len(lab), 1), 4),
                    f"{tag}_found_where_no_label": round(sum(r.get(kk) is not None for r in R if r["label_k"] is None)
                                                         / max(len(R) - len(lab), 1), 4),
                    f"{tag}_px_le2": round(float(np.mean(np.array(px) <= 2)), 4) if px else None,
                    f"{tag}_px_le10": round(float(np.mean(np.array(px) <= 10)), 4) if px else None,
                    f"{tag}_px_median": round(float(np.median(px)), 2) if px else None,
                })
            summary[f"{setting}/{sp}"].update({
                "geo_px_le10": round(float(np.mean(np.array(gpx) <= 10)), 4) if gpx else None,
                "geo_px_le30": round(float(np.mean(np.array(gpx) <= 30)), 4) if gpx else None,
                "geo_px_median": round(float(np.median(gpx)), 2) if gpx else None,
            })
    # action agreement (setting-independent: take one setting)
    A = [r for r in rows if r["setting"] == SETTINGS[0]]
    for sp in ("train", "val"):
        S = [r for r in A if split(r) == sp]
        if not S:
            continue
        cm = np.zeros((4, 4), dtype=int)
        for r in S:
            if r["spf_action"] is not None:
                cm[r["gt_action"], r["spf_action"]] += 1
        stop_gt = [r for r in S if r["gt_action"] == 0]
        summary[f"action/{sp}"] = {
            "frames": len(S), "spf_agree": round(float(np.trace(cm) / max(cm.sum(), 1)), 4),
            "spf_agree_non_stop": round(float(np.trace(cm[1:, 1:]) / max(cm[1:].sum(), 1)), 4),
            "gt_stop_d2goal_median": round(float(np.median([r["d2goal"] for r in stop_gt])), 3) if stop_gt else None,
            "gt_stop_within_3m": round(float(np.mean([r["d2goal"] < 3.0 for r in stop_gt])), 4) if stop_gt else None,
            "confusion_gt_rows_spf_cols": cm.tolist(),
        }

    # figures ---------------------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(24, 4))
    for ax, key, title in ((axs[0], "rule_uv", "rule v1 vs label"), (axs[1], "rule2_uv", "rule v2 vs label"),
                           (axs[2], "rule3_uv", "rule v3 (learned) vs label"), (axs[3], "geo_uv", "geodesic path vs label")):
        for setting in SETTINGS:
            for sp in ("train", "val"):
                e = [math.dist(r[key], r["label_uv"]) for r in rows
                     if r["setting"] == setting and split(r) == sp and r["label_k"] is not None and r[key] is not None]
                if e:
                    ax.hist(np.clip(e, 0, 200), bins=40, histtype="step", label=f"{setting}/{sp} (n={len(e)})", density=True)
        ax.set_title(f"pixel error: {title}")
        ax.set_xlabel("px (clipped at 200)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "px_error_hist.png"), dpi=90)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    for setting in SETTINGS:
        for kk, ls in (("rule_k", ":"), ("rule2_k", "--"), ("rule3_k", "-")):
            dk = [r[kk] - r["label_k"] for r in rows if r["setting"] == setting
                  and r["label_k"] is not None and r.get(kk) is not None]
            ax.hist(np.clip(dk, -10, 20), bins=np.arange(-10.5, 21.5), histtype="step", ls=ls, density=True,
                    label=f"{setting} {dict(rule_k='v1', rule2_k='v2', rule3_k='v3')[kk]}")
    ax.set_title("look-ahead k: rule - label (dotted v1, dashed v2, solid v3)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "k_diff_hist.png"), dpi=90)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, sp in zip(axs, ("train", "val")):
        s = summary.get(f"action/{sp}")
        if not s:
            continue
        cm = np.array(s["confusion_gt_rows_spf_cols"], dtype=float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(4), [ACT[i] for i in range(4)], fontsize=8)
        ax.set_yticks(range(4), [ACT[i] for i in range(4)], fontsize=8)
        ax.set_xlabel("ShortestPathFollower")
        ax.set_ylabel("dataset (GT)")
        ax.set_title(f"{sp}: agree {s['spf_agree']:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "action_confusion.png"), dpi=90)
    plt.close()

    samples(rows, split, out)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def samples(rows, split, out, n=6):
    """Look-down frames with label (green), empirical rule (red), geodesic reference (blue)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = np.random.default_rng(0)
    for setting in SETTINGS:
        for sp in ("train", "val"):
            lab = [r for r in rows if r["setting"] == setting and split(r) == sp and r["label_k"] is not None]
            agree = [r for r in lab if r["rule_uv"] and math.dist(r["rule_uv"], r["label_uv"]) <= 10]
            disagree = [r for r in lab if r["rule_uv"] and math.dist(r["rule_uv"], r["label_uv"]) > 30]
            picks = ([("agree", agree[i]) for i in rng.permutation(len(agree))[:n // 2]]
                     + [("disagree", disagree[i]) for i in rng.permutation(len(disagree))[:n // 2]])
            if not picks:
                continue
            fig, axs = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 3.2), squeeze=False)
            for ax, (tag, r) in zip(axs[0], picks):
                img = f"{TRAJ}/{r['scene']}/videos/chunk-000/observation.images.rgb.{setting}/episode_{r['episode']:06d}_{r['t']}.jpg"
                if os.path.exists(img):
                    ax.imshow(Image.open(img))
                for key, c, lbl in (("label_uv", "lime", "label"), ("rule_uv", "red", "rule v1"),
                                    ("rule2_uv", "orange", "rule v2"), ("rule3_uv", "magenta", "rule v3"),
                                    ("geo_uv", "deepskyblue", "geodesic")):
                    if r[key]:
                        ax.plot(*r[key], "o", ms=9, mfc="none", mec=c, mew=2.2, label=lbl)
                ax.set_title(f"{tag} {r['scene'][:6]} ep{r['episode']} t{r['t']}\nk label={r['label_k']} v1={r['rule_k']} "
                             f"v2={r.get('rule2_k')} v3={r.get('rule3_k')}", fontsize=7)
                ax.axis("off")
            axs[0][0].legend(fontsize=6, loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"samples_{setting}_{sp}.png"), dpi=80)
            plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps-per-scene", type=int, default=2)
    ap.add_argument("--ep-offset", type=int, default=0, help="skip the first N episodes per scene (held-out check)")
    ap.add_argument("--v3-dir", default=None, help="dir with v3_model_<setting>.pkl (fit_label_rule_v3.py) to add rule v3")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--worker", nargs=2, metavar=("SCENE", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args.worker[0], args.eps_per_scene, args.worker[1], args.ep_offset, args.v3_dir)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "scenes"), exist_ok=True)
    habitat_scenes = {e["scene_id"].split("/")[-2] for e in json.load(gzip.open(RAW))["episodes"]}
    scenes = sorted(s for s in os.listdir(TRAJ) if s in habitat_scenes)
    # the trainer's split: last ceil(n * val_ratio) scenes of the dataset are validation
    n_val = max(1, math.ceil(len(scenes) * args.val_ratio))
    val_scenes = set(scenes[-n_val:])

    def run(scene):
        path = os.path.join(out, "scenes", f"{scene}.jsonl")
        if not os.path.exists(path):
            subprocess.run([sys.executable, __file__, "--out", out, "--eps-per-scene", str(args.eps_per_scene),
                            "--ep-offset", str(args.ep_offset)] + (["--v3-dir", args.v3_dir] if args.v3_dir else [])
                           + ["--worker", scene, path], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return scene, os.path.exists(path)

    with ThreadPoolExecutor(args.jobs) as ex:
        done = list(ex.map(run, scenes))
    failed = [s for s, ok in done if not ok]
    rows = [json.loads(line) for f in sorted(glob.glob(os.path.join(out, "scenes", "*.jsonl"))) for line in open(f)]
    with open(os.path.join(out, "frames.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    summary = summarize(rows, val_scenes, out)
    summary["_meta"] = {"scenes": len(scenes), "val_scenes": sorted(val_scenes), "failed_scenes": failed,
                        "eps_per_scene": args.eps_per_scene, "ep_offset": args.ep_offset, "rule": LR.EMPIRICAL_RULE, "rule2": LR.EMPIRICAL_RULE_V2}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
