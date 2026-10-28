"""Tool 3: dataset GT check of the DEPLOYED model stack (dual_server -> vln-deploy S2/S1 modules, e.g. GGUF + TRT).

Replays recorded VLN-CE episodes (InternData-N1 lerobot, teacher forcing: every frame of the GT path in order)
through the same HTTP API the bridge uses (/eval_dual): forward 0° rgb/depth as image/depth, the 30° frame as
lookdown_image/lookdown_depth (sim GUIDEDOG_LOOKDOWN_CAM=1 setting). Use a dual_server in --mode sync (S2 every
plan_step_gap+1 frames, blocking) so the cadence is the habitat evaluator's (dataset_test.sh starts one).

Ground truth = what training uses (internvla_n1_lerobot_dataset.py, setting 125cm_30deg):
  S2 at frame i: goal.125cm_30deg [x, y] (= col, row of the 30° frame, text "x y") with relative_goal_frame_id k
                 -> answer "↓ > x y"; k == -1 -> discrete action actions[i+1] (0 STOP, 2 ←, 3 →; 1 = no S2 sample)
  S1 at frame i under the S2 answer of frame s (GT pixel, k_s): poses[i : s+k_s+1] relative to frame i
                 (get_trajectory_relative_to_frame(camera_deg=30) + interpolate_and_resample_trajectory, 33 points, m)
  The two helpers are loaded from the dataset source itself (ast), not copied.
Metrics: S2 output type agreement, pixel L2 / hit@px (pred pixel_frame must be "lookdown"), action agreement;
         S1 ADE / FDE vs GT path, end-point lateral sign agreement (y: left +).
Visualisation: <out>/png/<ep>_<frame>.jpg = 30° frame with GT pixel (red) / pred pixel (green) + inset top view of
         S1 pred (green) vs GT path (red), and <out>/summary.png.
Usage: python3 scripts/sim/verify/dataset_check.py --server http://127.0.0.1:5894 --episodes 30 --out DIR
"""
import argparse
import ast
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]  # InternNav
DS_SRC = ROOT / "internnav/dataset/internvla_n1_lerobot_dataset.py"
SCENE_ROOT = "data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r"
ACT = {0: "STOP", 1: "↑", 2: "←", 3: "→", 5: "↓"}


def load_gt_helpers():
    """Exec only the trajectory helpers of the training dataset module (its imports need decord etc.)."""
    names = {"interpolate_and_resample_trajectory", "get_trajectory_relative_to_frame",
             "smooth_and_resample_trajectory", "xy_to_delta_xyt"}
    tree = ast.parse(DS_SRC.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = names - {n.name for n in body}
    assert not missing, f"helpers not found in {DS_SRC}: {missing}"
    from scipy.interpolate import CubicSpline
    g = {"np": np, "CubicSpline": CubicSpline}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(DS_SRC), "exec"), g)
    return g


def gt_path(G, poses):
    rel = G["get_trajectory_relative_to_frame"](np.asarray(poses, np.float64), camera_deg=30)
    traj, _ = G["interpolate_and_resample_trajectory"](rel, 32)
    return np.asarray(traj)[:, :2]


def load_episodes(root, n, scenes_skip):
    import pandas as pd

    eps = []
    for scene in sorted(Path(root).iterdir())[scenes_skip:]:
        for line in open(scene / "meta/episodes.jsonl"):
            e = json.loads(line)
            idx = e["episode_index"]
            v = scene / "videos/chunk-000"
            L = e["length"]
            fr = lambda kind, deg, i, ext: v / f"observation.images.{kind}.125cm_{deg}deg/episode_{idx:06d}_{i}.{ext}"  # noqa
            frames = [(fr("rgb", 0, i, "jpg"), fr("depth", 0, i, "png"), fr("rgb", 30, i, "jpg"), fr("depth", 30, i, "png"))
                      for i in range(L)]
            if not all(p.exists() for f in frames for p in f):
                continue
            df = pd.read_parquet(scene / f"data/chunk-000/episode_{idx:06d}.parquet")
            acts = df["action"].tolist()
            eps.append({"id": f"{scene.name}_{idx}", "instruction": e["tasks"][0].strip(), "frames": frames,
                        "actions": acts[1:] + [0],  # action taken AT frame i (dataset convention)
                        "k": [int(x) for x in df["relative_goal_frame_id.125cm_30deg"]],
                        "goal": [list(map(int, g)) for g in df["goal.125cm_30deg"]],
                        "poses": [np.stack(p).astype(np.float64) for p in df["pose.125cm_30deg"]]})
            break  # one episode per scene
        if len(eps) >= n:
            break
    return eps


def png_bytes(arr, mode=None):
    b = io.BytesIO()
    Image.fromarray(arr, mode).save(b, format="PNG")
    return b.getvalue()


def depth_req(path):
    """dataset depth png (mm) -> request png (1e-4 m, uint16) as the bridge sends."""
    d_m = np.asarray(Image.open(path)).astype(np.float32) / 1000.0
    return png_bytes(np.clip(d_m * 10000.0, 0, 65535).astype(np.uint16))


def jpg_bytes(path):
    return Path(path).read_bytes()


def parse_pred(rep):
    """-> ("pixel", [row, col], frame) | ("action", [ids]) | ("none",)"""
    if rep.get("discrete_action") is not None:
        return ("action", [int(a) for a in rep["discrete_action"]])
    if rep.get("pixel_goal") is not None:
        return ("pixel", [int(v) for v in rep["pixel_goal"]], rep.get("pixel_frame"))
    return ("none",)


def gt_kind(ep, i):
    if ep["k"][i] != -1:
        col, row = ep["goal"][i]
        return ("pixel", [row, col])
    a = ep["actions"][i]
    return ("action", [a])


def draw(img30, gt, pred, traj, gtp, text):
    vis = cv2.cvtColor(img30, cv2.COLOR_RGB2BGR)
    if gt and gt[0] == "pixel":
        cv2.circle(vis, (gt[1][1], gt[1][0]), 9, (0, 0, 255), 2)
    if pred and pred[0] == "pixel":
        cv2.circle(vis, (pred[1][1], pred[1][0]), 9, (0, 255, 0), 2)
        if gt and gt[0] == "pixel":
            cv2.line(vis, (gt[1][1], gt[1][0]), (pred[1][1], pred[1][0]), (0, 255, 255), 1)
    if traj is not None:  # inset top view, 4 m x 4 m, robot at bottom centre, x forward = up, y left = left
        S, sc = 160, 160 / 4.0
        ins = np.full((S, S, 3), 255, np.uint8)
        to = lambda p: (int(S / 2 - p[1] * sc), int(S - 8 - p[0] * sc))  # noqa: E731
        for path, col in ((gtp, (0, 0, 255)), (traj, (0, 160, 0))):
            if path is not None:
                pts = np.array([to(p) for p in path], np.int32)
                cv2.polylines(ins, [pts], False, col, 2)
        cv2.rectangle(ins, (0, 0), (S - 1, S - 1), (0, 0, 0), 1)
        vis[-S - 4:-4, -S - 4:-4] = ins
    for a, b in (("↑", "F"), ("←", "L"), ("→", "R"), ("↓", "D"), ("  >  ", ">")):  # cv2 fonts are ASCII only
        text = text.replace(a, b)
    cv2.putText(vis, text[:90],(6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:5894")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--scenes_skip", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = whole episode")
    ap.add_argument("--hit_px", type=float, default=30.0)
    ap.add_argument("--save_png", type=int, default=1, help="1: one jpg per S2 step")
    ap.add_argument("--scene_root", default=SCENE_ROOT)
    ap.add_argument("--out", default="scripts/sim/verify/out/dataset")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "png").mkdir(parents=True, exist_ok=True)
    G = load_gt_helpers()
    h = requests.get(args.server + "/health", timeout=5).json()
    assert h.get("mode") == "sync", f"use a --mode sync dual_server (got {h})"
    eps = load_episodes(args.scene_root, args.episodes, args.scenes_skip)
    print(f"[dataset_check] {len(eps)} episodes, server {h}", flush=True)
    rows = []
    fo = open(out / "steps.jsonl", "w")
    for ep in eps:
        n = len(ep["frames"]) if args.max_steps <= 0 else min(args.max_steps, len(ep["frames"]))
        s_frame, s_gt = None, None  # frame of the S2 answer S1 currently works under, and its GT
        for i in range(n):
            rp, dp, lp, ldp = ep["frames"][i]
            files = {"image": ("i.jpg", jpg_bytes(rp), "image/jpeg"), "depth": ("d.png", depth_req(dp), "image/png"),
                     "lookdown_image": ("l.jpg", jpg_bytes(lp), "image/jpeg"),
                     "lookdown_depth": ("ld.png", depth_req(ldp), "image/png")}
            data = {"instruction": ep["instruction"], "reset": i == 0, "frame_id": i, "seed": 1000 + i}
            t0 = time.time()
            rep = requests.post(args.server + "/eval_dual", files=files, data={"json": json.dumps(data)},
                                timeout=120).json()
            dt = time.time() - t0
            pred = parse_pred(rep)
            gt = gt_kind(ep, i)
            r = {"ep": ep["id"], "i": i, "s2": bool(rep.get("is_s2_step")), "llm": rep.get("llm_output"),
                 "pred": pred, "gt": gt, "dt": round(dt, 3)}
            if r["s2"]:
                s_frame, s_gt = i, gt
                if pred[0] == "pixel" and gt[0] == "pixel":
                    r["px_l2"] = float(np.hypot(pred[1][0] - gt[1][0], pred[1][1] - gt[1][1]))
                    r["px_frame_ok"] = pred[2] == "lookdown"
            traj = np.asarray(rep["trajectory"])[:, :2] if rep.get("trajectory") is not None else None
            gtp = None
            if traj is not None and s_gt is not None and s_gt[0] == "pixel":
                end = s_frame + ep["k"][s_frame]
                if end - i >= 2:  # GT path of at least 3 poses left
                    gtp = gt_path(G, ep["poses"][i:end + 1])
                    m = min(len(gtp), len(traj))
                    r["ade"] = float(np.mean(np.linalg.norm(traj[:m] - gtp[:m], axis=1)))
                    r["fde"] = float(np.linalg.norm(traj[m - 1] - gtp[m - 1]))
                    r["end_pred"] = traj[-1].round(3).tolist()
                    r["end_gt"] = gtp[-1].round(3).tolist()
                    r["gt_len"] = float(np.sum(np.linalg.norm(np.diff(gtp, axis=0), axis=1)))
            if args.save_png and r["s2"]:
                img30 = np.asarray(Image.open(lp).convert("RGB"))
                txt = f"{ep['id']} f{i} gt={gt[0]}{gt[1]} pred={(r['llm'] or '').strip()}"
                if "px_l2" in r:
                    txt += f" L2={r['px_l2']:.0f}"
                cv2.imwrite(str(out / "png" / f"{ep['id']}_{i:03d}.jpg"), draw(img30, gt, pred, traj, gtp, txt))
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            fo.flush()
            rows.append(r)
        print(f"[dataset_check] {ep['id']}: {n} frames, S2 {sum(1 for r in rows if r['ep'] == ep['id'] and r['s2'])}",
              flush=True)
    res = summarize(rows, args.hit_px)
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    plot_summary(rows, out)
    print("[dataset_check] " + json.dumps(res, ensure_ascii=False), flush=True)


def pct(a, q):
    a = np.asarray(a, float)
    return round(float(np.percentile(a, q)), 3) if a.size else None


def summarize(rows, hit_px):
    s2 = [r for r in rows if r["s2"]]
    res = {"episodes": len({r["ep"] for r in rows}), "frames": len(rows), "s2_calls": len(s2)}
    # S2 type agreement (GT action 1 = no S2 sample in training -> excluded)
    judged = [r for r in s2 if not (r["gt"][0] == "action" and r["gt"][1] == [1])]
    same_type = [r for r in judged if r["pred"][0] == r["gt"][0]]
    res["s2_type_agree"] = round(len(same_type) / len(judged), 3) if judged else None
    conf = {}
    for r in judged:
        g = "pixel" if r["gt"][0] == "pixel" else ACT.get(r["gt"][1][0], "?")
        p = "pixel" if r["pred"][0] == "pixel" else (ACT.get(r["pred"][1][0], "?") if r["pred"][0] == "action" else "none")
        conf.setdefault(g, {}).setdefault(p, 0)
        conf[g][p] += 1
    res["s2_confusion_gt_to_pred"] = conf
    l2 = [r["px_l2"] for r in s2 if "px_l2" in r]
    gt_px = [r for r in s2 if r["gt"][0] == "pixel"]
    res["pixel"] = {"n": len(l2), "l2_p50": pct(l2, 50), "l2_p90": pct(l2, 90),
                    f"hit@{hit_px:.0f}": round(float(np.mean(np.array(l2) <= hit_px)), 3) if l2 else None,
                    f"hit@{hit_px:.0f}_over_gt_pixel": round(sum(1 for v in l2 if v <= hit_px) / len(gt_px), 3)
                    if gt_px else None,
                    "frame_lookdown": round(float(np.mean([r["px_frame_ok"] for r in s2 if "px_l2" in r])), 3)
                    if l2 else None}
    s1 = [r for r in rows if "ade" in r]
    ys = [(r["end_pred"][1], r["end_gt"][1]) for r in s1 if abs(r["end_gt"][1]) > 0.3]
    res["s1"] = {"n": len(s1), "ade_p50": pct([r["ade"] for r in s1], 50), "ade_p90": pct([r["ade"] for r in s1], 90),
                 "fde_p50": pct([r["fde"] for r in s1], 50), "fde_p90": pct([r["fde"] for r in s1], 90),
                 "y_sign_agree(|y_gt|>0.3)": round(float(np.mean([np.sign(a) == np.sign(b) for a, b in ys])), 3)
                 if ys else None, "n_y": len(ys)}
    return res


def plot_summary(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    l2 = [r["px_l2"] for r in rows if "px_l2" in r]
    ax[0].hist(l2, bins=40, range=(0, 400))
    ax[0].axvline(30, color="r", ls="--")
    ax[0].set_title(f"S2 pixel L2 to GT [px] (n={len(l2)})")
    s1 = [r for r in rows if "ade" in r]
    ax[1].hist([r["ade"] for r in s1], bins=40, range=(0, 2))
    ax[1].set_title(f"S1 ADE vs GT path [m] (n={len(s1)})")
    e = np.array([r["end_pred"] + r["end_gt"] for r in s1]).reshape(-1, 4)
    ax[2].scatter(e[:, 3], e[:, 1], s=4)
    lim = [-3, 3]
    ax[2].plot(lim, lim, "r--", lw=0.8)
    ax[2].set_xlabel("GT end y (left +) [m]")
    ax[2].set_ylabel("pred end y [m]")
    ax[2].set_title("S1 end point lateral")
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
