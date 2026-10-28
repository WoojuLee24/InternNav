"""Replay recorded VLN-CE episodes against a VLN HTTP server (vln_server.py or dual_server.py) like the bridge does.

Same frames, instruction, frame_id and per-step torch seed for every run -> outputs are comparable:
  run :  python3 scripts/sim/verify/bench_dual.py run --url http://127.0.0.1:5801 --tag mono --period 0
  cmp :  python3 scripts/sim/verify/bench_dual.py cmp out/bench/mono.npy out/bench/split_sync.npy
--period: seconds between requests (0 = back-to-back, as in sync equivalence; 0.3 = bridge plan_period).
Reported per run: client latency per reply kind, server S1 / S2 times, S2 period, replies/s.
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_quant import SCENE_ROOT, load_episodes  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "bench"


def enc(rgb, depth_m):
    b1, b2 = io.BytesIO(), io.BytesIO()
    Image.fromarray(rgb).save(b1, format="JPEG", quality=95)
    Image.fromarray(np.clip(depth_m * 10000.0, 0, 65535).astype(np.uint16)).save(b2, format="PNG")
    return b1.getvalue(), b2.getvalue()


def run(a):
    eps = load_episodes(a.scene_root, a.episodes, a.max_steps)
    rows, fid = [], 0
    t_start = time.time()
    for ep in eps:
        for i, (rp, dp, _) in enumerate(ep["frames"]):
            t_tick = time.time()
            rgb = np.asarray(Image.open(rp).convert("RGB"))
            d = np.asarray(Image.open(dp)).astype(np.float32)
            depth = d / 1000.0 if d.max() > 100 else d
            ib, db = enc(rgb, depth)
            fid += 1
            data = {"reset": i == 0, "instruction": ep["instruction"], "frame_id": fid, "seed": 1000 + i}
            t0 = time.time()
            r = requests.post(a.url + "/eval_dual", files={"image": ("rgb.jpg", ib, "image/jpeg"),
                                                          "depth": ("depth.png", db, "image/png")},
                              data={"json": json.dumps(data)}, timeout=600).json()
            lat = time.time() - t0
            rows.append({"ep": ep["id"], "step": i, "frame_id": fid, "client_ms": 1000 * lat, "wall": t0 - t_start,
                         "is_s2": bool(r.get("is_s2_step")), "llm": r.get("llm_output"),
                         "action": r.get("discrete_action"), "pixel": r.get("pixel_goal"),
                         "pixel_frame_id": r.get("pixel_frame_id"),
                         "traj": np.asarray(r["trajectory"]) if "trajectory" in r else None,
                         "server_ms": 1000 * r.get("t_infer", 0), "timing": r.get("timing")})
            if a.period > 0:
                time.sleep(max(0.0, a.period - (time.time() - t_tick)))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{a.tag}.npy", np.array(rows, dtype=object), allow_pickle=True)
    print(json.dumps(summary(rows), indent=1))
    print(f"[bench] saved {OUT / (a.tag + '.npy')}")


def pct(x, q):
    return round(float(np.percentile(x, q)), 1) if len(x) else None


def summary(rows):
    s2 = [r for r in rows if r["is_s2"]]
    s1 = [r for r in rows if not r["is_s2"]]
    s2_ms = [r["timing"]["s2_ms"] for r in s2 if r.get("timing") and r["timing"].get("s2_ms")]
    s1_ms = [r["timing"]["s1_ms"] for r in rows if r.get("timing") and r["timing"].get("s1_ms")]
    walls = [r["wall"] for r in s2]
    dur = rows[-1]["wall"] + rows[-1]["client_ms"] / 1000 - rows[0]["wall"]
    return {
        "n": len(rows), "n_s2_replies": len(s2), "duration_s": round(dur, 1), "replies_per_s": round(len(rows) / dur, 2),
        "client_ms_all_p50/p95": [pct([r["client_ms"] for r in rows], 50), pct([r["client_ms"] for r in rows], 95)],
        "client_ms_s1_only_p50/p95": [pct([r["client_ms"] for r in s1], 50), pct([r["client_ms"] for r in s1], 95)],
        "client_ms_with_s2_p50/p95": [pct([r["client_ms"] for r in s2], 50), pct([r["client_ms"] for r in s2], 95)],
        "server_s2_ms_p50/p95": [pct(s2_ms, 50), pct(s2_ms, 95)],
        "server_s1_ms_p50/p95": [pct(s1_ms, 50), pct(s1_ms, 95)],
        "s2_period_s_p50": round(float(np.median(np.diff(walls))), 2) if len(walls) > 2 else None,
        "pixel_goal_age_frames_p50": pct([r["frame_id"] - r["pixel_frame_id"] for r in rows
                                          if r.get("pixel_frame_id") is not None], 50),
    }


def cmp(a):
    A = list(np.load(a.a, allow_pickle=True))
    B = list(np.load(a.b, allow_pickle=True))
    n = min(len(A), len(B))
    same_kind = same_action = same_text = same_pixel = 0
    tdiff, n_s2, n_traj = [], 0, 0
    first_div = None
    for i in range(n):
        x, y = A[i], B[i]
        assert x["frame_id"] == y["frame_id"]
        k = (x["action"] is None) == (y["action"] is None)
        same_kind += k
        same_action += x["action"] == y["action"]
        if x["is_s2"] or y["is_s2"]:
            n_s2 += 1
            same_text += x["llm"] == y["llm"]
            same_pixel += x["pixel"] == y["pixel"]
        if x["traj"] is not None and y["traj"] is not None:
            n_traj += 1
            tdiff.append(float(np.abs(x["traj"] - y["traj"]).max()))
        if first_div is None and (x["llm"] != y["llm"] or x["action"] != y["action"] or (tdiff and tdiff[-1] > 1e-6)):
            first_div = (i, x["ep"], x["step"])
    res = {"n": n, "same_reply_kind": same_kind / n, "same_action": same_action / n, "n_s2": n_s2,
           "same_s2_text": same_text / max(1, n_s2), "same_pixel": same_pixel / max(1, n_s2), "n_traj": n_traj,
           "traj_maxabs_max_m": max(tdiff) if tdiff else None,
           "traj_bit_identical": float(np.mean(np.array(tdiff) == 0.0)) if tdiff else None,
           "first_divergence": first_div}
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:5801")
    r.add_argument("--tag", required=True)
    r.add_argument("--period", type=float, default=0.0)
    r.add_argument("--episodes", type=int, default=6)
    r.add_argument("--max_steps", type=int, default=40)
    r.add_argument("--scene_root", default=SCENE_ROOT)
    c = sub.add_parser("cmp")
    c.add_argument("a")
    c.add_argument("b")
    s = sub.add_parser("summary")
    s.add_argument("a")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "cmp":
        cmp(a)
    elif a.cmd == "summary":
        print(json.dumps(summary(list(np.load(a.a, allow_pickle=True))), indent=1))
    else:
        assert False, f"unreachable cmd={a.cmd!r}"


if __name__ == "__main__":
    main()
