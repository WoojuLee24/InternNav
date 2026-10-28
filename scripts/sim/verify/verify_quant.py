"""Quantization check per system (S2 / S1 / dual) on recorded VLN-CE episodes: GPU memory, latency, accuracy.

Each quant (bf16 / nf4 / int8) replays the same frames step by step through SimVLNAgent (S2 every
plan_step_gap, S1 every step, fixed torch seed per step so S1 sampling noise is identical). When S2
answers "↓" the 30° look-down frame of the same step is fed (VLN-CE protocol, = quantization_kit).

Measured per system
  S2 (Qwen2.5-VL, quantized part): weight GiB, peak activation GiB, latency per S2 call,
      accuracy vs bf16 (output type + action ids, exact text, pixel L2, latent cosine)
      accuracy vs ground truth (VLN-CE goal.125cm_30deg label, stored [x, y]): pixel-goal rate when a GT pixel exists,
      pixel L2 to GT, hit rate within --hit_px
  S1 (DiT / NavDP, always bf16; differs only through the latent): weight GiB, peak activation GiB,
      latency per S1 call, waypoint ADE / FDE vs bf16
  dual: total peak GiB, step latency

Usage (inside gd_vln container):
  python3 scripts/sim/verify/verify_quant.py --model_path /checkpoints/InternVLA-N1-DualVLN --quants bf16,nf4,int8 --episodes 6 --max_steps 40
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vln_agent import SimVLNAgent  # noqa: E402

SCENE_ROOT = "data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r"
S2_PREFIXES = ("visual", "model.layers", "model.embed_tokens", "model.norm", "model.latent_queries", "lm_head")


class ProfiledAgent(SimVLNAgent):
    """Times S2 / S1 calls and records their peak activation memory (weights excluded)."""

    def __init__(self, args):
        super().__init__(args)
        self.prof = {"s2": [], "s1": []}

    def _measure(self, key, fn, *a, **k):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        self.prof[key].append((time.time() - t0, (peak - base) / 2**30, peak / 2**30))
        return r

    def step_s2(self, *a, **k):
        return self._measure("s2", super().step_s2, *a, **k)

    def step_s1(self, *a, **k):
        return self._measure("s1", super().step_s1, *a, **k)


def weight_gib(model):
    s2 = s1 = 0
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        b = t.numel() * t.element_size()
        if name.startswith(S2_PREFIXES):
            s2 += b
        else:
            s1 += b
    return s2 / 2**30, s1 / 2**30


def load_episodes(root, n_episodes, max_steps):
    import pandas as pd

    eps = []
    for scene in sorted(Path(root).iterdir()):
        for line in open(scene / "meta/episodes.jsonl"):
            e = json.loads(line)
            idx = e["episode_index"]
            v = scene / "videos/chunk-000"
            n = min(e["length"], max_steps)
            frames = [(v / f"observation.images.rgb.125cm_0deg/episode_{idx:06d}_{i}.jpg",
                       v / f"observation.images.depth.125cm_0deg/episode_{idx:06d}_{i}.png",
                       v / f"observation.images.rgb.125cm_30deg/episode_{idx:06d}_{i}.jpg") for i in range(n)]
            if not all(all(p.exists() for p in f) for f in frames):
                continue
            df = pd.read_parquet(scene / f"data/chunk-000/episode_{idx:06d}.parquet",
                                 columns=["frame_index", "goal.125cm_30deg"])
            gt = {int(r.frame_index): [int(x) for x in r["goal.125cm_30deg"]] for _, r in df.iterrows()}
            eps.append({"id": f"{scene.name}/{idx}", "instruction": e["tasks"][0].strip(), "frames": frames,
                        "gt_pixel": [gt.get(i, [-1, -1]) for i in range(n)]})
            break  # one episode per scene for diversity
        if len(eps) >= n_episodes:
            break
    return eps


def run(args, quant, episodes):
    a = argparse.Namespace(**vars(args))
    a.quant = quant
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    agent = ProfiledAgent(a)
    load_s = time.time() - t
    w_s2, w_s1 = weight_gib(agent.model)
    mem_load = torch.cuda.memory_allocated() / 2**30
    out = []
    for ep in episodes:
        agent.reset()
        for i, (rp, dp, lp) in enumerate(ep["frames"]):
            rgb = np.asarray(Image.open(rp).convert("RGB"))
            d = np.asarray(Image.open(dp)).astype(np.float32)
            depth = d / 1000.0 if d.max() > 100 else d  # mm png or metres
            torch.manual_seed(1000 + i)
            torch.cuda.synchronize()
            t0 = time.time()
            o = agent.step(rgb, depth, np.eye(4), ep["instruction"], None)
            if o.output_action is not None and list(o.output_action) == [5]:
                look = np.asarray(Image.open(lp).convert("RGB")) if args.lookdown_source == "lookdown30" else rgb
                torch.manual_seed(1000 + i)
                o = agent.step(look, depth, np.eye(4), ep["instruction"], None, look_down=True)
                o.is_s2_step = True
            torch.cuda.synchronize()
            dt = time.time() - t0
            lat = agent.output_latent
            out.append({
                "ep": ep["id"], "step": i, "s2": bool(o.is_s2_step), "dt": dt, "gt_pixel": ep["gt_pixel"][i],
                "action": None if o.output_action is None else [int(x) for x in o.output_action],
                "pixel": None if o.output_pixel is None else [int(x) for x in o.output_pixel],
                "traj": None if o.output_trajectory is None else np.asarray(o.output_trajectory)[:, :2],
                "latent": lat.float().cpu().numpy().reshape(-1) if (o.is_s2_step and lat is not None) else None,
                "llm": o.llm_output,
            })
    prof = {k: np.array(v) for k, v in agent.prof.items()}
    del agent
    gc.collect()
    torch.cuda.empty_cache()
    p = lambda x, q: float(np.percentile(x, q)) if len(x) else float("nan")  # noqa: E731
    res = {
        "load_s": load_s,
        "dual_weights_GiB": mem_load,
        "dual_peak_GiB": float(max(v[:, 2].max() for v in prof.values() if len(v))),
        "dual_step_p50_ms": 1000 * p([r["dt"] for r in out], 50),
        "S2_weights_GiB": w_s2,
        "S2_act_peak_GiB": float(prof["s2"][:, 1].max()) if len(prof["s2"]) else float("nan"),
        "S2_calls": int(len(prof["s2"])),
        "S2_latency_p50_ms": 1000 * p(prof["s2"][:, 0], 50) if len(prof["s2"]) else float("nan"),
        "S2_latency_p95_ms": 1000 * p(prof["s2"][:, 0], 95) if len(prof["s2"]) else float("nan"),
        "S1_weights_GiB": w_s1,
        "S1_act_peak_GiB": float(prof["s1"][:, 1].max()) if len(prof["s1"]) else float("nan"),
        "S1_calls": int(len(prof["s1"])),
        "S1_latency_p50_ms": 1000 * p(prof["s1"][:, 0], 50) if len(prof["s1"]) else float("nan"),
        "S1_latency_p95_ms": 1000 * p(prof["s1"][:, 0], 95) if len(prof["s1"]) else float("nan"),
    }
    res.update(gt_accuracy(out, args.hit_px))
    return out, res


def gt_accuracy(out, hit_px):
    """S2 vs VLN-CE ground truth goal.125cm_30deg in the 640x480 look-down frame.

    The label is stored [x, y] = [col, row] (first value reaches 500 > 480 rows; it is also the order the LLM
    prints, "x y") — prepare_dataset.py's "[y,x]" comment is wrong. Agent pixels are [row, col]
    (internvla_n1_agent_realworld.py swaps the two printed numbers), so compare against [gt[1], gt[0]].
    """
    s2_gt = [r for r in out if r["s2"] and r["gt_pixel"][0] != -1]
    with_px = [r for r in s2_gt if r["pixel"] is not None]
    l2 = [float(np.hypot(r["pixel"][0] - r["gt_pixel"][1], r["pixel"][1] - r["gt_pixel"][0])) for r in with_px]
    return {
        "S2_gt_n": len(s2_gt),
        "S2_gt_pixel_rate": len(with_px) / len(s2_gt) if s2_gt else float("nan"),
        "S2_gt_pixel_l2_median": float(np.median(l2)) if l2 else float("nan"),
        f"S2_gt_hit@{hit_px}px": float(np.mean(np.array(l2) <= hit_px)) if l2 else float("nan"),
        f"S2_gt_hit@{hit_px}px_over_all": float(np.sum(np.array(l2) <= hit_px) / len(s2_gt)) if s2_gt else float("nan"),
    }


def compare(ref, pred):
    pl2, cos, ade, fde = [], [], [], []
    for r, p in zip(ref, pred):
        assert (r["ep"], r["step"]) == (p["ep"], p["step"])
        if r["pixel"] is not None and p["pixel"] is not None and r["s2"] and p["s2"]:
            pl2.append(float(np.hypot(r["pixel"][0] - p["pixel"][0], r["pixel"][1] - p["pixel"][1])))
        if r["latent"] is not None and p["latent"] is not None:
            a, b = r["latent"], p["latent"]
            cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        if r["traj"] is not None and p["traj"] is not None:
            d = np.linalg.norm(r["traj"] - p["traj"], axis=1)
            ade.append(float(d.mean()))
            fde.append(float(d[-1]))
    s2 = [(r, p) for r, p in zip(ref, pred) if r["s2"]]
    s2_agree = [r["action"] == p["action"] and (r["pixel"] is None) == (p["pixel"] is None) for r, p in s2]
    med = lambda x: float(np.median(x)) if x else float("nan")  # noqa: E731
    return {
        "S2_vs_bf16_n": len(s2),
        "S2_vs_bf16_action_agree": float(np.mean(s2_agree)) if s2 else float("nan"),
        "S2_vs_bf16_exact_text": float(np.mean([r["llm"] == p["llm"] for r, p in s2])) if s2 else float("nan"),
        "S2_vs_bf16_pixel_l2_median": med(pl2),
        "S2_vs_bf16_pixel_l2_p90": float(np.percentile(pl2, 90)) if pl2 else float("nan"),
        "S2_vs_bf16_latent_cos_median": med(cos),
        "S2_vs_bf16_latent_cos_min": float(min(cos)) if cos else float("nan"),
        "S1_vs_bf16_n": len(ade),
        "S1_vs_bf16_ade_median_m": med(ade),
        "S1_vs_bf16_fde_median_m": med(fde),
    }


def recompute(args, quants):
    outdir = Path(args.out) / f"{Path(args.model_path).name}_{args.lookdown_source}"
    report = json.loads((outdir / "report.json").read_text())
    res = {q: list(np.load(outdir / f"{q}_outputs.npy", allow_pickle=True)) for q in quants}
    for q in quants:
        report[q].update(gt_accuracy(res[q], args.hit_px))
        if q != "bf16":
            report[q].update(compare(res["bf16"], res[q]))
    (outdir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--quants", default="bf16,nf4,int8")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--scene_root", default=SCENE_ROOT)
    ap.add_argument("--lookdown_source", default="lookdown30", choices=["lookdown30", "same"],
                    help="frame fed after S2 says ↓: the recorded 30° frame (VLN-CE protocol) or the same FPV frame")
    ap.add_argument("--hit_px", type=float, default=30.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resize_w", type=int, default=384)
    ap.add_argument("--resize_h", type=int, default=384)
    ap.add_argument("--num_history", type=int, default=8)
    ap.add_argument("--plan_step_gap", type=int, default=4)
    ap.add_argument("--s1_depth_clip", type=float, default=5.0)
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--out", default="scripts/sim/verify/out/quant")
    ap.add_argument("--recompute", action="store_true", help="recompute metrics from saved *_outputs.npy (no GPU)")
    args = ap.parse_args()
    quants = args.quants.split(",")
    if args.recompute:
        return recompute(args, quants)
    assert quants[0] == "bf16", "bf16 must come first (reference)"

    episodes = load_episodes(args.scene_root, args.episodes, args.max_steps)
    print(f"[verify_quant] {len(episodes)} episodes, {sum(len(e['frames']) for e in episodes)} steps", flush=True)
    outdir = Path(args.out) / f"{Path(args.model_path).name}_{args.lookdown_source}"
    outdir.mkdir(parents=True, exist_ok=True)

    results, report = {}, {}
    for q in quants:
        print(f"[verify_quant] === {q} ===", flush=True)
        try:
            results[q], res = run(args, q, episodes)
        except Exception as e:  # e.g. bnb kernels missing for sm_120
            import traceback

            traceback.print_exc()
            report[q] = {"error": repr(e)}
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if q != "bf16" and "bf16" in results:
            res.update(compare(results["bf16"], results[q]))
        report[q] = res
        print(json.dumps({q: res}, indent=1), flush=True)
        np.save(outdir / f"{q}_outputs.npy", np.array(results[q], dtype=object), allow_pickle=True)
    (outdir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"[verify_quant] report -> {outdir / 'report.json'}")


if __name__ == "__main__":
    main()
