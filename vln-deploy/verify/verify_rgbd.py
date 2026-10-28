"""RGB-D frame folder -> S2 / S1 / S2+S1 on this device (5090 or Thor): latency, outputs, sanity checks, and a
cross-device comparison of two runs on the same frames (e.g. 5090 vs Thor).

frames folder:  rgb/<NNNN>.jpg|png        front camera, required
                depth/<NNNN>.png          uint16 millimetres (RealSense / dataset convention), optional -> zeros
                lookdown/<NNNN>.jpg|png   30° down camera, fed when S2 answers "↓", optional
                instruction.txt           navigation instruction (or --instruction)

  export  dataset episode -> frames folder
          verify_rgbd.py export --scene 17DRP5sb8fy [--episode 0] [--steps 40] --out DIR
  run     S2 every --gap frames (history in between), S1 on every frame with the latest pixel-goal latent
          verify_rgbd.py run --frames DIR --mode dual|s2|s1 --out RUN [--latents OTHER_RUN] [--no-trt]
            s2   : S2 only
            s1   : S1 only; latents from --latents (another device's run -> isolates S1) or from an untimed S2 pre-pass
            dual : both (S1 runs right after each S2, so per-call latency is measured uncontended)
  cmp     compare two runs on the same frames (S2 decisions / text / pixel / latent cos, S1 first action / ADE)
          verify_rgbd.py cmp RUN_A RUN_B [--min-cmd 0.99]

Same S2 settings as the deploy stack (GGUF HF parity, resize 384, embedding cache 512) and the same fixed S1 initial
noise per frame as verify_s1s2_full.py, so two devices see identical inputs. DualVLN S1 ignores depth (nextdit path
uses RGB only); depth is still passed through prep_images so a depth-using S1 would get it.
Run from vln-deploy/ with the deploy venv: quantization_kit/convenv_{x86,aarch64}/bin/python verify/verify_rgbd.py ...
"""
import argparse
import json
import os
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]          # vln-deploy
QK = ROOT / "quantization_kit"
INTERNNAV = ROOT.parent
DATASET = INTERNNAV / "data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r"
sys.path[:0] = [str(QK), str(ROOT), str(INTERNNAV)]


def img(p):
    return np.asarray(Image.open(p).convert("RGB"))


def depth(p):   # uint16 mm -> metres / 10, the scale s1_inference.prep_images expects (it multiplies by 10)
    return np.asarray(Image.open(p), np.float32) / 10000.0


def frames(d):
    d = Path(d)
    rgb = sorted(p for p in (d / "rgb").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    out = []
    for p in rgb:
        dp = d / "depth" / f"{p.stem}.png"
        lk = [q for q in (d / "lookdown").glob(f"{p.stem}.*")] if (d / "lookdown").is_dir() else []
        out.append({"id": p.stem, "rgb": p, "depth": dp if dp.exists() else None, "look": lk[0] if lk else None})
    return out


# ------------------------------------------------------------------ export
def cmd_export(a):
    sc = DATASET / a.scene
    eps = [json.loads(line) for line in open(sc / "meta/episodes.jsonl")]
    e = next(x for x in eps if x["episode_index"] == a.episode)
    v, n = sc / "videos/chunk-000", min(e["length"], a.steps)
    out = Path(a.out)
    for sub in ("rgb", "depth", "lookdown"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        stem = f"episode_{a.episode:06d}_{i}"
        shutil.copy(v / f"observation.images.rgb.125cm_0deg/{stem}.jpg", out / f"rgb/{i:04d}.jpg")
        shutil.copy(v / f"observation.images.rgb.125cm_30deg/{stem}.jpg", out / f"lookdown/{i:04d}.jpg")
        d = np.asarray(Image.open(v / f"observation.images.depth.125cm_0deg/{stem}.png"))
        Image.fromarray(np.clip(d, 0, 65535).astype(np.uint16)).save(out / f"depth/{i:04d}.png")
    (out / "instruction.txt").write_text(e["tasks"][0].strip() + "\n")
    print(f"[export] {a.scene}/{a.episode}: {n} frames -> {out}")


# ------------------------------------------------------------------ run
def s2_backend(a):
    from verify_s2_gguf import GGUFBackend

    return GGUFBackend(Namespace(text=a.text, mmproj=a.mmproj, resize=384, embd_cache=a.embd_cache, hf_parity=True))


def s1_model(a):
    import torch

    import quantization_kit.s1_inference as s1i
    import quantization_kit.trt_s1 as trt_s1
    from quantization_kit.s1_inference import S1LiteModel, prep_images
    from verify_s1s2_full import FixedNoise, noise_for, timed, wp_act

    noise = FixedNoise()
    s1i.randn_tensor = noise
    trt_s1.randn_tensor = noise
    m = S1LiteModel(a.ckpt).load_s1_weights(a.ckpt).to("cuda:0", torch.float32).eval()
    if not a.no_trt:
        trt_s1.attach_trt_denoise(m, onnx_dir=a.onnx_dir, engine_dir=a.engine_dir, trt_cond=True)

    def run(lat, goal, goal_d, cur, cur_d, key):
        noise.cur = noise_for(key)
        with torch.no_grad():
            def go():
                rgbs, depths = prep_images(goal, goal_d, cur, cur_d, "cuda:0", torch.float32)
                return m.generate_traj(traj_latents=torch.from_numpy(lat).to("cuda:0"), images_dp=rgbs,
                                       depths_dp=depths, num_inference_steps=a.s1_steps)
            dp, t = timed(go)
        return (*wp_act(dp.float()), t)

    return run


def s2_pass(a, fr, instr):
    """S2 on every --gap frame (history frames in between), like the agent / verify_s2_gguf.py schedule."""
    b, rows = s2_backend(a), []
    b.reset()
    for i, f in enumerate(fr):
        rgb = img(f["rgb"])
        if i % a.gap:
            b.step_no_infer(rgb)
            continue
        t0 = time.perf_counter()
        act, lat, pix, text = b.s2(rgb, instr, False)
        looked = False
        if act is not None and list(act) == [5] and f["look"] is not None:
            act, lat, pix, text2 = b.s2(img(f["look"]), instr, True)
            text, looked = text + "  >  " + text2, True
        rows.append({"i": i, "id": f["id"], "text": text, "action": None if act is None else [int(x) for x in act],
                     "pixel": None if pix is None else [int(x) for x in pix], "looked_down": looked,
                     "latent": None if lat is None else np.asarray(lat, np.float32).reshape(-1),
                     "ms": (time.perf_counter() - t0) * 1000})
        print(f"[s2] {f['id']}  {rows[-1]['ms']:6.0f} ms  {text!r}", flush=True)
    return rows


def cmd_run(a):
    import torch

    fr = frames(a.frames)
    instr = a.instruction or (Path(a.frames) / "instruction.txt").read_text().strip()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "s1" and a.latents:
        rows = list(np.load(Path(a.latents) / "s2.npy", allow_pickle=True))
    else:
        rows = s2_pass(a, fr, instr)
    if a.mode != "s2":
        s1 = s1_model(a)
        by = {r["i"]: r for r in rows}
        res, lat = [], None
        for i, f in enumerate(fr):
            if i in by:   # a new S2 answer: pixel goal -> new latent + goal frame; discrete action -> S1 idles
                r = by[i]
                lat = None if r["latent"] is None else r["latent"].reshape(1, 4, -1)
                gsrc = f["look"] if r["looked_down"] else f["rgb"]
                goal, goal_d = img(gsrc), depth(f["depth"]) if f["depth"] else None
            if lat is None:
                continue
            cur = img(f["rgb"])
            cur_d = depth(f["depth"]) if f["depth"] else np.zeros(cur.shape[:2], np.float32)
            gd = goal_d if goal_d is not None else np.zeros(goal.shape[:2], np.float32)
            wp, act, t = s1(lat, goal, gd, cur, cur_d, f"rgbd|{f['id']}")
            res.append({"id": f["id"], "wp": wp, "act": act, "ms": t})
        np.savez(out / "s1.npz", id=np.array([r["id"] for r in res]), wp=np.array([r["wp"] for r in res], dtype=object),
                 act=np.array([json.dumps(r["act"]) for r in res]), ms=np.array([r["ms"] for r in res]))
        print(f"[s1] {len(res)} calls", flush=True)
    np.save(out / "s2.npy", np.array(rows, dtype=object), allow_pickle=True)

    # ---- summary + sanity (no ground truth on robot frames: check the outputs are well-formed)
    pct = lambda v: None if not len(v) else [round(float(np.percentile(v, q)), 1) for q in (50, 95)]  # noqa: E731
    s2ms = [r["ms"] for r in rows]
    checks = {"s2_latent_shape_4x3584": all(r["latent"] is None or r["latent"].size == 4 * 3584 for r in rows),
              "s2_latent_finite": all(r["latent"] is None or np.isfinite(r["latent"]).all() for r in rows)}
    summ = {"device": torch.cuda.get_device_name(0), "sm": "sm%d%d" % torch.cuda.get_device_capability(),
            "arch": os.uname().machine, "mode": a.mode, "frames": len(fr), "instruction": instr,
            "s2": {"calls": len(rows), "ms_p50_p95": pct(s2ms[1:] or s2ms),
                   "pixel_goal": sum(r["latent"] is not None for r in rows),
                   "discrete": sum(r["latent"] is None for r in rows), "from": a.latents or "this run"}}
    if a.mode != "s2":
        s1ms = [r["ms"] for r in res]
        summ["s1"] = {"calls": len(res), "ms_p50_p95": pct(s1ms[3:] or s1ms),
                      "hz_p50": round(1000 / np.median(s1ms[3:] or s1ms), 1) if res else None,
                      "trt": not a.no_trt, "steps": a.s1_steps}
        checks["s1_waypoints_finite"] = all(np.isfinite(r["wp"]).all() for r in res)
        checks["s1_waypoints_within_10m"] = all(np.abs(r["wp"]).max() < 10 for r in res)
        checks["s1_actions_valid"] = all(set(r["act"]) <= {1, 2, 3} for r in res)
    summ["checks"] = checks
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    sys.exit(0 if all(checks.values()) else 1)


# ------------------------------------------------------------------ cmp
def cmd_cmp(a):
    A, B = Path(a.run_a), Path(a.run_b)
    sa = {r["i"]: r for r in np.load(A / "s2.npy", allow_pickle=True)}
    sb = {r["i"]: r for r in np.load(B / "s2.npy", allow_pickle=True)}
    both = sorted(set(sa) & set(sb))
    final = lambda r: r["text"].split("  >  ")[-1]  # noqa: E731
    kind = lambda r: "traj" if r["latent"] is not None else json.dumps(r["action"])  # noqa: E731
    lat = [(sa[i]["latent"], sb[i]["latent"]) for i in both if sa[i]["latent"] is not None and sb[i]["latent"] is not None]
    cos = [float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in lat]
    rep = {"s2": {"calls": len(both), "decision_agree": float(np.mean([kind(sa[i]) == kind(sb[i]) for i in both])),
                  "text_agree": float(np.mean([final(sa[i]) == final(sb[i]) for i in both])),
                  "pixel_exact": float(np.mean([sa[i]["pixel"] == sb[i]["pixel"] for i in both])),
                  "latent_cos_median_min": [round(float(np.median(cos)), 5), round(float(np.min(cos)), 5)] if cos else None}}
    ok = True
    if (A / "s1.npz").exists() and (B / "s1.npz").exists():
        la, lb = np.load(A / "s1.npz", allow_pickle=True), np.load(B / "s1.npz", allow_pickle=True)
        ta = {k: (w, json.loads(c)) for k, w, c in zip(la["id"], la["wp"], la["act"])}
        tb = {k: (w, json.loads(c)) for k, w, c in zip(lb["id"], lb["wp"], lb["act"])}
        ks = sorted(set(ta) & set(tb))
        ade = [float(np.linalg.norm(np.asarray(ta[k][0], np.float32) - np.asarray(tb[k][0], np.float32), axis=-1).mean()) for k in ks]
        cmd = float(np.mean([ta[k][1][:1] == tb[k][1][:1] for k in ks])) if ks else None
        rep["s1"] = {"calls_both": len(ks), "only_a": len(set(ta) - set(tb)), "only_b": len(set(tb) - set(ta)),
                     "command_agree(first action)": cmd,
                     "traj_agree(ADE<5cm)": float(np.mean(np.array(ade) < 0.05)) if ade else None,
                     "ade_median_cm": round(float(np.median(ade)) * 100, 3) if ade else None,
                     "ade_max_cm": round(float(np.max(ade)) * 100, 3) if ade else None}
        ok = cmd is not None and cmd >= a.min_cmd and rep["s1"]["traj_agree(ADE<5cm)"] >= a.min_cmd
    for r, run in (("a", A), ("b", B)):
        s = run / "summary.json"
        rep[f"device_{r}"] = json.loads(s.read_text())["device"] if s.exists() else None
    rep["pass"] = bool(ok)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    (B / f"cmp_vs_{A.name}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export")
    e.add_argument("--scene", required=True)
    e.add_argument("--episode", type=int, default=0)
    e.add_argument("--steps", type=int, default=40)
    e.add_argument("--out", required=True)
    r = sp.add_parser("run")
    r.add_argument("--frames", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--mode", choices=["dual", "s2", "s1"], default="dual")
    r.add_argument("--instruction", default="")
    r.add_argument("--latents", default="", help="[s1] take S2 outputs from another run folder (cross-device S1 check)")
    r.add_argument("--gap", type=int, default=5, help="S2 every N frames (verify_s2_gguf.py schedule)")
    r.add_argument("--text", default=str(QK / "out/gguf/InternVLA-S2-text-Q8_0-latent-embf16.gguf"))
    r.add_argument("--mmproj", default=str(QK / "out/gguf/InternVLA-S2-mmproj-f16.gguf"))
    r.add_argument("--embd-cache", dest="embd_cache", type=int, default=512)
    r.add_argument("--ckpt", default=str(ROOT / "checkpoints/InternVLA-N1-DualVLN"))
    r.add_argument("--s1-steps", dest="s1_steps", type=int, default=10)
    r.add_argument("--no-trt", dest="no_trt", action="store_true", help="S1 PyTorch fp32 instead of TensorRT FP16")
    r.add_argument("--onnx-dir", dest="onnx_dir", default=str(QK / "out/onnx"))
    r.add_argument("--engine-dir", dest="engine_dir", default=str(QK / "out/trt_engines"))
    c = sp.add_parser("cmp")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--min-cmd", dest="min_cmd", type=float, default=0.99, help="pass threshold for S1 command / ADE<5cm")
    a = ap.parse_args()
    {"export": cmd_export, "run": cmd_run, "cmp": cmd_cmp}[a.cmd](a)


if __name__ == "__main__":
    main()
