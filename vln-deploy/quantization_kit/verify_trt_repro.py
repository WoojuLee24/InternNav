"""Reproduce the S1 TensorRT FP16 engines from the PyTorch checkpoint and verify them against it.

Plan (diffusion loop is not a static graph -> only the per-step denoise net goes to TRT):
  deploy/tools/export_s1_onnx.py  checkpoint -> quantization_kit/out/onnx/*.onnx
  this script `build`             out/onnx -> quantization_kit/out/trt_engines/*.engine (trt_s1._build_engine, FP16)

Two phases because the original HF model needs transformers 4.51 (gd_vln habitat image) while the TRT
toolchain lives in vln-deploy's convenv (isaacsim image, transformers 5.x / TensorRT 10.16):
  ref   (habitat image)  real S1 inputs (VLN-CE frames + bf16 S2 latents) + fixed initial noise -> original
                         checkpoint S1 (internvla_n1.generate_traj) in bf16 and fp32, 10 and 5 steps -> out/ref.npz
  trt   (isaacsim image) build repro engines, then compare: ONNX/weight provenance, per-step denoise output
                         (PyTorch fp32 vs original engine vs repro engine), full S1 (PyTorch fp32 lite / original
                         TRT / repro TRT / repro TRT + rgb/cond engines) vs the ref, and latency -> out/report.json

  /usr/bin/python3 quantization_kit/verify_trt_repro.py ref                                  # habitat image
  quantization_kit/convenv_x86/bin/python quantization_kit/verify_trt_repro.py trt           # isaacsim image
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

QK = Path(__file__).resolve().parent
ROOT = QK.parent                      # vln-deploy
INTERNNAV = ROOT.parent
OUT = QK / "verify_trt_repro_out"
SCENES = INTERNNAV / "data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r"
Q_OUT = INTERNNAV / "scripts/sim/verify/out/quant/InternVLA-N1-DualVLN_lookdown30/bf16_outputs.npy"
STEPS = (10, 5)


def frame(ep, step):
    scene, idx = ep.split("/")
    p = SCENES / scene / f"videos/chunk-000/observation.images.rgb.125cm_0deg/episode_{int(idx):06d}_{step}.jpg"
    from PIL import Image

    return np.asarray(Image.open(p).convert("RGB"))


def build_cases():
    """(latent, goal_rgb, cur_rgb) from the bf16 S2 replay (verify_quant.py): each S2 pixel-goal step, current
    frame = the same step and +3 steps (S1 keeps running on newer frames with the same latent)."""
    rows = list(np.load(Q_OUT, allow_pickle=True))
    by = {(r["ep"], r["step"]) for r in rows}
    cases = []
    for r in rows:
        if r["s2"] and r["latent"] is not None:
            for k in (0, 3):
                if (r["ep"], r["step"] + k) in by:
                    cases.append({"ep": r["ep"], "step": r["step"], "cur": r["step"] + k,
                                  "latent": r["latent"].reshape(1, 4, 3584).astype(np.float32)})
    return cases


class FixedNoise:
    """Replaces diffusers randn_tensor so every backend starts from the identical initial noise."""

    def __init__(self):
        self.cur = None

    def __call__(self, shape, generator=None, device=None, dtype=None, **kw):
        import torch

        return torch.from_numpy(self.cur).reshape(shape).to(device=device, dtype=dtype)


def actions_and_wp(dp):
    from internnav.model.utils.vln_utils import traj_to_actions

    wp = np.asarray(traj_to_actions(dp.clone(), use_discrate_action=False), np.float32)
    act = [int(a) for a in traj_to_actions(dp.clone()) if a != 0][:4]
    return wp, act


def timed_gpu(fn):
    import torch

    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000


# ===================================================================================== ref
def phase_ref(a):
    sys.path[:0] = [str(INTERNNAV), str(INTERNNAV / "scripts/sim"), str(ROOT), str(INTERNNAV / "src/diffusion-policy")]
    import torch

    import internnav.model.basemodel.internvla_n1.internvla_n1 as n1
    from deploy.s1_module.service_hf import load_s1_only

    noise = FixedNoise()
    n1.randn_tensor = noise
    cases = build_cases()[: a.max_cases]
    rng = np.random.default_rng(0)
    noises = rng.standard_normal((len(cases), 32, 32, 3)).astype(np.float32)
    model = load_s1_only(a.ckpt, torch.device("cuda:0"))
    from deploy.s1_module.service_hf import s1_infer  # noqa: F401  (verbatim agent preprocessing, reused below)
    from PIL import Image

    def prep(goal, cur):  # verbatim InternVLAN1AsyncAgent.step S1 preprocessing (depth ignored by DualVLN)
        pr = lambda x: np.array(Image.fromarray(x).resize((224, 224))) / 255  # noqa: E731
        rgbs = torch.stack([torch.from_numpy(pr(goal)), torch.from_numpy(pr(cur))]).unsqueeze(0).cuda()
        depths = torch.zeros(1, 2, 224, 224, 1, dtype=torch.float64).cuda()
        return rgbs, depths

    res = {"goal": [], "cur": []}
    for dtype_name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        model.to(dt)
        for steps in STEPS:
            key = f"ref_{dtype_name}_{steps}"
            wps, acts, dps, ms = [], [], [], []
            for i, c in enumerate(cases):
                goal, cur = frame(c["ep"], c["step"]), frame(c["ep"], c["cur"])
                if dtype_name == "bf16" and steps == STEPS[0]:
                    res["goal"].append(goal)
                    res["cur"].append(cur)
                rgbs, depths = prep(goal, cur)
                lat = torch.from_numpy(c["latent"]).cuda().to(dt)
                noise.cur = noises[i]
                with torch.inference_mode():
                    dp, t = timed_gpu(lambda: model.generate_traj(lat, rgbs, depths, num_inference_steps=steps))
                wp, act = actions_and_wp(dp.float())
                wps.append(wp), acts.append(act), dps.append(dp.float().cpu().numpy()), ms.append(t)
            res[key + "_wp"], res[key + "_dp"] = np.stack(wps), np.stack(dps)
            res[key + "_act"] = np.array([json.dumps(x) for x in acts])
            res[key + "_ms"] = np.array(ms)
            print(f"[ref] {key}: {len(cases)} cases, generate_traj p50 {np.median(ms[3:]):.1f} ms", flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "ref.npz", latent=np.stack([c["latent"] for c in cases]), noise=noises,
             goal=np.stack(res.pop("goal")), cur=np.stack(res.pop("cur")), **res)
    print(f"[ref] saved {OUT / 'ref.npz'}")


# ===================================================================================== trt
def init_hashes(path):
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(path))
    hs = {}
    for t in m.graph.initializer:
        arr = numpy_helper.to_array(t)
        if arr.size >= 16:  # skip scalar/shape constants
            hs[hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()] = arr.shape
    return m, hs


def phase_trt(a):
    sys.path[:0] = [str(ROOT), str(QK)]
    import torch

    import quantization_kit.s1_inference as s1i
    import quantization_kit.trt_s1 as trt_s1
    from quantization_kit.s1_inference import S1LiteModel, prep_images

    report = {}
    # ---------------- build repro engines (FP16, static shapes as the originals)
    eng_dir = QK / "out/trt_engines"
    eng_dir.mkdir(exist_ok=True)
    sm = trt_s1._sm_tag()
    plan = [("s1_denoise", "s1_denoise_step.onnx", trt_s1.DENOISE_SHAPES),
            ("rgb_dinov2", "rgb_model_dinov2.onnx", trt_s1.RGB_SHAPES),
            ("rgb_dinov2_b1", "rgb_model_dinov2.onnx", trt_s1.RGB_SHAPES_B1),
            ("memenc", "memory_encoder.onnx", trt_s1.MEMENC_SHAPES),
            ("resampler", "rgb_resampler.onnx", trt_s1.RESAMPLER_SHAPES)]
    report["build"] = {}
    for tag, onnx_name, shapes in plan:
        ep = eng_dir / f"{tag}_native_fp16_{sm}.engine"
        t0 = time.time()
        if not ep.exists():
            ep.write_bytes(trt_s1._build_engine(str(QK / "out/onnx" / onnx_name), shapes))
        orig = QK / "trt_engines" / f"{tag}_native_fp16_{sm}.engine"
        report["build"][tag] = {"build_s": round(time.time() - t0, 1), "repro_MB": round(ep.stat().st_size / 2**20, 1),
                                "orig_MB": round(orig.stat().st_size / 2**20, 1) if orig.exists() else None}
        print(f"[trt] engine {tag}: {report['build'][tag]}", flush=True)

    # ---------------- ONNX / weight provenance
    report["provenance"] = {}
    for name in ("s1_denoise_step", "rgb_model_dinov2", "memory_encoder", "rgb_resampler", "cond_projector"):
        mo, ho = init_hashes(QK / "onnx" / f"{name}.onnx")
        mr, hr = init_hashes(QK / "out/onnx" / f"{name}.onnx")
        report["provenance"][name] = {
            "orig_weight_tensors": len(ho), "repro_weight_tensors": len(hr),
            "bit_identical_tensors": len(set(ho) & set(hr)),
            "orig_nodes": len(mo.graph.node), "repro_nodes": len(mr.graph.node),
            "io_orig": [i.name for i in mo.graph.input] + [o.name for o in mo.graph.output],
            "io_repro": [i.name for i in mr.graph.input] + [o.name for o in mr.graph.output]}
        print(f"[trt] provenance {name}: {report['provenance'][name]}", flush=True)

    # ---------------- per-step denoise network: PyTorch fp32 vs original engine vs repro engine
    dev = "cuda:0"
    ref = np.load(OUT / "ref.npz", allow_pickle=True)
    m = S1LiteModel(a.ckpt).load_s1_weights(a.ckpt).to(dev, torch.float32).eval()
    sys.path.insert(0, str(ROOT / "deploy/tools"))
    from export_s1_onnx import DenoiseStep

    step_pt = DenoiseStep(m).eval()
    e_orig = trt_s1._engine_for(f"s1_denoise_native_fp16_{sm}", "s1_denoise_step.onnx", trt_s1.DENOISE_SHAPES,
                                str(QK / "onnx"), str(QK / "trt_engines"))
    e_rep = trt_s1._engine_for(f"s1_denoise_native_fp16_{sm}", "s1_denoise_step.onnx", trt_s1.DENOISE_SHAPES,
                               str(QK / "out/onnx"), str(eng_dir))
    errs = {"orig_vs_pt": [], "repro_vs_pt": [], "repro_vs_orig": []}
    scale = []
    with torch.no_grad():
        for i in range(min(20, len(ref["latent"]))):  # realistic conditioning: hidden states of real cases
            lat = torch.from_numpy(ref["latent"][i]).to(dev)
            rgbs, depths = prep_images(ref["goal"][i], np.zeros((480, 640), np.float32), ref["cur"][i],
                                       np.zeros((480, 640), np.float32), dev, torch.float32)
            tl = m.cond_projector(lat)
            imgs = (rgbs.permute(0, 1, 4, 2, 3) - m._resnet_mean) / m._resnet_std
            feat = m._encode_rgb(imgs.flatten(0, 1), 2)
            mem = m.memory_encoder(feat.flatten(1, 2))
            hid = torch.cat([m.rgb_resampler(torch.cat([feat.flatten(1, 2), mem], -1)), tl], 1)
            z = torch.cat([torch.zeros_like(hid), hid], 0).repeat_interleave(32, 0)
            x = torch.from_numpy(ref["noise"][i]).to(dev)
            for t in (999, 500, 100):
                ts = torch.full((64,), t, dtype=torch.long, device=dev)
                pos = torch.arange(32, device=dev).reshape(1, 32)
                y_pt = step_pt(x, ts, z, pos)
                y_o = e_orig({"latents": x, "timestep": ts, "z_latents": z, "pos_ids": pos}).float()
                y_r = e_rep({"latents": x, "timestep": ts, "z_latents": z, "pos_ids": pos}).float()
                errs["orig_vs_pt"].append((y_o - y_pt).abs().max().item())
                errs["repro_vs_pt"].append((y_r - y_pt).abs().max().item())
                errs["repro_vs_orig"].append((y_r - y_o).abs().max().item())
                scale.append(y_pt.abs().max().item())
    report["step"] = {k: {"maxabs_max": float(np.max(v)), "maxabs_median": float(np.median(v))} for k, v in errs.items()}
    report["step"]["output_absmax_median"] = float(np.median(scale))
    print(f"[trt] per-step denoise: {report['step']}", flush=True)

    # ---------------- full S1 (same inputs + identical initial noise as the ref phase)
    noise = FixedNoise()
    s1i.randn_tensor = noise
    trt_s1.randn_tensor = noise

    def make(kind):
        mm = S1LiteModel(a.ckpt).load_s1_weights(a.ckpt).to(dev, torch.float32).eval()
        if kind == "pt_fp32":
            return mm
        if kind == "trt_orig":
            return trt_s1.attach_trt_denoise(mm, trt_rgb=False, trt_cond=False)
        if kind == "trt_repro":
            return trt_s1.attach_trt_denoise(mm, onnx_dir=str(QK / "out/onnx"), engine_dir=str(eng_dir),
                                             trt_rgb=False, trt_cond=False)
        if kind == "trt_repro_full":   # deploy configuration: denoise + DINOv2 + memory encoder + resampler on TRT
            return trt_s1.attach_trt_denoise(mm, onnx_dir=str(QK / "out/onnx"), engine_dir=str(eng_dir),
                                             trt_rgb=True, trt_cond=True)
        if kind == "trt_orig_full":
            return trt_s1.attach_trt_denoise(mm, trt_rgb=True, trt_cond=True)
        assert False, kind

    zeros = np.zeros((480, 640), np.float32)
    outs = {}
    for kind in ("pt_fp32", "trt_orig", "trt_repro", "trt_orig_full", "trt_repro_full"):
        mm = make(kind)
        for steps in STEPS:
            wps, acts, ms = [], [], []
            for i in range(len(ref["latent"])):
                lat = torch.from_numpy(ref["latent"][i]).to(dev)
                rgbs, depths = prep_images(ref["goal"][i], zeros, ref["cur"][i], zeros, dev, torch.float32)
                noise.cur = ref["noise"][i]
                with torch.no_grad():
                    dp, t = timed_gpu(lambda: mm.generate_traj(traj_latents=lat, images_dp=rgbs, depths_dp=depths,
                                                               num_inference_steps=steps))
                wp, act = actions_and_wp(dp.float())
                wps.append(wp), acts.append(act), ms.append(t)
            outs[f"{kind}_{steps}"] = (np.stack(wps), acts, np.array(ms))
            print(f"[trt] {kind}@{steps}: generate_traj p50 {np.median(ms[3:]):.1f} ms", flush=True)
        del mm
        torch.cuda.empty_cache()
    for steps in STEPS:
        for rk in ("bf16", "fp32"):
            outs[f"ref_{rk}_{steps}"] = (ref[f"ref_{rk}_{steps}_wp"], [json.loads(x) for x in ref[f"ref_{rk}_{steps}_act"]],
                                         ref[f"ref_{rk}_{steps}_ms"])

    def cmp(x, y):
        wx, ax, _ = outs[x]
        wy, ay, _ = outs[y]
        d = np.linalg.norm(wx - wy, axis=-1)  # [cases, T+1]
        return {"ade_median_m": float(np.median(d.mean(1))), "ade_p90_m": float(np.percentile(d.mean(1), 90)),
                "fde_median_m": float(np.median(d[:, -1])), "max_m": float(d.max()),
                "first_action_agree": float(np.mean([(p[:1] == q[:1]) for p, q in zip(ax, ay)])),
                "actions_agree": float(np.mean([p == q for p, q in zip(ax, ay)]))}

    report["full_s1"] = {"n_cases": int(len(ref["latent"]))}
    for steps in STEPS:
        for k in ("ref_fp32", "pt_fp32", "trt_orig", "trt_repro", "trt_orig_full", "trt_repro_full"):
            report["full_s1"][f"{k}@{steps} vs ref_bf16@10"] = cmp(f"{k}_{steps}", "ref_bf16_10")
        report["full_s1"][f"ref_bf16@{steps} vs ref_bf16@10"] = cmp(f"ref_bf16_{steps}", "ref_bf16_10")
        for k in ("trt_orig", "trt_repro", "trt_repro_full"):
            report["full_s1"][f"{k}@{steps} vs pt_fp32@{steps}"] = cmp(f"{k}_{steps}", f"pt_fp32_{steps}")
        report["full_s1"][f"trt_repro@{steps} vs trt_orig@{steps}"] = cmp(f"trt_repro_{steps}", f"trt_orig_{steps}")
    report["latency_ms_p50"] = {k: float(np.median(v[2][3:])) for k, v in outs.items()}
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["ref", "trt"])
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/InternVLA-N1-DualVLN"))
    ap.add_argument("--max_cases", type=int, default=80)
    a = ap.parse_args()
    if a.phase == "ref":
        phase_ref(a)
    elif a.phase == "trt":
        phase_trt(a)
    else:
        assert False, a.phase


if __name__ == "__main__":
    main()
