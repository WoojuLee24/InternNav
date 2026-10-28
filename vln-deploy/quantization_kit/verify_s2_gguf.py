"""System-2 compression check: HF PyTorch vs llama.cpp GGUF (f16 / Q4_K_M) on the same replayed inputs.

Fixed S2 schedule on recorded VLN-CE episodes (6 scenes x 40 steps): S2 runs at every 5th step (agent
PLAN_STEP_GAP=4), other steps append history (step_no_infer). If S2 answers "↓" the recorded 30° look-down frame of
that step is fed at full resolution (agent look_down semantics). Each backend records per S2 call: final text,
looked_down, pixel [row, col], latent [4, 3584], latency. Phases (different images/toolchains):

  gguf  (isaacsim image, vln-deploy convenv + llama-s2-runner)  --tag <name> --text <gguf> --mmproj <gguf> --resize 448
  hf    (gd_vln habitat image, transformers 4.51)               --tag <name> --resize 384|448 [--quant bf16]
  cmp   (gd_vln habitat image)  all tags vs a reference tag: text / action / pixel agreement, latent cosine, S1 ADE
        (HF S1 bf16, same seed per case), GT pixel accuracy (VLN-CE goal.125cm_30deg, stored [x, y]), latency.

  quantization_kit/convenv_x86/bin/python quantization_kit/verify_s2_gguf.py gguf --tag q4_repro --text quantization_kit/out/gguf/InternVLA-S2-text-Q4_K_M-latent-embf16.gguf --mmproj quantization_kit/out/gguf/InternVLA-S2-mmproj-f16.gguf
  /usr/bin/python3 quantization_kit/verify_s2_gguf.py hf --tag hf_bf16_384 --resize 384
  /usr/bin/python3 quantization_kit/verify_s2_gguf.py cmp --ref hf_bf16_448
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

QK = Path(__file__).resolve().parent
ROOT = QK.parent                          # vln-deploy
INTERNNAV = ROOT.parent
OUT = QK / "verify_s2_gguf_out"
SCENES = INTERNNAV / "data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r"
GAP = 5


def episodes(n_eps=6, max_steps=40):
    eps = []
    for scene in sorted(SCENES.iterdir()):
        e = json.loads(open(scene / "meta/episodes.jsonl").readline())
        idx, n = e["episode_index"], min(e["length"], max_steps)
        v = scene / "videos/chunk-000"
        eps.append({"id": f"{scene.name}/{idx}", "instruction": e["tasks"][0].strip(), "parquet":
                    str(scene / f"data/chunk-000/episode_{idx:06d}.parquet"),
                    "fpv": [str(v / f"observation.images.rgb.125cm_0deg/episode_{idx:06d}_{i}.jpg") for i in range(n)],
                    "look": [str(v / f"observation.images.rgb.125cm_30deg/episode_{idx:06d}_{i}.jpg") for i in range(n)]})
        if len(eps) >= n_eps:
            break
    return eps


def load(p):
    from PIL import Image

    return np.asarray(Image.open(p).convert("RGB"))


def replay(policy, tag, meta):
    """policy: reset(), step_no_infer(rgb), s2(rgb, instr, look_down) -> (action|None, latent|None, pixel|None, text)."""
    rows, acc_ms = [], []
    for ep in episodes():
        policy.reset()
        for i, p in enumerate(ep["fpv"]):
            rgb = load(p)
            if i % GAP != 0:
                t0 = time.perf_counter()
                policy.step_no_infer(rgb)
                acc_ms.append((time.perf_counter() - t0) * 1000)   # includes the embd_cache prefetch, if on
                continue
            t0 = time.perf_counter()
            act, lat, pix, text = policy.s2(rgb, ep["instruction"], False)
            looked = False
            if act is not None and list(act) == [5]:
                act, lat, pix, text2 = policy.s2(load(ep["look"][i]), ep["instruction"], True)
                text, looked = text + "  >  " + text2, True
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"ep": ep["id"], "step": i, "text": text, "looked_down": looked,
                         "action": None if act is None else [int(a) for a in act],
                         "pixel": None if pix is None else [int(x) for x in pix],
                         "latent": None if lat is None else np.asarray(lat, np.float32).reshape(4, -1), "ms": ms})
            print(f"[{tag}] {ep['id']} step {i:2d} {ms:7.0f} ms  {text!r}", flush=True)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / f"{tag}.npy", np.array(rows, dtype=object), allow_pickle=True)
    meta = dict(meta, accumulate_ms_p50=float(np.median(acc_ms)) if acc_ms else None)
    (OUT / f"{tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"[{tag}] saved {len(rows)} S2 calls -> {OUT / (tag + '.npy')}")


# ------------------------------------------------------------------ backends
class GGUFBackend:
    def __init__(self, a):
        sys.path[:0] = [str(INTERNNAV), str(ROOT)]
        b = QK / "llama.cpp" / ("build" if os.uname().machine == "aarch64" else "build_x86") / "bin"   # Thor / x86
        kw = dict(runner_bin=str(b / "llama-s2-runner"), model_gguf=a.text, mmproj_gguf=a.mmproj, lib_dir=str(b),
                  resize=a.resize, load_s1=False, embd_cache=a.embd_cache)
        if a.hf_parity:   # same image resolution as the HF model (384 -> 392, look-down 640x480 -> 644x476)
            from internnav.model.basemodel.internvla_n1.internvla_n1_policy_llamacpp_hfparity import (
                InternVLAN1LlamaCppPolicyHFParity,
            )

            self.p = InternVLAN1LlamaCppPolicyHFParity(**kw)
        else:
            from internnav.model.basemodel.internvla_n1.internvla_n1_policy_llamacpp import InternVLAN1LlamaCppPolicy

            self.p = InternVLAN1LlamaCppPolicy(lookdown_full_res=True, **kw)

    def reset(self):
        self.p.reset()

    def step_no_infer(self, rgb):
        self.p.step_no_infer(rgb, None, None)

    def s2(self, rgb, instr, look_down):
        o = self.p.s2_step(rgb, None, None, instr, None, look_down=look_down)
        lat = None if o.output_latent is None else o.output_latent.numpy()
        return o.output_action, lat, o.output_pixel, self.p.llm_output


class HFBackend:
    def __init__(self, a):
        sys.path[:0] = [str(INTERNNAV / "scripts/sim"), str(INTERNNAV)]
        import torch
        from argparse import Namespace

        from vln_agent import SimVLNAgent

        self.torch = torch
        self.ag = SimVLNAgent(Namespace(model_path=a.ckpt, device="cuda:0", quant=a.quant, resize_w=a.resize,
                                        resize_h=a.resize, num_history=8, plan_step_gap=4, log_dir=None,
                                        s1_depth_clip=None))

    def reset(self):
        self.ag.reset()

    def step_no_infer(self, rgb):
        self.ag.step_no_infer(rgb, None, None)

    def s2(self, rgb, instr, look_down):
        with self.torch.inference_mode():
            act, lat, pix = self.ag.step_s2(rgb, None, None, instr, None, look_down)
        self.torch.cuda.synchronize()
        return act, None if lat is None else lat.float().cpu().numpy(), pix, self.ag.llm_output


# ------------------------------------------------------------------ compare
def compare(a):
    sys.path[:0] = [str(INTERNNAV / "scripts/sim"), str(INTERNNAV), str(ROOT), str(INTERNNAV / "src/diffusion-policy")]
    import pandas as pd

    tags = sorted(p.stem for p in OUT.glob("*.npy"))
    runs = {t: list(np.load(OUT / f"{t}.npy", allow_pickle=True)) for t in tags}
    gt = {}
    for ep in episodes():
        df = pd.read_parquet(ep["parquet"], columns=["frame_index", "goal.125cm_30deg"])
        for _, r in df.iterrows():
            gt[(ep["id"], int(r.frame_index))] = [int(x) for x in r["goal.125cm_30deg"]]
    wp = None   # --no_s1 (Thor): no HF S1 (transformers 4.51) on device -> skip the latent->trajectory ADE column
    if not a.no_s1:
        import torch
        from PIL import Image

        import internnav.model.basemodel.internvla_n1.internvla_n1 as n1
        from deploy.s1_module.service_hf import load_s1_only
        from internnav.model.utils.vln_utils import traj_to_actions

        s1 = load_s1_only(a.ckpt, torch.device("cuda:0"))
        noise = np.random.default_rng(0).standard_normal((32, 32, 3)).astype(np.float32)
        n1.randn_tensor = lambda shape, generator=None, device=None, dtype=None, **k: torch.from_numpy(noise).reshape(shape).to(device, dtype)
        frames = {(ep["id"], i): (ep["fpv"][i], ep["look"][i]) for ep in episodes() for i in range(len(ep["fpv"]))}
        pr = lambda x: np.array(Image.fromarray(x).resize((224, 224))) / 255  # noqa: E731

        def s1_wp(r):
            f, l = frames[(r["ep"], r["step"])]
            goal = load(l) if r["looked_down"] else load(f)
            rgbs = torch.stack([torch.from_numpy(pr(goal)), torch.from_numpy(pr(load(f)))]).unsqueeze(0).cuda()
            lat = torch.from_numpy(r["latent"]).reshape(1, 4, -1).cuda().to(torch.bfloat16)
            with torch.inference_mode():
                dp = s1.generate_traj(lat, rgbs, torch.zeros(1, 2, 224, 224, 1, dtype=torch.float64).cuda())
            return np.asarray(traj_to_actions(dp.float(), use_discrate_action=False))

        wp = {t: {i: s1_wp(r) for i, r in enumerate(rs) if r["latent"] is not None} for t, rs in runs.items()}
    ref = runs[a.ref]
    final = lambda r: r["text"].split("  >  ")[-1]  # noqa: E731
    report = {}
    for t, rs in runs.items():
        n = min(len(rs), len(ref))
        pairs = [(x, y) for x, y in zip(rs[:n], ref[:n]) if (x["ep"], x["step"]) == (y["ep"], y["step"])]
        both_px = [(x, y) for x, y in pairs if x["pixel"] is not None and y["pixel"] is not None]
        both_lat = [(i, x, y) for i, (x, y) in enumerate(pairs) if x["latent"] is not None and y["latent"] is not None]
        cos = [float((x["latent"] * y["latent"]).sum() / (np.linalg.norm(x["latent"]) * np.linalg.norm(y["latent"])))
               for _, x, y in both_lat]
        ade = [] if wp is None else [float(np.linalg.norm(wp[t][i] - wp[a.ref][i], axis=1).mean()) for i, _, _ in both_lat]
        gtp = [(r, gt.get((r["ep"], r["step"]), [-1, -1])) for r in rs]
        gtv = [(r, g) for r, g in gtp if g[0] != -1]
        l2 = [np.hypot(r["pixel"][0] - g[1], r["pixel"][1] - g[0]) for r, g in gtv if r["pixel"] is not None]
        report[t] = {
            "n_s2": len(rs), "ms_p50": float(np.median([r["ms"] for r in rs])), "ms_p95": float(np.percentile([r["ms"] for r in rs], 95)),
            "vs_ref_final_text": float(np.mean([final(x) == final(y) for x, y in pairs])),
            "vs_ref_output_type+action": float(np.mean([x["action"] == y["action"] and (x["pixel"] is None) == (y["pixel"] is None) for x, y in pairs])),
            "vs_ref_lookdown_same": float(np.mean([x["looked_down"] == y["looked_down"] for x, y in pairs])),
            "vs_ref_pixel_l2_median": float(np.median([np.hypot(x["pixel"][0] - y["pixel"][0], x["pixel"][1] - y["pixel"][1]) for x, y in both_px])) if both_px else None,
            "vs_ref_pixel_exact": float(np.mean([x["pixel"] == y["pixel"] for x, y in both_px])) if both_px else None,
            "vs_ref_latent_cos_median": float(np.median(cos)) if cos else None,
            "vs_ref_latent_cos_min": float(np.min(cos)) if cos else None,
            "vs_ref_s1_ade_median_m": float(np.median(ade)) if ade else None,
            "gt_pixel_rate": len(l2) / max(1, len(gtv)), "gt_pixel_l2_median": float(np.median(l2)) if l2 else None,
            "gt_hit@30px": float(np.mean(np.array(l2) <= 30)) if l2 else None,
            "meta": json.loads((OUT / f"{t}.json").read_text()),
        }
    (OUT / f"report_vs_{a.ref}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "meta"} for k, v in report.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gguf", "hf", "cmp"])
    ap.add_argument("--tag")
    ap.add_argument("--text")
    ap.add_argument("--mmproj")
    ap.add_argument("--resize", type=int, default=448)
    ap.add_argument("--quant", default="bf16")
    ap.add_argument("--embd_cache", type=int, default=0, help="[gguf] vision-embedding cache entries (0 = off)")
    ap.add_argument("--hf_parity", action="store_true", help="[gguf] HF image resolution (needs --resize 384)")
    ap.add_argument("--ref", default="hf_bf16_448")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/InternVLA-N1-DualVLN"))
    ap.add_argument("--no_s1", action="store_true", help="[cmp] skip the HF S1 trajectory ADE (device without HF stack)")
    a = ap.parse_args()
    if a.phase == "gguf":
        replay(GGUFBackend(a), a.tag, {"backend": "gguf", "text": a.text, "mmproj": a.mmproj, "resize": a.resize,
                                        "hf_parity": a.hf_parity, "embd_cache": a.embd_cache})
    elif a.phase == "hf":
        replay(HFBackend(a), a.tag, {"backend": "hf", "quant": a.quant, "resize": a.resize})
    elif a.phase == "cmp":
        compare(a)
    else:
        assert False, a.phase


if __name__ == "__main__":
    main()
