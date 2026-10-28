"""Fidelity + speed of the S1 optimizations #3 (TRT memory_encoder+QFormer) and #4 (GPU prep),
measured against the fp32 PyTorch reference (PIL prep, eager conditioning) on a REAL S2 latent
and a realistic frame sequence (fixed goal, panning current). The denoise RNG is seeded identically
each call, so the only variable is the optimization under test.

For each variant: how many of N steps produce the SAME discrete actions as the fp32 reference,
plus the full run_s1 latency (prep + generate_traj + traj_to_actions).

  PYTHONPATH=/usr/lib/python3.12/dist-packages convenv/bin/python quantization_kit/verify_s1_opt.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_QK = Path(__file__).resolve().parent
_ROOT = _QK.parent
sys.path[:0] = [str(_ROOT), str(_QK)]

from internnav.model.utils.vln_utils import traj_to_actions  # noqa: E402
from quantization_kit.s1_inference import S1LiteModel, prep_images, prep_images_gpu  # noqa: E402
from quantization_kit.trt_s1 import attach_trt_denoise  # noqa: E402
from quantization_kit.verify_rgb_cache import real_latent  # noqa: E402

DEV = "cuda:0"
SEED = 1234
STEPS = 5


def build(trt=False, trt_cond=True):
    s1 = S1LiteModel(str(_ROOT / "checkpoints/InternVLA-N1-DualVLN")).load_s1_weights(
        str(_ROOT / "checkpoints/InternVLA-N1-DualVLN")).to(DEV, torch.float32).eval()
    if trt:
        attach_trt_denoise(s1, trt_cond=trt_cond)
    return s1


def main():
    goal = np.array(Image.open(_QK / "dbg/test448.png").convert("RGB"))
    latent = real_latent(goal)
    currents = [np.roll(goal, s, axis=1) for s in range(0, 96, 8)]   # 12 frames, panning scene
    z = np.zeros(goal.shape[:2], np.float32)
    print(f"[verify] real S2 latent {tuple(latent.shape)} | {len(currents)} frames | {STEPS} denoise steps\n")

    def acts(model, prep, cur):
        rgbs, depths = prep(goal, z, cur, z, DEV, torch.float32)
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        dp = model.generate_traj(traj_latents=latent, images_dp=rgbs, depths_dp=depths, num_inference_steps=STEPS)
        return [a for a in traj_to_actions(dp.clone()) if a != 0][:4]

    def timed(model, prep, n=25, w=6):
        cur = currents[1]
        f = lambda: acts(model, prep, cur)
        for _ in range(w):
            f()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n):
            f()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000

    # reference: fp32 PyTorch, PIL prep, eager conditioning
    ref_model = build(trt=False)
    ref = [acts(ref_model, prep_images, c) for c in currents]
    ms_ref = timed(ref_model, prep_images)

    trt_base = build(trt=True, trt_cond=False)   # current default (eager cond)
    trt_cond = build(trt=True, trt_cond=True)    # #3

    variants = [
        ("fp32 reference (PIL prep, eager cond)", ref_model, prep_images, ms_ref),
        ("TRT baseline   (PIL prep, eager cond)", trt_base, prep_images, None),
        ("#3 TRT +cond   (PIL prep, TRT  cond)", trt_cond, prep_images, None),
        ("#3+#4 +gpuprep (GPU prep, TRT  cond)", trt_cond, prep_images_gpu, None),
    ]

    print(f"{'variant':<42} {'action match':>13} {'run_s1':>9} {'Hz':>6}  {'vs ref':>7}")
    print("-" * 86)
    for name, model, prep, ms in variants:
        a = [acts(model, prep, c) for c in currents]
        match = sum(x == y for x, y in zip(a, ref))
        ms = ms if ms is not None else timed(model, prep)
        speed = "" if ms_ref == 0 else f"{ms_ref / ms:.2f}x"
        print(f"{name:<42} {match:>4}/{len(currents):<8} {ms:>6.1f}ms {1000/ms:>5.1f} {speed:>7}")

    print(f"\n  (action match = steps whose discrete actions equal the fp32 reference; "
          f"reference itself is {len(currents)}/{len(currents)} by definition)")


if __name__ == "__main__":
    main()
