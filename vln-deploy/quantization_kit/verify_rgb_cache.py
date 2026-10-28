"""Equivalence check for the S1 goal-frame DINOv2 feature cache (S1LiteModel.enable_rgb_cache).

The cache reuses the goal/memory frame's DINOv2 features while the goal is unchanged. DINOv2 is
a per-image function (LayerNorm-only, no cross-frame attention), so this is mathematically exact.
This harness measures it three ways on a REAL S2 latent + realistic frames (fixed goal, panning
current), with the denoise RNG seeded identically each call so the ONLY variable is the conditioning:

  ref    = production: both frames through DINOv2 as one batch=2 call (no cache)
  split  = both frames encoded separately (batch=1), goal recomputed every step (no reuse)
  cache  = both frames encoded separately (batch=1), goal reused after step 0 (the cache)

Reported:
  cache vs split  -> isolates the CACHE itself (reuse vs recompute). Expect EXACTLY 0.
  cache vs ref    -> the only difference from production, which is batch-1-vs-batch-2 GPU fp
                     noise (NOT the cache), plus whether discrete ACTIONS are preserved.

  PYTHONPATH=/usr/lib/python3.12/dist-packages convenv/bin/python quantization_kit/verify_rgb_cache.py
"""
import base64
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_QK = Path(__file__).resolve().parent
_ROOT = _QK.parent
sys.path[:0] = [str(_ROOT), str(_QK)]

from internnav.model.basemodel.internvla_n1.internvla_n1_policy_llamacpp import InternVLAN1LlamaCppPolicy  # noqa: E402
from internnav.model.utils.vln_utils import traj_to_actions  # noqa: E402
from quantization_kit.s1_inference import S1LiteModel, prep_images  # noqa: E402

DEV = "cuda:0"
SEED = 1234
STEPS = 10
SCRATCH = os.environ.get("TMPDIR", "/tmp")


def real_latent(goal_img):
    """One real S2 traj-latent for the goal frame (force the LATENT probe even if S2 chose a
    discrete action, so we exercise S1 on a genuine plan). Cached to disk for fast reruns."""
    cache = os.path.join(SCRATCH, "verify_latent.npy")
    if os.path.exists(cache):
        return torch.from_numpy(np.load(cache)).to(DEV, torch.float32)
    import platform  # [fix] Thor/5090 자동 분기
    B = os.path.join(_QK, "llama.cpp", "build" if platform.machine() == "aarch64" else "build_x86", "bin")
    G = os.path.join(_QK, "gguf")
    # [fix] f16-latent 없으면 Q4_K_M-latent fallback
    model_gguf = os.path.join(G, "InternVLA-S2-text-f16-latent.gguf")
    if not os.path.exists(model_gguf):
        model_gguf = os.path.join(G, "InternVLA-S2-text-Q4_K_M-latent.gguf")
    pol = InternVLAN1LlamaCppPolicy(
        runner_bin=os.path.join(B, "llama-s2-runner"),
        model_gguf=model_gguf,
        mmproj_gguf=os.path.join(G, "InternVLA-S2-mmproj-f16.gguf"),
        lib_dir=B, s1_model_path=str(_ROOT / "checkpoints/InternVLA-N1-DualVLN"),
        resize=336, load_s1=False)
    out = pol.s2_step(goal_img, None, None, "Walk to the chair and stop.", None)
    if out.output_latent is None:
        v = np.frombuffer(base64.b64decode(pol._cmd("LATENT")), np.float32).reshape(4, -1)
        out.output_latent = torch.from_numpy(v.copy()).unsqueeze(0)
    lat = out.output_latent.float().cpu().numpy()
    del pol
    np.save(cache, lat)
    return torch.from_numpy(lat).to(DEV, torch.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trt", action="store_true", help="verify the native TRT FP16 path (else PyTorch fp32)")
    args = ap.parse_args()

    goal = np.array(Image.open(_QK / "dbg/test448.png").convert("RGB"))
    latent = real_latent(goal)
    print(f"[verify] real S2 latent {tuple(latent.shape)}  |  goal=test448.png  |  "
          f"backend={'native TRT FP16' if args.trt else 'PyTorch fp32'}\n")

    s1 = S1LiteModel(str(_ROOT / "checkpoints/InternVLA-N1-DualVLN")).load_s1_weights(
        str(_ROOT / "checkpoints/InternVLA-N1-DualVLN")).to(DEV, torch.float32).eval()
    if args.trt:
        from quantization_kit.trt_s1 import attach_trt_denoise
        attach_trt_denoise(s1)   # replaces s1.generate_traj with the TRT FP16 loop (+ batch-1 rgb engine)

    # realistic current frames: a horizontal pan of the real scene (goal stays fixed)
    currents = [np.roll(goal, s, axis=1) for s in (0, 8, 16, 24, 32, 40, 48, 56, 64, 72)]
    z = np.zeros(goal.shape[:2], np.float32)

    def gen(cur, key):
        rgbs, depths = prep_images(goal, z, cur, z, DEV, torch.float32)
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        return s1.generate_traj(traj_latents=latent, images_dp=rgbs, depths_dp=depths,
                                num_inference_steps=STEPS, rgb_cache_key=key)

    def acts(t):
        return [a for a in traj_to_actions(t.clone()) if a != 0][:4]

    # pass REF (batch-2, no cache)
    s1._rgb_cache_enabled = False
    ref = [gen(c, None) for c in currents]
    # pass SPLIT (batch-1, unique key each step -> always miss -> recompute goal, no reuse)
    s1.enable_rgb_cache(True)
    split = [gen(c, ("split", i)) for i, c in enumerate(currents)]
    # pass CACHE (batch-1, constant key -> goal encoded once, reused)
    s1.enable_rgb_cache(True)
    cache = [gen(c, "GOAL") for c in currents]
    cache_misses = s1._rgb_cache_misses

    print(f"{'step':>4} | {'cache vs split':>16} | {'cache vs ref':>13} | actions(ref -> cache)   match")
    print("-" * 78)
    all_reuse_exact, all_actions_match, max_cr = True, True, 0.0
    for i in range(len(currents)):
        d_cs = (cache[i] - split[i]).abs().max().item()
        d_cr = (cache[i] - ref[i]).abs().max().item()
        a_ref, a_cache = acts(ref[i]), acts(cache[i])
        ok = a_ref == a_cache
        all_reuse_exact &= (d_cs == 0.0)
        all_actions_match &= ok
        max_cr = max(max_cr, d_cr)
        print(f"{i:>4} | {d_cs:>16.2e} | {d_cr:>13.2e} | {str(a_ref):>10} -> {str(a_cache):<10} {'OK' if ok else 'DIFF'}")

    print("\n=== verdict ===")
    print(f"  cache reuse exact (cache==split, max|Δ|=0) ............ {all_reuse_exact}   "
          f"[goal encoded {cache_misses}x over {len(currents)} steps -> reused {len(currents)-cache_misses}x]")
    print(f"  discrete actions identical to production (batch-2) .... {all_actions_match}")
    print(f"  trajectory max|Δ| vs production (batch-1 vs batch-2 fp). {max_cr:.2e}  (NOT from caching; the batch-split)")
    if all_reuse_exact and all_actions_match:
        print(f"\n  => The cache itself is BIT-EXACT (0 error). Encoding the goal frame separately\n"
              f"     adds only ~{max_cr:.0e} fp GPU batch-kernel noise that does NOT change any action.")
    else:
        print("\n  => see rows above")

    # ---- speed: cache off (re-encode both frames) vs cache on (reuse goal, encode current only) ----
    import time

    def timed(fn, n=30, w=5):
        for _ in range(w):
            fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000

    cur = currents[1]
    s1._rgb_cache_enabled = False
    ms_off = timed(lambda: gen(cur, None))
    s1.enable_rgb_cache(True); gen(cur, "WARM")                 # prime the cache (1 miss)
    ms_on = timed(lambda: gen(cur, "WARM"))                     # steady-state: goal reused
    print(f"\n=== S1 generate_traj latency ({'TRT FP16' if args.trt else 'PyTorch fp32'}, {STEPS} steps, incl. prep) ===")
    print(f"  rgb_cache OFF (DINOv2 on 2 frames) : {ms_off:6.1f} ms")
    print(f"  rgb_cache ON  (DINOv2 on 1 frame)  : {ms_on:6.1f} ms   ({ms_off / ms_on:.2f}x faster, -{ms_off - ms_on:.1f} ms)")


if __name__ == "__main__":
    main()
