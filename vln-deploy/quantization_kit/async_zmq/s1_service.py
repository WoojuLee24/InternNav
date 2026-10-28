"""S1 service: the fast (~30 Hz) trajectory policy as a standalone ZMQ process.

Owns ONLY System-1 (lite diffusion + optional native TensorRT FP16 denoise loop). No 7B
Qwen, no llama.cpp. Given a traj-latent from S2 plus the goal frame + current frame, it
denoises a trajectory and discretizes it to actions. Runs in its own process/GPU stream so
its 30 Hz cadence is never blocked by S2's 1-2 Hz reasoning.

Ports (binds):  PULL s1_in (latent + rgb)   PUSH s1_out (discrete actions)
Run standalone:  PYTHONPATH=/usr/lib/python3.12/dist-packages \
                   convenv/bin/python -m quantization_kit.async_zmq.s1_service
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zmq

_QK = Path(__file__).resolve().parents[1]
_ROOT = _QK.parent
sys.path[:0] = [str(_ROOT), str(_QK)]

from internnav.model.utils.vln_utils import traj_to_actions  # noqa: E402
from quantization_kit.async_zmq.zmq_proto import DEFAULT_PORTS, make_socket, recv_msg, send_msg  # noqa: E402
from quantization_kit.s1_inference import S1LiteModel, prep_images, prep_images_gpu  # noqa: E402


def build_s1(args):
    path = str(_ROOT / "checkpoints/InternVLA-N1-DualVLN")
    s1 = S1LiteModel(path).load_s1_weights(path).to(args.device, torch.float32).eval()
    if not args.no_trt:
        from quantization_kit.trt_s1 import attach_trt_denoise
        attach_trt_denoise(s1, trt_cond=True)   # TRT FP16: denoise + DINOv2 + memory_encoder/QFormer (#3)
    return s1


def _dp_to_waypoints(dp):
    """dp: [B*num_sample_trajs, T, 3] per-step deltas (dx,dy,dyaw). Average the sampled trajectories and
    cumsum (dx,dy) -> one representative xy path [T+1,2] in the robot frame (x forward, y left), from the
    origin. Visualization only; discrete actions still come from traj_to_actions on the full dp."""
    arr = dp.detach().to("cpu", torch.float32).numpy()
    arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])      # [N, T, 3]
    xy = np.cumsum(arr.mean(axis=0)[:, :2], axis=0)          # [T, 2]
    return np.vstack([[0.0, 0.0], xy]).astype(np.float32)    # [T+1, 2]


@torch.no_grad()
def run_s1(s1, latent, rgb_mem, rgb_cur, device, steps, cache_key=None, prep_fn=prep_images):
    depth0 = np.zeros(rgb_cur.shape[:2], np.float32)          # RGB-only DualVLN path
    rgbs, depths = prep_fn(rgb_mem, depth0, rgb_cur, depth0, device, torch.float32)
    latent = torch.from_numpy(latent).to(device, torch.float32)
    dp = s1.generate_traj(traj_latents=latent, images_dp=rgbs, depths_dp=depths,
                          num_inference_steps=steps, rgb_cache_key=cache_key)
    waypoints = _dp_to_waypoints(dp)                         # [T+1,2] robot-frame path, for viz
    actions = [a for a in traj_to_actions(dp.clone()) if a != 0]
    return actions[:4], waypoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--in-port", type=int, default=DEFAULT_PORTS["s1_in"])
    ap.add_argument("--out-port", type=int, default=DEFAULT_PORTS["s1_out"])
    ap.add_argument("--s1-steps", type=int, default=5, help="diffusion steps: 5 ~= 30 Hz")
    ap.add_argument("--no-trt", action="store_true", help="PyTorch fp32 instead of native TRT")
    ap.add_argument("--rgb-cache", action="store_true",
                    help="reuse goal-frame DINOv2 features (bit-exact; but only ~0.3 ms/tick here, "
                         "since DINOv2 is ~1.3 ms — the denoise loop is the real cost)")
    ap.add_argument("--no-gpu-prep", action="store_true",
                    help="use CPU PIL resize instead of GPU resize (GPU prep is ~20 ms faster but "
                         "bicubic differs from PIL; action-preserving on the test set — see verify_s1_opt.py)")
    args = ap.parse_args()

    print("[s1] building lite S1 (+TRT memory_encoder/QFormer) ...", flush=True)
    s1 = build_s1(args)                                       # attach_trt_denoise -> #3 TRT cond by default
    prep_fn = prep_images if args.no_gpu_prep else prep_images_gpu   # #4 GPU resize by default
    rgb_cache = args.rgb_cache
    if rgb_cache:
        s1.enable_rgb_cache(True)   # reuse goal-frame DINOv2 features (bit-exact; see verify_rgb_cache.py)

    # warm up engines/CUDA so the first real request already runs at speed, and measure the
    # UNCONTENDED S1 latency (no S2 sharing the GPU) for an honest baseline.
    dummy = np.zeros((336, 336, 3), np.uint8)   # matches cam-size; S1 re-preps to 224 either way
    warm_ms = 0.0
    for i in range(5):
        t = time.perf_counter()
        run_s1(s1, np.zeros((1, 4, 3584), np.float32), dummy, dummy, args.device, args.s1_steps,
               cache_key=(-1 if rgb_cache else None), prep_fn=prep_fn)   # warm the batch-1 engine too
        torch.cuda.synchronize()
        warm_ms = (time.perf_counter() - t) * 1000.0

    ctx = zmq.Context.instance()
    pull = make_socket(ctx, zmq.PULL, args.in_port, bind=True)
    push = make_socket(ctx, zmq.PUSH, args.out_port, bind=True)
    backend = "PyTorch fp32" if args.no_trt else "native TRT FP16"
    print(f"S1_SERVICE READY  in=PULL:{args.in_port}  out=PUSH:{args.out_port}  "
          f"| uncontended {warm_ms:.1f} ms/infer ({backend}, {args.s1_steps} steps -> {1000/warm_ms:.0f} Hz)"
          f" | prep={'CPU/PIL' if args.no_gpu_prep else 'GPU'} rgb_cache={'on' if rgb_cache else 'off'}",
          flush=True)

    while True:
        try:
            meta, arrays = recv_msg(pull)
        except (KeyboardInterrupt, zmq.ContextTerminated):
            break
        seq = meta.get("seq", -1)
        t0 = time.perf_counter()
        actions, waypoints = run_s1(s1, arrays["latent"], arrays["rgb_mem"], arrays["rgb_cur"],
                                    args.device, args.s1_steps, cache_key=meta.get("cache_key"), prep_fn=prep_fn)
        t_ms = (time.perf_counter() - t0) * 1000.0
        send_msg(push, {"seq": seq, "actions": [int(a) for a in actions], "t_infer_ms": t_ms},
                 {"waypoints": waypoints})


if __name__ == "__main__":
    main()
