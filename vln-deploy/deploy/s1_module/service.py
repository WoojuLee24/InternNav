"""S1 module — deploy-owned System-1 service (lite diffusion + TensorRT, ~30 Hz).

Deploy-local fork of quantization_kit/async_zmq/s1_service.py. The wire ships the
DENOISED, robot-facing waypoint path — exactly what the original repo hands to control
(internnav/model/utils/vln_utils.py):

    waypoints = traj_to_actions(dp, use_discrate_action=False)     # [T+1, 2]
      = /4.0 unnormalize -> per-sample cumsum -> np.mean(all_trajectory, axis=0)

i.e. the SAME trajectory the discrete actions are derived from — metric, robot frame
(x fwd, y left), starting at the origin. Individual diffusion samples are not sent.

Ports (binds):  PULL s1_in (latent + rgb)   PUSH s1_out (actions + waypoints)
Health: REP `s1_health` beacon, live from process start (ready flips after warm-up).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.common.config import DeployConfig  # noqa: E402
from deploy.common.interface import HealthBeacon  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--in-port", type=int, default=None)
    ap.add_argument("--out-port", type=int, default=None)
    ap.add_argument("--s1-steps", type=int, default=10,
                    help="diffusion steps: 10 (default) = original agent; 5 ~= 2x faster but waypoints ~13 mm off "
                         "(scripts/sim/report.md)")
    ap.add_argument("--no-trt", action="store_true", help="PyTorch fp32 instead of native TRT")
    ap.add_argument("--rgb-cache", action="store_true", help="reuse goal-frame DINOv2 features (bit-exact)")
    ap.add_argument("--no-gpu-prep", action="store_true", help="CPU PIL resize instead of GPU resize")
    ap.add_argument("--trt-onnx-dir", default=None, help="[lite] ONNX dir for TRT builds (default quantization_kit/onnx)")
    ap.add_argument("--trt-engine-dir", default=None,
                    help="[lite] TRT engine dir (default quantization_kit/trt_engines; e.g. quantization_kit/out/trt_engines)")
    ap.add_argument("--s1-backend", choices=["lite", "hf"], default="lite",
                    help="lite (default) = S1LiteModel (+TRT); hf = transformers checkpoint S1 (service_hf.py)")
    ap.add_argument("--hf-model-path", default=str(_ROOT / "checkpoints/InternVLA-N1-DualVLN"),
                    help="[hf] checkpoint dir (DualVLN / w-NavDP / wo-dagger)")
    args = ap.parse_args()

    cfg = DeployConfig.from_env()
    in_port = args.in_port or cfg.ports["s1_in"]
    out_port = args.out_port or cfg.ports["s1_out"]
    beacon = HealthBeacon("s1", cfg.ports["s1_health"])
    beacon.start()                                    # answers ready:false during the load below
    if args.s1_backend == "hf":
        from deploy.s1_module.service_hf import run_hf

        return run_hf(args, in_port, out_port, beacon)

    # heavy imports AFTER the beacon is up
    import torch
    import zmq
    from internnav.model.utils.vln_utils import traj_to_actions
    from quantization_kit.async_zmq.zmq_proto import make_socket, recv_msg, send_msg
    from quantization_kit.s1_inference import S1LiteModel, prep_images, prep_images_gpu

    print("[s1] building lite S1 (+TRT memory_encoder/QFormer) ...", flush=True)
    path = str(_ROOT / "checkpoints/InternVLA-N1-DualVLN")
    s1 = S1LiteModel(path).load_s1_weights(path).to(args.device, torch.float32).eval()
    if not args.no_trt:
        from quantization_kit.trt_s1 import attach_trt_denoise
        attach_trt_denoise(s1, onnx_dir=args.trt_onnx_dir, engine_dir=args.trt_engine_dir, trt_cond=True)
    prep_fn = prep_images if args.no_gpu_prep else prep_images_gpu
    if args.rgb_cache:
        s1.enable_rgb_cache(True)

    @torch.no_grad()
    def run_s1(latent, rgb_mem, rgb_cur, cache_key=None):
        depth0 = np.zeros(rgb_cur.shape[:2], np.float32)          # RGB-only DualVLN path
        rgbs, depths = prep_fn(rgb_mem, depth0, rgb_cur, depth0, args.device, torch.float32)
        lat = torch.from_numpy(latent).to(args.device, torch.float32)
        dp = s1.generate_traj(traj_latents=lat, images_dp=rgbs, depths_dp=depths,
                              num_inference_steps=args.s1_steps, rgb_cache_key=cache_key)
        # THE robot-facing denoised path (original code, verbatim): [T+1, 2] metric
        waypoints = traj_to_actions(dp.clone(), use_discrate_action=False).astype(np.float32)
        actions = [a for a in traj_to_actions(dp.clone()) if a != 0]   # original discretizer
        return actions[:4], waypoints

    # warm up (CUDA graphs / TRT) + measure the uncontended latency
    dummy = np.zeros((336, 336, 3), np.uint8)
    warm_ms = 0.0
    for _ in range(5):
        t = time.perf_counter()
        run_s1(np.zeros((1, 4, 3584), np.float32), dummy, dummy,
               cache_key=(-1 if args.rgb_cache else None))
        torch.cuda.synchronize()
        warm_ms = (time.perf_counter() - t) * 1000.0

    ctx = zmq.Context.instance()
    pull = make_socket(ctx, zmq.PULL, in_port, bind=True)
    push = make_socket(ctx, zmq.PUSH, out_port, bind=True)
    beacon.mark_ready()
    backend = "PyTorch fp32" if args.no_trt else "native TRT FP16"
    print(f"S1_SERVICE READY  in=PULL:{in_port}  out=PUSH:{out_port}  "
          f"| uncontended {warm_ms:.1f} ms/infer ({backend}, {args.s1_steps} steps -> {1000 / warm_ms:.0f} Hz)"
          f" | denoised waypoints (robot-facing)", flush=True)

    while True:
        try:
            meta, arrays = recv_msg(pull)
        except (KeyboardInterrupt, zmq.ContextTerminated):
            break
        seq = meta.get("seq", -1)
        t0 = time.perf_counter()
        actions, waypoints = run_s1(arrays["latent"], arrays["rgb_mem"], arrays["rgb_cur"],
                                    cache_key=meta.get("cache_key"))
        t_ms = (time.perf_counter() - t0) * 1000.0
        send_msg(push, {"seq": seq, "actions": [int(a) for a in actions], "t_infer_ms": t_ms},
                 {"waypoints": waypoints})


if __name__ == "__main__":
    main()
