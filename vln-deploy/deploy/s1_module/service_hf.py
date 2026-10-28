"""S1 module — HF (transformers) backend (`service.py --s1-backend hf`). The lite/TRT backend is untouched.

Runs the System-1 of any InternVLA-N1 HF checkpoint (DualVLN nextdit_async, w-NavDP / wo-dagger navdp_async)
exactly as the original agent does it: the checkpoint is loaded on CPU, the System-2 modules (LLM layers,
vision tower, lm_head, embeddings) are deleted, the rest goes to CUDA in bf16, and the input processing +
generate_traj call is a verbatim copy of InternVLAN1AsyncAgent.step() (internnav/agent/internvla_n1_agent_realworld.py)
-> for the same latent / frames / torch seed the waypoints equal the monolithic server's.

Wire: same as the lite service (interface.py S1 request/reply) plus OPTIONAL fields
    meta   "seed": int            torch.manual_seed before diffusion (reproducible comparisons)
    arrays "depth_mem", "depth_cur": f32 [H,W] metres (NavDP uses depth; zeros if absent, as the lite path)
reply waypoints are f32 [T+1, 2] (protocol dtype) = traj_to_actions(dp, use_discrate_action=False).
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]       # vln-deploy
_INTERNNAV = _ROOT.parent                          # InternNav repo
S2_ONLY = [("model", "layers"), ("model", "embed_tokens"), ("model", "norm"), ("", "visual"), ("", "lm_head")]
READY_TOKEN = "S1_SERVICE READY"


def load_s1_only(model_path, device):
    import torch

    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
    from quant import legacy_config  # scripts/sim (wo-dagger legacy config -> navdp_async)

    # attn_implementation only affects the LLM / vision attention, which is deleted below (flash-attn refuses CPU)
    kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cpu"})
    cfg = legacy_config(model_path)
    if cfg is not None:
        kwargs["config"] = cfg
    model = InternVLAN1ForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    for parent, name in S2_ONLY:
        owner = model.get_submodule(parent) if parent else model
        if hasattr(owner, name):
            delattr(owner, name)
    gc.collect()
    return model.to(device)


def s1_infer(model, latent, goal_rgb, goal_depth, rgb, depth, device):
    import torch
    from PIL import Image

    from internnav.model.utils.vln_utils import traj_to_actions

    # --- verbatim from InternVLAN1AsyncAgent.step (S1 branch) ---
    processed_pixel_rgb = np.array(Image.fromarray(goal_rgb).resize((224, 224))) / 255
    processed_pixel_depth = np.array(Image.fromarray(goal_depth).resize((224, 224)))
    processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
    processed_depth = np.array(Image.fromarray(depth).resize((224, 224)))
    rgbs = torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)]).unsqueeze(0).to(device)
    depths = (
        torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(device)
    )
    dp = model.generate_traj(latent, rgbs, depths)
    waypoints = traj_to_actions(dp, use_discrate_action=False)
    actions = [a for a in traj_to_actions(dp) if a != 0]   # same discretizer as the lite service
    return actions[:4], np.asarray(waypoints, np.float32)


def run_hf(args, in_port, out_port, beacon):
    sys.path[:0] = [str(_INTERNNAV / "scripts" / "sim"), str(_INTERNNAV), str(_INTERNNAV / "src/diffusion-policy")]
    import torch
    import zmq

    from quantization_kit.async_zmq.zmq_proto import make_socket, recv_msg, send_msg

    device = torch.device(args.device)
    print(f"[s1] loading HF System-1 from {args.hf_model_path} ...", flush=True)
    model = load_s1_only(args.hf_model_path, device)

    def run(meta, a):
        latent = torch.from_numpy(a["latent"]).to(device=device, dtype=torch.bfloat16)
        h, w = a["rgb_cur"].shape[:2]
        dm = a.get("depth_mem", np.zeros(a["rgb_mem"].shape[:2], np.float32))
        dc = a.get("depth_cur", np.zeros((h, w), np.float32))
        if meta.get("seed") is not None:
            torch.manual_seed(int(meta["seed"]))
        with torch.inference_mode():
            return s1_infer(model, latent, a["rgb_mem"], dm, a["rgb_cur"], dc, device)

    dummy = {"latent": np.zeros((1, model.config.n_query, 3584), np.float32),
             "rgb_mem": np.zeros((480, 640, 3), np.uint8), "rgb_cur": np.zeros((480, 640, 3), np.uint8)}
    warm_ms = 0.0
    for _ in range(3):
        t = time.perf_counter()
        run({}, dummy)
        torch.cuda.synchronize()
        warm_ms = (time.perf_counter() - t) * 1000.0

    ctx = zmq.Context.instance()
    pull = make_socket(ctx, zmq.PULL, in_port, bind=True)
    push = make_socket(ctx, zmq.PUSH, out_port, bind=True)
    beacon.mark_ready()
    print(f"{READY_TOKEN}  in=PULL:{in_port}  out=PUSH:{out_port}  | HF {model.config.system1} bf16 "
          f"| uncontended {warm_ms:.1f} ms/infer | cuda alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB",
          flush=True)
    while True:
        try:
            meta, arrays = recv_msg(pull)
        except (KeyboardInterrupt, zmq.ContextTerminated):
            break
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        actions, waypoints = run(meta, arrays)
        torch.cuda.synchronize()
        t_ms = (time.perf_counter() - t0) * 1000.0
        send_msg(push, {"seq": meta.get("seq", -1), "actions": [int(x) for x in actions], "t_infer_ms": t_ms},
                 {"waypoints": waypoints})
