"""S2 service: the slow (~1-2 Hz) high-level reasoner as a standalone ZMQ process.

Owns the System-2 brain (prompt / sampled history / look-down two-step / pixel parse /
latent extraction) via InternVLAN1LlamaCppPolicy(load_s1=False), which here drives the
C++ llama-s2-runner over its NEW ZMQ REP mode (--zmq). No System-1, no diffusion, no
TensorRT is loaded in this process -> it cannot stall the 30 Hz S1 loop.

Ports (binds):  PULL s2_in (frames to infer)   PUB s2_out (pixel goal + latent | discrete action)
Run standalone:  convenv/bin/python -m quantization_kit.async_zmq.s2_service
"""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zmq

_QK = Path(__file__).resolve().parents[1]
_ROOT = _QK.parent
sys.path[:0] = [str(_ROOT), str(_QK)]

from quantization_kit.async_zmq.zmq_proto import (  # noqa: E402
    DEFAULT_PORTS,
    ep,
    make_socket,
    recv_msg,
    send_msg,
)


def build_policy(args):
    # imported here (not at module level) so `--s2-backend hf` does not need the llama.cpp policy installed
    from internnav.model.basemodel.internvla_n1.internvla_n1_policy_llamacpp import InternVLAN1LlamaCppPolicy

    if args.hf_parity:  # HF image resolution (384 -> 392, look-down smart_resize; MTMD_QWEN25VL_HF_ALIGN=1)
        from internnav.model.basemodel.internvla_n1.internvla_n1_policy_llamacpp_hfparity import (
            InternVLAN1LlamaCppPolicyHFParity as InternVLAN1LlamaCppPolicy,
        )

    build = "build" if platform.machine() == "aarch64" else "build_x86"  # Jetson native vs x86 dev
    bin_dir = os.path.join(_QK, "llama.cpp", build, "bin")
    gguf = os.path.join(_QK, "gguf")
    text_gguf = {"f16": "InternVLA-S2-text-f16-latent.gguf",
                 "q4": "InternVLA-S2-text-Q4_K_M-latent.gguf"}[args.s2_model]
    # S2_RUNNER_PIPE=1 -> drive the runner over stdin/stdout instead of ZMQ (--zmq). Use on hosts whose
    # llama-s2-runner predates the --zmq REP mode (e.g. an older x86 build); pipe mode needs no rebuild.
    use_pipe = os.environ.get("S2_RUNNER_PIPE") == "1"
    return InternVLAN1LlamaCppPolicy(
        runner_bin=os.path.join(bin_dir, "llama-s2-runner"),
        # --gguf-text / --gguf-mmproj override the bundled files (e.g. quantization_kit/out/gguf/*)
        model_gguf=args.gguf_text or os.path.join(gguf, text_gguf),  # Q4 = 3.3x smaller -> faster decode
        mmproj_gguf=args.gguf_mmproj or os.path.join(gguf, "InternVLA-S2-mmproj-f16.gguf"),
        **({} if args.hf_parity else {"lookdown_full_res": args.lookdown_full_res}),
        lib_dir=bin_dir,
        s1_model_path=os.path.join(_ROOT, "checkpoints/InternVLA-N1-DualVLN"),
        device=args.device,
        resize=args.resize,   # S2 input res; 112-aligned (448/336/224). Set via --resize / demo --cam-size.
        num_history=args.num_history,
        kv_reuse=args.kv_reuse,
        load_s1=False,                                   # <-- S2 only; no diffusion in this process
        runner_endpoint=(None if use_pipe else ep(args.runner_port, args.host)),  # ZMQ (--zmq) unless pipe
        spawn_runner=True,
        embd_cache=args.embd_cache,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--in-port", type=int, default=DEFAULT_PORTS["s2_in"])
    ap.add_argument("--out-port", type=int, default=DEFAULT_PORTS["s2_out"])
    ap.add_argument("--runner-port", type=int, default=DEFAULT_PORTS["s2_runner"])
    ap.add_argument("--num-history", type=int, default=8)
    ap.add_argument("--resize", type=int, default=336,
                    help="S2 input square resolution; use a 112-aligned value (224/336/448). "
                         "448 = HF-parity reference (safest nav); 336 = ~11%% faster, latent cos ~0.93-0.96. "
                         "MUST match the camera --size so pixel goals map 1:1 (the demos keep them in sync).")
    ap.add_argument("--kv-reuse", action="store_true")
    ap.add_argument("--s2-model", choices=["f16", "q4"], default="q4",
                    help="q4 (default) = Q4_K_M-latent: ~3.3x smaller, ~1.2x faster decode, vln-ce-validated; "
                         "f16 = reference (parity-verified, slower)")
    ap.add_argument("--gguf-text", default=None, help="[gguf] text GGUF path (default: gguf/<--s2-model file>)")
    ap.add_argument("--gguf-mmproj", default=None, help="[gguf] mmproj GGUF path (default: gguf/InternVLA-S2-mmproj-f16.gguf)")
    ap.add_argument("--hf-parity", action="store_true",
                    help="[gguf] same image resolution as the HF model (use with --resize 384; implies look-down full res)")
    ap.add_argument("--embd-cache", type=int, default=0,
                    help="[gguf] vision-embedding cache size (entries, LRU); history frames pre-encoded on accumulate. "
                         "Bit-identical output; 0 = off (re-encode every image each call)")
    ap.add_argument("--lookdown-full-res", action="store_true",
                    help="[gguf] look-down frame at input resolution (HF policy parity; default: resized)")
    ap.add_argument("--s2-backend", choices=["gguf", "hf"], default="gguf",
                    help="gguf (default) = llama-s2-runner Q4/f16; hf = transformers InternVLA-N1 (s2_policy_hf.py)")
    ap.add_argument("--hf-model-path", default=os.path.join(_ROOT, "checkpoints/InternVLA-N1-DualVLN"),
                    help="[hf] checkpoint dir (DualVLN / w-NavDP / wo-dagger)")
    ap.add_argument("--hf-quant", choices=["bf16", "nf4", "int8"], default="bf16", help="[hf] LLM precision")
    ap.add_argument("--hf-resize", type=int, default=384, help="[hf] S2 input resize (training: 384)")
    ap.add_argument("--hf-plan-step-gap", type=int, default=4, help="[hf] agent PLAN_STEP_GAP (unused by the loop)")
    args = ap.parse_args()

    if args.s2_backend == "hf":
        from quantization_kit.async_zmq.s2_policy_hf import build_hf_policy

        print(f"[s2] loading HF InternVLA-N1 S2 [{args.hf_quant}] {args.hf_model_path} @ {args.hf_resize}px ...",
              flush=True)
        policy = build_hf_policy(args)
    else:
        policy = None
    if policy is None and args.resize % 112 != 0 and not args.hf_parity:
        print(f"[s2] WARNING: --resize {args.resize} is NOT 112-aligned; the runner (clip.cpp) will round it "
              f"to {args.resize // 112 * 112 or 112}. Use 224/336/448 to avoid a silent mismatch with the camera.",
              flush=True)
    if policy is None:
        print(f"[s2] loading Qwen2.5-VL GGUF runner [{args.s2_model.upper()}] @ {args.resize}px "
              f"(ZMQ {ep(args.runner_port, args.host)}) ...", flush=True)
        policy = build_policy(args)

    ctx = zmq.Context.instance()
    pull = make_socket(ctx, zmq.PULL, args.in_port, bind=True)
    pub = make_socket(ctx, zmq.PUB, args.out_port, bind=True)
    print(f"S2_SERVICE READY  in=PULL:{args.in_port}  out=PUB:{args.out_port}", flush=True)

    n_infer = 0
    while True:
        try:
            meta, arrays = recv_msg(pull)
        except (KeyboardInterrupt, zmq.ContextTerminated):
            break
        mode = meta.get("mode", "infer")
        seq = meta.get("seq", -1)

        if mode == "reset":
            policy.reset()
            continue
        if mode == "accumulate":            # keep S2 history correct without inferring
            policy.step_no_infer(arrays["rgb"], None, None)
            continue

        # ---- mode == "infer" : run the full S2 decision (with the look-down two-step) ----
        rgb = arrays["rgb"]
        instr = meta.get("instruction", "")
        t0 = time.perf_counter()
        try:
            out = policy.s2_step(rgb, None, None, instr, None, look_down=False)
            text = policy.llm_output
            goal_rgb = rgb
            if out.output_action and 5 in out.output_action and "lookdown_rgb" in arrays:
                out = policy.s2_step(arrays["lookdown_rgb"], None, None, instr, None, look_down=True)
                goal_rgb = arrays["lookdown_rgb"]
                text = text + "  >  " + policy.llm_output
            # DEMO AID: force a latent even when S2 chose a discrete action, so the 30 Hz S1
            # path is always exercised. The runner KV still holds the last TEXT, so a LATENT
            # probe is valid. On real navigation leave force_latent off and honour S2's choice.
            if meta.get("force_latent") and out.output_latent is None:
                import base64
                v = np.frombuffer(base64.b64decode(policy._cmd("LATENT")), np.float32).reshape(4, -1)
                out.output_latent = torch.from_numpy(v.copy()).unsqueeze(0)
        except Exception as e:              # never wedge the orchestrator: fall back to STOP
            print(f"[s2] infer error: {e}", flush=True)
            policy.reset()
            send_msg(pub, {"seq": seq, "kind": "action", "action": [0], "pixel": None,
                           "text": f"ERR:{e}", "t_infer_ms": 0.0})
            continue
        t_ms = (time.perf_counter() - t0) * 1000.0
        n_infer += 1

        if out.output_latent is not None:   # pixel goal -> hand the latent to S1
            latent = out.output_latent.detach().to("cpu", dtype=torch.float32).numpy()
            send_msg(
                pub,
                {"seq": seq, "kind": "latent",
                 "pixel": (out.output_pixel.tolist() if out.output_pixel is not None else None),
                 "action": None, "text": text, "t_infer_ms": t_ms},
                {"latent": latent.astype(np.float32), "goal_rgb": goal_rgb.astype(np.uint8)},
            )
            print(f"[s2] #{n_infer} seq={seq} LATENT pixel={out.output_pixel} ({t_ms:.0f} ms)", flush=True)
        else:                               # discrete action(s) -> executed directly, no S1
            acts = [int(a) for a in (out.output_action or [])]
            send_msg(pub, {"seq": seq, "kind": "action", "action": acts, "pixel": None,
                           "text": text, "t_infer_ms": t_ms})
            print(f"[s2] #{n_infer} seq={seq} ACTION {acts} ({t_ms:.0f} ms)", flush=True)

    del policy


if __name__ == "__main__":
    main()
