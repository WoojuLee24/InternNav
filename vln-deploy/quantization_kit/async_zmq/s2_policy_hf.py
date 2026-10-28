"""HF (transformers) System-2 backend for s2_service.py — drop-in for InternVLAN1LlamaCppPolicy(load_s1=False).

`s2_service.py --s2-backend hf` builds this instead of the GGUF/llama.cpp policy; the service loop is unchanged
because the same surface is provided:
    reset()                                   new episode
    step_no_infer(rgb, depth, pose)           "accumulate": history only
    s2_step(rgb, depth, pose, instr, intrinsic, look_down) -> S2Output(output_action | output_latent+output_pixel)
    llm_output                                last generated text
    _cmd("LATENT")                            base64 f32 [4, 3584] latent of the last turn (force_latent demo aid)

The HF model is InternNav's InternVLAN1ForCausalLM (bf16 / bitsandbytes nf4 / int8, legacy wo-dagger config) as
loaded by scripts/sim/quant.py, driven by the realworld agent's own step_s2 / step_no_infer
(internnav/agent/internvla_n1_agent_realworld.py via scripts/sim/vln_agent.SimVLNAgent) — identical prompt,
history sampling, look-down two-step and latent extraction to the monolithic HTTP server (scripts/sim/vln_server.py).
The System-1 modules of the checkpoint are dropped after loading (the s1 module owns them).

Pixel goal convention: [row, col] in the pixel space of the frame passed to s2_step (the agent resizes to
resize x resize internally but the LLM answers in input-frame coordinates — .claude/memory/261002_sim_vln_quant_result.md).
"""
import base64
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

_QK = Path(__file__).resolve().parents[1]
_DEPLOY_ROOT = _QK.parent                      # vln-deploy
_INTERNNAV = _DEPLOY_ROOT.parent               # InternNav repo (vln-deploy/internnav is a symlink to it)
sys.path[:0] = [str(_INTERNNAV / "scripts" / "sim"), str(_INTERNNAV)]

from internnav.model.utils.vln_utils import S2Output  # noqa: E402
from vln_agent import SimVLNAgent  # noqa: E402  (scripts/sim)

# System-1 parts of InternVLAN1ForCausalLM.model — not used by S2 (the s1 module owns them)
S1_MODULES = ("traj_dit", "action_encoder", "action_decoder", "cond_projector", "pos_encoding", "rgb_model",
              "memory_encoder", "rgb_resampler", "navdp")


class HFS2Policy:
    def __init__(self, model_path, device="cuda:0", quant="bf16", resize=384, num_history=8, plan_step_gap=4,
                 log_dir=None):
        args = Namespace(model_path=model_path, device=device, quant=quant, resize_w=resize, resize_h=resize,
                         num_history=num_history, plan_step_gap=plan_step_gap, log_dir=log_dir, s1_depth_clip=None)
        self.agent = SimVLNAgent(args)
        inner = self.agent.model.get_model()
        for name in S1_MODULES:
            if hasattr(inner, name):
                delattr(inner, name)
        torch.cuda.empty_cache()
        self.last_latent = None

    # -- InternVLAN1LlamaCppPolicy surface used by s2_service.py --
    @property
    def llm_output(self):
        return self.agent.llm_output

    def reset(self):
        self.agent.reset()
        self.last_latent = None

    def step_no_infer(self, rgb, depth, pose):
        self.agent.step_no_infer(rgb, depth, pose)

    @torch.inference_mode()
    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        action, latent, pixel = self.agent.step_s2(rgb, depth, pose, instruction, intrinsic, look_down)
        out = S2Output()
        if action is not None:
            out.output_action = list(action)
        else:
            out.output_latent = latent
            out.output_pixel = np.asarray(pixel)
            self.last_latent = latent
        return out

    def _cmd(self, cmd):
        """Only "LATENT" (force_latent demo aid): latent of the last pixel-goal turn."""
        assert cmd == "LATENT", f"HFS2Policy supports only the LATENT command, got {cmd!r}"
        if self.last_latent is None:
            v = np.zeros((4, 3584), np.float32)
        else:
            v = self.last_latent.float().cpu().numpy().reshape(-1, 3584)
        return base64.b64encode(np.ascontiguousarray(v, np.float32).tobytes())


def build_hf_policy(args):
    return HFS2Policy(args.hf_model_path, device=args.device, quant=args.hf_quant, resize=args.hf_resize,
                      num_history=args.num_history, plan_step_gap=args.hf_plan_step_gap)
