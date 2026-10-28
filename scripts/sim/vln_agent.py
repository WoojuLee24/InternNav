"""InternVLAN1AsyncAgent variant for the sim bridge.

Differences from internnav/agent/internvla_n1_agent_realworld.py (kept untouched):
  * model loaded through quant.load_internvla (bf16 / nf4 / int8)
  * no per-step debug jpg dumps (step_no_infer wrote one per frame); llm text is logged
    only when log_dir is set
  * optional S1 depth clip (habitat agent uses 5.0 m: internvla_n1_agent.py:60)
  * step() runs under torch.inference_mode() and also reports whether S2 ran + the raw LLM text
"""
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # .../InternNav
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/diffusion-policy"))

from transformers import AutoProcessor  # noqa: E402

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent  # noqa: E402

from quant import load_internvla  # noqa: E402


class SimVLNAgent(InternVLAN1AsyncAgent):
    def __init__(self, args):
        # Same state as the parent __init__, minus the hard-wired bf16 load and test_data dir.
        self.device = torch.device(args.device)
        self.log_dir = Path(args.log_dir) if getattr(args, "log_dir", None) else None
        self.save_dir = self.log_dir or Path("/tmp/vln_agent")
        os.makedirs(self.save_dir, exist_ok=True)
        self.model = load_internvla(args.model_path, device=args.device, quant=args.quant)
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.processor.tokenizer.padding_side = "left"
        self.s1_depth_clip = getattr(args, "s1_depth_clip", None)

        self.resize_w = args.resize_w
        self.resize_h = args.resize_h
        self.num_history = args.num_history
        self.PLAN_STEP_GAP = args.plan_step_gap

        prompt = (
            "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go "
            "next to stay on track? Please output the next waypoint's coordinates in the image. Please "
            "output STOP when you have successfully completed the task."
        )
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]
        self.actions2idx = OrderedDict({'STOP': [0], "↑": [1], "←": [2], "→": [3], "↓": [5]})
        self.last_s2_idx = -100
        self.reset()

    def reset(self):
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        self.last_s2_idx = -100
        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

    def step_no_infer(self, rgb, depth, pose):
        image = Image.fromarray(rgb).convert('RGB').resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        self.episode_idx += 1

    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        if self.s1_depth_clip is not None:
            depth = np.minimum(depth, self.s1_depth_clip).astype(np.float32)
        prev_s2 = self.last_s2_idx
        # the parent runs S1 (generate_traj) outside no_grad -> traj_to_actions(.numpy()) fails on a grad tensor
        with torch.inference_mode():
            out = super().step(rgb, depth, pose, instruction, intrinsic, look_down=look_down)
        out.is_s2_step = self.last_s2_idx != prev_s2 or look_down
        out.llm_output = self.llm_output if out.is_s2_step else None
        return out
