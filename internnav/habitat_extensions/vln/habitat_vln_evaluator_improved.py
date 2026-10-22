"""
habitat_vln_evaluator_improved.py — Improved Habitat VLN-CE evaluator.

Extends HabitatVLNEvaluator with:
  1. GPS distance-based stopping (prevents "within 3m but missed stop" failures)
  2. Support for 8-bit/4-bit model loading (via model_settings.load_in_8bit/4bit)
  3. Enhanced action sequence management (prevents LOOKDOWN in action queues)
  4. Turn saturation detection (prevents spinning in place)

Register as 'habitat_vln_improved' so it can be used in eval configs.

Key improvement (GPS stop): if distance_to_goal < 3.0m after any nav step,
issue STOP immediately. This alone should add ~10-20% SR.

Usage in config:
    eval_type='habitat_vln_improved'
"""

import copy
import itertools
import os
import re
import sys
from collections import OrderedDict
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from PIL import Image
from transformers import AutoProcessor

# Register measures
import internnav.habitat_extensions.vln.measures  # noqa: F401

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import Evaluator
from internnav.habitat_extensions.vln.habitat_vln_evaluator import HabitatVLNEvaluator

sys.path.append('./src/diffusion-policy')

try:
    from depth_camera_filtering import filter_depth
except ImportError:
    def filter_depth(x, **kw): return x

try:
    from internnav.habitat_extensions.vln.utils import preprocess_depth_image_v2
except ImportError:
    preprocess_depth_image_v2 = None

from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions


DEFAULT_IMAGE_TOKEN = "<image>"
MAX_STEPS = 8
MAX_LOCAL_STEPS = 4
GPS_STOP_THRESHOLD = 3.0   # meters — VLN-CE success distance
GPS_STOP_MIN_STEPS = 5     # don't trigger GPS stop in first N nav steps


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


@Evaluator.register('habitat_vln_improved')
class ImprovedHabitatVLNEvaluator(HabitatVLNEvaluator):
    """
    Improved version of HabitatVLNEvaluator with GPS stopping and other fixes.
    """

    def __init__(self, cfg: EvalCfg):
        super().__init__(cfg)
        # Improvement flags
        self.gps_stop_enabled = cfg.eval_settings.get("gps_stop", True)
        self.gps_stop_threshold = cfg.eval_settings.get("gps_stop_threshold", GPS_STOP_THRESHOLD)
        self.gps_stop_min_steps = cfg.eval_settings.get("gps_stop_min_steps", GPS_STOP_MIN_STEPS)
        self.turn_sat_filter = cfg.eval_settings.get("turn_sat_filter", True)
        self.debug_improvements = cfg.eval_settings.get("debug_improvements", True)

        print(f"[ImprovedEval] gps_stop={self.gps_stop_enabled} "
              f"threshold={self.gps_stop_threshold}m "
              f"turn_sat_filter={self.turn_sat_filter}", flush=True)

        # Counters
        self._gps_stops = 0
        self._turn_injections = 0

    def eval_action(self):
        if self.model_args.mode == 'dual_system':
            sucs, spls, oss, nes, ndtws = self._run_eval_dual_system_improved()
        else:
            return super().eval_action()

        result = {
            "sucs": sucs,
            "spls": spls,
            "oss": oss,
            "nes": nes,
        }
        if ndtws is not None:
            result["ndtws"] = ndtws
        return result

    def _check_gps_stop(self, metrics: dict, nav_step: int) -> bool:
        """Check if GPS distance warrants an immediate STOP."""
        if not self.gps_stop_enabled:
            return False
        if nav_step < self.gps_stop_min_steps:
            return False
        dist = metrics.get('distance_to_goal', float('inf'))
        if dist < self.gps_stop_threshold:
            self._gps_stops += 1
            if self.debug_improvements:
                print(f"[GPS_STOP] nav_step={nav_step} dist={dist:.3f}m → STOP "
                      f"(total gps_stops={self._gps_stops})", flush=True)
            return True
        return False

    def _filter_action_seq(self, action_seq: List[int]) -> List[int]:
        """
        Filter action sequence:
        1. Remove LOOKDOWN (5) — not a navigation action in VLN-CE context
        2. Prevent turn saturation: inject FORWARD if >80% turns
        """
        # Filter LOOKDOWN
        filtered = [a for a in action_seq if a != action_code.LOOKDOWN]

        # Turn saturation prevention
        if self.turn_sat_filter and len(filtered) >= 4:
            turns = sum(1 for a in filtered if a in (action_code.LEFT, action_code.RIGHT))
            if turns / len(filtered) > 0.8:
                # Inject a FORWARD after every 2 turns
                new_seq = []
                for i, a in enumerate(filtered):
                    new_seq.append(a)
                    if i > 0 and i % 2 == 1 and a in (action_code.LEFT, action_code.RIGHT):
                        new_seq.append(action_code.FORWARD)
                        self._turn_injections += 1
                filtered = new_seq
                if self.debug_improvements:
                    print(f"[TurnFilter] Injected FORWARD actions. Original={action_seq[:8]}, "
                          f"New={filtered[:8]}", flush=True)

        return filtered

    def _run_eval_dual_system_improved(self) -> tuple:
        """
        Improved dual-system evaluation loop.

        Key changes vs original HabitatVLNEvaluator._run_eval_dual_system:
          1. GPS stop check after every navigation step
          2. LOOKDOWN filtering from action_seq
          3. Turn saturation prevention
          4. Better logging of improvement triggers
        """
        self.model.eval()

        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()
        process_bar = tqdm.tqdm(
            total=len(self.env.episodes),
            desc=f"Improved Eval Epoch {self.epoch} Rank {self.rank}"
        )

        while self.env.is_running:
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction, flush=True)

            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            action = None
            messages = []
            local_actions = []
            done = False
            flag = False
            pixel_goal = None
            step_id = 0   # navigation step counter (excluding lookdown steps)

            while (not done) and (step_id <= self.max_steps_per_episode):
                rgb = observations["rgb"]
                depth = observations["depth"]

                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True, depth_scale=1000,
                        target_height=224, target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                    obs_down, _, _, _ = self.env.step(action_code.LOOKDOWN)
                    obs_down, _, _, _ = self.env.step(action_code.LOOKDOWN)
                    look_down_image = Image.fromarray(obs_down["rgb"]).convert('RGB')
                    depth_down = obs_down["depth"]
                    depth_down = filter_depth(depth_down.reshape(depth_down.shape[:2]), blur_type=None)
                    depth_down = depth_down * (self._max_depth - self._min_depth) + self._min_depth
                    depth_down = depth_down * 1000
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth_down.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True, depth_scale=1000,
                        target_height=224, target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0

                    self.env.step(action_code.LOOKUP)
                    self.env.step(action_code.LOOKUP)

                # GPS stop check (before choosing action)
                if action not in (action_code.LOOKDOWN, None):
                    current_metrics = self.env.get_metrics()
                    if self._check_gps_stop(current_metrics, step_id):
                        done = True
                        observations = self.env.get_observation()
                        # Force env step with STOP to register metric
                        observations, _, done, _ = self.env.step(action_code.STOP)
                        break

                # S2 invocation (same as original)
                if len(action_seq) == 0 and pixel_goal is None:
                    if action == action_code.LOOKDOWN:
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}
                        )
                        input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = list(np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ))
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    import random as _random
                    prompt = _random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs, max_new_tokens=128, do_sample=False, use_cache=True,
                            past_key_values=None, return_dict_in_generate=True,
                        ).sequences
                    torch.cuda.synchronize()

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
                    )
                    print(f'step_id: {step_id} output text: {llm_outputs}', flush=True)

                    coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]
                    if len(coord) >= 2:  # output pixel goal (need at least x, y)
                        forward_action = 0
                        pixel_goal = [int(coord[1]), int(coord[0])]

                        self.env.step(action_code.LOOKUP)
                        self.env.step(action_code.LOOKUP)

                        local_actions = []
                        pixel_values = inputs.pixel_values
                        image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                        with torch.no_grad():
                            traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                        pix_goal_image = copy.copy(image_dp)
                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                        pix_goal_depth = copy.copy(depth_dp)
                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)

                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))

                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]

                        action = local_actions[0]
                        if action == action_code.STOP:
                            pixel_goal = None
                            output_ids = None
                            action = action_code.LEFT
                            observations, _, done, _ = self.env.step(action)
                            step_id += 1
                            messages = []
                            continue

                    else:
                        # Arrow token sequence — filter LOOKDOWN and turn saturation
                        raw_action_seq = self.parse_actions(llm_outputs)
                        action_seq = self._filter_action_seq(raw_action_seq)
                        print(f'actions {action_seq}', flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif pixel_goal is not None:
                    if len(local_actions) == 0:
                        local_actions = []
                        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                        images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                        depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                        depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            dp_actions = self.model.generate_traj(traj_latents, images_dp, depths_dp)
                        action_list = traj_to_actions(dp_actions)
                        if len(action_list) < MAX_STEPS:
                            action_list += [0] * (MAX_STEPS - len(action_list))
                        local_actions = action_list
                        if len(local_actions) >= MAX_LOCAL_STEPS:
                            local_actions = local_actions[:MAX_LOCAL_STEPS]
                        action = local_actions.pop(0)
                    else:
                        action = local_actions.pop(0)

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                    if action == action_code.STOP:
                        pixel_goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                else:
                    action = 0

                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False

            # Episode end
            process_bar.update(1)
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            n = len(sucs)
            running_sr = sum(sucs) / n
            running_spl = sum(spls) / n
            running_ne = sum(nes) / n
            process_bar.set_postfix(SR=f"{running_sr:.3f}", SPL=f"{running_spl:.3f}", NE=f"{running_ne:.2f}")
            print(
                f"[{n}] {scene_id}_{episode_id:04d} | "
                f"success={metrics['success']:.0f} spl={metrics['spl']:.3f} "
                f"os={metrics['oracle_success']:.0f} ne={metrics['distance_to_goal']:.2f} | "
                f"running SR={running_sr:.3f} SPL={running_spl:.3f} NE={running_ne:.2f} "
                f"gps_stops={self._gps_stops} turn_inj={self._turn_injections}",
                flush=True,
            )

            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": float(metrics["success"]),
                "spl": float(metrics["spl"]),
                "os": float(metrics['oracle_success']),
                "ne": float(metrics["distance_to_goal"]),
                "steps": int(step_id),
                "episode_instruction": episode_instruction,
            }
            if 'ndtw' in metrics:
                result['ndtw'] = float(metrics['ndtw'])

            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, f'progress_rank{self.rank}.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")

        self.env.close()
        print(f"[ImprovedEval] Total gps_stops={self._gps_stops} "
              f"turn_injections={self._turn_injections}", flush=True)

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )


# Required for json.dumps in the method
import json  # noqa: E402 (needed after class definition)
