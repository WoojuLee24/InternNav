import copy
import itertools
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent.parent))
ROOT = Path(__file__).resolve().parents[2]

from collections import OrderedDict

from PIL import Image
from transformers import AutoProcessor

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import S2Output, split_and_clean, traj_to_actions

DEFAULT_IMAGE_TOKEN = "<image>"


class InternVLAN1AsyncAgent:
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.require_flash_attn = bool(getattr(args, 'require_flash_attn', True))
        self.use_tf32 = bool(getattr(args, 'tf32', False))
        # self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = ROOT / "test_data" / datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"args.model_path{args.model_path}")
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": self.device},
        )
        self.model.eval()
        self.model.to(self.device)

        if self.device.type != 'cuda':
            raise RuntimeError("GPU is required for realworld debug pipeline")

        if self.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
            print("[Runtime] TF32 enabled")

        attn_impl = getattr(self.model.config, '_attn_implementation', None)
        if attn_impl is None:
            attn_impl = getattr(self.model.config, 'attn_implementation', None)
        print(f"[Runtime] device={self.device} attn_impl={attn_impl}")
        if self.require_flash_attn and attn_impl != 'flash_attention_2':
            raise RuntimeError(
                f"FlashAttention-2 required but got attn_impl={attn_impl}"
            )

        self.processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
        self.processor.tokenizer.padding_side = 'left'

        self.resize_w = args.resize_w
        self.resize_h = args.resize_h
        self.num_history = args.num_history
        self.PLAN_STEP_GAP = args.plan_step_gap
        self.use_kv_cache = bool(getattr(args, 'kv_cache', False))
        self.max_new_tokens = int(getattr(args, 'max_new_tokens', 128))
        self.use_tensorrt = bool(getattr(args, 'tensorrt', False))
        self.use_quantization = bool(getattr(args, 'quantization', False))
        self.quant_method = str(getattr(args, 'quant_method', 'dynamic'))
        self.tensorrt_engine = getattr(args, 'tensorrt_engine', None)

        self._init_safe_acceleration_modes()

        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        self.last_s2_idx = -100

        # output
        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

    def _init_safe_acceleration_modes(self):
        if self.use_tensorrt:
            try:
                import tensorrt  # noqa: F401
                if self.tensorrt_engine and os.path.exists(self.tensorrt_engine):
                    print(f"[TensorRT] Engine path configured: {self.tensorrt_engine} (integration pending)")
                else:
                    print("[TensorRT] Enabled but no valid engine path provided; fallback to PyTorch")
            except Exception:
                print("[TensorRT] Not available in runtime; fallback to PyTorch")

        if self.use_quantization:
            if self.device.type == 'cuda':
                print("[Quantization] CUDA runtime detected; skip quantization to preserve behavior")
                return

            if self.quant_method != 'dynamic':
                print(f"[Quantization] Unsupported method '{self.quant_method}' in safe mode; skip")
                return

            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear},
                    dtype=torch.qint8,
                )
                self.model.eval()
                print("[Quantization] Applied dynamic INT8 quantization (CPU mode)")
            except Exception as e:
                print(f"[Quantization] Failed to apply dynamic quantization: {repr(e)}")

    def reset(self):
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None

        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

        # self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = ROOT / "test_data" / datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def step_no_infer(self, rgb, depth, pose):
        image = Image.fromarray(rgb).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}.jpg")
        self.episode_idx += 1

    def trajectory_tovw(self, trajectory, kp=1.0):
        subgoal = trajectory[-1]
        linear_vel, angular_vel = kp * np.linalg.norm(subgoal[:2]), kp * subgoal[2]
        linear_vel = np.clip(linear_vel, 0, 0.5)
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        return linear_vel, angular_vel

    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False, temperature=1.0, repetition_penalty=1.0):
        dual_sys_output = S2Output()
        no_output_flag = self.output_action is None and self.output_latent is None
        if (self.episode_idx - self.last_s2_idx > self.PLAN_STEP_GAP) or look_down or no_output_flag:
            t_s2_start = time.time()
            self.output_action, self.output_latent, self.output_pixel = self.step_s2(
                rgb, depth, pose, instruction, intrinsic, look_down,
                temperature, repetition_penalty
            )
            self.last_s2_idx = self.episode_idx
            dual_sys_output.output_pixel = self.output_pixel
            self.pixel_goal_rgb = copy.deepcopy(rgb)
            self.pixel_goal_depth = copy.deepcopy(depth)
            t_s2_end = time.time()
            print(f"[System 2] inference time: {t_s2_end - t_s2_start:.4f}s")
        else:
            self.step_no_infer(rgb, depth, pose)

        if self.output_action is not None:
            dual_sys_output.output_action = copy.deepcopy(self.output_action)
            self.output_action = None
        elif self.output_latent is not None:
            t_s1_start = time.time()
            processed_pixel_rgb = np.array(Image.fromarray(self.pixel_goal_rgb).resize((224, 224))) / 255
            processed_pixel_depth = np.array(Image.fromarray(self.pixel_goal_depth).resize((224, 224)))
            processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
            processed_depth = np.array(Image.fromarray(depth).resize((224, 224)))
            rgbs = (
                torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
                .unsqueeze(0)
                .to(self.device)
            )
            depths = (
                torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )
            trajectories = self.step_s1(self.output_latent, rgbs, depths)

            dual_sys_output.output_trajectory = traj_to_actions(trajectories, use_discrate_action=False)
            t_s1_end = time.time()
            print(f"[System 1] inference time: {t_s1_end - t_s1_start:.4f}s")

        return dual_sys_output

    def step_s2(self, rgb, depth, pose, instruction, intrinsic, look_down=False, temperature=1.0, repetition_penalty=1.0):
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
            # image.save(f"{self.save_dir}/debug_raw_{self.episode_idx:04d}.jpg")
        else:
            # image.save(f"{self.save_dir}/debug_raw_{self.episode_idx:04d}_look_down.jpg")
            pass
        if not look_down:
            self.conversation_history = []
            self.past_key_values = None

            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
                placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'

            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1
        else:
            self.input_images.append(image)
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        prompt = self.conjunctions[0] + DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        prompt_instruction = copy.deepcopy(sources[0]["value"])
        parts = split_and_clean(prompt_instruction)

        content = []
        for i in range(len(parts)):
            if parts[i] == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({"type": "text", "text": parts[i]})

        self.conversation_history.append({'role': 'user', 'content': content})

        text = self.processor.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)

        proc_images = self.input_images
        for _ in range(2):
            try:
                inputs = self.processor(text=[text], images=proc_images, return_tensors="pt")
                break
            except IndexError as e:
                if "image_grid_thw" not in str(e) or len(proc_images) <= 1:
                    raise
                proc_images = proc_images[:-1]
                print(f"[System 2] Processor image mismatch; retry with {len(proc_images)} images")
        else:
            inputs = self.processor(text=[text], images=proc_images, return_tensors="pt")
        vocab_size = getattr(self.model.config, "vocab_size", None)
        if vocab_size is None and getattr(self.model.config, "text_config", None) is not None:
            vocab_size = getattr(self.model.config.text_config, "vocab_size", None)
        if vocab_size is not None and hasattr(inputs, "input_ids"):
            input_ids = inputs.input_ids
            invalid_mask = (input_ids < 0) | (input_ids >= vocab_size)
            if invalid_mask.any():
                eos_id = int(getattr(self.model.config, "eos_token_id", 0) or 0)
                bad_count = int(invalid_mask.sum().item())
                print(f"[System 2] Found {bad_count} invalid token ids; replacing with eos={eos_id}")
                input_ids = input_ids.clone()
                input_ids[invalid_mask] = eos_id
                inputs["input_ids"] = input_ids
        inputs = inputs.to(self.device)
        t0 = time.time()
        gen_kwargs = {
            'max_new_tokens': self.max_new_tokens,
            'do_sample': False,
            'return_dict_in_generate': True,
            'use_cache': bool(self.use_kv_cache),
            'temperature': temperature,
            'repetition_penalty': repetition_penalty,
        }
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
                # past_key_values=self.past_key_values,
                # raw_input_ids=copy.deepcopy(inputs.input_ids),
            )
        output_ids = outputs.sequences

        t1 = time.time()
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        with open(f"{self.save_dir}/llm_output_{self.episode_idx: 04d}.txt", 'w') as f:
            f.write(self.llm_output)
        self.last_output_ids = copy.deepcopy(output_ids[0])
        self.past_key_values = copy.deepcopy(outputs.past_key_values)
        print(f"output {self.episode_idx}  {self.llm_output} cost: {t1 - t0}s")
        if bool(re.search(r'\d', self.llm_output)):
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            if len(coord) < 2:
                action_seq = self.parse_actions(self.llm_output)
                return action_seq, None, None
            pixel_goal = [int(coord[1]), int(coord[0])]
            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            pixel_values = inputs.pixel_values
            t0 = time.time()
            with torch.no_grad():
                traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)
                return None, traj_latents, pixel_goal

        else:
            action_seq = self.parse_actions(self.llm_output)
            return action_seq, None, None

    def step_s1(self, latent, rgb, depth):
        all_trajs = self.model.generate_traj(latent, rgb, depth)
        return all_trajs
