#!/usr/bin/env python3
"""
High-Performance InternVLA Server ( Optimized) version 1 for Real-World Robot

Optimizations applied:
- AsyncS2 (no rate limiting, max throughput)
- FP16 precision (PyTorch automatic)
- Flash Attention via transformers
- Dynamic INT8 Quantization for LLM
- Vision embedding caching
- Optimized generation settings
- Warmup for fast first inference

Usage:
    python http_internvla_server_debug_optimized.py
    python http_internvla_server_debug_optimized.py --no-quantization  # disable INT8
    python http_internvla_server_debug_optimized.py --cache-size 100  # larger cache
"""

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import traceback
from flask import Flask, jsonify, request
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformation import Calibration
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM, InternVLAN1ModelConfig
from internnav.model.utils.vln_utils import S2Output, split_and_clean, traj_to_actions

try:
    from vision_cache import VisionEmbeddingCache as ExternalVisionEmbeddingCache
except Exception:
    ExternalVisionEmbeddingCache = None

try:
    from quantization import quantize_for_deployment
except Exception:
    quantize_for_deployment = None

try:
    from tensorrt_converter import TensorRTConverter
except Exception:
    TensorRTConverter = None

app = Flask(__name__)
idx = 0
DEFAULT_INSTRUCTION = "Exit the door, then Turn left and go straight until you find small fire extinguisher. Then stop."


def get_best_attention():
    if not torch.cuda.is_available():
        print("[Agent] CUDA not available, using 'eager' attention")
        return "eager"
    try:
        import flash_attn
        return "flash_attention_2"
    except:
        return "sdpa"


class VisionImageCache:
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.cache: OrderedDict[str, Image.Image] = OrderedDict()
        self.lock = threading.RLock()

    def _hash(self, image: Image.Image) -> str:
        arr = np.asarray(image.resize((16, 16)))
        return str(hash(arr.tobytes()))

    def get(self, image: Image.Image) -> Optional[Image.Image]:
        key = self._hash(image)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key].copy()
            return None

    def put(self, image: Image.Image):
        key = self._hash(image)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                self.cache[key] = image.copy()

    def clear(self):
        with self.lock:
            self.cache.clear()

    def get_stats(self):
        with self.lock:
            return {'size': len(self.cache), 'max_size': self.max_size}


class AsyncS2:
    def __init__(self, step_func):
        self.step_func = step_func
        self.queue = queue.Queue(maxsize=1)
        self.latest_result = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.stats = {'s2_updates': 0, 'total_ms': 0, 'errors': 0}
        self.update_times = []
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def request(self, *args):
        try:
            self.queue.put_nowait(args)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(args)
            except:
                pass
    
    def get_result(self):
        with self.lock:
            return self.latest_result
    
    def _run(self):
        while self.running:
            try:
                args = self.queue.get(timeout=0.01)
                t0 = time.time()
                result = self.step_func(*args)
                ms = (time.time() - t0) * 1000
                
                with self.lock:
                    self.latest_result = result
                    self.stats['s2_updates'] += 1
                    self.stats['total_ms'] += ms
                    self.update_times.append(time.time())
                    if len(self.update_times) > 200:
                        self.update_times.pop(0)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[S2] Error: {e}")
                traceback.print_exc()
                self.stats['errors'] += 1


def apply_deployment_quantization(model, enabled: bool, method: str):
    if not enabled:
        print("[Quantization] Skipped (disabled)")
        return model, "disabled"
    if quantize_for_deployment is None:
        print("[Quantization] Helper module unavailable, fallback to FP16")
        return model, "unavailable"
    if torch.cuda.is_available():
        print("[Quantization] Dynamic torch INT8 is CPU-centric; keeping FP16 on GPU")
        return model, "gpu_fp16"
    try:
        q_model, metadata = quantize_for_deployment(model, method=method)
        return q_model, metadata.get('method', method)
    except Exception as e:
        print(f"[Quantization] Failed: {e}")
        return model, "failed"


class OptimizedAgent:
    def __init__(
        self,
        model_path,
        device='cuda:0',
        cache_size=50,
        use_quantization=True,
        quant_method='dynamic',
        use_tensorrt=False,
        max_history=2,
        max_new_tokens=16,
        use_kv_cache=True,
        s2_mode='async',
        s2_step_gap=8,
        disable_traj=False,
        s2_output_mode='auto',
        s2_decode_mode='normal',
        s2_discrete_to_pixel=False,
        s2_discrete_pixel_step=56,
    ):
        self.device = torch.device(device)
        self.use_quantization = use_quantization
        self.quant_method = quant_method
        self.use_tensorrt = use_tensorrt
        self.max_history = max_history
        self.max_new_tokens = max_new_tokens
        self.use_kv_cache = use_kv_cache
        self.s2_mode = s2_mode.lower()
        self.s2_step_gap = s2_step_gap
        self.disable_traj = disable_traj
        self.s2_output_mode = s2_output_mode.lower()
        self.s2_decode_mode = s2_decode_mode.lower()
        self.s2_discrete_to_pixel = bool(s2_discrete_to_pixel)
        self.s2_discrete_pixel_step = int(s2_discrete_pixel_step)
        self.tensorrt_status = "disabled"
        print(f"[Agent] Loading model on {self.device}...")
        print(f"[Agent] Quantization: {'INT8' if use_quantization else 'FP16 only'}")
        
        # Load model config
        self.config = InternVLAN1ModelConfig.from_pretrained(model_path)
        attn = get_best_attention()
        print(f"[Agent] Attention: {attn}")
        
        # Load model with FP16
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch.float16,
            attn_implementation=attn,
            device_map={"": self.device}
        )
        self.model.eval()
        
        self.model, self.quant_method_applied = apply_deployment_quantization(
            self.model,
            enabled=use_quantization,
            method=quant_method,
        )
        self._materialize_navdp_meta()

        if use_tensorrt:
            self._init_tensorrt(model_path)
        
        # Load processor
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = 'left'
        
        self.prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next? Please output the next waypoint's coordinates in the image. Please output STOP when done."
        self.actions2idx = OrderedDict({'STOP': [0], "↑": [1], "←": [2], "→": [3], "↓": [5]})
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]
        
        # State
        self.rgb_list = []
        self.conversation_history = []
        self.llm_output = ""
        self.episode_idx = 0
        self._last_s2_idx = -100
        self.input_images = []
        
        # Vision cache
        self.vision_embed_cache = None
        if ExternalVisionEmbeddingCache is not None:
            try:
                self.vision_embed_cache = ExternalVisionEmbeddingCache(max_size=cache_size)
                print("[Agent] External vision embedding cache enabled")
            except Exception as e:
                print(f"[Agent] External vision embedding cache init failed: {e}")
                self.vision_embed_cache = None
        self.vision_cache = VisionImageCache(max_size=cache_size)
        
        # Async worker is created once and used when mode is async.
        self.async_s2 = AsyncS2(self._step_s2)
        self.async_s2.start()
        self.sync_s2_stats = {'s2_updates': 0, 'total_ms': 0.0, 'errors': 0}
        self.sync_update_times = []
        
        self.cached = (None, None, None, False)
        self.frame_times = []
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self.s2_output_stats = {
            'pixel_coord': 0,
            'coord_parse_fail': 0,
            'discrete': 0,
            'empty': 0,
            'traj_latent': 0,
            'retry_count': 0,
            'retry_success': 0,
            'discrete_mapped_traj': 0,
        }
        self.last_llm_output = ""
        self.last_output_class = "none"
        self.s2_prompt_variant = 'legacy'
        self.s2_retry_on_discrete = False
        self.last_discrete_actions = None
        self.last_traj_error = ""
        self.navdp_meta_detected = 0
        
        # Warmup
        self._warmup()
        
        print(f"[Agent] Model ready")
        print(f"[Agent] Vision cache: {cache_size}")
        print(f"[Agent] TensorRT: {self.tensorrt_status}")
        print(f"[Agent] Warmup complete")

    def _init_tensorrt(self, model_path: str):
        if TensorRTConverter is None:
            self.tensorrt_status = "helper_unavailable"
            print("[TensorRT] Converter helper unavailable")
            return
        try:
            converter = TensorRTConverter(model_path)
            if not converter.check_trt_available():
                self.tensorrt_status = "python_unavailable"
                print("[TensorRT] Python runtime not available")
                return
            self.tensorrt_status = "runtime_available"
            print("[TensorRT] Runtime detected and initialized")
        except Exception as e:
            self.tensorrt_status = f"init_failed:{type(e).__name__}"
            print(f"[TensorRT] Initialization failed: {e}")

    def _materialize_navdp_meta(self):
        try:
            navdp = self.model.get_model().navdp
        except Exception:
            return

        device = torch.device(self.device)
        fixed = 0

        for module in navdp.modules():
            for name, param in list(module._parameters.items()):
                if param is None:
                    continue
                if getattr(param, "is_meta", False):
                    self.navdp_meta_detected += 1
                    dtype = torch.float16
                    module._parameters[name] = nn.Parameter(
                        torch.zeros(param.shape, dtype=dtype, device=device),
                        requires_grad=param.requires_grad,
                    )
                    fixed += 1
            for name, buf in list(module._buffers.items()):
                if buf is None:
                    continue
                if getattr(buf, "is_meta", False):
                    self.navdp_meta_detected += 1
                    module._buffers[name] = torch.zeros(buf.shape, dtype=torch.float16, device=device)
                    fixed += 1

        if fixed > 0:
            print(f"[Agent] Materialized {fixed} NavDP meta tensors")
    
    def _warmup(self):
        print("[Agent] Warming up model...")
        dummy_img = Image.new('RGB', (224, 224), color='gray')
        
        sources = [{"from": "human", "value": self.prompt.replace('<instruction>.', 'test')}, {"from": "gpt", "value": ""}]
        sources[0]["value"] += " you can see <image>."
        parts = split_and_clean(sources[0]["value"])
        
        content = []
        for p in parts:
            if p == "<image>":
                content.append({"type": "image", "image": dummy_img})
            else:
                content.append({"type": "text", "text": p})
        
        self.conversation_history = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(
            self.conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        try:
            inputs = self.processor(text=[text], images=[dummy_img], return_tensors="pt")
            inputs = {k: v for k, v in inputs.items() if k != 'mm_token_type_ids'}
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    use_cache=self.use_kv_cache
                )
            
            print("[Agent] Warmup complete")
        except Exception as e:
            print(f"[Agent] Warmup warning: {e}")
    
    @property
    def last_s2_idx(self):
        return self._last_s2_idx
    
    @last_s2_idx.setter
    def last_s2_idx(self, val):
        self._last_s2_idx = val
    
    def reset(self):
        self.rgb_list = []
        self.conversation_history = []
        self.llm_output = ""
        self.episode_idx = 0
        self._last_s2_idx = -100
        self.input_images = []
        self.cached = (None, None, None, False)
        self.frame_times = []
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self.vision_cache.clear()
        self.s2_output_stats = {
            'pixel_coord': 0,
            'coord_parse_fail': 0,
            'discrete': 0,
            'empty': 0,
            'traj_latent': 0,
            'retry_count': 0,
            'retry_success': 0,
            'discrete_mapped_traj': 0,
        }
        self.last_llm_output = ""
        self.last_output_class = "none"
        self.last_discrete_actions = None
        self.last_traj_error = ""
        self.navdp_meta_detected = 0
    
    def _step_s2(self, rgb, depth, pose, instruction, intrinsic, look_down):
        def _resolve_image_grid_thw(processor_inputs):
            grid = processor_inputs.get('image_grid_thw', None)
            if grid is None:
                raise KeyError("image_grid_thw not found in processor inputs")
            if isinstance(grid, list):
                return torch.cat([thw.unsqueeze(0) for thw in grid], dim=0)
            if torch.is_tensor(grid):
                return grid
            raise TypeError(f"Unsupported image_grid_thw type: {type(grid)}")

        def _compute_traj_latents(output_ids, processor_inputs):
            image_grid_thw = _resolve_image_grid_thw(processor_inputs)
            pixel_values = processor_inputs['pixel_values']
            with torch.no_grad():
                try:
                    return self.model.generate_latents(output_ids, pixel_values, image_grid_thw)
                except Exception as e:
                    if self.s2_output_mode == 'strict':
                        raise
                    if self.s2_output_mode in ('auto', 'legacy'):
                        print(f"[S2] generate_latents fallback: {e}")
                        hidden = self.model.get_hidden_states(output_ids, pixel_values, image_grid_thw)
                        n_query = self.model.get_n_query()
                        return hidden[:, -n_query:, :]
                    raise

        def _generate_and_decode(current_images):
            text = self.processor.apply_chat_template(
                self.conversation_history,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(text=[text], images=current_images, return_tensors="pt")
            inputs = {k: v for k, v in inputs.items() if k != 'mm_token_type_ids'}
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            input_len = inputs['input_ids'].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=self.use_kv_cache,
                    return_dict_in_generate=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id or 0,
                )
            output_ids = outputs.sequences if hasattr(outputs, 'sequences') else outputs
            llm_output = self.processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
            return llm_output, output_ids, inputs

        image = Image.fromarray(rgb).convert('RGB')
        cached_img = self.vision_cache.get(image)
        
        if cached_img is not None:
            image = cached_img
        else:
            image = image.resize((224, 224))
            self.vision_cache.put(image)
        
        if not look_down:
            self.rgb_list.append(image)
            self.conversation_history = []
            self.past_key_values = None
            if self.s2_decode_mode == 'coord_constrained':
                sources = [
                    {
                        "from": "human",
                        "value": (
                            "Navigation task: "
                            f"{instruction}. "
                            "Return only one next waypoint as row,col with integers. "
                            "If complete, return STOP only."
                        ),
                    },
                    {"from": "gpt", "value": ""}
                ]
            else:
                if self.s2_prompt_variant == 'coord_strict':
                    sources = [
                        {
                            "from": "human",
                            "value": (
                                "You are an autonomous navigation assistant. "
                                f"Your task is to {instruction}. "
                                "Output exactly one waypoint as two integers in image coordinates only, format: row,col. "
                                "If task is completed, output STOP only."
                            ),
                        },
                        {"from": "gpt", "value": ""}
                    ]
                else:
                    sources = [
                        {
                            "from": "human",
                            "value": (
                                "You are an autonomous navigation assistant. "
                                f"Your task is to {instruction}. "
                                "Where should you go next to stay on track? "
                                "Please output the next waypoint's coordinates in the image. "
                                "Please output STOP when you have successfully completed the task."
                            ),
                        },
                        {"from": "gpt", "value": ""}
                    ]
            cur_images = self.rgb_list[-1:]
            
            if len(self.rgb_list) <= 1:
                history_id = []
            else:
                max_hist_idx = len(self.rgb_list) - 2
                history_slots = max(0, int(self.max_history))
                history_id = np.unique(
                    np.linspace(0, max_hist_idx, history_slots, dtype=np.int32)
                ).tolist()
                placeholder = "<image>\n" * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'
            
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
        else:
            self.input_images.append(image)
            input_img_id = -1
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append({'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]})
        
        sources[0]["value"] += f" {self.conjunctions[0]}<image>."
        parts = split_and_clean(sources[0]["value"])
        
        content = []
        for p in parts:
            if p == "<image>":
                if 0 <= input_img_id < len(self.input_images):
                    content.append({"type": "image", "image": self.input_images[input_img_id]})
                    input_img_id += 1
                elif len(self.input_images) > 0:
                    content.append({"type": "image", "image": self.input_images[-1]})
                    input_img_id += 1
            else:
                content.append({"type": "text", "text": p})
        
        self.conversation_history.append({'role': 'user', 'content': content})
        
        self.llm_output, output_ids, inputs = _generate_and_decode(self.input_images)
        self.last_llm_output = self.llm_output.strip()
        
        # Check for pixel coordinates (trajectory mode)
        has_pixel_coord = bool(self.llm_output) and any(c.isdigit() for c in self.llm_output)
        if has_pixel_coord:
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            if len(coord) >= 2:
                pixel_goal = [int(coord[1]), int(coord[0])]
                self.s2_output_stats['pixel_coord'] += 1
                if self.disable_traj:
                    self.last_output_class = 'pixel_only'
                    return (None, None, pixel_goal)
                traj_latents = _compute_traj_latents(output_ids, inputs)
                self.s2_output_stats['traj_latent'] += 1
                self.last_output_class = 'pixel_plus_traj'
                return (None, traj_latents, pixel_goal)
            self.s2_output_stats['coord_parse_fail'] += 1

        if (not self.disable_traj) and (self.s2_output_mode in ('trajectory', 'force_traj')):
            traj_latents = _compute_traj_latents(output_ids, inputs)
            pixel_goal = [int(depth.shape[0] // 2), int(depth.shape[1] // 2)]
            self.s2_output_stats['traj_latent'] += 1
            self.last_output_class = 'forced_traj'
            return (None, traj_latents, pixel_goal)
        
        # Check for discrete actions
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(self.llm_output)
        actions = []
        for match in matches:
            actions.extend(self.actions2idx[match])

        if (
            actions
            and self.s2_retry_on_discrete
            and self.s2_prompt_variant == 'coord_strict'
            and not look_down
        ):
            self.s2_output_stats['retry_count'] += 1
            retry_text = (
                "Your previous output was not coordinate format. "
                "Now output exactly one waypoint as row,col (two integers) only. "
                "If done, output STOP only."
            )
            self.conversation_history.append({'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]})
            self.conversation_history.append({'role': 'user', 'content': [{'type': 'text', 'text': retry_text}]})
            retry_output, retry_ids, retry_inputs = _generate_and_decode(self.input_images)
            self.llm_output = retry_output
            self.last_llm_output = self.llm_output.strip()

            if any(c.isdigit() for c in self.llm_output):
                coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
                if len(coord) >= 2:
                    pixel_goal = [int(coord[1]), int(coord[0])]
                    self.s2_output_stats['pixel_coord'] += 1
                    self.s2_output_stats['retry_success'] += 1
                    if self.disable_traj:
                        self.last_output_class = 'pixel_only_retry'
                        return (None, None, pixel_goal)
                    traj_latents = _compute_traj_latents(retry_ids, retry_inputs)
                    self.s2_output_stats['traj_latent'] += 1
                    self.last_output_class = 'pixel_plus_traj_retry'
                    return (None, traj_latents, pixel_goal)

        if actions:
            if self.s2_discrete_to_pixel and (not self.disable_traj):
                h, w = depth.shape[:2]
                r, c = int(h // 2), int(w // 2)
                step = max(8, self.s2_discrete_pixel_step)
                a0 = actions[0]
                if a0 == 1:
                    r -= step
                elif a0 == 2:
                    c -= step
                elif a0 == 3:
                    c += step
                elif a0 == 5:
                    r += step
                r = max(0, min(h - 1, r))
                c = max(0, min(w - 1, c))
                pixel_goal = [r, c]
                traj_latents = _compute_traj_latents(output_ids, inputs)
                self.s2_output_stats['discrete_mapped_traj'] += 1
                self.s2_output_stats['traj_latent'] += 1
                self.last_discrete_actions = actions
                self.last_output_class = 'discrete_mapped_traj'
                return (None, traj_latents, pixel_goal)

            self.s2_output_stats['discrete'] += 1
            self.last_discrete_actions = actions
            self.last_output_class = 'discrete'
            return (actions, None, None)

        self.s2_output_stats['empty'] += 1
        self.last_output_class = 'empty'
        return (None, None, None)
    
    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        output = S2Output()
        mode = self.s2_mode
        
        # Decide if we need S2 inference
        need_s2 = (self.episode_idx - self.last_s2_idx > self.s2_step_gap) or look_down or not self.cached[3]

        if mode == 'sync':
            if need_s2:
                t0 = time.time()
                try:
                    result = self._step_s2(rgb, depth, pose, instruction, intrinsic, look_down)
                    elapsed_ms = (time.time() - t0) * 1000.0
                    self.sync_s2_stats['s2_updates'] += 1
                    self.sync_s2_stats['total_ms'] += elapsed_ms
                    self.sync_update_times.append(time.time())
                    if len(self.sync_update_times) > 200:
                        self.sync_update_times.pop(0)
                    if result is not None:
                        self.cached = result + (True,)
                except Exception:
                    self.sync_s2_stats['errors'] += 1
                    raise
                self.last_s2_idx = self.episode_idx
        else:
            if need_s2:
                self.async_s2.request(rgb, depth, pose, instruction, intrinsic, look_down)
                self.last_s2_idx = self.episode_idx
            elif not look_down:
                # Keep history cadence closer to legacy sync behavior.
                self.rgb_list.append(Image.fromarray(rgb).convert('RGB').resize((224, 224)))

            # Get latest result from async S2
            result = self.async_s2.get_result()
            if result:
                self.cached = result + (True,)
        
        action, latent, pixel, valid = self.cached
        
        # Handle discrete action
        if action is not None:
            output.output_action = action
            # Legacy behavior: consume action and do not keep previous latent/action cache.
            self.cached = (None, None, pixel, True)
        
        # Handle trajectory
        elif latent is not None:
            curr_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
            curr_depth = np.array(Image.fromarray(depth).resize((224, 224)))
            goal_rgb = np.array(Image.fromarray(self.pixel_goal_rgb).resize((224, 224))) / 255 if self.pixel_goal_rgb is not None else curr_rgb
            goal_depth = np.array(Image.fromarray(self.pixel_goal_depth).resize((224, 224))) if self.pixel_goal_depth is not None else curr_depth
            
            rgbs = torch.stack([torch.from_numpy(goal_rgb), torch.from_numpy(curr_rgb)]).unsqueeze(0).to(self.device)
            depths = torch.stack([torch.from_numpy(goal_depth), torch.from_numpy(curr_depth)]).unsqueeze(0).unsqueeze(-1).to(self.device)
            
            try:
                with torch.no_grad():
                    trajectories = self.model.generate_traj(latent, rgbs, depths)
                output.output_trajectory = traj_to_actions(trajectories, use_discrate_action=False)
            except Exception as e:
                self.s2_output_stats['coord_parse_fail'] += 1
                self.last_traj_error = f"{type(e).__name__}: {e}"
                print(f"[S1] Trajectory head error: {self.last_traj_error}")
                if self.last_discrete_actions is not None:
                    output.output_action = self.last_discrete_actions
                    self.last_output_class = f'traj_head_fail_fallback_discrete:{type(e).__name__}'
                    self.cached = (None, None, pixel, True)
                else:
                    self.last_output_class = f'traj_head_fail:{type(e).__name__}'
        
        # Update pixel goal
        if output.output_pixel is None and pixel is not None:
            output.output_pixel = pixel
            self.pixel_goal_rgb = rgb.copy()
            self.pixel_goal_depth = depth.copy()
        
        if not look_down:
            self.episode_idx += 1
        
        self.frame_times.append(time.time())
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
        
        return output
    
    def get_stats(self):
        hz = 0
        if len(self.frame_times) > 1:
            intervals = np.diff(self.frame_times[-10:])
            if len(intervals) > 0 and np.mean(intervals) > 0:
                hz = 1.0 / np.mean(intervals)
        
        s2_stats = self.async_s2.stats if self.s2_mode == 'async' else self.sync_s2_stats
        s2_times = self.async_s2.update_times if self.s2_mode == 'async' else self.sync_update_times

        avg_s2 = s2_stats['total_ms'] / s2_stats['s2_updates'] if s2_stats['s2_updates'] > 0 else 0
        s2_model_hz = 1000 / avg_s2 if avg_s2 > 0 else 0
        s2_effective_hz = 0.0
        if len(s2_times) >= 2:
            dt = s2_times[-1] - s2_times[0]
            if dt > 0:
                s2_effective_hz = (len(s2_times) - 1) / dt
        
        return {
            'current_hz': round(hz, 2),
            's2_updates': s2_stats['s2_updates'],
            's2_avg_ms': round(avg_s2, 2),
            's2_hz': round(s2_model_hz, 2),
            's2_effective_hz': round(s2_effective_hz, 2),
            's2_errors': s2_stats['errors'],
            's2_mode': self.s2_mode,
            's2_output_class': self.last_output_class,
            's2_pixel_coord_count': self.s2_output_stats['pixel_coord'],
            's2_traj_latent_count': self.s2_output_stats['traj_latent'],
            's2_discrete_count': self.s2_output_stats['discrete'],
            's2_empty_count': self.s2_output_stats['empty'],
            's2_coord_parse_fail_count': self.s2_output_stats['coord_parse_fail'],
            's2_retry_count': self.s2_output_stats['retry_count'],
            's2_retry_success_count': self.s2_output_stats['retry_success'],
            's2_discrete_mapped_traj_count': self.s2_output_stats['discrete_mapped_traj'],
            'last_llm_output': self.last_llm_output[:200],
            'last_traj_error': self.last_traj_error[:240],
            'navdp_meta_detected': self.navdp_meta_detected,
            'quantization': self.quant_method_applied,
            'tensorrt': self.tensorrt_status,
            'vision_cache': self.vision_cache.get_stats()
        }


agent = None


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, DEFAULT_INSTRUCTION
    
    image = Image.open(request.files['image'].stream).convert('RGB')
    depth = Image.open(request.files['depth'].stream)
    depth = np.asarray(depth).astype(np.float32) / 10000.0
    
    data = json.loads(request.form['json'])

    requested_mode = data.get('s2_mode') if isinstance(data, dict) else None
    if requested_mode in ('async', 'sync'):
        agent.s2_mode = requested_mode

    # Instruction source priority:
    # 1) per-request JSON instruction
    # 2) server default instruction (from CLI/env)
    instruction = data.get('instruction') if isinstance(data, dict) else None
    if not instruction:
        instruction = DEFAULT_INSTRUCTION
    
    if data.get('reset', False):
        idx = 0
        agent.reset()
    
    idx += 1
    
    try:
        result = agent.step(
            np.asarray(image), depth,
            np.eye(4),
            instruction,
            np.eye(4),
            False
        )
    except Exception as e:
        return jsonify({'status': 'error', 'error': type(e).__name__, 'message': str(e)}), 500
    
    output = {}
    if result.output_action is not None:
        output['discrete_action'] = result.output_action
    elif result.output_trajectory is not None:
        output['trajectory'] = result.output_trajectory.tolist()
        if result.output_pixel:
            output['pixel_goal'] = result.output_pixel
    else:
        output['status'] = 'waiting'
    
    return jsonify(output)


@app.route("/stats", methods=['GET'])
def get_stats():
    return jsonify(agent.get_stats())


@app.route("/health", methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': time.time()})


@app.route("/cache/clear", methods=['POST'])
def clear_cache():
    if agent and agent.vision_cache:
        agent.vision_cache.clear()
        return jsonify({'status': 'ok', 'message': 'Cache cleared'})
    return jsonify({'status': 'ok', 'message': 'No cache'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='High-Performance InternVLA Server')
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-w-NavDP")
    parser.add_argument("--cache_size", type=int, default=50, help="Vision cache size")
    parser.add_argument("--use-quantization", action="store_true", help="Enable quantization (alias)")
    parser.add_argument("--no-quantization", action="store_true", help="Disable INT8 quantization")
    parser.add_argument("--quant-method", type=str, default="dynamic", choices=["dynamic", "static", "qat"])
    parser.add_argument("--use-tensorrt", action="store_true", help="Enable TensorRT runtime initialization")
    parser.add_argument("--max-history", type=int, default=2, help="Max historical frames for S2 prompt")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Max generated tokens for S2")
    parser.add_argument("--use-kv-cache", action="store_true", help="Enable KV cache during generation")
    parser.add_argument("--s2-mode", type=str, default="async", choices=["async", "sync"],
                        help="Planner execution mode")
    parser.add_argument("--s2-step-gap", type=int, default=8, help="S1 steps before triggering new S2 plan")
    parser.add_argument("--disable-traj", action="store_true", help="Skip trajectory generation for faster S2 discrete-action mode")
    parser.add_argument("--s2-output-mode", type=str, default="auto", choices=["auto", "legacy", "strict", "trajectory", "force_traj"],
                        help="S2 output policy: auto, or force trajectory generation")
    parser.add_argument("--s2-decode-mode", type=str, default="normal", choices=["normal", "coord_constrained"],
                        help="S2 decode mode for coordinate-biased prompting")
    parser.add_argument("--s2-discrete-to-pixel", action="store_true",
                        help="Map discrete output to a pixel goal and run trajectory head")
    parser.add_argument("--s2-discrete-pixel-step", type=int, default=56,
                        help="Pixel step size used by discrete-to-pixel mapping")
    parser.add_argument("--default-instruction", type=str, default=None,
                        help="Default instruction if request does not provide one")
    parser.add_argument("--s2-prompt-variant", type=str, default="legacy", choices=["legacy", "coord_strict"],
                        help="S2 prompt variant for coordinate-vs-discrete behavior")
    parser.add_argument("--s2-retry-on-discrete", action="store_true",
                        help="If enabled, retry once with strict coordinate format when S2 returns discrete output")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HIGH-PERFORMANCE SERVER ( Optimized) version 1 for REAL-WORLD ROBOT")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Model: {args.model_path}")
    print(f"Cache Size: {args.cache_size}")
    effective_quant = args.use_quantization and not args.no_quantization
    effective_kv_cache = args.use_kv_cache
    print(f"Quantization: {'INT8 (requested)' if effective_quant else 'Disabled/FP16 fallback'}")
    print(f"KV cache: {'enabled' if effective_kv_cache else 'disabled'}")
    print(f"S2 mode: {args.s2_mode}")
    print(f"S2 output mode: {args.s2_output_mode}")
    print(f"S2 decode mode: {args.s2_decode_mode}")
    print(f"S2 prompt variant: {args.s2_prompt_variant}")
    print(f"S2 retry on discrete: {'enabled' if args.s2_retry_on_discrete else 'disabled'}")
    print(f"S2 discrete->pixel: {'enabled' if args.s2_discrete_to_pixel else 'disabled'} (step={args.s2_discrete_pixel_step})")
    print(f"S2 step gap: {args.s2_step_gap}")
    print(f"Trajectory generation: {'disabled' if args.disable_traj else 'enabled'}")

    # Resolve server default instruction from CLI or environment
    env_instruction = os.getenv("INTERNNAV_DEFAULT_INSTRUCTION")
    selected_instruction = args.default_instruction if args.default_instruction else env_instruction
    if selected_instruction:
        DEFAULT_INSTRUCTION = selected_instruction
    print(f"Default instruction: {DEFAULT_INSTRUCTION}")
    print("=" * 70)
    
    agent = OptimizedAgent(
        args.model_path,
        args.device,
        args.cache_size,
        use_quantization=effective_quant,
        quant_method=args.quant_method,
        use_tensorrt=args.use_tensorrt,
        max_history=args.max_history,
        max_new_tokens=args.max_new_tokens,
        use_kv_cache=effective_kv_cache,
        s2_mode=args.s2_mode,
        s2_step_gap=args.s2_step_gap,
        disable_traj=args.disable_traj,
        s2_output_mode=args.s2_output_mode,
        s2_decode_mode=args.s2_decode_mode,
        s2_discrete_to_pixel=args.s2_discrete_to_pixel,
        s2_discrete_pixel_step=args.s2_discrete_pixel_step,
    )
    agent.s2_prompt_variant = args.s2_prompt_variant
    agent.s2_retry_on_discrete = args.s2_retry_on_discrete
    agent.reset()
    
    print(f"[Server] Starting on 0.0.0.0:5802")
    print(f"[Server] Stats: http://localhost:5802/stats")
    print(f"[Server] Health: http://localhost:5802/health")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5802, threaded=True)
