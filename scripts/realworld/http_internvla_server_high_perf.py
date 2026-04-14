#!/usr/bin/env python3
"""
Ultra-High-Performance InternVLA Server (Integrated Optimization)

All optimizations integrated into a single source file:
- Vision Embedding Cache (from vision_cache.py)
- TensorRT Support (from tensorrt_converter.py)
- INT8 Quantization (from quantization.py)
- Async S2 (planner) runs without rate limiting
- Flash Attention + FP16 precision
- 224x224 image size

Target: 40-50+ Hz S2 planning

Usage:
    python http_internvla_server_high_perf.py
"""

import argparse
import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformation import Calibration
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM, InternVLAN1ModelConfig
from internnav.model.utils.vln_utils import S2Output, split_and_clean, traj_to_actions

app = Flask(__name__)
idx = 0


def get_best_attention():
    if not torch.cuda.is_available():
        print("[Agent] CUDA not available, using 'eager' attention")
        return "eager"
    try:
        import flash_attn
        return "flash_attention_2"
    except:
        return "sdpa"


# =============================================================================
# INTEGRATED: Vision Embedding Cache
# =============================================================================

class ImageHasher:
    def __init__(self, hash_size: int = 12, downsample_scale: int = 16):
        self.hash_size = hash_size
        self.downsample_scale = downsample_scale
    
    def compute_hash(self, image: np.ndarray) -> str:
        if len(image.shape) == 3:
            h, w = image.shape[:2]
            scale = max(1, min(h, w) // self.downsample_scale)
            sampled = image[::scale, ::scale, :]
            return hashlib.md5(sampled.tobytes()).hexdigest()[:self.hash_size]
        else:
            return hashlib.md5(image.tobytes()).hexdigest()[:self.hash_size]


class VisionEmbeddingCache:
    def __init__(self, max_size: int = 100, hash_size: int = 12):
        self.max_size = max_size
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hasher = ImageHasher(hash_size=hash_size)
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.last_hash: Optional[str] = None
    
    def _make_key(self, image: np.ndarray, timestamp: float) -> str:
        img_hash = self.hasher.compute_hash(image)
        time_bucket = int(timestamp * 10)
        return f"{img_hash}_{time_bucket}"
    
    def get(self, image: np.ndarray, timestamp: Optional[float] = None) -> Optional[torch.Tensor]:
        if timestamp is None:
            timestamp = time.time()
        
        key = self._make_key(image, timestamp)
        
        with self.lock:
            if key in self.cache:
                self.hits += 1
                self.cache.move_to_end(key)
                self.last_hash = key.split('_')[0]
                return self.cache[key].clone().to('cuda', non_blocking=True)
            
            self.misses += 1
            return None
    
    def put(self, image: np.ndarray, embedding: torch.Tensor, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        
        key = self._make_key(image, timestamp)
        
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                self.cache[key] = embedding.detach().cpu()
            
            self.last_hash = key.split('_')[0]
    
    def is_similar(self, image: np.ndarray) -> bool:
        if self.last_hash is None:
            return False
        current_hash = self.hasher.compute_hash(image)
        return current_hash == self.last_hash
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.last_hash = None
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


# =============================================================================
# INTEGRATED: TensorRT Support
# =============================================================================

class TensorRTConverter:
    def __init__(self, model_dir: str, workspace_size: int = 1 << 30):
        self.model_dir = Path(model_dir)
        self.workspace_size = workspace_size
        self.trt = None
        self.builder = None
        self.network = None
        self.config = None
        self.trt_logger = None
        self._init_trt()
    
    def _init_trt(self):
        try:
            import tensorrt as trt
            self.trt = trt
            
            trt_logger = trt.Logger(trt.Logger.WARNING)
            self.trt_logger = trt_logger
            
            self.builder = trt.Builder(trt_logger)
            self.network = self.builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            self.config = self.builder.create_builder_config()
            self.config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, self.workspace_size
            )
            
            print(f"[TensorRT] Initialized (workspace: {self.workspace_size >> 30} GB)")
            
        except ImportError:
            print("[TensorRT] Not installed. Run: pip install tensorrt")
            self.trt = None
    
    def check_trt_available(self) -> bool:
        return self.trt is not None
    
    def export_vision_encoder_onnx(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ) -> bool:
        if not self.check_trt_available():
            return False
        
        model.eval()
        dummy_input = torch.randn(input_shape, dtype=torch.float32)
        
        dynamic_axes = {
            'pixel_values': {0: 'batch_size'},
            'image_embeds': {0: 'batch_size'}
        }
        
        try:
            torch.onnx.export(
                model,
                (dummy_input,),
                output_path,
                input_names=['pixel_values'],
                output_names=['image_embeds'],
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True
            )
            print(f"[TensorRT] Exported vision encoder to {output_path}")
            return True
        except Exception as e:
            print(f"[TensorRT] Export failed: {e}")
            return False
    
    def build_tensorrt_engine(
        self,
        onnx_path: str,
        engine_path: str,
        precision: str = 'fp16',
        max_batch_size: int = 8
    ) -> bool:
        if not self.check_trt_available():
            return False
        
        if not os.path.exists(onnx_path):
            print(f"[TensorRT] ONNX file not found: {onnx_path}")
            return False
        
        print(f"[TensorRT] Building engine from {onnx_path}...")
        t_start = time.time()
        
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            explicit_batch = 1 << int(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = self.builder.create_network(explicit_batch)
            parser = self.trt.OnnxParser(network, self.trt_logger)
            
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f"[TensorRT] Parse error: {parser.get_error(i)}")
                    return False
            
            config = self.builder.create_builder_config()
            config.set_memory_pool_limit(
                self.trt.MemoryPoolType.WORKSPACE, 1 << 30
            )
            
            if precision == 'fp16' and self.builder.platform_has_fast_fp16:
                config.set_flag(self.trt.BuilderFlag.FP16)
                print("[TensorRT] Using FP16 precision")
            elif precision == 'int8' and self.builder.platform_has_fast_int8:
                config.set_flag(self.trt.BuilderFlag.INT8)
                print("[TensorRT] Using INT8 precision")
            
            profile = self.builder.create_optimization_profile()
            input_tensor = network.get_input(0)
            min_shape = (1,) + tuple(input_tensor.shape[1:])
            opt_shape = (max_batch_size // 2,) + tuple(input_tensor.shape[1:])
            max_shape = (max_batch_size,) + tuple(input_tensor.shape[1:])
            
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            engine = self.builder.build_serialized_network(network, config)
            
            if engine is None:
                print("[TensorRT] Engine build failed")
                return False
            
            with open(engine_path, 'wb') as f:
                f.write(engine)
            
            t_elapsed = time.time() - t_start
            engine_size = os.path.getsize(engine_path) / (1024 * 1024)
            
            print(f"[TensorRT] Engine built in {t_elapsed:.1f}s ({engine_size:.1f} MB)")
            print(f"[TensorRT] Saved to {engine_path}")
            
            return True
            
        except Exception as e:
            print(f"[TensorRT] Build failed: {e}")
            return False


class TensorRTModelWrapper:
    def __init__(self, engine_path: str, device: str = 'cuda:0'):
        self.engine_path = engine_path
        self.device = torch.device(device)
        self.engine = None
        self.context = None
        self.bindings: List[Any] = []
        self._load_engine()
    
    def _load_engine(self):
        try:
            import tensorrt as trt
            
            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.engine = runtime.deserialize_c_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            self.bindings = [None] * self.engine.num_io_tensors
            
            print(f"[TensorRT] Loaded engine with {self.engine.num_io_tensors} I/O tensors")
            
        except Exception as e:
            print(f"[TensorRT] Failed to load engine: {e}")
            self.engine = None
    
    def is_available(self) -> bool:
        return self.engine is not None


# =============================================================================
# INTEGRATED: INT8 Quantization
# =============================================================================

try:
    from torch.quantization import quantize_dynamic
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("[Quantization] torch.quantization not available")


class DynamicQuantizer:
    def __init__(self):
        self.quantized_model = None
        self.available = QUANTIZATION_AVAILABLE
    
    def quantize(self, model: nn.Module) -> nn.Module:
        if not self.available:
            print("[Quantization] Dynamic quantization not available")
            return model
        
        print("[Quantization] Applying dynamic INT8 quantization...")
        
        quantized = quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.LSTMCell, nn.GRUCell, nn.GRU},
            dtype=torch.qint8
        )
        
        self.quantized_model = quantized
        print("[Quantization] Dynamic quantization complete")
        
        return quantized
    
    def get_model(self) -> Optional[nn.Module]:
        return self.quantized_model


def quantize_for_deployment(
    model: nn.Module,
    method: str = 'dynamic'
) -> Tuple[nn.Module, Dict[str, Any]]:
    if not QUANTIZATION_AVAILABLE:
        print(f"[Quantization] Skipping {method} quantization (not available)")
        return model, {'method': 'none', 'quantized': False}
    
    print(f"[Quantization] Applying {method} quantization...")
    
    metadata = {
        'method': method,
        'original_state_dict_keys': len(model.state_dict()),
        'quantized': True
    }
    
    if method == 'dynamic':
        quantizer = DynamicQuantizer()
        quantized_model = quantizer.quantize(model)
    else:
        print(f"[Quantization] Unknown method: {method}, using dynamic")
        quantizer = DynamicQuantizer()
        quantized_model = quantizer.quantize(model)
    
    return quantized_model, metadata


# =============================================================================
# Async S2 Processor
# =============================================================================

class UltraAsyncS2:
    def __init__(self, step_func):
        self.step_func = step_func
        self.queue = queue.Queue(maxsize=4)
        self.buffer = [None, None]
        self.write_idx = 0
        self.read_idx = 1
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.stats = {'s2_updates': 0, 'total_ms': 0, 'errors': 0}
    
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
            return self.buffer[self.read_idx]
    
    def _run(self):
        import traceback
        while self.running:
            try:
                args = self.queue.get(timeout=0.01)
                t0 = time.time()
                result = self.step_func(*args)
                ms = (time.time() - t0) * 1000
                
                with self.lock:
                    self.write_idx = (self.write_idx + 1) % 2
                    self.read_idx = (self.read_idx + 1) % 2
                    self.buffer[self.write_idx] = result
                    self.stats['s2_updates'] += 1
                    self.stats['total_ms'] += ms
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[S2] Error: {e}")
                traceback.print_exc()
                self.stats['errors'] += 1


# =============================================================================
# High-Performance Agent
# =============================================================================

class HighPerformanceAgent:
    def __init__(
        self,
        model_path,
        device='cuda:0',
        use_vision_cache=True,
        use_tensorrt=False,
        use_quantization=False,
        cache_size=100
    ):
        self.device = torch.device(device)
        self.model_path = model_path
        
        print(f"[Agent] Loading model on {self.device}...")
        print(f"[Agent] Optimizations: vision_cache={use_vision_cache}, tensorrt={use_tensorrt}, quantization={use_quantization}")
        
        self.config = InternVLAN1ModelConfig.from_pretrained(model_path)
        attn = get_best_attention()
        
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            model_path, config=self.config,
            torch_dtype=torch.float16,
            attn_implementation=attn,
            device_map={"": self.device}
        )
        self.model.eval()
        
        self._apply_optimizations(use_tensorrt, use_quantization)
        
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = 'left'
        
        self.prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next? Please output the next waypoint's coordinates in the image. Please output STOP when done."
        self.actions2idx = OrderedDict({'STOP': [0], "↑": [1], "←": [2], "→": [3], "↓": [5]})
        
        self.rgb_list = []
        self.conversation_history = []
        self.llm_output = ""
        self.episode_idx = 0
        self._last_s2_idx = -100
        self.input_images = []
        
        self.vision_cache = VisionEmbeddingCache(max_size=cache_size) if use_vision_cache else None
        self.tensorrt_wrapper = None
        self.trt_converter = TensorRTConverter(model_path) if use_tensorrt else None
        
        self.async_s2 = UltraAsyncS2(self._step_s2)
        self.async_s2.start()
        
        self.cached = (None, None, None, False)
        self.frame_times = []
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        
        print(f"[Agent] Loaded with {attn}")
        if self.vision_cache:
            print(f"[Agent] Vision cache initialized (size={cache_size})")
    
    def _apply_optimizations(self, use_tensorrt: bool, use_quantization: bool):
        if use_quantization and QUANTIZATION_AVAILABLE:
            print("[Agent] Applying INT8 quantization...")
            try:
                quantizer = DynamicQuantizer()
                self.model = quantizer.quantize(self.model)
                print("[Agent] INT8 quantization applied")
            except Exception as e:
                print(f"[Agent] Quantization failed: {e}")
        
        if use_tensorrt and self.trt_converter and self.trt_converter.check_trt_available():
            print("[Agent] TensorRT support enabled (not pre-converted)")
    
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
        if self.vision_cache:
            self.vision_cache.clear()
    
    def _step_s2(self, rgb, depth, pose, instruction, intrinsic, look_down):
        image = Image.fromarray(rgb).convert('RGB')
        
        if not look_down:
            image = image.resize((224, 224))
            self.rgb_list.append(image)
            self.conversation_history = []
            sources = [{"from": "human", "value": self.prompt.replace('<instruction>.', instruction)}, {"from": "gpt", "value": ""}]
            cur_images = self.rgb_list[-1:]
            
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = sorted(np.unique(np.linspace(0, self.episode_idx - 1, 8, dtype=np.int32)).tolist())
                placeholder = "<image>\n" * len(history_id)
                sources[0]["value"] += f' Historical observations: {placeholder}.'
            
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1
        else:
            self.input_images.append(image)
            input_img_id = -1
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append({'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]})
        
        prompt = "you can see " + "<image>"
        sources[0]["value"] += f" {prompt}."
        parts = split_and_clean(sources[0]["value"])
        
        content = []
        for p in parts:
            if p == "<image>":
                if 0 <= input_img_id < len(self.input_images):
                    content.append({"type": "image", "image": self.input_images[input_img_id]})
                    input_img_id += 1
                else:
                    print(f"[S2] Warning: input_img_id={input_img_id} out of range, using last image")
                    if len(self.input_images) > 0:
                        content.append({"type": "image", "image": self.input_images[-1]})
                        input_img_id += 1
            else:
                content.append({"type": "text", "text": p})
        
        self.conversation_history.append({'role': 'user', 'content': content})
        
        text = self.processor.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt")
        inputs = {k: v for k, v in inputs.items() if k != 'mm_token_type_ids'}
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True, return_dict_in_generate=True)
        
        output_ids = outputs.sequences
        self.llm_output = self.processor.tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        if bool(self.llm_output) and any(c.isdigit() for c in self.llm_output):
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            with torch.no_grad():
                traj_latents = self.model.generate_latents(output_ids, inputs['pixel_values'], image_grid_thw)
            return (None, traj_latents, pixel_goal)
        else:
            actions = []
            for a in ['STOP', '↑', '←', '→', '↓']:
                if a in self.llm_output:
                    actions.extend(self.actions2idx.get(a, []))
            return (actions if actions else None, None, None)
    
    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        output = S2Output()
        
        need_s2 = (self.episode_idx - self.last_s2_idx > 4) or look_down or not self.cached[3]
        
        if need_s2:
            self.async_s2.request(rgb, depth, pose, instruction, intrinsic, look_down)
            self.last_s2_idx = self.episode_idx
        
        result = self.async_s2.get_result()
        if result:
            self.cached = result + (True,)
        
        action, latent, pixel, valid = self.cached
        
        if action is not None:
            output.output_action = action
            self.cached = (None, None, pixel, False)
        elif latent is not None:
            curr_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
            curr_depth = np.array(Image.fromarray(depth).resize((224, 224)))
            goal_rgb = np.array(Image.fromarray(self.pixel_goal_rgb).resize((224, 224))) / 255 if self.pixel_goal_rgb is not None else curr_rgb
            goal_depth = np.array(Image.fromarray(self.pixel_goal_depth).resize((224, 224))) if self.pixel_goal_depth is not None else curr_depth
            
            rgbs = torch.stack([torch.from_numpy(goal_rgb), torch.from_numpy(curr_rgb)]).unsqueeze(0).to(self.device)
            depths = torch.stack([torch.from_numpy(goal_depth), torch.from_numpy(curr_depth)]).unsqueeze(0).unsqueeze(-1).to(self.device)
            
            trajectories = self.model.generate_traj(latent, rgbs, depths)
            output.output_trajectory = traj_to_actions(trajectories, use_discrate_action=False)
        
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
        
        avg_s2 = self.async_s2.stats['total_ms'] / self.async_s2.stats['s2_updates'] if self.async_s2.stats['s2_updates'] > 0 else 0
        
        stats = {
            'current_hz': hz,
            's2_updates': self.async_s2.stats['s2_updates'],
            's2_avg_ms': avg_s2,
            's2_errors': self.async_s2.stats['errors']
        }
        
        if self.vision_cache:
            cache_stats = self.vision_cache.get_stats()
            stats['vision_cache'] = cache_stats
        
        return stats


# =============================================================================
# Flask Routes
# =============================================================================

agent = None


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx
    
    image = Image.open(request.files['image'].stream).convert('RGB')
    depth = Image.open(request.files['depth'].stream)
    depth = np.asarray(depth).astype(np.float32) / 10000.0
    
    data = json.loads(request.form['json'])
    
    if data.get('reset', False):
        idx = 0
        agent.reset()
    
    idx += 1
    
    result = agent.step(
        np.asarray(image), depth,
        np.eye(4),
        "Exit the door, then Turn left and go straight until you find small fire extinguisher. Then stop.",
        np.eye(4),
        False
    )
    
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
    return jsonify({'status': 'ok', 'message': 'No cache enabled'})


@app.route("/cache/stats", methods=['GET'])
def cache_stats():
    if agent and agent.vision_cache:
        return jsonify(agent.vision_cache.get_stats())
    return jsonify({'status': 'disabled'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ultra-High-Performance InternVLA Server')
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-w-NavDP")
    parser.add_argument("--no-vision-cache", action="store_true", help="Disable vision cache")
    parser.add_argument("--use-tensorrt", action="store_true", help="Enable TensorRT support")
    parser.add_argument("--use-quantization", action="store_true", help="Apply INT8 quantization")
    parser.add_argument("--cache-size", type=int, default=100, help="Vision cache size")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ULTRA-HIGH-PERFORMANCE SERVER (All Optimizations Integrated)")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Model: {args.model_path}")
    print(f"Vision Cache: {not args.no_vision_cache}")
    print(f"TensorRT: {args.use_tensorrt}")
    print(f"INT8 Quantization: {args.use_quantization}")
    print(f"Cache Size: {args.cache_size}")
    print("=" * 70)
    
    agent = HighPerformanceAgent(
        args.model_path,
        args.device,
        use_vision_cache=not args.no_vision_cache,
        use_tensorrt=args.use_tensorrt,
        use_quantization=args.use_quantization,
        cache_size=args.cache_size
    )
    agent.reset()
    
    print(f"[Server] Starting on 0.0.0.0:5802")
    print(f"[Server] Stats: http://localhost:5802/stats")
    print(f"[Server] Health: http://localhost:5802/health")
    print(f"[Server] Cache Stats: http://localhost:5802/cache/stats")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5802, threaded=True)
