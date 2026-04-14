"""
Ultra-High-Performance InternVLA Agent

Optimized for maximum Hz (50+ Hz):
- S2 (planner) runs async at maximum speed
- S1 (controller) returns immediately with cached/async results
- No rate limiting - runs as fast as possible
- Uses cached S2 outputs until new ones arrive

Usage:
    agent = UltraHighPerformanceAgent(args)
    result = agent.step(rgb, depth, pose, instruction, intrinsic)
"""

import copy
import itertools
import os
import re
import sys
import time
import threading
import queue
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent.parent))
ROOT = Path(__file__).resolve().parents[2]

from PIL import Image
from transformers import AutoProcessor

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM, InternVLAN1ModelConfig
from internnav.model.utils.vln_utils import S2Output, split_and_clean, traj_to_actions


DEFAULT_IMAGE_TOKEN = "<image>"


def get_best_attention_implementation() -> str:
    """Auto-select the best available attention implementation."""
    try:
        import flash_attn
        print(f"[Attention] Flash Attention {flash_attn.__version__} (FASTEST)")
        return "flash_attention_2"
    except (ImportError, OSError):
        print(f"[Attention] Using SDPA")
        return "sdpa"


@dataclass
class UltraPerformanceStats:
    """Ultra performance statistics tracking."""
    s2_count: int = 0
    s1_count: int = 0
    s2_total_ms: float = 0.0
    s1_total_ms: float = 0.0
    frame_times: List[float] = field(default_factory=list)
    request_count: int = 0
    
    @property
    def avg_s2_ms(self) -> float:
        return self.s2_total_ms / self.s2_count if self.s2_count > 0 else 0
    
    @property
    def avg_s1_ms(self) -> float:
        return self.s1_total_ms / self.s1_count if self.s1_count > 0 else 0
    
    @property
    def current_hz(self) -> float:
        if len(self.frame_times) < 2:
            return 0
        recent = self.frame_times[-20:] if len(self.frame_times) >= 20 else self.frame_times
        intervals = np.diff(recent)
        if len(intervals) == 0 or np.mean(intervals) == 0:
            return 0
        return 1.0 / np.mean(intervals)


class UltraAsyncS2Processor:
    """
    Ultra-fast async S2 processor.
    
    - No rate limiting - runs S2 as fast as possible
    - Double buffering for seamless output updates
    - Non-blocking reads for maximum throughput
    """
    
    def __init__(self, step_func: Callable, device: torch.device):
        self.step_func = step_func
        self.device = device
        self.command_queue: queue.Queue = queue.Queue(maxsize=4)
        self.result_buffer = [None, None]
        self.write_idx = 0
        self.read_idx = 1
        self.lock = threading.RLock()
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.stats = {'s2_updates': 0, 's2_total_ms': 0, 'dropped': 0}
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print(f"[UltraAsyncS2] Started (no rate limit)")
    
    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
        print("[UltraAsyncS2] Stopped")
    
    def request_s2(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        """Request S2 (non-blocking)."""
        try:
            self.command_queue.put_nowait((rgb, depth, pose, instruction, intrinsic, look_down))
        except queue.Full:
            self.stats['dropped'] += 1
            try:
                self.command_queue.get_nowait()
                self.command_queue.put_nowait((rgb, depth, pose, instruction, intrinsic, look_down))
            except queue.Full:
                pass
    
    def get_latest_result(self) -> Tuple:
        """Get latest S2 result (non-blocking). Returns (action, latent, pixel, valid)."""
        with self.lock:
            if self.result_buffer[self.read_idx] is not None:
                return self.result_buffer[self.read_idx]
            return None, None, None, False
    
    def _worker_loop(self):
        """Worker loop - runs S2 as fast as possible."""
        while self.running:
            try:
                cmd = self.command_queue.get(timeout=0.01)
                rgb, depth, pose, instruction, intrinsic, look_down = cmd
                
                t_start = time.time()
                action, latent, pixel = self.step_func(rgb, depth, pose, instruction, intrinsic, look_down)
                latency_ms = (time.time() - t_start) * 1000
                
                with self.lock:
                    self.write_idx = (self.write_idx + 1) % 2
                    self.read_idx = (self.read_idx + 1) % 2
                    self.result_buffer[self.write_idx] = (action, latent, pixel, True)
                    
                    self.stats['s2_updates'] += 1
                    self.stats['s2_total_ms'] += latency_ms
                    
                    if self.stats['s2_updates'] % 5 == 0:
                        print(f"[UltraAsyncS2] S2: {latency_ms:.0f}ms | Avg: {self.stats['s2_total_ms']/self.stats['s2_updates']:.0f}ms")
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[UltraAsyncS2] Error: {e}")


class UltraHighPerformanceAgent:
    """
    Ultra-high-performance InternVLA agent.
    
    Optimized for 50+ Hz:
    - S2 runs async without rate limiting
    - S1 returns immediately with latest cached S2 result
    - No blocking - maximum throughput
    """
    
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.save_dir = ROOT / "test_data" / datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)
        
        print("=" * 60)
        print("ULTRA-HIGH-PERFORMANCE InternVLA AGENT")
        print("=" * 60)
        print(f"[Config] Device: {self.device}")
        
        self.config = InternVLAN1ModelConfig.from_pretrained(args.model_path)
        self.attn_impl = get_best_attention_implementation()
        
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_path,
            config=self.config,
            torch_dtype=torch.float16,
            attn_implementation=self.attn_impl,
            device_map={"": self.device},
        )
        self.model.eval()
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.processor.tokenizer.padding_side = 'left'
        
        self.resize_w = args.resize_w
        self.resize_h = args.resize_h
        self.num_history = args.num_history
        self.PLAN_STEP_GAP = args.plan_step_gap
        
        self.async_processor = UltraAsyncS2Processor(
            step_func=self._step_s2_impl,
            device=self.device
        )
        self.async_processor.start()
        
        self.prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next? Please output the next waypoint's coordinates in the image. Please output STOP when done."
        self.conversation = [{"from": "human", "value": self.prompt}, {"from": "gpt", "value": ""}]
        self.conjunctions = ['you can see ', 'in front of you is ', 'there is ']
        
        self.actions2idx = OrderedDict({
            'STOP': [0], "↑": [1], "←": [2], "→": [3], "↓": [5],
        })
        
        self._init_state()
        self.stats = UltraPerformanceStats()
        
        print(f"[Agent] Optimizations: FP16 + {self.attn_impl}")
        print(f"[Agent] Ultra Async S2: enabled (no rate limit)")
        print("=" * 60)
    
    def _init_state(self):
        """Initialize agent state."""
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
        
        self.cached_action = None
        self.cached_latent = None
        self.cached_pixel = None
        self.cached_valid = False
    
    def reset(self):
        """Reset agent for new episode."""
        self._init_state()
        if hasattr(self, 'async_processor'):
            self.async_processor.stop()
            self.async_processor.start()
        self.stats = UltraPerformanceStats()
        self.save_dir = ROOT / "test_data" / datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)
        print("[Agent] Reset complete")
    
    def parse_actions(self, output: str) -> List[int]:
        """Parse action symbols from LLM output."""
        action_patterns = '|'.join(re.escape(a) for a in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[m] for m in matches]
        return list(itertools.chain.from_iterable(actions))
    
    def step(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        instruction: str,
        intrinsic: np.ndarray,
        look_down: bool = False
    ) -> S2Output:
        """
        Main step function - ultra fast, runs at maximum Hz.
        
        Returns cached S2 result immediately if available.
        """
        frame_start = time.time()
        dual_sys_output = S2Output()
        
        need_s2 = (self.episode_idx - self.last_s2_idx > self.PLAN_STEP_GAP) or look_down or not self.cached_valid
        
        if need_s2:
            self.async_processor.request_s2(
                rgb, depth, pose, instruction, intrinsic, look_down
            )
            self.last_s2_idx = self.episode_idx
        
        action, latent, pixel, valid = self.async_processor.get_latest_result()
        if valid:
            self.cached_action = action
            self.cached_latent = latent
            self.cached_pixel = pixel
            self.cached_valid = True
            self.stats.s2_count += 1
        
        if self.cached_action is not None:
            dual_sys_output.output_action = copy.deepcopy(self.cached_action)
            self.cached_action = None
        elif self.cached_latent is not None:
            t_s1 = time.time()
            trajectories = self._step_s1(rgb, depth)
            dual_sys_output.output_trajectory = traj_to_actions(trajectories, use_discrate_action=False)
            self.stats.s1_total_ms += (time.time() - t_s1) * 1000
            self.stats.s1_count += 1
        
        if dual_sys_output.output_pixel is None and self.cached_pixel is not None:
            dual_sys_output.output_pixel = self.cached_pixel
            self.pixel_goal_rgb = copy.deepcopy(rgb)
            self.pixel_goal_depth = copy.deepcopy(depth)
        
        if not look_down:
            image = Image.fromarray(rgb).convert('RGB')
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
            self.episode_idx += 1
        
        self.stats.frame_times.append(time.time())
        if len(self.stats.frame_times) > 100:
            self.stats.frame_times.pop(0)
        
        self.stats.request_count += 1
        
        frame_time = (time.time() - frame_start) * 1000
        if self.stats.request_count % 30 == 0:
            print(f"[Agent] Frame: {frame_time:.1f}ms | Hz: {self.stats.current_hz:.1f}")
        
        return dual_sys_output
    
    def _step_s2_impl(self, rgb, depth, pose, instruction, intrinsic, look_down=False) -> Tuple:
        """S2 implementation - synchronous call for async processor."""
        image = Image.fromarray(rgb).convert('RGB')
        
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
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
                sources[0]["value"] += f' Historical observations: {placeholder}.'
            
            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
        else:
            self.input_images.append(image)
            input_img_id = -1
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append({'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]})
        
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
        
        text = self.processor.apply_chat_template(
            self.conversation_history, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt")
        inputs = {k: v for k, v in inputs.items() if k != 'mm_token_type_ids'}
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
            )
        
        output_ids = outputs.sequences
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )
        
        if bool(re.search(r'\d', self.llm_output)):
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            
            with torch.no_grad():
                traj_latents = self.model.generate_latents(output_ids, inputs['pixel_values'], image_grid_thw)
            return None, traj_latents, pixel_goal
        else:
            action_seq = self.parse_actions(self.llm_output)
            return action_seq, None, None
    
    def _step_s1(self, rgb: np.ndarray, depth: np.ndarray) -> torch.Tensor:
        """S1 implementation - low-level control."""
        if self.pixel_goal_rgb is None:
            return torch.zeros(1, 8, 3)
        
        processed_rgb = np.array(Image.fromarray(self.pixel_goal_rgb).resize((224, 224))) / 255
        processed_depth = np.array(Image.fromarray(self.pixel_goal_depth).resize((224, 224)))
        curr_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
        curr_depth = np.array(Image.fromarray(depth).resize((224, 224)))
        
        rgbs = torch.stack([torch.from_numpy(processed_rgb), torch.from_numpy(curr_rgb)]).unsqueeze(0).to(self.device)
        depths = torch.stack([torch.from_numpy(processed_depth), torch.from_numpy(curr_depth)]).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        return self.model.generate_traj(self.cached_latent, rgbs, depths)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            's2_avg_ms': self.stats.avg_s2_ms,
            's1_avg_ms': self.stats.avg_s1_ms,
            'current_hz': self.stats.current_hz,
            's2_count': self.stats.s2_count,
            's1_count': self.stats.s1_count,
            'request_count': self.stats.request_count,
            'async_s2_stats': self.async_processor.stats if hasattr(self, 'async_processor') else {}
        }
    
    def start(self):
        if hasattr(self, 'async_processor'):
            self.async_processor.start()
    
    def stop(self):
        if hasattr(self, 'async_processor'):
            self.async_processor.stop()


def create_ultra_high_performance_agent(
    model_path: str,
    device: str = 'cuda:0',
    resize: int = 224,
    **kwargs
) -> UltraHighPerformanceAgent:
    """Factory function to create ultra-optimized agent."""
    
    class Args:
        def __init__(self):
            self.model_path = model_path
            self.device = device
            self.resize_w = resize
            self.resize_h = resize
            self.num_history = 8
            self.plan_step_gap = 4
    
    agent = UltraHighPerformanceAgent(Args())
    return agent
