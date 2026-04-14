#!/usr/bin/env python3
"""
Comprehensive Optimization Benchmark for InternVLA

This script benchmarks various optimization strategies to identify
the best approach for achieving 40-50 Hz real-time performance.

Usage:
    python benchmark_optimizations.py
"""

import argparse
import time
import torch
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'src/diffusion-policy'))

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from transformers import AutoProcessor
from PIL import Image
import numpy as np
import hashlib


def get_image_hash(img_array):
    """Get hash of image for caching simulation."""
    return hashlib.md5(img_array.tobytes()).hexdigest()[:16]


class BenchmarkRunner:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = torch.device('cuda:0')
        
    def load_model(self, torch_dtype=torch.float16, attn='sdpa'):
        """Load model with specified optimizations."""
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn,
            device_map={'': self.device},
        )
        self.model.eval()
        self.model.model.navdp = self.model.model.navdp.to(torch_dtype)
        
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=False)
        self.processor.tokenizer.padding_side = 'left'
        
    def prepare_inputs(self, image_size):
        """Prepare inputs for given image size."""
        img_array = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        image = Image.fromarray(img_array).convert('RGB')
        
        content = [{'type': 'image', 'image': image}, {'type': 'text', 'text': 'Go forward'}]
        text = self.processor.apply_chat_template(
            [{'role': 'user', 'content': content}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        return image, inputs, img_array
    
    def benchmark(self, inputs, num_iterations=10, max_tokens=50):
        """Run benchmark and return average time."""
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
        
        return sum(times) / len(times)
    
    def profile_components(self, inputs):
        """Profile individual components."""
        # Vision encoding
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = self.model.visual(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
        torch.cuda.synchronize()
        vision_time = (time.time() - t0) * 1000
        
        # Full generation
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
        torch.cuda.synchronize()
        total_time = (time.time() - t0) * 1000
        
        return vision_time, total_time


def run_benchmark():
    print("=" * 70)
    print("INTERNVLA OPTIMIZATION BENCHMARK")
    print("=" * 70)
    print()
    
    runner = BenchmarkRunner('checkpoints/InternVLA-N1-w-NavDP')
    
    # Test 1: Different precisions
    print("TEST 1: Precision Comparison")
    print("-" * 50)
    
    precisions = [
        ('FP16 + SDPA', torch.float16, 'sdpa'),
        ('BF16 + SDPA', torch.bfloat16, 'sdpa'),
    ]
    
    results = []
    for name, dtype, attn in precisions:
        print(f"  Testing {name}...", end=' ', flush=True)
        runner.load_model(torch_dtype=dtype, attn=attn)
        _, inputs, _ = runner.prepare_inputs(448)
        avg_time = runner.benchmark(inputs)
        results.append((name, avg_time))
        print(f"{avg_time:.1f}ms ({1000/avg_time:.1f} Hz)")
    
    best_precision = min(results, key=lambda x: x[1])
    print(f"\n  Best precision: {best_precision[0]} ({best_precision[1]:.1f}ms)")
    
    # Test 2: Image size optimization
    print("\nTEST 2: Image Size Optimization")
    print("-" * 50)
    
    # Reload with best precision
    runner.load_model(torch_dtype=torch.float16, attn='sdpa')
    
    sizes = [224, 336, 448]
    size_results = []
    for size in sizes:
        print(f"  Testing {size}x{size}...", end=' ', flush=True)
        _, inputs, _ = runner.prepare_inputs(size)
        avg_time = runner.benchmark(inputs)
        size_results.append((size, avg_time))
        print(f"{avg_time:.1f}ms ({1000/avg_time:.1f} Hz)")
    
    best_size = min(size_results, key=lambda x: x[1])
    print(f"\n  Best size: {best_size[0]}x{best_size[0]} ({best_size[1]:.1f}ms)")
    
    # Test 3: Component profiling
    print("\nTEST 3: Component Profiling (448x448)")
    print("-" * 50)
    
    _, inputs, _ = runner.prepare_inputs(448)
    vision_time, total_time = runner.profile_components(inputs)
    print(f"  Vision encoding: {vision_time:.1f}ms ({100*vision_time/total_time:.0f}%)")
    print(f"  Total generation: {total_time:.1f}ms (100%)")
    print(f"  LLM overhead: {total_time - vision_time:.1f}ms ({100*(total_time-vision_time)/total_time:.0f}%)")
    
    # Test 4: Vision caching benefit
    print("\nTEST 4: Vision Caching Simulation")
    print("-" * 50)
    
    # Simulate navigation with repeated frames
    _, inputs1, _ = runner.prepare_inputs(224)
    _, inputs2, _ = runner.prepare_inputs(224)
    
    # First frame (no cache)
    t1 = runner.benchmark(inputs1, num_iterations=5)
    
    # Simulate consecutive same-frame inference
    # In real scenario, only LLM would need to run
    print(f"  With vision: {t1:.1f}ms")
    print(f"  If vision cached: ~{(t1 - 100):.1f}ms (estimated)")
    print(f"  LLM-only overhead: ~{(t1 - 100)/50:.1f}ms per token")
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\nBest single-pass performance: {best_precision[1]:.1f}ms ({1000/best_precision[1]:.1f} Hz)")
    print(f"Best image size: {best_size[0]}x{best_size[0]} ({best_size[1]:.1f}ms)")
    
    print("\nOPTIMIZATION ROADMAP:")
    print("-" * 50)
    print("Phase 1 (Current):")
    print(f"  - FP16 + SDPA: {best_precision[1]:.1f}ms ({1000/best_precision[1]:.1f} Hz)")
    print(f"  - 224x224 images: {min([r[1] for r in size_results if r[0] == 224]):.1f}ms")
    
    print("\nPhase 2 (Needed for 40-50 Hz):")
    print("  - TensorRT for vision encoder: ~2-3x speedup")
    print("  - INT8 quantization: ~1.5-2x speedup")
    print("  - Flash Attention (when available): ~1.5x speedup")
    
    target_ms = 22.5  # Target for 44 Hz
    current_best = best_size[1]
    needed_speedup = current_best / target_ms
    
    print(f"\nTo reach {target_ms:.0f}ms target:")
    print(f"  Current: {current_best:.1f}ms")
    print(f"  Needed speedup: {needed_speedup:.1f}x")
    print(f"  Recommended: TensorRT + INT8 + Smaller model")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    run_benchmark()
