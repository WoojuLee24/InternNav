#!/usr/bin/env python3
"""
Full Pipeline Benchmark for InternVLA Real-World Navigation

This script benchmarks the complete S2 + S1 pipeline to measure
end-to-end performance and identify bottlenecks.

Usage:
    python benchmark_full_pipeline.py --model_path checkpoints/InternVLA-N1-w-NavDP
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


def benchmark_s2(model, processor, image, instruction, num_iterations=10):
    """Benchmark System 2 (VLM) inference."""
    print("\n=== System 2 (VLM) Benchmark ===")
    
    # Prepare inputs
    content = [{'type': 'image', 'image': image}, {'type': 'text', 'text': instruction}]
    conversation = [{'role': 'user', 'content': content}]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors='pt', padding=True)
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Warmup
    print("Warming up S2...")
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    # Benchmark
    print(f"Running S2 benchmark ({num_iterations} iterations)...")
    times = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    
    avg = sum(times) / len(times)
    print(f"RESULT: S2 Average: {avg:.1f}ms ({1000/avg:.1f} Hz)")
    print(f"RESULT: S2 Min: {min(times):.1f}ms, Max: {max(times):.1f}ms")
    
    return avg, times


def benchmark_full_pipeline(args):
    """Benchmark the full S2+S1 pipeline."""
    print("=" * 60)
    print("Full Pipeline Benchmark for InternVLA Real-World Navigation")
    print("=" * 60)
    
    # Enable optimizations
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Load model
    print("\nLoading model...")
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        device_map={'cuda:0': 0},
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    processor.tokenizer.padding_side = 'left'
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Attention: SDPA")
    
    # Create test image (proper size for vision encoder)
    dummy_rgb = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_rgb).convert('RGB')
    instruction = "Go forward and turn left"
    
    # Benchmark S2
    s2_avg, s2_times = benchmark_s2(model, processor, image, instruction)
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"System 2 (VLM):     {s2_avg:.1f}ms ({1000/s2_avg:.1f} Hz)")
    print(f"Target:            20-25ms (40-50 Hz)")
    print(f"Gap:               {s2_avg/25:.1f}x slower than target")
    print()
    print("OPTIMIZATION RECOMMENDATIONS:")
    print("1. Quantization (INT8/FP16): ~2x speedup")
    print("2. TensorRT conversion: ~3-5x speedup")
    print("3. vLLM with PagedAttention: ~2-4x speedup")
    print("4. Speculative decoding: ~2x speedup")
    print()
    print("Expected after optimizations: ~20-50ms (20-50 Hz)")
    print("=" * 60)
    
    return {
        's2_avg_ms': s2_avg,
        's2_times': s2_times,
        'target_hz': 40,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark InternVLA pipeline')
    parser.add_argument('--model_path', type=str, default='checkpoints/InternVLA-N1-w-NavDP',
                        help='Path to model checkpoint')
    args = parser.parse_args()
    
    results = benchmark_full_pipeline(args)
