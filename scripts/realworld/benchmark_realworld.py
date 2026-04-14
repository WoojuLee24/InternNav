#!/usr/bin/env python3
"""
Comprehensive Benchmarking Script for InternVLA Real-World Navigation

This script benchmarks the current performance of the dual-system navigation pipeline
and identifies bottlenecks.

Usage:
    python benchmark_realworld.py --model_path checkpoints/InternVLA-N1-w-NavDP

Output:
    - Detailed timing for each module
    - Current Hz and target Hz comparison
    - Bottleneck identification
    - Recommendations for optimization
"""

import argparse
import json
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# Add project paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'src/diffusion-policy'))


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.iterations = 0
        self.start_time = None
        self.end_time = None
    
    def record(self, module_name: str, duration_ms: float):
        """Record a timing measurement."""
        self.timings[module_name].append(duration_ms)
    
    def start(self):
        """Start benchmarking."""
        self.start_time = time.time()
    
    def end(self):
        """End benchmarking."""
        self.end_time = time.time()
    
    def get_stats(self, module_name: str) -> dict:
        """Get statistics for a module."""
        if module_name not in self.timings or not self.timings[module_name]:
            return None
        
        values = self.timings[module_name]
        return {
            "count": len(values),
            "total_ms": sum(values),
            "mean_ms": np.mean(values),
            "median_ms": np.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "std_ms": np.std(values),
            "p95_ms": np.percentile(values, 95),
            "p99_ms": np.percentile(values, 99),
        }
    
    def summary(self) -> dict:
        """Get full summary."""
        stats = {
            "iterations": self.iterations,
            "total_time_s": self.end_time - self.start_time if self.end_time else 0,
            "modules": {},
        }
        
        for module_name in self.timings:
            module_stats = self.get_stats(module_name)
            if module_stats:
                stats["modules"][module_name] = module_stats
        
        # Calculate overall frequency
        if stats["total_time_s"] > 0 and self.iterations > 0:
            mean_step_time = stats["total_time_s"] * 1000 / self.iterations
            stats["mean_step_time_ms"] = mean_step_time
            stats["estimated_hz"] = 1000.0 / mean_step_time if mean_step_time > 0 else 0
        else:
            stats["mean_step_time_ms"] = 0
            stats["estimated_hz"] = 0
        
        return stats


class ModuleBenchmark:
    """Context manager for benchmarking a module."""
    
    def __init__(self, name: str, result: BenchmarkResult):
        self.name = name
        self.result = result
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        self.result.record(self.name, duration_ms)


def benchmark_model_loading(args) -> dict:
    """Benchmark model loading time."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Model Loading")
    print("=" * 60)
    
    result = BenchmarkResult()
    result.start()
    
    with ModuleBenchmark("import_modules", result):
        from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
        from transformers import AutoProcessor
    
    with ModuleBenchmark("load_model", result):
        model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": torch.device(args.device)},
        )
        model.eval()
    
    with ModuleBenchmark("load_processor", result):
        processor = AutoProcessor.from_pretrained(args.model_path)
    
    result.end()
    
    stats = result.summary()
    
    print(f"\nModel loading benchmark:")
    for module, module_stats in stats["modules"].items():
        print(f"  {module}: {module_stats['mean_ms']:.2f}ms (total: {module_stats['total_ms']:.2f}ms)")
    
    return {"model": model, "processor": processor, "stats": stats}


def benchmark_s2_inference(model, processor, args, num_iterations: int = 10) -> dict:
    """Benchmark System 2 (VLM) inference."""
    print("\n" + "=" * 60)
    print("BENCHMARK: System 2 (VLM) Inference")
    print("=" * 60)
    
    # Create dummy inputs
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_depth = np.random.rand(480, 640).astype(np.float32) * 5 + 0.5
    instruction = "Go forward and turn left at the door."
    
    result = BenchmarkResult()
    result.start()
    
    for i in range(num_iterations):
        iter_start = time.time()
        
        with ModuleBenchmark("s2_preprocess", result):
            from PIL import Image
            image = Image.fromarray(dummy_image).convert('RGB')
            image = image.resize((args.resize_w, args.resize_h))
        
        with ModuleBenchmark("s2_prompt_build", result):
            prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next?"
            prompt = prompt.replace("<instruction>", instruction)
        
        with ModuleBenchmark("s2_processor", result):
            inputs = processor(text=[prompt], images=[image], return_tensors="pt")
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with ModuleBenchmark("s2_model_forward", result):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
        
        with ModuleBenchmark("s2_token_decode", result):
            llm_output = processor.tokenizer.decode(
                outputs.sequences[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
        
        iter_time = (time.time() - iter_start) * 1000
        result.record("s2_iteration_total", iter_time)
        result.iterations += 1
    
    result.end()
    
    stats = result.summary()
    
    print(f"\nSystem 2 benchmark ({num_iterations} iterations):")
    for module, module_stats in stats["modules"].items():
        print(f"  {module}:")
        print(f"    Mean: {module_stats['mean_ms']:.2f}ms")
        print(f"    Median: {module_stats['median_ms']:.2f}ms")
        print(f"    P95: {module_stats['p95_ms']:.2f}ms")
        print(f"    Max: {module_stats['max_ms']:.2f}ms")
    
    if "s2_iteration_total" in stats["modules"]:
        total_stats = stats["modules"]["s2_iteration_total"]
        print(f"\n  Total S2 time: {total_stats['mean_ms']:.2f}ms")
        print(f"  Estimated S2 frequency: {1000.0/total_stats['mean_ms']:.1f}Hz")
    
    return stats


def benchmark_s1_inference(model, args, num_iterations: int = 10) -> dict:
    """Benchmark System 1 (DiT) inference."""
    print("\n" + "=" * 60)
    print("BENCHMARK: System 1 (DiT) Inference")
    print("=" * 60)
    
    # Create dummy inputs
    dummy_rgb = np.random.rand(2, 224, 224, 3).astype(np.float32)
    dummy_depth = np.random.rand(2, 224, 224, 1).astype(np.float32) * 5
    dummy_latent = torch.randn(1, 4, 3584).to(model.device)
    
    result = BenchmarkResult()
    result.start()
    
    for i in range(num_iterations):
        iter_start = time.time()
        
        with ModuleBenchmark("s1_preprocess", result):
            rgbs = torch.from_numpy(dummy_rgb).unsqueeze(0).to(model.device)
            depths = torch.from_numpy(dummy_depth).unsqueeze(0).unsqueeze(-1).to(model.device)
        
        with ModuleBenchmark("s1_diffusion", result):
            with torch.no_grad():
                trajectories = model.generate_traj(dummy_latent, rgbs, depths)
        
        iter_time = (time.time() - iter_start) * 1000
        result.record("s1_iteration_total", iter_time)
        result.iterations += 1
    
    result.end()
    
    stats = result.summary()
    
    print(f"\nSystem 1 benchmark ({num_iterations} iterations):")
    for module, module_stats in stats["modules"].items():
        print(f"  {module}:")
        print(f"    Mean: {module_stats['mean_ms']:.2f}ms")
        print(f"    Median: {module_stats['median_ms']:.2f}ms")
    
    if "s1_iteration_total" in stats["modules"]:
        total_stats = stats["modules"]["s1_iteration_total"]
        print(f"\n  Total S1 time: {total_stats['mean_ms']:.2f}ms")
        print(f"  Estimated S1 frequency: {1000.0/total_stats['mean_ms']:.1f}Hz")
    
    return stats


def benchmark_end_to_end(model, processor, args, num_iterations: int = 5) -> dict:
    """Benchmark end-to-end pipeline."""
    print("\n" + "=" * 60)
    print("BENCHMARK: End-to-End Pipeline")
    print("=" * 60)
    
    from internnav.model.utils.vln_utils import traj_to_actions
    
    # Create dummy inputs
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_depth = np.random.rand(480, 640).astype(np.float32) * 5 + 0.5
    instruction = "Go forward and turn left."
    
    result = BenchmarkResult()
    result.start()
    
    # State for iterative benchmark
    rgb_list = []
    episode_idx = 0
    conversation_history = []
    llm_output = ""
    past_key_values = None
    last_s2_idx = -100
    PLAN_STEP_GAP = args.plan_step_gap
    num_history = args.num_history
    resize_w = args.resize_w
    resize_h = args.resize_h
    
    for i in range(num_iterations):
        iter_start = time.time()
        
        from PIL import Image
        from internnav.model.utils.vln_utils import split_and_clean
        
        # Image preprocessing
        with ModuleBenchmark("e2e_preprocess", result):
            image = Image.fromarray(dummy_image).convert('RGB')
            image_resized = image.resize((resize_w, resize_h))
            rgb_list.append(image_resized)
        
        # Prompt building
        with ModuleBenchmark("e2e_prompt", result):
            if episode_idx == 0:
                history_id = []
            else:
                history_id = np.unique(np.linspace(0, episode_idx - 1, num_history, dtype=np.int32)).tolist()
            history_id = sorted(history_id)
            input_images = [rgb_list[i] for i in history_id] + [image_resized]
        
        # Processor
        with ModuleBenchmark("e2e_processor", result):
            prompt = "You are an autonomous navigation assistant. Your task is " + instruction + ". Where should you go next?"
            inputs = processor(
                text=[prompt],
                images=input_images,
                return_tensors="pt"
            ).to(model.device)
        
        # Model forward
        with ModuleBenchmark("e2e_model", result):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
        
        # Token decode
        with ModuleBenchmark("e2e_decode", result):
            llm_output = processor.tokenizer.decode(
                outputs.sequences[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
        
        # Check if output has coordinates
        import re
        if bool(re.search(r'\d', llm_output)):
            # Generate latent
            with ModuleBenchmark("e2e_latent", result):
                image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
                pixel_values = inputs.pixel_values
                with torch.no_grad():
                    traj_latents = model.generate_latents(outputs.sequences, pixel_values, image_grid_thw)
            
            # Generate trajectory
            with ModuleBenchmark("e2e_traj", result):
                processed_rgb = np.array(image.resize((224, 224))) / 255
                processed_depth = np.array(Image.fromarray((dummy_depth * 10000).astype(np.uint16)).resize((224, 224))) / 255
                rgbs = torch.from_numpy(processed_rgb).unsqueeze(0).unsqueeze(0).to(model.device)
                depths = torch.from_numpy(processed_depth).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(model.device)
                with torch.no_grad():
                    trajectories = model.generate_traj(traj_latents, rgbs, depths)
                
                action_list = traj_to_actions(trajectories, use_discrate_action=False)
        
        episode_idx += 1
        
        iter_time = (time.time() - iter_start) * 1000
        result.record("e2e_total", iter_time)
        result.iterations += 1
        
        print(f"  Iteration {i+1}: {iter_time:.2f}ms")
    
    result.end()
    
    stats = result.summary()
    
    print(f"\nEnd-to-end benchmark ({num_iterations} iterations):")
    
    if "e2e_total" in stats["modules"]:
        total_stats = stats["modules"]["e2e_total"]
        print(f"\n  TOTAL TIME: {total_stats['mean_ms']:.2f}ms (mean)")
        print(f"  Median: {total_stats['median_ms']:.2f}ms")
        print(f"  P95: {total_stats['p95_ms']:.2f}ms")
        print(f"  Target: <25ms (40Hz)")
        
        hz = 1000.0 / total_stats['mean_ms'] if total_stats['mean_ms'] > 0 else 0
        print(f"\n  ESTIMATED FREQUENCY: {hz:.1f}Hz")
        
        if hz >= 40:
            print("  ✓ TARGET ACHIEVED!")
        elif hz >= 20:
            print("  ◐ PARTIAL - Needs optimization")
        else:
            print("  ✗ FAR FROM TARGET - Major optimization needed")
    
    return stats


def print_bottleneck_analysis(s2_stats: dict, s1_stats: dict, e2e_stats: dict):
    """Analyze and print bottleneck analysis."""
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    # Get critical paths
    critical_path = []
    
    if "s2_model_forward" in s2_stats.get("modules", {}):
        critical_path.append(("S2 Model Forward", s2_stats["modules"]["s2_model_forward"]["mean_ms"]))
    
    if "s1_diffusion" in s1_stats.get("modules", {}):
        critical_path.append(("S1 Diffusion", s1_stats["modules"]["s1_diffusion"]["mean_ms"]))
    
    # Sort by time
    critical_path.sort(key=lambda x: x[1], reverse=True)
    
    print("\nCritical Path (sorted by time):")
    total_critical = sum(t for _, t in critical_path)
    
    for name, time_ms in critical_path:
        pct = (time_ms / total_critical * 100) if total_critical > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {name:<25} {time_ms:>8.2f}ms {bar} {pct:>5.1f}%")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if critical_path and critical_path[0][1] > 100:
        print(f"  1. System 2 (VLM) is the main bottleneck ({critical_path[0][1]:.0f}ms)")
        print("     → Apply KV-cache optimization")
        print("     → Consider INT8 quantization")
        print("     → Use vLLM with PagedAttention")
    
    if len(critical_path) > 1 and critical_path[1][1] > 50:
        print(f"  2. System 1 (DiT) needs optimization ({critical_path[1][1]:.0f}ms)")
        print("     → Apply TensorRT optimization")
        print("     → Reduce diffusion steps if possible")
    
    print("  3. Consider async pipeline for parallel execution")


def main():
    parser = argparse.ArgumentParser(description='Benchmark InternVLA Real-World Navigation')
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-w-NavDP")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("INTERNVLA REAL-WORLD BENCHMARK")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("\nWARNING: No CUDA GPU available!")
    
    # Benchmark model loading
    load_results = benchmark_model_loading(args)
    model = load_results["model"]
    processor = load_results["processor"]
    
    # Benchmark System 2
    s2_results = benchmark_s2_inference(model, processor, args, args.iterations)
    
    # Benchmark System 1
    s1_results = benchmark_s1_inference(model, args, args.iterations)
    
    # Benchmark end-to-end
    e2e_results = benchmark_end_to_end(model, processor, args, min(args.iterations, 5))
    
    # Bottleneck analysis
    print_bottleneck_analysis(s2_results, s1_results, e2e_results)
    
    # Export results
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    all_results = {
        "args": vars(args),
        "model_loading": load_results["stats"],
        "system2": s2_results,
        "system1": s1_results,
        "end_to_end": e2e_results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults exported to: {output_file}")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
