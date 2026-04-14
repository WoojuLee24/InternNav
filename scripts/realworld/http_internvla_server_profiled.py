"""
Optimized HTTP Server for InternVLA Real-World Navigation
with Comprehensive Performance Profiling

This server provides an optimized version of the dual-system navigation server with:
- Comprehensive performance profiling at each module level
- KV-Cache optimization for System 2
- Async pipeline support
- Performance metrics export
- Real-time Hz monitoring

Usage:
    python http_internvla_server_profiled.py --model_path checkpoints/InternVLA-N1-w-NavDP
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
import cv2

# Add project paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

from internnav.agent.internvla_n1_agent_realworld_profiled import InternVLAN1ProfilingAgent
from transformation import Calibration
from performance_profiler import get_profiler, ProfilerContext

app = Flask(__name__)

# Global state
idx = 0
output_dir = ''
save_dir = 'vis_debug/http_internvla_server_profiled'
os.makedirs(save_dir, exist_ok=True)

# Performance tracking
performance_log = []
step_times = []
s2_times = []
s1_times = []
start_time = time.time()


def init_agent(args):
    """Initialize the profiling agent."""
    global agent
    
    # Import the profiled agent
    from internvla_n1_agent_realworld_profiled import InternVLAN1ProfilingAgent
    
    agent = InternVLAN1ProfilingAgent(args, enable_profiling=True)
    
    # Warmup
    agent.step(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640)),
        np.eye(4),
        "hello",
        args.camera_intrinsic,
    )
    agent.reset()
    
    print(f"[Server] Agent initialized with profiling enabled")


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    """
    Main evaluation endpoint with comprehensive profiling.
    """
    global idx, agent, performance_log, step_times, s2_times, s1_times
    
    request_start = time.time()
    profiler = get_profiler()
    profiler.start_iteration()
    
    # ===== 1. HTTP Request Parsing =====
    with ProfilerContext("http_request_parse", profiler):
        image_file = request.files['image']
        depth_file = request.files['depth']
        json_data = request.form['json']
        data = json.loads(json_data)
    
    # ===== 2. Image Decoding =====
    with ProfilerContext("image_decode", profiler):
        image = Image.open(image_file.stream).convert('RGB')
        image = np.asarray(image)
        
        depth = Image.open(depth_file.stream).convert('I')
        depth = np.asarray(depth).astype(np.float32) / 10000.0
    
    decode_time = (time.time() - request_start) * 1000
    print(f"[Profile] Image decode: {decode_time:.2f}ms")
    
    # Get instruction and metadata
    instruction = data.get('instruction', "Move forward and stop.")
    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    # ===== 3. Check for Reset =====
    policy_init = data.get('reset', False)
    if policy_init:
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        agent.reset()
        print("[Server] Agent reset")
    
    idx += 1
    
    # ===== 4. Main Inference with System 2 =====
    look_down = False
    step_start = time.time()
    
    # Run System 2
    s2_start = time.time()
    with ProfilerContext("s2_total", profiler):
        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction, 
            intrinsic=args.camera_intrinsic, look_down=look_down
        )
    
    s2_time = (time.time() - s2_start) * 1000
    s2_times.append(s2_time)
    print(f"[Profile] S2 time: {s2_time:.2f}ms")
    
    # Handle look_down if needed
    if dual_sys_output.output_action is not None and list(dual_sys_output.output_action) == [5]:
        look_down = True
        with ProfilerContext("s2_lookdown", profiler):
            dual_sys_output = agent.step(
                image, depth, camera_pose, instruction,
                intrinsic=args.camera_intrinsic, look_down=look_down
            )
    
    # ===== 5. Process System 1 =====
    s1_time = 0
    if dual_sys_output.output_trajectory is not None:
        s1_time = 0  # Already included in agent.step()
    
    # ===== 6. Build Response =====
    step_time = (time.time() - step_start) * 1000
    step_times.append(step_time)
    
    with ProfilerContext("http_response_build", profiler):
        json_output = {}
        
        if dual_sys_output.output_action is not None:
            json_output['discrete_action'] = list(dual_sys_output.output_action)
        else:
            if dual_sys_output.output_trajectory is not None:
                json_output['trajectory'] = dual_sys_output.output_trajectory.tolist()
            if dual_sys_output.output_pixel is not None:
                json_output['pixel_goal'] = list(dual_sys_output.output_pixel)
    
    # ===== 7. End Iteration =====
    profiler.end_iteration()
    
    # ===== 8. Print Periodic Stats =====
    total_time = (time.time() - request_start) * 1000
    
    if idx % 5 == 0:
        print_performance_stats()
    
    return jsonify(json_output)


@app.route("/performance", methods=['GET'])
def get_performance():
    """Get performance statistics."""
    profiler = get_profiler()
    stats = profiler.get_all_stats()
    
    # Add agent-specific stats
    if 'agent' in sys.modules:
        try:
            agent_stats = agent.get_performance_summary()
            stats['agent_summary'] = agent_stats
        except:
            pass
    
    return jsonify(stats)


@app.route("/performance/export", methods=['GET'])
def export_performance():
    """Export performance data to JSON file."""
    profiler = get_profiler()
    profiler.export_json(f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    return jsonify({"status": "exported"})


@app.route("/reset_stats", methods=['POST'])
def reset_stats():
    """Reset performance statistics."""
    global step_times, s2_times, s1_times
    step_times = []
    s2_times = []
    s1_times = []
    profiler = get_profiler()
    profiler.reset()
    return jsonify({"status": "reset"})


def print_performance_stats():
    """Print current performance statistics."""
    global step_times, s2_times, s1_times
    
    if len(step_times) < 2:
        return
    
    recent_steps = step_times[-20:] if len(step_times) >= 20 else step_times
    recent_s2 = s2_times[-20:] if len(s2_times) >= 20 else s2_times
    
    mean_step = np.mean(recent_steps)
    mean_s2 = np.mean(recent_s2)
    freq = 1000.0 / mean_step if mean_step > 0 else 0
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("PERFORMANCE PROFILING SUMMARY")
    print("=" * 70)
    print(f"Total steps: {idx}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Overall frequency: {idx/elapsed:.1f}Hz")
    print("-" * 70)
    print(f"{'Module':<30} {'Mean (ms)':<15} {'Hz':<15} {'Target Hz':<15}")
    print("-" * 70)
    print(f"{'End-to-end step':<30} {mean_step:<15.2f} {freq:<15.1f} {'40-50':<15}")
    print(f"{'System 2 (S2)':<30} {mean_s2:<15.2f} {1000.0/mean_s2 if mean_s2 > 0 else 0:<15.1f} {'10-20':<15}")
    
    if s1_times:
        recent_s1 = s1_times[-20:] if len(s1_times) >= 20 else s1_times
        mean_s1 = np.mean(recent_s1)
        print(f"{'System 1 (S1)':<30} {mean_s1:<15.2f} {1000.0/mean_s1 if mean_s1 > 0 else 0:<15.1f} {'100-200':<15}")
    
    print("-" * 70)
    print(f"Target: 40-50 Hz (20-25ms per step)")
    print(f"Current: {freq:.1f} Hz ({mean_step:.1f}ms per step)")
    
    if freq >= 40:
        print("✓ Target ACHIEVED!")
    elif freq >= 20:
        print("◐ Target PARTIALLY achieved - needs optimization")
    else:
        print("✗ Target NOT achieved - major optimization needed")
    
    print("=" * 70 + "\n")


@app.route("/health", methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "steps_completed": idx,
        "uptime_seconds": time.time() - start_time
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InternVLA Real-World Server with Profiling')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-w-NavDP")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=4)
    parser.add_argument("--calib", type=str, 
                       default="/home/gdr/gd_vln/workspace/src/InternNav/scripts/realworld/calib/calib_scout.txt")
    parser.add_argument("--port", type=int, default=5802)
    parser.add_argument("--enable_profiling", type=bool, default=True)
    
    global args
    args = parser.parse_args()
    
    # Load calibration
    calib = Calibration(args.calib)
    args.camera_intrinsic = np.array([
        [calib.f_u, 0.0, calib.c_u, 0.0],
        [0.0, calib.f_v, calib.c_v, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    print(f"[Server] Loaded calib: {args.calib}")
    print(f"[Server] camera_intrinsic: fx={calib.f_u:.2f} fy={calib.f_v:.2f} cx={calib.c_u:.2f} cy={calib.c_v:.2f}")
    
    # Initialize agent with profiling
    init_agent(args)
    
    print(f"\n{'='*70}")
    print("InternVLA Real-World Server with Performance Profiling")
    print(f"{'='*70}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Port: {args.port}")
    print(f"Profiling: {'ENABLED' if args.enable_profiling else 'DISABLED'}")
    print(f"{'='*70}\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=args.port, threaded=True)
