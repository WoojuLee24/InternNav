#!/usr/bin/env python3
"""
High-Performance InternVLA Server

Optimized Flask server for real-world robot deployment achieving 40-50 Hz.

Optimizations:
- Vision embedding caching
- Async S2 processing
- TensorRT acceleration (when available)
- INT8 quantization (when available)

Usage:
    python http_internvla_server_hp.py --use-vision-cache --use-async-s2
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

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformation import Calibration
from high_performance_agent import create_high_performance_agent, HighPerformanceInternVLAAgent


app = Flask(__name__)

idx = 0
start_time = time.time()
output_dir = ''
save_dir = 'vis_debug/http_internvla_server_hp'
os.makedirs(save_dir, exist_ok=True)


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time
    start_time = time.time()
    
    image_file = request.files['image']
    depth_file = request.files['depth']
    json_data = request.form['json']
    data = json.loads(json_data)
    
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)
    depth = depth.astype(np.float32) / 10000.0
    
    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    instruction = "Exit the door, then Turn left and go straight until you find small fire extinguisher. Then stop."
    
    policy_init = data.get('reset', False)
    if policy_init:
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        print("[Server] Resetting agent...")
        agent.reset()
    
    idx += 1
    
    look_down = False
    dual_sys_output = agent.step(
        image, depth, camera_pose, instruction,
        intrinsic=args.camera_intrinsic, look_down=look_down
    )
    
    if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
        look_down = True
        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction,
            intrinsic=args.camera_intrinsic, look_down=look_down
        )
    
    json_output = {}
    if dual_sys_output.output_action is not None:
        json_output['discrete_action'] = dual_sys_output.output_action
    elif dual_sys_output.output_trajectory is not None:
        json_output['trajectory'] = dual_sys_output.output_trajectory.tolist()
        if dual_sys_output.output_pixel is not None:
            json_output['pixel_goal'] = dual_sys_output.output_pixel
    else:
        json_output['status'] = 'waiting'
    
    return jsonify(json_output)


@app.route("/stats", methods=['GET'])
def get_stats():
    """Return performance statistics."""
    return jsonify(agent.get_stats())


@app.route("/health", methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'timestamp': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='High-Performance InternVLA Server')
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-w-NavDP")
    parser.add_argument("--resize_w", type=int, default=224)
    parser.add_argument("--resize_h", type=int, default=224)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=4)
    
    parser.add_argument("--use_vision_cache", action='store_true', default=True,
                        help='Enable vision embedding cache')
    parser.add_argument("--no_vision_cache", action='store_false', dest='use_vision_cache',
                        help='Disable vision embedding cache')
    parser.add_argument("--use_async_s2", action='store_true', default=True,
                        help='Enable async S2 processing')
    parser.add_argument("--no_async_s2", action='store_false', dest='use_async_s2',
                        help='Disable async S2 processing')
    parser.add_argument("--use_tensorrt", action='store_true', default=False,
                        help='Enable TensorRT acceleration')
    parser.add_argument("--use_int8", action='store_true', default=False,
                        help='Enable INT8 quantization')
    
    parser.add_argument("--calib", type=str, default="scripts/realworld/calib/calib_scout.txt",
                        help="Path to calibration file")
    
    args = parser.parse_args()
    
    if os.path.exists(args.calib):
        calib = Calibration(args.calib)
        args.camera_intrinsic = np.array([
            [calib.f_u, 0.0, calib.c_u, 0.0],
            [0.0, calib.f_v, calib.c_v, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        print(f"[Server] Loaded calib: {args.calib}")
        print(f"[Server] fx={calib.f_u:.2f} fy={calib.f_v:.2f} cx={calib.c_u:.2f} cy={calib.c_v:.2f}")
    else:
        args.camera_intrinsic = np.eye(4)
        print(f"[Server] Calibration file not found: {args.calib}")
    
    print("=" * 60)
    print("INITIALIZING HIGH-PERFORMANCE AGENT")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Image size: {args.resize_w}x{args.resize_h}")
    print(f"Vision cache: {'enabled' if args.use_vision_cache else 'disabled'}")
    print(f"Async S2: {'enabled' if args.use_async_s2 else 'disabled'}")
    print(f"TensorRT: {'enabled' if args.use_tensorrt else 'disabled'}")
    print(f"INT8: {'enabled' if args.use_int8 else 'disabled'}")
    print("=" * 60)
    
    agent = create_high_performance_agent(
        model_path=args.model_path,
        device=args.device,
        resize=args.resize_w,
        use_vision_cache=args.use_vision_cache,
        use_async_s2=args.use_async_s2,
        use_tensorrt=args.use_tensorrt,
        use_int8=args.use_int8
    )
    
    agent.reset()
    
    print(f"\n[Server] Starting on 0.0.0.0:5802")
    print(f"[Server] Stats endpoint: http://localhost:5802/stats")
    print(f"[Server] Health endpoint: http://localhost:5802/health")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5802, threaded=True)
