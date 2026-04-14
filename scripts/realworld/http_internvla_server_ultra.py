#!/usr/bin/env python3
"""
Ultra-High-Performance InternVLA Server

Optimized for maximum Hz (50+ Hz):
- S2 (planner) runs async without rate limiting
- S1 returns immediately with cached results
- No blocking - maximum throughput

Usage:
    python http_internvla_server_ultra.py
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
from ultra_high_performance_agent import create_ultra_high_performance_agent

app = Flask(__name__)

idx = 0
start_time = time.time()
output_dir = ''


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
        image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
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
    return jsonify(agent.get_stats())


@app.route("/health", methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ultra-High-Performance InternVLA Server')
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-w-NavDP")
    parser.add_argument("--resize_w", type=int, default=224)
    parser.add_argument("--resize_h", type=int, default=224)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=4)
    parser.add_argument("--calib", type=str, default="scripts/realworld/calib/calib_scout.txt")
    
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
        print(f"[Server] Warning: Calibration file not found: {args.calib}")
    
    print("=" * 60)
    print("ULTRA-HIGH-PERFORMANCE InternVLA SERVER")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Image size: {args.resize_w}x{args.resize_h}")
    print("Optimizations: Ultra Async S2 (no rate limit)")
    print("=" * 60)
    
    agent = create_ultra_high_performance_agent(
        model_path=args.model_path,
        device=args.device,
        resize=args.resize_w
    )
    
    agent.reset()
    
    print(f"\n[Server] Starting on 0.0.0.0:5802")
    print(f"[Server] Stats: http://localhost:5802/stats")
    print(f"[Server] Health: http://localhost:5802/health")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5802, threaded=True)
