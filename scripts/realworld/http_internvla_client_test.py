#!/usr/bin/env python3
"""
Simple test client for http_internvla_server_debug_optimized.py

This tests the server without ROS2. For real robot deployment,
use http_internvla_client_debug_optimized.py which requires ROS2.

Usage:
    python http_internvla_client_test.py --visualize
"""

import argparse
import io
import json
import time
import numpy as np
from PIL import Image
import requests
from datetime import datetime

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from transformation import Calibration
    HAS_CALIB = True
except:
    HAS_CALIB = False
    print("[Client] Calibration not available")


class TestClient:
    def __init__(self, server_url='http://127.0.0.1:5802', calib_path=None):
        self.server_url = server_url
        self.reset = True
        self.request_count = 0
        self.start_time = None
        
        if HAS_CALIB and calib_path:
            self.calib = Calibration(calib_path)
        else:
            self.calib = None
        
        print(f"[Client] Connected to {server_url}")
    
    def send_request(self, rgb_image, depth_image):
        """Send RGB-Depth to server and get response."""
        rgb_bytes = io.BytesIO()
        rgb_image.save(rgb_bytes, format='JPEG')
        rgb_bytes = rgb_bytes.getvalue()
        
        depth_bytes = io.BytesIO()
        depth_image.save(depth_bytes, format='PNG')
        depth_bytes = depth_bytes.getvalue()
        
        data = {'reset': self.reset, 'idx': self.request_count}
        self.reset = False
        
        try:
            response = requests.post(
                self.server_url + '/eval_dual',
                files={'image': rgb_bytes, 'depth': depth_bytes},
                data={'json': json.dumps(data)},
                timeout=30
            )
            
            self.request_count += 1
            
            if self.start_time is None:
                self.start_time = time.time()
            
            elapsed = time.time() - self.start_time
            hz = self.request_count / elapsed if elapsed > 0 else 0
            
            return response.json(), hz
            
        except Exception as e:
            print(f"[Client] Error: {e}")
            return None, 0
    
    def create_test_images(self, width=640, height=480):
        """Create test RGB and depth images."""
        rgb = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        depth = np.random.randint(100, 1000, (height, width), dtype=np.uint16)
        return Image.fromarray(rgb), Image.fromarray(depth)


def run_test(client, num_requests=50, interval=0.02):
    """Run test loop."""
    print("=" * 60)
    print("TEST CLIENT RUNNING")
    print("=" * 60)
    print(f"Requests: {num_requests}")
    print(f"Interval: {interval}s")
    print("=" * 60)
    
    latencies = []
    
    for i in range(num_requests):
        rgb, depth = client.create_test_images()
        
        t0 = time.time()
        result, hz = client.send_request(rgb, depth)
        latency = (time.time() - t0) * 1000
        
        latencies.append(latency)
        
        status = 'OK'
        if result:
            if 'discrete_action' in result:
                status = f"Action: {result['discrete_action']}"
            elif 'trajectory' in result:
                status = f"Trajectory: {len(result['trajectory'])} pts"
            elif result.get('status') == 'waiting':
                status = 'Waiting...'
        else:
            status = 'ERROR'
        
        print(f"[{i+1:3d}] {latency:6.1f}ms | Hz: {hz:5.1f} | {status}")
        
        time.sleep(interval)
    
    avg_latency = sum(latencies) / len(latencies)
    print("=" * 60)
    print(f"Average latency: {avg_latency:.1f}ms")
    print(f"Requests/sec: {1000/avg_latency:.1f} Hz")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Test client for optimized server')
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:5802')
    parser.add_argument('--calib', type=str, default='scripts/realworld/calib/calib_scout.txt',
                        help='Calibration file path')
    parser.add_argument('--requests', type=int, default=50, help='Number of requests')
    parser.add_argument('--interval', type=float, default=0.02, help='Interval between requests (s)')
    args = parser.parse_args()
    
    client = TestClient(args.server_url, args.calib if HAS_CALIB else None)
    run_test(client, args.requests, args.interval)


if __name__ == '__main__':
    main()
