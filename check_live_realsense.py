import sys
import os
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = "/home/gdr/gd_vln/workspace/src/InternNav"
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "scripts/iros_challenge/onsite_competition/sdk"))

from internnav.configs.agent import AgentCfg
from internnav.utils import AgentClient
from cam_ros import RosRealSense 

# 1. Initialize Agent Client with FULL configuration (to avoid 422 error)
agent_cfg = AgentCfg(
      server_host='localhost',
      server_port=8087,
      model_name='internvla_n1',
      ckpt_path='',
      model_settings={
            'policy_name': "InternVLAN1_Policy",
            'state_encoder': None,
            'env_num': 1,
            'sim_num': 1,
            'model_path': "checkpoints/InternVLA-N1",
            'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            'width': 640,
            'height': 480,
            'hfov': 79,
            'resize_w': 384,
            'resize_h': 384,
            'max_new_tokens': 1024,
            'num_frames': 32,
            'num_history': 8,
            'num_future_steps': 4,
            'device': 'cuda:0',
            'predict_step_nums': 32,
            'continuous_traj': True,
            'vis_debug': True,
            'infer_mode': 'partial_async',
            'vis_debug_path': 'vis_debug/test_live_realsense',
      }
)
agent = AgentClient(agent_cfg)

# 2. Initialize ROS Bridge with EXACT topic names from your 'ros2 topic list'
print("Connecting to ROS topics...")
cam = RosRealSense(
    rgb_topic='/camera/camera/color/image_raw', 
    depth_topic='/camera/camera/aligned_depth_to_color/image_raw' # '/camera/camera/depth/image_rect_raw'
)

# 3. Inference Loop
try:
    print("Starting Live Inference Loop...")
    while True:
        # This calls rclpy under the hood
        obs = cam.get_observation()
        if obs['rgb'] is None or obs['depth'] is None:
            print("Empty observation! Skipping this frame.")
            continue
        obs['instruction'] = 'go to the red car'

        # Inference
        start = time.time()
        result = agent.step([obs])
        print(f"Model Inference: {time.time() - start:.4f}s")
        
        # Action map: 0: STOP, 1: Forward, 2: Left, 3: Right
        action = result[0]['action'][0]
        print(f"Action: {action} | Timestamp: {obs['timestamp_s']:.2f}")
        
        time.sleep(0.2)
except KeyboardInterrupt:
    print("Stopping...")
finally:
    cam.stop()


# Configure data directory (single scene per folder)
scene_dir = '../../assets/realworld_sample_data1'

# Check if instruction file exists
instruction_path = os.path.join(scene_dir, 'instruction.txt')
if not os.path.exists(instruction_path):
    print(f"Error: instruction.txt not found in {scene_dir}")
else:
    print(f"Scene directory: {scene_dir}")
    
    # Read instruction
    with open(instruction_path, 'r') as f:
        instruction = f.read().strip()
    print(f"Instruction: {instruction}")
    

    