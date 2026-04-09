import argparse
import json
import os
import threading
import time
from datetime import datetime
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, ImageDraw, ImageFont
import cv2


# Add project path
# project_root = Path("/home/gdr/gd_vln/workspace/src/InternNav")
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformation import Calibration

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''
save_dir = 'vis_debug/http_internvla_server_debug'
os.makedirs(save_dir, exist_ok=True)
agent_lock = threading.Lock()
SERVER_MODE = "sync"
SERVER_OPT_FLAGS = {
    "kv_cache": False,
    "tensorrt": False,
    "quantization": False,
    "vision_cache": False,
    "methods": [],
}


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time
    try:
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
        print(f"read http data cost {time.time() - start_time}")

        camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        #instruction = "Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor"
        #instruction = "Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor"
        # instruction = "Stop. Just stop. Move forward one step. Move forward two step. Turn right and stop."
        # instruction = "Go straight along the walkway and turn right at the crosswalk. Go straight to the end of the crosswalk and stop."
        # instruction = "Go straight along the walkway until you see a crosswalk. Go straight again until you see a second crosswalk. Turn right at the second crosswalk and go straight to the end of the crosswalk. Stop at the end of the crosswalk."
        instruction = "Exit door. Turn left and go straight until you find fire extinguisher. Then stop."
        policy_init = data['reset']
        req_mode = data.get('mode', SERVER_MODE)
        if req_mode != 'sync':
            print(f"[Server] Requested mode '{req_mode}' not implemented yet; using sync")
        req_opts = data.get('optimizations', {})
        if req_opts:
            print(f"[Server] Received optimization request (scaffold): {req_opts}")
        if policy_init:
            start_time = time.time()
            idx = 0
            output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
            os.makedirs(output_dir, exist_ok=True)
            print("init reset model!!!")
            with agent_lock:
                agent.reset()

        idx += 1

        look_down = False
        t0 = time.time()
        dual_sys_output = {}

        with agent_lock:
            dual_sys_output = agent.step(
                image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
            )
            if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
                look_down = True
                dual_sys_output = agent.step(
                    image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
                )

        t1 = time.time()
        generate_time = t1 - t0
        print(f"dual sys step time: {generate_time}")

        # 클라이언트에서 보낸 idx 추출
        image_id = data.get('idx', 0)
        filename = f"frame_{image_id:05d}"  # 예: frame_00001.jpg

        # image_id = int(time.time() * 1000)
        # filename = f"rec_{image_id}.jpg"

        json_output = {}
        if dual_sys_output.output_action is not None:
            json_output['discrete_action'] = dual_sys_output.output_action
            # annotate_image(image_id, image, agent.llm_output, dual_sys_output.output_trajectory, dual_sys_output.output_pixel, save_dir, filename)

        elif dual_sys_output.output_trajectory is not None:
            json_output['trajectory'] = dual_sys_output.output_trajectory.tolist()
            if dual_sys_output.output_pixel is not None:
                json_output['pixel_goal'] = dual_sys_output.output_pixel
                # annotate_image(image_id, image, 'traj', dual_sys_output.output_trajectory.tolist(), dual_sys_output.output_pixel, save_dir, filename)
            else:
                # annotate_image(image_id, image, 'traj_cached_latent', dual_sys_output.output_trajectory.tolist(), dual_sys_output.output_pixel, save_dir, filename)
                pass
        else:
            json_output['status'] = 'waiting'

        # print(f"json_output {json_output}")
        return jsonify(json_output)
    except Exception as e:
        print(f"[Server] eval_dual exception: {repr(e)}")
        return jsonify({'status': 'waiting', 'error': str(e)})



def annotate_image(idx, image, llm_output, trajectory, pixel_goal, output_dir, filename):
    import matplotlib
    matplotlib.use('Agg')  # 반드시 pyplot을 import하기 전에 실행해야 합니다.
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if 'look_down' not in filename:
        filename = f'{filename}_z'
    image = Image.fromarray(image)#.save(f'rgb_{idx}.png')
    draw = ImageDraw.Draw(image)
    font_size = 20
    font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    text_content = []
    text_content.append(f"Frame    Id  : {idx}")
    text_content.append(f"Actions      : {llm_output}" )
    max_width = 0
    total_height = 0
    for line in text_content:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = 26
        max_width = max(max_width, text_width)
        total_height += text_height

    padding = 10
    box_x, box_y = 10, 10
    box_width = max_width + 2 * padding
    box_height = total_height + 2 * padding

    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill='black')

    text_color = 'white'
    y_position = box_y + padding
    
    for line in text_content:
        draw.text((box_x + padding, y_position), line, fill=text_color, font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        text_height = 26
        y_position += text_height
    image = np.array(image)
    
    # Draw trajectory visualization in the top-right corner using matplotlib
    if trajectory is not None and len(trajectory) > 0:
        img_height, img_width = image.shape[:2]
        
        # Window parameters
        window_size = 200  # Window size in pixels
        window_margin = 0  # Margin from edge
        window_x = img_width - window_size - window_margin
        window_y = window_margin
        
        # Extract trajectory points
        traj_points = []
        for point in trajectory:
            if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                traj_points.append([float(point[0]), float(point[1])])
        
        if len(traj_points) > 0:
            traj_array = np.array(traj_points)
            x_coords = traj_array[:, 0]
            y_coords = traj_array[:, 1]
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            fig.patch.set_alpha(0.6)  # Semi-transparent background
            fig.patch.set_facecolor('gray')
            ax.set_facecolor('lightgray')
            
            # Plot trajectory
            # Coordinate system: x-axis points up, y-axis points left
            # Origin at bottom center
            ax.plot(y_coords, x_coords, 'b-', linewidth=2, label='Trajectory')
            
            # Mark start point (green) and end point (red)
            ax.plot(y_coords[0], x_coords[0], 'go', markersize=6, label='Start')
            ax.plot(y_coords[-1], x_coords[-1], 'ro', markersize=6, label='End')
            
            # Mark origin
            ax.plot(0, 0, 'w+', markersize=10, markeredgewidth=2, label='Origin')
            
            # Set axis labels
            ax.set_xlabel('Y (left +)', fontsize=8)
            ax.set_ylabel('X (up +)', fontsize=8)
            ax.invert_xaxis()
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Set equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
            
            # Add legend
            ax.legend(fontsize=6, loc='upper right')
            
            # Adjust layout
            plt.tight_layout(pad=0.3)
            
            # Convert matplotlib figure to numpy array
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            # plot_img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
            # plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # plt.close(fig)

            # Patch for RGBA to RGB conversion
            # 1. 버퍼로부터 RGBA(4채널) 데이터를 가져옵니다.
            rgba_buffer = canvas.buffer_rgba()
            plot_img = np.frombuffer(rgba_buffer, dtype=np.uint8)
            
            # 2. 4채널 모양으로 먼저 reshape 합니다.
            # (height, width, 4) 형태로 복원
            width, height = canvas.get_width_height()
            plot_img = plot_img.reshape((height, width, 4))
            
            # 3. OpenCV를 이용해 RGBA를 RGB로 변환합니다. (Alpha 채널 제거)
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2RGB)
            
            # 메모리 해제
            plt.close(fig)

            # Resize plot to fit window
            plot_img = cv2.resize(plot_img, (window_size, window_size))
            
            # Overlay plot on image
            image[window_y:window_y+window_size, window_x:window_x+window_size] = plot_img
    
    if pixel_goal is not None:
        cv2.circle(image, (pixel_goal[1], pixel_goal[0]), 5, (255, 0, 0), -1)
    image = Image.fromarray(image).convert('RGB')
    image.save(f'{output_dir}/{filename}.jpg')
    # to numpy array
    return np.array(image)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-w-NavDP")
    parser.add_argument("--resize_w", type=int, default=256)
    parser.add_argument("--resize_h", type=int, default=256)
    parser.add_argument("--num_history", type=int, default=1)
    parser.add_argument("--plan_step_gap", type=int, default=12)
    parser.add_argument("--mode", type=str, default="sync", choices=["sync", "async"],
                        help="Execution mode. async is scaffold-only for now.")
    parser.add_argument("--kv-cache", action="store_true", help="Enable KV-cache optimization (scaffold flag).")
    parser.add_argument("--tensorrt", action="store_true", help="Enable TensorRT optimization (scaffold flag).")
    parser.add_argument("--quantization", action="store_true", help="Enable quantization optimization (scaffold flag).")
    parser.add_argument("--quant-method", type=str, default="dynamic", choices=["dynamic", "static", "qat"],
                        help="Quantization method (safe mode currently supports dynamic CPU fallback).")
    parser.add_argument("--tensorrt-engine", type=str, default="",
                        help="Path to TensorRT engine (optional, safe fallback if unavailable).")
    parser.add_argument("--vision-cache", action="store_true", help="Enable vision-cache optimization (scaffold flag).")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max new tokens for language generation.")
    parser.add_argument("--require-flash-attn", action="store_true", default=True,
                        help="Require FlashAttention-2 at runtime (enabled by default).")
    parser.add_argument("--method", action="append", default=[],
                        help="Additional optimization method tag (repeatable, scaffold only).")
    parser.add_argument("--calib", type=str, default="/home/gdr/gd_vln/workspace/src/InternNav/scripts/realworld/calib/calib_scout.txt",
                        help="Path to calibration file (e.g. calib/calib_scout.txt)")
    args = parser.parse_args()

    if args.mode != "sync":
        print(f"[Server] mode={args.mode} requested, but async path is not implemented yet. Falling back to sync.")
    SERVER_MODE = "sync"
    SERVER_OPT_FLAGS = {
        "kv_cache": bool(args.kv_cache),
        "tensorrt": bool(args.tensorrt),
        "quantization": bool(args.quantization),
        "quant_method": str(args.quant_method),
        "tensorrt_engine": str(args.tensorrt_engine),
        "vision_cache": bool(args.vision_cache),
        "methods": list(args.method),
    }
    if any([args.kv_cache, args.tensorrt, args.quantization, args.vision_cache, len(args.method) > 0]):
        print(f"[Server] Optimization flags enabled (scaffold only): {SERVER_OPT_FLAGS}")

    calib = Calibration(args.calib)
    args.camera_intrinsic = np.array([
        [calib.f_u, 0.0,      calib.c_u, 0.0],
        [0.0,       calib.f_v, calib.c_v, 0.0],
        [0.0,       0.0,      1.0,       0.0],
        [0.0,       0.0,      0.0,       1.0],
    ])
    print(f"[Server] Loaded calib: {args.calib}")
    print(f"[Server] camera_intrinsic fx={calib.f_u:.2f} fy={calib.f_v:.2f} cx={calib.c_u:.2f} cy={calib.c_v:.2f}")
    agent = InternVLAN1AsyncAgent(args)
    # agent.step(
    #     np.zeros((480, 640, 3)),
    #     np.zeros((480, 640)),
    #     np.eye(4),
    #     "hello",
    #     args.camera_intrinsic,
    # )
    # Warmup can trigger CUDA asserts on some checkpoints/runtime combos; skip by default.
    if os.environ.get("INTERNNAV_SERVER_WARMUP", "0") == "1":
        agent.step(
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640)),
            np.eye(4),
            "hello",
            args.camera_intrinsic,
        )
        agent.reset()

    app.run(host='0.0.0.0', port=5802)
