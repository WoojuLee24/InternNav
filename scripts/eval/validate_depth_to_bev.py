"""
Validates depth_to_bev() by overlaying the depth-derived BEV against the
GT topdown_camera_500 image captured in Isaac Sim.

Teleports the robot to each GT reference_path waypoint, captures:
  - obs['depth']       — FPV depth (normalised [0,1])
  - obs['topdown_rgb'] — 500×500 GT bird's-eye view

Converts FPV depth → BEV via depth_to_bev(), then saves a 4-panel image:
  [FPV RGB | Depth (jet) | BEV | GT topdown + BEV overlay]

Usage:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/validate_depth_to_bev.py \
        --config scripts/eval/configs/h1_custom_cam_cfg.py \
        --save_dir vln_pe/traj_data/r2r_debug \
        --max_episodes 3
"""
import sys

sys.path.append('.')
sys.path.append('./third_party/diffusion-policy')

import argparse
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import torch

from internnav.model.utils.depth_rgb_to_bev2 import (
    depth_rgb_to_bev as depth_rgb_to_bev_torch,
    depth_to_bev_occ,
)


def load_eval_cfg(config_path):
    spec = importlib.util.spec_from_file_location('eval_config_module', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


def step_until_finish(env, action, robot_name):
    while True:
        obs_list, _, terminated, _, _ = env.step(action)
        obs = obs_list[0][robot_name]
        if obs['finish_action'] or terminated[0]:
            return obs, terminated[0]


def get_yaw(quat):
    from omni.isaac.core.utils.rotations import quat_to_euler_angles
    _, _, yaw = quat_to_euler_angles(np.array(quat))
    return float(yaw)


def teleport_to(env, robot_name, position, yaw_rad, obs):
    """Rotate then forward-flash to approximate a GT waypoint."""
    TURN_ANGLE_RAD = math.radians(15.0)
    CLOSE_ENOUGH = 0.15
    MAX_STEPS = 300
    terminated = False
    for _ in range(MAX_STEPS):
        curr_pos = np.array(obs['globalgps'])[:2]
        tgt_pos = np.array(position)[:2]
        dist = np.linalg.norm(tgt_pos - curr_pos)
        if dist < CLOSE_ENOUGH:
            break
        dp = tgt_pos - curr_pos
        delta = math.atan2(dp[1], dp[0]) - get_yaw(obs['globalrotation'])
        delta = ((delta + math.pi) % (2 * math.pi)) - math.pi
        action_idx = (2 if delta > 0 else 3) if abs(delta) > TURN_ANGLE_RAD * 0.5 else 1
        obs, terminated = step_until_finish(env, [{robot_name: {'move_by_flash': [action_idx]}}], robot_name)
        if terminated:
            break
    return obs, terminated


def depth_rgb_to_bev(rgb_np, depth_np,
                     fx=585.0, fy=585.0, cx=320.0, cy=240.0,
                     cam_height=1.25, cam_pitch_deg=30.0,
                     cam_x_offset=0.0, cam_z_offset=0.0,
                     bev_range=5.0, bev_size=224, depth_scale=10.0,
                     z_min=-0.2, z_max=2.5):
    """depth image → BEV RGB via explicit 4-step pipeline.

    Step 1  K⁻¹        : pixel (u,v) + Z-depth D → camera frame 3-D point
    Step 2  R_cam→robot : camera frame → robot frame  (X=fwd, Y=left, Z=up)
    Step 3  BEV project : keep floor points (z_min ≤ Z_w ≤ z_max)
    Step 4  image plane : (X_w, Y_w) → BEV pixel (top=fwd, left=left)

    Camera convention (Isaac Sim / OpenGL):
      looks along local -Z,  +Y = up,  +X = right
    R_cam→robot (yaw=0, pitch down by cam_pitch_deg):
      [[ 0,     sin_p, -cos_p ],
       [-1,     0,      0     ],
       [ 0,     cos_p,  sin_p ]]

    Returns [bev_size, bev_size, 3] uint8.
    """
    H, W = depth_np.shape
    D = depth_np.clip(0, 1) * depth_scale          # Z-depth [m]

    # Step 1: K⁻¹ — pixel → camera frame
    v_g, u_g = np.mgrid[0:H, 0:W].astype(np.float32)
    X_c =  (u_g - cx) / fx * D    # image +u = cam +X
    Y_c = -(v_g - cy) / fy * D    # image +v = cam -Y  →  cam +Y = -(v-cy)/fy * D
    Z_c = -D                        # cam looks -Z  →  Z_c = -D

    # Step 2: R_cam→robot — camera frame → robot/world frame
    cos_p = math.cos(math.radians(cam_pitch_deg))
    sin_p = math.sin(math.radians(cam_pitch_deg))
    X_w =  sin_p * Y_c - cos_p * Z_c + cam_x_offset           # forward (+ camera offset from robot origin)
    Y_w = -X_c                                                  # left
    Z_w =  cos_p * Y_c + sin_p * Z_c + cam_height + cam_z_offset  # torso_height + cam_z_offset above ground

    # Step 3: BEV projection — keep floor points
    mask = (Z_w >= z_min) & (Z_w <= z_max) & (D > 0.05)

    # Step 4: (X_w, Y_w) → BEV image plane
    scale = bev_size / (2.0 * bev_range)
    i_idx = (bev_size / 2.0 - X_w * scale).astype(int).clip(0, bev_size - 1)
    j_idx = (bev_size / 2.0 - Y_w * scale).astype(int).clip(0, bev_size - 1)

    # write far → near so closer pixels overwrite farther ones
    order = np.argsort(-(X_w * X_w + Y_w * Y_w)[mask])
    bev_out = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    bev_out[i_idx[mask][order], j_idx[mask][order]] = rgb_np[mask][order]
    return bev_out


def _add_label(img, text, color=(255, 255, 255)):
    """Write label text at top-left of img (in-place)."""
    import cv2
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def _draw_bev_annotations(bev_bgr, size):
    """Draw robot marker, forward arrow, range circles, and axis labels on BEV image."""
    import cv2
    cx = cy = size // 2
    px_per_m = size / 10.0  # BEV covers 10 m total (±5 m)

    # range circles at 1 m, 3 m, 5 m
    for r_m, lbl in [(1, '1m'), (3, '3m'), (5, '5m')]:
        r_px = int(r_m * px_per_m)
        cv2.circle(bev_bgr, (cx, cy), r_px, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(bev_bgr, lbl, (cx + r_px + 3, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    # forward arrow (row↑ = robot forward in BEV)
    fwd_len = int(2.5 * px_per_m)
    cv2.arrowedLine(bev_bgr, (cx, cy), (cx, cy - fwd_len),
                    (0, 220, 0), 2, tipLength=0.25, line_type=cv2.LINE_AA)

    # axis labels
    cv2.putText(bev_bgr, 'FWD', (cx - 16, cy - fwd_len - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
    cv2.putText(bev_bgr, 'L', (4, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(bev_bgr, 'R', (size - 14, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # robot dot
    cv2.circle(bev_bgr, (cx, cy), 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(bev_bgr, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)


def save_comparison_pair(save_dir, frame_idx, fpv_rgb, depth_norm, topdown_rgb,
                         robot_yaw_rad=0.0, bev_rot_offset_deg=0.0,
                         fx=585.0, fy=585.0, cx=320.0, cy=240.0,
                         cam_height=1.25, cam_x_offset=0.0, cam_z_offset=0.0):
    """Save 4 images per frame: A_rgb, B_depth, C_bev, D_gt_bev.

    C (BEV) and D (GT) are both world-aligned (top = world +X assumed).
    FWD arrow on C is drawn in robot frame *before* rotation so it always
    matches BEV content regardless of coordinate-system assumptions.
    """
    import cv2
    GT_APERTURE = 200  # topdown_camera_500 horizontal aperture; matches world_to_pixel() in path_plan.py
    SIZE = topdown_rgb.shape[1]
    c = SIZE // 2
    px_m = 10 / GT_APERTURE * SIZE       # 25 px/m — same formula as world_to_pixel: 10/aperture*width
    gt_range = GT_APERTURE / 20          # 10m — GT camera radius
    fa = int(gt_range * 0.25 * px_m)     # 2.5m forward arrow

    fpv = fpv_rgb if fpv_rgb.dtype == np.uint8 \
        else (fpv_rgb.clip(0, 1) * 255).astype(np.uint8)

    # --- A: FPV RGB ---
    img_a = cv2.resize(cv2.cvtColor(fpv, cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    _add_label(img_a, 'A: FPV RGB')

    # --- B: GT depth (JET, blue=near red=far) ---
    depth_u8 = (depth_norm.clip(0, 1) * 255).astype(np.uint8)
    img_b = cv2.resize(cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET), (SIZE, SIZE))
    _add_label(img_b, 'B: GT depth  blue=near  red=far')

    # --- C: RGB→BEV world-aligned ---
    # Step 1: project in robot frame (z_max=0.3 = floor only)
    bev_rgb = depth_rgb_to_bev(fpv, depth_norm,
                                fx=fx, fy=fy, cx=cx, cy=cy,
                                cam_height=cam_height, cam_x_offset=cam_x_offset,
                                cam_z_offset=cam_z_offset,
                                bev_range=gt_range, z_min=-0.15, z_max=0.3)
    img_c = cv2.resize(cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    # Step 2: annotate in robot frame — FWD arrow always points straight up
    for r_m in [1, 3, 5]:
        cv2.circle(img_c, (c, c), int(r_m * px_m), (50, 50, 50), 1, cv2.LINE_AA)
    cv2.arrowedLine(img_c, (c, c), (c, c - fa), (0, 220, 0), 3,
                    tipLength=0.25, line_type=cv2.LINE_AA)
    cv2.putText(img_c, 'FWD', (c - 15, c - fa - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
    cv2.circle(img_c, (c, c), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img_c, (c, c), 8, (255, 255, 255), 1, cv2.LINE_AA)
    # Step 3: rotate image+arrow together → world-align (top = world +X, left = world +Y)
    rot_bev = math.degrees(robot_yaw_rad) + bev_rot_offset_deg
    if abs(rot_bev) > 0.5:
        M = cv2.getRotationMatrix2D((c, c), rot_bev, 1.0)
        img_c = cv2.warpAffine(img_c, M, (SIZE, SIZE))
    _add_label(img_c, f'C: BEV(world,floor)  rot={rot_bev:.0f}deg', color=(200, 255, 200))

    # --- D: GT topdown + robot forward arrow ---
    # GT is world-aligned; draw robot fwd arrow using yaw (world +X = image up assumed)
    gt = topdown_rgb if topdown_rgb.dtype == np.uint8 \
        else (topdown_rgb.clip(0, 1) * 255).astype(np.uint8)
    img_d = cv2.resize(cv2.cvtColor(gt, cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    total_yaw = robot_yaw_rad + math.radians(bev_rot_offset_deg)
    for r_m in [1, 3, 5]:
        cv2.circle(img_d, (c, c), int(r_m * px_m), (50, 50, 50), 1, cv2.LINE_AA)
    fdc = c - int(math.sin(total_yaw) * fa)   # world +Y = left = -col
    fdr = c - int(math.cos(total_yaw) * fa)   # world +X = up   = -row
    cv2.arrowedLine(img_d, (c, c), (fdc, fdr), (0, 220, 0), 3,
                    tipLength=0.25, line_type=cv2.LINE_AA)
    cv2.putText(img_d, 'FWD', (fdc - 15, fdr - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
    cv2.circle(img_d, (c, c), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img_d, (c, c), 8, (255, 255, 255), 1, cv2.LINE_AA)
    _add_label(img_d, 'D: GT topdown (world)', color=(255, 255, 200))

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f'{frame_idx:05d}_A_rgb.jpg'), img_a)
    cv2.imwrite(str(out / f'{frame_idx:05d}_B_depth.jpg'), img_b)
    cv2.imwrite(str(out / f'{frame_idx:05d}_C_bev.jpg'), img_c)
    cv2.imwrite(str(out / f'{frame_idx:05d}_D_gt_bev.jpg'), img_d)


def save_comparison_pair_v2(save_dir, frame_idx, fpv_rgb, depth_norm, topdown_rgb,
                           robot_yaw_rad=0.0, bev_rot_offset_deg=0.0,
                           fx=585.0, fy=585.0, cx=320.0, cy=240.0,
                           cam_height=1.25, cam_x_offset=0.0, cam_z_offset=0.0):
    """depth_rgb_to_bev2 torch 버전으로 생성한 BEV 이미지 저장.

    A: FPV RGB
    B: Depth (jet)
    C: depth_rgb_to_bev_torch (colored BEV)  ← 새 torch 구현
    D: depth_to_bev_occ (occupancy BEV) + GT topdown overlay
    """
    import cv2
    GT_APERTURE = 200
    SIZE = topdown_rgb.shape[1]
    c = SIZE // 2
    px_m = 10 / GT_APERTURE * SIZE
    gt_range = GT_APERTURE / 20
    fa = int(gt_range * 0.25 * px_m)

    H, W = depth_norm.shape
    fpv = fpv_rgb if fpv_rgb.dtype == np.uint8 \
        else (fpv_rgb.clip(0, 1) * 255).astype(np.uint8)

    # --- A: FPV RGB ---
    img_a = cv2.resize(cv2.cvtColor(fpv, cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    _add_label(img_a, 'A: FPV RGB')

    # --- B: Depth (jet) ---
    depth_u8 = (depth_norm.clip(0, 1) * 255).astype(np.uint8)
    img_b = cv2.resize(cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET), (SIZE, SIZE))
    _add_label(img_b, 'B: GT depth  blue=near  red=far')

    # --- C: depth_rgb_to_bev_torch (colored BEV) ---
    BEV_RANGE = 10.0   # match GT topdown ±10m so both C and D use 25 px/m
    bev_px_m = SIZE / (2.0 * BEV_RANGE)   # 25 px/m — same as GT after resize
    depth_t = torch.from_numpy(depth_norm).float().unsqueeze(0)   # [1, H, W]
    rgb_t = torch.from_numpy(fpv.astype(np.float32) / 255.0).unsqueeze(0)  # [1, H, W, 3]
    with torch.no_grad():
        bev_color = depth_rgb_to_bev_torch(
            depth_t, rgb_t,
            fx=fx, fy=fy, cx=cx, cy=cy,
            cam_height=cam_height, cam_x_offset=cam_x_offset, cam_z_offset=cam_z_offset,
            bev_range=BEV_RANGE,
            z_min=-0.15, z_max=2.0,
        )  # [1, 3, 224, 224]
    bev_color_np = (bev_color[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [224,224,3] RGB
    img_c = cv2.resize(cv2.cvtColor(bev_color_np, cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    bev_fa = int(2.5 * bev_px_m)
    for r_m in [1, 3, 5]:
        cv2.circle(img_c, (c, c), int(r_m * bev_px_m), (50, 50, 50), 1, cv2.LINE_AA)
    cv2.arrowedLine(img_c, (c, c), (c, c - bev_fa), (0, 220, 0), 3, tipLength=0.25, line_type=cv2.LINE_AA)
    cv2.putText(img_c, 'FWD', (c - 15, c - bev_fa - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
    cv2.circle(img_c, (c, c), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img_c, (c, c), 8, (255, 255, 255), 1, cv2.LINE_AA)
    rot_bev = math.degrees(robot_yaw_rad) + bev_rot_offset_deg
    if abs(rot_bev) > 0.5:
        M = cv2.getRotationMatrix2D((c, c), rot_bev, 1.0)
        img_c = cv2.warpAffine(img_c, M, (SIZE, SIZE))
    _add_label(img_c, f'C: BEV_v2(colored) rot={rot_bev:.0f}deg', color=(200, 255, 200))

    # --- D: depth_to_bev_occ overlay on GT topdown ---
    with torch.no_grad():
        bev_occ = depth_to_bev_occ(
            depth_t,
            fx=fx, fy=fy, cx=cx, cy=cy,
            cam_height=cam_height, cam_x_offset=cam_x_offset, cam_z_offset=cam_z_offset,
            bev_range=BEV_RANGE,
            z_min=-0.15, z_max=2.0,
        )  # [1, 224, 224]
    bev_occ_np = (bev_occ[0].numpy().clip(0, 1) * 255).astype(np.uint8)  # [224, 224]
    gt = topdown_rgb if topdown_rgb.dtype == np.uint8 \
        else (topdown_rgb.clip(0, 1) * 255).astype(np.uint8)
    img_d = cv2.resize(cv2.cvtColor(gt, cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    total_yaw = robot_yaw_rad + math.radians(bev_rot_offset_deg)
    for r_m in [1, 3, 5]:
        cv2.circle(img_d, (c, c), int(r_m * px_m), (50, 50, 50), 1, cv2.LINE_AA)
    fdc = c - int(math.sin(total_yaw) * fa)
    fdr = c - int(math.cos(total_yaw) * fa)
    cv2.arrowedLine(img_d, (c, c), (fdc, fdr), (0, 220, 0), 3, tipLength=0.25, line_type=cv2.LINE_AA)
    cv2.circle(img_d, (c, c), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img_d, (c, c), 8, (255, 255, 255), 1, cv2.LINE_AA)
    bev_scaled = cv2.resize(bev_occ_np, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    occupied = bev_scaled > 25
    img_d[occupied] = (
        0.35 * img_d[occupied].astype(np.float32) +
        0.65 * np.array([255, 255, 0], dtype=np.float32)
    ).astype(np.uint8)
    _add_label(img_d, 'D: GT topdown + occ_v2', color=(255, 255, 200))

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f'{frame_idx:05d}_A_rgb.jpg'), img_a)
    cv2.imwrite(str(out / f'{frame_idx:05d}_B_depth.jpg'), img_b)
    cv2.imwrite(str(out / f'{frame_idx:05d}_C_bev.jpg'), img_c)
    cv2.imwrite(str(out / f'{frame_idx:05d}_D_gt_bev.jpg'), img_d)


def save_panel(save_path, fpv_rgb, depth_norm, bev, topdown_rgb,
               robot_yaw_rad=0.0, bev_rot_offset_deg=0.0):
    """4-panel: FPV RGB | Depth (GT sensor) | RGB→BEV | GT(robot-frame) + BEV overlay."""
    import cv2

    SIZE = 500

    # --- Panel 1: FPV RGB ---
    fpv = fpv_rgb if fpv_rgb.dtype == np.uint8 else (fpv_rgb.clip(0, 1) * 255).astype(np.uint8)
    fpv_bgr = cv2.resize(cv2.cvtColor(fpv, cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    _add_label(fpv_bgr, '1) FPV RGB')

    # --- Panel 2: GT depth sensor (simulator output, blue=near red=far) ---
    depth_u8 = (depth_norm.clip(0, 1) * 255).astype(np.uint8)
    depth_jet = cv2.resize(cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET), (SIZE, SIZE))
    _add_label(depth_jet, '2) GT depth sensor  blue=near  red=far')

    # --- Panel 3: FPV RGB projected to BEV via depth (robot-relative) ---
    bev_rgb_224 = depth_rgb_to_bev(fpv, depth_norm)            # [224,224,3] uint8
    bev_rgb_bgr = cv2.resize(cv2.cvtColor(bev_rgb_224, cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    _draw_bev_annotations(bev_rgb_bgr, SIZE)
    _add_label(bev_rgb_bgr, '3) RGB->BEV  red=robot  green=fwd', color=(200, 255, 200))

    # --- Panel 4: GT topdown rotated to robot frame + BEV overlay ---
    # Strategy: GT image top = world +X (fixed quaternion). Rotate GT by +(yaw + offset)
    # CCW to put robot-forward at image top, matching BEV frame.
    gt = topdown_rgb if topdown_rgb.dtype == np.uint8 else (topdown_rgb.clip(0, 1) * 255).astype(np.uint8)
    gt_bgr = cv2.resize(cv2.cvtColor(gt, cv2.COLOR_RGB2BGR), (SIZE, SIZE))

    rot_deg = math.degrees(robot_yaw_rad) + bev_rot_offset_deg
    if abs(rot_deg) > 0.5:
        M = cv2.getRotationMatrix2D((SIZE // 2, SIZE // 2), rot_deg, 1.0)
        gt_bgr = cv2.warpAffine(gt_bgr, M, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)

    # Binary BEV occupancy overlay as cyan
    bev_u8 = (bev.clip(0, 1) * 255).astype(np.uint8)
    bev_scaled = cv2.resize(bev_u8, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    occupied = bev_scaled > 25

    overlay = gt_bgr.copy()
    overlay[occupied] = (
        0.35 * gt_bgr[occupied].astype(np.float32) +
        0.65 * np.array([255, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    cx = cy = SIZE // 2
    cv2.circle(overlay, (cx, cy), 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)
    _add_label(overlay, '4) GT(robot-frame)+BEV', color=(255, 255, 200))
    cv2.putText(overlay, f'cyan=BEV  red=robot  rot={rot_deg:.0f}deg', (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, f'cyan=BEV  red=robot  rot={rot_deg:.0f}deg', (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)

    panel = np.concatenate([fpv_bgr, depth_jet, bev_rgb_bgr, overlay], axis=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, panel)


def run_validation(args):
    eval_cfg = load_eval_cfg(args.config)

    # Force topdown sensor on (needed to get GT map obs)
    eval_cfg.eval_settings['vis_output'] = True
    eval_cfg.eval_settings['use_agent_server'] = False

    from internnav.configs.evaluator.vln_default_config import get_config
    from internnav.env.base import Env

    evaluator_cfg = get_config(eval_cfg)

    # Inject fields that the evaluator normally sets before Env.init()
    evaluator_cfg.env.env_settings['dataset'] = evaluator_cfg.dataset
    evaluator_cfg.env.env_settings.setdefault('rank', 0)
    evaluator_cfg.env.env_settings.setdefault('local_rank', 0)
    evaluator_cfg.env.env_settings.setdefault('world_size', 1)

    env = Env.init(evaluator_cfg.env, evaluator_cfg.task)

    robot_name = evaluator_cfg.task.robot_name
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir2 = save_dir.parent / (save_dir.name + '2')
    save_dir2.mkdir(parents=True, exist_ok=True)

    # camera intrinsics from eval_cfg (if provided), else fall back to defaults
    cam_intr = eval_cfg.agent.model_settings.get('camera_intrinsic', None)
    if cam_intr is not None:
        _fx = float(cam_intr[0][0])
        _fy = float(cam_intr[1][1])
        _cx = float(cam_intr[0][2])
        _cy = float(cam_intr[1][2])
    else:
        _fx, _fy, _cx, _cy = 585.0, 585.0, 320.0, 240.0

    # camera_translation = (x_fwd, y_left, z_up) offset from torso_link
    # cam_height = torso_link height above floor (robot constant from USD)
    # cam_z_offset = camera_translation[2] (camera z above torso_link)
    # total camera height = cam_height + cam_z_offset  (e.g. 1.05 + 0.2 = 1.25 m)
    _TORSO_LINK_HEIGHT = 1.05  # from USD: h1_1_25 camera z=0.2 above torso → torso at 1.05m
    cam_translation = getattr(eval_cfg.task, 'camera_translation', None)
    _cam_x_offset = float(cam_translation[0]) if cam_translation is not None else 0.0
    _cam_z_offset = float(cam_translation[2]) if cam_translation is not None else 0.0
    print(f'camera_translation={cam_translation}  '
          f'cam_height={_TORSO_LINK_HEIGHT}  cam_z_offset={_cam_z_offset}  '
          f'total={_TORSO_LINK_HEIGHT + _cam_z_offset:.3f}m  cam_x_offset={_cam_x_offset}')

    frame_idx = 0
    ep_count = 0

    # Initial reset — returns (obs_list, reset_info_list)
    obs_list, reset_info_list = env.reset()

    while True:
        info = reset_info_list[0] if reset_info_list else None
        if info is None:
            break
        if args.max_episodes and ep_count >= args.max_episodes:
            break

        ref_path = info.data.get('reference_path', [])
        obs = obs_list[0][robot_name]

        # warm-up stand_still step (same as eval_gt_collect)
        obs, _ = step_until_finish(env, [{robot_name: {'stand_still': []}}], robot_name)

        for wp_idx, wp in enumerate(ref_path[1:], start=1):
            obs, terminated = teleport_to(env, robot_name, wp, 0.0, obs)

            depth_norm = obs.get('depth')
            rgb = obs.get('rgb')
            topdown_rgb = obs.get('topdown_rgb')

            if depth_norm is not None and topdown_rgb is not None:
                if depth_norm.ndim == 3:
                    depth_norm = depth_norm[..., 0]

                fpv_rgb = rgb[..., :3] if rgb is not None \
                    else np.zeros((224, 224, 3), dtype=np.uint8)
                robot_yaw = get_yaw(obs['globalrotation'])
                print(f'  ep{ep_count:03d} wp{wp_idx:03d}  '
                      f'yaw={math.degrees(robot_yaw):.1f}deg  '
                      f'rot={-(math.degrees(robot_yaw)+args.bev_rot_offset_deg):.1f}deg')

                save_comparison_pair(
                    save_dir, frame_idx,
                    fpv_rgb, depth_norm, topdown_rgb,
                    robot_yaw_rad=robot_yaw,
                    bev_rot_offset_deg=args.bev_rot_offset_deg,
                    fx=_fx, fy=_fy, cx=_cx, cy=_cy,
                    cam_height=_TORSO_LINK_HEIGHT,
                    cam_x_offset=_cam_x_offset,
                    cam_z_offset=_cam_z_offset,
                )
                save_comparison_pair_v2(
                    save_dir2, frame_idx,
                    fpv_rgb, depth_norm, topdown_rgb,
                    robot_yaw_rad=robot_yaw,
                    bev_rot_offset_deg=args.bev_rot_offset_deg,
                    fx=_fx, fy=_fy, cx=_cx, cy=_cy,
                    cam_height=_TORSO_LINK_HEIGHT,
                    cam_x_offset=_cam_x_offset,
                    cam_z_offset=_cam_z_offset,
                )
                frame_idx += 1

            if terminated:
                break

        ep_count += 1
        print(f'Episode {ep_count} done — {frame_idx} pairs saved so far')

        obs_list, reset_info_list = env.reset([0])

    env.close()
    print(f'Saved {frame_idx} pairs to {save_dir}  (v1: numpy)')
    print(f'Saved {frame_idx} pairs to {save_dir2}  (v2: torch depth_rgb_to_bev2)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Eval config py file (must have topdown sensor)')
    parser.add_argument('--save_dir', default='vln_pe/traj_data/r2r_gt_bev')
    parser.add_argument('--max_episodes', type=int, default=3)
    parser.add_argument('--bev_rot_offset_deg', type=float, default=0.0,
                        help='Extra rotation (deg, CCW) added to yaw when aligning GT to robot frame. '
                             'Adjust if cyan BEV is still rotated relative to GT.')
    args = parser.parse_args()
    run_validation(args)


if __name__ == '__main__':
    main()
