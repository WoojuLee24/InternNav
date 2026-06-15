"""Validates depth→BEV projection in Habitat VLN-CE environment.

Habitat version of validate_depth_to_bev.py.
- Isaac Sim GT = obs['topdown_rgb'] (500×500 camera)
- Habitat  GT  = TopDownMap measurement cropped ±bev_range m around agent

Saves 4 files per frame (same naming as validate_depth_to_bev.py):
  XXXXX_A_rgb.jpg     FPV RGB
  XXXXX_B_depth.jpg   Depth (JET, blue=near red=far)
  XXXXX_C_bev.jpg     BEV world-aligned (rot by yaw)
  XXXXX_D_gt_bev.jpg  TopDownMap crop + BEV occupancy overlay (cyan)

save_dir  = numpy BEV (depth_rgb_to_bev local impl)
save_dir2 = torch BEV (depth_rgb_to_bev_torch + depth_to_bev_occ)

Usage:
  python3 scripts/eval/validate_depth_to_bev_habitat.py \\
      --config scripts/eval/configs/habitat_dual_system_mini_5090_cfg.py \\
      --save_dir /ws/src/InternNav/data/.../r2r_bev_gt \\
      --max_episodes 3
"""

import importlib.util
import math
import os
import sys
from pathlib import Path

sys.path.append('.')

import cv2
import numpy as np
import torch
from depth_camera_filtering import filter_depth
import argparse

import habitat
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.maps import (
    calculate_meters_per_pixel,
    colorize_topdown_map,
)
from habitat_baselines.config.default import get_config as get_habitat_config

import internnav.habitat_extensions.vln.measures  # noqa: F401 — registry side-effects
from internnav.habitat_extensions.vln.utils import get_intrinsic_matrix
from internnav.model.utils.depth_rgb_to_bev_torch import (
    depth_rgb_to_bev as depth_rgb_to_bev_torch,
    depth_to_bev_occ,
)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_eval_cfg(config_path):
    spec = importlib.util.spec_from_file_location('eval_config_module', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


# ---------------------------------------------------------------------------
# Numpy BEV (v1) — local implementation (same as validate_depth_to_bev.py)
# ---------------------------------------------------------------------------

def depth_rgb_to_bev_np(rgb_np, depth_norm,
                         fx=388.4, fy=388.4, cx=319.5, cy=239.5,
                         cam_height=1.25, cam_pitch_deg=0.0,
                         cam_x_offset=0.0, cam_z_offset=0.0,
                         bev_range=5.0, bev_size=224, depth_scale=10.0,
                         z_min=-0.2, z_max=2.5):
    H, W = depth_norm.shape
    D = depth_norm.clip(0, 1) * depth_scale

    v_g, u_g = np.mgrid[0:H, 0:W].astype(np.float32)
    X_c =  (u_g - cx) / fx * D
    Y_c = -(v_g - cy) / fy * D
    Z_c = -D

    cos_p = math.cos(math.radians(cam_pitch_deg))
    sin_p = math.sin(math.radians(cam_pitch_deg))
    X_w =  sin_p * Y_c - cos_p * Z_c + cam_x_offset
    Y_w = -X_c
    Z_w =  cos_p * Y_c + sin_p * Z_c + cam_height + cam_z_offset

    mask = (Z_w >= z_min) & (Z_w <= z_max) & (D > 0.05)

    scale = bev_size / (2.0 * bev_range)
    i_idx = (bev_size / 2.0 - X_w * scale).astype(int).clip(0, bev_size - 1)
    j_idx = (bev_size / 2.0 - Y_w * scale).astype(int).clip(0, bev_size - 1)

    order = np.argsort(-(X_w * X_w + Y_w * Y_w)[mask])
    bev_out = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    bev_out[i_idx[mask][order], j_idx[mask][order]] = rgb_np[mask][order]
    return bev_out


def depth_to_bev_occ_np(depth_norm,
                         fx=388.4, fy=388.4, cx=319.5, cy=239.5,
                         cam_height=1.25, cam_pitch_deg=0.0,
                         bev_range=5.0, bev_size=224, depth_scale=10.0,
                         z_min=-0.2, z_max=2.5):
    H, W = depth_norm.shape
    D = depth_norm.clip(0, 1) * depth_scale
    v_g, u_g = np.mgrid[0:H, 0:W].astype(np.float32)
    X_c =  (u_g - cx) / fx * D
    Y_c = -(v_g - cy) / fy * D
    Z_c = -D
    cos_p = math.cos(math.radians(cam_pitch_deg))
    sin_p = math.sin(math.radians(cam_pitch_deg))
    X_w =  sin_p * Y_c - cos_p * Z_c
    Y_w = -X_c
    Z_w =  cos_p * Y_c + sin_p * Z_c + cam_height
    mask = (Z_w >= z_min) & (Z_w <= z_max) & (D > 0.05)
    scale = bev_size / (2.0 * bev_range)
    i_idx = (bev_size / 2.0 - X_w * scale).astype(int).clip(0, bev_size - 1)
    j_idx = (bev_size / 2.0 - Y_w * scale).astype(int).clip(0, bev_size - 1)
    occ = np.zeros((bev_size, bev_size), dtype=np.float32)
    np.add.at(occ, (i_idx[mask], j_idx[mask]), 1.0)
    m = occ.max()
    if m > 0:
        occ /= m
    return occ


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_label(img, text, color=(255, 255, 255)):
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def _crop_tdm_around_agent(tdm_rgb, agent_row, agent_col, tdm_mpp, bev_range, size):
    """Crop TopDownMap ±bev_range metres around agent, resize to size×size.

    Returns (crop_rgb, scale_px_per_m) where scale is pixels/metre in the
    returned SIZE×SIZE image — same physical scale as the BEV image.
    """
    H, W = tdm_rgb.shape[:2]
    half_px = int(math.ceil(bev_range / tdm_mpp))

    r0 = agent_row - half_px;  r1 = agent_row + half_px
    c0 = agent_col - half_px;  c1 = agent_col + half_px

    # Pad if out of TDM bounds
    pad_top    = max(0, -r0);   pad_bot  = max(0, r1 - H)
    pad_left   = max(0, -c0);   pad_right = max(0, c1 - W)
    crop_src = tdm_rgb[max(0, r0):min(H, r1), max(0, c0):min(W, c1)]
    crop = cv2.copyMakeBorder(crop_src, pad_top, pad_bot, pad_left, pad_right,
                               cv2.BORDER_CONSTANT, value=(30, 30, 30))

    crop_resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    px_per_m = size / (2.0 * bev_range)
    return crop_resized, px_per_m


# ---------------------------------------------------------------------------
# Panel-image helpers (v1 numpy, v2 torch)
# ---------------------------------------------------------------------------

def _make_panel_c(bev_rgb_uint8, agent_angle, bev_rot_offset_deg, bev_range, size):
    """BEV image world-aligned: rotate so robot-forward aligns with TDM direction.

    Habitat TDM + get_polar_angle source (nav.py):
      phi = cartesian_to_polar(heading_z, -heading_x)[1] = atan2(-heading_x, heading_z)
      → heading_x = -sin(phi),  heading_z = cos(phi)
      Row ~ world Z (row+→+Z), Col ~ world X (col+→+X)

    phi = π:  facing -Z (default) → row-  → UP
    phi = 0:  facing +Z (back)    → row+  → DOWN
    phi = π/2: facing -X (left)   → col-  → LEFT
    phi = -π/2: facing +X (right) → col+  → RIGHT

    cv2 rotation by θ maps top (0,-1) → (sin(θ), -cos(θ)).
    Need (sin(θ), -cos(θ)) = (heading_x, heading_z) = (-sin(φ), cos(φ)).
    Solving: sin(θ)=-sin(φ), cos(θ)=-cos(φ)  →  θ = φ + π  →  rot_deg = degrees(π + agent_angle)
    """
    c = size // 2
    px_m = size / (2.0 * bev_range)
    fa = int(2.5 * px_m)

    img = cv2.resize(cv2.cvtColor(bev_rgb_uint8, cv2.COLOR_RGB2BGR), (size, size))

    # Range circles (drawn before rotation so they stay circular)
    step = max(1, int(bev_range / 5))
    for r_m in range(step, int(bev_range) + 1, step):
        cv2.circle(img, (c, c), int(r_m * px_m), (50, 50, 50), 1, cv2.LINE_AA)

    # Rotate BEV to world frame
    rot_deg = math.degrees(math.pi + agent_angle) + bev_rot_offset_deg
    if abs(rot_deg % 360) > 0.5:
        M = cv2.getRotationMatrix2D((c, c), rot_deg, 1.0)
        img = cv2.warpAffine(img, M, (size, size))

    # Draw FWD arrow AFTER rotation using same formula as Panel D
    # (independent of rotation — directly from agent_angle in TDM space)
    arrow_col = int(c + fa * math.sin(agent_angle))
    arrow_row = int(c + fa * math.cos(agent_angle))
    cv2.arrowedLine(img, (c, c), (arrow_col, arrow_row), (0, 220, 0), 3, tipLength=0.25, line_type=cv2.LINE_AA)
    cv2.putText(img, 'FWD', (arrow_col - 15, arrow_row - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
    cv2.circle(img, (c, c), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), 8, (255, 255, 255), 1, cv2.LINE_AA)

    _add_label(img, f'C: BEV(world)  rot={rot_deg:.0f}deg', color=(200, 255, 200))
    return img


def _make_panel_d(tdm_crop, bev_rgb_world_bgr, agent_angle, px_m, size):
    """TDM crop (world-aligned) + BEV RGB overlay + FWD arrow.

    bev_rgb_world_bgr: already rotated to world frame, BGR, size×size.
    """
    c = size // 2
    fa = int(2.5 * px_m)

    img_bgr = cv2.cvtColor(tdm_crop, cv2.COLOR_RGB2BGR) if tdm_crop.ndim == 3 else cv2.cvtColor(tdm_crop, cv2.COLOR_GRAY2BGR)

    # Blend BEV RGB (already world-aligned) where non-black
    bev_resized = cv2.resize(bev_rgb_world_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    non_black = np.any(bev_resized > 10, axis=2)
    img_bgr[non_black] = (
        0.35 * img_bgr[non_black].astype(np.float32) +
        0.65 * bev_resized[non_black].astype(np.float32)
    ).astype(np.uint8)

    bev_range_m = size / (2.0 * px_m)
    step_d = max(1, int(bev_range_m / 5))
    for r_m in range(step_d, int(bev_range_m) + 1, step_d):
        cv2.circle(img_bgr, (c, c), int(r_m * px_m), (50, 50, 50), 1, cv2.LINE_AA)

    arrow_col = int(c + fa * math.sin(agent_angle))
    arrow_row = int(c + fa * math.cos(agent_angle))
    cv2.arrowedLine(img_bgr, (c, c), (arrow_col, arrow_row), (0, 220, 0), 3, tipLength=0.25, line_type=cv2.LINE_AA)
    cv2.putText(img_bgr, 'FWD', (arrow_col - 15, arrow_row - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
    cv2.circle(img_bgr, (c, c), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img_bgr, (c, c), 8, (255, 255, 255), 1, cv2.LINE_AA)

    _add_label(img_bgr, 'D: GT TDMap + BEV RGB', color=(200, 255, 200))
    return img_bgr


# ---------------------------------------------------------------------------
# Per-frame save functions
# ---------------------------------------------------------------------------

def save_comparison_pair(save_dir, frame_idx,
                         fpv_rgb, depth_norm, tdm_crop, tdm_px_m,
                         agent_angle=0.0, bev_rot_offset_deg=0.0,
                         fx=388.4, fy=388.4, cx=319.5, cy=239.5,
                         cam_height=1.25, cam_pitch_deg=0.0,
                         depth_scale=10.0, bev_range=5.0,
                         z_min=-0.2, z_max=2.5,
                         debug=False, debug_dir=None):
    """v1: numpy BEV."""
    SIZE = tdm_crop.shape[0]
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    fpv = fpv_rgb if fpv_rgb.dtype == np.uint8 else (fpv_rgb.clip(0, 1) * 255).astype(np.uint8)

    # debug=True: f,g,h,i,j,k (a-e reserved for main-loop debug images)
    # debug=False: a,b,c,d
    # debug=True: f,g,(h,i,k debug-only),j,l  |  debug=False: a,b,c,d
    p_rgb, p_depth, p_bev, p_gt = ('f', 'g', 'j', 'l') if debug else ('a', 'b', 'c', 'd')

    img_rgb = cv2.resize(cv2.cvtColor(fpv, cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    _add_label(img_rgb, 'A: FPV RGB')

    depth_u8 = (depth_norm.clip(0, 1) * 255).astype(np.uint8)
    img_depth = cv2.resize(cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET), (SIZE, SIZE))
    _add_label(img_depth, 'B: Depth  blue=near  red=far')

    # h: BEV raw (camera-forward frame)
    bev_rgb = depth_rgb_to_bev_np(
        fpv, depth_norm,
        fx=fx, fy=fy, cx=cx, cy=cy,
        cam_height=cam_height, cam_pitch_deg=cam_pitch_deg,
        bev_range=bev_range, depth_scale=depth_scale,
        z_min=z_min, z_max=z_max,
    )
    if debug:
        cv2.imwrite(str(out / f'{frame_idx:05d}_h_bev_raw.jpg'),
                    cv2.resize(cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2BGR), (SIZE, SIZE)))

    # i: occ raw (camera-forward frame)
    bev_occ = depth_to_bev_occ_np(
        depth_norm,
        fx=fx, fy=fy, cx=cx, cy=cy,
        cam_height=cam_height, cam_pitch_deg=cam_pitch_deg,
        bev_range=bev_range, depth_scale=depth_scale,
        z_min=z_min, z_max=z_max,
    )
    if debug:
        cv2.imwrite(str(out / f'{frame_idx:05d}_i_occ_raw.jpg'),
                    cv2.resize((bev_occ.clip(0, 1) * 255).astype(np.uint8), (SIZE, SIZE)))

    # j: BEV world-aligned (rotated)
    img_bev = _make_panel_c(bev_rgb, agent_angle, bev_rot_offset_deg, bev_range, SIZE)

    # rotate bev_rgb and bev_occ to world frame
    rot_deg = math.degrees(math.pi + agent_angle) + bev_rot_offset_deg
    bev_bgr = cv2.resize(cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    occ_sized = cv2.resize((bev_occ.clip(0, 1) * 255).astype(np.uint8), (SIZE, SIZE),
                            interpolation=cv2.INTER_NEAREST)
    if abs(rot_deg % 360) > 0.5:
        M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), rot_deg, 1.0)
        bev_bgr_world = cv2.warpAffine(bev_bgr, M, (SIZE, SIZE))
        occ_world = cv2.warpAffine(occ_sized, M, (SIZE, SIZE))
    else:
        bev_bgr_world = bev_bgr
        occ_world = occ_sized

    # k: occ world-aligned (rotated)
    if debug:
        cv2.imwrite(str(out / f'{frame_idx:05d}_k_occ_rotated.jpg'), occ_world)

    # l: GT TDMap + BEV RGB overlay
    img_gt = _make_panel_d(tdm_crop, bev_bgr_world, agent_angle, tdm_px_m, SIZE)

    cv2.imwrite(str(out / f'{frame_idx:05d}_{p_rgb}_rgb.jpg'), img_rgb)
    cv2.imwrite(str(out / f'{frame_idx:05d}_{p_depth}_depth.jpg'), img_depth)
    cv2.imwrite(str(out / f'{frame_idx:05d}_{p_bev}_bev.jpg'), img_bev)
    cv2.imwrite(str(out / f'{frame_idx:05d}_{p_gt}_gt_bev.jpg'), img_gt)


def save_comparison_pair_v2(save_dir, frame_idx,
                             fpv_rgb, depth_norm, tdm_crop, tdm_px_m,
                             agent_angle=0.0, bev_rot_offset_deg=0.0,
                             fx=388.4, fy=388.4, cx=319.5, cy=239.5,
                             cam_height=1.25, cam_pitch_deg=0.0,
                             depth_scale=10.0, bev_range=5.0,
                             z_min=-0.2, z_max=2.5,
                             debug=False, debug_dir=None):
    """v2: torch depth_rgb_to_bev_torch + depth_to_bev_occ."""
    SIZE = tdm_crop.shape[0]
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    fpv = fpv_rgb if fpv_rgb.dtype == np.uint8 else (fpv_rgb.clip(0, 1) * 255).astype(np.uint8)

    p_rgb, p_depth, p_bev, p_gt = ('f', 'g', 'j', 'l') if debug else ('a', 'b', 'c', 'd')

    img_rgb = cv2.resize(cv2.cvtColor(fpv, cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    _add_label(img_rgb, 'A: FPV RGB')

    depth_u8 = (depth_norm.clip(0, 1) * 255).astype(np.uint8)
    img_depth = cv2.resize(cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET), (SIZE, SIZE))
    _add_label(img_depth, 'B: Depth  blue=near  red=far')

    depth_t = torch.from_numpy(depth_norm).float().unsqueeze(0)
    rgb_t = torch.from_numpy(fpv.astype(np.float32) / 255.0).unsqueeze(0)

    # h: BEV raw (camera-forward frame)
    with torch.no_grad():
        bev_color = depth_rgb_to_bev_torch(
            depth_t, rgb_t,
            fx=fx, fy=fy, cx=cx, cy=cy,
            cam_height=cam_height, cam_pitch_deg=cam_pitch_deg,
            depth_scale=depth_scale,
            bev_range=bev_range,
            z_min=z_min, z_max=z_max,
        )  # [1,3,224,224]
    bev_color_np = (bev_color[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if debug:
        cv2.imwrite(str(out / f'{frame_idx:05d}_h_bev_raw.jpg'),
                    cv2.resize(cv2.cvtColor(bev_color_np, cv2.COLOR_RGB2BGR), (SIZE, SIZE)))

    # i: occ raw (camera-forward frame)
    with torch.no_grad():
        bev_occ_t = depth_to_bev_occ(
            depth_t,
            fx=fx, fy=fy, cx=cx, cy=cy,
            cam_height=cam_height, cam_pitch_deg=cam_pitch_deg,
            depth_scale=depth_scale,
            bev_range=bev_range,
            z_min=z_min, z_max=z_max,
        )  # [1,224,224]
    bev_occ_np = bev_occ_t[0].numpy()
    if debug:
        cv2.imwrite(str(out / f'{frame_idx:05d}_i_occ_raw.jpg'),
                    cv2.resize((np.clip(bev_occ_np, 0, 1) * 255).astype(np.uint8), (SIZE, SIZE)))

    # j: BEV world-aligned (rotated)
    img_bev = _make_panel_c(bev_color_np, agent_angle, bev_rot_offset_deg, bev_range, SIZE)
    _add_label(img_bev, f'C: BEV_v2(torch)', color=(200, 255, 200))

    # rotate bev and occ to world frame
    rot_deg = math.degrees(math.pi + agent_angle) + bev_rot_offset_deg
    bev_bgr = cv2.resize(cv2.cvtColor(bev_color_np, cv2.COLOR_RGB2BGR), (SIZE, SIZE))
    occ_sized = cv2.resize((np.clip(bev_occ_np, 0, 1) * 255).astype(np.uint8), (SIZE, SIZE),
                            interpolation=cv2.INTER_NEAREST)
    if abs(rot_deg % 360) > 0.5:
        M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), rot_deg, 1.0)
        bev_bgr_world = cv2.warpAffine(bev_bgr, M, (SIZE, SIZE))
        occ_world = cv2.warpAffine(occ_sized, M, (SIZE, SIZE))
    else:
        bev_bgr_world = bev_bgr
        occ_world = occ_sized

    # k: occ world-aligned (rotated)
    if debug:
        cv2.imwrite(str(out / f'{frame_idx:05d}_k_occ_rotated.jpg'), occ_world)

    # l: GT TDMap + BEV RGB overlay
    img_gt = _make_panel_d(tdm_crop, bev_bgr_world, agent_angle, tdm_px_m, SIZE)
    _add_label(img_gt, 'D: GT TDMap + BEV RGB_v2', color=(200, 255, 200))

    cv2.imwrite(str(out / f'{frame_idx:05d}_{p_rgb}_rgb.jpg'), img_rgb)
    cv2.imwrite(str(out / f'{frame_idx:05d}_{p_depth}_depth.jpg'), img_depth)
    cv2.imwrite(str(out / f'{frame_idx:05d}_{p_bev}_bev.jpg'), img_bev)
    cv2.imwrite(str(out / f'{frame_idx:05d}_{p_gt}_gt_bev.jpg'), img_gt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(args):
    eval_cfg = load_eval_cfg(args.config)

    config = get_habitat_config(eval_cfg.env.env_settings['config_path'])
    sim_sensors_cfg = config.habitat.simulator.agents.main_agent.sim_sensors

    depth_cfg = sim_sensors_cfg.depth_sensor
    min_depth  = depth_cfg.min_depth
    max_depth  = depth_cfg.max_depth

    # 1. Defaults from yaml sensor config (vln_r2r_mini.yaml)
    K = get_intrinsic_matrix(depth_cfg)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    cam_height   = float(sim_sensors_cfg.rgb_sensor.position[1])  # 1.25 m
    cam_pitch_deg = 0.0

    # 2. Override from eval_cfg.agent.model_settings (habitat_custom_cam_cfg.py)
    ms = getattr(eval_cfg.agent, 'model_settings', {}) or {}
    K_cfg = ms.get('camera_intrinsic')
    if K_cfg is not None:
        fx, fy = float(K_cfg[0][0]), float(K_cfg[1][1])
        cx, cy = float(K_cfg[0][2]), float(K_cfg[1][2])
    if ms.get('cam_height') is not None:
        cam_height = float(ms['cam_height'])
    if ms.get('cam_pitch_deg') is not None:
        cam_pitch_deg = float(ms['cam_pitch_deg'])

    z_min = args.z_min
    z_max = args.z_max

    print(f'[camera]  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}  '
          f'height={cam_height}m  pitch={cam_pitch_deg}deg  z=[{z_min},{z_max}]')

    # Programmatically add TopDownMap measurement
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update({
            'top_down_map': TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=args.map_resolution,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=True,
                draw_view_points=True,
                draw_goal_positions=True,
                fog_of_war=FogOfWarConfig(draw=True, visibility_dist=args.bev_range, fov=79),
            ),
            'collisions': CollisionsMeasurementConfig(),
        })

    env = habitat.Env(config)

    save_dir  = Path(args.save_dir)
    save_dir2 = save_dir.parent / (save_dir.name + '2')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir2.mkdir(parents=True, exist_ok=True)

    debug_dir  = save_dir  if args.debug else None
    debug_dir2 = save_dir2 if args.debug else None
    if args.debug:
        print(f'[debug] intermediates → {debug_dir}  /  {debug_dir2}')

    tdm_mpp = None   # computed once after first reset
    frame_idx = 0
    ep_count = 0

    while ep_count < args.max_episodes:
        obs = env.reset()
        if obs is None:
            break

        episode = env.current_episode
        scene_id = episode.scene_id.split('/')[-2]
        ep_id = int(episode.episode_id)
        instruction = getattr(getattr(episode, 'instruction', None), 'instruction_text', '')
        print(f'\n[Episode {ep_count+1}/{args.max_episodes}]  {scene_id}  id={ep_id}')
        print(f'  {instruction}')

        if tdm_mpp is None:
            tdm_mpp = calculate_meters_per_pixel(args.map_resolution, pathfinder=env.sim.pathfinder)
            print(f'  TopDownMap {args.map_resolution}px  {tdm_mpp*100:.2f} cm/px')

        follower = ShortestPathFollower(env.sim, goal_radius=0.5, return_one_hot=False)
        goal_pos = episode.goals[0].position

        step_id = 0
        done = False

        while not done and step_id < args.max_steps:
            rgb      = obs['rgb']      # (H,W,3) uint8
            raw_dep  = obs['depth']    # (H,W) float [0,1]

            if args.debug:
                dep_pre_u8 = (raw_dep.reshape(raw_dep.shape[:2]).clip(0, 1) * 255).astype(np.uint8)
                dep_pre_img = cv2.applyColorMap(dep_pre_u8, cv2.COLORMAP_JET)
                cv2.imwrite(str(save_dir / f'{frame_idx:05d}_a_depth_prefilter.jpg'), dep_pre_img)
                cv2.imwrite(str(save_dir2 / f'{frame_idx:05d}_a_depth_prefilter.jpg'), dep_pre_img)
            raw_dep = filter_depth(raw_dep.reshape(raw_dep.shape[:2]), blur_type=None)
            # Keep as normalised [0,1] for BEV functions (depth_scale=max_depth)
            depth_norm = raw_dep.astype(np.float32)

            # TopDownMap metrics
            metrics    = env.get_metrics()
            tdm_info   = metrics.get('top_down_map')
            if tdm_info is None:
                step_id += 1
                action = follower.get_next_action(goal_pos)
                if action is None or action == 0:
                    break
                obs, _, done, *_ = env.step(action)
                continue

            raw_map    = tdm_info['map']
            fog_mask   = tdm_info['fog_of_war_mask']
            agent_row, agent_col = tdm_info['agent_map_coord'][0]
            agent_angle = float(tdm_info['agent_angle'][0])
            tdm_rgb    = colorize_topdown_map(raw_map, fog_mask)  # (H_map, W_map, 3)
            if args.debug:
                raw_map_f = raw_map.astype(np.float32)
                _m = raw_map_f.max()
                raw_map_vis = cv2.applyColorMap(
                    (raw_map_f / _m * 255).astype(np.uint8) if _m > 0 else raw_map_f.astype(np.uint8),
                    cv2.COLORMAP_JET)
                fog_vis = (fog_mask.astype(np.float32) * 255).astype(np.uint8)
                tdm_full_img = cv2.cvtColor(tdm_rgb, cv2.COLOR_RGB2BGR)
                for _d in (save_dir, save_dir2):
                    cv2.imwrite(str(_d / f'{frame_idx:05d}_b_raw_map.jpg'), raw_map_vis)
                    cv2.imwrite(str(_d / f'{frame_idx:05d}_c_fog_mask.jpg'), fog_vis)
                    cv2.imwrite(str(_d / f'{frame_idx:05d}_d_tdm_full.jpg'), tdm_full_img)

            # Crop TDM ±bev_range m around agent (same physical scale as BEV)
            tdm_crop, tdm_px_m = _crop_tdm_around_agent(
                tdm_rgb, agent_row, agent_col,
                tdm_mpp, args.bev_range, size=500,
            )
            if args.debug:
                tdm_crop_img = cv2.cvtColor(tdm_crop, cv2.COLOR_RGB2BGR)
                for _d in (save_dir, save_dir2):
                    cv2.imwrite(str(_d / f'{frame_idx:05d}_e_tdm_crop.jpg'), tdm_crop_img)

            kw = dict(
                fx=fx, fy=fy, cx=cx, cy=cy,
                cam_height=cam_height,
                cam_pitch_deg=cam_pitch_deg,
                depth_scale=max_depth,
                bev_range=args.bev_range,
                z_min=z_min,
                z_max=z_max,
            )

            save_comparison_pair(
                save_dir, frame_idx,
                rgb, depth_norm, tdm_crop, tdm_px_m,
                agent_angle=agent_angle,
                bev_rot_offset_deg=args.bev_rot_offset_deg,
                debug=args.debug, debug_dir=debug_dir,
                **kw,
            )
            save_comparison_pair_v2(
                save_dir2, frame_idx,
                rgb, depth_norm, tdm_crop, tdm_px_m,
                agent_angle=agent_angle,
                bev_rot_offset_deg=args.bev_rot_offset_deg,
                debug=args.debug, debug_dir=debug_dir2,
                **kw,
            )

            print(f'  step {step_id:03d}  frame {frame_idx:05d}  '
                  f'agent_angle={math.degrees(agent_angle):.1f}deg')
            frame_idx += 1

            action = follower.get_next_action(goal_pos)
            if action is None or action == 0:
                print(f'  Oracle STOP at step {step_id}')
                break
            obs, _, done, *_ = env.step(action)
            step_id += 1

        ep_count += 1
        print(f'  Episode done — {frame_idx} frames total')

    env.close()
    print(f'\nSaved {frame_idx} pairs to {save_dir}  (v1: numpy)')
    print(f'Saved {frame_idx} pairs to {save_dir2}  (v2: torch)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='Eval config .py (e.g. habitat_dual_system_mini_5090_cfg.py or habitat_custom_cam_cfg.py)')
    parser.add_argument('--save_dir', default='./logs/r2r_bev_gt')
    parser.add_argument('--max_episodes', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Max steps per episode for oracle agent')
    parser.add_argument('--bev_range', type=float, default=5.0,
                        help='BEV covers ±bev_range metres around robot')
    parser.add_argument('--bev_rot_offset_deg', type=float, default=0.0,
                        help='Extra CCW rotation (deg) for BEV world-alignment fine-tuning')
    parser.add_argument('--map_resolution', type=int, default=1024,
                        help='TopDownMap resolution in pixels')
    # Camera params read from config model_settings; z_min/z_max are debug controls
    parser.add_argument('--z_min', type=float, default=-0.2,
                        help='Min world Z [m] for BEV occupancy (default -0.2)')
    parser.add_argument('--z_max', type=float, default=2.5,
                        help='Max world Z [m] for BEV occupancy (default 2.5)')
    parser.add_argument('--debug', action='store_true',
                        help='Save intermediate images per frame: X1 depth pre-filter, X2 full TDM, X3 TDM crop, E BEV raw (pre-rotation), F occupancy raw (pre-rotation)')
    args = parser.parse_args()
    run_validation(args)


if __name__ == '__main__':
    main()
