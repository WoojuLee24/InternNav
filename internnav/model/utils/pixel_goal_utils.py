import math
import os

import cv2
import numpy as np


def fpv_pixel_to_normalized(pixel_col_row, img_wh):
    """FPV 픽셀 [col, row] -> normalized [x, y] in [0, 1].

    Args:
        pixel_col_row: array-like [col, row] in S2's resized image space
        img_wh: (W, H) of S2's resized image (resize_w, resize_h)
    Returns:
        np.array([x_norm, y_norm]) in [0, 1]
    """
    col = int(pixel_col_row[0])
    row = int(pixel_col_row[1])
    W, H = img_wh
    return np.array([col / W, row / H], dtype=np.float32)


def fpv_pixel_to_bev_normalized(pixel_col_row, depth_hw, fx, fy, cx, cy,
                                 bev_range, bev_size, cam_pitch_deg=0.0):
    """FPV 픽셀 + depth -> BEV normalized coordinate (BEV config 전용).

    Coordinate transform following depth_rgb_to_bev_torch.py:
      xyz_c = D * K_inv @ [u, v, 1]          (camera frame)
      X_w = z_c * cos(pitch) - y_c * sin(pitch)  (world forward)
      Y_w = -x_c                                   (world left)
      bev_row = bev_size/2 - X_w * scale
      bev_col = bev_size/2 - Y_w * scale

    Args:
        pixel_col_row: [col, row] in FPV image space
        depth_hw: (H, W) depth array in metres
        fx, fy, cx, cy: camera intrinsics (scaled to depth_hw resolution)
        bev_range: half-range of BEV grid in metres
        bev_size: BEV grid size in pixels
        cam_pitch_deg: camera pitch angle in degrees (positive = tilted down)
    Returns:
        np.array([bev_col_norm, bev_row_norm]) in [0, 1]
    """
    col = int(pixel_col_row[0])
    row = int(pixel_col_row[1])
    d = float(depth_hw[row, col])
    if d <= 0:
        return np.array([0.5, 0.5], dtype=np.float32)

    x_c = (col - cx) / fx * d
    y_c = (row - cy) / fy * d
    z_c = d

    pitch = math.radians(cam_pitch_deg)
    X_w = z_c * math.cos(pitch) - y_c * math.sin(pitch)
    Y_w = -x_c

    scale = bev_size / (2.0 * bev_range)
    bev_row = int(bev_size / 2.0 - X_w * scale)
    bev_col = int(bev_size / 2.0 - Y_w * scale)
    bev_row = max(0, min(bev_size - 1, bev_row))
    bev_col = max(0, min(bev_size - 1, bev_col))
    return np.array([bev_col / bev_size, bev_row / bev_size], dtype=np.float32)


def visualize_pixel_goal(image_rgb, pixel_col_row, pixel_norm, save_dir,
                          step=None, mode='fpv'):
    """pixel_goal 위치에 마커와 좌표 텍스트를 오버레이하고 저장.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image
        pixel_col_row: [col, row] pixel coordinates on the image
        pixel_norm: [x_norm, y_norm] normalized coordinates
        save_dir: directory to save visualization
        step: episode step for filename suffix
        mode: 'fpv' or 'bev' for filename prefix
    Returns:
        vis: annotated image (RGB uint8)
    """
    vis = image_rgb.copy()
    col = int(pixel_col_row[0])
    row = int(pixel_col_row[1])

    cv2.circle(vis, (col, row), 8, (0, 255, 0), 2)
    cv2.circle(vis, (col, row), 2, (0, 255, 0), -1)

    text1 = f"px: [{col}, {row}]"
    text2 = f"norm: [{pixel_norm[0]:.3f}, {pixel_norm[1]:.3f}]"
    cv2.putText(vis, text1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(vis, text2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    os.makedirs(save_dir, exist_ok=True)
    suffix = f"_{step:06d}" if step is not None else ""
    out_path = os.path.join(save_dir, f"pixel_goal_{mode}{suffix}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return vis
