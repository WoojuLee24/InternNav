"""FPV depth → Bird's-Eye View (BEV) conversion.

Camera assumptions (r2r_125cm_0_30 dataset / h1 robot):
  - Camera height h = 1.25 m above ground
  - Camera pitch  = 30° downward
  - USD convention: camera looks along local -Z, +Y=up
  - `distance_to_image_plane` depth is positive Z-depth (metres after denorm)

Coordinate system (torso_link / world-aligned):
  X = forward,  Y = left,  Z = up

Verified rotation matrix (cam → world, quaternion (0.612, 0.354, -0.354, -0.612)):
  R = [[ 0,      0.5,  -0.866],
       [-1,      0,     0    ],
       [ 0,      0.866, 0.5  ]]

Unproject formulas (with D = depth in metres):
  X_w =  D * (0.866 - 0.5*(v-cy)/fy)
  Y_w = -D * (u-cx) / fx
  Z_w =  D * (-0.5 - 0.866*(v-cy)/fy) + h
"""

import os

import torch
import torch.nn.functional as F


def depth_to_bev(
    depth: torch.Tensor,
    cam_height: float = 1.25,
    cam_pitch_deg: float = 30.0,
    fx: float = 204.75,
    fy: float = 272.9,
    cx: float = 112.0,
    cy: float = 112.0,
    bev_range: float = 5.0,
    bev_size: int = 224,
    depth_scale: float = 10.0,
    floor_z_min: float = -0.1,
    floor_z_max: float = 0.3,
) -> torch.Tensor:
    """Convert normalised FPV depth to BEV density map.

    Args:
        depth: [B, H, W] float, normalised [0, 1]  (multiply by depth_scale for metres)
        Returns: [B, bev_size, bev_size] float [0, 1]  (1 = occupied cell)
    """
    import math

    B, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    D = depth * depth_scale  # metres, [B, H, W]

    cos_p = math.cos(math.radians(cam_pitch_deg))   # 0.866
    sin_p = math.sin(math.radians(cam_pitch_deg))   # 0.500

    # pixel grids [H, W]
    v = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    u = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)

    dv = (v - cy) / fy  # [H, W]
    du = (u - cx) / fx  # [H, W]

    # unproject to world frame [B, H, W]
    X_w =  D * (cos_p - sin_p * dv)       # forward
    Y_w = -D * du                           # left
    Z_w =  D * (-sin_p - cos_p * dv) + cam_height  # height

    # floor mask
    mask = (Z_w >= floor_z_min) & (Z_w <= floor_z_max) & (D > 0.1)  # [B, H, W]

    # BEV pixel indices (float)
    scale = bev_size / (2.0 * bev_range)
    i_f = bev_size / 2.0 - X_w * scale   # forward → smaller i
    j_f = bev_size / 2.0 - Y_w * scale   # left    → smaller j

    i_idx = i_f.long().clamp(0, bev_size - 1)
    j_idx = j_f.long().clamp(0, bev_size - 1)

    # scatter into BEV grid
    bev = torch.zeros(B, bev_size, bev_size, device=device, dtype=dtype)
    flat_idx = (i_idx * bev_size + j_idx).view(B, -1)          # [B, H*W]
    flat_mask = mask.view(B, -1).to(dtype)                      # [B, H*W]

    bev.view(B, -1).scatter_add_(1, flat_idx, flat_mask)

    # normalise to [0, 1]
    max_val = bev.amax(dim=(1, 2), keepdim=True).clamp(min=1.0)
    bev = bev / max_val

    return bev  # [B, bev_size, bev_size]


def save_train_debug(
    step: int,
    fpv: torch.Tensor,
    depth: torch.Tensor,
    bev: torch.Tensor,
    save_dir: str,
) -> None:
    """Save 3-panel debug image: FPV RGB | Depth (jet) | BEV.

    Args:
        step:  training step counter
        fpv:   [H, W, 3] float [0,1] — original FPV RGB (HWC)
        depth: [H, W]    float [0,1] — normalised depth
        bev:   [H, W]    float [0,1] — BEV density map
    """
    import numpy as np

    try:
        import cv2
    except ImportError:
        return

    os.makedirs(save_dir, exist_ok=True)

    def _to_uint8(t: torch.Tensor) -> "np.ndarray":
        return (t.detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

    fpv_np = _to_uint8(fpv)                             # [H, W, 3]
    depth_np = _to_uint8(depth)                         # [H, W]
    bev_np = _to_uint8(bev)                             # [H, W]

    depth_jet = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)  # [H, W, 3]
    bev_rgb = cv2.cvtColor(bev_np, cv2.COLOR_GRAY2BGR)         # [H, W, 3]
    fpv_bgr = cv2.cvtColor(fpv_np, cv2.COLOR_RGB2BGR)

    H, W = fpv_np.shape[:2]
    depth_jet_r = cv2.resize(depth_jet, (W, H))
    bev_rgb_r   = cv2.resize(bev_rgb,   (W, H))

    panel = np.concatenate([fpv_bgr, depth_jet_r, bev_rgb_r], axis=1)
    cv2.imwrite(os.path.join(save_dir, f'bev_debug_{step:06d}.jpg'), panel)
