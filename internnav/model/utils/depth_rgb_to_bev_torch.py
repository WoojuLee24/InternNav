"""FPV depth + RGB image → coloured / occupancy Bird's-Eye View (BEV).

Pipeline (mirrors DenseLoc grd_img2cam / grd2cam2world):
  1. xyz_c_norm = K_inv @ [u, v, 1]   — normalised ray in camera frame
  2. xyz_c      = D * xyz_c_norm       — metric 3-D point in camera frame
  3. xyz_w      = R_c2w @ xyz_c + t    — world frame (X=fwd, Y=left, Z=up)
  4. (X_w, Y_w) → BEV cell            — scatter RGB colour or occupancy

Coordinate conventions:
  Camera : x=right, y=down, z=forward  (standard pinhole)
  World  : X=forward, Y=left, Z=up
  Camera is pitched `cam_pitch_deg`° downward at `cam_height` m above ground.
"""

import math

import torch


# ---------------------------------------------------------------------------
# Geometry helpers  (equivalent to DenseLoc grd_img2cam internals)
# ---------------------------------------------------------------------------

def _build_ray_grid(
    H: int,
    W: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalised ray directions via K_inv @ [u, v, 1].

    Mirrors DenseLoc::grd_img2cam —
        camera_k_inv = torch.inverse(camera_k)
        xyz_w = sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)

    Standard pinhole convention: x_c=right, y_c=down, z_c=forward (depth direction).
    Isaac Sim outputs depth as Z-depth (positive = distance in front of camera).
    K_inv @ [u, v, 1] gives [(u-cx)/fx, (v-cy)/fy, 1] — scale by depth D to get xyz_c.

    Returns:
        [3, H, W]  —  x_c=right, y_c=down, z_c=forward
    """
    K_inv = torch.tensor(
        [[1 / fx,      0, -cx / fx],
         [     0, 1 / fy, -cy / fy],
         [     0,      0,        1]],
        dtype=dtype, device=device,
    )  # [3, 3]

    v = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    u = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=0)  # [3, H, W]

    return torch.einsum("ij,jhw->ihw", K_inv, uv1)  # [3, H, W]


def _build_R_c2w(
    cam_pitch_deg: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Camera-to-world rotation matrix [3, 3].

    Camera frame : x=right, y=down, z=forward
    World frame  : X=forward, Y=left, Z=up
    Positive pitch = camera tilts downward.
    """
    cos_p = math.cos(math.radians(cam_pitch_deg))
    sin_p = math.sin(math.radians(cam_pitch_deg))
    return torch.tensor(
        [[ 0, -sin_p,  cos_p],   # X_w (forward)
         [-1,  0,      0    ],   # Y_w (left)
         [ 0, -cos_p, -sin_p]],  # Z_w (up; cam_height added separately)
        dtype=dtype, device=device,
    )  # [3, 3]


# ---------------------------------------------------------------------------
# Unprojection  (mirrors DenseLoc grd2cam2world with gt_depth)
# ---------------------------------------------------------------------------

def unproject_depth(
    depth: torch.Tensor,
    cam_height: float = 1.25,
    cam_pitch_deg: float = 30.0,
    cam_x_offset: float = 0.0,
    cam_z_offset: float = 0.0,
    fx: float = 585.0,
    fy: float = 585.0,
    cx: float = 320.0,
    cy: float = 240.0,
    depth_scale: float = 10.0,
) -> torch.Tensor:
    """Unproject normalised depth map to world-frame 3-D points.

    Mirrors DenseLoc::grd2cam2world (gt_depth branch):
        xyz_grd = xyz_w * depth.permute(0, 2, 3, 1)   — scale ray by depth

    Args:
        depth: [B, H, W] normalised [0, 1]
    Returns:
        xyz_w: [B, H, W, 3]  world frame  (X=fwd, Y=left, Z=up)
    """
    B, H, W = depth.shape
    device, dtype = depth.device, depth.dtype

    D = depth * depth_scale  # metric depth [B, H, W]

    # Step 1 — ray directions in camera frame  (K_inv @ [u, v, 1])
    xyz_c_norm = _build_ray_grid(H, W, fx, fy, cx, cy, device, dtype)  # [3, H, W]

    # Step 2 — 3-D point in camera frame  (scale by metric depth)
    xyz_c = D.unsqueeze(1) * xyz_c_norm.unsqueeze(0)  # [B, 3, H, W]

    # Step 3 — rotate to world frame + camera height translation
    R_c2w = _build_R_c2w(cam_pitch_deg, dtype, device)  # [3, 3]
    xyz_w = torch.einsum("ij,bjhw->bihw", R_c2w, xyz_c)  # [B, 3, H, W]
    xyz_w[:, 2] += cam_height + cam_z_offset  # Z_w = torso_height + cam_z_offset above ground
    xyz_w[:, 0] += cam_x_offset               # X_w: camera is cam_x_offset ahead of robot origin

    return xyz_w.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 3]


# ---------------------------------------------------------------------------
# BEV projection helpers
# ---------------------------------------------------------------------------

def _world_to_bev_idx(
    X_w: torch.Tensor,
    Y_w: torch.Tensor,
    bev_range: float,
    bev_size: int,
):
    """World (X=fwd, Y=left) → BEV integer pixel indices.

    BEV origin is at the robot position; X increases toward the top of the
    image (smaller row index), Y increases toward the left (smaller col index).

    Returns:
        i_idx, j_idx: [B, H, W] long, clamped to [0, bev_size-1]
    """
    scale = bev_size / (2.0 * bev_range)
    i_idx = (bev_size / 2.0 - X_w * scale).long().clamp(0, bev_size - 1)
    j_idx = (bev_size / 2.0 - Y_w * scale).long().clamp(0, bev_size - 1)
    return i_idx, j_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def depth_rgb_to_bev(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    cam_height: float = 1.25,
    cam_pitch_deg: float = 30.0,
    cam_x_offset: float = 0.0,
    cam_z_offset: float = 0.0,
    fx: float = 585.0,
    fy: float = 585.0,
    cx: float = 320.0,
    cy: float = 240.0,
    bev_range: float = 5.0,
    bev_size: int = 224,
    depth_scale: float = 10.0,
    z_min: float = -0.1,
    z_max: float = 2.0,
) -> torch.Tensor:
    """FPV depth + RGB → coloured BEV image.

    Each BEV cell receives the mean RGB of all 3-D points that project into it
    (mirrors DenseLoc's ``bev.color`` mode: get_valid_batch_points2 with RGB).

    Args:
        depth: [B, H, W]    normalised depth [0, 1]
        rgb  : [B, H, W, 3] float RGB [0, 1]  (HWC layout)
    Returns:
        bev  : [B, 3, bev_size, bev_size]  coloured BEV [0, 1]
    """
    B, H, W = depth.shape
    device, dtype = depth.device, depth.dtype

    xyz_w = unproject_depth(depth, cam_height, cam_pitch_deg, cam_x_offset, cam_z_offset, fx, fy, cx, cy, depth_scale)
    X_w, Y_w, Z_w = xyz_w[..., 0], xyz_w[..., 1], xyz_w[..., 2]

    D = depth * depth_scale
    mask = (Z_w >= z_min) & (Z_w <= z_max) & (D > 0.1)  # [B, H, W]

    i_idx, j_idx = _world_to_bev_idx(X_w, Y_w, bev_range, bev_size)
    flat_idx = (i_idx * bev_size + j_idx).view(B, -1)           # [B, H*W]
    flat_mask = mask.view(B, -1).to(dtype)                       # [B, H*W]

    # RGB scatter: sum colour values then divide by count
    rgb_flat = rgb.reshape(B, H * W, 3)                          # [B, H*W, 3]
    rgb_flat = rgb_flat * flat_mask.unsqueeze(-1)                 # zero invalid pixels

    bev_rgb   = torch.zeros(B, 3, bev_size * bev_size, device=device, dtype=dtype)
    bev_count = torch.zeros(B, 1, bev_size * bev_size, device=device, dtype=dtype)

    bev_rgb.scatter_add_(2, flat_idx.unsqueeze(1).expand(-1, 3, -1), rgb_flat.permute(0, 2, 1))
    bev_count.scatter_add_(2, flat_idx.unsqueeze(1), flat_mask.unsqueeze(1))

    bev_rgb   = bev_rgb.view(B, 3, bev_size, bev_size)
    bev_count = bev_count.view(B, 1, bev_size, bev_size).clamp(min=1.0)

    return (bev_rgb / bev_count).clamp(0.0, 1.0)  # [B, 3, bev_size, bev_size]


def depth_to_bev_occ(
    depth: torch.Tensor,
    cam_height: float = 1.25,
    cam_pitch_deg: float = 30.0,
    cam_x_offset: float = 0.0,
    cam_z_offset: float = 0.0,
    fx: float = 585.0,
    fy: float = 585.0,
    cx: float = 320.0,
    cy: float = 240.0,
    bev_range: float = 5.0,
    bev_size: int = 224,
    depth_scale: float = 10.0,
    z_min: float = -0.1,
    z_max: float = 0.3,
) -> torch.Tensor:
    """FPV depth → BEV occupancy map using K_inv-based unprojection.

    Drop-in replacement for depth_to_bev() but restructured with explicit
    K_inv and R_c2w matrices, mirroring DenseLoc's grd_img2cam approach.

    Args:
        depth: [B, H, W]  normalised [0, 1]
    Returns:
        bev  : [B, bev_size, bev_size]  float [0, 1]
    """
    B, H, W = depth.shape
    device, dtype = depth.device, depth.dtype

    xyz_w = unproject_depth(depth, cam_height, cam_pitch_deg, cam_x_offset, cam_z_offset, fx, fy, cx, cy, depth_scale)
    X_w, Y_w, Z_w = xyz_w[..., 0], xyz_w[..., 1], xyz_w[..., 2]

    D = depth * depth_scale
    mask = (Z_w >= z_min) & (Z_w <= z_max) & (D > 0.1)

    i_idx, j_idx = _world_to_bev_idx(X_w, Y_w, bev_range, bev_size)
    flat_idx  = (i_idx * bev_size + j_idx).view(B, -1)
    flat_mask = mask.view(B, -1).to(dtype)

    bev = torch.zeros(B, bev_size, bev_size, device=device, dtype=dtype)
    bev.view(B, -1).scatter_add_(1, flat_idx, flat_mask)

    bev = bev / bev.amax(dim=(1, 2), keepdim=True).clamp(min=1.0)
    return bev  # [B, bev_size, bev_size]
