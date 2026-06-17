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
import os

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


def _build_R_c2w(cam_pitch_deg, dtype, device):
    """Camera-to-world rotation matrix.

    Args:
        cam_pitch_deg: scalar float or [B] tensor (degrees, positive = tilt down).
    Returns:
        [3, 3] for scalar input, [B, 3, 3] for batched input.
    """
    if isinstance(cam_pitch_deg, torch.Tensor) and cam_pitch_deg.ndim > 0:
        rad = cam_pitch_deg.to(device=device, dtype=torch.float64) * (math.pi / 180)
        cos_p = rad.cos().to(dtype)   # [B]
        sin_p = rad.sin().to(dtype)   # [B]
        B = cos_p.shape[0]
        z = torch.zeros(B, device=device, dtype=dtype)
        o = torch.ones(B, device=device, dtype=dtype)
        return torch.stack([
            torch.stack([ z, -sin_p,  cos_p], dim=1),   # X_w (forward)
            torch.stack([-o,  z,      z    ], dim=1),   # Y_w (left)
            torch.stack([ z, -cos_p, -sin_p], dim=1),  # Z_w (up)
        ], dim=1)  # [B, 3, 3]
    cos_p = math.cos(math.radians(float(cam_pitch_deg)))
    sin_p = math.sin(math.radians(float(cam_pitch_deg)))
    return torch.tensor(
        [[ 0, -sin_p,  cos_p],
         [-1,  0,      0    ],
         [ 0, -cos_p, -sin_p]],
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
    R_c2w = _build_R_c2w(cam_pitch_deg, dtype, device)  # [3,3] or [B,3,3]
    if R_c2w.ndim == 3:
        xyz_w = torch.einsum("bij,bjhw->bihw", R_c2w, xyz_c)  # [B, 3, H, W]
    else:
        xyz_w = torch.einsum("ij,bjhw->bihw",  R_c2w, xyz_c)  # [B, 3, H, W]
    if isinstance(cam_height, torch.Tensor) and cam_height.ndim > 0:
        xyz_w[:, 2] += cam_height.to(device=device, dtype=dtype).view(B, 1, 1) + cam_z_offset
    else:
        xyz_w[:, 2] += cam_height + cam_z_offset
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


def _raycast_free_torch(occ_mask: torch.Tensor) -> torch.Tensor:
    """[S, S] bool → [S, S] float32: 0.0=unknown, 0.5=free (raycasted), 1.0=occupied.

    For each occupied cell, casts a ray from the BEV center and marks all
    intermediate unknown cells as free — equivalent to the numpy _raycast_free.
    """
    S = occ_mask.shape[0]
    device = occ_mask.device
    ci = cj = S // 2

    result = torch.zeros(S, S, device=device, dtype=torch.float32)
    result[occ_mask] = 1.0

    occ_r, occ_c = occ_mask.nonzero(as_tuple=True)   # [K]
    K = occ_r.shape[0]
    if K == 0:
        return result

    # n_k: Chebyshev distance from center to each occupied cell
    n_k = torch.maximum((occ_r - ci).abs(), (occ_c - cj).abs()).float().clamp(min=1)  # [K]
    max_n = int(n_k.max().item())

    t_range  = torch.arange(max_n, device=device, dtype=torch.float32).unsqueeze(0)   # [1, max_n]
    n_k_exp  = n_k.unsqueeze(1)                                                        # [K, 1]
    valid    = t_range < n_k_exp                                                        # [K, max_n]

    fracs  = t_range / n_k_exp                                                          # [K, max_n]
    ray_r  = (ci + fracs * (occ_r.float().unsqueeze(1) - ci)).long().clamp(0, S - 1)  # [K, max_n]
    ray_c  = (cj + fracs * (occ_c.float().unsqueeze(1) - cj)).long().clamp(0, S - 1)

    flat_idx = (ray_r * S + ray_c).view(-1)[valid.view(-1)]   # [M]

    free_marker = torch.zeros(S, S, device=device, dtype=torch.bool)
    free_marker.view(-1)[flat_idx] = True

    result[free_marker & (result == 0.0)] = 0.5
    return result


def depth_to_bev_occ_ros2(
    depth: torch.Tensor,
    cam_height: float = 1.25,
    cam_pitch_deg: float = 0.0,
    cam_x_offset: float = 0.0,
    cam_z_offset: float = 0.0,
    fx: float = 585.0,
    fy: float = 585.0,
    cx: float = 320.0,
    cy: float = 240.0,
    bev_range: float = 5.0,
    bev_size: int = 224,
    depth_scale: float = 1.0,
    z_min: float = 0.05,
    z_max: float = 2.0,
) -> torch.Tensor:
    """FPV depth → raycasted BEV occupancy matching build_occupancy_grid (ros2).

    Pipeline:
      1. Unproject depth with floor-based BEV indexing (not truncation).
      2. Exclude OOB points (not clamped to border).
      3. Strict > / < height filter (ros2 convention).
      4. Raycast: for each occupied cell cast a ray from BEV center, mark
         intermediate unknown cells as free.

    Args:
        depth: [B, H, W] metric metres (depth_scale=1.0) or raw (set depth_scale)
    Returns:
        bev: [B, bev_size, bev_size] float32
             0.0 = unknown, 0.5 = free (raycasted), 1.0 = occupied
    """
    B = depth.shape[0]
    device = depth.device

    xyz_w = unproject_depth(depth.float(), cam_height, cam_pitch_deg,
                            cam_x_offset, cam_z_offset, fx, fy, cx, cy, depth_scale)
    X_w, Y_w, Z_w = xyz_w[..., 0], xyz_w[..., 1], xyz_w[..., 2]

    D = depth.float() * depth_scale
    mask = (Z_w > z_min) & (Z_w < z_max) & (D > 0.1)

    scale = bev_size / (2.0 * bev_range)
    i_idx = torch.floor(bev_size / 2.0 - X_w * scale).long()
    j_idx = torch.floor(bev_size / 2.0 - Y_w * scale).long()
    oob = (i_idx < 0) | (i_idx >= bev_size) | (j_idx < 0) | (j_idx >= bev_size)
    mask = mask & ~oob

    flat_idx  = (i_idx.clamp(0, bev_size - 1) * bev_size + j_idx.clamp(0, bev_size - 1)).view(B, -1)
    flat_mask = mask.float().view(B, -1)

    bev = torch.zeros(B, bev_size, bev_size, device=device, dtype=torch.float32)
    bev.view(B, -1).scatter_add_(1, flat_idx, flat_mask)

    return torch.stack([_raycast_free_torch(b > 0) for b in bev], dim=0)


def bev_3state_to_binary(bev: torch.Tensor) -> torch.Tensor:
    """[B,S,S] 3-state (0=unk,0.5=free,1=occ) → [B,3,S,S]: [free_ch, occ_ch, unk_ch]."""
    free = ((bev >= 0.25) & (bev < 0.75)).float()
    occ  = (bev >= 0.75).float()
    unk  = (bev < 0.25).float()
    return torch.stack([free, occ, unk], dim=1)


def bev_3state_to_prob(bev: torch.Tensor) -> torch.Tensor:
    """[B,S,S] 3-state → [B,S,S]: free=0.0, unknown=0.5, occupied=1.0 (probability)."""
    return torch.where(bev >= 0.75, torch.ones_like(bev),
           torch.where(bev >= 0.25, torch.zeros_like(bev),
           torch.full_like(bev, 0.5)))


def bev_3state_to_dist(bev: torch.Tensor) -> torch.Tensor:
    """[B,S,S] 3-state → [B,S,S]: EDT from obstacle(occ|unk), normalized [0,1].

    Conservative: unknown treated as obstacle. Free space near obstacles = 0,
    far from obstacles = 1. Suitable as a continuous costmap for trajectory models.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    result = torch.zeros_like(bev)
    for i in range(bev.shape[0]):
        free_map = ((bev[i] >= 0.25) & (bev[i] < 0.75)).cpu().numpy().astype(np.uint8)
        dist = distance_transform_edt(free_map).astype(np.float32)
        max_d = dist.max()
        if max_d > 0:
            dist /= max_d
        result[i] = torch.from_numpy(dist).to(bev.device)
    return result


def bev_3state_to_dist_sep(bev: torch.Tensor) -> torch.Tensor:
    """[B,S,S] 3-state → [B,3,S,S]: [dist_ch, occ_ch, unk_ch].

    Ch0 (dist): EDT from obstacle(occ|unk), normalized [0,1]. Same as bev_3state_to_dist.
    Ch1 (occ) : 1.0 where occupied, 0.0 elsewhere.
    Ch2 (unk) : 1.0 where unknown,  0.0 elsewhere.
    """
    dist = bev_3state_to_dist(bev)
    occ  = (bev >= 0.75).float()
    unk  = (bev <  0.25).float()
    return torch.stack([dist, occ, unk], dim=1)


def save_image(img, path: str) -> None:
    """Save a single image to disk with cv2.imwrite.

    Accepts torch.Tensor, numpy.ndarray, or PIL.Image in any of:
      - float [0, 1]  or  uint8 [0, 255]
      - shape [H, W], [H, W, C], or [C, H, W]
    RGB inputs are converted to BGR automatically.
    """
    import cv2
    import numpy as np

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # --- convert to float32 numpy ---
    if isinstance(img, torch.Tensor):
        arr = img.detach().float().cpu().numpy()
    else:
        try:
            from PIL import Image as _PILImage
            if isinstance(img, _PILImage.Image):
                arr = np.array(img).astype(np.float32)
                if arr.max() > 1.5:
                    arr = arr / 255.0
            else:
                arr = np.asarray(img, dtype=np.float32)
        except ImportError:
            arr = np.asarray(img, dtype=np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0

    arr = arr.clip(0.0, 1.0)

    # [C, H, W] → [H, W, C]  (heuristic: first dim ≤ 4 and smaller than spatial dims)
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = arr.transpose(1, 2, 0)

    u8 = (arr * 255).astype(np.uint8)

    if u8.ndim == 3 and u8.shape[2] == 3:
        u8 = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    elif u8.ndim == 3 and u8.shape[2] == 1:
        u8 = u8[:, :, 0]

    cv2.imwrite(path, u8)


def save_train_debug(
    step: int,
    fpv: torch.Tensor,
    depth: torch.Tensor,
    bev: torch.Tensor,
    save_dir: str,
) -> None:
    """Save 3-panel debug image: FPV RGB | Depth (jet colourmap) | BEV.

    Args:
        fpv  : [H, W, 3] float [0, 1]
        depth: [H, W]    float [0, 1]
        bev  : [H, W]    float [0, 1]
    """
    import cv2
    import numpy as np

    os.makedirs(save_dir, exist_ok=True)

    def _to_u8(t: torch.Tensor) -> np.ndarray:
        return (t.detach().float().cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)

    fpv_np   = _to_u8(fpv)
    depth_np = _to_u8(depth)
    bev_np   = _to_u8(bev)

    depth_jet = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
    bev_bgr   = cv2.cvtColor(bev_np, cv2.COLOR_GRAY2BGR)
    fpv_bgr   = cv2.cvtColor(fpv_np, cv2.COLOR_RGB2BGR)

    H, W = fpv_bgr.shape[:2]
    panel = np.concatenate(
        [fpv_bgr, cv2.resize(depth_jet, (W, H)), cv2.resize(bev_bgr, (W, H))],
        axis=1,
    )
    cv2.imwrite(os.path.join(save_dir, f"bev_debug_{step:06d}.jpg"), panel)
