"""Camera geometry for the sim VLN bridge (pure numpy, no ROS) — unit-tested in tests/.

Pixel convention: (u, v) = (col, row), pixel centers at integer coordinates, so a resize
by factor s maps u -> (u + 0.5) * s - 0.5.  S2 pixel goals come as [row, col]
(internvla_n1_agent_realworld.py: pixel_goal = [coord[1], coord[0]]).

Frames: camera *optical* frame (x right, y down, z forward) as published by the sim
(camera_color_optical_frame).  Transforms are 4x4 homogeneous T_a_b (maps b -> a).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_K(cls, K, width, height):
        K = np.asarray(K, np.float64).reshape(3, 3)
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2], int(width), int(height))

    def K(self):
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


@dataclass
class CropResize:
    """Center crop to the target aspect ratio, then uniform resize (no squash)."""

    x0: int
    y0: int
    crop_w: int
    crop_h: int
    out_w: int
    out_h: int

    @property
    def scale(self):
        return self.out_w / self.crop_w

    @classmethod
    def center(cls, in_w, in_h, out_w, out_h):
        aspect = out_w / out_h
        crop_w = min(in_w, int(round(in_h * aspect)))
        crop_h = min(in_h, int(round(crop_w / aspect)))
        return cls((in_w - crop_w) // 2, (in_h - crop_h) // 2, crop_w, crop_h, out_w, out_h)

    def apply_intrinsics(self, intr: Intrinsics) -> Intrinsics:
        s = self.scale
        return Intrinsics(
            intr.fx * s,
            intr.fy * s,
            (intr.cx - self.x0 + 0.5) * s - 0.5,
            (intr.cy - self.y0 + 0.5) * s - 0.5,
            self.out_w,
            self.out_h,
        )

    def to_out(self, u, v):
        s = self.scale
        return (np.asarray(u) - self.x0 + 0.5) * s - 0.5, (np.asarray(v) - self.y0 + 0.5) * s - 0.5

    def to_in(self, u, v):
        s = self.scale
        return (np.asarray(u) + 0.5) / s - 0.5 + self.x0, (np.asarray(v) + 0.5) / s - 0.5 + self.y0

    def apply_rgb(self, rgb):
        import cv2

        crop = rgb[self.y0 : self.y0 + self.crop_h, self.x0 : self.x0 + self.crop_w]
        return cv2.resize(crop, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)

    def apply_depth(self, depth):
        # NEAREST: never blend valid depth with the 0 (invalid) value.
        import cv2

        crop = depth[self.y0 : self.y0 + self.crop_h, self.x0 : self.x0 + self.crop_w]
        return cv2.resize(crop, (self.out_w, self.out_h), interpolation=cv2.INTER_NEAREST)


def rescale_pixel(row, col, src_wh, dst_wh):
    """[row, col] in an image of size src_wh=(w, h) -> (u, v) in an image of size dst_wh (squash resize)."""
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    return (col + 0.5) * sx - 0.5, (row + 0.5) * sy - 0.5


def sample_depth(depth_m, u, v, radius=4, min_depth=0.1, max_depth=50.0, min_valid_frac=0.3):
    """Median of valid depth in a (2r+1)^2 window around (u, v). Returns (depth or None, valid_frac)."""
    h, w = depth_m.shape[:2]
    ui, vi = int(round(float(u))), int(round(float(v)))
    if not (0 <= ui < w and 0 <= vi < h):
        return None, 0.0
    win = depth_m[max(0, vi - radius) : vi + radius + 1, max(0, ui - radius) : ui + radius + 1]
    valid = win[np.isfinite(win) & (win > min_depth) & (win < max_depth)]
    frac = valid.size / max(1, win.size)
    if frac < min_valid_frac:
        return None, frac
    return float(np.median(valid)), frac


def backproject(u, v, depth, intr: Intrinsics):
    """Pixel + z-depth -> 3D point in the optical frame."""
    return np.array([(u - intr.cx) / intr.fx * depth, (v - intr.cy) / intr.fy * depth, depth], np.float64)


def project(p_cam, intr: Intrinsics):
    """3D points [N,3] in the optical frame -> (uv [N,2], in_front [N])."""
    p = np.asarray(p_cam, np.float64).reshape(-1, 3)
    z = p[:, 2]
    in_front = z > 1e-6
    zs = np.where(in_front, z, 1.0)
    uv = np.column_stack([intr.fx * p[:, 0] / zs + intr.cx, intr.fy * p[:, 1] / zs + intr.cy])
    return uv, in_front


def transform_points(T, p):
    p = np.asarray(p, np.float64).reshape(-1, 3)
    return p @ T[:3, :3].T + T[:3, 3]


def ray_ground_intersection(u, v, intr: Intrinsics, T_base_cam, ground_z, max_range=30.0):
    """Cast the pixel ray, intersect plane z = ground_z expressed in the base frame.

    Returns the 3D point in the base frame, or None if the ray does not hit the plane in
    front of the camera within max_range.
    """
    d_cam = np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
    R, o = T_base_cam[:3, :3], T_base_cam[:3, 3]
    d = R @ d_cam
    if d[2] >= -1e-6:
        return None
    t = (ground_z - o[2]) / d[2]
    p = o + t * d
    if t <= 0 or np.linalg.norm(p[:2] - o[:2]) > max_range:
        return None
    return p


def pixel_to_base(u, v, depth_m, intr: Intrinsics, T_base_cam, ground_z, depth_kwargs=None):
    """Pixel goal -> base-frame 3D point. Depth median first, ground-plane ray as fallback.

    Returns (point or None, method) with method in {"depth", "ground", "none"}.
    """
    d, _ = sample_depth(depth_m, u, v, **(depth_kwargs or {}))
    if d is not None:
        return transform_points(T_base_cam, backproject(u, v, d, intr))[0], "depth"
    p = ray_ground_intersection(u, v, intr, T_base_cam, ground_z)
    if p is not None:
        return p, "ground"
    return None, "none"


def base_traj_to_pixels(traj_xy, intr: Intrinsics, T_base_cam, ground_z):
    """Robot-frame ground waypoints [N,2] (x fwd, y left) -> pixels [N,2] + in_front mask."""
    xy = np.asarray(traj_xy, np.float64).reshape(-1, 2)
    p_base = np.column_stack([xy, np.full(len(xy), ground_z)])
    p_cam = transform_points(np.linalg.inv(T_base_cam), p_base)
    return project(p_cam, intr)


def se2(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = [x, y]
    return T


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_T(t_xyz, q_xyzw):
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_R(q_xyzw)
    T[:3, 3] = t_xyz
    return T


def yaw_of(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


def estimate_ground_z(depth_m, intr: Intrinsics, T_base_cam, v_frac=(0.8, 1.0), u_frac=(0.3, 0.7), step=8,
                      max_spread=0.05, min_points=30):
    """Floor height in the base frame from the bottom-centre of the depth image (assumed floor).

    Returns the median z, or None when too few valid pixels or the region is not flat
    (p90 - p10 > max_spread, e.g. an obstacle right in front).
    """
    h, w = depth_m.shape[:2]
    vs = np.arange(int(h * v_frac[0]), int(h * v_frac[1]), step)
    us = np.arange(int(w * u_frac[0]), int(w * u_frac[1]), step)
    uu, vv = np.meshgrid(us, vs)
    d = depth_m[vv, uu]
    ok = np.isfinite(d) & (d > 0.1)
    if ok.sum() < min_points:
        return None
    pts = np.column_stack([(uu[ok] - intr.cx) / intr.fx * d[ok], (vv[ok] - intr.cy) / intr.fy * d[ok], d[ok]])
    z = transform_points(T_base_cam, pts)[:, 2]
    if np.percentile(z, 90) - np.percentile(z, 10) > max_spread:
        return None
    return float(np.median(z))
