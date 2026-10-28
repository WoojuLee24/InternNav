"""Camera model for projecting S1 waypoints into the image — conventions copied verbatim
from the original repo (do NOT invent new ones):

  * intrinsics from HFOV      internnav/agent/internvla_n1_agent.py:get_intrinsic_matrix
        fx = (W/2)/tan(hfov/2),  fy = fx,  cx = (W-1)/2,  cy = (H-1)/2
  * camera->world rotation +  internnav/model/utils/depth_rgb_to_bev_torch.py
    ground frame               (_build_R_c2w / unproject_depth):
        world X = forward, Y = left, Z = up;  cam_pitch_deg pitches the camera DOWN;
        xyz_w = R_c2w @ xyz_c + [cam_x_offset, 0, cam_height]
  * S2 pixel convention       internvla_n1_policy(_llamacpp).py:_parse_pixel
        output_pixel = [row, col]  (i.e. [y, x])

`project_ground` is the exact algebraic INVERSE of `unproject_depth` restricted to
ground-plane points (z_w = 0), which is what S1 waypoints are (robot-frame x fwd, y left).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class CamModel:
    size: int             # published square frame size (camera --size == S2 --resize)
    hfov_deg: float       # HFOV of the published (center-cropped, square) frame
    height_m: float       # camera height above ground; repo default 1.25 m
    pitch_deg: float      # downward pitch; repo convention +down (0 = level forward cam)
    x_offset_m: float = 0.0   # camera mounted this far ahead of the robot origin

    @property
    def K(self):
        """get_intrinsic_matrix port (square frame: W = H = size)."""
        fx = (self.size / 2.0) / np.tan(np.deg2rad(self.hfov_deg / 2.0))
        return fx, fx, (self.size - 1.0) / 2.0, (self.size - 1.0) / 2.0

    @property
    def R_c2w(self):
        """Verbatim _build_R_c2w(cam_pitch_deg): cam (x right, y down, z fwd) -> world."""
        p = np.deg2rad(self.pitch_deg)
        s, c = np.sin(p), np.cos(p)
        return np.array([[0.0, -s, c],
                         [-1.0, 0.0, 0.0],
                         [0.0, -c, -s]], np.float64)

    @property
    def t_w(self):
        """Camera position in the world/robot frame (unproject_depth's translation)."""
        return np.array([self.x_offset_m, 0.0, self.height_m], np.float64)


def project_ground(xy, cam: CamModel):
    """Robot-frame ground points [N,2] (x fwd, y left, z=0)  ->  pixel (u,v) [N,2].

    Inverse of unproject_depth:  xyz_c = R_c2w^T @ (xyz_w - t_w);  u = fx*x_c/z_c + cx ...
    Returns (uv float32 [N,2], in_front bool [N], in_frame bool [N]).
    """
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    world = np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy))])
    cam_pts = (world - cam.t_w) @ cam.R_c2w          # row-vector form of R^T @ (w - t)
    fx, fy, cx, cy = cam.K
    z = cam_pts[:, 2]
    in_front = z > 0.05                              # behind/at the pinhole -> not drawable
    zs = np.where(in_front, z, 1.0)
    u = fx * cam_pts[:, 0] / zs + cx
    v = fy * cam_pts[:, 1] / zs + cy
    uv = np.column_stack([u, v]).astype(np.float32)
    in_frame = in_front & (u >= 0) & (u < cam.size) & (v >= 0) & (v < cam.size)
    return uv, in_front, in_frame


def pixel_to_ground(row, col, cam: CamModel):
    """S2 pixel goal [row, col] -> robot-frame ground point (x fwd, y left), or None.

    Casts the pixel ray through R_c2w and intersects the z_w = 0 ground plane — the same
    geometry unproject_depth applies with a depth map, but depth-free (ground assumption).
    """
    fx, fy, cx, cy = cam.K
    ray_c = np.array([(col - cx) / fx, (row - cy) / fy, 1.0])
    ray_w = cam.R_c2w @ ray_c
    if ray_w[2] >= -1e-6:                            # ray does not descend -> above horizon
        return None
    scale = cam.height_m / (-ray_w[2])
    p = cam.t_w + scale * ray_w
    return float(p[0]), float(p[1])
