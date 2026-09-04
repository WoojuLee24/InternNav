"""Synthetic observation camera profiles for Stage 04.

Dataset GT cameras remain owned by ``dataset_utils``.  Profiles here are used
only when a scene has no GT camera, so changing one cannot silently alter the
legacy InternNav-N1 path.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraProfile:
    name: str
    width: int
    height: int
    k: np.ndarray
    depth_scale_m: float
    depth_min_m: float
    depth_max_m: float
    render_near_m: float
    render_far_m: float

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'width': self.width,
            'height': self.height,
            'intrinsic': self.k.tolist(),
            'depth_scale_m': self.depth_scale_m,
            'depth_min_m': self.depth_min_m,
            'depth_max_m': self.depth_max_m,
            'render_near_m': self.render_near_m,
            'render_far_m': self.render_far_m,
        }


def square_pixel_k_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Return a zero-distortion, square-pixel pinhole intrinsic matrix."""
    focal_px = (width * 0.5) / np.tan(np.deg2rad(hfov_deg) * 0.5)
    return np.array([
        [focal_px, 0.0, width * 0.5],
        [0.0, focal_px, height * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# Nominal synthetic profile, not a calibration dump from a particular device.
# A real deployment must replace K with librealsense get_intrinsics() output and
# verify depth_scale_m with get_depth_scale().  0.001 m/raw is the D400 default
# Z16 unit; 10 m is this synthetic dataset's storage cutoff, not a claim about
# D455 measurement accuracy at 10 m.
D455_NOMINAL = CameraProfile(
    name='d455_nominal',
    width=480,
    height=270,
    k=square_pixel_k_from_hfov(480, 270, 90.0),
    depth_scale_m=0.001,
    depth_min_m=0.1,
    depth_max_m=10.0,
    render_near_m=0.05,
    render_far_m=12.0,
)

# 30 m 저장 범위 변형. K/해상도는 d455_nominal과 동일하게 두어 "범위만 바꿨을 때의 차이"를
# 분리해서 볼 수 있게 한다. render_far는 저장 상한보다 커야 한다 — generate_episode의 valid
# 조건이 depth < render_far*0.99 이므로 30 m를 담으려면 far가 최소 30/0.99 = 30.3 m다.
D455_30M = CameraProfile(
    name='d455_30m',
    width=480,
    height=270,
    k=square_pixel_k_from_hfov(480, 270, 90.0),
    depth_scale_m=0.001,
    depth_min_m=0.1,
    depth_max_m=30.0,
    render_near_m=0.05,
    render_far_m=35.0,
)

PROFILES = {D455_NOMINAL.name: D455_NOMINAL, D455_30M.name: D455_30M}


def names() -> tuple:
    return tuple(PROFILES)


def get(name: str) -> CameraProfile:
    if name not in PROFILES:
        raise KeyError(f'unknown camera profile: {name!r}')
    return PROFILES[name]
