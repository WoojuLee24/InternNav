"""V2 / V6 unit tests: crop-resize intrinsics, depth sampling, back-projection, ground ray."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projection import (  # noqa: E402
    CropResize,
    Intrinsics,
    backproject,
    base_traj_to_pixels,
    make_T,
    pixel_to_base,
    project,
    ray_ground_intersection,
    rescale_pixel,
    sample_depth,
    transform_points,
)

# sim D455: fx=fy=1.93*1280/3.896, centered principal point
SIM = Intrinsics(634.086, 634.086, 640.0, 360.0, 1280, 720)


def _sim_T_base_cam():
    # base_link -> os_sensor -> camera_link (sensor_configuration.yaml) -> optical
    T_base_os = make_T([0.3, 0.0, 0.05], [0, 0, 0, 1])
    T_os_link = make_T([0.065151, 0.002019, -0.015822], [-0.000391, 0.128428, 0.001329, 0.991718])
    T_link_opt = make_T([0, 0, 0], [-0.5, 0.5, -0.5, 0.5])
    return T_base_os @ T_os_link @ T_link_opt


def test_crop_geometry():
    cr = CropResize.center(1280, 720, 640, 480)
    assert (cr.x0, cr.y0, cr.crop_w, cr.crop_h) == (160, 0, 960, 720)
    assert cr.scale == pytest.approx(2 / 3)


def test_crop_intrinsics_consistent_with_pixel_mapping():
    """Projecting with K' must equal projecting with K then mapping pixels (V2: <0.01 px)."""
    cr = CropResize.center(1280, 720, 640, 480)
    k2 = cr.apply_intrinsics(SIM)
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-2, 2, 200), rng.uniform(-1, 1, 200), rng.uniform(0.5, 10, 200)])
    uv, _ = project(pts, SIM)
    u2, v2 = cr.to_out(uv[:, 0], uv[:, 1])
    uv_k2, _ = project(pts, k2)
    assert np.abs(uv_k2 - np.column_stack([u2, v2])).max() < 1e-2
    # round trip
    ub, vb = cr.to_in(u2, v2)
    assert np.allclose(np.column_stack([ub, vb]), uv)


def test_rescale_pixel_round_trip():
    u, v = rescale_pixel(191.5, 191.5, (384, 384), (640, 480))
    assert u == pytest.approx(319.5) and v == pytest.approx(239.5)


def test_sample_depth_median_and_invalid():
    d = np.full((480, 640), 3.0, np.float32)
    d[200:210, 300:310] = 0.0  # hole
    val, frac = sample_depth(d, 100, 100, radius=3)
    assert val == pytest.approx(3.0) and frac == 1.0
    val, frac = sample_depth(d, 305, 205, radius=3)
    assert val is None and frac < 0.3
    assert sample_depth(d, -5, 10)[0] is None


def test_backproject_project_round_trip():
    p = backproject(400.0, 100.0, 4.2, SIM)
    uv, front = project(p, SIM)
    assert front[0] and np.allclose(uv[0], [400.0, 100.0])


def test_ground_ray_matches_depth_on_plane():
    """Synthetic flat floor: depth backprojection and ground-ray must agree (V6 round trip <1 cm)."""
    T = _sim_T_base_cam()
    ground_z = -0.55
    for u, v in [(640, 600), (300, 500), (1000, 450)]:
        p_ray = ray_ground_intersection(u, v, SIM, T, ground_z)
        assert p_ray is not None and p_ray[2] == pytest.approx(ground_z)
        z_depth = transform_points(np.linalg.inv(T), p_ray)[0, 2]
        d = np.full((720, 1280), z_depth, np.float32)
        p_depth, method = pixel_to_base(u, v, d, SIM, T, ground_z, {"radius": 0})
        assert method == "depth"
        assert np.linalg.norm(p_depth - p_ray) < 0.01


def test_ground_ray_above_horizon_is_none():
    T = _sim_T_base_cam()
    assert ray_ground_intersection(640, 20, SIM, T, -0.55) is None


def test_pixel_to_base_falls_back_to_ground():
    T = _sim_T_base_cam()
    d = np.zeros((720, 1280), np.float32)
    p, method = pixel_to_base(640, 600, d, SIM, T, -0.55)
    assert method == "ground" and p[0] > 0.3


def test_traj_projection_inverse_of_ground_ray():
    T = _sim_T_base_cam()
    xy = np.array([[1.0, 0.0], [2.0, 0.5], [3.0, -0.8]])
    uv, front = base_traj_to_pixels(xy, SIM, T, -0.55)
    assert front.all()
    for (u, v), (x, y) in zip(uv, xy):
        p = ray_ground_intersection(u, v, SIM, T, -0.55)
        assert math.hypot(p[0] - x, p[1] - y) < 1e-6


def test_camera_pitch_down():
    """Sanity: sim camera pitched ~14.8 deg down -> image center ray hits floor ahead."""
    T = _sim_T_base_cam()
    p = ray_ground_intersection(640, 360, SIM, T, -0.55)
    assert p is not None
    expected = 0.365 + (0.55 + 0.034) / math.tan(math.radians(14.76))
    assert p[0] == pytest.approx(expected, rel=0.03)


def test_estimate_ground_z_flat_and_obstacle():
    from projection import estimate_ground_z

    T = _sim_T_base_cam()
    true_z = -0.476
    d = np.zeros((720, 1280), np.float32)
    for v in range(360, 720):  # synthesize a flat floor from the ground ray
        for u in range(0, 1280, 4):
            p = ray_ground_intersection(u, v, SIM, T, true_z)
            if p is not None:
                d[v, u:u + 4] = transform_points(np.linalg.inv(T), p)[0, 2]
    assert estimate_ground_z(d, SIM, T) == pytest.approx(true_z, abs=0.01)
    d2 = d.copy()
    d2[600:720, 400:800] = 0.8  # box right in front -> not flat -> None
    assert estimate_ground_z(d2, SIM, T) is None
