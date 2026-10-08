"""05 RTX LiDAR NPZ를 생성 코드와 독립적으로 검증한다."""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


DEFAULT_NPZ = Path(
    'scripts/dataset_converters/gs_vlnpe/apply_real/obs/'
    'noeun_station_mid_random_isaac_d455_nominal/lidar_rtx/smoke_test/'
    'episode_000000_frame_0000.npz'
)
DEFAULT_USDZ = Path('data/noeun_station/noeun_station_collision.usdz')


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz', type=Path, default=DEFAULT_NPZ)
    parser.add_argument('--scene_usdz', type=Path, default=DEFAULT_USDZ)
    parser.add_argument('--sample_stride', type=int, default=20)
    parser.add_argument('--median_surface_tol_m', type=float, default=0.002)
    parser.add_argument('--p95_surface_tol_m', type=float, default=0.01)
    parser.add_argument('--calibration_median_tol_m', type=float, default=0.05)
    parser.add_argument('--calibration_p95_tol_m', type=float, default=0.25)
    parser.add_argument('--report_json', type=Path, default=None)
    return parser


def load_usdz_mesh(path: Path) -> trimesh.Trimesh:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        if 'mesh.ply' not in archive.namelist():
            raise ValueError(f'{path} 안에 mesh.ply가 없습니다.')
        with tempfile.TemporaryDirectory(prefix='noeun_lidar_validate_') as temp_dir:
            mesh_path = Path(archive.extract('mesh.ply', temp_dir))
            return trimesh.load(mesh_path, process=False, force='mesh')


def validate(args) -> dict:
    if args.sample_stride < 1:
        raise ValueError('--sample_stride는 1 이상이어야 합니다.')
    with np.load(args.npz) as data:
        required = {
            'points_world', 'ranges_m', 'valid_mask', 'sensor_position_world',
            'sensor_orientation_world_wxyz', 'camera_pose_c2w', 'episode_index', 'frame_index',
            'points_lidar_m', 'points_robot_m', 'T_world_robot', 'T_robot_lidar',
            'T_world_lidar', 'emitter_id', 'azimuth',
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f'필수 필드 누락: {missing}')
        points = np.asarray(data['points_world'])
        ranges = np.asarray(data['ranges_m'])
        valid = np.asarray(data['valid_mask'])
        sensor_position = np.asarray(data['sensor_position_world'])
        points_lidar = np.asarray(data['points_lidar_m'])
        points_robot = np.asarray(data['points_robot_m'])
        t_world_robot = np.asarray(data['T_world_robot'])
        t_robot_lidar = np.asarray(data['T_robot_lidar'])
        t_world_lidar = np.asarray(data['T_world_lidar'])
        camera_c2w = np.asarray(data['camera_pose_c2w'])
        emitter_id = np.asarray(data['emitter_id'])
        azimuth = np.asarray(data['azimuth'])
        episode = int(data['episode_index'])
        frame = int(data['frame_index'])

    shape_ok = points.ndim == 2 and points.shape[1:] == (3,)
    aligned = shape_ok and ranges.shape == (len(points),) and valid.shape == (len(points),)
    dtype_ok = points.dtype == np.float32 and ranges.dtype == np.float32 and valid.dtype == np.bool_
    finite_fraction = float(np.isfinite(points).all(axis=1).mean()) if shape_ok and len(points) else 0.0
    recomputed = np.linalg.norm(points - sensor_position[None, :], axis=1) if shape_ok else np.array([])
    range_consistent = bool(aligned and np.allclose(ranges, recomputed, atol=1e-5, rtol=1e-5))
    transform_shape_ok = all(x.shape == (4, 4) for x in (t_world_robot, t_robot_lidar, t_world_lidar))
    chain_error = float(np.max(np.abs(t_world_robot @ t_robot_lidar - t_world_lidar)))
    lidar_h = np.concatenate([points_lidar, np.ones((len(points_lidar), 1))], axis=1)
    robot_h = np.concatenate([points_robot, np.ones((len(points_robot), 1))], axis=1)
    lidar_to_robot_error = float(np.max(np.abs((lidar_h @ t_robot_lidar.T)[:, :3] - points_robot)))
    robot_to_world_error = float(np.max(np.abs((robot_h @ t_world_robot.T)[:, :3] - points)))
    channel_ids = np.unique(emitter_id).astype(int)
    channel_ok = np.array_equal(channel_ids, np.arange(32))
    azimuth_span = float(azimuth.max() - azimuth.min()) if azimuth.size else 0.0

    mesh = load_usdz_mesh(args.scene_usdz)
    sampled = points[valid][::args.sample_stride]
    surface_m = np.abs(trimesh.proximity.ProximityQuery(mesh).signed_distance(sampled))
    surface_median = float(np.median(surface_m))
    surface_p95 = float(np.percentile(surface_m, 95))

    obs_root = args.npz.parents[2]
    ep_dir = obs_root / f'episode_{episode:06d}'
    intrinsic = np.load(ep_dir / 'intrinsic.npy').astype(np.float64)
    depth_raw = np.asarray(Image.open(ep_dir / 'depth' / f'frame_{frame:04d}.png'))
    depth_m = depth_raw.astype(np.float64) * 0.001
    points_camera = (points.astype(np.float64) - camera_c2w[:3, 3]) @ camera_c2w[:3, :3]
    z = points_camera[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = intrinsic[0, 0] * points_camera[:, 0] / z + intrinsic[0, 2]
        v = intrinsic[1, 1] * points_camera[:, 1] / z + intrinsic[1, 2]
    ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    h, w = depth_m.shape
    projected = valid & (z > 0) & (z <= 10.0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    rgbd_depth = depth_m[vi[projected], ui[projected]]
    lidar_depth = z[projected]
    overlap = rgbd_depth > 0
    calibration_abs = np.abs(lidar_depth[overlap] - rgbd_depth[overlap])
    calibration_median = float(np.median(calibration_abs)) if calibration_abs.size else float('inf')
    calibration_p95 = float(np.percentile(calibration_abs, 95)) if calibration_abs.size else float('inf')

    checks = {
        'shape_alignment': bool(aligned),
        'dtype': bool(dtype_ok),
        'all_points_finite': finite_fraction == 1.0,
        'range_recomputation': range_consistent,
        'transform_shapes': transform_shape_ok,
        'transform_chain': chain_error <= 1e-5,
        'lidar_to_robot_transform': lidar_to_robot_error <= 1e-5,
        'robot_to_world_transform': robot_to_world_error <= 1e-5,
        'os1_32_channel_coverage': bool(channel_ok),
        'azimuth_360_coverage': azimuth_span >= 359.0,
        'nonempty_valid_points': bool(len(sampled) > 0),
        'mesh_surface_median': surface_median <= args.median_surface_tol_m,
        'mesh_surface_p95': surface_p95 <= args.p95_surface_tol_m,
        'rgbd_lidar_overlap': bool(calibration_abs.size >= 100),
        'rgbd_lidar_median': calibration_median <= args.calibration_median_tol_m,
        'rgbd_lidar_p95': calibration_p95 <= args.calibration_p95_tol_m,
    }
    return {
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'input_npz': str(args.npz),
        'scene_usdz': str(args.scene_usdz),
        'checks': checks,
        'point_count': int(len(points)) if shape_ok else 0,
        'sample_count': int(len(sampled)),
        'finite_fraction': finite_fraction,
        'surface_distance_m': {
            'median': surface_median,
            'p95': surface_p95,
            'max': float(surface_m.max()),
        },
        'coordinate_transform_max_abs_error_m': {
            'chain': chain_error,
            'lidar_to_robot': lidar_to_robot_error,
            'robot_to_world': robot_to_world_error,
        },
        'os1_scan': {
            'emitter_ids': channel_ids.tolist(),
            'azimuth_span_deg': azimuth_span,
        },
        'rgbd_lidar_calibration': {
            'projected_count': int(projected.sum()),
            'valid_depth_overlap_count': int(calibration_abs.size),
            'median_abs_error_m': calibration_median,
            'p95_abs_error_m': calibration_p95,
        },
        'threshold_m': {
            'median': args.median_surface_tol_m,
            'p95': args.p95_surface_tol_m,
        },
    }


def main() -> int:
    args = build_argparser().parse_args()
    result = validate(args)
    report_path = args.report_json or args.npz.with_suffix('.validation.json')
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
