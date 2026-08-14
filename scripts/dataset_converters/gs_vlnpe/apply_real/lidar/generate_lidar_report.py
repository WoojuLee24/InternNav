"""노은역 RTX LiDAR smoke 결과를 사람이 읽는 self-contained report.html로 만든다."""

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from viz_utils import blink_widget_html, image_to_data_uri, save_gallery

DEFAULT_NPZ = Path(
    'scripts/dataset_converters/gs_vlnpe/apply_real/obs/'
    'noeun_station_mid_random_isaac_d455_nominal/lidar_rtx/smoke_test/'
    'episode_000000_frame_0000.npz'
)
DEFAULT_USDZ = Path('data/noeun_station/noeun_station_collision.usdz')


def emit(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()
    return path


def load_mesh(path: Path):
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory(prefix='lidar_report_') as tmp:
        mesh_path = Path(archive.extract('mesh.ply', tmp))
        return trimesh.load(mesh_path, process=False, force='mesh')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz', type=Path, default=DEFAULT_NPZ)
    parser.add_argument('--scene_usdz', type=Path, default=DEFAULT_USDZ)
    parser.add_argument('--sample_stride', type=int, default=20)
    args = parser.parse_args()

    validation_path = args.npz.with_suffix('.validation.json')
    result = json.loads(validation_path.read_text(encoding='utf-8'))
    with np.load(args.npz) as data:
        points = np.asarray(data['points_world'])
        ranges = np.asarray(data['ranges_m'])
        valid = np.asarray(data['valid_mask'])
        sensor = np.asarray(data['sensor_position_world'])
        points_robot = np.asarray(data['points_robot_m'])
        camera_c2w = np.asarray(data['camera_pose_c2w'])
        emitter_id = np.asarray(data['emitter_id'])
        intensity = np.asarray(data['intensity'])
        episode = int(data['episode_index'])
        frame = int(data['frame_index'])

    points, ranges = points[valid], ranges[valid]
    draw_stride = max(1, len(points) // 30000)
    draw_points, draw_ranges = points[::draw_stride], ranges[::draw_stride]
    assets = args.npz.parent / 'report_assets'

    plt.figure(figsize=(8, 7))
    plt.scatter(draw_points[:, 0], draw_points[:, 1], c=draw_ranges, s=.35, cmap='turbo')
    plt.scatter(sensor[0], sensor[1], c='white', edgecolors='black', s=100, marker='*', label='LiDAR')
    plt.colorbar(label='range [m]'); plt.xlabel('world X [m]'); plt.ylabel('world Y [m]')
    plt.axis('equal'); plt.legend(); plt.title('RTX LiDAR point cloud — top view')
    top = emit(assets / 'lidar_top_view.png')

    robot_draw = points_robot[valid][::draw_stride]
    plt.figure(figsize=(8, 7))
    plt.scatter(robot_draw[:, 0], robot_draw[:, 1], c=np.linalg.norm(robot_draw, axis=1),
                s=.35, cmap='turbo')
    plt.scatter(0, 0, c='white', edgecolors='black', s=100, marker='*', label='robot/base_link')
    plt.colorbar(label='distance from robot origin [m]'); plt.xlabel('robot X forward [m]')
    plt.ylabel('robot Y left [m]'); plt.axis('equal'); plt.legend()
    plt.title('OS1-32 point cloud in robot/base_link frame')
    robot_top = emit(assets / 'lidar_robot_frame_top.png')

    plt.figure(figsize=(9, 4.8))
    plt.scatter(draw_points[:, 0], draw_points[:, 2], c=draw_ranges, s=.35, cmap='turbo')
    plt.scatter(sensor[0], sensor[2], c='white', edgecolors='black', s=100, marker='*')
    plt.colorbar(label='range [m]'); plt.xlabel('world X [m]'); plt.ylabel('world Z [m]')
    plt.title('RTX LiDAR point cloud — side view')
    side = emit(assets / 'lidar_side_view.png')

    plt.figure(figsize=(8, 4.5))
    plt.hist(ranges, bins=100, color='#4fd1c5')
    plt.axvline(np.median(ranges), color='#e2725b', linestyle='--', label=f'median {np.median(ranges):.3f} m')
    plt.xlabel('range [m]'); plt.ylabel('returns'); plt.title('LiDAR range distribution'); plt.legend()
    range_hist = emit(assets / 'range_histogram.png')

    sampled = points[::args.sample_stride]
    surface_mm = np.abs(trimesh.proximity.ProximityQuery(load_mesh(args.scene_usdz)).signed_distance(sampled)) * 1000
    plt.figure(figsize=(8, 4.5))
    plt.hist(surface_mm, bins=100, color='#7f9cf5')
    plt.axvline(np.median(surface_mm), color='#57c785', linestyle='--', label=f'median {np.median(surface_mm):.3f} mm')
    plt.axvline(np.percentile(surface_mm, 95), color='#e0b34d', linestyle='--', label=f'p95 {np.percentile(surface_mm, 95):.3f} mm')
    plt.xlabel('nearest mesh surface distance [mm]'); plt.ylabel('sampled returns')
    plt.title('LiDAR ↔ source mesh alignment error'); plt.legend()
    error_hist = emit(assets / 'mesh_surface_error_histogram.png')

    obs_root = args.npz.parents[2]
    ep_dir = obs_root / f'episode_{episode:06d}'
    rgb = ep_dir / 'rgb' / f'frame_{frame:04d}.jpg'
    depth = ep_dir / 'depth' / f'frame_{frame:04d}.png'
    context = []
    if rgb.is_file():
        context.append(('04 Isaac RGB', rgb))
    if depth.is_file():
        raw = np.asarray(Image.open(depth)); meters = raw.astype(np.float32) * .001
        meters[(raw == 0) | (raw >= 10000)] = np.nan
        plt.figure(figsize=(8, 4.5)); plt.imshow(meters, cmap='turbo', vmin=.1, vmax=10)
        plt.colorbar(label='depth [m]'); plt.axis('off'); plt.title('04 Isaac D455 depth')
        context.append(('04 Isaac D455 depth', emit(assets / 'camera_depth.png')))

    overlay = residual_hist = channel_plot = intensity_hist = None
    if rgb.is_file() and depth.is_file():
        rgb_image = np.asarray(Image.open(rgb).convert('RGB'))
        intrinsic = np.load(ep_dir / 'intrinsic.npy').astype(np.float64)
        raw_depth = np.asarray(Image.open(depth))
        depth_m = raw_depth.astype(np.float64) * .001
        points_camera = (points.astype(np.float64) - camera_c2w[:3, 3]) @ camera_c2w[:3, :3]
        z = points_camera[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = intrinsic[0, 0] * points_camera[:, 0] / z + intrinsic[0, 2]
            v = intrinsic[1, 1] * points_camera[:, 1] / z + intrinsic[1, 2]
        h, w = raw_depth.shape
        projected = valid & (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ui, vi = np.rint(u[projected]).astype(int), np.rint(v[projected]).astype(int)
        ui = np.clip(ui, 0, w - 1); vi = np.clip(vi, 0, h - 1)
        projected_z = z[projected]
        projected_range = ranges[projected]
        order = np.argsort(projected_z)[::-1]
        color_max = max(10.0, float(np.percentile(projected_range, 99)))
        plt.figure(figsize=(10, 5.8)); plt.imshow(rgb_image)
        scatter = plt.scatter(ui[order], vi[order], c=projected_range[order], s=2.2, cmap='turbo',
                              vmin=.3, vmax=color_max, alpha=.78)
        plt.colorbar(scatter, label=f'OS1 range [m] (display p99={color_max:.1f} m)')
        plt.axis('off'); plt.title('D455 RGB + all in-FOV OS1-32 returns (not clipped at 10 m)')
        overlay = emit(assets / 'rgb_lidar_projection_overlay.png')

        camera_depth = depth_m[vi, ui]
        overlap = (camera_depth > 0) & (projected_z <= 10.0)
        residual = np.abs(projected_z[overlap] - camera_depth[overlap])
        plt.figure(figsize=(8, 4.5)); plt.hist(residual, bins=120, range=(0, min(1, np.percentile(residual, 99))),
                                             color='#f6ad55')
        plt.axvline(np.median(residual), color='#57c785', linestyle='--',
                    label=f'median {np.median(residual)*100:.2f} cm')
        plt.axvline(np.percentile(residual, 95), color='#e2725b', linestyle='--',
                    label=f'p95 {np.percentile(residual, 95)*100:.2f} cm')
        plt.xlabel('|LiDAR camera-Z - D455 depth| [m]'); plt.ylabel('projected returns')
        plt.title('RGB-D ↔ LiDAR calibration residual'); plt.legend()
        residual_hist = emit(assets / 'rgbd_lidar_residual_histogram.png')

    plt.figure(figsize=(9, 4.5)); plt.hist(intensity[valid], bins=100, color='#9f7aea')
    plt.xlabel('intensity'); plt.ylabel('returns'); plt.title('OS1-32 intensity distribution')
    intensity_hist = emit(assets / 'intensity_histogram.png')
    channel_ids, channel_counts = np.unique(emitter_id, return_counts=True)
    sane = (channel_ids >= 0) & (channel_ids < 32)
    plt.figure(figsize=(9, 4.5)); plt.bar(channel_ids[sane], channel_counts[sane], color='#4fd1c5')
    plt.xticks(np.arange(32)); plt.xlabel('emitter/channel ID'); plt.ylabel('returns')
    plt.title('OS1-32 channel coverage')
    channel_plot = emit(assets / 'channel_coverage.png')

    surface = result['surface_distance_m']; badge = 'good' if result['status'] == 'PASS' else 'bad'
    calibration = result.get('rgbd_lidar_calibration', {})
    rows = ''.join(
        f'<tr><td>{name}</td><td><span class="pill {"good" if ok else "bad"}">{"PASS" if ok else "FAIL"}</span></td></tr>'
        for name, ok in result['checks'].items()
    )
    summary = f'''
<div class="stat-row">
 <div class="stat"><div class="label">판정</div><div class="value"><span class="pill {badge}">{result['status']}</span></div></div>
 <div class="stat"><div class="label">RTX returns</div><div class="value">{result['point_count']:,}</div></div>
 <div class="stat"><div class="label">finite</div><div class="value">{result['finite_fraction']*100:.1f}%</div></div>
 <div class="stat"><div class="label">mesh median</div><div class="value">{surface['median']*1000:.3f} mm</div></div>
 <div class="stat"><div class="label">mesh p95</div><div class="value">{surface['p95']*1000:.3f} mm</div></div>
 <div class="stat"><div class="label">RGB-D/LiDAR median</div><div class="value">{calibration.get('median_abs_error_m', float('nan'))*100:.2f} cm</div></div>
 <div class="stat"><div class="label">RGB-D/LiDAR p95</div><div class="value">{calibration.get('p95_abs_error_m', float('nan'))*100:.2f} cm</div></div>
</div>
<p>노은역 중간층 episode {episode}, frame {frame}. 04 RGB-D와 같은 pose의 RTX LiDAR 결과이며 원본 USDZ mesh 표면 정합까지 검사한다.</p>
<table><tr><th>검증 gate</th><th>결과</th></tr>{rows}</table>'''
    body = '<div class="grid">'
    if context:
        body += '<div class="card"><div class="caption">같은 frame의 04 RGB-D</div>' + blink_widget_html('context', context) + '</div>'
    body += '<div class="card"><div class="caption">LiDAR 점군 상면/측면</div>' + blink_widget_html('views', [('top view', top), ('side view', side)]) + '</div>'
    body += '<div class="card"><div class="caption">robot/base_link 좌표계</div>' + f'<img src="{image_to_data_uri(robot_top)}"></div>'
    if overlay is not None:
        body += '<div class="card"><div class="caption">RGB 위 OS1-32 projection — pose/extrinsic calibration 확인</div>' + f'<img src="{image_to_data_uri(overlay)}"></div>'
    if residual_hist is not None:
        body += '<div class="card"><div class="caption">RGB-D ↔ LiDAR depth 잔차</div>' + f'<img src="{image_to_data_uri(residual_hist)}"></div>'
    body += f'<div class="card"><div class="caption">거리 분포</div><img src="{image_to_data_uri(range_hist)}"></div>'
    body += f'<div class="card"><div class="caption">intensity</div><img src="{image_to_data_uri(intensity_hist)}"></div>'
    body += f'<div class="card"><div class="caption">OS1 32채널 coverage</div><img src="{image_to_data_uri(channel_plot)}"></div>'
    body += f'<div class="card"><div class="caption">mesh 표면 정합 오차</div><img src="{image_to_data_uri(error_hist)}"></div></div>'
    report = save_gallery(args.npz.parent, 'report.html', '05 RTX LiDAR — 노은역 smoke test', summary, body, eyebrow='gs_vlnpe lidar validation')
    print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
