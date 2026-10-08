"""검증된 여러 OS1-32 frame을 한 페이지에서 비교하는 self-contained HTML report."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from viz_utils import image_to_data_uri, save_gallery


DEFAULT_DIR = Path(
    'scripts/dataset_converters/gs_vlnpe/apply_real/obs/'
    'noeun_station_mid_random_isaac_d455_nominal/lidar_rtx/smoke_test'
)
DEFAULT_FRAMES = '0,142,284'


def emit(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def parse_frames(text: str) -> list[int]:
    result = [int(item.strip()) for item in text.split(',') if item.strip()]
    if not result or len(result) != len(set(result)) or any(item < 0 for item in result):
        raise ValueError(f'--frames는 중복 없는 0 이상 정수 목록이어야 합니다: {text!r}')
    return result


def load_frame(npz_path: Path) -> dict:
    validation_path = npz_path.with_suffix('.validation.json')
    if not validation_path.is_file():
        raise FileNotFoundError(f'먼저 독립 검증을 실행해야 합니다: {validation_path}')
    validation = json.loads(validation_path.read_text(encoding='utf-8'))
    with np.load(npz_path) as data:
        payload = {name: np.array(data[name], copy=True) for name in (
            'points_robot_m', 'points_world', 'ranges_m', 'valid_mask',
            'camera_pose_c2w', 'intensity', 'emitter_id', 'episode_index', 'frame_index',
        )}
    payload['validation'] = validation
    payload['npz_path'] = npz_path
    return payload


def project_to_rgb(frame: dict, obs_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episode = int(frame['episode_index'])
    index = int(frame['frame_index'])
    ep_dir = obs_root / f'episode_{episode:06d}'
    rgb = np.asarray(Image.open(ep_dir / 'rgb' / f'frame_{index:04d}.jpg').convert('RGB'))
    depth_raw = np.asarray(Image.open(ep_dir / 'depth' / f'frame_{index:04d}.png'))
    intrinsic = np.load(ep_dir / 'intrinsic.npy').astype(np.float64)
    points = frame['points_world'].astype(np.float64)
    c2w = frame['camera_pose_c2w'].astype(np.float64)
    camera = (points - c2w[:3, 3]) @ c2w[:3, :3]
    z = camera[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]
        v = intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]
    h, w = depth_raw.shape
    mask = frame['valid_mask'] & (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return rgb, depth_raw.astype(np.float32) * .001, u[mask], v[mask]


def render_frame_card(frame: dict, obs_root: Path, assets: Path) -> Path:
    episode = int(frame['episode_index'])
    index = int(frame['frame_index'])
    valid = frame['valid_mask']
    robot = frame['points_robot_m'][valid]
    ranges = frame['ranges_m'][valid]
    rgb, depth_m, u, v = project_to_rgb(frame, obs_root)

    # project_to_rgb와 같은 mask를 다시 만들어 overlay 색을 정확히 맞춘다.
    points = frame['points_world'].astype(np.float64)
    c2w = frame['camera_pose_c2w'].astype(np.float64)
    camera = (points - c2w[:3, 3]) @ c2w[:3, :3]
    z = camera[:, 2]
    intrinsic = np.load(
        obs_root / f'episode_{episode:06d}' / 'intrinsic.npy'
    ).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        all_u = intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]
        all_v = intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]
    h, w = depth_m.shape
    projected = valid & (z > 0) & (all_u >= 0) & (all_u < w) & (all_v >= 0) & (all_v < h)
    overlay_ranges = frame['ranges_m'][projected]
    order = np.argsort(z[projected])[::-1]

    stride = max(1, len(robot) // 18000)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].imshow(rgb)
    scatter = axes[0, 0].scatter(
        u[order], v[order], c=overlay_ranges[order], s=1.5, cmap='turbo',
        vmin=.3, vmax=max(10, float(np.percentile(overlay_ranges, 99))), alpha=.78,
    )
    figure.colorbar(scatter, ax=axes[0, 0], label='OS1 range [m]')
    axes[0, 0].set_title(f'frame {index}: RGB + all in-FOV OS1 returns'); axes[0, 0].axis('off')

    shown_depth = depth_m.copy(); shown_depth[shown_depth <= 0] = np.nan
    depth_image = axes[0, 1].imshow(shown_depth, cmap='turbo', vmin=.1, vmax=10)
    figure.colorbar(depth_image, ax=axes[0, 1], label='D455 depth [m]')
    axes[0, 1].set_title('D455 metric depth (calibration valid to 10 m)'); axes[0, 1].axis('off')

    points_draw, ranges_draw = robot[::stride], ranges[::stride]
    robot_scatter = axes[1, 0].scatter(
        points_draw[:, 0], points_draw[:, 1], c=ranges_draw, s=.3, cmap='turbo'
    )
    axes[1, 0].scatter(0, 0, marker='*', s=90, c='white', edgecolors='black')
    figure.colorbar(robot_scatter, ax=axes[1, 0], label='OS1 range [m]')
    axes[1, 0].set_aspect('equal'); axes[1, 0].set_xlabel('robot X forward [m]')
    axes[1, 0].set_ylabel('robot Y left [m]'); axes[1, 0].set_title('robot/base_link point cloud')

    axes[1, 1].hist(ranges, bins=100, color='#4fd1c5', alpha=.9)
    axes[1, 1].axvline(np.median(ranges), color='#e2725b', linestyle='--',
                       label=f'median {np.median(ranges):.2f} m')
    axes[1, 1].set_xlabel('range [m]'); axes[1, 1].set_ylabel('returns')
    axes[1, 1].set_title('Full OS1 range distribution'); axes[1, 1].legend()
    figure.suptitle(f'Noeun middle level — episode {episode} frame {index}', fontsize=16)
    return emit(assets / f'frame_{index:04d}_summary.png')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--frames', default=DEFAULT_FRAMES)
    parser.add_argument('--report_name', default='three_frame_report.html')
    args = parser.parse_args()

    frame_indices = parse_frames(args.frames)
    paths = [args.input_dir / f'episode_{args.episode:06d}_frame_{index:04d}.npz'
             for index in frame_indices]
    frames = [load_frame(path) for path in paths]
    if any(int(frame['episode_index']) != args.episode for frame in frames):
        raise ValueError('NPZ 내부 episode_index가 요청 episode와 다릅니다.')

    obs_root = args.input_dir.parents[1]
    assets = args.input_dir / 'three_frame_report_assets'
    images = [render_frame_card(frame, obs_root, assets) for frame in frames]
    validations = [frame['validation'] for frame in frames]
    overall_pass = all(item['status'] == 'PASS' for item in validations)

    point_counts = np.array([item['point_count'] for item in validations])
    mesh_median_mm = np.array([item['surface_distance_m']['median'] * 1000 for item in validations])
    mesh_p95_mm = np.array([item['surface_distance_m']['p95'] * 1000 for item in validations])
    calib_median_cm = np.array([
        item['rgbd_lidar_calibration']['median_abs_error_m'] * 100 for item in validations
    ])
    calib_p95_cm = np.array([
        item['rgbd_lidar_calibration']['p95_abs_error_m'] * 100 for item in validations
    ])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(frame_indices, point_counts, marker='o'); axes[0, 0].set_title('return count')
    axes[0, 1].plot(frame_indices, mesh_median_mm, marker='o', label='median')
    axes[0, 1].plot(frame_indices, mesh_p95_mm, marker='o', label='p95')
    axes[0, 1].axhline(2, color='gray', linestyle=':', label='median gate 2 mm')
    axes[0, 1].axhline(10, color='black', linestyle=':', label='p95 gate 10 mm')
    axes[0, 1].set_title('mesh alignment [mm]'); axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(frame_indices, calib_median_cm, marker='o', label='median')
    axes[1, 0].plot(frame_indices, calib_p95_cm, marker='o', label='p95')
    axes[1, 0].axhline(5, color='gray', linestyle=':', label='median gate 5 cm')
    axes[1, 0].axhline(25, color='black', linestyle=':', label='p95 gate 25 cm')
    axes[1, 0].set_title('RGB-D/LiDAR residual [cm]'); axes[1, 0].legend(fontsize=8)
    axes[1, 1].axis('off')
    trend = emit(assets / 'three_frame_metrics.png')

    rows = []
    for index, validation in zip(frame_indices, validations):
        badge = 'good' if validation['status'] == 'PASS' else 'bad'
        rows.append(
            f'<tr><td>{index}</td><td><span class="pill {badge}">{validation["status"]}</span></td>'
            f'<td>{validation["point_count"]:,}</td>'
            f'<td>{validation["surface_distance_m"]["median"]*1000:.3f}</td>'
            f'<td>{validation["surface_distance_m"]["p95"]*1000:.3f}</td>'
            f'<td>{validation["rgbd_lidar_calibration"]["median_abs_error_m"]*100:.3f}</td>'
            f'<td>{validation["rgbd_lidar_calibration"]["p95_abs_error_m"]*100:.3f}</td></tr>'
        )
    badge = 'good' if overall_pass else 'bad'
    summary = f'''
<div class="stat-row">
 <div class="stat"><div class="label">3-frame 판정</div><div class="value"><span class="pill {badge}">{"PASS" if overall_pass else "FAIL"}</span></div></div>
 <div class="stat"><div class="label">frames</div><div class="value">{len(frames)}</div></div>
 <div class="stat"><div class="label">returns 범위</div><div class="value">{point_counts.min():,}–{point_counts.max():,}</div></div>
 <div class="stat"><div class="label">mesh median 최악</div><div class="value">{mesh_median_mm.max():.3f} mm</div></div>
 <div class="stat"><div class="label">RGB-D/LiDAR p95 최악</div><div class="value">{calib_p95_cm.max():.3f} cm</div></div>
</div>
<p>episode {args.episode}의 총 285 frames 중 시작·중간·끝({', '.join(map(str, frame_indices))})을 실제 OS1-32로 생성해 비교했다. 각 frame은 독립 16-gate 검증을 먼저 통과해야 이 report에 포함된다.</p>
<table><tr><th>frame</th><th>status</th><th>returns</th><th>mesh median [mm]</th><th>mesh p95 [mm]</th><th>RGB-D/LiDAR median [cm]</th><th>RGB-D/LiDAR p95 [cm]</th></tr>{''.join(rows)}</table>
'''
    body = f'<div class="card"><div class="caption">frame별 수치 추세</div><img src="{image_to_data_uri(trend)}"></div><div class="grid">'
    for index, image in zip(frame_indices, images):
        body += f'<div class="card"><div class="caption">episode {args.episode} / frame {index}</div><img src="{image_to_data_uri(image)}"></div>'
    body += '</div>'
    report = save_gallery(
        args.input_dir, args.report_name, 'OS1-32 다중-frame 검증 — 노은역 중간층',
        summary, body, eyebrow='gs_vlnpe lidar multi-frame validation',
    )
    aggregate = {
        'status': 'PASS' if overall_pass else 'FAIL',
        'episode': args.episode,
        'frames': frame_indices,
        'inputs': [str(path) for path in paths],
        'report_html': str(report),
    }
    (args.input_dir / 'three_frame_validation.json').write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0 if overall_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
