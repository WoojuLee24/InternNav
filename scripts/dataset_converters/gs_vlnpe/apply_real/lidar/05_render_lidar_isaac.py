"""노은역 04 Isaac 관측 pose에서 RTX LiDAR를 재생한다.

첫 단계는 ``--smoke_test`` 전용이다. Isaac Sim 5.x의 OmniLidar/LidarRtx가 노은역
USDZ geometry에서 실제 return을 만드는지 한 pose로 확인한다. 센서 모델과 장착 자세가
확정되기 전에는 전체 8,378 frame 생성을 허용하지 않는다.

실행:
    /workspace/isaaclab/_isaac_sim/python.sh \
      scripts/dataset_converters/gs_vlnpe/apply_real/lidar/05_render_lidar_isaac.py --smoke_test
"""

from isaacsim import SimulationApp

# RTX LiDAR 한 개에는 multi-GPU가 필요 없다. 이 장비의 4-GPU 자동 경로에서 첫 render 때
# copy-queue semaphore timeout이 실측됐으므로 GPU 0 하나로 고정한다.
simulation_app = SimulationApp({'headless': True, 'multi_gpu': False, 'active_gpu': 0})

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleXFormPrim  # noqa: E402
from isaacsim.sensors.rtx import LidarRtx  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[5]
APPLY_REAL_ROOT = REPO_ROOT / 'scripts/dataset_converters/gs_vlnpe/apply_real'
sys.path.insert(0, str(APPLY_REAL_ROOT))
import usdz_scene_utils  # noqa: E402
from geometry_utils import action_to_c2w  # noqa: E402
DEFAULT_SCENE_USD = REPO_ROOT / 'data/noeun_station/noeun_station_collision.usdz'
DEFAULT_OBS_ROOT = (
    REPO_ROOT
    / 'scripts/dataset_converters/gs_vlnpe/apply_real/obs'
    / 'noeun_station_mid_random_isaac_d455_nominal'
)
DEFAULT_OUTPUT_ROOT = DEFAULT_OBS_ROOT / 'lidar_rtx'
DEFAULT_PATH_JSON = (
    APPLY_REAL_ROOT / 'paths/noeun_station_mid_random.json'
)
DEFAULT_PROFILE = 'OS1_REV6_32ch10hz1024res'
DEFAULT_MODEL = 'OS1'
ANNOTATOR = 'IsaacCreateRTXLidarScanBuffer'
SCAN_BUFFER_OPTIONS = {
    'outputIntensity': True,
    'outputDistance': True,
    'outputObjectId': False,
    'outputVelocity': False,
    'outputAzimuth': True,
    'outputElevation': True,
    'outputNormal': False,
    'outputTimestamp': True,
    'outputEmitterId': True,
    'outputBeamId': True,
    'outputMaterialId': False,
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke_test', action='store_true', help='episode/frame 하나만 검사하고 저장')
    parser.add_argument('--scene_usd', type=Path, default=DEFAULT_SCENE_USD)
    parser.add_argument('--obs_root', type=Path, default=DEFAULT_OBS_ROOT)
    parser.add_argument('--output_root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--path_json', type=Path, default=DEFAULT_PATH_JSON)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--profile', default=DEFAULT_PROFILE)
    parser.add_argument('--warmup_frames', type=int, default=30)
    return parser


def load_episode_robot_contract(path_json: Path, episode: int) -> dict:
    payload = json.loads(path_json.read_text(encoding='utf-8'))
    matches = [item for item in payload['episodes'] if int(item['episode_id']) == episode]
    if len(matches) != 1:
        raise ValueError(f'episode {episode}의 robot 메타데이터를 하나로 결정할 수 없습니다: {path_json}')
    return matches[0]


def load_camera_pose(obs_root: Path, episode: int, frame: int) -> tuple[np.ndarray, Path]:
    path = obs_root / f'episode_{episode:06d}' / 'extrinsic' / f'frame_{frame:04d}.npy'
    if not path.is_file():
        raise FileNotFoundError(f'04 pose가 없습니다: {path}')
    pose = np.asarray(np.load(path), dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f'pose shape이 (4,4)가 아닙니다: {pose.shape} ({path})')
    if not np.all(np.isfinite(pose)):
        raise ValueError(f'pose에 NaN/Inf가 있습니다: {path}')
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f'pose homogeneous 마지막 행이 잘못됐습니다: {pose[3]}')
    return pose, path


def camera_pose_to_level_lidar_pose(camera_action_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """카메라 위치와 yaw만 사용해 수평 LiDAR pose(wxyz)를 만든다.

    04 카메라는 약 15도 아래를 보지만, 임시 rotary LiDAR smoke test는 world-up 기준 수평으로
    둔다. 실제 장착 자세가 확정되면 이 정책을 profile 계약으로 교체해야 한다.
    """
    camera_pose_c2w = action_to_c2w(camera_action_pose, 'cam2world_gl')
    forward = camera_pose_c2w[:3, 2].copy()
    forward[2] = 0.0
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError('카메라 forward를 XY 평면에 투영할 수 없습니다.')
    forward /= norm
    yaw = float(np.arctan2(forward[1], forward[0]))
    quat_xyzw = Rotation.from_euler('z', yaw).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return camera_pose_c2w[:3, 3].copy(), quat_wxyz


def pose_matrix(position: np.ndarray, orientation_wxyz: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(np.roll(np.asarray(orientation_wxyz), -1)).as_matrix()
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def matrix_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quat_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return transform[:3, 3].copy(), np.roll(quat_xyzw, 1)


def load_scene(scene_usd: Path) -> None:
    if not scene_usd.is_file():
        raise FileNotFoundError(f'노은역 USDZ가 없습니다: {scene_usd}')
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(scene_usd),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )
    cfg.func('/World/Scene', cfg)
    stage = omni.usd.get_context().get_stage()
    changed = usdz_scene_utils.expose_collision_meshes_for_rendering(stage, scene_usd, UsdGeom)
    print(f'[lidar-smoke] RTX-visible collision meshes={changed}', flush=True)


def acquire_smoke_frame(args) -> dict:
    camera_action_pose, pose_path = load_camera_pose(args.obs_root, args.episode, args.frame)
    camera_pose_c2w = action_to_c2w(camera_action_pose, 'cam2world_gl')
    lidar_position, lidar_orientation = camera_pose_to_level_lidar_pose(camera_action_pose)
    robot_contract = load_episode_robot_contract(args.path_json, args.episode)
    floor_z = float(robot_contract['floor_z'])
    h_b = float(robot_contract['h_b'])

    load_scene(args.scene_usd)
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    _, sensor_prim = omni.kit.commands.execute(
        'IsaacSensorCreateRtxLidar',
        path='/World/NoeunLidar',
        parent=None,
        config=DEFAULT_MODEL,
        variant=args.profile,
        translation=Gf.Vec3d(*[float(v) for v in lidar_position]),
        orientation=Gf.Quatd(
            float(lidar_orientation[0]),
            Gf.Vec3d(*[float(v) for v in lidar_orientation[1:]]),
        ),
    )
    if sensor_prim is None:
        raise RuntimeError(f'OS1 variant 생성 실패: model={DEFAULT_MODEL}, variant={args.profile}')
    asset_path = sensor_prim.GetPath()
    stage = omni.usd.get_context().get_stage()
    lidar_prims = [
        prim for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(asset_path) and prim.GetTypeName() == 'OmniLidar'
    ]
    if len(lidar_prims) != 1:
        found = [(str(prim.GetPath()), prim.GetTypeName()) for prim in stage.Traverse()
                 if prim.GetPath().HasPrefix(asset_path)]
        raise RuntimeError(f'OS1 asset의 OmniLidar prim을 하나로 결정할 수 없습니다: {found}')
    lidar_prim_path = str(lidar_prims[0].GetPath())
    lidar = LidarRtx(
        prim_path=lidar_prim_path,
        name='noeun_lidar',
        **{'omni:sensor:Core:outputFrameOfReference': 'SENSOR'},
        **{'omni:sensor:Core:auxOutputType': 'FULL'},
    )
    world.reset()
    lidar.initialize()

    asset_xform = SingleXFormPrim(str(asset_path), reset_xform_properties=False)
    root_position, root_orientation = asset_xform.get_world_pose()
    sensor_position, sensor_orientation = lidar.get_world_pose()
    t_root_sensor = np.linalg.inv(pose_matrix(root_position, root_orientation)) @ pose_matrix(
        sensor_position, sensor_orientation
    )
    desired_t_world_sensor = pose_matrix(lidar_position, lidar_orientation)
    desired_t_world_root = desired_t_world_sensor @ np.linalg.inv(t_root_sensor)
    desired_root_position, desired_root_orientation = matrix_pose(desired_t_world_root)
    asset_xform.set_world_pose(desired_root_position, desired_root_orientation)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(3):
        world.step(render=True)
    actual_lidar_position, actual_lidar_orientation = lidar.get_world_pose()
    actual_lidar_position = np.asarray(actual_lidar_position, dtype=np.float64)
    actual_lidar_orientation = np.asarray(actual_lidar_orientation, dtype=np.float64)
    mount_position_error = float(np.linalg.norm(actual_lidar_position - lidar_position))
    if mount_position_error > 1e-5:
        raise RuntimeError(f'OS1 내부 offset 역보정 실패: position error={mount_position_error:.9f} m')
    lidar.attach_annotator(ANNOTATOR, transformPoints=False, **SCAN_BUFFER_OPTIONS)
    data = None
    used_frames = 0
    for used_frames in range(1, args.warmup_frames + 1):
        world.step(render=True)
        current = lidar.get_current_frame().get(ANNOTATOR)
        if current and 'data' in current and np.asarray(current['data']).size > 0:
            # Annotator 배열은 다음 render/timeline 상태 변경 때 재사용되는 view일 수 있다.
            # return을 확인한 바로 이 지점에서 전부 소유 메모리로 동결한다.
            candidate = {}
            for key, value in current.items():
                if key == 'info':
                    candidate[key] = {subkey: np.array(subvalue, copy=True)
                                      for subkey, subvalue in value.items()}
                else:
                    candidate[key] = np.array(value, copy=True)
            emitter = candidate.get('emitterId', np.empty(0))
            azimuth = candidate.get('azimuth', np.empty(0))
            channel_complete = np.array_equal(np.unique(emitter), np.arange(32))
            azimuth_complete = azimuth.size > 0 and float(azimuth.max() - azimuth.min()) >= 359.0
            if not channel_complete or not azimuth_complete:
                print(
                    f'[lidar-smoke] incomplete/corrupt candidate at render {used_frames}: '
                    f'channels={np.unique(emitter).tolist()} azimuth_span='
                    f'{float(azimuth.max() - azimuth.min()) if azimuth.size else 0.0:.3f}',
                    flush=True,
                )
                continue
            data = candidate
            break
    timeline.stop()

    if data is None:
        raise RuntimeError(
            f'{args.warmup_frames} render frame 동안 LiDAR return이 없습니다. '
            'scene visibility/profile/timeline을 확인해야 합니다.'
        )

    points_lidar = np.array(data['data'], dtype=np.float32, copy=True)
    if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
        raise RuntimeError(f'예상하지 못한 pointcloud shape: {points_lidar.shape}')
    finite = np.all(np.isfinite(points_lidar), axis=1)
    ranges = np.linalg.norm(points_lidar, axis=1).astype(np.float32)
    valid = finite & (ranges > 0.0)
    if not np.any(valid):
        raise RuntimeError('반환점은 있으나 유효한 finite positive-range 점이 없습니다.')

    out_dir = args.output_root / 'smoke_test'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'episode_{args.episode:06d}_frame_{args.frame:04d}'
    npz_path = out_dir / f'{stem}.npz'
    lidar_rotation = Rotation.from_quat(np.roll(actual_lidar_orientation, -1)).as_matrix()
    t_world_lidar = np.eye(4, dtype=np.float64)
    t_world_lidar[:3, :3] = lidar_rotation
    t_world_lidar[:3, 3] = actual_lidar_position
    yaw = float(np.arctan2(lidar_rotation[1, 0], lidar_rotation[0, 0]))
    t_world_robot = np.eye(4, dtype=np.float64)
    t_world_robot[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    t_world_robot[:3, 3] = [lidar_position[0], lidar_position[1], floor_z]
    t_robot_lidar = np.linalg.inv(t_world_robot) @ t_world_lidar
    points_world = points_lidar @ t_world_lidar[:3, :3].T + t_world_lidar[:3, 3]
    points_robot = ((points_world - t_world_robot[:3, 3]) @ t_world_robot[:3, :3]).astype(np.float32)

    auxiliary = {}
    for source_name, output_name in (
        ('distance', 'ranges_annotator_m'),
        ('intensity', 'intensity'),
        ('timestamp', 'timestamps_ns'),
        ('azimuth', 'azimuth'),
        ('elevation', 'elevation'),
        ('beamId', 'beam_id'),
        ('emitterId', 'emitter_id'),
    ):
        if source_name in data:
            auxiliary[output_name] = np.array(data[source_name], copy=True)

    np.savez_compressed(
        npz_path,
        points_lidar_m=points_lidar,
        points_robot_m=points_robot,
        points_world=points_world.astype(np.float32),
        ranges_m=ranges,
        valid_mask=valid,
        sensor_position_world=actual_lidar_position.astype(np.float32),
        sensor_orientation_world_wxyz=actual_lidar_orientation.astype(np.float32),
        requested_mount_position_world=lidar_position.astype(np.float32),
        requested_mount_orientation_world_wxyz=lidar_orientation.astype(np.float32),
        source_camera_action_pose_gl=camera_action_pose.astype(np.float32),
        camera_pose_c2w=camera_pose_c2w.astype(np.float32),
        T_world_robot=t_world_robot.astype(np.float32),
        T_robot_lidar=t_robot_lidar.astype(np.float32),
        T_world_lidar=t_world_lidar.astype(np.float32),
        floor_z_m=np.float32(floor_z),
        camera_height_m=np.float32(h_b),
        scanbuffer_transform=np.array(data.get('info', {}).get('transform', []), dtype=np.float64, copy=True),
        episode_index=np.int64(args.episode),
        frame_index=np.int64(args.frame),
        **auxiliary,
    )

    valid_ranges = ranges[valid]
    result = {
        'status': 'PASS',
        'purpose': 'OS1-32 ScanBuffer contract smoke test before full generation',
        'profile': args.profile,
        'annotator': ANNOTATOR,
        'output_frame': 'raw SENSOR plus explicit robot/base_link and world transforms',
        'mount_policy': 'robot x-forward/y-left/z-up; LiDAR at camera position, level and yaw-aligned',
        'asset_internal_sensor_offset_m': t_root_sensor[:3, 3].tolist(),
        'mount_position_error_m': mount_position_error,
        'scene_usd': str(args.scene_usd),
        'source_pose': str(pose_path),
        'output_npz': str(npz_path),
        'warmup_frames_used': used_frames,
        'point_count': int(len(points_lidar)),
        'valid_point_count': int(valid.sum()),
        'finite_fraction': float(finite.mean()),
        'range_m': {
            'min': float(valid_ranges.min()),
            'median': float(np.median(valid_ranges)),
            'p95': float(np.percentile(valid_ranges, 95)),
            'max': float(valid_ranges.max()),
        },
        'arrays': {
            key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
            for key, value in auxiliary.items()
        },
        'robot_frame': {
            'origin_world_m': t_world_robot[:3, 3].tolist(),
            'camera_height_m': h_b,
            'floor_z_m': floor_z,
            'axes': 'x forward, y left, z up',
        },
        'scanbuffer_contract': {},
    }
    for key, value in data.items():
        if key == 'info':
            continue
        array = np.asarray(value)
        result['scanbuffer_contract'][key] = {
            'shape': list(array.shape),
            'dtype': str(array.dtype),
        }
    info = data.get('info', {})
    for key, value in info.items():
        array = np.asarray(value)
        entry = {'shape': list(array.shape), 'dtype': str(array.dtype)}
        if array.size == 1:
            scalar = array.reshape(-1)[0]
            entry['value'] = scalar.item() if hasattr(scalar, 'item') else scalar
        result['scanbuffer_contract'][f'info.{key}'] = entry
    if 'timestamps_ns' in auxiliary and auxiliary['timestamps_ns'].size:
        timestamp = auxiliary['timestamps_ns'].astype(np.uint64, copy=False)
        result['timestamp_ns'] = {
            'min': int(timestamp.min()),
            'max': int(timestamp.max()),
            'span': int(timestamp.max() - timestamp.min()),
            'monotonic_non_decreasing': bool(np.all(np.diff(timestamp) >= 0)),
        }
    if 'azimuth' in auxiliary and auxiliary['azimuth'].size:
        azimuth = auxiliary['azimuth'].astype(np.float64, copy=False)
        result['azimuth_observed'] = {
            'min': float(azimuth.min()),
            'max': float(azimuth.max()),
            'unit_not_assumed_until_verified': True,
        }
    if 'beam_id' in auxiliary and auxiliary['beam_id'].size:
        result['beam_id_unique'] = [int(v) for v in np.unique(auxiliary['beam_id'])]
    report_path = out_dir / f'{stem}.json'
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    result['report_json'] = str(report_path)
    return result


def main() -> int:
    args = build_argparser().parse_args()
    exit_code = 0
    try:
        if not args.smoke_test:
            raise RuntimeError(
                '전체 생성은 smoke test와 geometry 검증 통과 후 활성화합니다. 먼저 --smoke_test를 사용하세요.'
            )
        print(f'[lidar-smoke] scene={args.scene_usd}', flush=True)
        print(f'[lidar-smoke] obs={args.obs_root} ep={args.episode} frame={args.frame}', flush=True)
        result = acquire_smoke_frame(args)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    except Exception:
        # SimulationApp.close()가 종료 중 예외 출력을 가릴 수 있어 close 전에 명시적으로 남긴다.
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
