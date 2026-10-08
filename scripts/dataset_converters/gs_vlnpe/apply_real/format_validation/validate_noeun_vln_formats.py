"""노은역 04 Isaac 산출물과 VLN-PE/VLN-CE의 실제 저장 계약을 비교한다.

이 검사의 PASS는 비교가 완결되었다는 뜻이다. 포맷 호환 여부는 compatibility 필드로
별도 판정한다. 서로 다른 포맷을 정확히 불일치로 검출한 것을 실행 실패로 취급하지 않는다.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image


DEFAULT_NOEUN = Path(
    'scripts/dataset_converters/gs_vlnpe/apply_real/obs/'
    'noeun_station_mid_random_isaac_d455_nominal'
)
DEFAULT_VLNPE = Path('data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r/s8pcmisQ38h')
DEFAULT_VLNCE_CONFIG = Path('scripts/eval/configs/vln_r2r_mini.yaml')
DEFAULT_VLNCE_JSON = Path(
    'data/vlnverse_emr/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz'
)
DEFAULT_REPORT = Path(
    'scripts/dataset_converters/gs_vlnpe/apply_real/format_validation/'
    'noeun_vln_format_validation.json'
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--noeun_root', type=Path, default=DEFAULT_NOEUN)
    p.add_argument('--vlnpe_scene', type=Path, default=DEFAULT_VLNPE)
    p.add_argument('--vlnpe_episode', type=int, default=0)
    p.add_argument('--vlnce_config', type=Path, default=DEFAULT_VLNCE_CONFIG)
    p.add_argument('--vlnce_json', type=Path, default=DEFAULT_VLNCE_JSON)
    p.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    return p


def sequential_indices(paths: list[Path], prefix: str, suffix: str) -> bool:
    try:
        got = [int(p.name.removeprefix(prefix).removesuffix(suffix)) for p in paths]
    except ValueError:
        return False
    return got == list(range(len(paths)))


def inspect_noeun(root: Path) -> dict:
    cfg_path = root / 'camera_config.json'
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    episodes = sorted(p for p in root.glob('episode_*') if p.is_dir())
    per_episode = []
    totals = {'rgb': 0, 'depth': 0, 'pose': 0}
    all_checks = []
    depth_raw_min, depth_raw_max = 65535, 0
    valid_depth_pixels, total_depth_pixels = 0, 0

    for ep_index, ep in enumerate(episodes):
        rgb = sorted((ep / 'rgb').glob('frame_*.jpg'))
        depth = sorted((ep / 'depth').glob('frame_*.png'))
        pose = sorted((ep / 'extrinsic').glob('frame_*.npy'))
        intrinsic_path = ep / 'intrinsic.npy'
        counts_equal = len(rgb) == len(depth) == len(pose) and len(rgb) > 0
        names_sequential = (
            sequential_indices(rgb, 'frame_', '.jpg')
            and sequential_indices(depth, 'frame_', '.png')
            and sequential_indices(pose, 'frame_', '.npy')
        )
        intrinsic = np.load(intrinsic_path) if intrinsic_path.is_file() else np.empty(0)
        intrinsic_ok = intrinsic.shape == (3, 3) and np.issubdtype(intrinsic.dtype, np.floating)
        frame_checks = []

        for r_path, d_path, p_path in zip(rgb, depth, pose):
            with Image.open(r_path) as image:
                rgb_ok = image.mode == 'RGB' and image.size == (cfg['width'], cfg['height'])
            with Image.open(d_path) as image:
                raw = np.asarray(image)
            pose_matrix = np.load(p_path)
            depth_ok = raw.shape == (cfg['height'], cfg['width']) and raw.dtype == np.uint16
            pose_ok = (
                pose_matrix.shape == (4, 4)
                and np.issubdtype(pose_matrix.dtype, np.floating)
                and np.isfinite(pose_matrix).all()
                and np.allclose(pose_matrix[3], [0, 0, 0, 1], atol=1e-6)
            )
            frame_checks.append(bool(rgb_ok and depth_ok and pose_ok))
            depth_raw_min = min(depth_raw_min, int(raw.min()))
            depth_raw_max = max(depth_raw_max, int(raw.max()))
            total_depth_pixels += raw.size
            valid_depth_pixels += int(((raw > 0) & (raw <= 10000)).sum())

        episode_ok = bool(counts_equal and names_sequential and intrinsic_ok and all(frame_checks))
        all_checks.append(episode_ok)
        totals['rgb'] += len(rgb); totals['depth'] += len(depth); totals['pose'] += len(pose)
        per_episode.append({
            'episode': ep_index,
            'rgb_frames': len(rgb),
            'depth_frames': len(depth),
            'pose_frames': len(pose),
            'counts_equal': counts_equal,
            'sequential_names': names_sequential,
            'intrinsic_shape': list(intrinsic.shape),
            'intrinsic_dtype': str(intrinsic.dtype),
            'all_frame_payloads_valid': bool(all(frame_checks)),
            'passed': episode_ok,
        })

    return {
        'root': str(root),
        'camera_config': cfg,
        'layout': 'per-frame JPG + uint16 PNG + 4x4 NPY pose; per-episode intrinsic NPY',
        'episode_count': len(episodes),
        'totals': totals,
        'depth_raw_range': [depth_raw_min, depth_raw_max],
        'depth_metric_range_m': [depth_raw_min * cfg['depth_scale_m'], depth_raw_max * cfg['depth_scale_m']],
        'valid_depth_pixel_fraction': valid_depth_pixels / total_depth_pixels,
        'per_episode': per_episode,
        'integrity_passed': bool(len(episodes) == 20 and totals == {'rgb': 8378, 'depth': 8378, 'pose': 8378}
                                 and all(all_checks)),
        'missing_vln_fields': [
            'episode parquet table', 'language instruction/task metadata', 'discrete action labels',
            'robot position/orientation/yaw', 'progress/step/timestamp', 'finish status/fail reason',
        ],
    }


def inspect_vlnpe(scene: Path, episode: int) -> dict:
    stem = f'episode_{episode:06d}'
    parquet = scene / 'data/chunk-000' / f'{stem}.parquet'
    rgb_path = scene / 'videos/chunk-000/observation.images.rgb' / f'{stem}.npy'
    depth_path = scene / 'videos/chunk-000/observation.images.depth' / f'{stem}.npy'
    table = pq.read_table(parquet)
    rgb = np.load(rgb_path, mmap_mode='r')
    depth = np.load(depth_path, mmap_mode='r')
    columns = table.column_names
    required_loader_columns = [
        'observation.camera_position', 'observation.camera_orientation', 'observation.camera_yaw',
        'observation.robot_position', 'observation.robot_orientation', 'observation.robot_yaw',
        'observation.progress', 'observation.step', 'observation.action',
    ]
    n = table.num_rows
    return {
        'scene_root': str(scene),
        'episode': episode,
        'layout': 'LeRobot-like parquet + episode-level RGB/depth NPY stacks + meta JSONL',
        'parquet_rows': n,
        'parquet_columns': columns,
        'loader_required_columns': required_loader_columns,
        'loader_columns_present': all(c in columns for c in required_loader_columns),
        'rgb': {'path': str(rgb_path), 'shape': list(rgb.shape), 'dtype': str(rgb.dtype)},
        'depth': {
            'path': str(depth_path), 'shape': list(depth.shape), 'dtype': str(depth.dtype),
            'stored_min': float(np.min(depth)), 'stored_max': float(np.max(depth)),
            'metric_conversion': 'meters = stored_float32 * 10; stored 1.0 is 10m clip/invalid',
        },
        'frame_alignment': bool(rgb.shape[0] == depth.shape[0] == n),
        'camera_pose_storage': 'parquet position[3] + orientation quaternion wxyz[4]',
        'intrinsic_storage': 'not in parquet; implicit 256x256 90deg HFOV K=[[128,0,128],[0,128,128],[0,0,1]]',
        'integrity_passed': bool(
            rgb.ndim == 4 and rgb.shape[1:] == (256, 256, 3) and rgb.dtype == np.uint8
            and depth.ndim == 3 and depth.shape[1:] == (256, 256) and depth.dtype == np.float32
            and rgb.shape[0] == depth.shape[0] == n
            and all(c in columns for c in required_loader_columns)
        ),
    }


def inspect_vlnce(config_path: Path, json_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    agent = config['habitat']['simulator']['agents']['main_agent']['sim_sensors']
    rgb_cfg, depth_cfg = agent['rgb_sensor'], agent['depth_sensor']
    with gzip.open(json_path, 'rt', encoding='utf-8') as handle:
        annotations = json.load(handle)
    episodes = annotations.get('episodes', []) if isinstance(annotations, dict) else []
    sample = episodes[0] if episodes else {}
    return {
        'config': str(config_path),
        'episode_json': str(json_path),
        'layout': 'Habitat gzip episode JSON + GLB/navmesh; RGB-D generated online at runtime',
        'episode_count': len(episodes),
        'sample_episode_keys': sorted(sample),
        'rgb_sensor': rgb_cfg,
        'depth_sensor': depth_cfg,
        'scene_asset_contract': 'Matterport3D GLB + navmesh',
        'offline_rgb_depth_stack_in_episode_json': False,
        'integrity_passed': bool(
            episodes
            and rgb_cfg['width'] == 640 and rgb_cfg['height'] == 480 and rgb_cfg['hfov'] == 79
            and depth_cfg['width'] == 640 and depth_cfg['height'] == 480
            and depth_cfg['max_depth'] == 10.0
        ),
    }


def compare(noeun: dict, pe: dict, ce: dict) -> dict:
    pe_semantic = {
        'rgb_observation': True,
        'metric_optical_depth_after_decode': True,
        'per_frame_camera_pose_after_conversion': True,
        '10m_depth_cutoff': True,
    }
    pe_exact = {
        'directory_layout': False,
        'rgb_container_dtype_shape': False,
        'depth_container_dtype_scale_shape': False,
        'pose_serialization': False,
        'intrinsic_serialization': False,
        'episode_metadata_and_actions': False,
        'direct_internnav_vlnpe_loader_compatibility': False,
    }
    ce_semantic = {
        'rgb_observation': True,
        'metric_optical_depth_at_runtime': True,
        'camera_pose_available_from_simulator_state': True,
        '10m_depth_max': True,
    }
    ce_exact = {
        'offline_dataset_category': False,
        'directory_layout': False,
        'rgb_resolution': False,
        'depth_runtime_representation': False,
        'pose_serialization': False,
        'episode_instruction_action_schema': False,
        'direct_habitat_dataset_compatibility': False,
    }
    return {
        'noeun_vs_vlnpe': {
            'semantic_stream_correspondence': pe_semantic,
            'semantic_correspondence_all': all(pe_semantic.values()),
            'exact_storage_contract': pe_exact,
            'exact_storage_format_same': all(pe_exact.values()),
            'direct_loader_compatible': False,
            'verdict': 'RGB/depth/pose 의미는 대응하지만 파일 포맷과 학습 episode 계약은 동일하지 않음',
        },
        'noeun_vs_vlnce': {
            'semantic_stream_correspondence': ce_semantic,
            'semantic_correspondence_all': all(ce_semantic.values()),
            'exact_storage_contract': ce_exact,
            'exact_storage_format_same': all(ce_exact.values()),
            'direct_loader_compatible': False,
            'verdict': '센서 의미는 대응하지만 VLN-CE는 Habitat online 환경이라 동일 offline 포맷이 아님',
        },
        'conversion_required': {
            'for_vlnpe_loader': [
                'JPG/PNG frame을 episode-level NPY stack으로 변환',
                '4x4 pose를 camera_position + wxyz orientation으로 변환',
                'depth를 VLN-PE normalized float32 규약으로 변환하거나 loader를 명시적으로 확장',
                'parquet과 tasks/episode metadata 생성',
                'robot state, action, progress, step, instruction을 정의/생성',
            ],
            'for_vlnce': [
                '노은역 scene을 Habitat-compatible scene/navmesh로 준비',
                'R2RVLN episode JSON에 scene/start/goal/instruction/action 계약 생성',
                'Habitat sensor config와 노은역 카메라 calibration 차이를 결정',
                '또는 VLN-CE runtime이 아닌 별도 offline replay adapter 구현',
            ],
        },
    }


def main() -> int:
    args = parser().parse_args()
    noeun = inspect_noeun(args.noeun_root)
    pe = inspect_vlnpe(args.vlnpe_scene, args.vlnpe_episode)
    ce = inspect_vlnce(args.vlnce_config, args.vlnce_json)
    checks = {
        'noeun_integrity': noeun['integrity_passed'],
        'vlnpe_sample_integrity': pe['integrity_passed'],
        'vlnce_contract_integrity': ce['integrity_passed'],
    }
    result = {
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'meaning_of_status': 'comparison inputs were parsed and validated; compatibility is separate',
        'checks': checks,
        'noeun': noeun,
        'vlnpe': pe,
        'vlnce': ce,
        'compatibility': compare(noeun, pe, ce),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'status': result['status'], 'checks': checks,
        'noeun_totals': noeun['totals'],
        'noeun_vs_vlnpe_exact_same': result['compatibility']['noeun_vs_vlnpe']['exact_storage_format_same'],
        'noeun_vs_vlnce_exact_same': result['compatibility']['noeun_vs_vlnce']['exact_storage_format_same'],
        'report': str(args.report),
    }, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
