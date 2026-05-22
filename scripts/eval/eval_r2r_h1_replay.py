"""
r2r traj_data의 각 step position에 H1 로봇을 직접 teleport하여
새로운 RGB/depth 이미지를 캡처하고 r2r_h1_replay를 생성한다.

핵심 설계:
- r2r 수집 방식: vln_move_by_speed (연속 이동, step=0,50,100,...)
- 재현 방식: parquet의 각 row position/orientation에 직접 teleport → 이미지 캡처
  → action replay가 아닌 position replay → 정확히 같은 시점/위치에서 이미지 생성
- parquet row 하나 = 하나의 synthetic episode
- RGB/depth: 새로 캡처 / parquet, meta: r2r에서 그대로 복사

Usage:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/eval_r2r_h1_replay.py \
        --config   scripts/eval/configs/h1_internvla_n1_async_cfg_orig.py \
        --r2r_dir  data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r \
        --save_dir data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r_h1_replay \
        --scenes   sT4fr6TAbpF ur6pFq6Qu1A
"""
import sys
sys.path.append('.')
sys.path.append('./third_party/diffusion-policy')

import argparse
import collections
import gzip
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────
# 좌표 역변환: Isaac → Habitat
# ─────────────────────────────────────────────────────────────────

def _isaac_pos_to_habitat(isaac_pos):
    px, py, pz = float(isaac_pos[0]), float(isaac_pos[1]), float(isaac_pos[2])
    return [px, pz - 1.05, -py]


def _quat_mul(q1, q2):
    """quaternion 곱, [w,x,y,z] 포맷"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def _isaac_ori_to_habitat(isaac_ori):
    """Isaac robot_orientation [w,x,y,z] → Habitat start_rotation [x,y,z,w]"""
    c = math.cos(math.pi / 4)
    s = -math.sin(math.pi / 4)
    q_z90_inv = [c, 0.0, 0.0, s]
    qi = _quat_mul(list(isaac_ori), q_z90_inv)
    return [qi[1], -qi[3], qi[2], -qi[0]]


# ─────────────────────────────────────────────────────────────────
# r2r traj_data 로드 — 모든 row의 position/orientation 포함
# ─────────────────────────────────────────────────────────────────

def load_r2r_episodes(r2r_dir, target_scenes=None, max_eps=None):
    """
    반환: {scene: [(ep_idx, [(pos_0,ori_0), (pos_1,ori_1), ...], instruction, tokens), ...]}
    """
    import pyarrow.parquet as pq

    scenes = sorted(os.listdir(r2r_dir))
    if target_scenes:
        scenes = [s for s in scenes if s in target_scenes]

    result = {}
    for scene in scenes:
        data_dir = os.path.join(r2r_dir, scene, 'data', 'chunk-000')
        if not os.path.isdir(data_dir):
            continue

        tasks_path = os.path.join(r2r_dir, scene, 'meta', 'tasks.jsonl')
        task_by_idx = {}
        if os.path.exists(tasks_path):
            for line in open(tasks_path):
                t = json.loads(line)
                task_by_idx[t['task_index']] = t

        parquets = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
        if max_eps is not None:
            parquets = parquets[:max_eps]

        episodes = []
        for pf in parquets:
            ep_idx = int(pf.split('_')[1].split('.')[0])
            tbl    = pq.read_table(os.path.join(data_dir, pf)).to_pandas()
            poses  = [
                (np.array(tbl['observation.robot_position'].iloc[i]),
                 np.array(tbl['observation.robot_orientation'].iloc[i]))
                for i in range(len(tbl))
            ]
            task = task_by_idx.get(ep_idx, {})
            episodes.append((ep_idx, poses,
                             task.get('task', ''),
                             task.get('instruction_tokens', [])))

        result[scene] = episodes
        total_rows = sum(len(poses) for _, poses, _, _ in episodes)
        print(f'  {scene}: {len(episodes)} episodes, {total_rows} rows loaded')

    return result


# ─────────────────────────────────────────────────────────────────
# synthetic dataset — parquet row 하나 = synthetic episode 하나
# ─────────────────────────────────────────────────────────────────

def create_synthetic_dataset(r2r_data, out_dir):
    """
    각 parquet row를 하나의 synthetic episode로 변환.
    반환: mapping = [(scene, ep_idx, row_idx), ...]  (env 처리 순서와 동일)
    """
    episodes = []
    mapping  = []

    for scene in sorted(r2r_data.keys()):
        for ep_idx, poses, instruction, tokens in r2r_data[scene]:
            for row_idx, (pos, ori) in enumerate(poses):
                hab_pos = _isaac_pos_to_habitat(pos)
                hab_rot = _isaac_ori_to_habitat(ori)
                global_idx = len(mapping)
                episodes.append({
                    'episode_id':     global_idx,
                    'trajectory_id':  3_000_000 + global_idx,
                    'scene_id':       f'mp3d/{scene}/{scene}.glb',
                    'start_position': hab_pos,
                    'start_rotation': hab_rot,
                    'info':           {'geodesic_distance': 1.0},
                    'goals':          [{'position': hab_pos, 'radius': 3.0}],
                    'instruction':    {'instruction_text': instruction,
                                       'instruction_tokens': tokens},
                    'reference_path': [hab_pos, hab_pos],
                })
                mapping.append((scene, ep_idx, row_idx))

    split_dir = Path(out_dir) / 'train'
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / 'train.json.gz'
    with gzip.open(out_path, 'wt', encoding='utf-8') as f:
        json.dump({'episodes': episodes}, f)

    print(f'Synthetic dataset: {len(episodes)} episodes (1 per row) → {out_path}')
    return mapping


# ─────────────────────────────────────────────────────────────────
# 저장 / 복사
# ─────────────────────────────────────────────────────────────────

def save_images(save_dir, scan, episode_index, frame_list):
    """frame_list: [{'rgb': arr, 'depth': arr}, ...]. 반환: 저장된 frame 수"""
    import cv2

    scene_dir = Path(save_dir) / scan
    rgb_dir   = scene_dir / 'videos' / 'chunk-000' / 'observation.images.rgb'
    dep_dir   = scene_dir / 'videos' / 'chunk-000' / 'observation.images.depth'
    rgb_dir.mkdir(parents=True, exist_ok=True)
    dep_dir.mkdir(parents=True, exist_ok=True)

    ep_fname = f'episode_{episode_index:06d}'
    rgbs, depths = [], []

    for frame in frame_list:
        rgb = frame.get('rgb')
        if rgb is not None:
            if rgb.shape[:2] != (256, 256):
                rgb = cv2.resize(rgb, (256, 256))
            rgbs.append(rgb.astype(np.uint8))

        depth = frame.get('depth')
        if depth is not None:
            if depth.ndim == 3:
                depth = depth[..., 0]
            if depth.shape != (256, 256):
                depth = cv2.resize(depth, (256, 256))
            depths.append(depth.astype(np.float32))

    if rgbs:
        np.save(rgb_dir / f'{ep_fname}.npy', np.array(rgbs,   dtype=np.uint8))
    if depths:
        np.save(dep_dir / f'{ep_fname}.npy', np.array(depths, dtype=np.float32))

    return len(rgbs)


def copy_parquet(r2r_dir, save_dir, scan, episode_index):
    src = Path(r2r_dir) / scan / 'data' / 'chunk-000' / f'episode_{episode_index:06d}.parquet'
    dst = Path(save_dir) / scan / 'data' / 'chunk-000' / f'episode_{episode_index:06d}.parquet'
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_meta(r2r_dir, save_dir, scan):
    src_meta = Path(r2r_dir) / scan / 'meta'
    dst_meta = Path(save_dir) / scan / 'meta'
    dst_meta.mkdir(parents=True, exist_ok=True)

    for fname in ['episodes.jsonl', 'tasks.jsonl', 'episodes_stats.jsonl']:
        src = src_meta / fname
        if src.exists():
            shutil.copy2(src, dst_meta / fname)

    info_src = src_meta / 'info.json'
    if info_src.exists():
        with open(info_src) as f:
            info = json.load(f)
        info['robot_type'] = 'h1'
        with open(dst_meta / 'info.json', 'w') as f:
            json.dump(info, f, indent=2)


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='scripts/eval/configs/h1_internvla_n1_async_cfg_orig.py')
    parser.add_argument('--r2r_dir',  default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r')
    parser.add_argument('--save_dir', default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r_h1_replay')
    parser.add_argument('--scenes',   nargs='*', default=None)
    parser.add_argument('--ds_tmp',   default='/tmp/r2r_h1_replay_ds')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--max_eps',  type=int, default=None, help='scene당 최대 episode 수 (테스트용)')
    args = parser.parse_args()

    # 1. r2r 로드 (모든 row position/orientation)
    print('=== Step 1: Load r2r episodes ===')
    r2r_data = load_r2r_episodes(args.r2r_dir, target_scenes=args.scenes, max_eps=args.max_eps)
    total_eps  = sum(len(v) for v in r2r_data.values())
    total_rows = sum(len(poses) for scene_eps in r2r_data.values()
                     for _, poses, _, _ in scene_eps)
    print(f'Total: {len(r2r_data)} scenes, {total_eps} episodes, {total_rows} rows\n')

    # 2. synthetic dataset (row 하나 = episode 하나)
    print('=== Step 2: Create synthetic dataset ===')
    mapping = create_synthetic_dataset(r2r_data, args.ds_tmp)
    print()

    # 3. EvalCfg 로드
    spec = importlib.util.spec_from_file_location('eval_config_module', args.config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    eval_cfg = module.eval_cfg

    eval_cfg.dataset.dataset_settings['base_data_dir']    = args.ds_tmp
    eval_cfg.dataset.dataset_settings['split_data_types'] = ['train']
    eval_cfg.dataset.dataset_settings['filter_stairs']    = False

    from internnav.configs.evaluator.vln_default_config import get_config
    eval_cfg.task.task_name = f'h1_replay_{eval_cfg.task.task_name}'
    eval_cfg = get_config(eval_cfg)
    eval_cfg.env.env_settings['rank']       = 0
    eval_cfg.env.env_settings['local_rank'] = 0
    eval_cfg.env.env_settings['world_size'] = 1
    eval_cfg.env.env_settings['dataset']    = eval_cfg.dataset

    # fall/stuck 종료 비활성화, warm-up 1 step (position teleport 후 즉시 capture)
    eval_cfg.task.task_settings['check_fall_and_stuck'] = False
    eval_cfg.task.task_settings['warm_up_step'] = 1

    robot_name = eval_cfg.task.robot_name

    # 4. env 초기화
    print('=== Step 3: Initialize env ===')
    from internnav.env import Env
    env = Env.init(eval_cfg.env, eval_cfg.task)
    os.makedirs(args.save_dir, exist_ok=True)

    # image buffers: {(scene, ep_idx): [frame_dict, ...]}  indexed by row_idx
    # ResumablePathKeyEpisodeloader reverses episode order, so we iterate reversed(mapping)
    ep_n_rows = {}
    for scene, ep_idx, row_idx in mapping:
        key = (scene, ep_idx)
        ep_n_rows[key] = max(ep_n_rows.get(key, 0), row_idx + 1)
    image_buffers = {key: [None] * n for key, n in ep_n_rows.items()}

    print(f'\n=== Step 4: Position-based capture ({len(mapping)} rows) ===')
    obs_list, reset_info_list = env.reset()

    done, skipped, errors = 0, 0, 0
    skipped_eps = set()
    seen_eps    = set()

    for i, (scene, ep_idx, row_idx) in enumerate(reversed(mapping)):
        info = reset_info_list[0] if reset_info_list else None
        if info is None:
            print('[WARN] env episodes exhausted early')
            break

        # episode 단위 skip 판정 (해당 ep 첫 등장 시 확인)
        if (scene, ep_idx) not in seen_eps:
            seen_eps.add((scene, ep_idx))
            if args.skip_existing:
                rgb_path = (Path(args.save_dir) / scene / 'videos' / 'chunk-000'
                            / 'observation.images.rgb' / f'episode_{ep_idx:06d}.npy')
                if rgb_path.exists():
                    print(f'  SKIP {scene} ep{ep_idx}')
                    skipped += 1
                    skipped_eps.add((scene, ep_idx))

        if (scene, ep_idx) in skipped_eps:
            obs_list, reset_info_list = env.reset([0])
            continue

        try:
            # stand_still 1 step → warm-up 완료 → RGB/depth 캡처
            obs_list2, _, terminated, _, _ = env.step([{robot_name: {'stand_still': []}}])
            obs   = obs_list2[0][robot_name]
            rgb   = obs.get('rgb')
            depth = obs.get('depth')
            image_buffers[(scene, ep_idx)][row_idx] = {'rgb': rgb, 'depth': depth}
            done += 1
        except Exception as e:
            print(f'  ERROR {scene} ep{ep_idx} row{row_idx}: {e}')
            image_buffers[(scene, ep_idx)][row_idx] = {'rgb': None, 'depth': None}
            errors += 1

        obs_list, reset_info_list = env.reset([0])

    # 5. 이미지/parquet/meta 저장
    print('\n=== Step 5: Save images & copy parquet/meta ===')
    saved_eps = set()
    for (scene, ep_idx) in sorted(image_buffers.keys()):
        frames = image_buffers[(scene, ep_idx)]
        n = save_images(args.save_dir, scene, ep_idx, frames)
        copy_parquet(args.r2r_dir, args.save_dir, scene, ep_idx)
        print(f'  {scene} ep{ep_idx}: {n} frames')
        saved_eps.add(scene)

    for scene in saved_eps:
        copy_meta(args.r2r_dir, args.save_dir, scene)
        print(f'  {scene}: meta copied')

    print(f'\n=== Summary ===')
    print(f'  Rows captured : {done}')
    print(f'  Episodes skipped: {skipped}')
    print(f'  Errors        : {errors}')
    print(f'  Output        : {args.save_dir}')

    env.close()


if __name__ == '__main__':
    main()
