"""
Verify WebDataset shard data matches the original pipeline.

Compares per-episode processed arrays from the shard directory against
values re-derived from the original data directory (data-vol1 extracted).

Usage:
    # Quick: 20 random episodes
    python scripts/train/base_train/verify_webdataset.py \\
        --ref-dir ~/data-vol1/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i \\
        --shard-dir ~/data-vol1/InternData-N1-v0.5-mini/vln_n1/webdataset_shards/matterport3d_d435i \\
        --n-episodes 20

    # Full: all episodes
    python scripts/train/base_train/verify_webdataset.py \\
        --ref-dir ... --shard-dir ... --all

    # Single scene
    python scripts/train/base_train/verify_webdataset.py \\
        --ref-dir ... --shard-dir ... --scene 17DRP5sb8fy --all
"""
import argparse
import io
import json
import os
import random
import sys

import cv2
import jsonlines
import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ---- reference loading (identical to dataset fallback paths) ----

def ref_process_image(path: str, image_size: int) -> np.ndarray:
    image = np.array(Image.open(path), np.uint8)
    H, W, C = image.shape
    prop = image_size / max(H, W)
    image = cv2.resize(image, (-1, -1), fx=prop, fy=prop)
    pad_width = max((image_size - image.shape[1]) // 2, 0)
    pad_height = max((image_size - image.shape[0]) // 2, 0)
    pad_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                       mode='constant', constant_values=0)
    image = cv2.resize(pad_image, (image_size, image_size))
    return np.array(image, np.float32) / 255.0


def ref_process_depth(path: str, image_size: int) -> np.ndarray:
    depth = np.array(Image.open(path), np.uint16) / 10000.0
    H, W = depth.shape
    prop = image_size / max(H, W)
    depth = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
    pad_width = max((image_size - depth.shape[1]) // 2, 0)
    pad_height = max((image_size - depth.shape[0]) // 2, 0)
    pad_depth = np.pad(depth, ((pad_height, pad_height), (pad_width, pad_width)),
                       mode='constant', constant_values=0)
    pad_depth[pad_depth > 5.0] = 0
    pad_depth[pad_depth < 0.1] = 0
    depth = cv2.resize(pad_depth, (image_size, image_size))
    return np.array(depth, np.float32)[:, :, np.newaxis]


def ref_load_parquet(path: str):
    df = pd.read_parquet(path)
    intrinsic = np.vstack(np.array(df['observation.camera_intrinsic'].tolist()[0])).reshape(3, 3)
    extrinsic = np.vstack(np.array(df['observation.camera_extrinsic'].tolist()[0])).reshape(4, 4)
    traj_len = len(df['action'].tolist())
    trajectory = np.array([np.stack(frame) for frame in df['action']], dtype=np.float64).reshape(-1, 4, 4)
    return intrinsic, extrinsic, trajectory, traj_len


def ref_load_obstacle(ply_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_path)
    scene_color = np.array(pcd.colors)
    scene_points = np.array(pcd.points)
    color_distance = np.abs(scene_color - np.array([0, 0, 0.5])).sum(axis=-1)
    return scene_points[color_distance < 0.05].astype(np.float32)


# ---- verification helpers ----

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'


def check(name, ref, got, rtol=1e-5, atol=1e-5):
    if not np.allclose(ref, got, rtol=rtol, atol=atol, equal_nan=True):
        max_diff = np.abs(ref.astype(np.float64) - got.astype(np.float64)).max()
        print(f'  {FAIL} {name}: max_diff={max_diff:.6e}  shape ref={ref.shape} got={got.shape}')
        return False
    print(f'  {PASS} {name}')
    return True


def verify_episode(ref_dir: str, shard_dir: str, scene: str, ep_idx: int, image_size: int) -> bool:
    """Return True if all checks pass."""
    ok = True
    ep_str = f'{ep_idx:06d}'

    # ---- load shard data ----
    scene_shard = os.path.join(shard_dir, scene)
    rgb_shard = np.load(os.path.join(scene_shard, f'{ep_str}.rgb.npy'))
    depth_shard = np.load(os.path.join(scene_shard, f'{ep_str}.depth.npy'))
    meta_shard = np.load(os.path.join(scene_shard, f'{ep_str}.meta.npz'))
    obstacle_shard = np.load(os.path.join(shard_dir, f'{scene}_obstacle.npy'))

    intrinsic_s = meta_shard['intrinsic']
    extrinsic_s = meta_shard['extrinsic']
    trajectory_s = meta_shard['trajectory']

    # ---- load reference data ----
    # Parquet
    chunk_name = os.listdir(os.path.join(ref_dir, scene, 'data'))[0]
    parquet_path = os.path.join(ref_dir, scene, 'data', chunk_name, f'episode_{ep_str}.parquet')
    intrinsic_r, extrinsic_r, trajectory_r, traj_len = ref_load_parquet(parquet_path)

    # Obstacle
    ply_path = os.path.join(ref_dir, scene, 'meta', 'pointcloud.ply')
    obstacle_r = ref_load_obstacle(ply_path)

    # RGB / Depth paths
    rgb_dir = os.path.join(ref_dir, scene, 'videos', chunk_name, 'observation.images.rgb')
    depth_dir = os.path.join(ref_dir, scene, 'videos', chunk_name, 'observation.images.depth')

    ep_rgb_files = sorted(
        [p for p in os.listdir(rgb_dir) if p.startswith(f'episode_{ep_str}_')],
        key=lambda x: int(x.split('_')[-1].split('.')[0]),
    )
    ep_depth_files = sorted(
        [p for p in os.listdir(depth_dir) if p.startswith(f'episode_{ep_str}_')],
        key=lambda x: int(x.split('_')[-1].split('.')[0]),
    )

    # ---- comparisons ----
    ok &= check('intrinsic', intrinsic_r, intrinsic_s)
    ok &= check('extrinsic', extrinsic_r, extrinsic_s)
    ok &= check('trajectory', trajectory_r, trajectory_s)
    ok &= check('obstacle', obstacle_r, obstacle_shard)

    if len(ep_rgb_files) != rgb_shard.shape[0]:
        print(f'  {FAIL} rgb frame count: ref={len(ep_rgb_files)} shard={rgb_shard.shape[0]}')
        ok = False
    else:
        print(f'  {PASS} rgb frame count={len(ep_rgb_files)}')

        # Check all frames
        rgb_fail = 0
        depth_fail = 0
        for i, (rgb_fname, depth_fname) in enumerate(zip(ep_rgb_files, ep_depth_files)):
            rgb_ref = ref_process_image(os.path.join(rgb_dir, rgb_fname), image_size)
            depth_ref = ref_process_depth(os.path.join(depth_dir, depth_fname), image_size)
            if not np.allclose(rgb_ref, rgb_shard[i], rtol=1e-5, atol=1e-5):
                rgb_fail += 1
            if not np.allclose(depth_ref, depth_shard[i], rtol=1e-5, atol=1e-5):
                depth_fail += 1

        if rgb_fail == 0:
            print(f'  {PASS} all {len(ep_rgb_files)} rgb frames match')
        else:
            print(f'  {FAIL} {rgb_fail}/{len(ep_rgb_files)} rgb frames differ')
            ok = False

        if depth_fail == 0:
            print(f'  {PASS} all {len(ep_depth_files)} depth frames match')
        else:
            print(f'  {FAIL} {depth_fail}/{len(ep_depth_files)} depth frames differ')
            ok = False

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref-dir', required=True, help='Reference extracted data (data-vol1)')
    parser.add_argument('--shard-dir', required=True, help='WebDataset shard directory')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--n-episodes', type=int, default=20, help='Number of random episodes to check')
    parser.add_argument('--all', action='store_true', help='Check all episodes (slow)')
    parser.add_argument('--scene', help='Restrict to one scene')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    index_path = os.path.join(args.shard_dir, 'shard_index.json')
    with open(index_path) as f:
        shard_index = json.load(f)

    # Build candidate list
    candidates = []
    for scene, episodes in shard_index.items():
        if args.scene and scene != args.scene:
            continue
        for ep_info in episodes:
            ep_idx = int(ep_info[0])
            if ep_info[2] >= 0:  # skip episodes that were skipped during convert (no orig_H)
                candidates.append((scene, ep_idx))

    if not args.all:
        n = min(args.n_episodes, len(candidates))
        candidates = random.sample(candidates, n)

    print(f'Verifying {len(candidates)} episodes ...')

    passed = failed = 0
    for scene, ep_idx in tqdm(candidates, desc='Verifying'):
        print(f'\n[{scene} ep={ep_idx}]')
        try:
            ok = verify_episode(args.ref_dir, args.shard_dir, scene, ep_idx, args.image_size)
        except Exception as e:
            import traceback
            print(f'  ERROR: {e}')
            traceback.print_exc()
            ok = False

        if ok:
            passed += 1
        else:
            failed += 1

    print(f'\n{"="*50}')
    print(f'Result: {passed} passed, {failed} failed')
    print('="*50')

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
