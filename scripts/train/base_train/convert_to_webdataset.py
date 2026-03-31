"""
Convert per-scene tar.gz archives to episode-packed WebDataset shard format.

Source (tar.gz layout inside each {scene}.tar.gz):
    {scene}/data/chunk-000/episode_{ep:06d}.parquet
    {scene}/meta/pointcloud.ply
    {scene}/meta/episodes_stats.jsonl
    {scene}/videos/chunk-000/observation.images.rgb/episode_{ep:06d}_{frame:03d}.jpg
    {scene}/videos/chunk-000/observation.images.depth/episode_{ep:06d}_{frame:03d}.png

Output layout:
    {out_dir}/
        shard_index.json
        {scene}/
            {ep_idx:06d}.rgb.npy    (N, 224, 224, 3) float32
            {ep_idx:06d}.depth.npy  (N, 224, 224, 1) float32
            {ep_idx:06d}.meta.npz   intrinsic, extrinsic, trajectory, orig_H, orig_W
        {scene}_obstacle.npy        (M, 3) float32

Usage:
    python scripts/train/base_train/convert_to_webdataset.py \\
        --src-dir ~/data-vol1/InternData-N1-v0.5-mini_backup/vln_n1/traj_data/matterport3d_d435i \\
        --out-dir ~/data-vol1/InternData-N1-v0.5-mini/vln_n1/webdataset_shards/matterport3d_d435i \\
        --workers 8

    # Single scene test
    python scripts/train/base_train/convert_to_webdataset.py \\
        --src-dir ... --out-dir ... --scenes 17DRP5sb8fy --workers 1
"""
import argparse
import io
import json
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import jsonlines
import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image
from tqdm import tqdm

# -----------------------------------------------------------------------
# Processing functions — identical logic to NavDP_Base_Datset methods
# -----------------------------------------------------------------------

IMAGE_SIZE = 224  # will be overridden by --image-size arg


def process_image(img_bytes: bytes, image_size: int) -> np.ndarray:
    """Same as NavDP_Base_Datset.process_image() but takes raw bytes."""
    image = np.array(Image.open(io.BytesIO(img_bytes)), np.uint8)
    orig_H, orig_W = image.shape[:2]
    H, W, C = image.shape
    prop = image_size / max(H, W)
    image = cv2.resize(image, (-1, -1), fx=prop, fy=prop)
    pad_width = max((image_size - image.shape[1]) // 2, 0)
    pad_height = max((image_size - image.shape[0]) // 2, 0)
    pad_image = np.pad(
        image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
        mode='constant', constant_values=0,
    )
    image = cv2.resize(pad_image, (image_size, image_size))
    return np.array(image, np.float32) / 255.0, orig_H, orig_W  # float32 (H,W,3)


def process_depth(depth_bytes: bytes, image_size: int) -> np.ndarray:
    """Same as NavDP_Base_Datset.process_depth() but takes raw bytes."""
    depth = np.array(Image.open(io.BytesIO(depth_bytes)), np.uint16) / 10000.0
    H, W = depth.shape
    prop = image_size / max(H, W)
    depth = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
    pad_width = max((image_size - depth.shape[1]) // 2, 0)
    pad_height = max((image_size - depth.shape[0]) // 2, 0)
    pad_depth = np.pad(
        depth, ((pad_height, pad_height), (pad_width, pad_width)),
        mode='constant', constant_values=0,
    )
    pad_depth[pad_depth > 5.0] = 0
    pad_depth[pad_depth < 0.1] = 0
    depth = cv2.resize(pad_depth, (image_size, image_size))
    return np.array(depth, np.float32)[:, :, np.newaxis]  # float32 (H,W,1)


def load_parquet(parquet_bytes: bytes):
    """Same as NavDP_Base_Datset.process_data_parquet() fallback path."""
    df = pd.read_parquet(io.BytesIO(parquet_bytes))
    camera_intrinsic = np.vstack(np.array(df['observation.camera_intrinsic'].tolist()[0])).reshape(3, 3)
    camera_extrinsic = np.vstack(np.array(df['observation.camera_extrinsic'].tolist()[0])).reshape(4, 4)
    trajectory_length = len(df['action'].tolist())
    camera_trajectory = np.array(
        [np.stack(frame) for frame in df['action']], dtype=np.float64
    ).reshape(-1, 4, 4)
    return camera_intrinsic, camera_extrinsic, camera_trajectory, trajectory_length


def load_obstacle_points(ply_bytes: bytes) -> np.ndarray:
    """Same as NavDP_Base_Datset.process_obstacle_points() fallback path."""
    tmp_path = f'/tmp/_convert_ply_{os.getpid()}.ply'
    with open(tmp_path, 'wb') as f:
        f.write(ply_bytes)
    pcd = o3d.io.read_point_cloud(tmp_path)
    os.unlink(tmp_path)
    scene_color = np.array(pcd.colors)
    scene_points = np.array(pcd.points)
    color_distance = np.abs(scene_color - np.array([0, 0, 0.5])).sum(axis=-1)
    return scene_points[color_distance < 0.05].astype(np.float32)


# -----------------------------------------------------------------------
# Per-scene conversion
# -----------------------------------------------------------------------

def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def convert_scene(scene_tgz: str, out_dir: str, image_size: int, overwrite: bool) -> dict:
    """
    Convert one scene tar.gz to per-episode npy files in out_dir/{scene}/.
    Returns summary dict.
    """
    scene = Path(scene_tgz).stem.replace('.tar', '')  # handles .tar.gz → stem = scene
    scene_out = os.path.join(out_dir, scene)
    os.makedirs(scene_out, exist_ok=True)

    obstacle_out = os.path.join(out_dir, f'{scene}_obstacle.npy')

    result = {'scene': scene, 'episodes': [], 'skipped': 0, 'failed': 0}

    with tarfile.open(scene_tgz, 'r:gz') as tf:
        all_names = tf.getnames()

        # ---- obstacle ----
        if not os.path.isfile(obstacle_out) or overwrite:
            ply_name = f'{scene}/meta/pointcloud.ply'
            try:
                ply_bytes = tf.extractfile(ply_name).read()
                obstacle = load_obstacle_points(ply_bytes)
                np.save(obstacle_out, obstacle)
            except Exception as e:
                print(f'[{scene}] obstacle FAILED: {e}', file=sys.stderr)
                result['failed'] += 1

        # ---- episodes_stats ----
        stats_name = f'{scene}/meta/episodes_stats.jsonl'
        stats_bytes = tf.extractfile(stats_name).read()
        with jsonlines.Reader(io.StringIO(stats_bytes.decode())) as reader:
            episode_stats = list(reader)

        # chunk name (assume chunk-000)
        chunk = 'chunk-000'

        # build maps: rgb / depth / parquet by episode_idx
        rgb_map = {}   # ep_idx → sorted list of (frame_idx, member_name)
        depth_map = {}
        rgb_prefix = f'{scene}/videos/{chunk}/observation.images.rgb/'
        depth_prefix = f'{scene}/videos/{chunk}/observation.images.depth/'
        data_prefix = f'{scene}/data/{chunk}/'

        for name in all_names:
            if name.startswith(rgb_prefix) and name.endswith('.jpg'):
                fname = os.path.basename(name)
                parts = fname.replace('.jpg', '').split('_')
                # episode_000000_000 → parts = ['episode', '000000', '000']
                ep = int(parts[1])
                frame = int(parts[2])
                rgb_map.setdefault(ep, []).append((frame, name))
            elif name.startswith(depth_prefix) and name.endswith('.png'):
                fname = os.path.basename(name)
                parts = fname.replace('.png', '').split('_')
                ep = int(parts[1])
                frame = int(parts[2])
                depth_map.setdefault(ep, []).append((frame, name))

        # ---- process each episode ----
        for ep_stat in episode_stats:
            ep_idx = ep_stat['episode_index']
            ep_str = f'{ep_idx:06d}'

            rgb_out = os.path.join(scene_out, f'{ep_str}.rgb.npy')
            depth_out = os.path.join(scene_out, f'{ep_str}.depth.npy')
            meta_out = os.path.join(scene_out, f'{ep_str}.meta.npz')

            if all(os.path.isfile(p) for p in [rgb_out, depth_out, meta_out]) and not overwrite:
                result['skipped'] += 1
                frame_count = ep_stat['image_index']['count']
                result['episodes'].append([ep_idx, frame_count, -1, -1])
                continue

            try:
                # parquet
                parquet_name = f'{data_prefix}episode_{ep_str}.parquet'
                parquet_bytes = tf.extractfile(parquet_name).read()
                intrinsic, extrinsic, trajectory, traj_len = load_parquet(parquet_bytes)

                # rgb frames
                ep_rgb_list = sorted(rgb_map.get(ep_idx, []), key=lambda x: x[0])
                ep_depth_list = sorted(depth_map.get(ep_idx, []), key=lambda x: x[0])

                n_frames = len(ep_rgb_list)
                orig_H, orig_W = -1, -1

                rgb_arr = np.zeros((n_frames, image_size, image_size, 3), np.float32)
                for i, (frame_idx, member_name) in enumerate(ep_rgb_list):
                    img_bytes = tf.extractfile(member_name).read()
                    processed, h, w = process_image(img_bytes, image_size)
                    rgb_arr[i] = processed
                    if orig_H < 0:
                        orig_H, orig_W = h, w

                depth_arr = np.zeros((n_frames, image_size, image_size, 1), np.float32)
                for i, (frame_idx, member_name) in enumerate(ep_depth_list):
                    dep_bytes = tf.extractfile(member_name).read()
                    depth_arr[i] = process_depth(dep_bytes, image_size)

                # save
                np.save(rgb_out, rgb_arr)
                np.save(depth_out, depth_arr)
                buf = io.BytesIO()
                np.savez(
                    buf,
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    trajectory=trajectory,
                    orig_H=np.array(orig_H),
                    orig_W=np.array(orig_W),
                )
                with open(meta_out, 'wb') as f:
                    f.write(buf.getvalue())

                result['episodes'].append([ep_idx, n_frames, orig_H, orig_W])

            except Exception as e:
                print(f'[{scene}] ep={ep_idx} FAILED: {e}', file=sys.stderr)
                import traceback; traceback.print_exc(file=sys.stderr)
                result['failed'] += 1

    return result


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', required=True, help='Directory containing {scene}.tar.gz files')
    parser.add_argument('--out-dir', required=True, help='Output shard directory')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--scenes', nargs='*', help='Process only these scenes (optional filter)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tgz_files = sorted(Path(args.src_dir).glob('*.tar.gz'))
    if args.scenes:
        tgz_files = [t for t in tgz_files if t.name.replace('.tar.gz', '') in args.scenes]

    print(f'Found {len(tgz_files)} scene tar.gz files')
    print(f'Output dir: {args.out_dir}')
    print(f'Workers: {args.workers}')

    shard_index = {}  # {scene: [[ep_idx, length, orig_H, orig_W], ...]}
    total_failed = 0

    if args.workers <= 1:
        for tgz in tqdm(tgz_files, desc='Scenes'):
            result = convert_scene(str(tgz), args.out_dir, args.image_size, args.overwrite)
            shard_index[result['scene']] = result['episodes']
            total_failed += result['failed']
            print(f"  {result['scene']}: episodes={len(result['episodes'])} skipped={result['skipped']} failed={result['failed']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(convert_scene, str(tgz), args.out_dir, args.image_size, args.overwrite): tgz
                for tgz in tgz_files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc='Scenes'):
                result = future.result()
                shard_index[result['scene']] = result['episodes']
                total_failed += result['failed']
                print(f"  {result['scene']}: episodes={len(result['episodes'])} skipped={result['skipped']} failed={result['failed']}")

    # Write shard index
    index_path = os.path.join(args.out_dir, 'shard_index.json')
    with open(index_path, 'w') as f:
        json.dump(shard_index, f, indent=2)
    print(f'\nShard index written to {index_path}')

    total_eps = sum(len(v) for v in shard_index.values())
    print(f'Total scenes: {len(shard_index)}  total episodes: {total_eps}  failed: {total_failed}')
    if total_failed > 0:
        print(f'WARNING: {total_failed} failures. Check stderr output.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
