"""
Convert PLY point clouds to pre-filtered obstacle NPZ files.

For each meta/pointcloud.ply, extracts obstacle points (color ≈ [0,0,0.5])
and saves them as meta/pointcloud_obstacle.npz.

Original .ply files are NOT deleted.

Usage:
    python scripts/train/base_train/convert_ply_to_npz.py \
        --root-dir /ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_n1/traj_data
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', required=True, help='Root trajectory data directory')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing .npz files')
args = parser.parse_args()

ply_paths = []
for dirpath, _, filenames in os.walk(args.root_dir):
    for fname in filenames:
        if fname == 'pointcloud.ply':
            ply_paths.append(os.path.join(dirpath, fname))
ply_paths.sort()

print(f"Found {len(ply_paths)} PLY files under {args.root_dir}\n")

skipped = converted = failed = 0

for ply_path in tqdm(ply_paths, unit='file'):
    npz_path = ply_path.replace('pointcloud.ply', 'pointcloud_obstacle.npz')

    if os.path.isfile(npz_path) and not args.overwrite:
        skipped += 1
        continue

    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        scene_points = np.array(pcd.points)   # (N, 3) float64
        scene_colors = np.array(pcd.colors)   # (N, 3) float64, normalized 0-1

        color_dist = np.abs(scene_colors - np.array([0.0, 0.0, 0.5])).sum(axis=-1)
        obstacle_points = scene_points[color_dist < 0.05].astype(np.float64)

        np.savez_compressed(npz_path, obstacle_points=obstacle_points)
        converted += 1
    except Exception as e:
        print(f"\nFAILED: {ply_path} — {e}", file=sys.stderr)
        failed += 1

print(f"\nDone.  converted={converted}  skipped={skipped}  failed={failed}")
