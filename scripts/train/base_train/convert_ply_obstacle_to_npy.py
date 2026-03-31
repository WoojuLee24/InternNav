"""
Convert PLY point clouds to pre-filtered obstacle NPY files.

Saves obstacle points using the same pipeline as process_obstacle_points():
  pointcloud_obstacle.npy  shape=(M, 3) float64

Verify: reloads npy and compares against values re-derived from the original
PLY using the identical pipeline as process_obstacle_points().

Original .ply files are NOT deleted.

Usage:
    python scripts/train/base_train/convert_ply_obstacle_to_npy.py \
        --root-dir /path/to/traj_data
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
from tqdm import tqdm


def load_obstacle_points(ply_path):
    """Identical to process_obstacle_points() fallback path in dataset."""
    pcd = o3d.io.read_point_cloud(ply_path)
    scene_color = np.array(pcd.colors)
    scene_points = np.array(pcd.points)
    color_distance = np.abs(scene_color - np.array([0, 0, 0.5])).sum(axis=-1)
    return scene_points[color_distance < 0.05]


parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', required=True)
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

ply_paths = []
for dirpath, _, filenames in os.walk(args.root_dir):
    for fname in filenames:
        if fname == 'pointcloud.ply':
            ply_paths.append(os.path.join(dirpath, fname))
ply_paths.sort()
print(f"Found {len(ply_paths)} PLY files under {args.root_dir}\n")

skipped = converted = failed = verify_failed = 0

for ply_path in tqdm(ply_paths, unit='file'):
    npy_path = ply_path.replace('pointcloud.ply', 'pointcloud_obstacle.npy')

    if os.path.isfile(npy_path) and not args.overwrite:
        skipped += 1
        continue

    try:
        obstacle_points = load_obstacle_points(ply_path)

        np.save(npy_path, obstacle_points)

        # --- verify: reload npy and compare against original ply values ---
        ref = load_obstacle_points(ply_path)
        if not np.array_equal(np.load(npy_path), ref):
            print(f"\nVERIFY FAILED: {ply_path}", file=sys.stderr)
            verify_failed += 1
        else:
            converted += 1

    except Exception as e:
        print(f"\nFAILED: {ply_path} — {e}", file=sys.stderr)
        failed += 1

print(f"\nDone.  converted={converted}  skipped={skipped}  failed={failed}  verify_failed={verify_failed}")
if verify_failed or failed:
    sys.exit(1)
