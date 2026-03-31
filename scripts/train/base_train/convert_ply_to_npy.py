"""
Convert PLY point clouds to NPY — full point cloud, identical to original PLY.

Saves all points and colors as a single float64 array:
  pointcloud.npy  shape=(N, 6)  columns: [x, y, z, r, g, b]

Values are identical to np.array(pcd.points) and np.array(pcd.colors)
from open3d — no filtering, no transformation.

Verifies every converted file by reloading and comparing with np.array_equal.
Original .ply files are NOT deleted.

Usage:
    python scripts/train/base_train/convert_ply_to_npy.py \
        --root-dir /path/to/traj_data
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
from tqdm import tqdm

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
    npy_path = ply_path.replace('pointcloud.ply', 'pointcloud.npy')

    if os.path.isfile(npy_path) and not args.overwrite:
        skipped += 1
        continue

    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.array(pcd.points)          # (N, 3) float64
        colors = np.array(pcd.colors)          # (N, 3) float64, normalized 0-1
        data = np.concatenate([points, colors], axis=1)  # (N, 6) float64

        np.save(npy_path, data)

        # --- verify: reload must be bit-identical ---
        loaded = np.load(npy_path)
        if not np.array_equal(loaded, data):
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
