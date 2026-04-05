"""
Convert PLY point clouds to pre-filtered obstacle NPY files.

For each meta/pointcloud.ply, extracts obstacle points (color ≈ [0,0,0.5])
and saves them as meta/pointcloud_obstacle.npy  (float64, shape (N,3)).

The filtering logic is identical to NavDP_Base_Datset.process_obstacle_points:
    color_distance = |color - [0, 0, 0.5]|.sum(axis=-1) < 0.05

If meta/pointcloud_obstacle.npz exists, the output is verified against it.
Original .ply files are NOT deleted.

Usage:
    python scripts/train/base_train/convert_ply_to_npy.py \
        --root-dir /ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_n1/traj_data

    # overwrite existing .npy files:
    python scripts/train/base_train/convert_ply_to_npy.py \
        --root-dir ... --overwrite

    # verify .npy against .ply without writing:
    python scripts/train/base_train/convert_ply_to_npy.py \
        --root-dir ... --verify-only
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
from tqdm import tqdm


def extract_obstacle_points(ply_path: str) -> np.ndarray:
    """Identical filter to NavDP_Base_Datset.process_obstacle_points."""
    pcd = o3d.io.read_point_cloud(ply_path)
    scene_points = np.array(pcd.points)   # (N, 3) float64
    scene_colors = np.array(pcd.colors)   # (N, 3) float64, normalized 0-1
    color_dist = np.abs(scene_colors - np.array([0.0, 0.0, 0.5])).sum(axis=-1)
    return scene_points[color_dist < 0.05].astype(np.float64)


parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', required=True, help='Root trajectory data directory')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing .npy files')
parser.add_argument('--verify-only', action='store_true', help='Verify existing .npy against .ply without writing')
args = parser.parse_args()

ply_paths = []
for dirpath, _, filenames in os.walk(args.root_dir):
    for fname in filenames:
        if fname == 'pointcloud.ply':
            ply_paths.append(os.path.join(dirpath, fname))
ply_paths.sort()

print(f"Found {len(ply_paths)} PLY files under {args.root_dir}\n")

skipped = converted = failed = verified_ok = verified_fail = 0

for ply_path in tqdm(ply_paths, unit='file'):
    npy_path = ply_path.replace('pointcloud.ply', 'pointcloud_obstacle.npy')
    npz_path = ply_path.replace('pointcloud.ply', 'pointcloud_obstacle.npz')

    if args.verify_only:
        # Verify .npy matches .ply output
        if not os.path.isfile(npy_path):
            print(f"\nMISSING npy: {npy_path}", file=sys.stderr)
            verified_fail += 1
            continue
        try:
            from_ply = extract_obstacle_points(ply_path)
            from_npy = np.load(npy_path)
            if np.array_equal(from_ply, from_npy):
                verified_ok += 1
            else:
                print(f"\nMISMATCH: {ply_path}  ply={from_ply.shape}  npy={from_npy.shape}", file=sys.stderr)
                verified_fail += 1
        except Exception as e:
            print(f"\nFAILED verify: {ply_path} — {e}", file=sys.stderr)
            verified_fail += 1
        continue

    if os.path.isfile(npy_path) and not args.overwrite:
        skipped += 1
        continue

    try:
        obstacle_points = extract_obstacle_points(ply_path)

        # Cross-check against existing .npz if available
        if os.path.isfile(npz_path):
            from_npz = np.load(npz_path)['obstacle_points']
            if not np.array_equal(obstacle_points, from_npz):
                print(
                    f"\nWARN mismatch vs .npz: {ply_path}  "
                    f"ply={obstacle_points.shape}  npz={from_npz.shape}",
                    file=sys.stderr,
                )

        np.save(npy_path, obstacle_points)
        converted += 1
    except Exception as e:
        print(f"\nFAILED: {ply_path} — {e}", file=sys.stderr)
        failed += 1

if args.verify_only:
    print(f"\nVerification done.  ok={verified_ok}  fail={verified_fail}")
else:
    print(f"\nDone.  converted={converted}  skipped={skipped}  failed={failed}")
