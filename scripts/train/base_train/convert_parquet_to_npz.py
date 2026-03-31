"""
Convert episode parquet files to NPZ for fast numpy loading.

For each episode_*.parquet, extracts camera_intrinsic, camera_extrinsic,
and trajectory arrays and saves them as episode_*.npz alongside the original.

Original .parquet files are NOT deleted.

Usage:
    python scripts/train/base_train/convert_parquet_to_npz.py \
        --root-dir /ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_n1/traj_data
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', required=True, help='Root trajectory data directory')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing .npz files')
args = parser.parse_args()

parquet_paths = []
for dirpath, _, filenames in os.walk(args.root_dir):
    for fname in filenames:
        if fname.endswith('.parquet'):
            parquet_paths.append(os.path.join(dirpath, fname))
parquet_paths.sort()

print(f"Found {len(parquet_paths)} parquet files under {args.root_dir}\n")

skipped = converted = failed = 0

for pq_path in tqdm(parquet_paths, unit='file'):
    cache_dir = os.path.join(os.path.dirname(pq_path), 'npz_cache')
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, os.path.basename(pq_path).replace('.parquet', '.npz'))

    if os.path.isfile(npz_path) and not args.overwrite:
        skipped += 1
        continue

    try:
        df = pd.read_parquet(pq_path)

        camera_intrinsic = np.vstack(
            np.array(df['observation.camera_intrinsic'].tolist()[0])
        ).reshape(3, 3).astype(np.float64)

        camera_extrinsic = np.vstack(
            np.array(df['observation.camera_extrinsic'].tolist()[0])
        ).reshape(4, 4).astype(np.float64)

        trajectory = np.array(
            [np.stack(frame) for frame in df['action']], dtype=np.float64
        ).reshape(-1, 4, 4)

        np.savez_compressed(
            npz_path,
            camera_intrinsic=camera_intrinsic,
            camera_extrinsic=camera_extrinsic,
            trajectory=trajectory,
        )
        converted += 1
    except Exception as e:
        print(f"\nFAILED: {pq_path} — {e}", file=sys.stderr)
        failed += 1

print(f"\nDone.  converted={converted}  skipped={skipped}  failed={failed}")
