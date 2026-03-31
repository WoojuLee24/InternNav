"""
Convert episode parquet files to NPY for fast numpy loading.

Saves three arrays per episode using the same dtype as process_data_parquet():
  npy_cache/<stem>_intrinsic.npy   shape=(3,3)
  npy_cache/<stem>_extrinsic.npy   shape=(4,4)
  npy_cache/<stem>_trajectory.npy  shape=(N,4,4) float64

Verify: reloads npy and compares against values re-derived from the original
parquet using the identical pipeline as process_data_parquet().

Original .parquet files are NOT deleted.

Usage:
    python scripts/train/base_train/convert_parquet_to_npy.py \
        --root-dir /path/to/traj_data
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_parquet(pq_path):
    """Identical to process_data_parquet() fallback path in dataset."""
    df = pd.read_parquet(pq_path)
    intrinsic = np.vstack(
        np.array(df['observation.camera_intrinsic'].tolist()[0])
    ).reshape(3, 3)
    extrinsic = np.vstack(
        np.array(df['observation.camera_extrinsic'].tolist()[0])
    ).reshape(4, 4)
    trajectory = np.array(
        [np.stack(frame) for frame in df['action']], dtype=np.float64
    ).reshape(-1, 4, 4)
    return intrinsic, extrinsic, trajectory


parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', required=True)
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

parquet_paths = []
for dirpath, _, filenames in os.walk(args.root_dir):
    for fname in filenames:
        if fname.endswith('.parquet'):
            parquet_paths.append(os.path.join(dirpath, fname))
parquet_paths.sort()
print(f"Found {len(parquet_paths)} parquet files under {args.root_dir}\n")

skipped = converted = failed = verify_failed = 0

for pq_path in tqdm(parquet_paths, unit='file'):
    cache_dir = os.path.join(os.path.dirname(pq_path), 'npy_cache')
    stem = os.path.splitext(os.path.basename(pq_path))[0]
    p_intr = os.path.join(cache_dir, stem + '_intrinsic.npy')
    p_extr = os.path.join(cache_dir, stem + '_extrinsic.npy')
    p_traj = os.path.join(cache_dir, stem + '_trajectory.npy')

    if all(os.path.isfile(p) for p in [p_intr, p_extr, p_traj]) and not args.overwrite:
        skipped += 1
        continue

    try:
        intrinsic, extrinsic, trajectory = load_parquet(pq_path)

        os.makedirs(cache_dir, exist_ok=True)
        np.save(p_intr, intrinsic)
        np.save(p_extr, extrinsic)
        np.save(p_traj, trajectory)

        # --- verify: reload npy and compare against original parquet values ---
        ref_intr, ref_extr, ref_traj = load_parquet(pq_path)
        ok = (
            np.array_equal(np.load(p_intr), ref_intr)
            and np.array_equal(np.load(p_extr), ref_extr)
            and np.array_equal(np.load(p_traj), ref_traj)
        )
        if not ok:
            print(f"\nVERIFY FAILED: {pq_path}", file=sys.stderr)
            verify_failed += 1
        else:
            converted += 1

    except Exception as e:
        print(f"\nFAILED: {pq_path} — {e}", file=sys.stderr)
        failed += 1

print(f"\nDone.  converted={converted}  skipped={skipped}  failed={failed}  verify_failed={verify_failed}")
if verify_failed or failed:
    sys.exit(1)
