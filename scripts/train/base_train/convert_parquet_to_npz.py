"""
Convert episode parquet files to NPZ for fast numpy loading.

For each episode_*.parquet, extracts camera_intrinsic, camera_extrinsic,
and trajectory arrays and saves them as npz_cache/episode_*.npz alongside
the original parquet.

The extraction logic is identical to NavDP_Base_Datset.process_data_parquet:
    camera_intrinsic : (3, 3) float64
    camera_extrinsic : (4, 4) float64
    trajectory       : (T, 4, 4) float64

Original .parquet files are NOT deleted.

Usage:
    python scripts/train/base_train/convert_parquet_to_npz.py \
        --root-dir /ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_n1/traj_data

    # overwrite existing .npz files:
    python scripts/train/base_train/convert_parquet_to_npz.py \
        --root-dir ... --overwrite

    # verify .npz against .parquet without writing:
    python scripts/train/base_train/convert_parquet_to_npz.py \
        --root-dir ... --verify-only
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_parquet(pq_path: str) -> tuple:
    """Identical extraction to NavDP_Base_Datset.process_data_parquet."""
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
    return camera_intrinsic, camera_extrinsic, trajectory


def npz_path_for(pq_path: str) -> str:
    cache_dir = os.path.join(os.path.dirname(pq_path), 'npz_cache')
    return os.path.join(cache_dir, os.path.basename(pq_path).replace('.parquet', '.npz'))


parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', required=True, help='Root trajectory data directory')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing .npz files')
parser.add_argument('--verify-only', action='store_true', help='Verify existing .npz against .parquet without writing')
args = parser.parse_args()

parquet_paths = []
for dirpath, _, filenames in os.walk(args.root_dir):
    for fname in filenames:
        if fname.endswith('.parquet'):
            parquet_paths.append(os.path.join(dirpath, fname))
parquet_paths.sort()

print(f"Found {len(parquet_paths)} parquet files under {args.root_dir}\n")

skipped = converted = failed = verified_ok = verified_fail = 0

for pq_path in tqdm(parquet_paths, unit='file'):
    npz_path = npz_path_for(pq_path)

    if args.verify_only:
        if not os.path.isfile(npz_path):
            print(f"\nMISSING npz: {npz_path}", file=sys.stderr)
            verified_fail += 1
            continue
        try:
            intr_pq, extr_pq, traj_pq = extract_parquet(pq_path)
            data = np.load(npz_path)
            ok = (
                np.array_equal(intr_pq, data['camera_intrinsic']) and
                np.array_equal(extr_pq, data['camera_extrinsic']) and
                np.array_equal(traj_pq, data['trajectory'])
            )
            if ok:
                verified_ok += 1
            else:
                print(f"\nMISMATCH: {pq_path}", file=sys.stderr)
                verified_fail += 1
        except Exception as e:
            print(f"\nFAILED verify: {pq_path} — {e}", file=sys.stderr)
            verified_fail += 1
        continue

    if os.path.isfile(npz_path) and not args.overwrite:
        skipped += 1
        continue

    try:
        camera_intrinsic, camera_extrinsic, trajectory = extract_parquet(pq_path)
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
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

if args.verify_only:
    print(f"\nVerification done.  ok={verified_ok}  fail={verified_fail}")
else:
    print(f"\nDone.  converted={converted}  skipped={skipped}  failed={failed}")
