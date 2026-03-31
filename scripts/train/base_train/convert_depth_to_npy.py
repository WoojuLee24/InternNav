"""
Pre-process depth PNG images to float32 NPY for fast loading.

Applies the same pipeline as process_depth() and saves as float32.
Output: observation.images.depth/npy_cache/<stem>.npy  shape=(224,224,1) float32

Verify: reloads npy and compares against values re-derived from the original
PNG using the identical pipeline as process_depth().

Original .png files are NOT deleted.

Usage:
    python scripts/train/base_train/convert_depth_to_npy.py \
        --root-dir /path/to/traj_data [--image-size 224]
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', required=True)
parser.add_argument('--image-size', type=int, default=224)
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
image_size = args.image_size


def process_depth(depth_path):
    """Identical to NavDP_Base_Datset.process_depth()."""
    depth = np.array(Image.open(depth_path), np.uint16) / 10000.0  # float64
    H, W = depth.shape
    prop = image_size / max(H, W)
    depth = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
    pad_width = max((image_size - depth.shape[1]) // 2, 0)
    pad_height = max((image_size - depth.shape[0]) // 2, 0)
    pad_depth = np.pad(
        depth, ((pad_height, pad_height), (pad_width, pad_width)),
        mode='constant', constant_values=0
    )
    pad_depth[pad_depth > 5.0] = 0
    pad_depth[pad_depth < 0.1] = 0
    depth = cv2.resize(pad_depth, (image_size, image_size))
    depth = np.array(depth, np.float32)
    return depth[:, :, np.newaxis]


png_paths = []
for dirpath, _, filenames in os.walk(args.root_dir):
    if not dirpath.endswith('observation.images.depth'):
        continue
    for fname in filenames:
        if fname.endswith('.png'):
            png_paths.append(os.path.join(dirpath, fname))
png_paths.sort()
print(f"Found {len(png_paths)} PNG files under {args.root_dir}\n")

skipped = converted = failed = verify_failed = 0

for png_path in tqdm(png_paths, unit='file'):
    cache_dir = os.path.join(os.path.dirname(png_path), 'npy_cache')
    stem = os.path.splitext(os.path.basename(png_path))[0]
    npy_path = os.path.join(cache_dir, stem + '.npy')

    if os.path.isfile(npy_path) and not args.overwrite:
        skipped += 1
        continue

    try:
        depth = process_depth(png_path)

        os.makedirs(cache_dir, exist_ok=True)
        np.save(npy_path, depth)

        # --- verify: reload npy and compare against original png values ---
        ref = process_depth(png_path)
        if not np.array_equal(np.load(npy_path), ref):
            print(f"\nVERIFY FAILED: {png_path}", file=sys.stderr)
            verify_failed += 1
        else:
            converted += 1

    except Exception as e:
        print(f"\nFAILED: {png_path} — {e}", file=sys.stderr)
        failed += 1

print(f"\nDone.  converted={converted}  skipped={skipped}  failed={failed}  verify_failed={verify_failed}")
if verify_failed or failed:
    sys.exit(1)
