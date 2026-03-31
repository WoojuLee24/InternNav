"""
Pre-process depth PNG images to float16 NPZ for fast loading.

Applies the same resize+pad+clamp as process_depth() and saves the
result as float16 so __getitem__ can skip all processing with np.load().

Output: observation.images.depth/npz_cache/<stem>.npz  key='depth'  shape=(224,224,1) float16

Original .png files are NOT deleted.

Usage:
    python scripts/train/base_train/convert_depth_to_npz.py \
        --root-dir /ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_n1/traj_data
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', required=True, help='Root trajectory data directory')
parser.add_argument('--image-size', type=int, default=224)
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing .npz files')
args = parser.parse_args()

png_paths = []
for dirpath, dirnames, filenames in os.walk(args.root_dir):
    if not dirpath.endswith('observation.images.depth'):
        continue
    for fname in filenames:
        if fname.endswith('.png'):
            png_paths.append(os.path.join(dirpath, fname))
png_paths.sort()

print(f"Found {len(png_paths)} PNG files under {args.root_dir}\n")

skipped = converted = failed = 0
image_size = args.image_size

for png_path in tqdm(png_paths, unit='file'):
    cache_dir = os.path.join(os.path.dirname(png_path), 'npz_cache')
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, os.path.splitext(os.path.basename(png_path))[0] + '.npz')

    if os.path.isfile(npz_path) and not args.overwrite:
        skipped += 1
        continue

    try:
        depth = np.array(Image.open(png_path), np.uint16).astype(np.float32) / 10000.0
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
        depth_f16 = np.array(depth, np.float32).astype(np.float16)[:, :, np.newaxis]
        np.savez_compressed(npz_path, depth=depth_f16)
        converted += 1
    except Exception as e:
        print(f"\nFAILED: {png_path} — {e}", file=sys.stderr)
        failed += 1

print(f"\nDone.  converted={converted}  skipped={skipped}  failed={failed}")
