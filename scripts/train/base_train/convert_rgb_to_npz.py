"""
Pre-process RGB JPEG images to float16 NPZ for fast loading.

Applies the same resize+pad+normalize as process_image() and saves the
result as float16 so __getitem__ can skip all processing with np.load().

Output: observation.images.rgb/npz_cache/<stem>.npz  key='img'  shape=(224,224,3) float16

Original .jpg files are NOT deleted.

Usage:
    python scripts/train/base_train/convert_rgb_to_npz.py \
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

jpg_paths = []
for dirpath, dirnames, filenames in os.walk(args.root_dir):
    # only process observation.images.rgb directories
    if not dirpath.endswith('observation.images.rgb'):
        continue
    for fname in filenames:
        if fname.endswith('.jpg') or fname.endswith('.jpeg'):
            jpg_paths.append(os.path.join(dirpath, fname))
jpg_paths.sort()

print(f"Found {len(jpg_paths)} JPEG files under {args.root_dir}\n")

skipped = converted = failed = 0
image_size = args.image_size

for jpg_path in tqdm(jpg_paths, unit='file'):
    cache_dir = os.path.join(os.path.dirname(jpg_path), 'npz_cache')
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, os.path.splitext(os.path.basename(jpg_path))[0] + '.npz')

    if os.path.isfile(npz_path) and not args.overwrite:
        skipped += 1
        continue

    try:
        image = np.array(Image.open(jpg_path), np.uint8)
        H, W, C = image.shape
        prop = image_size / max(H, W)
        image = cv2.resize(image, (-1, -1), fx=prop, fy=prop)
        pad_width = max((image_size - image.shape[1]) // 2, 0)
        pad_height = max((image_size - image.shape[0]) // 2, 0)
        pad_image = np.pad(
            image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
            mode='constant', constant_values=0
        )
        image = cv2.resize(pad_image, (image_size, image_size))
        img_f16 = (np.array(image, np.float32) / 255.0).astype(np.float16)
        np.savez_compressed(npz_path, img=img_f16)
        converted += 1
    except Exception as e:
        print(f"\nFAILED: {jpg_path} — {e}", file=sys.stderr)
        failed += 1

print(f"\nDone.  converted={converted}  skipped={skipped}  failed={failed}")
