"""
Pre-process RGB JPEG images to float32 NPY for fast loading.

Applies the same pipeline as process_image() and saves as float32.
Output: observation.images.rgb/npy_cache/<stem>.npy  shape=(224,224,3) float32

Verify: reloads npy and compares against values re-derived from the original
JPEG using the identical pipeline as process_image().

Original .jpg files are NOT deleted.

Usage:
    python scripts/train/base_train/convert_rgb_to_npy.py \
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


def process_image(image_path):
    """Identical to NavDP_Base_Datset.process_image()."""
    image = np.array(Image.open(image_path), np.uint8)
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
    return np.array(image, np.float32) / 255.0


jpg_paths = []
for dirpath, _, filenames in os.walk(args.root_dir):
    if not dirpath.endswith('observation.images.rgb'):
        continue
    for fname in filenames:
        if fname.endswith('.jpg') or fname.endswith('.jpeg'):
            jpg_paths.append(os.path.join(dirpath, fname))
jpg_paths.sort()
print(f"Found {len(jpg_paths)} JPEG files under {args.root_dir}\n")

skipped = converted = failed = verify_failed = 0

for jpg_path in tqdm(jpg_paths, unit='file'):
    cache_dir = os.path.join(os.path.dirname(jpg_path), 'npy_cache')
    stem = os.path.splitext(os.path.basename(jpg_path))[0]
    npy_path = os.path.join(cache_dir, stem + '.npy')

    if os.path.isfile(npy_path) and not args.overwrite:
        skipped += 1
        continue

    try:
        img = process_image(jpg_path)

        os.makedirs(cache_dir, exist_ok=True)
        np.save(npy_path, img)

        # --- verify: reload npy and compare against original jpg values ---
        ref = process_image(jpg_path)
        if not np.array_equal(np.load(npy_path), ref):
            print(f"\nVERIFY FAILED: {jpg_path}", file=sys.stderr)
            verify_failed += 1
        else:
            converted += 1

    except Exception as e:
        print(f"\nFAILED: {jpg_path} — {e}", file=sys.stderr)
        failed += 1

print(f"\nDone.  converted={converted}  skipped={skipped}  failed={failed}  verify_failed={verify_failed}")
if verify_failed or failed:
    sys.exit(1)
