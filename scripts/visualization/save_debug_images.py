"""
r2r 포맷 데이터셋의 observation.rgb / observation.depth 를
r2r_debug 디렉터리에 개별 JPG/PNG 로 저장한다.

출력 구조:
  r2r_debug/<dataset_name>/<scan>/episode_XXXXXX/
      rgb/frame_0000.jpg  ...
      depth/frame_0000.png  ...

Usage:
    python scripts/eval/save_debug_images.py \
        --src_dir data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r \
        [--out_dir data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r_debug] \
        [--scan s8pcmisQ38h] [--ep 0]
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont


def _get_font(size=13):
    for p in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _depth_to_rgb(depth_frame):
    """float32 depth (H,W) → uint8 RGB (H,W,3) with a simple warm colormap."""
    d = depth_frame.copy()
    valid = d > 0
    if valid.any():
        lo, hi = d[valid].min(), d[valid].max()
        d = np.clip((d - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    r = np.clip(255 * (1.5 * d - 0.2),              0, 255).astype(np.uint8)
    g = np.clip(255 * np.sin(np.pi * d),             0, 255).astype(np.uint8)
    b = np.clip(255 * (1.0 - 2.0 * np.abs(d - 0.25)), 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def save_episode(src_dir: Path, out_dir: Path, scan: str, ep: int):
    rgb_npy   = src_dir / scan / 'videos' / 'chunk-000' / 'observation.images.rgb'   / f'episode_{ep:06d}.npy'
    depth_npy = src_dir / scan / 'videos' / 'chunk-000' / 'observation.images.depth' / f'episode_{ep:06d}.npy'

    if not rgb_npy.exists():
        print(f'  [SKIP] rgb not found: {rgb_npy}')
        return

    rgb_arr   = np.load(rgb_npy)                                   # (T, H, W, 3) uint8
    depth_arr = np.load(depth_npy) if depth_npy.exists() else None # (T, H, W) float32

    T = len(rgb_arr)
    ep_dir = out_dir / scan / f'episode_{ep:06d}'
    rgb_dir   = ep_dir / 'rgb'
    depth_dir = ep_dir / 'depth'
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if depth_arr is not None:
        depth_dir.mkdir(parents=True, exist_ok=True)

    for fi in range(T):
        Image.fromarray(rgb_arr[fi]).save(rgb_dir / f'frame_{fi:04d}.jpg', quality=95)
        if depth_arr is not None:
            Image.fromarray(_depth_to_rgb(depth_arr[fi])).save(depth_dir / f'frame_{fi:04d}.png')

    print(f'  {scan}/ep{ep:06d}: {T} rgb frames saved to {ep_dir}')
    if depth_arr is not None:
        print(f'  {scan}/ep{ep:06d}: {T} depth frames saved to {depth_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True,
                        help='r2r 포맷 데이터셋 경로 (e.g. data/.../r2r 또는 r2r_h1_replay)')
    parser.add_argument('--out_dir', default=None,
                        help='출력 루트 (기본: <src_dir>/../r2r_debug/<dataset_name>)')
    parser.add_argument('--scan',    default=None, help='처리할 scan ID (미지정 시 전체)')
    parser.add_argument('--ep',      type=int, default=None, help='처리할 episode index (미지정 시 전체)')
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dataset_name = src_dir.name  # 'r2r' or 'r2r_h1_replay'

    if args.out_dir:
        out_root = Path(args.out_dir) / dataset_name
    else:
        out_root = src_dir.parent / 'r2r_debug' / dataset_name

    print(f'src : {src_dir}')
    print(f'out : {out_root}')

    scans = [args.scan] if args.scan else sorted(
        d for d in os.listdir(src_dir) if (src_dir / d).is_dir()
    )

    for scan in scans:
        data_dir = src_dir / scan / 'data' / 'chunk-000'
        if not data_dir.exists():
            continue

        if args.ep is not None:
            eps = [args.ep]
        else:
            eps = sorted(
                int(p.stem.split('_')[1])
                for p in data_dir.glob('episode_*.parquet')
            )

        for ep in eps:
            save_episode(src_dir, out_root, scan, ep)

    print('Done.')


if __name__ == '__main__':
    main()
