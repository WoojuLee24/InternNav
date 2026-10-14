"""P1-R: 릴리스 train action 을 그대로 재생해 렌더러를 검증한다 (action 생성기 오차는 배제).

각 episode: raw start 에 놓고 parquet action 을 sim.step 으로 재생하며
  - pose: 우리 센서 로컬 c2w vs 릴리스 `pose.<rig>` — 위치 오차(mm), 회전 오차(deg, rotvec)
  - RGB : 우리 렌더를 릴리스와 같은 JPEG 양자화 테이블로 인코딩한 뒤 디코딩한 픽셀 vs 릴리스 jpg 디코딩
          (MAE, PSNR, SSIM). 참고값으로 JPEG 잡음 바닥 |raw - jpeg(raw)| 도 같이 잰다.
  - depth: floor(m*1000) uint16 vs 릴리스 png — mm 완전일치 비율, 1 mm 초과 비율
md5(바이트) 일치는 불가능하다: 릴리스 chroma 양자화 테이블이 이 머신 인코더와 다르다 (탐색에서 확인).

출력: <out>/summary.json, <out>/per_episode.csv, <out>/viz/*.jpg (릴리스 | 우리 | |diff|x8), pose 오차 그래프.

    /usr/bin/python scripts/dataset_converters/instruction_labeling/render/p1r_replay_verify.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --scenes 17DRP5sb8fy,1LXtFkjw3qL,29hnd4uzFmX --gpu 0 --out logs/instruction_labeling/p1r
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from habitat_rig_sim import RigSim  # noqa: E402
from release_io import (load_cfg, match_raw, release_episode_data, release_episodes,  # noqa: E402
                        release_image_path, scene_glb)


def jpeg_roundtrip(rgb: np.ndarray, qtables, subsampling: int) -> np.ndarray:
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, 'JPEG', qtables=qtables, subsampling=subsampling)
    return np.asarray(Image.open(b).convert('RGB'))


def depth_to_mm(depth_m: np.ndarray, max_m: float) -> np.ndarray:
    return np.floor(np.clip(depth_m, 0.0, max_m) * 1000.0).astype(np.int64)


def pose_err(P: np.ndarray, G: np.ndarray):
    pos_mm = float(np.linalg.norm(P[:3, 3] - G[:3, 3]) * 1000.0)
    R = P[:3, :3].T @ G[:3, :3].astype(np.float64)
    U, _, Vt = np.linalg.svd(R)  # 릴리스는 float32 -> 정규직교화 후 각도
    rot_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(U @ Vt).as_rotvec())))
    return pos_mm, rot_deg


def compare_rgb(ours: np.ndarray, rel_path: Path, subsampling: int, ssim: bool) -> dict:
    rel_im = Image.open(rel_path)
    rel = np.asarray(rel_im.convert('RGB'))
    oq = jpeg_roundtrip(ours, rel_im.quantization, subsampling)
    d = {
        'mae': float(np.abs(oq.astype(np.int16) - rel).mean()),
        'psnr': float(cv2.PSNR(oq, rel)),
        'jpeg_floor_mae': float(np.abs(ours.astype(np.int16) - oq).mean()),
        'mean_shift': float(oq.astype(np.float64).mean() - rel.astype(np.float64).mean()),
    }
    if ssim:
        d['ssim'] = float(structural_similarity(oq, rel, channel_axis=2, data_range=255))
    return d, rel, oq


def save_viz(path: Path, rel: np.ndarray, oq: np.ndarray, title: str):
    diff = np.clip(np.abs(oq.astype(np.int16) - rel) * 8, 0, 255).astype(np.uint8)
    vis = np.concatenate([rel, oq, diff], 1)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    for x, t in ((10, 'release'), (650, 'ours (jpeg)'), (1290, '|diff| x8')):
        cv2.putText(vis, t, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, title, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def run_scene(cfg, scene, args, out: Path) -> list:
    rigs = list(cfg['camera']['rigs'])
    sub = int(cfg['image']['jpeg_subsampling'])
    dmax = float(cfg['camera']['depth_max_m'])
    eps = release_episodes(cfg, scene)
    if args.max_eps:
        eps = eps[:args.max_eps]
    sim = RigSim(cfg, scene_glb(cfg, scene), gpu=args.gpu)
    rows = []
    for k, ep in enumerate(eps):
        raw = match_raw(cfg, scene, ep['task'])
        if raw is None:
            rows.append({'scene': scene, 'episode_index': ep['episode_index'], 'status': 'unmatched'})
            continue
        d = release_episode_data(cfg, scene, ep['episode_index'], rigs)
        sim.set_start(raw['start_position'], raw['start_rotation'])
        acc = {r: {'pos': [], 'rot': [], 'mae': [], 'psnr': [], 'ssim': [], 'floor': [], 'shift': [],
                   'dexact': [], 'dgt1': []} for r in rigs}
        T = len(d['action'])
        for f in range(T):
            if f > 0:
                sim.step(d['action'][f])
            do_px = (f % args.pixel_every == 0) or f == T - 1
            obs = sim.observe() if do_px else None
            for r in rigs:
                pm, rd = pose_err(sim.local_pose_cv(r), d[f'pose.{r}'][f])
                acc[r]['pos'].append(pm)
                acc[r]['rot'].append(rd)
                if not do_px:
                    continue
                c, rel, oq = compare_rgb(obs[r][0], release_image_path(cfg, scene, 'rgb', r, ep['episode_index'], f),
                                         sub, ssim=True)
                acc[r]['mae'].append(c['mae'])
                acc[r]['psnr'].append(c['psnr'])
                acc[r]['ssim'].append(c['ssim'])
                acc[r]['floor'].append(c['jpeg_floor_mae'])
                acc[r]['shift'].append(c['mean_shift'])
                dep = cv2.imread(str(release_image_path(cfg, scene, 'depth', r, ep['episode_index'], f)), -1)
                dd = np.abs(depth_to_mm(obs[r][1], dmax) - dep.astype(np.int64))
                acc[r]['dexact'].append(float((dd == 0).mean()))
                acc[r]['dgt1'].append(float((dd > 1).mean()))
                if k < args.viz_eps and f in (0, T // 2, T - 1):
                    save_viz(out / 'viz' / f'{scene}_ep{ep["episode_index"]:06d}_{r}_f{f:03d}.jpg', rel, oq,
                             f'{scene} ep{ep["episode_index"]} {r} f{f} MAE {c["mae"]:.2f} PSNR {c["psnr"]:.1f}')
        row = {'scene': scene, 'episode_index': ep['episode_index'], 'raw_episode_id': raw['episode_id'],
               'trajectory_id': raw['trajectory_id'], 'frames': T, 'status': 'ok'}
        for r in rigs:
            a = acc[r]
            row.update({f'{r}.pos_mm_max': max(a['pos']), f'{r}.rot_deg_max': max(a['rot']),
                        f'{r}.rgb_mae': float(np.mean(a['mae'])), f'{r}.rgb_mae_max': max(a['mae']),
                        f'{r}.psnr_min': min(a['psnr']), f'{r}.ssim_mean': float(np.mean(a['ssim'])),
                        f'{r}.ssim_min': min(a['ssim']), f'{r}.jpeg_floor_mae': float(np.mean(a['floor'])),
                        f'{r}.mean_shift': float(np.mean(a['shift'])),
                        f'{r}.depth_exact': float(np.mean(a['dexact'])), f'{r}.depth_gt1mm': max(a['dgt1'])})
            row[f'{r}.pos_series'] = a['pos']
        rows.append(row)
        print(f'[{scene}] {k + 1}/{len(eps)} ep{ep["episode_index"]} T={T} '
              + ' '.join(f'{r}: pos {row[f"{r}.pos_mm_max"]:.2f}mm rot {row[f"{r}.rot_deg_max"]:.3f}° '
                         f'MAE {row[f"{r}.rgb_mae"]:.2f} SSIM {row[f"{r}.ssim_mean"]:.4f} '
                         f'depth= {row[f"{r}.depth_exact"]:.4f}' for r in rigs), flush=True)
    sim.close()
    return rows


def summarize(rows, rigs) -> dict:
    ok = [r for r in rows if r['status'] == 'ok']
    s = {'episodes': len(rows), 'matched': len(ok), 'unmatched': len(rows) - len(ok),
         'frames': int(sum(r['frames'] for r in ok))}
    for rig in rigs:
        def col(k):
            return np.array([r[f'{rig}.{k}'] for r in ok])
        pos = np.concatenate([r[f'{rig}.pos_series'] for r in ok])
        s[rig] = {
            'pos_mm': {'p50': float(np.median(pos)), 'p99': float(np.percentile(pos, 99)), 'max': float(pos.max())},
            'rot_deg_max': float(col('rot_deg_max').max()),
            'rgb_mae_mean': float(col('rgb_mae').mean()), 'rgb_mae_worst_ep': float(col('rgb_mae_max').max()),
            'psnr_min': float(col('psnr_min').min()),
            'ssim_mean': float(col('ssim_mean').mean()), 'ssim_min': float(col('ssim_min').min()),
            'jpeg_floor_mae_mean': float(col('jpeg_floor_mae').mean()),
            'mean_brightness_shift': float(col('mean_shift').mean()),
            'depth_exact_mean': float(col('depth_exact').mean()), 'depth_gt1mm_worst': float(col('depth_gt1mm').max()),
        }
    return s


def plot_pose(rows, rigs, path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(rigs), figsize=(6 * len(rigs), 4), squeeze=False)
    for ax, rig in zip(axes[0], rigs):
        for r in rows:
            if r['status'] == 'ok':
                ax.plot(r[f'{rig}.pos_series'], lw=0.5, alpha=0.5)
        ax.set_title(f'{rig}: position error per frame (mm)')
        ax.set_xlabel('frame')
        ax.set_ylabel('mm')
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--scenes', required=True, help='쉼표 구분')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max_eps', type=int, default=0, help='scene 당 episode 수 상한 (0=전부)')
    ap.add_argument('--pixel_every', type=int, default=5, help='픽셀 비교 프레임 간격 (마지막 프레임은 항상)')
    ap.add_argument('--viz_eps', type=int, default=2, help='scene 당 시각화할 episode 수')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    rigs = list(cfg['camera']['rigs'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    for scene in args.scenes.split(','):
        rows += run_scene(cfg, scene, args, out)
    s = summarize(rows, rigs)
    s['elapsed_s'] = round(time.time() - t0, 1)
    s['args'] = vars(args)
    (out / 'summary.json').write_text(json.dumps(s, indent=2))
    import pandas as pd
    pd.DataFrame([{k: v for k, v in r.items() if not k.endswith('pos_series')} for r in rows]).to_csv(
        out / 'per_episode.csv', index=False)
    plot_pose(rows, rigs, out / 'pose_error.png')
    print(json.dumps(s, indent=2))


if __name__ == '__main__':
    main()
