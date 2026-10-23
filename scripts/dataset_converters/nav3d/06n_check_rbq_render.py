"""06n — 06m (RBQ 가 물리로 걸은 자세로 렌더) 결과를 검사한다. vln_ce · vln_pe 둘 다.

06b 의 R2(전진 0.25 m·회전 15° 정확) · R3(경로 0.30 m) · R4(카메라 0.584 ± 0.06 m) 는 카메라를 경로 위에 바로 놓는 06 용이다.
06m 은 몸이 흔들리고 걸음이 정확히 0.25 m 가 아니므로 기준을 RBQ 에 맞춰 바꾼다.

## 용어
- **trunk**: RBQ 몸통 (base_link). 06k 가 action 마다 위치·자세를 기록했다 (walk json).
- **D455 장착**: trunk → 카메라. sim_docker `sensor_configuration.yaml` 값 (전방 0.365 m · 위 0.034 m · 아래로 14.76°).
- **카메라 바닥 높이**: 카메라에서 수직 아래로 쏜 광선이 mesh 에 닿기까지 거리.

## 판정 기준 (결과를 보기 전에 정함)
| | 묻는 것 | 기준 | 근거 |
|---|---|---|---|
| M1 형식 | 프레임 수 == walk 프레임 수 == parquet 행 · 이미지 크기 | 전 ep | vln_ce rgb 640×480 jpg · depth uint16 png / vln_pe rgb (T,256,256,3) uint8 · depth (T,256,256[,1]) float32 |
| M2 장착 | 저장된 카메라 자세 == walk trunk 자세 ∘ D455 장착 | 위치 ±1 mm · 회전 ±0.01° 전 프레임 | vln_pe 는 world 값 그대로, vln_ce 는 시작 기준 상대라 첫 프레임과의 거리·회전각으로 비교 |
| M3 높이 | 평지 ep 카메라 바닥 높이 | 프레임 ≥ 95 % 가 `--m3_lo` ~ `--m3_hi` (0.45 ~ 0.66 m) | 처음 기준 0.50 은 trunk 0.518(정지)~0.55(설계) 가정. **2026-10-21 결과를 본 뒤 0.45 로 낮춤 (사용자 결정)**: 걷는 trunk 중앙 0.483 m (p5 0.469) + 장착 0.034 = 0.517, 걸음 흔들림을 허용 |
| M4 경로 | trunk xy ↔ v4 경로 · 끝점 ↔ 도착 | ep 최대 ≤ 0.5 m · 끝 ≤ 0.5 m 전 ep | 06k 성공 판정과 같은 0.5 m |
| M5 depth | 유효 depth 비율 ≥ 0.5 인 프레임 | ≥ 99 % | 06b R5 와 같음 |
| M6 rgb | 평균 밝기 > 15/255 인 프레임 | ≥ 99 % | 06b R6 와 같음 |
| M7 action | 저장 action == walk action (vln_ce 그대로 · vln_pe 한 칸 당기고 끝 0) | 전 ep | 06 / 06p 규약 |
| M8 두 포맷 | vln_ce ↔ vln_pe **서로 직접** 대조: 프레임 수 · action (pe = ce 한 칸 당김) · 첫 프레임 기준 카메라 이동 거리·회전각 · 같은 프레임 이미지 | 프레임·action 전 ep 일치 · 자세 ±1 mm/±0.01° · rgb 상관 중앙 ≥ 0.9 · depth 차 중앙 ≤ 0.05 m | 같은 걷기·같은 렌더러여야 한다. 이미지는 vln_pe(90°) 가운데를 vln_ce 시야(≈69°)만큼 잘라 크기를 맞춰 비교 |

출력: summary json · 눈으로 보는 프레임 묶음 png (포맷마다).

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/06n_check_rbq_render.py
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = 'data/InternData-N1-v0.5-mini'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--walk_json', default='scripts/dataset_converters/nav3d/logs/06k_rbq_walk/walk_noeun_v4.json')
    ap.add_argument('--paths_json', default='scripts/dataset_converters/nav3d/logs/04_export_paths/noeun_station/noeun_station_paths_meshfloor_straight_v4.json')
    ap.add_argument('--vlnce_root', default=f'{D}/vln_ce_noeun_nav3d_rbq_v4/traj_data/r2r/noeunstation')
    ap.add_argument('--vlnpe_root', default=f'{D}/vln_pe_noeun_nav3d_rbq_v4/traj_data/r2r/noeunstation')
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--mesh', default='data/scene_data/noeun_station/noeun_station_mid.ply')
    ap.add_argument('--max_episodes', type=int, default=0, help='0 = 전부')
    ap.add_argument('--m3_lo', type=float, default=0.45, help='M3 아래 경계 (처음 0.50 → 2026-10-21 0.45, 걷는 trunk 0.483 m 실측)')
    ap.add_argument('--m3_hi', type=float, default=0.66)
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/06n_check_rbq_render')
    return ap.parse_args()


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def rot_angle_deg(Ra, Rb):
    """두 회전 사이 각 — acos 대신 atan2 (각이 0 근처일 때 acos 는 float32 저장값에서 ≈0.03° 오차를 낸다)."""
    Q = Ra.T @ Rb
    s = np.linalg.norm([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]]) / 2
    return math.degrees(math.atan2(s, (np.trace(Q) - 1) / 2))


def axis_angle_deg(a, b):
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b)))


def path_dev(xy, P):
    """xy (N,2) 각 점에서 경로 폴리라인 P (M,2) 까지 최소 거리."""
    a, b = P[:-1], P[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((xy[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0, 1)
    q = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(xy[:, None, :] - q, axis=2).min(1)


def cross_check(W, ce_root, pe_root, rig, every=10):
    """M8: vln_ce 와 vln_pe 를 서로 직접 대조 (walk json 을 거치지 않음)."""
    import pandas as pd
    from PIL import Image
    fk = 128.0 / 468.2242151                              # vln_ce 픽셀 -> vln_pe 픽셀 (초점거리 비)
    hw, hh = 320 * fk, 240 * fk
    box = (128 - hw, 128 - hh, 128 + hw, 128 + hh)
    n_bad, a_bad, pe_err, re_err, corr, ddiff = 0, 0, 0.0, 0.0, [], []
    for ei in range(len(W)):
        dc = pd.read_parquet(ce_root / 'data' / 'chunk-000' / f'episode_{ei:06d}.parquet')
        dp = pd.read_parquet(pe_root / 'data' / 'chunk-000' / f'episode_{ei:06d}.parquet')
        if len(dc) != len(dp):
            n_bad += 1
            continue
        a_ce, a_pe = dc['action'].to_numpy(), dp['observation.action'].to_numpy()
        a_bad += int(not np.array_equal(a_pe, np.r_[a_ce[1:], 0]))
        Pc = np.stack([np.stack(x).astype(np.float64) for x in dc[f'pose.{rig}']])
        cp = np.stack(dp['observation.camera_position'].to_numpy())
        cq = np.stack(dp['observation.camera_orientation'].to_numpy())
        ax0 = None
        for i in range(len(dc)):
            pe_err = max(pe_err, abs(np.linalg.norm(Pc[i, :3, 3] - Pc[0, :3, 3]) - np.linalg.norm(cp[i] - cp[0])))
            w, x, y, z = cq[i]
            ax = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])   # 저장 quat 의 x 축 = 광축
            ax0 = ax if ax0 is None else ax0
            re_err = max(re_err, abs(axis_angle_deg(Pc[0, :3, 2], Pc[i, :3, 2]) - axis_angle_deg(ax0, ax)))
        vd_ce, vd_pe = ce_root / 'videos' / 'chunk-000', pe_root / 'videos' / 'chunk-000'
        rgb = np.load(vd_pe / 'observation.images.rgb' / f'episode_{ei:06d}.npy', mmap_mode='r')
        dep = np.load(vd_pe / 'observation.images.depth' / f'episode_{ei:06d}.npy', mmap_mode='r')
        for i in range(0, len(dc), every):
            crop = Image.fromarray(np.asarray(rgb[i])).crop(box)
            ce = Image.open(vd_ce / f'observation.images.rgb.{rig}' / f'episode_{ei:06d}_{i}.jpg').convert('RGB').resize(crop.size)
            g1, g2 = np.asarray(crop.convert('L'), np.float64).ravel(), np.asarray(ce.convert('L'), np.float64).ravel()
            corr.append(float(np.corrcoef(g1, g2)[0, 1]))
            dpe = Image.fromarray(np.asarray(dep[i]).reshape(256, 256).astype(np.float32) * 10.0).crop(box)
            dce = np.asarray(Image.open(vd_ce / f'observation.images.depth.{rig}' / f'episode_{ei:06d}_{i}.png')).astype(np.float32) / 1000.0
            dce = np.asarray(Image.fromarray(dce).resize(dpe.size, Image.NEAREST))
            dpe = np.asarray(dpe)
            ok = (dce > 0) & (dpe < 9.99)
            if ok.any():
                ddiff.append(float(np.median(np.abs(dce[ok] - dpe[ok]))))
    r = {'frame_mismatch_ep': n_bad, 'action_mismatch_ep': a_bad, 'pos_err_m': pe_err, 'rot_err_deg': re_err,
         'rgb_corr_median': float(np.median(corr)), 'rgb_corr_min': float(np.min(corr)),
         'depth_diff_median_m': float(np.median(ddiff)), 'n_frames_img': len(corr)}
    r['pass'] = (n_bad == 0 and a_bad == 0 and pe_err <= 1e-3 and re_err <= 0.01 and r['rgb_corr_median'] >= 0.9
                 and r['depth_diff_median_m'] <= 0.05)
    return r


def main():
    import trimesh
    import pandas as pd
    from PIL import Image
    args = parse()
    m6m = _load('m06m', HERE / '06m_render_rbq.py')
    W = [r for r in json.loads(Path(args.walk_json).read_text())['results'] if r['success']]
    if args.max_episodes:
        W = W[:args.max_episodes]
    Pj = {str(e['episode_id']): e for e in json.loads(Path(args.paths_json).read_text())['episodes']}
    mesh = trimesh.load(args.mesh, force='mesh')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    M = m6m.trunk_to_cam_cv()
    R = {}

    # M4 · M3 (포맷 무관 — walk 자세와 mesh 로)
    dev, endd, hgt = [], [], []
    for r in W:
        e = Pj[r['key'].split(':')[1]]
        T = np.asarray(e['trajectory'])
        xy = np.array([f['pos'][:2] for f in r['frames']])
        dev.append(float(path_dev(xy, T[:, :2]).max()))
        endd.append(float(np.linalg.norm(xy[-1] - T[-1, :2])))
        if e['kind'] == 'flat':
            C = m6m.frames_to_poses(r['frames'])[:, :3, 3]
            loc, idx, _ = mesh.ray.intersects_location(C, np.tile([0, 0, -1.0], (len(C), 1)), multiple_hits=False)
            h = np.full(len(C), np.nan)
            h[idx] = C[idx, 2] - loc[:, 2]
            hgt.append(h)
    hgt = np.concatenate(hgt) if hgt else np.full(1, np.nan)
    R['M3'] = {'frames': int(len(hgt)), 'range_m': [args.m3_lo, args.m3_hi], 'in_range_frac': float(np.mean((hgt >= args.m3_lo) & (hgt <= args.m3_hi))),
               'pct_5_50_95': np.nanpercentile(hgt, [5, 50, 95]).round(3).tolist()}
    R['M3']['pass'] = R['M3']['in_range_frac'] >= 0.95      # 평지 ep 가 없으면 FAIL 로 드러난다
    R['M4'] = {'dev_max': max(dev), 'end_max': max(endd), 'pass': max(dev) <= 0.5 and max(endd) <= 0.5,
               'worst': sorted(zip(dev, [r['key'] for r in W]), reverse=True)[:3]}

    for fmt, root in (('vlnce', Path(args.vlnce_root)), ('vlnpe', Path(args.vlnpe_root))):
        if not root.exists():
            R[fmt] = {'skip': f'{root} 없음'}
            continue
        f1, pos_err, rot_err, act_bad, dvalid, bright, shown = [], 0.0, 0.0, [], [], [], []
        for ei, r in enumerate(W):
            fr = r['frames']
            n = len(fr)
            df = pd.read_parquet(root / 'data' / 'chunk-000' / f'episode_{ei:06d}.parquet')
            Cw = m6m.frames_to_poses(fr)
            a_ce = np.array([f['action'] for f in fr])
            if fmt == 'vlnce':
                vd = root / 'videos' / 'chunk-000'
                rgbs = [vd / f'observation.images.rgb.{args.rig}' / f'episode_{ei:06d}_{i}.jpg' for i in range(n)]
                deps = [vd / f'observation.images.depth.{args.rig}' / f'episode_{ei:06d}_{i}.png' for i in range(n)]
                ok = len(df) == n and all(p.exists() for p in rgbs + deps)
                if ok:
                    im0, d0 = Image.open(rgbs[0]), np.asarray(Image.open(deps[0]))
                    ok = im0.size == (640, 480) and d0.dtype == np.uint16 and d0.shape == (480, 640)
                f1.append(ok)
                if not ok:
                    continue
                Ps = np.stack([np.stack(x).astype(np.float64) for x in df[f'pose.{args.rig}']])
                for i in range(n):
                    pos_err = max(pos_err, abs(np.linalg.norm(Ps[i, :3, 3] - Ps[0, :3, 3]) - np.linalg.norm(Cw[i, :3, 3] - Cw[0, :3, 3])))
                    rot_err = max(rot_err, abs(rot_angle_deg(Ps[0, :3, :3], Ps[i, :3, :3]) - rot_angle_deg(Cw[0, :3, :3], Cw[i, :3, :3])))
                act_bad.append(not np.array_equal(df['action'].to_numpy(), a_ce))
                for i in range(0, n, 5):
                    dvalid.append(float((np.asarray(Image.open(deps[i])) > 0).mean()))
                    bright.append(float(np.asarray(Image.open(rgbs[i])).mean()))
                if ei % 25 == 0:
                    shown.append([np.asarray(Image.open(rgbs[i]).convert('RGB').resize((320, 240))) for i in np.linspace(0, n - 1, 6).astype(int)])
            else:
                vd = root / 'videos' / 'chunk-000'
                rgb = np.load(vd / 'observation.images.rgb' / f'episode_{ei:06d}.npy', mmap_mode='r')
                dep = np.load(vd / 'observation.images.depth' / f'episode_{ei:06d}.npy', mmap_mode='r')
                ok = (len(df) == n and rgb.shape == (n, 256, 256, 3) and rgb.dtype == np.uint8 and dep.shape[:3] == (n, 256, 256)
                      and dep.dtype == np.float32)
                f1.append(ok)
                if not ok:
                    continue
                cp = np.stack(df['observation.camera_position'].to_numpy())
                rp = np.stack(df['observation.robot_position'].to_numpy())
                tp = np.array([f['pos'] for f in fr])
                pos_err = max(pos_err, float(np.abs(cp - Cw[:, :3, 3]).max()), float(np.abs(rp - tp).max()))
                cq = np.stack(df['observation.camera_orientation'].to_numpy())
                for i in range(n):
                    w, x, y, z = cq[i]
                    Rq = m6m.quat_wxyz_to_R([w, x, y, z])
                    # 06p c2w_to_vlnpe_quat 규약이 OpenCV 와 다를 수 있다 → 광축(카메라 앞) 방향만 비교: OpenCV z 축 vs 저장 quat 의 x 축
                    rot_err = max(rot_err, axis_angle_deg(Rq[:, 0], Cw[i, :3, 2]))
                a_pe = np.r_[a_ce[1:], 0]
                act_bad.append(not np.array_equal(df['observation.action'].to_numpy(), a_pe))
                d = np.asarray(dep[::5]).reshape(-1, 256 * 256)
                dvalid += (d < 1.0).mean(1).tolist()
                bright += np.asarray(rgb[::5]).reshape(len(d), -1).mean(1).tolist()
                if ei % 25 == 0:
                    shown.append([np.asarray(rgb[i]) for i in np.linspace(0, n - 1, 6).astype(int)])
        dvalid, bright = np.array(dvalid), np.array(bright)
        R[fmt] = {'M1': {'ok': int(sum(f1)), 'n': len(f1), 'pass': all(f1)},
                  'M2': {'pos_err_m': pos_err, 'rot_err_deg': rot_err, 'pass': pos_err <= 1e-3 and rot_err <= 0.01,
                         'note': 'vln_pe 회전은 광축 방향 비교' if fmt == 'vlnpe' else '첫 프레임 기준 거리·회전각'},
                  'M5': {'frac': float(np.mean(dvalid >= 0.5)), 'pass': float(np.mean(dvalid >= 0.5)) >= 0.99},
                  'M6': {'frac': float(np.mean(bright > 15)), 'pass': float(np.mean(bright > 15)) >= 0.99},
                  'M7': {'bad': int(sum(act_bad)), 'pass': not any(act_bad)}}
        if shown:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(len(shown), 6, figsize=(18, 2.6 * len(shown) * (0.75 if fmt == 'vlnce' else 1.0)))
            for row, ims in zip(np.atleast_2d(axs), shown):
                for ax, im in zip(row, ims):
                    ax.imshow(im)
                    ax.axis('off')
            fig.tight_layout()
            fig.savefig(out / f'frames_{fmt}.jpg', dpi=70)
    if 'skip' not in R['vlnce'] and 'skip' not in R['vlnpe']:
        R['M8'] = cross_check(W, Path(args.vlnce_root), Path(args.vlnpe_root), args.rig)
    (out / 'summary.json').write_text(json.dumps(R, indent=1, default=float))
    if 'M8' in R:
        print('M8', 'PASS' if R['M8']['pass'] else 'FAIL', {a: b for a, b in R['M8'].items() if a != 'pass'})
    for k in ('M3', 'M4'):
        print(k, 'PASS' if R[k]['pass'] else 'FAIL', {a: b for a, b in R[k].items() if a != 'pass'})
    for fmt in ('vlnce', 'vlnpe'):
        if 'skip' in R[fmt]:
            print(fmt, R[fmt]['skip'])
            continue
        for k, v in R[fmt].items():
            print(fmt, k, 'PASS' if v['pass'] else 'FAIL', {a: b for a, b in v.items() if a != 'pass'})
    print('->', out)


if __name__ == '__main__':
    main()
