"""노은역 걸어 보기 지도 배경 — 층마다 paint_bg_<tag>.png + paint_meta_<tag>.json (teleop_view.py 가 읽는 형식).

logs/etri/paint_bg.py 와 같은 그림 (navmesh 바닥 높이 + 바닥 위 0.3~1.5 m 메쉬 = 장애물). 그쪽은 ETRI 경로·걷기
결과 파일을 같이 읽어서 노은역에 못 쓴다. 층 경계 = verify.json 층 높이의 중간.

    python3 scripts/dataset_converters/nav3d/noeun/tools/noeun_walk_bg.py   (habitat 컨테이너, 저장소 루트)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from matplotlib import cm
from PIL import Image
from scipy import ndimage

N = Path(__file__).resolve().parents[2]   # nav3d/
sys.path.insert(0, str(N / 'utils'))
import nav3d_path as npth  # noqa: E402
import nav3d_utils as nu  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--navmesh', default=str(N / 'logs/noeun/12_build_navmesh/noeun_station/noeun_station.navmesh'))
ap.add_argument('--mesh', default='data/data/noeun_eval/navmesh/noeun_station/noeun_station.glb')
ap.add_argument('--verify', default='data/data/noeun_eval/navmesh/noeun_station/verify.json')
ap.add_argument('--out', default=str(N / 'logs/noeun_walk/noeun_station'))
ap.add_argument('--res', type=float, default=0.1, help='m/px')
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

sc = npth.NavScene(nu.load_pathfinder(a.navmesh))
V = np.asarray(trimesh.load(a.mesh, force='mesh', process=False, skip_materials=True).vertices)
zs = [s['z'] for s in json.load(open(a.verify))['storeys']]
cuts = [-1e9] + [(zs[i] + zs[i + 1]) / 2 for i in range(len(zs) - 1)] + [1e9]
names = ['B2_platform', 'B1_concourse', 'G_exit'] if len(zs) == 3 else [f'S{i}' for i in range(len(zs))]

for i, tag in enumerate(names):
    ZLO, ZHI = cuts[i], cuts[i + 1]
    T = sc.tris[(sc.tris[:, :, 2].mean(1) >= ZLO) & (sc.tris[:, :, 2].mean(1) < ZHI)]
    x0, y0 = T[:, :, 0].min() - 5, T[:, :, 1].min() - 5
    x1, y1 = T[:, :, 0].max() + 5, T[:, :, 1].max() + 5
    W, H = int((x1 - x0) / a.res) + 1, int((y1 - y0) / a.res) + 1

    def px(x, y):
        return (np.asarray(x) - x0) / a.res, (y1 - np.asarray(y)) / a.res   # 위가 +y

    fz = np.full((H, W), np.nan)
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    for t in T[np.argsort(T[:, :, 2].mean(1))]:
        u, v = px(t[:, 0], t[:, 1])
        ua, ub = int(max(u.min(), 0)), int(min(u.max() + 1, W))
        va, vb = int(max(v.min(), 0)), int(min(v.max() + 1, H))
        if ua >= ub or va >= vb:
            continue
        X, Y = jj[va:vb, ua:ub] + 0.5, ii[va:vb, ua:ub] + 0.5
        (ax, bx, cx), (ay, by, cy) = u, v
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        l1 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
        l2 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
        m = (l1 >= -1e-6) & (l2 >= -1e-6) & (l1 + l2 <= 1 + 1e-6)
        fz[va:vb, ua:ub][m] = t[:, 2].mean()
    nav = ~np.isnan(fz)
    idx = ndimage.distance_transform_edt(~nav, return_distances=False, return_indices=True)
    floor = fz[idx[0], idx[1]]

    u, v = px(V[:, 0], V[:, 1])
    k = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (V[:, 2] >= ZLO - 1) & (V[:, 2] < ZHI + 3)
    u, v, z = u[k].astype(int), v[k].astype(int), V[k, 2]
    dz = z - floor[v, u]
    obs = np.zeros((H, W)); np.add.at(obs, (v[(dz > 0.3) & (dz < 1.5)], u[(dz > 0.3) & (dz < 1.5)]), 1)
    gnd = np.zeros((H, W)); np.add.at(gnd, (v[np.abs(dz) <= 0.3], u[np.abs(dz) <= 0.3]), 1)

    zmin, zmax = np.nanpercentile(fz, 1), np.nanpercentile(fz, 99)
    rgb = np.full((H, W, 3), 0.93)
    rgb[gnd > 0] = 0.80
    col = cm.get_cmap('viridis')(np.clip((fz - zmin) / max(zmax - zmin, 1e-3), 0, 1))[..., :3]
    rgb[nav] = 0.35 * rgb[nav] + 0.65 * col[nav]
    rgb[ndimage.binary_dilation(obs > 1) & ~nav] = 0.12
    rgb[(obs > 1) & nav] = 0.3
    Image.fromarray((rgb * 255).astype(np.uint8)).save(out / f'paint_bg_{tag}.png', optimize=True)
    meta = {'scene': 'noeun_station', 'tag': tag, 'zlo': ZLO, 'zhi': ZHI, 'x0': float(x0), 'y1': float(y1), 'res': a.res,
            'w': W, 'h': H, 'zmin': float(zmin), 'zmax': float(zmax), 'nav_m2': float(nav.sum() * a.res ** 2)}
    (out / f'paint_meta_{tag}.json').write_text(json.dumps(meta))
    print(f'[bg] {tag}: {W}x{H} px · 바닥 {meta["nav_m2"]:.0f} m² · z {zmin:.2f}~{zmax:.2f}')
