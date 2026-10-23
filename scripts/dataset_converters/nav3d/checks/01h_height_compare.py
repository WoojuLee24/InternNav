"""01h — 노은역 navmesh 를 로봇 키 기준 두 가지로 만들어 비교한다 (실제 로봇 0.9 m vs 계획 기준 1.8 m).

키(agent_height) = 이 높이까지 머리 위가 비어 있어야 걸을 수 있는 면으로 친다. 키를 키우면 천장 · 계단 밑 · 낮은 구조물
아래가 빠진다. 턱 높이(agent_max_climb) 0.20 m 는 둘 다 같다.

| 비교 | 묻는 것 |
|---|---|
| 면적 | 0.9 m navmesh 위 점(0.1 m 간격) 중 1.8 m navmesh 에 없는 비율 · 어디인가(그림 · ply) |
| 기존 경로 | 0.9 m 로 계획 · 다듬은 경로 중 1.8 m 기준으로 못 쓰는 것 (1.8 navmesh 밖 또는 몸 여유 < r_b — 구 중심 0.5 · h − r_b) |
| 다시 계획 | 같은 시작 · 끝을 1.8 m navmesh 로 계획 · 다듬기 — 끝점 없음 / 경로 없음 / 사용 가능 · 길이 변화 |

표본: 03 노은역 층간 100 쌍(같은 시작·끝) + 0.9 m 가장 큰 섬 위 무작위 100 쌍.

사용 (habitat 컨테이너, 저장소 루트) — 01 을 두 키로 먼저:
    /usr/bin/python scripts/dataset_converters/nav3d/01_build_navmesh.py --scene noeun_station --agent_height 0.9 --out_dir scripts/dataset_converters/nav3d/logs/01_height/h0.9
    /usr/bin/python scripts/dataset_converters/nav3d/01_build_navmesh.py --scene noeun_station --agent_height 1.8 --out_dir scripts/dataset_converters/nav3d/logs/01_height/h1.8
    /usr/bin/python scripts/dataset_converters/nav3d/checks/01h_height_compare.py
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/ (이 파일은 nav3d/checks/ 에 있다)
sys.path.insert(0, str(HERE))
import nav3d_path as npth  # noqa: E402
import nav3d_scenes as scenes  # noqa: E402
import nav3d_utils as nu  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--heights', nargs=2, type=float, default=[0.9, 1.8], help='(실제 로봇 키, 계획 기준 키)')
    ap.add_argument('--root', default='scripts/dataset_converters/nav3d/logs/01_height')
    ap.add_argument('--refine_dir', default='scripts/dataset_converters/nav3d/logs/03_refine_paths', help='03 노은역 경로(시작·끝)')
    ap.add_argument('--n_random', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def on_surface(sc, P, tol=0.3):
    """점 P (N,3) 각각: 같은 xy 에서 높이 tol 안에 navmesh 표면이 있는가."""
    ok = np.zeros(len(P), bool)
    for i, p in enumerate(P):
        c = sc.surf.candidates(p[:2])
        ok[i] = bool(c) and min(abs(z - p[2]) for z in c) <= tol
    return ok


def check(sc, md, X, r_b, h):
    """키 h 기준 사용 가능: 경로 모든 점이 sc navmesh 위 & 몸 여유 ≥ r_b. 양 끝 0.3 m 제외.

    몸 여유 = 반지름 r_b 구 두 개(중심 바닥 + 0.5 m, 바닥 + h − r_b)에서 mesh 까지 거리 — 위 구의 꼭대기가 정확히 키 h.
    옛 자(중심 바닥 + h − 0.05)는 천장이 h 에 있으면 거리 0.05 라 머리 위 h + r_b − 0.05 가 비어야 통과한다(너무 엄격) —
    비교용으로 `mesh_clear_old_min_m` 에 남긴다."""
    on = on_surface(sc, X[::2])
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1))]
    inner = (L > 0.3) & (L < L[-1] - 0.3)
    Y = X[inner][::2]
    if len(Y):
        lo = md(Y + [0, 0, 0.5])
        clr = np.minimum(lo, md(Y + [0, 0, h - r_b]))
        old = np.minimum(lo, md(Y + [0, 0, h - 0.05]))
    else:
        clr = old = np.array([np.inf])
    return {'on_frac': float(on.mean()), 'mesh_clear_min_m': float(clr.min()), 'mesh_clear_old_min_m': float(old.min()),
            'usable': bool(on.all() and clr.min() >= r_b), 'usable_old': bool(on.all() and old.min() >= r_b),
            'length_m': float(L[-1])}


def main():
    args = parse()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import trimesh
    h_lo, h_hi = args.heights
    root = Path(args.root)
    spec = scenes.SCENES['noeun_station']
    mesh = nu.add_box_blockers(nu.load_mesh_zup(spec['mesh']), spec.get('blockers', []))
    md = nu.MeshDistance(mesh)
    pf = {h: nu.load_pathfinder(str(root / f'h{h}' / 'noeun_station' / 'noeun_station.navmesh')) for h in (h_lo, h_hi)}
    sc = {h: npth.NavScene(pf[h]) for h in (h_lo, h_hi)}
    r_b = sc[h_lo].agent_radius
    out = root / 'compare'
    out.mkdir(parents=True, exist_ok=True)
    summ = {'heights': [h_lo, h_hi], 'r_b': r_b, 'max_climb': float(pf[h_lo].nav_mesh_settings.agent_max_climb)}

    # 1) 면적: 낮은 키 navmesh 위 점 중 높은 키 navmesh 에 없는 것
    V = sc[h_lo].tris.reshape(-1, 3)
    nm = trimesh.Trimesh(vertices=V, faces=np.arange(len(V)).reshape(-1, 3), process=False)
    pts, _ = trimesh.sample.sample_surface_even(nm, int(nm.area / 0.1 ** 2), seed=args.seed)
    kept = on_surface(sc[h_hi], pts)
    lost = pts[~kept]
    summ['area'] = {f'navmesh area h{h} (m2)': float(nu.tri_area_normal(sc[h].tris)[0].sum()) for h in (h_lo, h_hi)}
    summ['area']['points on low-h navmesh'] = int(len(pts))
    summ['area']['lost at high h (frac)'] = float((~kept).mean())
    summ['area']['lost at high h (m2, ≈ pts × 0.01)'] = float((~kept).sum() * 0.01)
    lv = nu.assign_level(lost[:, 2], sc[h_lo].levels, 0.6) if len(lost) else np.zeros(0, int)
    st = np.where(lv >= 0, sc[h_lo].lv2st[np.maximum(lv, 0)], -1) if len(lost) else lv
    summ['area']['lost m2 by storey (S0 B2, S1 B1, S2 ground, -1 = between)'] = {int(k): float((st == k).sum() * 0.01) for k in np.unique(st)}

    # 2) 경로: 03 층간 100 쌍 + 무작위 100 쌍
    A = np.load(Path(args.refine_dir) / 'scenes' / 'noeun_station_traj.npz')
    pairs = [('interstorey', A[k][0].astype(float), A[k][-1].astype(float)) for k in sorted(A.files, key=lambda s: int(s.split('_')[0]))
             if k.endswith('_raw')]
    rng = np.random.default_rng(args.seed)
    cand = pts                                                   # 같은 섬(get_island)인 쌍만 쓴다
    while len([p for p in pairs if p[0] == 'random']) < args.n_random:
        a, b = cand[rng.integers(len(cand))], cand[rng.integers(len(cand))]
        if np.linalg.norm(a[:2] - b[:2]) < 5:
            continue
        isl = [pf[h_lo].get_island(nu.ours_to_hab(p).astype(np.float32)) for p in (a, b)]
        if isl[0] != isl[1]:
            continue
        pairs.append(('random', a, b))
    rows = []
    for kind, a, b in pairs:
        row = {'kind': kind}
        P = sc[h_lo].find_path(a, b)
        if P is None:
            row['low'] = None
            rows.append(row)
            continue
        X = npth.refine(sc[h_lo], P)['traj']
        row['low'] = check(sc[h_lo], md, X, r_b, h_lo)                 # 낮은 키 기준 (기준선)
        row['low_path_at_high'] = check(sc[h_hi], md, X, r_b, h_hi)   # 같은 경로를 높은 키 기준으로
        ends = on_surface(sc[h_hi], np.array([a, b]))
        if not ends.all():
            row['high'] = 'endpoint_lost'
        else:
            P2 = sc[h_hi].find_path(a, b)
            if P2 is None:
                row['high'] = 'no_path'
            else:
                row['high'] = check(sc[h_hi], md, npth.refine(sc[h_hi], P2)['traj'], r_b, h_hi)
        rows.append(row)
        print(f"  {kind:11s} low {row['low']['usable']} · low path @high {row['low_path_at_high']['usable']} · "
              f"replan high {row['high'] if isinstance(row['high'], str) else row['high']['usable']}", flush=True)

    for kind in ('interstorey', 'random'):
        G = [r for r in rows if r['kind'] == kind]
        d = {'n': len(G)}
        d['low h: planned'] = sum(r['low'] is not None for r in G)
        d['low h: usable'] = sum(r['low'] is not None and r['low']['usable'] for r in G)
        d['low-h path, judged at high h: usable'] = sum(r['low'] is not None and r['low_path_at_high']['usable'] for r in G)
        d['  - off high-h navmesh'] = sum(r['low'] is not None and r['low_path_at_high']['on_frac'] < 1 for r in G)
        d['  - head clearance < r_b'] = sum(r['low'] is not None and r['low_path_at_high']['on_frac'] == 1 and
                                             r['low_path_at_high']['mesh_clear_min_m'] < r_b for r in G)
        d['replan high h: endpoint lost'] = sum(r.get('high') == 'endpoint_lost' for r in G)
        d['replan high h: no path'] = sum(r.get('high') == 'no_path' for r in G)
        H = [r for r in G if isinstance(r.get('high'), dict)]
        d['replan high h: usable'] = sum(r['high']['usable'] for r in H)
        lr = [r['high']['length_m'] / r['low']['length_m'] for r in H if r['low']['length_m'] > 0]
        d['replan length ratio high/low (median, max)'] = [float(np.median(lr)), float(np.max(lr))] if lr else None
        d['replan longer than +10 %'] = int(sum(x > 1.1 for x in lr))
        d['(old strict head check h-0.05) low h usable / low path @high usable / replan high usable'] = [
            sum(r['low'] is not None and r['low']['usable_old'] for r in G),
            sum(r['low'] is not None and r['low_path_at_high']['usable_old'] for r in G),
            sum(r['high']['usable_old'] for r in H)]
        summ[kind] = d
    json.dump({'summary': summ, 'rows': rows}, open(out / 'compare.json', 'w'), indent=1, default=float)
    print(json.dumps(summ, indent=1, ensure_ascii=False))

    # 그림: 층별 평면 — 회색 = 둘 다 걸을 수 있음, 빨강 = 높은 키에서 빠진 곳
    storeys = sorted(set(sc[h_lo].lv2st.tolist()))
    fig, axs = plt.subplots(1, len(storeys), figsize=(9 * len(storeys), 8), squeeze=False)
    cz = sc[h_lo].tris.mean(1)[:, 2]
    tl = nu.assign_level(cz, sc[h_lo].levels, 0.6)
    ts = np.where(tl >= 0, sc[h_lo].lv2st[np.maximum(tl, 0)], -1)
    for ax, s_ in zip(axs[0], storeys):
        m = ts == s_
        ax.add_collection(PolyCollection(sc[h_lo].tris[m][:, :, :2], facecolors='0.8', edgecolors='none'))
        sel = st == s_
        ax.scatter(lost[sel, 0], lost[sel, 1], s=2, c='red', label=f'lost at {h_hi} m ({sel.sum() * 0.01:.1f} m2)')
        ax.autoscale(); ax.set_aspect('equal'); ax.legend(fontsize=9)
        ax.set_title(f'S{s_}: gray = walkable at {h_lo} m, red = not walkable at {h_hi} m (head room)', fontsize=10)
    fig.savefig(out / 'lost_area.png', dpi=70, bbox_inches='tight'); plt.close(fig)
    # 3D 확인: 원본 mesh 점(회색, intensity 0) + 낮은 키 navmesh 점(파랑, 0.3) + 높은 키에서 빠진 곳(빨강, 1.0)
    sys.path.insert(0, str(HERE))
    s03 = importlib.util.spec_from_file_location('s03', HERE / 'checks' / '03s_stairs_label_check.py')
    m03 = importlib.util.module_from_spec(s03)
    s03.loader.exec_module(m03)
    sp, _ = trimesh.sample.sample_surface(mesh, int(min(3_000_000, mesh.area / 0.05 ** 2)), seed=0)
    xyz = np.vstack([sp, pts[kept] + [0, 0, 0.02], lost + [0, 0, 0.02]])
    rgb = np.vstack([np.tile((150, 150, 150), (len(sp), 1)), np.tile((60, 120, 230), (int(kept.sum()), 1)),
                     np.tile((230, 30, 30), (len(lost), 1))]).astype(np.uint8)
    it = np.r_[np.zeros(len(sp)), np.full(int(kept.sum()), 0.3), np.ones(len(lost))].astype(np.float32)
    m03.write_ply(out / f'noeun_lost_h{h_hi}.ply', xyz, rgb, it)
    print(f'  -> {out}/compare.json · lost_area.png · noeun_lost_h{h_hi}.ply')
    return 0


if __name__ == '__main__':
    sys.exit(main())
