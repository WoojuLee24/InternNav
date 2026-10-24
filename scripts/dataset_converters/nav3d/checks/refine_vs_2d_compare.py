"""03c — 2D occupancy 방식(gs_vlnpe) 과 nav3d 방식을 **같은 에피소드 · 같은 자** 로 비교한다 (vln_ce).

같은 자 (CLAUDE.md 9-1: 지도가 다르면 비교하지 않는다):
- **mesh 3D 거리**: 경로 점(바닥 + 0.5 m, 바닥 + 키 − 0.05 m)에서 원본 mesh 까지 — 두 방식 어느 지도와도 무관한 중립 자.
- **navmesh 위**: GT(R2R reference_path)가 나온 지도. nav3d 는 이 지도로 계획하므로 nav3d 에 유리한 자임을 감안한다.
- 각 방식 **자기 지도** 판정도 옆에 적는다(2D = ESDF `points_min_m ≥ r_b` 03u 결과, nav3d = 03 결과). 이건 서로 비교하지 않는다.

| 칸 | 뜻 (경로 하나는 한 칸) |
|---|---|
| 생성 실패 | 경로가 없음 (2D: 끝점 막힘·A* 실패·계단 입구 막힘·계단 길 없음 / nav3d: find_path 실패) |
| 관통 | navmesh 밖 점이 있거나 mesh 3D 거리 < 0.02 m |
| r_b 미달 | 관통 아님 & mesh 3D 거리 < r_b (0.10 m) |
| 사용 가능 | 둘 다 아님 |

비교 대상
- nav3d: 03 의 sg 경로(GT 시작 → 끝 find_path → 다듬기), 에피소드 = 03 표본(씬당 평지 10 + 층 이동 전부, trajectory 중복 제거).
- 2D 평지: gs_vlnpe 확정 설정 `vlnce_full/multifloor/on/paths`(= base, plan_r_b 0.15 + ②③④ + floor_tol 0.1).
- 2D 층 이동: `on_plan5/paths_mf`(plan — 계단 위치만 GT, 길은 새로) 과 `on/paths_mf`(gt — 계단 구간은 GT 길 그대로, 참고).
gs_vlnpe 파일은 읽기만 한다.

사용 (habitat 컨테이너, 저장소 루트):
    ls data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r | xargs -P 4 -I{} /usr/bin/python scripts/dataset_converters/nav3d/checks/refine_vs_2d_compare.py --scene {}
    /usr/bin/python scripts/dataset_converters/nav3d/checks/refine_vs_2d_compare.py --aggregate
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'utils'))
GS = HERE.parent / 'gs_vlnpe'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GS))
import nav3d_path as npth  # noqa: E402
import nav3d_utils as nu  # noqa: E402
from esdf_utils import discrete_frechet_distance  # noqa: E402

DOOR_WIDTH_M = 0.8
PENETRATE_M = 0.02
R_B = 0.10
MF = GS / 'logs' / 'vlnce_full' / 'multifloor'
SOURCES = {'2d': (MF / 'on' / 'paths', MF / 'on' / 'usable' / 'usable.json', 'usable'),
           '2d_plan': (MF / 'on_plan5' / 'paths_mf', MF / 'on_plan5' / 'usable' / 'usable.json', 'usable_multifloor'),
           '2d_gt': (MF / 'on' / 'paths_mf', MF / 'on' / 'usable' / 'usable.json', 'usable_multifloor')}


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', default=None)
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--in_dir', default='scripts/dataset_converters/nav3d/logs/common/refine_paths')
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/checks/refine_vs_2d_compare')
    ap.add_argument('--scene_root', default='data/scene_data/mp3d_ce/mp3d')
    return ap.parse_args()


def category(m):
    if m is None:
        return 'fail'
    if m['on_frac'] < 1.0 or m['mesh_clear_min_m'] < PENETRATE_M:
        return 'penetrate'
    if m['mesh_clear_min_m'] < R_B:
        return 'lt_rb'
    return 'usable'


def measure(sc, md, X, gt):
    """X (N,3) — z 는 각 방식이 준 바닥 높이. navmesh 위인 점은 navmesh 표면 z 로 바꿔 mesh 거리를 잰다."""
    z, _ = sc.surf.track_idx(X[:, :2], z0=X[0, 2])
    on = np.isfinite(z)
    Y = X.copy()
    Y[on, 2] = z[on]
    m = npth.evaluate(sc, Y, None, on, md, R_B)
    fr = discrete_frechet_distance(X[:, :2], gt[:, :2])
    off = np.where(~on)[0][::2]
    # navmesh 밖 점이 navmesh 에서 수평으로 얼마나 떨어졌나 (snap) — 높이 추적 실패와 진짜 밖을 가리려고
    m['off_max_m'] = float(max((np.linalg.norm(nu.hab_to_ours(np.asarray(sc.pf.snap_point(nu.ours_to_hab(X[i]).astype(np.float32))))[:2]
                                               - X[i, :2]) for i in off), default=0.0))
    m.update({'frechet_m': fr, 'same_route': bool(fr < DOOR_WIDTH_M),
              'len_ratio': m['length_m'] / max(1e-6, float(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1).sum()))})
    return m


def run_scene(args):
    s = args.scene
    j = json.load(open(Path(args.in_dir) / 'scenes' / f'{s}.json'))
    A = np.load(Path(args.in_dir) / 'scenes' / f'{s}_traj.npz')
    pf = nu.load_pathfinder(f'{args.scene_root}/{s}/{s}.navmesh')
    sc = npth.NavScene(pf)
    meta = json.load(open(GS / 'logs' / 'scene_meta' / f'{s}.json'))
    md = nu.MeshDistance(nu.load_mesh_zup(meta['mesh_path']))
    src = {}
    for k, (d, uj, ukey) in SOURCES.items():
        f = d / f'{s}_vlnce.json'
        eps = {e['episode_id']: e for e in json.load(open(f))['episodes']} if f.exists() else {}
        u = json.load(open(uj))[ukey].get(s, [])
        src[k] = (eps, set(u))
    rows, arrays = [], {}
    for r in j['rows']:
        if r.get('mode') != 'sg':
            continue
        ep, mf = r['ep'], r['multifloor']
        gt = A[f'{ep}_gt'].astype(np.float64)
        row = {'ep': ep, 'multifloor': mf}
        if r.get('found'):
            T = A[f'{ep}_sg_ref'].astype(np.float64)
            row['nav3d'] = measure(sc, md, T, gt)
        else:
            row['nav3d'] = None
        for k in (('2d_plan', '2d_gt') if mf else ('2d',)):
            eps, usable = src[k]
            e = eps.get(ep)
            if e is None or not e.get('trajectory'):
                row[k], row[f'{k}_own_usable'] = None, False
                continue
            xy = np.asarray(e['trajectory'], np.float64)
            zz = np.asarray(e['trajectory_z'], np.float64) if 'trajectory_z' in e else np.full(len(xy), float(e['floor_z']))
            X = np.column_stack([xy, zz])
            row[k] = measure(sc, md, X, gt)
            row[k]['start_err_m'] = float(np.linalg.norm(xy[0] - gt[0, :2]))   # 같은 에피소드인지 확인용
            row[f'{k}_own_usable'] = ep in usable
            arrays[f'{ep}_{k}'] = X.astype(np.float32)
        rows.append(row)
    d = Path(args.out_dir) / 'scenes'
    d.mkdir(parents=True, exist_ok=True)
    json.dump({'scene': s, 'rows': rows}, open(d / f'{s}.json', 'w'), indent=1)
    np.savez_compressed(d / f'{s}_traj.npz', **arrays)
    print(f'[03c] {s}: {len(rows)} ep')


def aggregate(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    out = Path(args.out_dir)
    rows = []
    for f in sorted((out / 'scenes').glob('*.json')):
        jj = json.load(open(f))
        rows += [dict(r, scene=jj['scene']) for r in jj['rows']]
    summ = {}
    bad_match = [(r['scene'], r['ep'], k) for r in rows for k in ('2d', '2d_plan', '2d_gt')
                 if r.get(k) and r[k]['start_err_m'] > 0.3]
    summ['episode match check (2D start vs GT start > 0.3 m)'] = len(bad_match)
    for grp, mf, methods in (('flat', False, ('nav3d', '2d')), ('multifloor', True, ('nav3d', '2d_plan', '2d_gt'))):
        G = [r for r in rows if r['multifloor'] == mf]
        cat = {}
        for k in methods:
            c = {'n': len(G)}
            for name in ('fail', 'penetrate', 'lt_rb', 'usable'):
                c[name] = sum(category(r.get(k)) == name for r in G)
            c['offmesh'] = sum(r.get(k) is not None and r[k]['on_frac'] < 1.0 for r in G)
            c['offmesh > 0.10 m'] = sum(r.get(k) is not None and r[k]['off_max_m'] > 0.10 for r in G)
            if k != 'nav3d':
                c['own_map_usable'] = sum(r.get(f'{k}_own_usable', False) for r in G)
                c['own_usable_but_mesh_says_not'] = sum(r.get(f'{k}_own_usable', False) and category(r.get(k)) != 'usable' for r in G)
            cat[k] = c
        both = [r for r in G if all(r.get(k) is not None for k in methods)]
        q = {'n (all methods generated)': len(both)}
        for k in methods:
            for m in ('same_route', 'len_ratio', 'turn_per_m_deg', 'mesh_clear_min_m', 'margin_p5_m', 'on_frac'):
                v = [r[k][m] for r in both]
                q[f'{k} {m}'] = (int(sum(v)) if m == 'same_route' else float(np.median(v))) if v else None
        summ[grp] = {'categories': cat, 'quality (median, same episodes)': q}
    json.dump(summ, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1, ensure_ascii=False))

    # 그림 1: 같은 에피소드 mesh 거리 · 회전 · Fréchet (가로 2D, 세로 nav3d)
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    for mf, k2, col, nm in ((False, '2d', 'tab:red', 'flat: 2D'), (True, '2d_plan', 'tab:blue', 'multifloor: 2D plan')):
        P = [r for r in rows if r['multifloor'] == mf and r.get(k2) and r.get('nav3d')]
        for ax, m in zip(axs, ('mesh_clear_min_m', 'turn_per_m_deg', 'frechet_m')):
            ax.scatter([r[k2][m] for r in P], [r['nav3d'][m] for r in P], s=6, alpha=0.5, color=col, label=nm)
            ax.set_xlabel(f'2D {m}'); ax.set_ylabel(f'nav3d {m}')
    for ax, t in zip(axs, ('min 3D distance to mesh (higher = safer)', 'turn deg/m', 'Frechet to GT (lower = closer)')):
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], 'k:', lw=0.8); ax.set_title(t); ax.legend(fontsize=8)
    axs[0].axhline(R_B, color='gray', lw=0.6); axs[0].axvline(R_B, color='gray', lw=0.6)
    fig.savefig(out / 'scatter_2d_vs_3d.png', dpi=80, bbox_inches='tight'); plt.close(fig)

    # 그림 2: 예시 — 평지 (2D 만 실패 · 둘 다 사용 가능) + 층 이동 (2D plan 실패 · 둘 다 사용 가능), 평면도 + 높이 단면
    def pick(mf, cond, n):
        out_, seen = [], set()
        for r in rows:
            if r['multifloor'] == mf and r['scene'] not in seen and cond(r):
                out_.append(r); seen.add(r['scene'])
            if len(out_) >= n:
                break
        return out_
    k_mf = '2d_plan'
    picks = (pick(False, lambda r: category(r.get('2d')) != 'usable' and category(r.get('nav3d')) == 'usable', 2) +
             pick(False, lambda r: category(r.get('nav3d')) != 'usable' and category(r.get('2d')) == 'usable', 1) +
             pick(False, lambda r: category(r.get('2d')) == 'usable' and category(r.get('nav3d')) == 'usable' and r['2d']['frechet_m'] > 1.5, 1) +
             pick(True, lambda r: category(r.get(k_mf)) != 'usable' and category(r.get('nav3d')) == 'usable' and r.get(k_mf), 2) +
             pick(True, lambda r: category(r.get(k_mf)) == 'usable' and category(r.get('nav3d')) == 'usable', 2))
    fig, axs = plt.subplots(len(picks), 2, figsize=(18, 6.5 * len(picks)), squeeze=False, gridspec_kw={'width_ratios': [1.1, 1]})
    cache = {}
    for row, r in enumerate(picks):
        s = r['scene']
        if s not in cache:
            pf = nu.load_pathfinder(f'{args.scene_root}/{s}/{s}.navmesh')
            cache[s] = (npth.NavScene(pf), np.load(Path(args.in_dir) / 'scenes' / f'{s}_traj.npz'),
                        np.load(out / 'scenes' / f'{s}_traj.npz'))
        sc, A, B = cache[s]
        ep = r['ep']
        G = A[f'{ep}_gt']
        T = A[f'{ep}_sg_ref'] if r.get('nav3d') else None
        k2 = k_mf if r['multifloor'] else '2d'
        D2 = B[f'{ep}_{k2}'] if f'{ep}_{k2}' in B.files else None
        allp = np.vstack([p for p in (G, T, D2) if p is not None])
        lo, hi = allp[:, :2].min(0) - 1.5, allp[:, :2].max(0) + 1.5
        zr = (allp[:, 2].min() - 0.6, allp[:, 2].max() + 0.6)
        c = sc.tris.mean(1)
        m = (c[:, 2] > zr[0]) & (c[:, 2] < zr[1])
        ax = axs[row][0]
        ax.add_collection(PolyCollection(sc.tris[m][:, :, :2], facecolors=np.where(sc.stairs_tri[m], 'salmon', '0.85'), edgecolors='none'))
        ax.plot(G[:, 0], G[:, 1], color='orange', lw=4, alpha=0.5, label='GT')
        lab2 = f"2D ({'plan' if r['multifloor'] else 'occupancy'}): {category(r.get(k2))}"
        lab3 = f"nav3d: {category(r.get('nav3d'))}"
        if D2 is not None:
            ax.plot(D2[:, 0], D2[:, 1], 'g-', lw=1.4, label=lab2)
        if T is not None:
            ax.plot(T[:, 0], T[:, 1], 'b-', lw=1.4, label=lab3)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal'); ax.legend(fontsize=7)

        def info(k):
            x = r.get(k)
            return 'fail' if x is None else f"mesh min {x['mesh_clear_min_m']:.2f} m, on {x['on_frac']:.3f}, Frechet {x['frechet_m']:.1f}"
        ax.set_title(f"{s} ep{ep} ({'multifloor' if r['multifloor'] else 'flat'})\n2D: {info(k2)}\nnav3d: {info('nav3d')}", fontsize=8)
        ax = axs[row][1]
        for X, col, nm in ((G, 'orange', 'GT'), (D2, 'g', '2D'), (T, 'b', 'nav3d')):
            if X is None:
                continue
            sx = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1))]
            ax.plot(sx, X[:, 2], color=col, lw=1.2, label=f'{nm} floor z')
        ax.set_xlabel('path length [m]'); ax.set_ylabel('floor z [m]'); ax.legend(fontsize=7)
        ax.set_title('height profile (salmon on left = stairs-like navmesh)', fontsize=8)
    fig.savefig(out / 'examples_2d_vs_3d.png', dpi=70, bbox_inches='tight'); plt.close(fig)

    def tbl(d):
        cols = list(d)
        keys = list(dict.fromkeys(k for c in cols for k in d[c]))
        h = '<tr><th></th>' + ''.join(f'<th>{c}</th>' for c in cols) + '</tr>'
        h += ''.join('<tr><td>' + k + '</td>' + ''.join(f'<td>{d[c].get(k, "")}</td>' for c in cols) + '</tr>' for k in keys)
        return f'<table>{h}</table>'
    body = ''.join(f"<h2>{g}</h2>{tbl(summ[g]['categories'])}<pre>{json.dumps(summ[g]['quality (median, same episodes)'], indent=1)}</pre>"
                   for g in ('flat', 'multifloor'))
    (out / 'report.html').write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>nav3d vs 2D</title>
<style>body{{font-family:sans-serif;max-width:1500px;margin:auto;padding:16px}}td,th{{border:1px solid #ccc;padding:2px 8px}}img{{max-width:100%}}</style>
</head><body><h1>03c — 2D occupancy vs nav3d (같은 에피소드 · 같은 자)</h1>{body}
<img src="scatter_2d_vs_3d.png"><img src="examples_2d_vs_3d.png"></body></html>""")
    print(f'  -> {out}/report.html')


def main():
    args = parse()
    if args.aggregate:
        aggregate(args)
    else:
        run_scene(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
