"""03s — "계단 같은 곳" 라벨 수정 전/후 비교 (03 의 계단 오탐지 수정 검증·시각화).

수정 전(`stairs_group='unleveled'`): 층에 안 든 삼각형 전부로 덩어리 → 층 사이 높이의 평평한 바닥까지 계단이 됐다.
수정 후(`stairs_group='sloped'`, 기본): 기울어진 삼각형끼리만 덩어리.

| 검사 | 묻는 것 | 기대 |
|---|---|---|
| 계단 면적 | 씬마다 계단 라벨 삼각형 면적 | 줄어듦 (평평한 바닥이 빠짐) |
| 평지 경로 계단 길이 | 03 raw 경로(평지 에피소드) 중 계단 라벨 위 길이 | 줄어듦 (오탐지) |
| 층 이동 높이 덮음 | 층 이동 경로가 오르내린 높이 중 계단 라벨 구간에서 생긴 비율 | 거의 그대로 (진짜 계단은 유지) |

그림: `scenes_before_after.png`(색 = navmesh 높이, 빨간 빗금 = 계단 같은 곳), `examples_before_after.png`, `distribution.png`.
3D: `ply/<scene>_cloud_{before,after}.ply` — 원본 mesh 표면 점 + navmesh 점, intensity 씬 0 · 바닥 0.3 · 계단 1.0
(RGB 도 회색 · 파랑 · 빨강), `ply/<scene>_navmesh_{before,after}.ply` — navmesh 삼각형(같은 intensity·색). Z-up, m.
CloudCompare: 열고 scalar field = intensity 로 색칠.

03 이 저장한 raw 경로(`logs/common/refine_paths/scenes/<S>_traj.npz`)를 다시 쓴다 — 경로 계획은 새로 하지 않는다.

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/refine_stairs_label_check.py --scene noeun_station
    ls data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r | xargs -P 4 -I{} /usr/bin/python scripts/dataset_converters/nav3d/checks/refine_stairs_label_check.py --scene {}
    /usr/bin/python scripts/dataset_converters/nav3d/checks/refine_stairs_label_check.py --aggregate
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'utils'))
sys.path.insert(0, str(HERE))
import nav3d_path as npth  # noqa: E402
import nav3d_utils as nu  # noqa: E402

GROUPS = ('unleveled', 'sloped')      # 수정 전, 수정 후


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', default=None)
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--ply', nargs='*', default=None, help='이 씬들의 3D ply 만 다시 낸다 (aggregate 없이)')
    ap.add_argument('--in_dir', default='scripts/dataset_converters/nav3d/logs/common/refine_paths')
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/checks/refine_stairs_label_check')
    ap.add_argument('--scene_root', default='data/scene_data/mp3d_ce/mp3d')
    ap.add_argument('--n_examples', type=int, default=6, help='그림에 넣을 경로 수 (평지 오탐지 큰 것 + 층 이동)')
    ap.add_argument('--n_scene_figs', type=int, default=6, help='평면도 전/후 그림에 넣을 vln_ce 씬 수 (계단 면적이 가장 많이 준 순) + 노은역')
    return ap.parse_args()


def load_pf(args, s):
    if s == 'noeun_station':
        return nu.load_pathfinder(str(HERE / 'logs' / 'noeun' / '12_build_navmesh' / 'noeun_station' / 'noeun_station.navmesh'))
    return nu.load_pathfinder(f'{args.scene_root}/{s}/{s}.navmesh')


def path_stairs(sc, X):
    """경로 X 의 계단 라벨 위 길이(m)와, 오르내린 높이 중 계단 구간 몫."""
    z, idx = sc.surf.track_idx(X[:, :2], z0=X[0, 2])
    lab = np.where(idx >= 0, sc.stairs_tri[np.maximum(idx, 0)], False)
    seg = np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1)
    dz = np.abs(np.diff(X[:, 2]))
    on = lab[:-1] | lab[1:]
    return float(seg[on].sum()), float(dz[on].sum()), float(dz.sum())


def run_scene(args):
    s = args.scene
    pf = load_pf(args, s)
    sc = {g: npth.NavScene(pf, stairs_group=g) for g in GROUPS}
    a, _ = nu.tri_area_normal(sc['sloped'].tris)
    j = json.load(open(Path(args.in_dir) / 'scenes' / f'{s}.json'))
    A = np.load(Path(args.in_dir) / 'scenes' / f'{s}_traj.npz')
    rows = []
    for r in j['rows']:
        if not r.get('found') or r.get('mode') == 'gt':
            continue
        pre = f"{r['id']}" if j['meta']['kind'] == 'noeun' else f"{r['ep']}_{r['mode']}"
        X = A[f'{pre}_raw'].astype(np.float64)
        row = {'pre': pre, 'multifloor': bool(r.get('multifloor', True)), 'length_m': float(r['raw_length_m'])}
        for g in GROUPS:
            L, dzs, dzt = path_stairs(sc[g], X)
            row[f'{g}_stairs_m'], row[f'{g}_dz_stairs'], row['dz_total'] = L, dzs, dzt
        rows.append(row)
    out = {'scene': s, 'kind': j['meta']['kind'],
           'stairs_area_m2': {g: float(a[sc[g].stairs_tri].sum()) for g in GROUPS}, 'rows': rows}
    d = Path(args.out_dir) / 'scenes'
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f'{s}.json', 'w'), indent=1)
    print(f"[03s] {s}: 계단 면적 {out['stairs_area_m2']['unleveled']:.0f} -> {out['stairs_area_m2']['sloped']:.0f} m² · 경로 {len(rows)}")


def draw_example(ax_pair, args, s, pre, cache):
    from matplotlib.collections import PolyCollection
    if s not in cache:
        pf = load_pf(args, s)
        cache[s] = ({g: npth.NavScene(pf, stairs_group=g) for g in GROUPS},
                    np.load(Path(args.in_dir) / 'scenes' / f'{s}_traj.npz'))
    scs, A = cache[s]
    R = A[f'{pre}_raw'].astype(np.float64)
    ep = pre.split('_')[0]
    G = A[f'{ep}_gt'] if f'{ep}_gt' in A.files else None
    pad = 2.0
    lo, hi = R[:, :2].min(0) - pad, R[:, :2].max(0) + pad
    zr = (R[:, 2].min() - 0.6, R[:, 2].max() + 0.6)
    res = {}
    for ax, g, nm in zip(ax_pair, GROUPS, ('before (unleveled)', 'after (sloped only)')):
        sc = scs[g]
        r = npth.refine(sc, R)
        T = r['traj']
        c = sc.tris.mean(1)
        m = (c[:, 2] > zr[0]) & (c[:, 2] < zr[1]) & (c[:, 0] > lo[0] - 3) & (c[:, 0] < hi[0] + 3) & \
            (c[:, 1] > lo[1] - 3) & (c[:, 1] < hi[1] + 3)
        ax.add_collection(PolyCollection(sc.tris[m][:, :, :2], facecolors=np.where(sc.stairs_tri[m], 'salmon', '0.85'),
                                         edgecolors='w', lw=0.2))
        if G is not None:
            ax.plot(G[:, 0], G[:, 1], color='orange', lw=3, alpha=0.6, label='GT')
        ax.plot(R[:, 0], R[:, 1], 'k-', lw=0.8, label='find_path (raw)')
        ax.plot(T[:, 0], T[:, 1], 'b-', lw=1.4, label='refined')
        st = r['stairs']
        ax.plot(T[st, 0], T[st, 1], 'r.', ms=2, label='refined: stairs part')
        L, _, _ = path_stairs(sc, R)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
        ax.set_title(f'{s} {pre} | {nm}\nstairs on path {L:.1f} m', fontsize=9)
        ax.legend(fontsize=6, loc='best')
        res[g] = T
    return res


INTENSITY = {'scene': 0.0, 'floor': 0.3, 'stairs': 1.0}
RGB = {'scene': (150, 150, 150), 'floor': (60, 120, 230), 'stairs': (230, 30, 30)}


def write_ply(path, xyz, rgb, intensity, faces=None):
    """x y z (float) + red green blue (uchar) + intensity (float) [+ face]. binary. CloudCompare 는 intensity 를 scalar field 로 읽는다."""
    vt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('intensity', '<f4')])
    v = np.empty(len(xyz), vt)
    v['x'], v['y'], v['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v['red'], v['green'], v['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    v['intensity'] = intensity
    head = ['ply', 'format binary_little_endian 1.0', f'element vertex {len(v)}'] + \
           [f'property {t} {n}' for n, t in (('x', 'float'), ('y', 'float'), ('z', 'float'), ('red', 'uchar'),
                                            ('green', 'uchar'), ('blue', 'uchar'), ('intensity', 'float'))]
    if faces is not None:
        head += [f'element face {len(faces)}', 'property list uchar int vertex_indices']
    with open(path, 'wb') as f:
        f.write(('\n'.join(head + ['end_header']) + '\n').encode())
        f.write(v.tobytes())
        if faces is not None:
            ft = np.dtype([('n', 'u1'), ('i', '<i4', (3,))])
            fa = np.empty(len(faces), ft)
            fa['n'], fa['i'] = 3, faces
            f.write(fa.tobytes())


def scene_mesh(s):
    if s == 'noeun_station':
        sys.path.insert(0, str(HERE))
        import nav3d_scenes
        spec = nav3d_scenes.SCENES['noeun_station']
        return nu.add_box_blockers(nu.load_mesh_zup(spec['mesh']), spec.get('blockers', []))
    meta = json.load(open(HERE.parent / 'gs_vlnpe' / 'logs' / 'scene_meta' / f'{s}.json'))
    return nu.load_mesh_zup(meta['mesh_path'])


def export_ply(args, s, d, spacing=0.05, max_scene_pts=3_000_000):
    """씬 하나 -> 수정 전/후 각각
    - `<s>_navmesh_{before,after}.ply`: navmesh 3D 삼각형(꼭짓점 색 + intensity, 바닥 0.3 · 계단 1.0)
    - `<s>_cloud_{before,after}.ply`: 원본 mesh 표면 점(intensity 0, 회색) + navmesh 표면 점(바닥 0.3 파랑 · 계단 1.0 빨강)
    좌표 = 우리 좌표(Z-up, m)."""
    import trimesh
    d.mkdir(parents=True, exist_ok=True)
    mesh = scene_mesh(s)
    n = int(min(max_scene_pts, mesh.area / spacing ** 2))
    sp, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    pf = load_pf(args, s)
    for g, tag in (('unleveled', 'before'), ('sloped', 'after')):
        sc = npth.NavScene(pf, stairs_group=g)
        kind = np.where(sc.stairs_tri, 'stairs', 'floor')
        V = sc.tris.reshape(-1, 3)
        kv = np.repeat(kind, 3)
        write_ply(d / f'{s}_navmesh_{tag}.ply', V, np.array([RGB[k] for k in kv], np.uint8),
                  np.array([INTENSITY[k] for k in kv], np.float32), faces=np.arange(len(V)).reshape(-1, 3))
        nm = trimesh.Trimesh(vertices=V, faces=np.arange(len(V)).reshape(-1, 3), process=False)
        npn = int(max(1000, nm.area / (spacing * 0.6) ** 2))       # navmesh 는 조금 더 촘촘히 (위에 덮여 보이게)
        pn, fi = trimesh.sample.sample_surface(nm, npn, seed=0)
        pn = pn + [0, 0, 0.02]                                     # 바닥 mesh 와 겹쳐 깜빡이지 않게 2 cm 띄움
        kn = kind[fi]
        xyz = np.vstack([sp, pn])
        rgb = np.vstack([np.tile(RGB['scene'], (len(sp), 1)), np.array([RGB[k] for k in kn])]).astype(np.uint8)
        it = np.r_[np.full(len(sp), INTENSITY['scene']), [INTENSITY[k] for k in kn]].astype(np.float32)
        write_ply(d / f'{s}_cloud_{tag}.ply', xyz, rgb, it)
    print(f'  ply {s}: 씬 점 {len(sp):,} + navmesh 점 {npn:,}', flush=True)


def aggregate(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    out = Path(args.out_dir)
    S = [json.load(open(f)) for f in sorted((out / 'scenes').glob('*.json'))]
    rows = [dict(r, scene=j['scene'], kind=j['kind']) for j in S for r in j['rows']]
    summ = {}
    for name, sel in (('vln_ce flat', lambda r: r['kind'] == 'vln_ce' and not r['multifloor']),
                      ('vln_ce multifloor', lambda r: r['kind'] == 'vln_ce' and r['multifloor']),
                      ('noeun', lambda r: r['kind'] == 'noeun')):
        G = [r for r in rows if sel(r)]
        d = {'n': len(G)}
        for g in GROUPS:
            d[f'{g}: paths with stairs'] = sum(r[f'{g}_stairs_m'] > 0 for r in G)
            d[f'{g}: stairs m on path (sum)'] = float(sum(r[f'{g}_stairs_m'] for r in G))
            dzt = sum(r['dz_total'] for r in G)
            d[f'{g}: dz covered by stairs'] = float(sum(r[f'{g}_dz_stairs'] for r in G) / dzt) if dzt > 0 else None
        summ[name] = d
    summ['stairs area m2 (all scenes)'] = {g: float(sum(j['stairs_area_m2'][g] for j in S)) for g in GROUPS}
    summ['scenes area shrunk > 50 %'] = sum(j['stairs_area_m2']['sloped'] < 0.5 * j['stairs_area_m2']['unleveled'] for j in S)
    json.dump(summ, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1, ensure_ascii=False))

    # 1) 분포: 씬별 계단 면적, 경로별 계단 길이(평지/층 이동), 층 이동 높이 덮음
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    ar = np.array([[j['stairs_area_m2'][g] for g in GROUPS] for j in S])
    axs[0].scatter(ar[:, 0], ar[:, 1], s=12)
    for j, (x, y) in zip(S, ar):
        if y < 0.5 * x and x > 30:
            axs[0].annotate(j['scene'][:5], (x, y), fontsize=7)
    mx = ar.max() * 1.05
    axs[0].plot([0, mx], [0, mx], 'k:', lw=0.8)
    axs[0].set_xlabel('stairs-like area before [m2]'); axs[0].set_ylabel('after [m2]'); axs[0].set_title('per scene: stairs-like navmesh area')
    for mf, col, nm in ((False, 'tab:red', 'flat episode'), (True, 'tab:blue', 'multifloor episode')):
        v = np.array([[r['unleveled_stairs_m'], r['sloped_stairs_m']] for r in rows if r['kind'] == 'vln_ce' and r['multifloor'] == mf])
        axs[1].scatter(v[:, 0], v[:, 1], s=6, alpha=0.5, color=col, label=nm)
    v = np.array([[r['unleveled_stairs_m'], r['sloped_stairs_m']] for r in rows if r['kind'] == 'noeun'])
    if len(v):
        axs[1].scatter(v[:, 0], v[:, 1], s=8, alpha=0.6, color='tab:green', label='noeun')
    mx = max(axs[1].get_xlim()[1], axs[1].get_ylim()[1])
    axs[1].plot([0, mx], [0, mx], 'k:', lw=0.8); axs[1].legend(fontsize=8)
    axs[1].set_xlabel('stairs length on path before [m]'); axs[1].set_ylabel('after [m]')
    axs[1].set_title('per path (03 raw): below diagonal = less stairs label')
    mfr = [r for r in rows if r['multifloor'] and r['dz_total'] > 0.5]
    cov = np.array([[r['unleveled_dz_stairs'] / r['dz_total'], r['sloped_dz_stairs'] / r['dz_total']] for r in mfr])
    axs[2].hist(cov[:, 0], bins=20, range=(0, 1), alpha=0.5, label='before')
    axs[2].hist(cov[:, 1], bins=20, range=(0, 1), alpha=0.5, label='after')
    axs[2].set_xlabel('share of height change inside stairs label'); axs[2].set_ylabel('# multifloor paths')
    axs[2].set_title('real stairs kept? (multifloor paths, 1.0 = all climbing on stairs label)'); axs[2].legend()
    fig.savefig(out / 'distribution.png', dpi=80, bbox_inches='tight'); plt.close(fig)

    # 2) 씬 평면도 전/후 (면적이 가장 많이 준 씬 4 + 노은역) — 색 = 높이, 빨간 빗금 = 계단 같은 곳. 같은 씬의 ply 도 낸다.
    order = sorted([j for j in S if j['kind'] == 'vln_ce'],
                   key=lambda j: -(j['stairs_area_m2']['unleveled'] - j['stairs_area_m2']['sloped']))[:args.n_scene_figs]
    order += [j for j in S if j['kind'] == 'noeun']
    cmap = plt.get_cmap('viridis')
    fig, axs = plt.subplots(len(order), 2, figsize=(18, 7.5 * len(order)), squeeze=False)
    for row, j in enumerate(order):
        pf = load_pf(args, j['scene'])
        for ax, g, nm in zip(axs[row], GROUPS, ('before (unleveled)', 'after (sloped)')):
            sc = npth.NavScene(pf, stairs_group=g)
            cz = sc.tris.mean(1)[:, 2]
            zlo, zhi = np.percentile(cz, 1), np.percentile(cz, 99)
            o = np.argsort(cz)                              # 위층이 아래층을 덮도록 높이 순
            t, st = sc.tris[o], sc.stairs_tri[o]
            pc = PolyCollection(t[:, :, :2], array=cz[o], cmap=cmap, edgecolors='none')
            pc.set_clim(zlo, zhi)
            ax.add_collection(pc)
            if st.any():
                ax.add_collection(PolyCollection(t[st][:, :, :2], facecolors='none', edgecolors='red', lw=0.6, hatch='////'))
            ax.autoscale(); ax.set_aspect('equal')
            fig.colorbar(pc, ax=ax, fraction=0.04, label='navmesh height z [m]')
            ax.set_title(f"{j['scene']} | {nm}\ncolor = height · red hatch = stairs-like ({j['stairs_area_m2'][g]:.0f} m2)", fontsize=10)
    fig.savefig(out / 'scenes_before_after.png', dpi=70, bbox_inches='tight'); plt.close(fig)
    for j in order:
        export_ply(args, j['scene'], out / 'ply')

    # 3) 예시 경로: 평지 에피소드 중 계단 길이가 가장 많이 준 것 + 층 이동 2 개(진짜 계단 유지 확인)
    flat = sorted([r for r in rows if r['kind'] == 'vln_ce' and not r['multifloor']],
                  key=lambda r: -(r['unleveled_stairs_m'] - r['sloped_stairs_m']))
    seen, picks = set(), []
    for r in flat:
        if r['scene'] not in seen and len(picks) < args.n_examples - 2:
            picks.append(r); seen.add(r['scene'])
    mf = sorted([r for r in rows if r['kind'] == 'vln_ce' and r['multifloor'] and r['dz_total'] > 2.0],
                key=lambda r: -r['length_m'])
    picks += [r for r in mf if r['scene'] not in seen][:2]
    fig, axs = plt.subplots(len(picks), 2, figsize=(16, 7 * len(picks)), squeeze=False)
    cache = {}
    for row, r in enumerate(picks):
        draw_example(axs[row], args, r['scene'], r['pre'], cache)
    fig.savefig(out / 'examples_before_after.png', dpi=70, bbox_inches='tight'); plt.close(fig)

    def tbl(d):
        h = ''.join(f'<tr><td>{k}</td><td>{v if not isinstance(v, float) else round(v, 3)}</td></tr>' for k, v in d.items())
        return f'<table>{h}</table>'
    body = ''.join(f'<h3>{k}</h3>{tbl(v) if isinstance(v, dict) else v}' for k, v in summ.items())
    (out / 'report.html').write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>nav3d 03s stairs label</title>
<style>body{{font-family:sans-serif;max-width:1500px;margin:auto;padding:16px}}td{{border:1px solid #ccc;padding:2px 8px}}img{{max-width:100%}}</style>
</head><body><h1>03s — 계단 라벨 수정 전/후</h1>{body}
<img src="distribution.png"><img src="scenes_before_after.png"><img src="examples_before_after.png"></body></html>""")
    print(f'  -> {out}/report.html')


def main():
    args = parse()
    if args.ply is not None:
        for s in args.ply:
            export_ply(args, s, Path(args.out_dir) / 'ply')
    elif args.aggregate:
        aggregate(args)
    else:
        run_scene(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
