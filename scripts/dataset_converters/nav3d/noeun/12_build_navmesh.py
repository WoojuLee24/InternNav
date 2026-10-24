"""01 — 씬 mesh 로 navmesh 를 만들고, **층(storey)·계단(연결부)** 을 찾고, 층 사이 경로가 되는지 검증한다.

    mesh(Z-up) -> glb -> habitat recompute_navmesh -> <out>/<scene>.navmesh
                                                 -> 평면(level) -> 층(storey) / 섬 / 연결부(계단·에스컬레이터·경사로)
                                                 -> 검증 V1~V5 + 그림 + report.html + verify.json

## 용어
- **평면(level)**: navmesh 의 거의 수평인 면적이 몰린 높이. 바닥이 완만히 기울면 한 층이 평면 2개로 쪼개진다
  (노은역 승강장·대합실이 끝에서 끝까지 약 0.5 m 기운다).
- **층(storey)**: 높이 차가 `--storey_gap` 미만인 평면끼리 묶은 것 (B2·B1·지상).
- **연결부(connector)**: 어느 평면에도 속하지 않는 navmesh 삼각형 덩어리 중 **서로 다른 두 층에 닿는 것**.
  같은 층 안의 완만한 경사 덩어리는 `intra` 로 따로 센다.
- **계단 판별**: 연결부 위 점에서 원본 mesh 를 아래로 쏴서 표면 높이의 평면 맞춤 잔차(`stepness_rms_m`)를 잰다.
  톱니(디딤판·챌판)면 계단, 매끈하면 경사로·에스컬레이터 덮개.

## 검증 (GT 없는 씬도 돌아가게 — GT 비교는 vln_ce 단계에서)
| | 묻는 것 | 기준 |
|---|---|---|
| V1 좌표 | navmesh 꼭짓점이 원본 mesh 표면 위에 있나 (좌표 규약) | 중앙값 < 0.05 m, p95 < 0.25 m. 틀린 규약과 대비 |
| V2 층 | 각 층 높이에 mesh 의 넓은 수평면(바닥)이 있나 | 층의 모든 평면이 0.3 m 안에 mesh 바닥 봉우리를 가짐 |
| V3 2D 대조 | 같은 층에서 navmesh 통행 영역 vs gs_vlnpe 2D 지도(`nav_mask_ref`) | 수치 보고. 2D 만 통행인 곳을 "mesh 바닥 있음 / 없음" 으로 나눔 |
| V4 연결 | 층들이 이어지나, 연결부 몇 개·계단인지 | 보고 |
| V5 층간 경로 | 서로 다른 층의 무작위 두 점(같은 섬) `find_path` | 성공 100 %, 수평으로 navmesh 위 ≥ 99 %. mesh 3D 여유는 보고 |

사용 (internnav-habitat-navmesh 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/12_build_navmesh.py --scene noeun_station
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
sys.path.insert(0, str(HERE))
import nav3d_utils as nu  # noqa: E402
from nav3d_scenes import SCENES, sample_storey_points  # noqa: E402,F401

STEP_RMS_M = 0.05          # 이 이상이면 계단 (실측: 계단 0.13~0.16, 매끈한 경사 ≤ 0.023)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True, help=f'{list(SCENES)} 또는 --mesh 와 함께 임의 이름')
    ap.add_argument('--mesh', default=None, help='Z-up mesh (obj/ply/glb). 생략 시 SCENES 표')
    ap.add_argument('--compare_2d', default=None, help='gs_vlnpe 02 esdf npz (V3). 생략 시 SCENES 표, "none" 이면 끔')
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/noeun/12_build_navmesh')
    ap.add_argument('--no_blockers', action='store_true', help='SCENES 의 blockers(유리 등 수동 벽)를 넣지 않음')
    # navmesh 설정 — 노은역 로봇: r_b 0.25, 실제 키 0.9 m 이지만 **계획 기준 키는 1.8 m**(사용자 결정 2026-10-14), 턱 0.20 m 통과
    ap.add_argument('--agent_radius', type=float, default=0.25)
    ap.add_argument('--agent_height', type=float, default=1.8, help='계획 기준 키 [m] — 머리 위가 이만큼 비어야 걷는다 (실제 로봇 0.9)')
    ap.add_argument('--agent_max_climb', type=float, default=0.20, help='넘을 수 있는 턱 [m] (사용자 확인 0.20)')
    ap.add_argument('--agent_max_slope', type=float, default=45.0)
    ap.add_argument('--cell_size', type=float, default=0.05)
    ap.add_argument('--cell_height', type=float, default=0.05, help='habitat 기본 0.2 — 계단 높이를 살리려 0.05')
    # 분석
    ap.add_argument('--min_level_area', type=float, default=10.0, help='평면으로 볼 최소 수평 면적 [m2]')
    ap.add_argument('--level_tol', type=float, default=0.3, help='삼각형이 평면에 속하는 높이 허용 [m]')
    ap.add_argument('--storey_gap', type=float, default=1.5, help='이보다 가까운 평면은 한 층 [m]')
    ap.add_argument('--n_pairs', type=int, default=200, help='V5 층간 경로 쌍 수')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


# ---------------------------------------------------------------------------
group_storeys = nu.group_storeys   # 03 도 같은 정의를 쓰도록 공용으로 옮김


def analyse_levels(tris, mesh, args):
    a, n = nu.tri_area_normal(tris)
    cz = tris[:, :, 2].mean(1)
    flat = n[:, 2] > np.cos(np.deg2rad(nu.UP_COS_DEG))
    levels = nu.find_levels(cz[flat], a[flat], min_area_m2=args.min_level_area)
    lv = nu.assign_level(cz, levels, args.level_tol)
    lv[~flat] = -1
    lv2st, storeys = group_storeys(levels, args.storey_gap)
    # mesh 의 위를 향한 수평면(바닥 후보) — V2 대조용. 지붕·선반 윗면도 섞인다.
    fa, fn = mesh.area_faces, mesh.face_normals
    up = fn[:, 2] > np.cos(np.deg2rad(nu.UP_COS_DEG))
    mz = mesh.triangles_center[up, 2]
    mesh_floors = nu.find_levels(mz, fa[up], min_area_m2=args.min_level_area)
    for k, L in enumerate(levels):
        d = [abs(L['z'] - m['z']) for m in mesh_floors]
        L['mesh_floor_dz_m'] = float(min(d)) if d else None
        L['storey'] = int(lv2st[k])
    return levels, storeys, lv, lv2st, a, n, cz, mesh_floors, (mz, fa[up], mesh.triangles[up])


def analyse_connectors(tris, lv, lv2st, a, n, isl, levels, storeys, mesh, args):
    comp = nu.tri_components(tris, lv < 0)
    conns, intra = [], 0
    Lz = np.array([L['z'] for L in levels]) if levels else np.zeros(0)
    for c in np.unique(comp[comp >= 0]):
        m = comp == c
        t = tris[m]
        zmin, zmax = float(t[:, :, 2].min()), float(t[:, :, 2].max())
        touch_lv = sorted({int(k) for k, z in enumerate(Lz) if zmin - 0.35 <= z <= zmax + 0.35})
        touch_st = sorted({int(lv2st[k]) for k in touch_lv})
        if len(touch_st) < 2:
            intra += int(len(touch_lv) >= 2)
            continue
        slope = float(np.degrees(np.arccos(np.clip(np.average(n[m, 2], weights=a[m]), -1, 1))))
        st = nu.stepness(mesh, t, seed=args.seed + int(c))
        rms = st['stepness_rms_m']
        if rms is None:
            kind = 'unknown'
        elif rms >= STEP_RMS_M:
            kind = 'stairs'
        else:
            kind = 'smooth'
        xy = t[:, :, :2].reshape(-1, 2)
        conns.append({'id': len(conns), 'comp': int(c), 'storeys': touch_st, 'levels': touch_lv, 'z_range': [zmin, zmax],
                      'area_m2': float(a[m].sum()), 'n_tris': int(m.sum()), 'slope_deg': slope,
                      'center_xy': xy.mean(0).tolist(), 'bbox_xy': [xy.min(0).tolist(), xy.max(0).tolist()],
                      'islands': sorted({int(i) for i in isl[m]}), 'kind': kind, **st})
    return comp, conns, intra


def v1_frame(pf, md, n_max=20000, seed=0):
    v = np.asarray(pf.build_navmesh_vertices(-1), dtype=np.float64).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    v = v[rng.choice(len(v), size=min(n_max, len(v)), replace=False)]
    d_ok = md(nu.hab_to_ours(v), exact_below=0.5)
    d_raw = md(v.copy(), exact_below=0.0)                # 틀린 규약: habitat 좌표를 그대로 우리 좌표로
    q = lambda d: {'median_m': float(np.median(d)), 'p95_m': float(np.percentile(d, 95)), 'max_m': float(d.max())}
    r = {'n': int(len(v)), 'ours': q(d_ok), 'wrong_identity': q(d_raw)}
    r['passed'] = bool(r['ours']['median_m'] < 0.05 and r['ours']['p95_m'] < 0.25)
    return r


def v3_compare_2d(npz_path, tris, lv, lv2st, levels, storeys, mesh_up):
    d = np.load(npz_path, allow_pickle=True)
    fz, mask2d, org, c = float(d['floor_z']), d['nav_mask_ref'], d['origin'], float(d['voxel_size'])
    if not storeys:
        return {'skipped': '층 없음'}
    s = int(np.argmin([min(abs(z - fz) for z in S['z_range']) for S in storeys]))
    if min(abs(z - fz) for z in storeys[s]['z_range']) > 0.6:
        return {'skipped': f'2D floor_z {fz} 근처 층 없음'}
    in_storey = (lv >= 0) & np.isin(lv, storeys[s]['levels'])
    nav = nu.rasterize_tris_xy(tris[in_storey], org[:2], c, mask2d.shape)
    # mesh 에 이 층 높이의 바닥(위향 수평면)이 있는 칸
    zlo, zhi = storeys[s]['z_range'][0] - 0.3, storeys[s]['z_range'][1] + 0.3
    ft = mesh_up[2][(mesh_up[0] > zlo) & (mesh_up[0] < zhi)]
    floor = nu.rasterize_tris_xy(ft, org[:2], c, mask2d.shape)
    c2 = c * c
    only2d = ~nav & mask2d
    return {'storey': s, 'storey_z_range': storeys[s]['z_range'], 'floor_z_2d': fz, 'r_b_2d': float(d['r_b']),
            'ref_h_b_2d': float(d['ref_h_b']),
            'iou': float((nav & mask2d).sum() / max((nav | mask2d).sum(), 1)),
            'iou_where_floor': float((nav & mask2d & floor).sum() / max(((nav | mask2d) & floor).sum(), 1)),
            'both_m2': float((nav & mask2d).sum() * c2), 'nav_only_m2': float((nav & ~mask2d).sum() * c2),
            '2d_only_m2': float(only2d.sum() * c2),
            '2d_only_no_floor_m2': float((only2d & ~floor).sum() * c2),
            '2d_only_with_floor_m2': float((only2d & floor).sum() * c2),
            'nav_mask': nav, 'mask2d': mask2d, 'floor': floor, 'origin': org[:2].tolist(), 'cell': c}


def project_path(surf, D):
    """촘촘히 찍은 경로 점을 navmesh 표면에 수직 투영(연속성 추적) — 꼭짓점 사이 직선 z 는 계단 구간에서 허공을 지난다.
    투영할 삼각형이 없는 점(수평으로 navmesh 밖)은 원래 z 를 두고 off=True."""
    z = surf.track(D[:, :2], z0=D[0, 2])
    off = ~np.isfinite(z)
    S = D.copy()
    S[~off, 2] = z[~off]
    return S, off


def v5_paths(pf, levels, lv2st, comp_ids, tris, md, args):
    surf = nu.NavSurface(tris)
    import habitat_sim
    rng = np.random.default_rng(args.seed)
    pts = sample_storey_points(pf, levels, lv2st, seed=args.seed)
    # 2개 이상 층을 가진 섬마다, 그 섬의 층 쌍에 경로 쌍을 고르게 나눈다
    by_isl = {}
    for (st, il), v in pts.items():
        by_isl.setdefault(il, {})[st] = v
    combos = [(il, i, j) for il, d in sorted(by_isl.items()) for i in sorted(d) for j in sorted(d) if i < j]
    pairs = []
    for n in range(args.n_pairs if combos else 0):
        il, i, j = combos[n % len(combos)]
        if rng.random() < 0.5:
            i, j = j, i
        A, B = by_isl[il][i], by_isl[il][j]
        pairs.append((i, j, A[rng.integers(len(A))], B[rng.integers(len(B))]))
    tc = tris.mean(1)
    from scipy.spatial import cKDTree
    tk = cKDTree(tc)
    rows, examples = [], []
    for i, j, a, b in pairs:
        sp = habitat_sim.ShortestPath()
        sp.requested_start, sp.requested_end = a, b
        found = bool(pf.find_path(sp))
        row = {'from_storey': i, 'to_storey': j, 'found': found}
        if found:
            P = nu.hab_to_ours(np.asarray(sp.points))
            S, off = project_path(surf, nu.densify(P, 0.05))
            # 몸 = 반지름 r_b 구 두 개(중심 0.5 m, 키 − r_b) — 위 구 꼭대기가 키. 옛 자(키 − 0.05)는 머리 위가 키 + r_b − 0.05 까지
            # 비어야 통과해 너무 엄격했다(노은역 1.8 m 에서 실측, 01h)
            clr = np.minimum(md(S + [0, 0, 0.5]), md(S + [0, 0, args.agent_height - args.agent_radius]))
            k = int(np.argmin(clr))
            on_conn = bool(np.isin(comp_ids[tk.query(S[k])[1]], list(args._conn_comps)))
            row.update(geodesic_m=float(sp.geodesic_distance), euclid_m=float(np.linalg.norm(P[-1] - P[0])),
                       on_navmesh_frac=float((~off).mean()), clearance_min_m=float(clr[k]),
                       clearance_min_xyz=S[k].round(2).tolist(), clearance_min_on_connector=on_conn,
                       clearance_ok=bool(clr[k] >= args.agent_radius - 0.05))
            if len(examples) < 6 and i != j:
                examples.append({'P': P, 'S': S, 'clr': clr, 'i': i, 'j': j})
        rows.append(row)
    f = [r for r in rows if r['found']]
    bad = [r for r in f if not r['clearance_ok']]
    summ = {'n_pairs': len(rows), 'found': len(f),
            'on_navmesh_frac_min': float(min((r['on_navmesh_frac'] for r in f), default=np.nan)),
            'clearance_ok': len(f) - len(bad),
            'clearance_min_median_m': float(np.median([r['clearance_min_m'] for r in f])) if f else None,
            'clearance_bad_on_connector': int(sum(r['clearance_min_on_connector'] for r in bad)),
            'clearance_bad_xyz': sorted({tuple(r['clearance_min_xyz']) for r in bad})[:20],
            'geodesic_over_euclid_median': float(np.median([r['geodesic_m'] / max(r['euclid_m'], 1e-6) for r in f]))
            if f else None,
            'storey_pairs': {f"{a}->{b}": sum(1 for r in rows if (r['from_storey'], r['to_storey']) == (a, b))
                             for a, b in sorted({(r['from_storey'], r['to_storey']) for r in rows})}}
    summ['passed'] = bool(rows and summ['found'] == len(rows) and summ['on_navmesh_frac_min'] >= 0.99)
    return summ, rows, examples


# ---------------------------------------------------------------------------
def figures(out, scene, tris, lv, lv2st, comp, conns, isl, levels, storeys, mesh_up, mesh, v3, examples, args, res_blockers=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    figs = {}
    cz = tris[:, :, 2].mean(1)
    is_conn = np.isin(comp, [c['comp'] for c in conns])
    st_of = np.where(lv >= 0, lv2st[np.maximum(lv, 0)], -1)
    cmap = plt.get_cmap('tab10')
    a, _ = nu.tri_area_normal(tris)

    # (1) 높이 히스토그램
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.arange(np.floor(min(cz.min(), mesh_up[0].min())), np.ceil(max(cz.max(), mesh_up[0].max())) + 0.1, 0.1)
    ax.hist(mesh_up[0], bins=bins, weights=mesh_up[1], alpha=0.4, label='mesh up-facing', color='gray')
    ax.hist(cz, bins=bins, weights=a, alpha=0.7, label='navmesh', color='tab:blue')
    for s, S in enumerate(storeys):
        ax.axvspan(S['z_range'][0] - 0.1, S['z_range'][1] + 0.1, color=cmap(s), alpha=0.2)
        ax.text(S['z'], 1, f'S{s}', color=cmap(s), fontsize=10, weight='bold')
    ax.set_xlabel('z [m]'); ax.set_ylabel('area [m2]'); ax.set_yscale('log'); ax.legend()
    ax.set_title(f'{scene} - area by height (bands = storeys)')
    figs['height_hist'] = f'{scene}_height_hist.png'; fig.savefig(out / figs['height_hist'], dpi=110, bbox_inches='tight'); plt.close(fig)

    # (2) 옆모습
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    mv = mesh.vertices[np.random.default_rng(0).choice(len(mesh.vertices), 60000, replace=False)]
    for ax, (i, nm) in zip(axs, [(0, 'x'), (1, 'y')]):
        ax.scatter(mv[:, i], mv[:, 2], s=0.2, c='lightgray')
        c = tris[:, :, [i, 2]].mean(1)
        for s in range(len(storeys)):
            m = st_of == s
            ax.scatter(c[m, 0], c[m, 1], s=1, color=cmap(s), label=f'S{s}')
        ax.scatter(c[is_conn, 0], c[is_conn, 1], s=2, c='red', label='connector')
        ax.set_xlabel(f'{nm} [m]'); ax.set_ylabel('z [m]'); ax.set_aspect('equal')
    axs[0].legend(loc='upper right', markerscale=5); axs[0].set_title(f'{scene} - side view (gray mesh, color storey, red connector)')
    figs['side'] = f'{scene}_side.png'; fig.savefig(out / figs['side'], dpi=110, bbox_inches='tight'); plt.close(fig)

    # (3) 층별 top-down
    nS = len(storeys)
    fig, axs = plt.subplots(1, max(nS, 1), figsize=(8 * max(nS, 1), 8), squeeze=False)
    icm = plt.get_cmap('tab20')
    for s in range(nS):
        ax = axs[0][s]
        z0, z1 = storeys[s]['z_range']
        w = mesh.vertices[(mesh.vertices[:, 2] > z0 + 0.1) & (mesh.vertices[:, 2] < z1 + args.agent_height)]
        ax.scatter(w[:, 0], w[:, 1], s=0.1, c='0.6')
        m = st_of == s
        ax.add_collection(PolyCollection(tris[m][:, :, :2], facecolors=[icm(i % 20) for i in isl[m]], edgecolors='none', alpha=0.8))
        cs = [c for c in conns if s in c['storeys']]
        for c in cs:
            mm = comp == c['comp']
            ax.add_collection(PolyCollection(tris[mm][:, :, :2], facecolors='red', edgecolors='none'))
            ax.annotate(f"C{c['id']} {c['kind']}\nS{c['storeys']}", c['center_xy'], color='darkred', fontsize=9, weight='bold')
        ax.autoscale(); ax.set_aspect('equal')
        ax.set_title(f"S{s} z={z0:+.2f}~{z1:+.2f} m, {storeys[s]['area_m2']:.0f} m2 (color=island, red=connector)")
    figs['storeys'] = f'{scene}_storeys_topdown.png'; fig.savefig(out / figs['storeys'], dpi=90, bbox_inches='tight'); plt.close(fig)

    # (3b) 층마다 한 장씩 크게 — 색 = 섬(이어진 덩어리), 범례에 섬 면적. 큰 섬(파랑)이 로봇이 다닐 수 있는 본 영역
    from matplotlib.patches import Patch
    isl_area = {int(i): float(a[isl == i].sum()) for i in np.unique(isl)}
    for s_ in range(nS):
        fig, ax = plt.subplots(figsize=(13, 11))
        z0, z1 = storeys[s_]['z_range']
        w = mesh.vertices[(mesh.vertices[:, 2] > z0 + 0.1) & (mesh.vertices[:, 2] < z1 + args.agent_height)]
        ax.scatter(w[:, 0], w[:, 1], s=0.1, c='0.6')
        m = st_of == s_
        ax.add_collection(PolyCollection(tris[m][:, :, :2], facecolors=[icm(i % 20) for i in isl[m]], edgecolors='none', alpha=0.85))
        for c in conns:
            if s_ in c['storeys']:
                mm = comp == c['comp']
                ax.add_collection(PolyCollection(tris[mm][:, :, :2], facecolors='red', edgecolors='none'))
                ax.annotate(f"C{c['id']} {c['kind']} -> S{[x for x in c['storeys'] if x != s_]}", c['center_xy'],
                            color='darkred', fontsize=10, weight='bold')
        for b_ in res_blockers:
            lo, hi = b_['min'], b_['max']
            if hi[2] < z0 - 0.5 or lo[2] > z1 + 0.5:
                continue
            ax.add_patch(matplotlib.patches.Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], color='magenta'))
        here = sorted({int(i) for i in isl[m]}, key=lambda i: -isl_area[i])
        handles = [Patch(color=icm(i % 20), label=f'island {i}: {isl_area[i]:.0f} m2 total') for i in here[:8]]
        handles += [Patch(color='red', label='stairs (to other storey)'), Patch(color='magenta', label='manual blocker (glass)'),
                    Patch(color='0.6', label='mesh (walls, floor+0.1 .. +robot height)')]
        ax.legend(handles=handles, loc='upper right', fontsize=9)
        ax.autoscale(); ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_title(f"S{s_} z={z0:+.2f}~{z1:+.2f} m - navmesh colored by island (connected piece). "
                     f"Same color = robot can walk between them.")
        figs[f'storey{s_}'] = f'{scene}_S{s_}_topdown.png'
        fig.savefig(out / figs[f'storey{s_}'], dpi=90, bbox_inches='tight'); plt.close(fig)

    # (4) 2D 대조
    if v3 and 'iou' in v3:
        nav, m2, fl = v3['nav_mask'], v3['mask2d'], v3['floor']
        img = np.ones(nav.shape + (3,))
        img[~nav & m2 & ~fl] = (0.8, 0.85, 1.0)
        img[~nav & m2 & fl] = (0.2, 0.3, 0.9)
        img[nav & m2] = (0.3, 0.7, 0.3); img[nav & ~m2] = (0.9, 0.3, 0.3)
        fig, ax = plt.subplots(figsize=(10, 10))
        o, c = v3['origin'], v3['cell']
        ax.imshow(img, origin='lower', extent=[o[0], o[0] + nav.shape[1] * c, o[1], o[1] + nav.shape[0] * c])
        ax.set_title(f"S{v3['storey']} navmesh vs 2D nav_mask_ref - IoU {v3['iou']:.3f} (where mesh floor: {v3['iou_where_floor']:.3f})\n"
                     f"green both, red navmesh only {v3['nav_only_m2']:.0f} m2, dark blue 2D only w/ floor {v3['2d_only_with_floor_m2']:.0f} m2,"
                     f" light blue 2D only no floor {v3['2d_only_no_floor_m2']:.0f} m2", fontsize=9)
        figs['compare_2d'] = f'{scene}_compare_2d.png'; fig.savefig(out / figs['compare_2d'], dpi=100, bbox_inches='tight'); plt.close(fig)

    # (5) 층간 경로 예시
    if examples:
        fig, axs = plt.subplots(len(examples), 2, figsize=(15, 3.6 * len(examples)), squeeze=False,
                                gridspec_kw={'width_ratios': [1, 1.6]})
        for r, e in enumerate(examples):
            ax = axs[r][0]
            ax.add_collection(PolyCollection(tris[:, :, :2], facecolors='0.85', edgecolors='none'))
            ax.add_collection(PolyCollection(tris[is_conn][:, :, :2], facecolors='salmon', edgecolors='none'))
            ax.plot(e['S'][:, 0], e['S'][:, 1], 'b-', lw=1.2); ax.plot(*e['P'][0, :2], 'go'); ax.plot(*e['P'][-1, :2], 'r*', ms=10)
            ax.autoscale(); ax.set_aspect('equal'); ax.set_title(f"S{e['i']} -> S{e['j']} (salmon = connector)")
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(e['S'], axis=0), axis=1))]
            ax = axs[r][1]
            ax.plot(s, e['S'][:, 2], label='path z (projected on navmesh)')
            ax2 = ax.twinx(); ax2.plot(s, e['clr'], 'g', lw=0.6, alpha=0.7); ax2.axhline(args.agent_radius, color='g', ls=':')
            ax2.set_ylabel('mesh clearance [m]', color='g')
            ax.set_xlabel('path length [m]'); ax.set_ylabel('z [m]'); ax.legend(loc='upper left', fontsize=8)
        figs['paths'] = f'{scene}_paths.png'; fig.savefig(out / figs['paths'], dpi=100, bbox_inches='tight'); plt.close(fig)

    # (6) 연결부 단면 — 계단 진행 방향(xy 주성분)으로 자른 옆모습: mesh 점 + navmesh + 위에서 쏜 mesh 표면
    if conns:
        P = mesh.sample(600000)
        fig, axs = plt.subplots(len(conns), 1, figsize=(15, 4.2 * len(conns)), squeeze=False)
        for r, c in enumerate(conns):
            ax = axs[r][0]
            xy = tris[comp == c['comp']][:, :, :2].reshape(-1, 2)
            ctr = xy.mean(0)
            d = np.linalg.svd(xy - ctr, full_matrices=False)[2][0]
            e = np.array([-d[1], d[0]])
            s_lo, s_hi = ((xy - ctr) @ d).min() - 6, ((xy - ctr) @ d).max() + 6
            m = (np.abs((P[:, :2] - ctr) @ e) < 0.4) & ((P[:, :2] - ctr) @ d > s_lo) & ((P[:, :2] - ctr) @ d < s_hi)
            ax.scatter((P[m, :2] - ctr) @ d, P[m, 2], s=0.4, c='gray', label='mesh')
            tcen = tris.mean(1)
            tm = (np.abs((tcen[:, :2] - ctr) @ e) < 0.4) & ((tcen[:, :2] - ctr) @ d > s_lo) & ((tcen[:, :2] - ctr) @ d < s_hi)
            for tr in tris[tm]:
                ax.fill((tr[:, :2] - ctr) @ d, tr[:, 2], color='tab:blue', alpha=0.5)
            ss = np.linspace(s_lo, s_hi, 800)
            line = ctr + ss[:, None] * d
            top = nu.surface_z_down(mesh, line, np.full(len(ss), c['z_range'][1] + 2.0), max_drop=c['z_range'][1] - c['z_range'][0] + 4)
            ax.plot(ss, top, 'k-', lw=0.7, label='mesh surface (ray down)')
            ax.set_aspect('equal'); ax.legend(fontsize=8, loc='upper left')
            ax.set_title(f"C{c['id']} {c['kind']} S{c['storeys']} slope {c['slope_deg']:.0f} deg, stepness rms {c['stepness_rms_m']:.3f} m, "
                         f"center ({ctr[0]:.1f},{ctr[1]:.1f}) - section along stair axis, blue = navmesh")
            ax.set_xlabel('distance along stair axis [m]'); ax.set_ylabel('z [m]')
        figs['sections'] = f'{scene}_connector_sections.png'; fig.savefig(out / figs['sections'], dpi=90, bbox_inches='tight'); plt.close(fig)

    # (7) 3D html
    try:
        import plotly.graph_objects as go
        rgb = lambda c: f'rgb({int(255 * c[0])},{int(255 * c[1])},{int(255 * c[2])})'
        names = {s: f"S{s} storey z {storeys[s]['z_range'][0]:+.1f}~{storeys[s]['z_range'][1]:+.1f}" for s in range(len(storeys))}
        groups = [(names[s], (st_of == s) & ~is_conn, rgb(cmap(s))) for s in range(len(storeys))]
        groups += [('stairs/connector (between storeys)', is_conn, 'red'),
                   ('other (slopes inside a storey, small pieces)', (st_of < 0) & ~is_conn, 'gray')]
        data = []
        for nm, m, col in groups:
            if not m.any():
                continue
            V = tris[m].reshape(-1, 3); I = np.arange(m.sum()) * 3
            data.append(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=I, j=I + 1, k=I + 2, color=col, opacity=1.0,
                                  name=nm, showlegend=True))
        sp = mesh.sample(80000)
        data.append(go.Scatter3d(x=sp[:, 0], y=sp[:, 1], z=sp[:, 2], mode='markers',
                                 marker=dict(size=1, color='lightgray'), name='mesh', opacity=0.3))
        for e in examples:
            data.append(go.Scatter3d(x=e['S'][:, 0], y=e['S'][:, 1], z=e['S'][:, 2] + 0.1, mode='lines',
                                     line=dict(width=5), name=f"S{e['i']}->S{e['j']}"))
        f3 = go.Figure(data)
        f3.update_layout(scene=dict(aspectmode='data'), title=f'{scene} navmesh (color storey, red connector)', height=900)
        figs['html3d'] = f'{scene}_navmesh3d.html'
        f3.write_html(out / figs['html3d'], include_plotlyjs=True)
    except Exception as ex:  # noqa: BLE001
        print(f'  [warn] 3D html 실패: {ex}')
    return figs


def write_report(out, scene, res, figs):
    def tbl(rows, cols):
        h = '<tr>' + ''.join(f'<th>{c}</th>' for c in cols) + '</tr>'
        for r in rows:
            h += '<tr>' + ''.join(f'<td>{r.get(c, "")}</td>' for c in cols) + '</tr>'
        return f'<table>{h}</table>'
    f2 = lambda x: f'{x:.3f}' if isinstance(x, float) else x
    st = [{'S': s, 'z_range': f"{S['z_range'][0]:+.2f}~{S['z_range'][1]:+.2f}", 'area_m2': f"{S['area_m2']:.0f}",
           'levels': S['levels']} for s, S in enumerate(res['storeys'])]
    lv = [{'L': k, 'storey': L['storey'], 'z': f2(L['z']), 'area_m2': f"{L['area_m2']:.0f}", 'mesh_floor_dz': f2(L['mesh_floor_dz_m'])}
          for k, L in enumerate(res['levels'])]
    cn = [{'C': c['id'], 'storeys': c['storeys'], 'kind': c['kind'], 'z': f"{c['z_range'][0]:+.2f}~{c['z_range'][1]:+.2f}",
           'area_m2': f"{c['area_m2']:.1f}", 'slope_deg': f"{c['slope_deg']:.1f}",
           'stepness_rms': f2(c['stepness_rms_m']), 'xy': [round(x, 1) for x in c['center_xy']], 'islands': c['islands']}
          for c in res['connectors']]
    isl = [{'island': r['island'], 'area_m2': f"{r['area_m2']:.1f}", 'storeys': r['storeys'],
            'z': f"{r['z_range'][0]:+.2f}~{r['z_range'][1]:+.2f}"} for r in res['islands'][:12]]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>nav3d 01 {scene}</title>
<style>body{{font-family:sans-serif;max-width:1500px;margin:auto;padding:16px}}table{{border-collapse:collapse;margin:8px 0}}
td,th{{border:1px solid #ccc;padding:3px 8px;font-size:13px}}img{{max-width:100%;border:1px solid #ddd}}
.ok{{color:green}}.ng{{color:#c00}}pre{{background:#f6f6f6;padding:8px;white-space:pre-wrap}}</style></head><body>
<h1>nav3d 01 — {scene} navmesh</h1>
<pre>{json.dumps(res['settings'])}\nnavmesh: area {res['navmesh']['area_m2']:.1f} m² · islands {res['navmesh']['num_islands']} · tris {res['navmesh']['n_tris']} · build {res['navmesh']['build_s']:.1f} s</pre>
<h2>판정</h2>{tbl([{'검증': k, '결과': ('<b class=ok>PASS</b>' if v else '<b class=ng>FAIL</b>') if isinstance(v, bool) else v}
                 for k, v in res['gates'].items()], ['검증', '결과'])}
<h2>V1 좌표</h2><pre>{json.dumps(res['v1'], indent=1)}</pre>
<h2>V2 층</h2>{tbl(st, ['S', 'z_range', 'area_m2', 'levels'])}{tbl(lv, ['L', 'storey', 'z', 'area_m2', 'mesh_floor_dz'])}
<img src="{figs.get('height_hist', '')}"><img src="{figs.get('side', '')}">
<h2>V4 연결부 (층 사이: 계단·에스컬레이터·경사로) · 같은 층 안 경사 덩어리 {res['intra_slopes']}개</h2>
{tbl(cn, ['C', 'storeys', 'kind', 'z', 'area_m2', 'slope_deg', 'stepness_rms', 'xy', 'islands'])}
<h3>섬 (큰 순 12개)</h3>{tbl(isl, ['island', 'area_m2', 'storeys', 'z'])}
<img src="{figs.get('storeys', '')}">{''.join(f'<img src="{v}">' for k, v in figs.items() if k.startswith('storey') and k != 'storeys')}<img src="{figs.get('sections', '')}">
<h2>V3 2D 지도 대조</h2><pre>{json.dumps(res['v3'], indent=1)}</pre><img src="{figs.get('compare_2d', '')}">
<h2>V5 층간 경로</h2><pre>{json.dumps(res['v5'], indent=1)}</pre><img src="{figs.get('paths', '')}">
<p>3D: <a href="{figs.get('html3d', '')}">{figs.get('html3d', '')}</a></p>
</body></html>"""
    (out / 'report.html').write_text(html)


def main():
    args = parse()
    spec = SCENES.get(args.scene, {})
    mesh_path = args.mesh or spec['mesh']
    cmp2d = None if args.compare_2d == 'none' else (args.compare_2d or spec.get('compare_2d'))
    out = Path(args.out_dir) / args.scene
    out.mkdir(parents=True, exist_ok=True)
    cfg = {k: getattr(args, k) for k in ['agent_radius', 'agent_height', 'agent_max_climb', 'agent_max_slope',
                                         'cell_size', 'cell_height']}
    print(f'[01_build_navmesh] {args.scene} mesh={mesh_path} cfg={cfg}')

    mesh = nu.load_mesh_zup(mesh_path)
    print(f'  mesh: {len(mesh.vertices)} v / {len(mesh.faces)} f · bounds {np.round(mesh.bounds, 2).tolist()}')
    blockers = [] if args.no_blockers else spec.get('blockers', [])
    if blockers:
        mesh = nu.add_box_blockers(mesh, blockers)
        print(f"  blockers {len(blockers)}개 추가: {[b['name'] for b in blockers]}")
    glb = out / f'{args.scene}.glb'
    nu.export_glb(mesh, str(glb))
    t = time.time()
    pf = nu.build_navmesh(str(glb), cfg, str(out / f'{args.scene}.navmesh'))
    build_s = time.time() - t
    tris, isl = nu.navmesh_triangles_islands(pf)
    isl_chk = nu.check_island_labels(pf, tris, isl, seed=args.seed)
    print(f"  섬 라벨 표본 검사(get_island 일치): {isl_chk['agree']:.3f} -> {'PASS' if isl_chk['passed'] else 'FAIL'}")
    print(f'  navmesh: area {pf.navigable_area:.1f} m2 · islands {pf.num_islands} · tris {len(tris)} · {build_s:.1f} s')

    md = nu.MeshDistance(mesh)
    v1 = v1_frame(pf, md, seed=args.seed)
    print(f"  V1 좌표: median {v1['ours']['median_m']:.3f} p95 {v1['ours']['p95_m']:.3f} m · "
          f"틀린 규약 median {v1['wrong_identity']['median_m']:.2f} m -> {'PASS' if v1['passed'] else 'FAIL'}")

    levels, storeys, lv, lv2st, a, n, cz, mesh_floors, mesh_up = analyse_levels(tris, mesh, args)
    v2_ok = bool(levels) and all(L['mesh_floor_dz_m'] is not None and L['mesh_floor_dz_m'] < 0.3 for L in levels)
    print(f'  V2 층 {len(storeys)}개 (평면 {len(levels)}개): ' +
          ', '.join(f"S{s} {S['z_range'][0]:+.2f}~{S['z_range'][1]:+.2f} ({S['area_m2']:.0f} m2)" for s, S in enumerate(storeys)) +
          f" · 평면↔mesh 바닥 최대 Δ {max(L['mesh_floor_dz_m'] for L in levels):.2f} m -> {'PASS' if v2_ok else 'FAIL'}")
    comp, conns, intra = analyse_connectors(tris, lv, lv2st, a, n, isl, levels, storeys, mesh, args)
    print(f'  V4 층 사이 연결부 {len(conns)}개 (같은 층 안 경사 {intra}개): ' +
          '; '.join(f"C{c['id']} S{c['storeys']} {c['kind']} {c['slope_deg']:.0f}° rms "
                    f"{'-' if c['stepness_rms_m'] is None else round(c['stepness_rms_m'], 3)} @({c['center_xy'][0]:.0f},{c['center_xy'][1]:.0f})"
                    for c in conns))
    isl_rows = []
    for i in range(pf.num_islands):
        m = isl == i
        isl_rows.append({'island': i, 'area_m2': float(a[m].sum()),
                         'storeys': sorted({int(lv2st[k]) for k in lv[m] if k >= 0}),
                         'z_range': [float(tris[m][:, :, 2].min()), float(tris[m][:, :, 2].max())]})
    isl_rows.sort(key=lambda r: -r['area_m2'])
    print(f"  섬: 가장 큰 섬 {isl_rows[0]['area_m2']:.0f} m2 · 층 {isl_rows[0]['storeys']}")

    v3 = v3_compare_2d(cmp2d, tris, lv, lv2st, levels, storeys, mesh_up) if cmp2d else {'skipped': 'compare_2d 없음'}
    if 'iou' in v3:
        print(f"  V3 2D 대조 (S{v3['storey']}): IoU {v3['iou']:.3f} · mesh 바닥 있는 곳만 IoU {v3['iou_where_floor']:.3f} · "
              f"navmesh만 {v3['nav_only_m2']:.0f} m2 · 2D만 {v3['2d_only_m2']:.0f} m2 "
              f"(바닥 없음 {v3['2d_only_no_floor_m2']:.0f} / 바닥 있음 {v3['2d_only_with_floor_m2']:.0f})")
    args._conn_comps = {c['comp'] for c in conns}
    if len(storeys) >= 2:
        v5, v5_rows, examples = v5_paths(pf, levels, lv2st, comp, tris, md, args)
        print(f"  V5 층간 경로: {v5['found']}/{v5['n_pairs']} 성공 · 수평 navmesh 위 최소 {v5['on_navmesh_frac_min']:.3f} · "
              f"mesh 여유 ≥ r_b-0.05: {v5['clearance_ok']}/{v5['found']} (경로별 최소 여유 중앙 {v5['clearance_min_median_m']:.3f} m, "
              f"미달 중 연결부 위 {v5['clearance_bad_on_connector']})")
    else:
        v5, v5_rows, examples = {'skipped': '층 < 2'}, [], []

    figs = figures(out, args.scene, tris, lv, lv2st, comp, conns, isl, levels, storeys, mesh_up, mesh, v3, examples, args, blockers)
    res = {'scene': args.scene, 'mesh': mesh_path, 'settings': cfg, 'blockers': blockers,
           'navmesh': {'area_m2': float(pf.navigable_area), 'num_islands': int(pf.num_islands), 'n_tris': int(len(tris)),
                       'build_s': build_s, 'file': f'{args.scene}.navmesh'},
           'v1': v1, 'levels': levels, 'storeys': storeys, 'mesh_floors': mesh_floors, 'islands': isl_rows,
           'connectors': conns, 'intra_slopes': intra,
           'v3': {k: v for k, v in v3.items() if k not in ('nav_mask', 'mask2d', 'floor')},
           'v5': v5, 'v5_pairs': v5_rows,
           'island_label_check': isl_chk,
           'gates': {'V1 좌표': v1['passed'], '섬 라벨': isl_chk['passed'], 'V2 층↔mesh 바닥': v2_ok, 'V5 층간 경로': v5.get('passed', 'skipped')},
           'figures': figs}
    json.dump(res, open(out / 'verify.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    write_report(out, args.scene, res, figs)
    print(f'  -> {out}/report.html · verify.json · gates {res["gates"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
