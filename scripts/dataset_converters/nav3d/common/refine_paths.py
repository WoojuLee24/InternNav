"""03 — find_path 경로를 다듬는다: 바닥은 벽에서 밀어내고 곡선으로, 계단은 폭 가운데로 곧게. 전후를 검증·시각화한다.

다듬는 방법은 `nav3d_path.py` (바닥 = 탄성 밴드 + Chaikin, 계단 = 폭 가운데 + 오프셋 평활).

| 검사 | 묻는 것 | 기준 |
|---|---|---|
| on | 다듬은 경로가 수평으로 navmesh 위인가 | 100 % |
| 경계 여유 | navmesh 경계까지 최소 거리 (양 끝 0.3 m 제외). 벽까지 실제 ≈ 여유 + agent_radius | 전보다 커짐 |
| mesh 여유 | 원본 mesh 까지 3D 거리 (바닥+0.5 m, 바닥+키−0.05 m) — navmesh 와 독립인 두 번째 자 | r_b 미만 비율 |
| 회전량 | °/m (0.25 m 재표본) | 보고 (GT 와 비교) |
| 계단 정면 | 계단 경사 방향과 진행 방향 사이 각 | 보고 |
| GT (vln_ce) | 다듬은 뒤에도 같은 루트(Fréchet < 0.8 m)인가 | 전후 비교 |

모드
- `--scene noeun_station`: 01 의 navmesh. 서로 다른 층 무작위 점 쌍(01 V5 와 같은 표본) start→goal.
- `--dataset vln_ce --scene <S>`: 공식 navmesh. 에피소드마다 (wp) reference_path 경유점을 find_path 로 이은 경로,
  (sg) GT 시작→끝 find_path. R2R 은 경로 하나에 지시문 3개라 `trajectory_id` 로 중복을 뺀다.
  `--max_flat` 로 씬당 평지 에피소드 수를 줄인다(층 이동 에피소드는 전부).

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/common/refine_paths.py --scene noeun_station
    ls data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r | xargs -P 4 -I{} /usr/bin/python scripts/dataset_converters/nav3d/common/refine_paths.py --dataset vln_ce --scene {}
    /usr/bin/python scripts/dataset_converters/nav3d/common/refine_paths.py --aggregate
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'utils'))
GS = HERE.parent / 'gs_vlnpe'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GS))
import nav3d_path as npth  # noqa: E402
import nav3d_utils as nu  # noqa: E402
import nav3d_scenes as scenes  # noqa: E402
from path_quality import DOOR_WIDTH_M, gt_metrics, stairs_points, one  # noqa: E402,F401

MULTIFLOOR_DZ_M = 0.5


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', default=None)
    ap.add_argument('--dataset', default=None, choices=[None, 'vln_ce'])
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/common/refine_paths')
    ap.add_argument('--n_pairs', type=int, default=100, help='노은역 층간 경로 수')
    ap.add_argument('--max_flat', type=int, default=10, help='vln_ce 씬당 평지 에피소드(중복 제거 후) 수, 0 = 전부')
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default=None, help='원본 R2R json.gz 폴더. None = vlnce_gt.DEFAULT_RAW_ROOT (mini, 이전 동작). full: data/InternData-N1/vln_ce/raw_data/r2r')
    ap.add_argument('--scene_root', default='data/scene_data/mp3d_ce/mp3d')
    ap.add_argument('--mesh_root', default=None,
                    help='vln_ce 씬 obj mesh 루트 (<root>/<씬>/matterport_mesh/*/*.obj, geometry_utils.find_scene_mesh). '
                         'None = gs_vlnpe/logs/scene_meta/<씬>.json 의 mesh_path (이전 동작 — 그 파일을 만드는 gs_vlnpe 01 은 pxr 이 필요). '
                         '예: data/InternData-N1-v0.5-mini/scene_data/mp3d_n1 (이 PC 61 씬에서 scene_meta 와 같은 파일)')
    ap.add_argument('--split', default='train', choices=['train', 'val_unseen'],
                    help='vln_ce. train = 릴리스 parquet GT (이전 동작). val_unseen = 릴리스 렌더가 없어 원본 json 만 쓴다 — '
                         'GT = reference_path 경유점을 find_path 로 이은 경로(= wp 다듬기 전), ep 키 = trajectory_id. --out_dir 를 split 마다 따로')
    ap.add_argument('--rig', default='125cm_30deg')
    ap.add_argument('--navmesh_dir', default=None,
                    help='vln_ce 씬 navmesh 를 이 폴더의 <scene>.navmesh 로 (예: vln_pe H1 반지름으로 다시 만든 것, build_navmesh_h1.py). '
                         'None = 공식 navmesh (vln_ce 로봇 반지름 0.10 m, 이전 동작). r_b 는 navmesh 의 agent_radius 를 따른다')
    ap.add_argument('--push_margin', type=float, default=npth.RefineCfg.push_margin_m)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stairs_mode', default='split', choices=['split', 'none'],
                    help='split = 계단은 폭 가운데로 곧게, 바닥은 곡선 (기본) / none = 계단 구별 없이 전부 바닥처럼 (비교용)')
    return ap.parse_args()


def run_noeun(args, cfg):
    spec = scenes.SCENES['noeun_station']
    d01 = HERE / 'logs' / 'noeun' / '12_build_navmesh' / 'noeun_station'
    pf = nu.load_pathfinder(str(d01 / 'noeun_station.navmesh'))
    sc = npth.NavScene(pf)
    sc_ref = npth.NavScene(pf, stairs_group='none') if args.stairs_mode == 'none' else sc
    mesh = nu.add_box_blockers(nu.load_mesh_zup(spec['mesh']), spec.get('blockers', []))
    md = nu.MeshDistance(mesh)
    pts = scenes.sample_storey_points(pf, sc.levels, sc.lv2st, seed=args.seed)
    by_isl = {}
    for (st, il), v in pts.items():
        by_isl.setdefault(il, {})[st] = v
    combos = [(il, i, j) for il, d in sorted(by_isl.items()) for i in sorted(d) for j in sorted(d) if i < j]
    rng = np.random.default_rng(args.seed)
    rows, arrays = [], {}
    for n in range(args.n_pairs):
        il, i, j = combos[n % len(combos)]
        A, B = by_isl[il][i], by_isl[il][j]
        a, b = nu.hab_to_ours(A[rng.integers(len(A))]), nu.hab_to_ours(B[rng.integers(len(B))])
        P = sc.find_path(a, b)
        if P is None:
            rows.append({'id': n, 'found': False})
            continue
        row, r = one(sc, P, md, sc.agent_radius, cfg, sc_ref=sc_ref)
        rows.append({'id': n, 'found': True, 'from_storey': i, 'to_storey': j, **row})
        arrays[f'{n}_raw'], arrays[f'{n}_ref'] = r['raw'].astype(np.float32), r['traj'].astype(np.float32)
        arrays[f'{n}_ref_st'] = r['stairs']
        print(f"  [{n}] S{i}->S{j} {row['raw_length_m']:.0f} m · 여유 p5 {row['raw_margin_p5_m']:.2f}->{row['ref_margin_p5_m']:.2f} · "
              f"회전 {row['raw_turn_per_m_deg']:.1f}->{row['ref_turn_per_m_deg']:.1f} · {row['sec']:.1f} s", flush=True)
    return {'scene': 'noeun_station', 'kind': 'noeun', 'r_b': sc.agent_radius}, rows, arrays


def vlnce_navmesh(args, s: str) -> str:
    if getattr(args, 'navmesh_dir', None):
        return f'{args.navmesh_dir}/{s}.navmesh'
    return f'{args.scene_root}/{s}/{s}.navmesh'


def run_vlnce(args, cfg):
    import vlnce_gt
    s = args.scene
    pf = nu.load_pathfinder(vlnce_navmesh(args, s))
    sc = npth.NavScene(pf)
    sc_ref = npth.NavScene(pf, stairs_group='none') if args.stairs_mode == 'none' else sc
    if args.mesh_root:
        from geometry_utils import find_scene_mesh
        mesh_path = str(find_scene_mesh(args.mesh_root, s))
    else:
        mesh_path = json.load(open(GS / 'logs' / 'scene_meta' / f'{s}.json'))['mesh_path']
    md = nu.MeshDistance(nu.load_mesh_zup(mesh_path))
    raw_root = args.raw_root or vlnce_gt.DEFAULT_RAW_ROOT

    def wp_path(raw):
        wps = [nu.hab_to_ours(np.asarray(p)) for p in raw['reference_path']]
        legs = [sc.find_path(a, b) for a, b in zip(wps[:-1], wps[1:])]
        return None if any(L is None for L in legs) else np.vstack([legs[0]] + [L[1:] for L in legs[1:]])

    def episodes():
        """(ep 키, raw, GT 바닥 경로 | None). train = 릴리스 ep · 릴리스 자세. val_unseen = trajectory_id · 경유점 find_path."""
        if args.split == 'val_unseen':
            import gzip
            eps = json.load(gzip.open(f'{raw_root}/val_unseen/val_unseen.json.gz'))['episodes']
            for raw in sorted((e for e in eps if Path(e['scene_id']).stem == s), key=lambda e: e['episode_id']):
                yield raw['trajectory_id'], raw, wp_path(raw)
            return
        rawidx = vlnce_gt.raw_index(raw_root)
        h_b = vlnce_gt.vlnce_rig.rig_height_m(args.rig)
        for ep in range(len(glob.glob(f'{args.data_root}/{s}/data/chunk-000/*.parquet'))):
            try:
                g = vlnce_gt.load_gt_episode_vlnce(args.data_root, s, ep, args.rig)
                raw = rawidx[(s, vlnce_gt.episode_instruction(args.data_root, s, ep))]
            except KeyError:
                continue
            yield ep, raw, g['cam_xyz'] - [0.0, 0.0, h_b]

    seen, flat_n, rows, arrays = set(), 0, [], {}
    for ep, raw, gt in episodes():
        if raw['trajectory_id'] in seen:
            continue
        seen.add(raw['trajectory_id'])
        if gt is None:   # val_unseen: 경유점끼리 경로가 없음 — wp · sg 둘 다 못 만든 것으로 센다
            rows += [{'ep': ep, 'mode': m, 'multifloor': False, 'found': False} for m in ('wp', 'sg')]
            continue
        mf = bool(abs(gt[-1, 2] - gt[0, 2]) > MULTIFLOOR_DZ_M)
        if not mf:
            if args.max_flat and flat_n >= args.max_flat:
                continue
            flat_n += 1
        for mode in ('wp', 'sg'):
            if mode == 'wp':
                P = wp_path(raw)
                if P is None:
                    rows.append({'ep': ep, 'mode': mode, 'multifloor': mf, 'found': False})
                    continue
            else:
                P = sc.find_path(gt[0], gt[-1])
                if P is None:
                    rows.append({'ep': ep, 'mode': mode, 'multifloor': mf, 'found': False})
                    continue
            row, r = one(sc, P, md, sc.agent_radius, cfg, gt, sc_ref=sc_ref)
            rows.append({'ep': ep, 'mode': mode, 'multifloor': mf, 'found': True, 'traj_id': raw['trajectory_id'], **row})
            arrays[f'{ep}_{mode}_raw'], arrays[f'{ep}_{mode}_ref'] = r['raw'].astype(np.float32), r['traj'].astype(np.float32)
            arrays[f'{ep}_{mode}_ref_st'] = r['stairs']
        # GT 도 같은 자(navmesh·mesh)로 잰다 — 우리 경로 충돌 수치를 읽을 기준선
        zg, _ = sc.surf.track_idx(gt[:, :2], z0=gt[0, 2])
        rows.append({'ep': ep, 'mode': 'gt', 'multifloor': mf, 'found': True, 'traj_id': raw['trajectory_id'],
                     **{f'ref_{k}': v for k, v in npth.evaluate(sc, gt, None, np.isfinite(zg), md, sc.agent_radius).items()}})
        arrays[f'{ep}_gt'] = gt.astype(np.float32)
    return {'scene': s, 'kind': 'vln_ce', 'r_b': sc.agent_radius}, rows, arrays


# ---------------------------------------------------------------------------
PENETRATE_M = 0.02   # mesh 3D 거리가 이보다 작으면 경로 중심이 벽면에 닿음(=관통으로 센다). mesh 거리는 부호가 없어 벽 안쪽을 따로 못 가린다


def collision_counts(G, who):
    """관통 = navmesh 밖 점이 있음 또는 mesh 3D 거리 < PENETRATE_M. r_b 미달 = 관통 아님 & mesh 3D 거리 < r_b.
    사용 가능 = 둘 다 아님."""
    pen = [r for r in G if r[f'{who}_on_frac'] < 1.0 or r[f'{who}_mesh_clear_min_m'] < PENETRATE_M]
    pen_ids = {id(r) for r in pen}
    rb = [r for r in G if id(r) not in pen_ids and r[f'{who}_mesh_clear_min_m'] < r['r_b']]
    n = len(G)
    return {'n': n, 'penetrate': len(pen), 'penetrate_offmesh': sum(r[f'{who}_on_frac'] < 1.0 for r in G),
            'penetrate_touch': sum(r[f'{who}_mesh_clear_min_m'] < PENETRATE_M for r in G),
            'lt_rb': len(rb), 'usable': n - len(pen) - len(rb),
            'lt_rb_point_frac_median': float(np.median([r[f'{who}_mesh_clear_lt_rb_frac'] for r in rb])) if rb else None}


def aggregate(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    out = Path(args.out_dir)
    groups, gt_groups = {}, {}
    allrows = []
    for f in sorted((out / 'scenes').glob('*.json')):
        j = json.load(open(f))
        for r in j['rows']:
            if not r.get('found'):
                continue
            r['scene'], r['kind'], r['r_b'] = j['meta']['scene'], j['meta']['kind'], j['meta']['r_b']
            key = 'noeun' if r['kind'] == 'noeun' else f"vln_ce_{r['mode']}_{'multifloor' if r['multifloor'] else 'flat'}"
            if r['kind'] == 'vln_ce' and r['mode'] == 'gt':
                gt_groups.setdefault(key, []).append(r)
                continue
            groups.setdefault(key, []).append(r)
            allrows.append(r)

    def med(v):
        v = [x for x in v if x is not None]
        return float(np.median(v)) if v else None

    summ = {}
    for k, G in sorted(groups.items()):
        d = {'n': len(G)}
        for m in ('on_frac', 'margin_min_m', 'margin_p5_m', 'turn_per_m_deg', 'mesh_clear_min_m', 'length_m',
                  'stairs_headon_med_deg', 'stairs_headon_p90_deg'):
            d[f'{m} raw->ref (median)'] = [med([r.get(f'raw_{m}') for r in G]), med([r.get(f'ref_{m}') for r in G])]
        d['ref on_frac == 1'] = sum(r['ref_on_frac'] == 1.0 for r in G)
        d['margin_min >= 0.05 raw->ref'] = [sum(r['raw_margin_min_m'] >= 0.05 for r in G), sum(r['ref_margin_min_m'] >= 0.05 for r in G)]
        d['mesh_clear_min >= r_b raw->ref'] = [sum(r['raw_mesh_clear_min_m'] >= r['r_b'] for r in G),
                                               sum(r['ref_mesh_clear_min_m'] >= r['r_b'] for r in G)]
        d['fallback paths'] = sum(r.get('n_fallback', 0) > 0 for r in G)
        if 'raw_gt_same_route' in G[0]:
            d['GT same route raw->ref'] = [sum(r['raw_gt_same_route'] for r in G), sum(r['ref_gt_same_route'] for r in G)]
            d['GT turn_per_m (median)'] = med([r['gt_turn_per_m_deg'] for r in G])
        d['sec median'] = med([r['sec'] for r in G])
        summ[k] = d
    # 충돌 표 — 경로 하나를 한 칸에만 넣는다(관통 > r_b 미달 > 사용 가능). GT 는 같은 자로 잰 기준선.
    coll = {}
    for k, G in sorted(groups.items()):
        for who in ('raw', 'ref'):
            coll[f'{k} | {who}'] = collision_counts(G, who)
    for k, G in sorted(gt_groups.items()):
        coll[f'{k.replace("_wp", "").replace("_sg", "")} | GT'] = collision_counts(G, 'ref')
    summ['collision'] = coll
    json.dump(summ, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1, ensure_ascii=False))

    # 그림: 예시 (노은역 3 + vln_ce 층 이동 3) — 위: 전체, 가운데: 확대(가장 많이 움직인 곳), 아래: 여유·높이
    picks = []
    if 'noeun' in groups:
        picks += [r for r in sorted(groups['noeun'], key=lambda r: -r['raw_length_m'])[:3]]
    for k in ('vln_ce_wp_multifloor', 'vln_ce_wp_flat'):
        if k in groups:
            picks += sorted(groups[k], key=lambda r: -(r['ref_margin_p5_m'] - r['raw_margin_p5_m']))[:2]
    figs = {}
    if picks:
        fig, axs = plt.subplots(len(picks), 3, figsize=(24, 5.5 * len(picks)), squeeze=False, gridspec_kw={'width_ratios': [1.2, 1, 1.4]})
        cache = {}
        for row, r in enumerate(picks):
            key = r['scene']
            if key not in cache:
                if r['kind'] == 'noeun':
                    pf = nu.load_pathfinder(str(HERE / 'logs' / 'noeun' / '12_build_navmesh' / 'noeun_station' / 'noeun_station.navmesh'))
                else:
                    pf = nu.load_pathfinder(vlnce_navmesh(args, key))
                cache[key] = (npth.NavScene(pf), np.load(out / 'scenes' / f'{key}_traj.npz'))
            sc, A = cache[key]
            pre = f"{r['id']}" if r['kind'] == 'noeun' else f"{r['ep']}_{r['mode']}"
            R, T = A[f'{pre}_raw'], A[f'{pre}_ref']
            zr = (min(R[:, 2].min(), T[:, 2].min()) - 0.6, max(R[:, 2].max(), T[:, 2].max()) + 0.6)
            m = (sc.tris.mean(1)[:, 2] > zr[0]) & (sc.tris.mean(1)[:, 2] < zr[1])
            fc = np.where(sc.stairs_tri[m], 'salmon', '0.85')
            for ax, zoom in ((axs[row][0], False), (axs[row][1], True)):
                ax.add_collection(PolyCollection(sc.tris[m][:, :, :2], facecolors=fc, edgecolors='w' if zoom else 'none', lw=0.3))
                if f"{r.get('ep')}_gt" in A and r['kind'] == 'vln_ce':
                    G = A[f"{r['ep']}_gt"]
                    ax.plot(G[:, 0], G[:, 1], color='orange', lw=3, alpha=0.6, label='GT')
                ax.plot(R[:, 0], R[:, 1], 'k-', lw=1, label='find_path (raw)')
                ax.plot(T[:, 0], T[:, 1], 'b-', lw=1.3, label='refined')
                ax.set_aspect('equal')
                if zoom:
                    from scipy.spatial import cKDTree
                    d, _ = cKDTree(R[:, :2]).query(T[:, :2])
                    c = T[np.argmax(d), :2]
                    ax.set_xlim(c[0] - 3, c[0] + 3); ax.set_ylim(c[1] - 3, c[1] + 3)
                    ax.set_title('zoom: largest change (salmon = stairs-like navmesh)', fontsize=9)
                else:
                    ax.autoscale(); ax.legend(fontsize=7)
                    ax.set_title(f"{key} {pre} | margin p5 {r['raw_margin_p5_m']:.2f}->{r['ref_margin_p5_m']:.2f} m, "
                                 f"turn {r['raw_turn_per_m_deg']:.1f}->{r['ref_turn_per_m_deg']:.1f} deg/m", fontsize=9)
            ax = axs[row][2]
            for X, col, nm in ((R, 'k', 'raw'), (T, 'b', 'refined')):
                s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1))]
                mg = np.array([sc.margin(p) for p in X[::2]])
                ax.plot(s[::2], mg, color=col, lw=0.8, label=f'navmesh margin {nm}')
            ax.axhline(npth.RefineCfg.push_margin_m, color='b', ls=':', lw=0.8)
            ax.set_ylim(0, 1.0); ax.set_ylabel('margin to navmesh edge [m]'); ax.set_xlabel('path length [m]')
            ax2 = ax.twinx()
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T[:, :2], axis=0), axis=1))]
            ax2.plot(s, T[:, 2], 'g', lw=1, label='refined z')
            st = A[f'{pre}_ref_st']
            ax2.fill_between(s, T[:, 2].min(), T[:, 2].max(), where=st, color='salmon', alpha=0.2, label='stairs part')
            ax2.set_ylabel('z [m]', color='g'); ax.legend(loc='upper left', fontsize=7); ax2.legend(loc='upper right', fontsize=7)
        figs['examples'] = 'examples.png'
        fig.savefig(out / figs['examples'], dpi=80, bbox_inches='tight'); plt.close(fig)

    # 분포: 여유 p5, 회전, mesh 여유
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    for k, G in sorted(groups.items()):
        for ax, m in zip(axs, ('margin_p5_m', 'turn_per_m_deg', 'mesh_clear_min_m')):
            v0 = [r.get(f'raw_{m}') for r in G if r.get(f'raw_{m}') is not None]
            v1 = [r.get(f'ref_{m}') for r in G if r.get(f'ref_{m}') is not None]
            ax.scatter(v0, v1, s=6, alpha=0.5, label=k)
            ax.set_xlabel(f'raw {m}'); ax.set_ylabel(f'refined {m}')
    for ax in axs:
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], 'k:', lw=0.8); ax.legend(fontsize=7)
    figs['scatter'] = 'scatter.png'
    fig.savefig(out / figs['scatter'], dpi=80, bbox_inches='tight'); plt.close(fig)

    def tbl(d):
        keys = sorted({k for g in d.values() for k in g})
        h = '<tr><th>지표</th>' + ''.join(f'<th>{g}</th>' for g in d) + '</tr>'
        for k in keys:
            def fmt(v):
                if isinstance(v, list):
                    return ' → '.join('-' if x is None else (f'{x:.3f}' if isinstance(x, float) else str(x)) for x in v)
                return '-' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))
            h += f'<tr><td>{k}</td>' + ''.join(f'<td>{fmt(d[g].get(k))}</td>' for g in d) + '</tr>'
        return f'<table>{h}</table>'
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>nav3d 03 refine</title>
<style>body{{font-family:sans-serif;max-width:1600px;margin:auto;padding:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:3px 8px;font-size:13px}}
img{{max-width:100%}}</style></head><body><h1>nav3d 03 — 경로 다듬기 (raw find_path → refined)</h1>
<h2>충돌 (관통 = navmesh 밖 또는 mesh 거리 &lt; {PENETRATE_M} m · r_b 미달 = 관통 아님 &amp; mesh 거리 &lt; r_b)</h2>{tbl(coll)}<h2>품질</h2>{tbl({k: v for k, v in summ.items() if k != 'collision'})}<img src="{figs.get('scatter', '')}"><img src="{figs.get('examples', '')}"></body></html>"""
    (out / 'report.html').write_text(html)
    print(f'  -> {out}/report.html')


def main():
    args = parse()
    if args.aggregate:
        aggregate(args)
        return 0
    cfg = npth.RefineCfg(push_margin_m=args.push_margin)
    out = Path(args.out_dir) / 'scenes'
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.dataset == 'vln_ce':
        meta, rows, arrays = run_vlnce(args, cfg)
    elif args.scene == 'noeun_station':
        meta, rows, arrays = run_noeun(args, cfg)
    else:
        assert False, f'unreachable scene={args.scene!r} dataset={args.dataset!r}'
    meta['sec'] = time.time() - t0
    meta['cfg'] = vars(cfg)
    meta['stairs_mode'] = args.stairs_mode
    json.dump({'meta': meta, 'rows': rows}, open(out / f"{meta['scene']}.json", 'w'), indent=1)
    np.savez_compressed(out / f"{meta['scene']}_traj.npz", **arrays)
    ok = [r for r in rows if r.get('found') and r.get('mode') != 'gt']
    print(f"[03] {meta['scene']}: {len(ok)}/{sum(r.get('mode') != 'gt' for r in rows)} · 여유 p5 중앙 {np.median([r['raw_margin_p5_m'] for r in ok]):.3f}"
          f" -> {np.median([r['ref_margin_p5_m'] for r in ok]):.3f} m · {meta['sec']:.0f} s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
