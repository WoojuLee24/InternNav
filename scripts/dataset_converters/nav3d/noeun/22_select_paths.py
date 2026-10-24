"""04 — 노은역 경로를 고르고(사용 가능 판정) 렌더 단계가 읽는 paths json 으로 내보낸다.

    03 층간 경로 (01 V5 와 같은 표본) ─┐
    평지 경로 (층마다 새로 뽑음)       ─┼─> 03 과 같은 다듬기(nav3d_path.refine) -> 사용 가능 판정 U1~U5
                                       └─> paths json (점마다 바닥 z · 평지/계단 구간) + 그림 + report.html

구성 (2026-10-15 사용자 결정): **층 이동 100 + 평지 100 (S0 승강장 50 · S1 대합실 50)**. 지상(S2)은 떨어진 섬이 많아 평지에서 뺀다.
층 이동은 03 결과를 쓰고, 판정에서 빠진 만큼 같은 방식으로 더 뽑는다.

## 용어
- **사용 가능**: 아래 U1~U5 를 모두 지킨 경로.
- **r_b**: 로봇 몸 반지름 0.25 m (01 navmesh 와 같음). **mesh 여유**: 경로 점(바닥 + 0.5 m, 키 − r_b)에서 mesh 까지 3D 거리.
- **큰 mesh 구멍**: 00 `holes.csv` 의 CRITICAL mesh 구멍(통행 구역 안 · 내접 ≥ 0.10 m).

## 사용 가능 기준 (결과를 보기 전에 정함)
| | 묻는 것 | 기준 | 근거 |
|---|---|---|---|
| U1 | 경로가 있고 다듬기가 끝났나 | find_path 성공 | |
| U2 | 다듬은 경로가 전부 navmesh 위인가 | navmesh 위 100 % | 03 과 같은 판정 |
| U3 | 벽·물체에 몸이 걸치지 않나 | mesh 여유 < r_b 인 점 0 · 최소 여유 ≥ 0.02 m | 03 "관통 0 · r_b 미달 0" |
| U4 | 00 의 큰 mesh 구멍 옆을 지나지 않나 | 구멍 중심에서 (r_b + 구멍 반지름) 안을 지나는 점 0 (같은 층 높이 ±0.5 m) | 구멍 위는 렌더 depth 가 뚫리고 로봇이 빠진다 |
| U5 | 평지 경로가 한 층에 머물고 짧지 않나 | 높이 폭 < 0.5 m · 길이 ≥ 3 m | 평지와 층 이동을 섞지 않음 |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/22_select_paths.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
sys.path.insert(0, str(HERE))
import nav3d_path as npth  # noqa: E402
import nav3d_report as R  # noqa: E402
import nav3d_utils as nu  # noqa: E402
import nav3d_scenes as scenes  # noqa: E402
from path_quality import load_holes, near_hole, segments, judge, one  # noqa: E402,F401


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_multifloor', type=int, default=100)
    ap.add_argument('--flat', default='0:50,1:50', help='층:개수 — 평지 경로 (S0 승강장 · S1 대합실)')
    ap.add_argument('--flat_min_m', type=float, default=3.0, help='U5 평지 최소 길이')
    ap.add_argument('--flat_max_m', type=float, default=40.0, help='평지 시작·끝 직선 거리 상한 (R2R 수준 길이)')
    ap.add_argument('--holes_csv', default='scripts/dataset_converters/nav3d/logs/noeun/11_check_assets/noeun_station/holes.csv')
    ap.add_argument('--nav03', default='scripts/dataset_converters/nav3d/logs/common/refine_paths/scenes/noeun_station')
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/noeun/paths')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def main():
    args = parse()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = scenes.SCENES['noeun_station']
    d01 = HERE / 'logs' / 'noeun' / '12_build_navmesh' / 'noeun_station'
    pf = nu.load_pathfinder(str(d01 / 'noeun_station.navmesh'))
    sc = npth.NavScene(pf)
    mesh = nu.add_box_blockers(nu.load_mesh_zup(spec['mesh']), spec.get('blockers', []))
    md = nu.MeshDistance(mesh)
    cfg = npth.RefineCfg()
    r_b = sc.agent_radius
    holes = load_holes(args.holes_csv)
    rng = np.random.default_rng(args.seed)
    cand = []                 # (kind, from, to, row, X, stairs)

    # 층 이동: 03 결과
    j03 = json.loads((Path(args.nav03).with_suffix('.json')).read_text())
    z03 = np.load(str(Path(args.nav03)) + '_traj.npz')
    for row in j03['rows']:
        if not row.get('found'):
            cand.append(('multifloor', None, None, row, None, None, '03'))
            continue
        i = row['id']
        cand.append(('multifloor', row['from_storey'], row['to_storey'], row, z03[f'{i}_ref'].astype(np.float64),
                     z03[f'{i}_ref_st'].astype(bool), '03'))

    pts = scenes.sample_storey_points(pf, sc.levels, sc.lv2st, per_bucket=400, seed=args.seed + 1)

    def gen(st_a, st_b, kind, tries=4000):
        keys = [k for k in pts if k[0] in (st_a, st_b)]
        isl = sorted({k[1] for k in keys if (st_a, k[1]) in pts and (st_b, k[1]) in pts},
                     key=lambda i: -(len(pts[(st_a, i)]) + len(pts[(st_b, i)])))
        for _ in range(tries):
            il = isl[0] if kind == 'flat' else isl[rng.integers(len(isl))]
            A, B = pts[(st_a, il)], pts[(st_b, il)]
            a, b = nu.hab_to_ours(A[rng.integers(len(A))]), nu.hab_to_ours(B[rng.integers(len(B))])
            if kind == 'flat' and not (args.flat_min_m <= np.linalg.norm(a[:2] - b[:2]) <= args.flat_max_m):
                continue
            P = sc.find_path(a, b)
            if P is None:
                continue
            row, r = one(sc, P, md, r_b, cfg)
            return row, r['traj'].astype(np.float64), np.asarray(r['stairs'], bool)
        return None

    # 층 이동이 판정에서 빠지면 같은 방식(03 의 층 쌍 순서)으로 채운다
    eps, rejected = [], []
    combos = sorted({(c[1], c[2]) for c in cand if c[1] is not None})
    k_extra = 0
    for c in cand:
        kind, a, b, row, X, st, src = c
        why = judge(row, X, kind, holes, r_b, args) if X is not None else ['U1 경로 없음']
        (rejected if why else eps).append(dict(kind=kind, from_storey=a, to_storey=b, row=row, X=X, st=st, src=src, why=why))
    while sum(e['kind'] == 'multifloor' for e in eps) < args.n_multifloor and k_extra < 400:
        a, b = combos[k_extra % len(combos)]
        k_extra += 1
        g = gen(a, b, 'multifloor')
        if g is None:
            continue
        row, X, st = g
        why = judge(row, X, 'multifloor', holes, r_b, args)
        (rejected if why else eps).append(dict(kind='multifloor', from_storey=a, to_storey=b, row=row, X=X, st=st, src='new', why=why))
    mf = [e for e in eps if e['kind'] == 'multifloor'][:args.n_multifloor]
    flat = []
    for tok in args.flat.split(','):
        s_, n_ = map(int, tok.split(':'))
        got, tries = 0, 0
        while got < n_ and tries < 20 * n_:
            tries += 1
            g = gen(s_, s_, 'flat')
            if g is None:
                continue
            row, X, st = g
            why = judge(row, X, 'flat', holes, r_b, args)
            e = dict(kind='flat', from_storey=s_, to_storey=s_, row=row, X=X, st=st, src='new', why=why)
            if why:
                rejected.append(e)
            else:
                flat.append(e)
                got += 1
        print(f'  flat S{s_}: {got}/{n_} (tries {tries})', flush=True)
    final = mf + flat

    # paths json
    episodes = []
    for k, e in enumerate(final):
        X = e['X']
        episodes.append({'episode_id': k, 'kind': e['kind'], 'from_storey': e['from_storey'], 'to_storey': e['to_storey'],
                         'source': e['src'], 'start': X[0].tolist(), 'goal': X[-1].tolist(),
                         'trajectory': X.tolist(), 'trajectory_z': X[:, 2].tolist(), 'floor_z': float(X[0, 2]),
                         'segments': segments(e['st'], X[:, 2]), 'path_length_m': float(e['row']['ref_length_m']),
                         'mesh_clear_min_m': float(e['row']['ref_mesh_clear_min_m']),
                         'margin_min_m': float(e['row']['ref_margin_min_m']), 'stairs_frac': float(np.mean(e['st']))})
    doc = {'scene_id': 'noeun_station', 'coords': 'nav3d (Z-up) = usdz world', 'r_b': r_b,
           'agent_height': sc.agent_height, 'navmesh': str(d01 / 'noeun_station.navmesh'),
           'criteria': 'U1~U5 (22_select_paths.py docstring)', 'episodes': episodes}
    (out / 'noeun_station_paths.json').write_text(json.dumps(doc))
    report(out, args, final, rejected, holes, sc, mesh)
    print(f'-> {out}/noeun_station_paths.json · {len(mf)} multifloor + {len(flat)} flat · rejected {len(rejected)}')


def report(out, args, final, rejected, holes, sc, mesh):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import Counter
    why = Counter(w for e in rejected for w in e['why'])
    rows = []
    for kind, st in [('multifloor', None), ('flat', 0), ('flat', 1)]:
        es = [e for e in final if e['kind'] == kind and (st is None or e['from_storey'] == st)]
        if not es:
            continue
        L = [e['row']['ref_length_m'] for e in es]
        rows.append({'구분': '층 이동' if kind == 'multifloor' else f'평지 S{st}', '경로': len(es),
                     '길이 중앙 / 최소 / 최대 [m]': f'{np.median(L):.1f} / {min(L):.1f} / {max(L):.1f}',
                     'mesh 여유 최소 [m]': f"{min(e['row']['ref_mesh_clear_min_m'] for e in es):.3f}",
                     '경계 여유 최소 [m]': f"{min(e['row']['ref_margin_min_m'] for e in es):.3f}",
                     '계단 비율 중앙': f"{np.median([np.mean(e['st']) for e in es]):.2f}",
                     '03 결과 / 새로 뽑음': f"{sum(e['src'] == '03' for e in es)} / {sum(e['src'] == 'new' for e in es)}"})
    figs = []
    tri = sc.tris
    for si, S in enumerate(sc.storeys):
        fig, ax = plt.subplots(figsize=(14, 9))
        cz = tri[:, :, 2].mean(1)
        k = (cz > S['z_range'][0] - 0.4) & (cz < S['z_range'][1] + 0.4)
        from matplotlib.collections import PolyCollection
        ax.add_collection(PolyCollection(tri[k][:, :, :2], facecolor='#dcdcdc', edgecolor='none'))
        for e in final:
            X = e['X']
            on = (X[:, 2] > S['z_range'][0] - 0.6) & (X[:, 2] < S['z_range'][1] + 0.6)
            if not on.any():
                continue
            col = '#e67e22' if e['kind'] == 'multifloor' else '#2b5fa8'
            ax.plot(np.where(on, X[:, 0], np.nan), np.where(on, X[:, 1], np.nan), color=col, lw=0.8, alpha=0.8)
            if e['kind'] == 'flat':
                ax.plot(*X[0, :2], 'o', color=col, ms=3)
        hz = holes[(holes[:, 2] > S['z_range'][0] - 0.6) & (holes[:, 2] < S['z_range'][1] + 0.6)] if len(holes) else holes
        for h in hz:
            ax.add_patch(plt.Circle(h[:2], 0.25 + h[3], color='#d00', fill=False, lw=1))
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_title(f'S{si} z {S["z_range"][0]:+.1f}~{S["z_range"][1]:+.1f}: orange = multifloor (part on this storey), blue = flat, red circle = 00 mesh hole (+r_b)')
        fn = f'paths_S{si}.png'
        fig.savefig(out / fn, dpi=80, bbox_inches='tight')
        plt.close(fig)
        figs.append((fn, f'S{si}: 회색 = navmesh, 주황 = 층 이동 경로(이 층 부분), 파랑 = 평지 경로(점 = 시작), 빨간 원 = 00 의 큰 mesh 구멍(+ r_b).'))
    fig, ax = plt.subplots(figsize=(14, 5))
    for e in final:
        X = e['X']
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1))]
        ax.plot(L, X[:, 2], color='#e67e22' if e['kind'] == 'multifloor' else '#2b5fa8', lw=0.6, alpha=0.6)
    ax.set_xlabel('distance along path [m]')
    ax.set_ylabel('floor z [m]')
    ax.set_title('height along path — orange multifloor, blue flat')
    fig.tight_layout()
    fig.savefig(out / 'height.png', dpi=80)
    plt.close(fig)
    figs.append(('height.png', '경로를 따라 바닥 높이 — 층 이동은 계단을 오르내리고, 평지는 한 층에 머문다.'))
    rej = [{'사유': k, '경로 수': v} for k, v in why.most_common()]
    n_mf = sum(e['kind'] == 'multifloor' for e in final)
    n_fl = sum(e['kind'] == 'flat' for e in final)
    body = R.header('nav3d 04 · 노은역 경로 내보내기', '노은역 경로 선택 · 내보내기',
                    f'층 이동 {n_mf} + 평지 {n_fl} (S0 {sum(e["kind"] == "flat" and e["from_storey"] == 0 for e in final)} · '
                    f'S1 {sum(e["kind"] == "flat" and e["from_storey"] == 1 for e in final)}) — 모두 U1~U5 통과. 판정에서 뺀 후보 {len(rejected)}. '
                    f'출력 <span class="mono">noeun_station_paths.json</span> (점마다 바닥 z · 평지/계단 구간).')
    body += R.section('사용 가능 기준', '<p class="muted">r_b = 로봇 몸 반지름 0.25 m. mesh 여유 = 경로 점(바닥 + 0.5 m, 키 − r_b)에서 mesh 까지 3D 거리. '
                      '큰 mesh 구멍 = 00 holes.csv 의 CRITICAL mesh 구멍.</p>' + R.table([
                          {'ID': 'U1', '기준': '경로 있음'}, {'ID': 'U2', '기준': 'navmesh 위 100 %'},
                          {'ID': 'U3', '기준': 'mesh 여유 < r_b 인 점 0 · 최소 여유 ≥ 0.02 m'},
                          {'ID': 'U4', '기준': '큰 mesh 구멍 중심에서 (r_b + 구멍 반지름) 안을 지나는 점 0 (높이 ±0.5 m)'},
                          {'ID': 'U5', '기준': f'평지: 높이 폭 < 0.5 m · 길이 ≥ {args.flat_min_m} m'}], ['ID', '기준']))
    body += R.section('고른 경로', R.table(rows, list(rows[0].keys())))
    body += R.section('뺀 후보 (사유별)', R.table(rej or [{'사유': '없음', '경로 수': 0}], ['사유', '경로 수']))
    body += R.section('그림', ''.join(R.figure(f, c) for f, c in figs))
    R.write(out, '노은역 경로 내보내기', body)


if __name__ == '__main__':
    main()
