"""03r — 경로를 habitat 안에서 1인칭으로 렌더해 눈으로 비교한다 (vln_ce, 125cm 리그 0° · 30° 룩다운).

두 가지 비교:
- `--what stairs_label`: 계단 라벨 수정 전(unleveled) / 후(sloped) 로 다듬은 경로. 판정이 갈린 점 · 둘 다 계단인 점 ·
  두 경로가 가장 많이 갈라진 점에서 찍고, 이미지마다 그 점의 판정을 표시한다(빨간 테두리 = STAIRS, 파란 테두리 = floor).
  **판정 = 카메라 발밑 navmesh 삼각형의 계단 라벨** — 정면·30° 사진에는 발밑이 안 찍히므로(화면 아래 끝 ≈ 0.7 m 앞)
  발밑(90°, 가운데 노란 십자 = 판정 지점, 사진 위쪽 = 진행 방향) 사진을 함께 찍는다.
- `--what stairs_ab`: 03ab(계단 따로 split vs 전부 바닥처럼 none) 에서 계단 정면 각도 차이가 가장 큰 경로, 계단 위 점에서.
렌더 원본은 낱장으로도 저장: `<out>/<scene>_<ep>/<번호>_<경로>_pitch<0|30|90>.png`.
  에피소드는 03s 결과(평지인데 계단 라벨이 가장 많이 줄어든 것 + 층 이동)에서 고른다.
- `--what compare_2d`: GT · 2D occupancy(gs_vlnpe) · nav3d 경로. 각 경로의 같은 진행 비율(0.1 … 0.9)에서 찍고,
  2D 경로가 navmesh 밖으로 가장 많이 나간 점이 있으면 그 점(다른 경로는 가장 가까운 점)도 찍는다.
  에피소드는 03c 결과(2D 만 실패 · nav3d 만 실패 · 층 이동)에서 고른다. 카메라 높이 = 각 방식이 준 바닥 z + 1.25 m.

렌더러 검증: GT 자세에서 렌더한 이미지 = 데이터셋 rgb.125cm_0deg / 30deg (평균 차 2~3.5 / 255, JPEG 수준).
`--check_gt` 로 다시 확인한다.

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/refine_render_views.py --what stairs_label
    /usr/bin/python scripts/dataset_converters/nav3d/checks/refine_render_views.py --what compare_2d
    /usr/bin/python scripts/dataset_converters/nav3d/checks/refine_render_views.py --check_gt --scene 1LXtFkjw3qL --ep 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
GS = HERE.parent / 'gs_vlnpe'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GS))
import nav3d_path as npth  # noqa: E402
import nav3d_render as nr  # noqa: E402
import nav3d_utils as nu  # noqa: E402

PITCHES = (0, 30)              # 정면 · vln_ce 룩다운
PITCHES_STAIRS = (0, 30, 90)   # + 발밑(90°): 계단 판정은 카메라 발밑 기준이라 그 지점을 직접 본다
H_B = 1.25


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--what', choices=['stairs_label', 'compare_2d', 'stairs_ab'], default=None)
    ap.add_argument('--check_gt', action='store_true')
    ap.add_argument('--scene', default='1LXtFkjw3qL')
    ap.add_argument('--ep', type=int, default=0)
    ap.add_argument('--n_examples', type=int, default=6)
    ap.add_argument('--n_views', type=int, default=4, help='경로당 찍는 곳 수')
    ap.add_argument('--log_root', default='scripts/dataset_converters/nav3d/logs')
    ap.add_argument('--scene_root', default='data/scene_data/mp3d_ce/mp3d')
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    return ap.parse_args()


def arclen(X):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1))]


def at_frac(X, f):
    L = arclen(X)
    return int(np.clip(np.searchsorted(L, f * L[-1]), 0, len(X) - 1))


def shoot(cam, X, i, pitches=PITCHES):
    """pitch 90 은 발밑 — 사진 위쪽 = 진행 방향, 가운데 십자 = 카메라 발밑(판정 지점)."""
    yaw = nr.yaw_along(X, i)
    out = []
    for p in pitches:
        im = cam.render(X[i], yaw, p, H_B)['rgb']
        if p == 90:
            im = im.copy()
            h, w = im.shape[:2]
            im[h // 2 - 1:h // 2 + 2, w // 2 - 25:w // 2 + 26] = (255, 255, 0)
            im[h // 2 - 25:h // 2 + 26, w // 2 - 1:w // 2 + 2] = (255, 255, 0)
        out.append(im)
    return out


def save_views(d, views, pitches):
    """렌더 원본을 낱장으로 저장: <d>/<k>_<경로>_pitch<p>.png"""
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for k, v in enumerate(views):
        for n, ims in v[1].items():
            for p, im in zip(pitches, ims):
                Image.fromarray(im).save(d / f'{k + 1}_{n}_pitch{p}.png')


def sheet(path_png, title, sc, paths, views, zr, sc_before=None, pitches=PITCHES):
    """paths: [(name, X, color)], views: [(label, {name: [rgb0, rgb30]}, {name: point})]."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    names = [n for n, X, _ in paths if X is not None and any(n in v[1] for v in views)]   # 찍은 경로만 열로
    nr_, nc = len(views), 1 + len(names) * len(pitches)
    fig = plt.figure(figsize=(4.2 * (nc - 1) + 8, 3.3 * nr_ + 0.6))
    gs = fig.add_gridspec(nr_, nc, width_ratios=[8 / 4.2] + [1] * (nc - 1))
    ax = fig.add_subplot(gs[:, 0])
    allp = np.vstack([X for _, X, _ in paths if X is not None])
    lo, hi = allp[:, :2].min(0) - 1.0, allp[:, :2].max(0) + 1.0
    c = sc.tris.mean(1)
    m = (c[:, 2] > zr[0]) & (c[:, 2] < zr[1])
    ax.add_collection(PolyCollection(sc.tris[m][:, :, :2], facecolors=np.where(sc.stairs_tri[m], 'salmon', '0.85'), edgecolors='none'))
    if sc_before is not None and sc_before.stairs_tri[m].any():     # 수정 전 계단 = 빨간 빗금 (수정 후 = 살구색 칠)
        mb = m & sc_before.stairs_tri
        ax.add_collection(PolyCollection(sc.tris[mb][:, :, :2], facecolors='none', edgecolors='red', lw=0.4, hatch='////'))
    for n, X, col in paths:
        if X is not None:
            ax.plot(X[:, 0], X[:, 1], color=col, lw=3 if n == 'GT' else 1.5, alpha=0.6 if n == 'GT' else 1.0, label=n)
    for k, v in enumerate(views):
        pts = v[2]
        for n, p in pts.items():
            col = dict((nn, cc) for nn, _, cc in paths)[n]
            ax.plot(p[0], p[1], 'o', ms=9, mfc='w', mec=col)
            ax.annotate(str(k + 1), p[:2], ha='center', va='center', fontsize=7)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal'); ax.legend(fontsize=7, loc='best')
    ax.set_title('top view: salmon fill = stairs (' + ('after' if sc_before is not None else 'current') + ')' +
                 ('\nred hatch = stairs (before)' if sc_before is not None else '') + '\nnumbers = camera spots', fontsize=9)
    for r, v in enumerate(views):
        lab, imgs, pts = v[:3]
        stl = v[3] if len(v) > 3 else {}            # {경로 이름: 그 점이 계단 라벨인가}
        col_i = 1
        for n in names:
            for pi, p in enumerate(pitches):
                a = fig.add_subplot(gs[r, col_i])
                if n in imgs:
                    a.imshow(imgs[n][pi])
                a.set_xticks([]); a.set_yticks([])
                t = f'[{r + 1}] {n} · ' + ('FOOT (down 90°, + = camera spot)' if p == 90 else f'pitch {p}°') + f' {lab if pi == 0 else ""}'
                if n in stl:                                # 계단 판정 표시: 빨간 테두리 = STAIRS, 파란 테두리 = FLOOR
                    t += f"\nlabel here: {'STAIRS' if stl[n] else 'floor'}"
                    for sp in a.spines.values():
                        sp.set_edgecolor('red' if stl[n] else 'royalblue'); sp.set_linewidth(4)
                a.set_title(t, fontsize=8, color=('red' if stl.get(n) else 'k'))
                col_i += 1
    fig.suptitle(title, fontsize=11)
    fig.savefig(path_png, dpi=60, bbox_inches='tight')
    plt.close(fig)


def point_labels(sc, X):
    """경로 점마다 그 아래 navmesh 삼각형이 계단 라벨인가."""
    _, idx = sc.surf.track_idx(X[:, :2], z0=X[0, 2])
    return np.where(idx >= 0, sc.stairs_tri[np.maximum(idx, 0)], False)


# ---------------------------------------------------------------------------
def run_stairs_label(args):
    out = Path(args.log_root) / 'checks' / 'refine_render_views' / 'stairs_label'
    out.mkdir(parents=True, exist_ok=True)
    S = [json.load(open(f)) for f in sorted((Path(args.log_root) / 'checks' / 'refine_stairs_label_check' / 'scenes').glob('*.json'))]
    rows = [dict(r, scene=j['scene']) for j in S if j['kind'] == 'vln_ce' for r in j['rows']]
    flat = sorted([r for r in rows if not r['multifloor']], key=lambda r: -(r['unleveled_stairs_m'] - r['sloped_stairs_m']))
    picks, seen = [], set()
    for r in flat:
        if r['scene'] not in seen and len(picks) < args.n_examples - 2:
            picks.append(r); seen.add(r['scene'])
    for r in sorted([r for r in rows if r['multifloor'] and r['dz_total'] > 2.0], key=lambda r: -r['length_m']):
        if r['scene'] not in seen and len(picks) < args.n_examples:
            picks.append(r); seen.add(r['scene'])
    index = []
    for r in picks:
        s, pre = r['scene'], r['pre']
        pf = nu.load_pathfinder(f'{args.scene_root}/{s}/{s}.navmesh')
        scs = {g: npth.NavScene(pf, stairs_group=g) for g in ('unleveled', 'sloped')}
        A = np.load(Path(args.log_root) / 'common' / 'refine_paths' / 'scenes' / f'{s}_traj.npz')
        R = A[f'{pre}_raw'].astype(np.float64)
        T = {g: npth.refine(scs[g], R)['traj'] for g in scs}
        G = A[f"{pre.split('_')[0]}_gt"].astype(np.float64)
        # 찍을 곳: (1) 판정이 갈린 점(before 계단 · after 바닥 또는 반대) 최대 2 곳 (2) 둘 다 계단인 점 1 곳 — 진짜 계단 유지 확인
        # (3) 나머지는 두 경로가 가장 많이 갈라진 점. 서로 1.5 m 이상 떨어지게. 판정 = 그 점 아래 navmesh 삼각형의 계단 라벨.
        from scipy.spatial import cKDTree
        Tb, Ta = T['unleveled'], T['sloped']
        d, jn = cKDTree(Ta[:, :2]).query(Tb[:, :2])
        lb, la = point_labels(scs['unleveled'], Tb), point_labels(scs['sloped'], Ta)[jn]
        L = arclen(Tb)
        chosen = []

        def take(idx, k):
            for i in idx:
                if sum(1 for _ in chosen) >= args.n_views or k <= 0:
                    break
                if all(abs(L[i] - L[j]) > 1.5 for j in chosen):
                    chosen.append(int(i)); k -= 1
        diff = np.where(lb != la)[0]
        take(diff[np.argsort(np.abs(np.arange(len(diff)) - len(diff) / 2))] if len(diff) else [], 2)   # 갈린 구간 가운데부터
        both = np.where(lb & la)[0]
        take(both[np.argsort(np.abs(np.arange(len(both)) - len(both) / 2))] if len(both) else [], 1)
        take(np.argsort(-d), args.n_views)
        chosen.sort()
        cam = nr.HabitatCam(f'{args.scene_root}/{s}/{s}.glb')
        views = []
        for i in chosen:
            j = int(jn[i])
            views.append((f'(gap {d[i]:.2f} m)', {'before': shoot(cam, Tb, i, PITCHES_STAIRS), 'after': shoot(cam, Ta, j, PITCHES_STAIRS)},
                          {'before': Tb[i], 'after': Ta[j]}, {'before': bool(lb[i]), 'after': bool(la[i])}))
        cam.close()
        allp = np.vstack([R, G])
        png = out / f'{s}_{pre}.png'
        sheet(png, f'{s} {pre} | stairs label before (unleveled, red) vs after (sloped, blue) — refined path, habitat 1st-person',
              scs['sloped'], [('GT', G, 'orange'), ('before', T['unleveled'], 'r'), ('after', T['sloped'], 'b')], views,
              (allp[:, 2].min() - 0.6, allp[:, 2].max() + 0.6), sc_before=scs['unleveled'], pitches=PITCHES_STAIRS)
        save_views(out / f'{s}_{pre}', views, PITCHES_STAIRS)
        index.append({'scene': s, 'pre': pre, 'png': png.name, 'max_gap_m': float(d.max()),
                      'views': [{'k': k + 1, 'before_stairs': v[3]['before'], 'after_stairs': v[3]['after'],
                                 'z': float(v[2]['before'][2])} for k, v in enumerate(views)],
                      'stairs_m_before': r['unleveled_stairs_m'], 'stairs_m_after': r['sloped_stairs_m']})
        print(f'[03r] {png.name} · 최대 갈라짐 {d.max():.2f} m', flush=True)
    json.dump(index, open(out / 'index.json', 'w'), indent=1)


def run_stairs_ab(args):
    """03ab 에서 계단 정면 각도 차이가 가장 큰 경로 — split(계단 따로) vs none(전부 바닥처럼). 계단 위 점 중 두 경로가 가장 많이 갈라진 곳."""
    from scipy.spatial import cKDTree
    ab = Path(args.log_root) / 'checks' / 'refine_stairs_mode_compare'
    out = Path(args.log_root) / 'checks' / 'refine_render_views' / 'stairs_ab'
    out.mkdir(parents=True, exist_ok=True)
    picks, seen = [], set()
    for e in json.load(open(ab / 'largest_diff.json')):
        if e['scene'] not in seen and e['scene'] != 'noeun_station' and len(picks) < args.n_examples:
            picks.append(e); seen.add(e['scene'])
    index = []
    for e in picks:
        s = e['scene']
        pre = f"{e['id']}_{e['mode']}"
        pf = nu.load_pathfinder(f'{args.scene_root}/{s}/{s}.navmesh')
        sc = npth.NavScene(pf)
        T = {m: np.load(ab / m / 'scenes' / f'{s}_traj.npz')[f'{pre}_ref'].astype(np.float64) for m in ('split', 'none')}
        G = np.load(ab / 'split' / 'scenes' / f'{s}_traj.npz')[f"{e['id']}_gt"].astype(np.float64)
        Ts, Tn = T['split'], T['none']
        d, jn = cKDTree(Tn[:, :2]).query(Ts[:, :2])
        ls = point_labels(sc, Ts)
        L = arclen(Ts)
        chosen = []
        for i in np.argsort(-(d + 10.0 * ls)):           # 계단 위 점 먼저, 그 안에서 많이 갈라진 순
            if all(abs(L[i] - L[j]) > 1.0 for j in chosen):
                chosen.append(int(i))
            if len(chosen) >= args.n_views:
                break
        chosen.sort()
        ln = point_labels(sc, Tn)
        cam = nr.HabitatCam(f'{args.scene_root}/{s}/{s}.glb')
        views = [(f'(gap {d[i]:.2f} m)', {'split': shoot(cam, Ts, i, PITCHES_STAIRS), 'none': shoot(cam, Tn, int(jn[i]), PITCHES_STAIRS)},
                  {'split': Ts[i], 'none': Tn[int(jn[i])]}, {'split': bool(ls[i]), 'none': bool(ln[int(jn[i])])}) for i in chosen]
        cam.close()
        allp = np.vstack([Ts, G])
        png = out / f'{s}_{pre}.png'
        sheet(png, f"{s} {pre} | split (stairs straight, blue) vs none (all as floor, red) | stairs head-on median "
                   f"{e['split_headon']:.0f} vs {e['none_headon']:.0f} deg", sc,
              [('GT', G, 'orange'), ('split', Ts, 'b'), ('none', Tn, 'r')], views,
              (allp[:, 2].min() - 0.6, allp[:, 2].max() + 0.6), pitches=PITCHES_STAIRS)
        save_views(out / f'{s}_{pre}', views, PITCHES_STAIRS)
        index.append(dict(e, png=png.name))
        print(f'[03r] {png.name} · 계단 정면 {e["split_headon"]:.0f} vs {e["none_headon"]:.0f}°', flush=True)
    json.dump(index, open(out / 'index.json', 'w'), indent=1)


def category(m):
    if m is None:
        return 'fail'
    if m['on_frac'] < 1.0 or m['mesh_clear_min_m'] < 0.02:
        return 'penetrate'
    return 'lt_rb' if m['mesh_clear_min_m'] < 0.10 else 'usable'


def run_compare_2d(args):
    out = Path(args.log_root) / 'checks' / 'refine_render_views' / 'compare_2d'
    out.mkdir(parents=True, exist_ok=True)
    cdir = Path(args.log_root) / 'checks' / 'refine_vs_2d_compare' / 'scenes'
    rows = []
    for f in sorted(cdir.glob('*.json')):
        jj = json.load(open(f))
        rows += [dict(r, scene=jj['scene']) for r in jj['rows']]

    def pick(cond, n, seen):
        res = []
        for r in rows:
            if r['scene'] not in seen and cond(r):
                res.append(r); seen.add(r['scene'])
            if len(res) >= n:
                break
        return res
    seen = set()
    k2 = lambda r: '2d_plan' if r['multifloor'] else '2d'
    picks = (pick(lambda r: not r['multifloor'] and category(r.get('2d')) != 'usable' and r.get('2d') and category(r.get('nav3d')) == 'usable', 2, seen) +
             pick(lambda r: not r['multifloor'] and category(r.get('nav3d')) != 'usable' and category(r.get('2d')) == 'usable', 1, seen) +
             pick(lambda r: r['multifloor'] and r.get('2d_plan') and category(r.get('nav3d')) == 'usable', 3, seen))
    index = []
    for r in picks:
        s, ep = r['scene'], r['ep']
        pf = nu.load_pathfinder(f'{args.scene_root}/{s}/{s}.navmesh')
        sc = npth.NavScene(pf)
        A = np.load(Path(args.log_root) / 'common' / 'refine_paths' / 'scenes' / f'{s}_traj.npz')
        B = np.load(cdir / f'{s}_traj.npz')
        G = A[f'{ep}_gt'].astype(np.float64)
        T = A[f'{ep}_sg_ref'].astype(np.float64) if f'{ep}_sg_ref' in A.files else None
        D = B[f'{ep}_{k2(r)}'].astype(np.float64) if f'{ep}_{k2(r)}' in B.files else None
        paths = [('GT', G, 'orange'), ('2D', D, 'g'), ('nav3d', T, 'b')]
        cam = nr.HabitatCam(f'{args.scene_root}/{s}/{s}.glb')
        views = []
        # 2D 경로가 navmesh 밖으로 가장 많이 나간 점 (있으면) — 다른 경로는 그 점에서 가장 가까운 점
        if D is not None:
            z, _ = sc.surf.track_idx(D[:, :2], z0=D[0, 2])
            off = np.where(~np.isfinite(z))[0]
            if len(off):
                dd = [np.linalg.norm(nu.hab_to_ours(np.asarray(pf.snap_point(nu.ours_to_hab(D[i]).astype(np.float32))))[:2] - D[i, :2]) for i in off]
                iw = int(off[int(np.argmax(dd))])
                imgs, pts = {}, {}
                for n, X, _ in paths:
                    if X is None:
                        continue
                    i = iw if n == '2D' else int(np.argmin(np.linalg.norm(X[:, :2] - D[iw, :2], axis=1)))
                    imgs[n], pts[n] = shoot(cam, X, i), X[i]
                views.append((f'(2D off navmesh {max(dd):.2f} m)', imgs, pts))
        for f in np.linspace(0.1, 0.9, args.n_views):
            imgs, pts = {}, {}
            for n, X, _ in paths:
                if X is None:
                    continue
                i = at_frac(X, f)
                imgs[n], pts[n] = shoot(cam, X, i), X[i]
            views.append((f'(at {f:.0%})', imgs, pts))
        cam.close()
        allp = np.vstack([X for _, X, _ in paths if X is not None])
        info = lambda m: 'fail' if m is None else f"{category(m)} (mesh min {m['mesh_clear_min_m']:.2f} m, Frechet {m['frechet_m']:.1f} m)"
        png = out / f'{s}_ep{ep}.png'
        sheet(png, f"{s} ep{ep} ({'multifloor' if r['multifloor'] else 'flat'}) | 2D: {info(r.get(k2(r)))} | nav3d: {info(r.get('nav3d'))}",
              sc, paths, views, (allp[:, 2].min() - 0.6, allp[:, 2].max() + 0.6))
        save_views(out / f'{s}_ep{ep}', views, PITCHES)
        index.append({'scene': s, 'ep': ep, 'multifloor': r['multifloor'], 'png': png.name,
                      '2d': category(r.get(k2(r))), 'nav3d': category(r.get('nav3d'))})
        print(f'[03r] {png.name} · 2D {index[-1]["2d"]} · nav3d {index[-1]["nav3d"]}', flush=True)
    json.dump(index, open(out / 'index.json', 'w'), indent=1)


def check_gt(args):
    import vlnce_gt
    from PIL import Image
    g = vlnce_gt.load_gt_episode_vlnce(args.data_root, args.scene, args.ep, '125cm_30deg')
    cam = nr.HabitatCam(f'{args.scene_root}/{args.scene}/{args.scene}.glb')
    errs = []
    for i in range(0, len(g['cam_xyz']), max(1, len(g['cam_xyz']) // 5)):
        f = g['poses_c2w'][i][:3, 2]
        yaw = float(np.arctan2(f[1], f[0]))
        for p in PITCHES:
            o = cam.render(g['cam_xyz'][i] - [0, 0, H_B], yaw, p, H_B)['rgb']
            ref = np.asarray(Image.open(f'{args.data_root}/{args.scene}/videos/chunk-000/observation.images.rgb.125cm_{p}deg/'
                                        f'episode_{args.ep:06d}_{i}.jpg').convert('RGB'))
            errs.append(float(np.abs(o.astype(int) - ref.astype(int)).mean()))
    cam.close()
    print(f'[03r check_gt] {args.scene} ep{args.ep}: 평균 |차| {np.mean(errs):.2f} / 255 (최대 {np.max(errs):.2f}) — 5 넘으면 규약 오류 의심')
    return 0 if np.max(errs) < 5 else 1


def main():
    args = parse()
    if args.check_gt:
        return check_gt(args)
    if args.what == 'stairs_label':
        run_stairs_label(args)
    elif args.what == 'compare_2d':
        run_compare_2d(args)
    elif args.what == 'stairs_ab':
        run_stairs_ab(args)
    else:
        assert False, f'unreachable what={args.what!r}'
    return 0


if __name__ == '__main__':
    sys.exit(main())
