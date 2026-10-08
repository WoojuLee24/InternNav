"""W 게이트 — 사람 주석 waypoint를 앵커로 leg별 국소 우회가 되는지 검증한다.

| 게이트 | 무엇을 묻나 | 통과 기준 |
|---|---|---|
| W1 | spine 앵커 품질: waypoint→최근접 GT 프레임 거리, 프레임 단조성 | mean < 0.15 m, 단조 100% |
| W2 | **VLN-CE 기준 재현**: `r_b=0.1`에서 leg 성공률 | ≈100% (원본이 이 기준으로 만들어졌으므로) |
| W3 | GT 충실도: mode별 GT와의 평균거리 | `nudge`/`corridor`가 `free`·순수 A*보다 작아야 |
| W4 | **커버리지 이득**: leg 단위 국소화로 살아남는 프레임 수 vs 전 구간 방식 | 프레임 커버리지 증가 |

W2가 가장 중요하다 — VLN-CE는 *"an agent can follow the shortest path to within 0.5 m of the next
waypoint"* 로 궤적 navigability를 판정해 데이터셋을 만들었고, 그때 로봇은 *"a 1.5m tall cylinder of
diameter of 0.2m"*(= `r_b` 0.1, `h` 1.5)였다. 그러니 **`r_b=0.1`에서 우리 leg가 거의 다 성공해야
정상**이다. 미달이면 우리 occupancy 맵이 recast보다 과대 장애물이라는 뜻이다.

실행:
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/05_validate_waypoints.py --scene s8pcmisQ38h --r_bs 0.10,0.20,0.30,0.45
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / 'gs_vlnpe')); sys.path.insert(0, str(_HERE))
from esdf_utils import (  # noqa: E402
    compute_esdf_2d, compute_scan_coverage_mask, derive_obstacle_2d, resample_by_arclength,
    truncate_navigable,
)
from viz_utils import floorplan_canvas, save_gallery  # noqa: E402
from embodiment_augment import (DETOUR_THRESH_M, H_NAV_HABITAT_M, H_OBS_HABITAT_M,  # noqa: E402
                                PLAN_MARGIN_M, EmbodimentAugmenter, densify)
from pixel_goal_utils import rig_height_m  # noqa: E402
_m_verify_pose_mesh = importlib.import_module('00_verify_pose_mesh')  # noqa: E402
load_episode = _m_verify_pose_mesh.load_episode  # noqa: E402
from waypoint_spine import LEG_TOL_M, build_spine  # noqa: E402
_m_verify_navmesh_gt = importlib.import_module('03_verify_navmesh_gt')  # noqa: E402
chain_find_path = _m_verify_navmesh_gt.chain_find_path  # noqa: E402
from navmesh_grid import DEFAULT_NAVMESH_DIR, cache_path  # noqa: E402

GT_COLOR, WP_COLOR, START_COLOR, GOAL_COLOR = (255, 255, 255), (0, 220, 255), (255, 200, 0), (255, 0, 255)
DETOUR_COLOR = (255, 120, 0)      # 우회가 일어난 지점 표시 — 선만 보면 10 cm 이탈은 육안으로 안 보인다
BASELINE_R_B = 0.10       # VLN-CE 정의값 (cylinder diameter 0.2 m)
MODES = ('none', 'nudge', 'corridor', 'corridor_gtc', 'free')



def _clr_stats(res, gt, esdf, origin, cell, r_b):
    """경로와 GT 대응점의 esdf 여유 비교. -> dict(gap, low, low_gt) or None

    `_gt_dist`처럼 GT를 `frames_covered`로 자르고 호길이 리샘플로 대응점을 만든다.
    gap = |경로 여유 − GT 여유| median, low = 경로 여유 < r_b+0.05 비율 (스플라인 후 최종 경로 기준).
    """
    cov = res.get('frames_covered')
    if not len(res['path_xy']) or cov is None:
        return None
    seg = gt[cov[0]:cov[1] + 1, :2]
    if len(seg) < 2:
        return None
    rp = resample_by_arclength(np.asarray(res['path_xy']), n=160)
    rg = resample_by_arclength(np.asarray(seg), n=160)

    def look(q):
        ij = np.floor((q - np.asarray(origin)[:2]) / cell).astype(int)
        ok = ((ij[:, 0] >= 0) & (ij[:, 0] < esdf.shape[1])
              & (ij[:, 1] >= 0) & (ij[:, 1] < esdf.shape[0]))
        v = np.full(len(q), np.nan)
        v[ok] = esdf[ij[ok][:, 1], ij[ok][:, 0]]
        return v

    cp, cg = look(rp), look(rg)
    m = np.isfinite(cp) & np.isfinite(cg)
    if not m.any():
        return None
    return dict(gap=float(np.median(np.abs(cp[m] - cg[m]))),
                low=float(np.mean(cp[m] < r_b + 0.05)),
                low_gt=float(np.mean(cg[m] < r_b + 0.05)))


def hab_leg_pass(navmesh_dir, scene, r_b, wps_hab, tol_m=LEG_TOL_M):
    """habitat navmesh(캐시된 r_b)로 leg별 통과 여부. -> (L,) bool or None

    **VLN-CE 논문 자신의 기준**을 그대로 쓴다 — leg 경로가 다음 waypoint의 `tol_m`(0.5 m) 안에 닿나.
    우리 ladder와 독립이므로 오라클이 된다. 캐시가 없으면 None(W6 생략).
    """
    if not navmesh_dir:
        return None
    f = cache_path(navmesh_dir, scene, r_b)
    if not f.exists():
        return None
    from habitat_sim.nav import PathFinder
    pf = PathFinder(); pf.load_nav_mesh(str(f))
    _p, reach = chain_find_path(pf, wps_hab)
    return reach <= float(tol_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='s8pcmisQ38h')
    ap.add_argument('--rig', default='125cm_0deg')
    ap.add_argument('--r_bs', default='0.10,0.20,0.30,0.45')
    ap.add_argument('--episodes', type=int, default=14)
    ap.add_argument('--n_frames', type=int, default=40, help='에피소드당 로드할 GT 프레임 수(균등 샘플)')
    ap.add_argument('--corridor_m', type=float, default=0.75,
                    help='회랑 반경[m] — GT에서 이만큼만 벗어남. leg 길이가 median 1.79 m라 2.0 m는 제약이 안 된다(corridor==free)')
    ap.add_argument('--detour_thresh_m', type=float, default=DETOUR_THRESH_M,
                    help="leg를 'detour'로 부를 최소 이탈량[m]. 파이프라인 노이즈(앵커 0.094 m·셀 0.05·"
                         "margin 0.05·반경 증가분)보다 위여야 한다")
    ap.add_argument('--max_z_spread', type=float, default=0.30,
                    help='카메라 높이 변동 상한[m]. 초과 = 계단 → 단일 floor_z로 2D 투영 불가')
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_n1')
    ap.add_argument('--esdf_dir', default='scripts/dataset_converters/gs_vlnpe/logs/esdf')
    ap.add_argument('--geo_dir', default='data/embodiment_aug/scene_geo',
                    help='씬 지오메트리 전용 ply 폴더 (01_build_scene_geo.py 산출물)')
    ap.add_argument('--map_source', default='occ', choices=['occ', 'navmesh'],
                    help='계획 격자 소스. occ=3D 스캔 밴드 투영(habitat 파라미터 정렬, 기본) · '
                         'navmesh=VLN-CE navmesh 래스터화(habitat 판정과 100%% 일치, habitat 의존)')
    ap.add_argument('--navmesh_dir', default=DEFAULT_NAVMESH_DIR,
                    help='navmesh 캐시 위치. map_source=occ여도 W6 오라클 대조에는 쓴다')
    ap.add_argument('--out_dir', default='logs/embodiment_augment/waypoints')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    r_bs = [float(x) for x in args.r_bs.split(',')]
    h_rig = rig_height_m(args.rig)

    aug = EmbodimentAugmenter(args.data_root, args.raw_root, args.mesh_root, args.esdf_dir,
                              args.geo_dir, map_source=args.map_source,
                              navmesh_dir=args.navmesh_dir or None)
    print(f'[W] map_source={args.map_source}' + (
        f" (밴드 h_nav={H_NAV_HABITAT_M} · h_obs={H_OBS_HABITAT_M} — habitat 정렬)"
        if args.map_source == 'occ' else f' (캐시 {args.navmesh_dir})'))
    raw_idx = aug._raw_index() if hasattr(aug, '_raw_index') else None
    tasks = [json.loads(l) for l in open(Path(args.data_root) / args.scene / 'meta' / 'episodes.jsonl')]
    ctx = aug._scene_ctx(args.scene); cell = float(ctx['args'].cell_m)

    body, rows = [], []
    anchor_d, n_spine_ok, n_tried = [], 0, 0
    per_rb = {rb: dict(ok=0, detour=0, blocked=0, gt=[], usable=0, total=0,
                       by_mode={m: 0 for m in MODES},
                       # W6 혼동행렬: (우리 통과?, habitat 통과?) -> 개수
                       cm={'both': 0, 'false_accept': 0, 'false_reject': 0, 'both_block': 0},
                       why={}) for rb in r_bs}
    mode_gt = {m: [] for m in MODES}
    mode_stat = {m: dict(blocked=0, detour=0) for m in MODES}
    clr_stat, p4_stat = {}, {}                      # 여유 스타일 (P1~P4 + ablation)

    for ep in range(min(args.episodes, len(tasks))):
        instr = tasks[ep]['tasks'][0]
        try:
            frames = load_episode(args.data_root, args.scene, args.rig, ep, args.n_frames)
            al = aug.align(args.scene, ep, frames, instr)
        except Exception as e:
            print(f'[W] ep{ep}: skip ({type(e).__name__})'); continue
        T = al['T']
        gt = np.stack([(T @ p)[:3, 3] for _f, p, _d, _v in frames])
        z_spread = float(gt[:, 2].max() - gt[:, 2].min())
        if z_spread > args.max_z_spread:
            print(f'[W] ep{ep}: skip (높이 변동 {z_spread:.2f} > {args.max_z_spread} = 계단)'); continue
        raw = aug._raw.get((args.scene, instr.strip())) if hasattr(aug, '_raw') else None
        if raw is None:
            from vlnce_align import build_rawdata_index
            if raw_idx is None:
                raw_idx = build_rawdata_index(Path(args.raw_root))
            raw = raw_idx.get((args.scene, instr.strip()))
        if raw is None:
            print(f'[W] ep{ep}: skip (raw 매칭 실패)'); continue

        n_tried += 1
        spine = build_spine(raw['reference_path'], gt)
        anchor_d.append(spine['anchor_dist_m'])
        if not spine['ok']:
            print(f"[W] ep{ep}: spine 거부 — {spine['reason']}"); continue
        n_spine_ok += 1
        # mesh z == habitat y라 `reference_path` y가 곧 절단 높이다 (rig 높이 역산보다 직접적, S2와 동일).
        wps_hab = np.asarray(raw['reference_path'], dtype=np.float64)
        fz = float(np.median(wps_hab[:, 1]))

        # --- mode별 GT 충실도 (W3) + 벽 여유 스타일 (P1~P3): baseline r_b에서 비교
        _c2, cell_b, esdf_b, _nav_b = aug._leg_grids(args.scene, BASELINE_R_B, 0.12, h_rig, fz,
                                                     PLAN_MARGIN_M)
        for m in MODES:
            res = aug.follow_waypoints(args.scene, spine, gt[:, :2], BASELINE_R_B, floor_z=fz,
                                       ladder=(m,), corridor_m=args.corridor_m)
            d = _gt_dist(res, gt)
            if d is not None:
                mode_gt[m].append(d)
            mode_stat[m]['blocked'] += res['n_blocked']; mode_stat[m]['detour'] += res['n_detour']
            st = _clr_stats(res, gt, esdf_b, ctx['origin'], cell_b, BASELINE_R_B)
            if st is not None:
                clr_stat.setdefault(m, []).append(st)
        # ablation: 고정 마진(GT 매칭 없음 — 좁은 문에서 blocked가 생길 것이라는 예측) + cap 절반
        for label, kw in (('corridor_gtc fixed+0.15', dict(gtc_fixed_m=0.15)),
                          ('corridor_gtc cap=0.15', dict(gtc_cap_m=0.15))):
            res = aug.follow_waypoints(args.scene, spine, gt[:, :2], BASELINE_R_B, floor_z=fz,
                                       ladder=('corridor_gtc',), corridor_m=args.corridor_m, **kw)
            d = _gt_dist(res, gt)
            if d is not None:
                mode_gt.setdefault(label, []).append(d)
            mode_stat.setdefault(label, dict(blocked=0, detour=0))
            mode_stat[label]['blocked'] += res['n_blocked']
            mode_stat[label]['detour'] += res['n_detour']
            st = _clr_stats(res, gt, esdf_b, ctx['origin'], cell_b, BASELINE_R_B)
            if st is not None:
                clr_stat.setdefault(label, []).append(st)
        # P4: 큰 r_b에서도 방향이 유지되나
        for rb4 in (0.20, 0.30):
            _c4, cell_4, esdf_4, _n4 = aug._leg_grids(args.scene, rb4, 0.12, h_rig, fz, PLAN_MARGIN_M)
            for m in ('corridor', 'corridor_gtc'):
                res = aug.follow_waypoints(args.scene, spine, gt[:, :2], rb4, floor_z=fz,
                                           ladder=(m,), corridor_m=args.corridor_m)
                st = _clr_stats(res, gt, esdf_4, ctx['origin'], cell_4, rb4)
                if st is not None:
                    p4_stat.setdefault((rb4, m), []).append(st)

        # --- r_b sweep (W2/W4)
        per_ep = {}
        for rb in r_bs:
            res = aug.follow_waypoints(args.scene, spine, gt[:, :2], rb, floor_z=fz,
                                       corridor_m=args.corridor_m,
                                       detour_thresh_m=args.detour_thresh_m)
            per_ep[rb] = res
            d = per_rb[rb]
            d['ok'] += res['n_ok']; d['detour'] += res['n_detour']; d['blocked'] += res['n_blocked']
            for lg in res['legs']:
                if lg['mode'] in d['by_mode']:
                    d['by_mode'][lg['mode']] += 1
            d['usable'] += res['frames_usable']; d['total'] += res['n_frames']
            # --- W6: 같은 leg를 habitat navmesh로도 판정해 혼동행렬을 만든다 (독립 오라클)
            hb = hab_leg_pass(args.navmesh_dir, args.scene, rb, wps_hab)
            if hb is not None:
                n_leg = len(spine['legs'])
                ours = np.zeros(n_leg, dtype=bool)
                for lg in res['legs']:
                    ours[lg['i']] = lg['status'] != 'blocked'
                for o, r in zip(ours, hb[:n_leg]):
                    d['cm']['both' if (o and r) else 'false_accept' if o else
                             'false_reject' if r else 'both_block'] += 1
            # --- W5 진단: baseline에서 `none`이 아닌 leg의 **사유**를 `_path_ok` 한 곳에서 뽑는다
            if abs(rb - BASELINE_R_B) < 1e-9:
                bad = [lg for lg in res['legs'] if lg['mode'] != 'none']
                if bad:
                    c, cell_, esdf_, _nav = aug._leg_grids(args.scene, rb, 0.12, h_rig, fz,
                                                           PLAN_MARGIN_M)
                    for lg in bad:
                        l0, l1 = lg['frames']
                        r_ = aug._path_ok(c, gt[l0:l1 + 1, :2], esdf_, cell_, rb,
                                          gt[l1, :2], LEG_TOL_M, why=True)
                        k = f'{r_} → {lg["mode"] or "blocked"}'
                        d['why'][k] = d['why'].get(k, 0) + 1
            gd = _gt_dist(res, gt)
            if gd is not None:
                d['gt'].append(gd)

        # --- 그림: 에피소드 계획 격자 배경 + GT + waypoint + r_b별 경로
        obs = derive_obstacle_2d(ctx['occ'], ctx['origin'], fz, h_rig, cell,
                                 ctx['args'].h_nav_ratio * h_rig)
        nav = truncate_navigable(compute_esdf_2d(obs, cell), BASELINE_R_B) & ctx['coverage']
        base, draw, emit = floorplan_canvas(nav, ctx['origin'], cell, out)
        img = base.copy()
        draw(img, densify(gt[:, :2], cell), GT_COLOR, 4)
        for rb in r_bs:
            res = per_ep[rb]
            if len(res['path_xy']):
                draw(img, res['path_xy'], _rb_color(rb, r_bs), 1)
        # waypoint는 **단색 하나**로만 찍는다. leg가 막힌 것은 색 선이 거기서 끊기는 걸로 읽힌다.
        draw(img, np.asarray(spine['waypoints_mesh'])[:, :2], WP_COLOR, 5)
        # 반면 **detour는 선만 보면 구분이 안 된다** — 임계가 10 cm(=2셀)라 육안으로 안 보인다.
        # 그래서 우회가 일어난 지점(최대 이탈점)에 주황 링을 찍고 캡션에 이탈량을 적는다.
        det_notes = []
        for rb in r_bs:
            for lg in per_ep[rb]['legs']:
                if lg['status'] == 'detour' and lg['detour_point'] is not None:
                    draw(img, np.asarray([lg['detour_point']]), DETOUR_COLOR, 7)
                    det_notes.append(f"r_b={rb} leg{lg['i']} {lg['detour_max_m'] * 100:.0f} cm")
        draw(img, gt[:1, :2], START_COLOR, 5)
        draw(img, gt[-1:, :2], GOAL_COLOR, 5)
        emit(img, f'wp_ep{ep:03d}.jpg')

        st = ' · '.join(
            f"r_b={rb}: ok {per_ep[rb]['n_ok']}/detour {per_ep[rb]['n_detour']}/blocked "
            f"{per_ep[rb]['n_blocked']} · 프레임 {per_ep[rb]['frames_usable']}/{per_ep[rb]['n_frames']}"
            for rb in r_bs)
        dn = (' · <b>우회</b>: ' + ', '.join(det_notes)) if det_notes else ' · 우회 없음'
        body.append(f'<h3>episode {ep}</h3>'
                    f'<p>waypoint {len(spine["waypoints_mesh"])}개 · leg {len(spine["legs"])}개 · '
                    f'앵커 거리 max {spine["anchor_dist_m"].max():.3f} m · 정합 resid {al["resid"]:.4f} m'
                    f'{dn}<br>{st}</p><img src="wp_ep{ep:03d}.jpg" style="width:70%">')
        print(f'[W] ep{ep}: wp={len(spine["waypoints_mesh"])} 앵커max={spine["anchor_dist_m"].max():.3f} | ' + st)
        rows.append(ep)

    if not rows:
        print('[W] 사용 가능한 에피소드가 없다'); return 1

    ad = np.concatenate(anchor_d)
    w1_ok = ad.mean() < 0.15 and n_spine_ok == n_tried
    base_leg = per_rb[min(r_bs)]
    base_total = base_leg['ok'] + base_leg['detour'] + base_leg['blocked']
    w2_rate = (base_leg['ok'] + base_leg['detour']) / max(base_total, 1)
    w2_ok = w2_rate > 0.95
    # W5 identity — baseline에서 GT를 손대지 않았나. 통과 못 하면 손댄 leg 수를 그대로 보고한다(무마 금지).
    w5_none, w5_fix = base_leg['by_mode']['none'], base_leg['by_mode']['nudge'] + base_leg['by_mode']['corridor']
    w5_ok = w5_none == base_total
    # W6 거짓 승인 — r_b 전체 합. 0이어야 한다.
    fa = sum(per_rb[rb]['cm']['false_accept'] for rb in r_bs)
    fr = sum(per_rb[rb]['cm']['false_reject'] for rb in r_bs)
    n_cm = sum(sum(per_rb[rb]['cm'].values()) for rb in r_bs)
    w6_ok = n_cm > 0 and fa == 0

    def rb_row(rb):
        g = per_rb[rb]
        tag = ' <b>(VLN-CE 정의값)</b>' if abs(rb - BASELINE_R_B) < 1e-9 else ''
        gd = f'{np.mean(g["gt"]) * 100:.1f} cm' if g['gt'] else '—'
        pct = 100 * g['usable'] / max(g['total'], 1)
        nn, nm, co = g['by_mode']['none'], g['by_mode']['nudge'], g['by_mode']['corridor']
        tot = g['ok'] + g['detour'] + g['blocked']
        fix = nm + co                       # 실제로 GT를 손댄 leg 수 (none은 손대지 않았다)
        cm = g['cm']
        w6 = (f'{cm["both"]} / <b>{cm["false_accept"]}</b> / {cm["false_reject"]} / {cm["both_block"]}'
              if sum(cm.values()) else '—')
        return (f'<tr><td>{rb}{tag}</td><td><b>{nn}</b> / {nm} / {co}</td>'
                f'<td><b>{fix}</b>/{tot} ({100 * fix / max(tot, 1):.0f}%)</td>'
                f'<td>{g["ok"]}</td><td>{g["detour"]}</td><td>{g["blocked"]}</td>'
                f'<td>{gd}</td><td><b>{g["usable"]}</b>/{g["total"]} ({pct:.0f}%)</td>'
                f'<td>{w6}</td></tr>')

    def mode_row(m, desc):
        v = f'<b>{np.mean(mode_gt[m]) * 100:.1f} cm</b>' if mode_gt[m] else '—'
        return f'<tr><td><code>{m}</code></td><td>{desc}</td><td>{v}</td></tr>'

    # --- 벽 여유 스타일 (P1~P4): corridor_gtc가 "벽에 안 붙으면서 GT를 따라가나"
    def _agg_clr(key, src=None):
        v = (src or clr_stat).get(key)
        if not v:
            return None
        return dict(gap=float(np.median([x['gap'] for x in v])),
                    low=float(np.mean([x['low'] for x in v])),
                    low_gt=float(np.mean([x['low_gt'] for x in v])))

    clr_rows, verdicts = [], []
    for key in ('corridor', 'corridor_gtc', 'corridor_gtc cap=0.15', 'corridor_gtc fixed+0.15'):
        a = _agg_clr(key)
        if a is None:
            continue
        gd = f'{np.mean(mode_gt[key]) * 100:.1f} cm' if mode_gt.get(key) else '—'
        st = mode_stat.get(key, dict(blocked=0, detour=0))
        clr_rows.append(f'<tr><td><code>{key}</code></td><td>{gd}</td>'
                        f'<td><b>{a["gap"] * 100:.1f} cm</b></td>'
                        f'<td>{100 * a["low"]:.1f}%</td><td>{100 * a["low_gt"]:.1f}%</td>'
                        f'<td>{st["blocked"]}</td><td>{st["detour"]}</td></tr>')
        print(f'[W] 여유 {key:26s} GT거리 {gd:>8s} · |Δ여유| {a["gap"]*100:5.1f} cm · '
              f'경로 여유<r_b+0.05 {100*a["low"]:5.1f}% (GT 자체 {100*a["low_gt"]:.1f}%) · '
              f'blocked {st["blocked"]} detour {st["detour"]}')
    c0, c1 = _agg_clr('corridor'), _agg_clr('corridor_gtc')
    if c0 and c1:
        p1 = c1['gap'] < c0['gap']
        d0 = np.mean(mode_gt['corridor']) if mode_gt.get('corridor') else np.nan
        d1 = np.mean(mode_gt['corridor_gtc']) if mode_gt.get('corridor_gtc') else np.nan
        p2 = np.isfinite(d0) and np.isfinite(d1) and (d1 - d0) <= 0.01
        # 사전 등록한 실패 조건은 "**새** blocked이 생기면"이다 — 감소는 실패가 아니다(개선).
        p3 = (mode_stat['corridor_gtc']['blocked'] <= mode_stat['corridor']['blocked'])
        verdicts += [('P1 스타일: |Δ여유| median이 corridor_gtc < corridor',
                      f'{c1["gap"]*100:.1f} vs {c0["gap"]*100:.1f} cm', p1),
                     ('P2 GT 거리: 악화 ≤ 1 cm',
                      f'{d1*100:.1f} vs {d0*100:.1f} cm (Δ{(d1-d0)*100:+.1f})', p2),
                     ('P3 무손실: 새 blocked 없음 (≤)',
                      f'{mode_stat["corridor_gtc"]["blocked"]} vs {mode_stat["corridor"]["blocked"]}',
                      p3)]
    p4_ok, p4_txt = True, []
    for rb4 in (0.20, 0.30):
        a0, a1 = _agg_clr((rb4, 'corridor'), p4_stat), _agg_clr((rb4, 'corridor_gtc'), p4_stat)
        if a0 and a1:
            p4_ok &= a1['gap'] < a0['gap']
            p4_txt.append(f'r_b={rb4}: {a1["gap"]*100:.1f} vs {a0["gap"]*100:.1f} cm')
    if p4_txt:
        verdicts.append(('P4 r_b 일반화: 큰 r_b에서도 P1 방향 유지', ' · '.join(p4_txt), p4_ok))
    for nm, val, ok in verdicts:
        print(f'[W] {nm}: {val} → {"PASS" if ok else "FAIL"}')

    clr_html = (
        f'<h3>실험결과 2b — 벽 여유 스타일: 벽에 안 붙으면서 GT를 따라가나 (@ r_b={BASELINE_R_B})</h3>'
        '<p>배경([[260820_gt_wall_clearance_result]]): 원본 GT는 벽에서 넉넉하게 가는데(스타일), '
        '우리 생성 경로(<code>refine_min_move</code>)는 여유가 r_b에 붙는다. '
        '<code>corridor_gtc</code>는 목표 여유를 <b>GT 최근접점의 여유</b>로 잡아 민다 — '
        '좁은 문(GT도 여유 없음)은 안 밀고, 넓은 방은 GT 쪽으로 민다. '
        '여유는 <b>스플라인 후 최종 경로</b>에서 잰다.</p>'
        '<table border=1 cellpadding=6><tr><th>mode</th><th>GT와 거리 (W3)</th>'
        '<th>|경로 여유 − GT 여유| median</th><th>경로 여유&lt;r_b+0.05</th>'
        '<th><small>GT 자체</small></th><th>blocked</th><th>detour</th></tr>'
        + ''.join(clr_rows) + '</table>'
        '<table border=1 cellpadding=6><tr><th>예측 (사전 등록)</th><th>실측</th><th>판정</th></tr>'
        + ''.join(f'<tr><td>{nm}</td><td>{val}</td>'
                  f'<td><b>{"PASS" if ok else "FAIL"}</b></td></tr>' for nm, val, ok in verdicts)
        + '</table>'
        '<p>ablation: <code>fixed+0.15</code>는 GT 매칭 없이 고정 마진 — <b>좁은 문에서 blocked가 '
        '생길 것</b>이라는 예측 포함(생기면 GT 매칭이 필요한 이유의 직접 증거). '
        '<code>cap=0.15</code>는 상한 절반.</p>') if clr_rows else ''

    summary = (
        f'<p><b>{args.scene}</b> · vln_ce · rig {args.rig} · 에피소드 <b>{len(rows)}개</b> / leg '
        f'<b>{base_total}개</b> · r_b {r_bs} · 회랑 반경 {args.corridor_m} m</p>'
        f'<p>계획 격자 <b>map_source={args.map_source}</b> — '
        + ('3D 스캔을 <code>h_nav={}</code>~<code>h_obs={}</code> m 밴드로 투영(habitat '
           '<code>agent_max_climb</code>/<code>agent_height</code> 직독값에 정렬)'.format(
               H_NAV_HABITAT_M, H_OBS_HABITAT_M)
           if args.map_source == 'occ' else
           'VLN-CE navmesh를 우리 격자에 래스터화(habitat 판정과 100% 일치, 대신 habitat 의존)')
        + '</p>'

        f'<h3>게이트</h3>'
        f'<table border=1 cellpadding=6><tr><th>게이트</th><th>묻는 것</th><th>결과</th></tr>'
        f'<tr><td><b>W1</b> spine 앵커</td><td>사람 주석 waypoint를 GT 프레임에 붙일 수 있나</td>'
        f'<td>거리 mean <b>{ad.mean():.3f} m</b> (max {ad.max():.3f}) · 채택 {n_spine_ok}/{n_tried} '
        f'→ {"PASS" if w1_ok else "CHECK"}</td></tr>'
        f'<tr><td><b>W2</b> VLN-CE 기준 재현</td>'
        f'<td>r_b={BASELINE_R_B}에서 leg가 거의 다 성공하나<br>'
        f'<small>원본이 이 기준(Ø0.2 m 원기둥, 다음 waypoint 0.5 m 내 도달)으로 만들어졌으므로 ≈100% 기대</small></td>'
        f'<td>leg 성공 <b>{100 * w2_rate:.1f}%</b> ({base_leg["ok"] + base_leg["detour"]}/{base_total}) '
        f'→ {"PASS" if w2_ok else "CHECK"}</td></tr>'
        f'<tr><td><b>W5</b> identity</td>'
        f'<td>r_b={BASELINE_R_B}에서 <b>GT를 건드리지 않나</b><br>'
        f'<small>ladder 밑단 <code>none</code>이 전 leg를 받아야 하고 GT와의 거리가 0이어야 한다</small></td>'
        f'<td><code>none</code> <b>{w5_none}</b>/{base_total} · 보정한 leg <b>{w5_fix}</b> · '
        f'GT 거리 {np.mean(base_leg["gt"]) * 100:.2f} cm → {"PASS" if w5_ok else "CHECK"}'
        + ('<br><small>실패 사유: ' + ' · '.join(f'<code>{k}</code> x{v}'
                                              for k, v in sorted(base_leg['why'].items()))
           + '</small>' if base_leg['why'] else '') + '</td></tr>'
        f'<tr><td><b>W6</b> habitat 오라클</td>'
        f'<td><b>거짓 승인이 0인가</b> — habitat이 막혔다는 leg를 우리가 승인하지 않나<br>'
        f'<small>같은 leg를 캐시된 r_b navmesh로 독립 판정(VLN-CE 기준 0.5 m). 우리 ladder와 무관</small></td>'
        f'<td>거짓 승인 <b>{fa}</b> · 거짓 기각 {fr} · 대조 leg {n_cm}개 → '
        f'{"PASS" if w6_ok else ("CHECK" if n_cm else "캐시 없음 — S1 먼저")}</td></tr></table>'

        f'<h3>실험결과 1 — r_b별 leg 결과와 프레임 커버리지 (W2 / W4 / W6)</h3>'
        f'<table border=1 cellpadding=6><tr><th>r_b</th>'
        f'<th>성공 mode<br><small><b>none</b>=GT 그대로 / nudge=밀어냄 / corridor=회랑 A*</small></th>'
        f'<th><b>보정 필요 leg</b><br><small>nudge+corridor</small></th>'
        f'<th>leg ok</th><th>detour</th><th>blocked</th>'
        f'<th>GT와 거리</th><th>쓸 수 있는 프레임</th>'
        f'<th>W6 habitat 대조<br><small>둘다통과 / <b>거짓승인</b> / 거짓기각 / 둘다막힘</small></th></tr>'
        + ''.join(rb_row(rb) for rb in r_bs) + '</table>'
        f'<p><b>"보정 필요 leg" 열이 이 리포트의 핵심 숫자다</b> — "몇 개 leg가 실제로 GT를 손대야 '
        f'했나"에 하나로 답한다. <code>none</code>은 GT를 <b>그대로</b> 쓴 것이므로 보정에 세지 않는다. '
        f'(이전 판에는 이 열이 없어 <code>nudge 47/47</code>을 "47개를 밀어냈다"로 오독했다 — 실제 '
        f'이동량은 median 0.000 m였다.)</p>'
        f'<p><b>W6이 독립 검증이다.</b> 같은 leg를 <b>habitat navmesh</b>로도 판정해 대조한다(VLN-CE '
        f'논문 기준: 다음 waypoint 0.5 m 내 도달). <b>거짓 승인</b>(우리 통과 ↔ habitat 막힘)은 '
        f'"갈 수 없는 길"을 정답으로 가르치는 것이라 <b>0이어야 한다</b>. 거짓 기각은 샘플 손실이므로 '
        f'수치로만 보고한다.</p>'
        f'<p>마지막 열이 <b>leg 단위 국소화의 이득</b>이다. leg 하나가 막혀도 그 앞 프레임은 학습에 쓸 수 '
        f'있다 — 학습 샘플이 에피소드 전체가 아니라 프레임별 ≤3.25 m 국소 창이기 때문이다. '
        f'전 구간 통과를 요구하면 이 열이 0%가 되는 에피소드가 대부분이다.</p>'

        f'<h3>실험결과 2 — 국소 우회 방식별 GT 충실도 (W3, @ r_b={BASELINE_R_B})</h3>'
        f'<table border=1 cellpadding=6><tr><th>mode</th><th>방법</th><th>GT와 거리</th></tr>'
        + mode_row('none', '<b>GT 서브궤적을 그대로</b>. refine·thin·spline 전부 생략 → '
                           '<b>이탈이 정의상 0</b>')
        + mode_row('nudge', '점별 국소 밀어내기(<code>refine_min_move</code>). 창 밖으로 못 나가 '
                            '<b>재라우팅 불가</b>')
        + mode_row('corridor', f'GT 서브궤적 주변 <b>회랑({args.corridor_m} m) 안에서만</b> A*. '
                               '가구 우회 O, 다른 방 X')
        + mode_row('corridor_gtc', 'corridor + <b>GT 여유 매칭 refine</b> — 목표 여유를 r_b 고정이 '
                                   '아니라 <b>GT의 로컬 여유</b>로 (벽에 안 붙으면서 GT 추종)')
        + mode_row('free', '전체 navigable에서 A*. waypoint 순서만 제약 → 루트 이탈 위험')
        + '</table>'
        + clr_html
        + f'<p>기본값 <code>ladder=(none, nudge, corridor)</code> — 자유도 낮은 쪽부터 시도해 <b>첫 성공에서 '
        f'멈춘다</b>. <b>통과 가능한 leg는 GT를 그대로 쓰고</b>(밑단 <code>none</code>), 밀어내기·회랑은 '
        f'정말 못 지나갈 때만 쓴다. rung을 위에 얹는 것은 커버리지를 낮추지 않는다. '
        f'실패한 leg는 기각하고 원본 baseline으로 폴백한다 — 국소 변형으로 못 뚫리는 구간은 '
        f'"이 embodiment로는 이 instruction의 루트를 따라갈 수 없다"는 물리적 사실이다.</p>'

        f'<h3>방법 — 왜 waypoint를 앵커로 하나</h3>'
        f'<p><code>start→goal</code> 순수 A*는 <b>instruction을 모른다</b>. R2R GT는 사람이 고른 경유지를 '
        f'지나도록 만들어졌는데 A*는 기하적 최단만 찾으므로 다른 방으로 돌아버리고, 그러면 instruction이 '
        f'거짓 라벨이 된다. 정량적 근거: <code>reference_path</code> 길이가 <code>geodesic_distance</code>'
        f'보다 <b>긴</b> 에피소드가 절반 가까이다(17DRP ep95: 5.70 vs 4.94 m) — 그 차이가 instruction을 '
        f'따르느라 돌아가는 부분이고 A*는 그걸 잘라낸다.</p>'
        f'<p><b>출처</b>: waypoint는 R2R(<a href="https://arxiv.org/abs/1711.07280">Anderson et al., '
        f'CVPR 2018</a>)에서 사람이 고른 Matterport3D 파노라마 뷰포인트이고, VLN-CE'
        f'(<a href="https://arxiv.org/abs/2004.02857">Krantz et al., ECCV 2020</a>)가 이를 '
        f'<i>"a ground-based agent represented by a 1.5m tall cylinder of diameter of 0.2m"</i>가 설 수 '
        f'있는 점으로 투영해 <i>waypoint locations</i>를 만든 뒤, <i>"We run this algorithm between each '
        f'waypoint in a trajectory to the next … navigable if … within 0.5 m of the next waypoint"</i>로 '
        f'궤적을 생성했다(R2R 궤적의 77% 통과). 즉 <b>r_b=0.1 · h=1.5 · leg_tol=0.5 m는 우리가 고른 값이 '
        f'아니라 데이터셋의 정의값</b>이고(배포 <code>.navmesh</code> 직독값과 일치), 우리는 그 절차에서 '
        f'<b>r_b만 바꿔 재적용</b>한다.</p>'
        f'<p><b>waypoint 표현</b>: leg별 status를 <b>프레임 플래그 <code>reach_ok</code>'
        f'(0 ok / 1 detour / 2 blocked)</b>로 투영해 둔다. 로더는 창이 걸치는 프레임만 O(1)로 보면 되고, '
        f'같은 캐시로 두 학습 모드를 지원한다 — <b>제외 모드</b>는 창에 2가 있으면 샘플을 버리고, '
        f'<b>도달불가 학습 모드</b>는 2가 처음 나오는 프레임을 stop 라벨로 쓴다(기존 <code>stop_list</code> '
        f'경로 재사용).</p>')

    trouble = (
        f'<p>이 리포트를 만드는 과정에서 <b>틀린 결과를 낸 측정</b>들이다. 본문 수치는 모두 아래를 고친 '
        f'뒤 재측정한 것이다.</p>'
        f'<table border=1 cellpadding=6><tr><th>증상</th><th>원인</th><th>조치</th></tr>'
        f'<tr><td>r_b=0.45에서도 <b>blocked가 0</b>. 모든 leg가 성공</td>'
        f'<td><code>_plan_leg</code>가 <b>끝점 거리만</b> 보고 <code>_path_ok</code> 4조건을 빼먹었다. '
        f'<code>nudge</code>는 실패를 반환하지 않고 점을 조금 밀기만 하므로 끝점이 거의 항상 GT 끝점이라 '
        f'leg_tol(0.5 m)을 통과한다 → <b>벽을 지나는 경로가 전부 성공으로 집계</b></td>'
        f'<td>leg마다 하드 충돌·몸통 침범·미관측 통과·target 도달 4조건 적용</td></tr>'
        f'<tr><td>GT와의 거리가 <b>225 cm / 409 cm</b>로 터짐</td>'
        f'<td>blocked에서 끊긴 <b>부분 경로를 전체 GT와 비교</b>했다. 호길이 리샘플이라 길이가 다르면 '
        f'대응점이 전부 어긋난다</td>'
        f'<td><code>frames_covered</code>로 GT를 잘라 비교 → 5.5 / 8.4 cm</td></tr>'
        f'<tr><td><code>corridor</code>와 <code>free</code>의 GT 거리가 <b>동일(20.9 cm)</b></td>'
        f'<td>회랑 반경 기본값 2.0 m가 leg 길이(median 1.79 m)보다 넓어 <b>제약이 전혀 안 됐다</b></td>'
        f'<td>기본값 0.75 m로 낮춤 → corridor 17.8 cm vs free 25.9 cm로 분리</td></tr>'
        f'<tr><td>정합 residual이 에피소드마다 0.003~0.035 m로 흔들림</td>'
        f'<td><code>vlnce_align.Z_OFFSET_M</code>이 0.20 m로 잘못 설정 — mesh 바닥에 맞추려고 역산한 '
        f'fudge였다. <code>reference_path</code> y가 <b>바닥 높이</b>고 rig 높이는 상대 pose 안에 이미 '
        f'있으므로 offset이 없어야 한다</td>'
        f'<td>0.0으로 정정 → residual 0.0003~0.0004 m. G1 depth 오차도 median 5→0.5 mm</td></tr>'
        f'</table>'
        f'<tr><td>detour 임계 0.10 m가 <b>노이즈를 우회로 오분류</b></td>'
        f'<td>10 cm는 파이프라인 노이즈 하한보다 작다 — 앵커 오차 mean 0.094 m(p90 0.227) · 격자 셀 0.05 · '
        f'<code>PLAN_MARGIN_M</code> 0.05 · <b>반경 증가분 자체</b>(r_b−0.10; 같은 벽을 그만큼 더 떨어져 '
        f'지나면 자동으로 생긴다). 실측 이탈량은 median <b>0.000 m</b>(전 r_b), p90 0.05~0.15</td>'
        f'<td>임계 <b>0.20 m</b>로 상향(임계 0.10에서 11~19% 오분류 → 0.20에서 r_b≥0.15는 0%). '
        f'반경 인식 임계는 불필요 — 이탈량이 r_b와 함께 커지지 않는다(r_b=0.3에서 max 0.05 m)</td></tr>'
        f'<tr><td><code>nudge</code> 모드의 이탈량이 <b>항상 0</b>으로 보고됨</td>'
        f'<td>거리장(<code>field</code>)을 <code>corridor</code> 모드에서만 만들고 나머지는 '
        f'<code>None</code>으로 뒀다. ladder 기본값이 nudge 먼저라 <b>detour가 거의 안 잡히는 구조</b>였다</td>'
        f'<td>mode와 무관하게 거리장을 항상 생성</td></tr>'
        f'<tr><td>회랑 A*가 실제로 쓰이는지 표에서 안 보였다</td>'
        f'<td>corridor로 구제된 leg의 이탈량이 임계 아래라 <code>ok</code>로 분류된다 — 결과만 보면 '
        f'국소 우회가 한 번도 안 일어난 것처럼 보인다(실제로는 17DRP r_b=0.3에서 12%가 corridor)</td>'
        f'<td>표에 <b>성공 mode 열</b> 추가</td></tr>'
        f'<p><b>구조적 교훈</b>: 위 1·2번은 둘 다 "성공/실패를 판정하는 자리에서 검증을 빼먹으면 지표가 '
        f'조용히 좋아진다"는 같은 실패다. 생성 라벨을 다루는 코드에서는 <b>기각 조건을 한 곳'
        f'(<code>_path_ok</code>)에 모아 모든 경로가 반드시 통과</b>하게 두는 것이 맞다.</p>')

    (out / 'summary.html').write_text(summary, encoding='utf-8')
    (out / 'trouble.html').write_text(trouble, encoding='utf-8')
    (out / 'body.html').write_text(''.join(body), encoding='utf-8')
    save_gallery(out, 'report.html', 'W — waypoint 앵커 + leg별 국소 우회', summary, ''.join(body))
    print(f'[W] W1 앵커 mean={ad.mean():.3f}m spine {n_spine_ok}/{n_tried} · '
          f'W2 leg 성공률 @r_b={min(r_bs)}: {100 * w2_rate:.1f}%')
    print(f'[W] W5 identity @r_b={BASELINE_R_B}: none {w5_none}/{base_total} · 보정한 leg {w5_fix} · '
          f'GT거리 {np.mean(base_leg["gt"]) * 100:.2f} cm → {"PASS" if w5_ok else "CHECK"}')
    if base_leg['why']:
        print('      W5 실패 사유: ' + ' · '.join(f'{k} x{v}' for k, v in sorted(base_leg['why'].items())))
    print(f'[W] W6 habitat 오라클: 거짓승인 {fa} · 거짓기각 {fr} · 대조 leg {n_cm}개 → '
          f'{"PASS" if w6_ok else ("CHECK" if n_cm else "캐시없음")}')
    for rb in r_bs:
        g = per_rb[rb]
        print(f'      r_b={rb:.2f} mode none/nudge/corridor = '
              f'{g["by_mode"]["none"]}/{g["by_mode"]["nudge"]}/{g["by_mode"]["corridor"]} · '
              f'혼동 {g["cm"]}')
    print(f'[W] report -> {out}/report.html')
    return 0


def _gt_dist(res, gt):
    """생성 경로와 원본 GT의 평균거리 [m]. **경로가 덮은 프레임 구간으로 GT를 자른다.**

    blocked에서 경로가 끊기므로 전체 GT와 비교하면 지표가 터진다(실측 225 cm/409 cm는 이 오류였다).
    """
    cov = res.get('frames_covered')
    if not len(res['path_xy']) or cov is None:
        return None
    seg = gt[cov[0]:cov[1] + 1, :2]
    if len(seg) < 2:
        return None
    return float(np.mean(np.linalg.norm(_match(res['path_xy'], seg), axis=1)))


def _match(path_xy, gt_xy, n=160):
    """두 경로를 호길이 등간격 n점으로 리샘플해 대응점 차이를 반환. (n,2)"""
    def rs(t):
        t = np.asarray(t, dtype=np.float64)[:, :2]
        if len(t) < 2:
            return np.repeat(t, n, axis=0)[:n]
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(t, axis=0), axis=1))]
        if d[-1] < 1e-9:
            return np.repeat(t[:1], n, axis=0)
        u = np.linspace(0, d[-1], n)
        return np.stack([np.interp(u, d, t[:, 0]), np.interp(u, d, t[:, 1])], axis=1)
    return rs(path_xy) - rs(gt_xy)


def _rb_color(rb, r_bs):
    palette = [(90, 190, 255), (255, 210, 60), (255, 90, 90), (0, 255, 90), (200, 120, 255)]
    return palette[r_bs.index(rb) % len(palette)]


if __name__ == '__main__':
    raise SystemExit(main())
