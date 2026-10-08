"""R9 — waypoint만 주면 GT를 얼마나 재현하나 (navmesh 맵 vs recast v2 맵).

GT 궤적은 planner에 **주지 않는다** — 입력은 사람 주석 waypoint(`reference_path`) 좌표뿐이고,
GT는 평가에만 쓴다. 두 맵은 **bool 마스크만 공급**하며, 마스크→(esdf, navigable) 변환·계획·
후처리는 맵과 무관한 단일 코드 경로다(map별 생성 분기 없음). r_b = 0.10 고정(VLN-CE 정의값).

| 게이트 | 무엇을 묻나 | 통과 기준 |
|---|---|---|
| G1 | GT 점 생존율(마스크 안 비율) — 맵 자체 확인 | ≥ 99% (두 맵 모두) |
| G2 | P0(waypoint 체인 A*) 전 leg 연결 | A* 실패 0 |
| G3 | 최선 연속 변형의 gt_dist median | ≤ 15 cm/씬 (노이즈 플로어 병기) |
| G4 | 두 맵의 gt_dist(최선 변형) 차 | ≤ 5 cm (planner가 지배 변수인가) |
| G5 | TL비율 ≤ 1.05 & SPL-if-followed ≥ GT 자신 − 0.02 | 경로 길이 효율 |

## GT는 어떻게 만들어졌나 (논문·코드로 확인 — 추정 아님)
- VLN-CE (Krantz et al., ECCV 2020, arXiv:2004.02857): 로봇 = *"1.5m tall cylinder of diameter
  of 0.2m"*. R2R nav-graph 노드를 navmesh에 스냅(2 m 하향 레이캐스트, 수평 변위 ≤ 0.5 m).
  waypoint 사이마다 *"an A*-based heuristic search algorithm to compute an approximate shortest
  path"* 를 돌려 *"within 0.5 m of the next waypoint"* 면 navigable로 판정(77% 전이 성공).
- `gt.json.gz`의 locations/actions = 그 최단경로를 구식 `ShortestPathFollowerCompat`
  (0.25 m 전진/15° 회전, 매 스텝 `get_straight_shortest_path_points` 재계획)로 실행한 흔적
  (github.com/jacobkrantz/VLN-CE `habitat_extensions/shortest_path_follower.py`).
→ **GT = "waypoint 경유 최단경로 + 이산 실행"** — waypoint만으로 근사 재현이 가능해야 한다.
  양자화(0.25 m/15°) 출력은 이번 범위에서 제외(사용자 결정, 추후 적용 가능). 단 GT 자신이
  양자화 실행 흔적이라 매끈한 경로의 gt_dist에는 잔물결 크기의 **노이즈 플로어**가 있다 —
  GT vs smooth(GT)로 같이 재서 해석 기준으로 쓴다.

## 변형 사다리 (사전 고정 — gt_dist 최소화가 목표, 두 맵 동일)
- P0: waypoint 체인 grid A*(`esdf_utils.astar`, 8-이웃) 그대로 — 후처리 없음.
- P1: P0 + `thin_waypoints`(0.8 m) + `smooth_cubic_spline` — 격자 계단 제거(03 실측 채택값).
- P2(w): P1 + A* `clearance_weight=w` — GT가 순수 최단경로보다 벽에서 먼 스타일(#04)이라
  벽에서 살짝 미는 게 gt_dist를 줄이는지 실측. clearance는 `CLR_CAP_M`으로 캡해서
  넓은 방에서 비용이 0으로 퇴화(개활지 우회)하는 것을 막는다.

실행:
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/09_waypoint_replan.py --scene 17DRP5sb8fy
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/09_waypoint_replan.py --scene s8pcmisQ38h

self-check(씬·habitat 불필요): /usr/bin/python scripts/dataset_converters/3dloader_vlnce/09_waypoint_replan.py --selfcheck
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
    astar, cell_to_world, check_path_navigable, resample_by_arclength, sample_esdf_at,
    smooth_cubic_spline, thin_waypoints, truncate_navigable, world_to_cell,
)
from viz_utils import floorplan_canvas, save_gallery  # noqa: E402
from embodiment_augment import PLAN_MARGIN_M, EmbodimentAugmenter, densify, snap_to_grid  # noqa: E402
from waypoint_spine import LEG_TOL_M, build_spine  # noqa: E402
import navmesh_grid  # noqa: E402
import recast_like  # noqa: E402
_m00 = importlib.import_module('00_verify_pose_mesh')  # noqa: E402
load_episode = _m00.load_episode

R_B = 0.10                 # VLN-CE 정의값 (cylinder diameter 0.2 m)
SPLIT_TOL_M = 0.20         # 03/04와 동일 — 대응점 거리가 이보다 크면 "다른 루트"
SUCCESS_RADIUS_M = 3.0     # VLN-CE SR 판정 반경
THIN_SPACING_M = 0.8       # 03 실측 채택값 (스무딩 전 솎기)
CLR_CAP_M = 0.30           # clearance_weight에 먹일 esdf 상한 — gt_clearance_refine.CAP_M과 같은
                           # 근거(GT가 r_b 위로 유지하는 여유의 상한). 캡이 없으면 개활지(esdf≥2 m)
                           # 비용이 1e-6으로 퇴화해 A*가 개활지로 크게 우회한다.
MAPS = ('navmesh', 'recast')

GT_COLOR = (255, 255, 255)
WP_COLOR = (0, 220, 255)
P0_COLOR = (90, 190, 255)
P1_COLOR = (0, 255, 90)
P2_COLOR = (255, 90, 90)
START_COLOR, GOAL_COLOR = (255, 200, 0), (255, 0, 255)


# ---------------------------------------------------------------------------
# 맵 — bool 마스크만 공급한다. 이후는 전부 공용 경로.
# ---------------------------------------------------------------------------
def build_mask(map_name, ctx, navmesh_dir, scene, r_b, floor_y, start_xy):
    """맵 이름 -> (Ny,Nx) bool 마스크. 두 맵 모두 r_b로 이미 깎인 configuration space다."""
    cell = float(ctx['args'].cell_m)
    if map_name == 'navmesh':
        return navmesh_grid.load_mask(navmesh_dir, scene, r_b, floor_y, ctx['origin'], cell,
                                      ctx['coverage'].shape)
    elif map_name == 'recast':
        grid = ctx.get('_recast_grid')
        if grid is None:      # cumsum 캐시(_recast_csum)가 이 dict에 상주하므로 씬당 1회만 만든다
            grid = dict(occ=ctx['occ'], origin=ctx['origin'], cell=cell, coverage=ctx['coverage'])
            ctx['_recast_grid'] = grid
        return recast_like.recast_mask(grid, floor_y, r_b, start_xy=start_xy)
    else:
        assert False, f'unreachable map_name={map_name!r}'


def grids_from_mask(mask, cell, r_b, coverage):
    """마스크 -> (esdf, navigable). **두 맵 공용** — 여기서부터 생성 코드는 맵을 모른다."""
    esdf = navmesh_grid.esdf_from_mask(mask, cell, r_b)
    navigable = truncate_navigable(esdf, float(r_b) + float(PLAN_MARGIN_M)) & coverage
    return esdf, navigable


# ---------------------------------------------------------------------------
# 생성 — waypoint 체인 A* (+옵션 스무딩·clearance tie-break). GT는 안 들어온다.
# ---------------------------------------------------------------------------
def chain_astar_legs(navigable, esdf, origin, cell, wps_xy, clearance_weight=0.0):
    """waypoint 체인 A*. -> (legs [(Ni,2) world xy or None], snap_d (T,) or None)

    03의 교훈: 실패한 leg를 vstack으로 이어붙이면 가짜 벽 뚫기가 생긴다 — leg별로 반환하고
    합치기는 호출부가 전 leg 성공일 때만 한다. waypoint는 navigable 밖이면 최근접 셀로 스냅
    (VLN-CE 자신도 0.5 m 스냅을 썼다). 스냅 실패(반경 내 navigable 없음)면 None.
    """
    snapped, snap_d = [], []
    for w in np.asarray(wps_xy, dtype=np.float64)[:, :2]:
        s = snap_to_grid(navigable, origin, cell, w, max_radius_m=LEG_TOL_M)
        if s is None:
            return None, None
        snapped.append(s); snap_d.append(float(np.linalg.norm(s - w)))
    cells = world_to_cell(np.asarray(snapped), origin, cell)
    clr = np.minimum(esdf, CLR_CAP_M) if clearance_weight else None
    legs = []
    for a, b in zip(cells[:-1], cells[1:]):
        p = astar(navigable, a, b, clearance=clr, clearance_weight=clearance_weight)
        legs.append(None if p is None else cell_to_world(p, origin, cell))
    return legs, np.asarray(snap_d)


def gen_path(navigable, esdf, origin, cell, wps_xy, smooth, clearance_weight=0.0):
    """변형 하나를 생성. -> dict(path_xy, legs, snap_d, n_fail) — path_xy는 전 leg 성공일 때만."""
    legs, snap_d = chain_astar_legs(navigable, esdf, origin, cell, wps_xy, clearance_weight)
    if legs is None:
        return dict(path_xy=None, legs=[], snap_d=None, n_fail=len(wps_xy) - 1)
    n_fail = sum(1 for l in legs if l is None)
    if n_fail:
        return dict(path_xy=None, legs=legs, snap_d=snap_d, n_fail=n_fail)
    raw = np.vstack([legs[0]] + [l[1:] for l in legs[1:]])
    path = np.asarray(smooth_cubic_spline(thin_waypoints(raw, THIN_SPACING_M), cell)) if smooth else raw
    return dict(path_xy=path, legs=legs, snap_d=snap_d, n_fail=0)


# ---------------------------------------------------------------------------
# 평가 — GT는 여기서만 쓴다.
# ---------------------------------------------------------------------------
def match_dists(a_xy, b_xy, n=200):
    """호길이 등간격 대응점 거리 (n,). 점 분포가 다른 두 경로는 인덱스로 비교하면 안 된다(03)."""
    ra, rb_ = resample_by_arclength(np.asarray(a_xy)[:, :2], n=n), \
        resample_by_arclength(np.asarray(b_xy)[:, :2], n=n)
    return np.linalg.norm(ra - rb_, axis=1)


def arclen(p):
    p = np.asarray(p, dtype=np.float64)
    return float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def noise_floor(gt_xy, cell):
    """GT vs smooth(GT) 대응점 거리 median — 매끈한 경로가 도달 가능한 gt_dist 하한.

    GT는 0.25 m/15° 이산 실행 흔적이라 잔물결이 있다. 우리 출력은 매끈한 스플라인이므로
    이 플로어 아래로는 정의상 못 내려간다.
    """
    sm = smooth_cubic_spline(thin_waypoints(np.asarray(gt_xy)[:, :2], THIN_SPACING_M), cell)
    return float(np.median(match_dists(gt_xy, sm)))


def eval_variant(res, gt_xy, spine, l_direct):
    """생성 결과 하나를 GT로 채점. -> dict or None(생성 실패).

    split_legs는 **스무딩 전 leg별 A* 경로**를 그 leg의 GT 서브궤적과 비교한다 — 전체 경로
    비교는 어긋난 구간을 평균이 희석하기 때문(#03: length ratio ≈ 1이 route-split을 숨겼다).
    """
    if res['path_xy'] is None:
        return None
    d = match_dists(res['path_xy'], gt_xy)
    p_len, g_len = arclen(res['path_xy']), arclen(gt_xy)
    split, leg_d = 0, []
    for lg, (lo, hi) in zip(res['legs'], spine['legs']):
        seg = np.asarray(gt_xy)[lo:hi + 1, :2]
        if lg is None or len(seg) < 2:
            continue
        m = float(np.mean(match_dists(lg, seg, n=80)))
        leg_d.append(m)
        split += int(m > SPLIT_TOL_M)
    end_err = float(np.linalg.norm(np.asarray(res['path_xy'])[-1] - np.asarray(gt_xy)[-1, :2]))
    return dict(gt_med=float(np.median(d)), gt_mean=float(d.mean()), gt_max=float(d.max()),
                frac_split=float(np.mean(d > SPLIT_TOL_M)), split_legs=split, n_legs=len(leg_d),
                leg_d=leg_d, len_ratio=(p_len / g_len if g_len > 0 else float('nan')),
                spl=(l_direct / max(l_direct, p_len) if p_len > 0 else 0.0),
                success=bool(end_err <= SUCCESS_RADIUS_M),
                snap_max=float(res['snap_d'].max()) if res['snap_d'] is not None else float('nan'))


def gt_survival(mask, origin, cell, gt_xy):
    """GT 점 생존율 — 맵 자체 확인(G1). densify 후 마스크 안 비율."""
    pts = densify(np.asarray(gt_xy)[:, :2], cell)
    ij = world_to_cell(pts, origin, cell)
    h, w = mask.shape
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    inside = np.zeros(len(ij), dtype=bool)
    inside[ok] = mask[ij[ok, 1], ij[ok, 0]]
    return int(inside.sum()), int(len(inside))


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='s8pcmisQ38h')
    ap.add_argument('--rig', default='125cm_0deg')
    ap.add_argument('--episodes', type=int, default=14)
    ap.add_argument('--n_frames', type=int, default=40)
    ap.add_argument('--max_z_spread', type=float, default=0.30,
                    help='카메라 높이 변동 상한[m]. 초과 = 계단 → 단일 높이 절단 불가')
    ap.add_argument('--maps', default='navmesh,recast')
    ap.add_argument('--clearance_weights', default='0.5,1,2',
                    help='P2 스윕 값들. clearance는 CLR_CAP_M으로 캡한 esdf')
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_n1')
    ap.add_argument('--esdf_dir', default='scripts/dataset_converters/gs_vlnpe/logs/esdf')
    ap.add_argument('--geo_dir', default='data/embodiment_aug/scene_geo')
    ap.add_argument('--navmesh_dir', default=navmesh_grid.DEFAULT_NAVMESH_DIR)
    ap.add_argument('--out_dir', default='logs/embodiment_augment/waypoint_replan')
    ap.add_argument('--selfcheck', action='store_true')
    args = ap.parse_args()
    if args.selfcheck:
        return _selfcheck()

    out = Path(args.out_dir) / args.scene
    out.mkdir(parents=True, exist_ok=True)
    maps = [m.strip() for m in args.maps.split(',') if m.strip()]
    weights = [float(w) for w in args.clearance_weights.split(',') if w.strip()]
    variants = [('p0', False, 0.0), ('p1', True, 0.0)] + [(f'p2_w{w:g}', True, w) for w in weights]

    aug = EmbodimentAugmenter(args.data_root, args.raw_root, args.mesh_root, args.esdf_dir,
                              args.geo_dir)
    from vlnce_align import build_rawdata_index
    raw_idx = build_rawdata_index(Path(args.raw_root))
    tasks = [json.loads(l) for l in open(Path(args.data_root) / args.scene / 'meta' / 'episodes.jsonl')]
    ctx = aug._scene_ctx(args.scene)
    cell = float(ctx['args'].cell_m)

    # 집계: agg[map][variant] = eval dict 리스트
    agg = {m: {v[0]: [] for v in variants} for m in maps}
    surv = {m: [0, 0] for m in maps}
    floors, body, n_used = [], [], 0

    for ep in range(min(args.episodes, len(tasks))):
        instr = tasks[ep]['tasks'][0]
        try:
            frames = load_episode(args.data_root, args.scene, args.rig, ep, args.n_frames)
            al = aug.align(args.scene, ep, frames, instr)
        except Exception as e:
            print(f'[R9] ep{ep}: skip ({type(e).__name__})'); continue
        gt = np.stack([(al['T'] @ p)[:3, 3] for _f, p, _d, _v in frames])
        z_spread = float(gt[:, 2].max() - gt[:, 2].min())
        if z_spread > args.max_z_spread:
            print(f'[R9] ep{ep}: skip (높이 변동 {z_spread:.2f} = 계단)'); continue
        raw = raw_idx.get((args.scene, instr.strip()))
        if raw is None:
            print(f'[R9] ep{ep}: skip (raw 매칭 실패)'); continue
        spine = build_spine(raw['reference_path'], gt)
        if not spine['ok']:
            print(f"[R9] ep{ep}: skip (spine — {spine['reason']})"); continue
        wps = np.asarray(spine['waypoints_mesh'])[:, :2]     # planner 입력은 이것뿐
        fz = float(np.median(np.asarray(raw['reference_path'], dtype=np.float64)[:, 1]))
        n_used += 1
        floors.append(noise_floor(gt[:, :2], cell))

        imgs, notes = [], []
        for m in maps:
            mask = build_mask(m, ctx, args.navmesh_dir, args.scene, R_B, fz, start_xy=wps[0])
            s_in, s_all = gt_survival(mask, ctx['origin'], cell, gt[:, :2])
            surv[m][0] += s_in; surv[m][1] += s_all
            esdf, navigable = grids_from_mask(mask, cell, R_B, ctx['coverage'])
            # SPL의 l = 같은 맵 start→goal 최단경로 길이 (waypoint 경유 아님 — SPL 정의 그대로)
            dl, _sd = chain_astar_legs(navigable, esdf, ctx['origin'], cell, wps[[0, -1]])
            l_direct = arclen(dl[0]) if dl is not None and dl[0] is not None else float('nan')
            note = [f'<b>{m}</b> GT생존 {s_in}/{s_all}']
            base, draw, emit = floorplan_canvas(navigable, ctx['origin'], cell, out)
            img = base.copy()
            draw(img, densify(gt[:, :2], cell), GT_COLOR, 2)
            for name, smooth, w in variants:
                res = gen_path(navigable, esdf, ctx['origin'], cell, wps, smooth, w)
                ev = eval_variant(res, gt[:, :2], spine, l_direct)
                agg[m][name].append(ev)
                if ev is None:
                    note.append(f'{name}: <b>실패</b>(leg {res["n_fail"]})')
                    continue
                note.append(f'{name}: {ev["gt_med"] * 100:.1f} cm')
                color = {'p0': P0_COLOR, 'p1': P1_COLOR}.get(name, P2_COLOR)
                if name in ('p0', 'p1') or name == variants[-1][0]:
                    draw(img, res['path_xy'], color, 1)
            draw(img, wps, WP_COLOR, 5)
            draw(img, gt[:1, :2], START_COLOR, 5)
            draw(img, gt[-1:, :2], GOAL_COLOR, 5)
            emit(img, f'ep{ep:03d}_{m}.jpg')
            imgs.append(f'<img src="ep{ep:03d}_{m}.jpg" style="width:48%">')
            notes.append(' · '.join(note))
        body.append(f'<h3>episode {ep}</h3><p>waypoint {len(wps)}개 · 플로어 '
                    f'{floors[-1] * 100:.1f} cm<br>{"<br>".join(notes)}</p>' + ''.join(imgs))
        print(f'[R9] ep{ep}: wp={len(wps)} 플로어={floors[-1] * 100:.1f}cm | '
              + ' | '.join(notes).replace('<b>', '').replace('</b>', ''))

    if not n_used:
        print('[R9] 사용 가능한 에피소드가 없다'); return 1

    # --- 집계 + 게이트 ------------------------------------------------------
    floor_med = float(np.median(floors))
    summary = dict(scene=args.scene, n_ep=n_used, floor_med=floor_med, maps={})
    for m in maps:
        summary['maps'][m] = dict(survival=surv[m][0] / max(surv[m][1], 1), variants={})
        for name, _s, _w in variants:
            evs = [e for e in agg[m][name] if e is not None]
            n_fail = sum(1 for e in agg[m][name] if e is None)
            if not evs:
                summary['maps'][m]['variants'][name] = dict(n_fail=n_fail); continue
            # leg 단위 분해: split leg(> SPLIT_TOL_M)를 빼고 남는 잔차 — planner 문제와
            # 루트 모호성(waypoint 사이 장애물 섬) 문제를 분리해 읽기 위함.
            leg_all = [d for e in evs for d in e['leg_d']]
            leg_ok = [d for d in leg_all if d <= SPLIT_TOL_M]
            summary['maps'][m]['variants'][name] = dict(
                n=len(evs), n_fail=n_fail,
                leg_med_ok=float(np.median(leg_ok)) if leg_ok else float('nan'),
                leg_med_split=float(np.median([d for d in leg_all if d > SPLIT_TOL_M]))
                if len(leg_ok) < len(leg_all) else float('nan'),
                gt_med=float(np.median([e['gt_med'] for e in evs])),
                gt_mean=float(np.mean([e['gt_mean'] for e in evs])),
                gt_max=float(np.max([e['gt_max'] for e in evs])),
                split_legs=int(np.sum([e['split_legs'] for e in evs])),
                n_legs=int(np.sum([e['n_legs'] for e in evs])),
                len_ratio=float(np.median([e['len_ratio'] for e in evs])),
                spl=float(np.median([e['spl'] for e in evs])),
                success=float(np.mean([e['success'] for e in evs])),
                snap_max=float(np.nanmax([e['snap_max'] for e in evs])))

    def best_variant(m):
        vs = {k: v for k, v in summary['maps'][m]['variants'].items() if 'gt_med' in v and not v['n_fail']}
        return min(vs.items(), key=lambda kv: kv[1]['gt_med']) if vs else (None, None)

    g1 = all(summary['maps'][m]['survival'] >= 0.99 for m in maps)
    g2 = all(summary['maps'][m]['variants']['p0'].get('n_fail', 1) == 0 for m in maps)
    bests = {m: best_variant(m) for m in maps}
    g3 = all(b[1] is not None and b[1]['gt_med'] <= 0.15 for b in bests.values())
    g4 = (len(maps) < 2 or (all(b[1] is not None for b in bests.values())
          and abs(bests[maps[0]][1]['gt_med'] - bests[maps[1]][1]['gt_med']) <= 0.05))
    g5 = all(b[1] is not None and b[1]['len_ratio'] <= 1.05 and b[1]['spl'] >= 1.0 - 0.02
             for b in bests.values())
    summary['gates'] = dict(G1=g1, G2=g2, G3=g3, G4=g4, G5=g5)
    summary['best'] = {m: bests[m][0] for m in maps}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    # --- 리포트 --------------------------------------------------------------
    rows = []
    for m in maps:
        for name, _s, _w in variants:
            v = summary['maps'][m]['variants'][name]
            if 'gt_med' not in v:
                rows.append(f'<tr><td>{m}</td><td>{name}</td>'
                            f'<td colspan=7>실패 {v["n_fail"]}에피소드</td></tr>')
                continue
            hl = ' style="font-weight:bold"' if name == bests[m][0] else ''
            rows.append(f'<tr{hl}><td>{m}</td><td>{name}</td><td>{v["gt_med"] * 100:.1f}</td>'
                        f'<td>{v["gt_mean"] * 100:.1f}</td><td>{v["gt_max"] * 100:.0f}</td>'
                        f'<td>{v["split_legs"]}/{v["n_legs"]}</td><td>{v["len_ratio"]:.3f}</td>'
                        f'<td>{v["spl"]:.3f}</td><td>{v["n_fail"]}</td></tr>')
    gates_html = ' · '.join(f'{k} {"✅" if v else "❌"}' for k, v in summary['gates'].items())
    summary_html = (
        f'<p>씬 <b>{args.scene}</b> · 에피소드 {n_used} · r_b={R_B} · '
        f'노이즈 플로어(GT vs smooth GT) median <b>{floor_med * 100:.1f} cm</b></p>'
        f'<p>GT 생존율: ' + ' · '.join(f'{m} <b>{100 * summary["maps"][m]["survival"]:.2f}%</b>'
                                      for m in maps) + f'</p><p>{gates_html}</p>'
        '<table border=1 cellpadding=4><tr><th>맵</th><th>변형</th><th>gt_med[cm]</th>'
        '<th>gt_mean[cm]</th><th>gt_max[cm]</th><th>split leg</th><th>len비</th>'
        '<th>SPL</th><th>실패</th></tr>' + ''.join(rows) + '</table>')
    # summary.html/body.html 조각은 발행기(publish_artifact_report.py)가 읽는다 (발행 규약 #4).
    (out / 'summary.html').write_text(summary_html, encoding='utf-8')
    (out / 'body.html').write_text(''.join(body), encoding='utf-8')
    save_gallery(out, 'report.html', 'R9 — waypoint만으로 GT 재현 (navmesh vs recast v2)',
                 summary_html, ''.join(body))

    print(f'[R9] 플로어 median {floor_med * 100:.1f} cm · 게이트 {gates_html.replace("✅", "PASS").replace("❌", "FAIL")}')
    for m in maps:
        print(f'[R9] {m}: 생존 {100 * summary["maps"][m]["survival"]:.2f}% · best={bests[m][0]} '
              + ' · '.join(f'{k}={v["gt_med"] * 100:.1f}cm' for k, v in
                           summary['maps'][m]['variants'].items() if 'gt_med' in v))
    print(f'[R9] report -> {out}/report.html')
    return 0


# ---------------------------------------------------------------------------
# self-check — 합성 마스크로 생성·평가 로직을 검사한다 (씬·habitat 불필요).
# ---------------------------------------------------------------------------
def _selfcheck():
    cell, n = 0.05, 120
    origin = np.zeros(3)
    # 6 m x 6 m 방, 가운데 세로 벽(문 하나)
    mask = np.zeros((n, n), dtype=bool)
    mask[10:110, 10:110] = True
    mask[10:110, 58:62] = False          # x=2.9~3.1 벽
    mask[55:65, 58:62] = True            # y=2.75~3.25 문
    coverage = np.ones_like(mask)
    esdf, navigable = grids_from_mask(mask, cell, R_B, coverage)

    # (1) 체인 A*가 문을 지나 연결된다
    wps = np.array([[1.0, 3.0], [3.0, 3.0], [5.0, 3.0]])
    res0 = gen_path(navigable, esdf, origin, cell, wps, smooth=False)
    assert res0['path_xy'] is not None and res0['n_fail'] == 0, '체인 A* 실패'
    # (2) 경로 전체가 navigable 안 (하드 충돌 0)
    chk = check_path_navigable(res0['path_xy'], esdf, origin, cell, R_B)
    assert chk['hard_ok'], f'P0가 마스크를 뚫었다 {chk}'
    # (3) 스무딩 후에도 유효 + 끝점 유지
    res1 = gen_path(navigable, esdf, origin, cell, wps, smooth=True)
    assert res1['path_xy'] is not None
    assert np.linalg.norm(res1['path_xy'][0] - wps[0]) < 0.10, '스무딩이 시작점을 옮겼다'
    assert np.linalg.norm(res1['path_xy'][-1] - wps[-1]) < 0.10, '스무딩이 끝점을 옮겼다'
    # (4) 자기 자신과의 gt_dist = 0, 평가 지표 sane
    fake = dict(path_xy=res1['path_xy'], legs=res0['legs'], snap_d=res0['snap_d'], n_fail=0)
    ev = eval_variant(fake, res1['path_xy'], dict(legs=[]), arclen(res1['path_xy']))
    assert ev['gt_med'] < 1e-9 and ev['spl'] > 0.999 and ev['success'], f'항등 평가 실패 {ev}'
    # (5) GT를 평행이동하면 gt_med가 그만큼 나온다
    ev2 = eval_variant(fake, res1['path_xy'] + [0.0, 0.30], dict(legs=[]), arclen(res1['path_xy']))
    assert abs(ev2['gt_med'] - 0.30) < 0.02, f'평행이동 평가 {ev2["gt_med"]}'
    # (6) split 검출: leg GT를 반대편으로 옮기면 split_legs가 선다
    gt_fake = np.vstack([res0['legs'][0] + [0.0, 0.5], res0['legs'][1]])
    sp = dict(legs=[(0, len(res0['legs'][0]) - 1),
                    (len(res0['legs'][0]) - 1, len(gt_fake) - 1)])
    ev3 = eval_variant(fake, gt_fake, sp, arclen(gt_fake))
    assert ev3['split_legs'] >= 1, f'split 미검출 {ev3}'
    # (7) clearance_weight가 경로를 벽에서 밀어낸다 (median clearance 증가)
    resw = gen_path(navigable, esdf, origin, cell, wps, smooth=False, clearance_weight=2.0)
    c0 = np.median(sample_esdf_at(esdf, res0['path_xy'], origin, cell))
    cw = np.median(sample_esdf_at(esdf, resw['path_xy'], origin, cell))
    assert cw >= c0 - 1e-9, f'clearance_weight 효과 없음 {c0} -> {cw}'
    # (8) 노이즈 플로어: 직선 GT는 플로어 ~0, 지그재그 GT는 > 0
    line = np.stack([np.linspace(1, 5, 80), np.full(80, 3.0)], axis=1)
    assert noise_floor(line, cell) < 0.01, '직선 플로어가 0이 아니다'
    zig = line.copy(); zig[1::2, 1] += 0.05
    assert noise_floor(zig, cell) > 0.005, '지그재그 플로어가 0이다'
    # (9) 생존율: 마스크 안 경로 100%, 벽 관통 경로 < 100%
    s_in, s_all = gt_survival(mask, origin, cell, line)
    assert s_in == s_all, '마스크 안 GT 생존율이 100%가 아니다'
    bad = np.stack([np.linspace(1, 5, 80), np.full(80, 0.2)], axis=1)   # 방 밖
    b_in, b_all = gt_survival(mask, origin, cell, bad)
    assert b_in < b_all, '마스크 밖 GT가 생존 판정됐다'

    print('[selfcheck] 09_waypoint_replan 9/9 통과 '
          '(체인A*·하드충돌0·스무딩끝점·항등평가·평행이동·split·clearance밀기·플로어·생존율)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
