"""#03 — **데이터셋이 실제로 쓴 지도에 두 경로를 올려 "벽을 뚫는가"를 본다.**

## 왜 이걸 먼저 하나
이 폴더는 **로봇 크기(`r_b`)를 바꿨을 때 정답 경로를 다시 그려주는 코드**다. 그러면 순서가 정해진다 —
**큰 로봇용을 새로 그리기 전에, 원래 크기에서 원래 정답을 그대로 뽑아낼 수 있는지 먼저 확인해야 한다.**
#03이 그 확인이고, 그것뿐이다. #04가 우리 맵을 이 지도에 맞추고, #05가 그 위에서 `r_b`를 흔든다.

## 두 경로 — 배포된 `<scan>.navmesh` 위에 올린다
| | 경로 | 색 | 무엇인가 |
|---|---|---|---|
| ⓐ | **원본 정답** | 노랑 | 저장된 프레임별 위치. **학습이 실제로 먹는 라벨** |
| ⓑ | **waypoint를 이어 만든 경로** | 주황 | leg별 `find_path`. **VLN-CE가 데이터셋을 만든 절차 그 자체** |

`r_b = 0.10`은 우리가 고른 값이 아니다 — VLN-CE 논문이 *"1.5 m tall cylinder of diameter of 0.2 m"*로
못박았고 배포 navmesh를 직독하면 정말 그 값이다. **"원본 파라미터를 따른다" = 바꿀 값이 0개**이고,
그것을 확정하는 것이 게이트 A다.

## 게이트 4개
| 게이트 | 묻는 것 | 기준 |
|---|---|---|
| A 파라미터 | 배포 navmesh가 habitat 기본값으로 만들어졌나 | `agent_radius` 0.10 · `agent_height` 1.50 불일치 0 |
| B 벽 뚫기 | ⓐ·ⓑ가 지도를 벗어나나 | 5 cm로 채운 점 `is_navigable` ≥99% |
| C leg 도달 | 이어붙인 경로가 다음 waypoint에 닿나 | `find_path` 끝점 ≤0.5 m **100%** (VLN-CE 논문 자신의 기준) |
| D 캐시 | `recompute_navmesh`가 배포본을 재현하고 `r_b`별로 저장되나 | area 오차 <1% |

## 진단 2개 (게이트가 아니다)
- **ⓑ는 ⓐ를 얼마나 재현하나** — 호길이 리샘플 대응점 거리 + 길이비. ⚠️ **길이비≈1은 "같은 경로"의
  증거가 아니다** — 두 우회로 길이가 비슷하면 측지 최단선이 **장애물의 반대편**으로 돈다(s8 median
  56 cm인데 길이비 1.002). 그래서 회랑 중심선은 GT 서브궤적이어야 한다.
- **"뚫는다"가 어디서 일어나나** — J1 `is_navigable`(진실) vs J2 `get_topdown_view` 5 cm 래스터
  (우리가 그리고 계획할 때 보는 것). **J1 통과·J2 막힘인 점은 전부 깊이 1칸이어야 한다**는 술어로 낸다.
  2칸 이상이 하나라도 나오면 격자 반올림이 아니라 지도가 실제로 다른 것이다.

## `clearance` 분석은 지웠다
navmesh는 **이미 `agent_radius` 0.10만큼 깎인** configuration space다. 물어야 할 것은 "칸 안이냐"뿐이고,
거기에 `clearance >= r_b`를 또 요구하면 **반경을 두 번 센다**(`navmesh_grid.esdf_from_mask`가
`esdf = 칸안 ? 거리 + r_b : 0`을 하는 이유). 이 분석에서 나온 결론은 이미 철회됐다
(`.claude/memory/260820_clearance_gt_vs_findpath_result.md` — `min`이라는 최악의 한 점을 전형값으로 읽음).

## 좌표 규약
- `reference_path` / `start_position` = habitat 세계좌표 (Y-up). navmesh API가 바로 먹는다.
- 저장 pose = 에피소드 시작 상대 mesh 좌표(Z-up) → `T_sf2mesh`로 mesh 절대 → habitat 역변환
  `(m_x, m_z - h_rig, -m_y)`. `h_rig`를 빼는 이유: pose는 **카메라**, navmesh는 **바닥**을 판정한다.
- `get_topdown_view(mpp, h)` → `tv[iz, ix]`, `ix=(x-lo[0])/mpp`, `iz=(z-lo[2])/mpp` (`lo=get_bounds()[0]`).
  **단일 높이 절단**이라 에피소드별 floor로 잘라야 하고, 계단 에피소드는 제외한다.
- `T_sf2mesh`가 **전부 해석적**이라(`vlnce_align.fit_sf2mesh`) 씬 mesh·Open3D가 필요 없다.
  우리 플래너와 우리 occ 맵은 이 리포트 범위 밖이다(#04 / #05).

실행: /usr/bin/python scripts/dataset_converters/3dloader_vlnce/03_verify_navmesh_gt.py --scenes 17DRP5sb8fy,s8pcmisQ38h --r_bs 0.10,0.15,0.20,0.30,0.45 --episodes 14 --out_dir logs/embodiment_augment/s1_navmesh
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / 'gs_vlnpe')); sys.path.insert(0, str(_HERE))
from esdf_utils import resample_by_arclength  # noqa: E402
from viz_utils import GT_COLOR, KNOT_COLOR, floorplan_canvas, save_gallery  # noqa: E402
from pixel_goal_utils import rig_height_m  # noqa: E402
from vlnce_align import _rigid, build_rawdata_index, hmap_translation, hmap_yaw  # noqa: E402

WP_COLOR, BAD_COLOR = (0, 220, 255), (255, 40, 40)
START_COLOR, GOAL_COLOR = (255, 200, 0), (255, 0, 255)
BASELINE_R_B = 0.10        # VLN-CE 정의값 (cylinder diameter 0.2 m). 배포 navmesh 직독값과 같다
LEG_TOL_M = 0.5            # VLN-CE 논문의 navigability 기준
NAV_TOL = 0.99             # 게이트 B. 옛 게이트 E와 같은 값 — 저장 GT 1517점 중 1점이 실패한 실측이 있다
AREA_TOL = 0.01            # 게이트 D: 배포본 대비 area 상대오차 상한
MAX_Y_DELTA = 0.5          # `is_navigable`의 수직 허용치. habitat 기본값을 그대로 쓴다
DENSE_M = 0.05             # 경로를 채우는 간격[m] — 격자 해상도와 같게 해야 코너를 놓치지 않는다
MAX_Z_SPREAD_M = 0.30      # 이보다 높이가 흔들리면 계단 에피소드로 보고 제외 (#05와 같은 값)
EXPECT = {'agent_radius': BASELINE_R_B, 'agent_height': 1.50, 'agent_max_climb': 0.20, 'cell_size': 0.05}
SETTING_KEYS = ['agent_radius', 'agent_height', 'agent_max_climb', 'agent_max_slope', 'cell_size',
                'cell_height', 'edge_max_error', 'edge_max_len', 'region_min_size', 'region_merge_size',
                'verts_per_poly', 'detail_sample_dist', 'detail_sample_max_error',
                'filter_low_hanging_obstacles', 'filter_ledge_spans',
                'filter_walkable_low_height_spans', 'include_static_objects']


# ---------------------------------------------------------------------------
# 좌표 — 하류(`02` / `04` / `05`)가 import한다. 시그니처를 바꾸지 마라.
# ---------------------------------------------------------------------------
def sf2mesh(raw_ep):
    """에피소드 start-frame -> mesh 절대 4x4. **해석적** (mesh 로드 불필요)."""
    return _rigid([hmap_yaw(raw_ep['start_rotation']), *hmap_translation(raw_ep['start_position'])])


def mesh_to_habitat_floor(m_xyz, h_rig):
    """mesh Z-up 카메라점 -> habitat Y-up **바닥점**. `(m_x, m_z - h_rig, -m_y)`.

    `hmap_translation`(habitat->mesh)의 역이고, 카메라 높이를 빼서 navmesh가 판정하는 바닥으로 내린다.
    """
    m = np.asarray(m_xyz, dtype=np.float64).reshape(-1, 3)
    return np.stack([m[:, 0], m[:, 2] - float(h_rig), -m[:, 1]], axis=1)


def load_poses(data_root, scene, rig, episode):
    """parquet `pose.{rig}` 전체. -> (N,4,4). depth는 읽지 않는다."""
    import pyarrow.parquet as pq
    t = pq.read_table(Path(data_root) / scene / 'data' / 'chunk-000' / f'episode_{episode:06d}.parquet')
    return np.stack([np.asarray(p, dtype=np.float64).reshape(4, 4) for p in t[f'pose.{rig}'].to_pylist()])


def topdown(pf, mpp, height):
    """`get_topdown_view` -> (mask (Nz,Nx) bool, origin (2,) = (lo_x, lo_z)). 규약은 모듈 docstring."""
    lo, _hi = pf.get_bounds()
    lo = np.asarray(lo, dtype=np.float64)
    return np.asarray(pf.get_topdown_view(mpp, float(height)), dtype=bool), np.array([lo[0], lo[2]])


# ---------------------------------------------------------------------------
# navmesh 질의
# ---------------------------------------------------------------------------
def chain_find_path(pf, wps, per_leg=False):
    """waypoint를 순서대로 `find_path`로 잇는다. -> (path (M,3) | [array|None]*L, reach (L,))

    **VLN-CE가 데이터셋을 만든 절차 그대로**: *"We run this algorithm between each waypoint in a
    trajectory to the next … navigable if … within 0.5 m of the next waypoint"*.
    `per_leg=True`가 아니면 실패한 leg를 버리고 이어붙이므로 **떨어진 두 구간 사이에 벽을 통과하는
    직선이 생긴다** — 경로를 점 단위로 판정할 때는 반드시 `per_leg=True`로 받아라.
    """
    import habitat_sim
    w = np.asarray(wps, dtype=np.float64)
    legs, pieces, reach = [], [], []
    for a, b in zip(w[:-1], w[1:]):
        sp = habitat_sim.ShortestPath(); sp.requested_start = a; sp.requested_end = b
        if not pf.find_path(sp) or not np.isfinite(sp.geodesic_distance) or len(sp.points) < 2:
            legs.append(None); reach.append(float('inf')); continue
        p = np.asarray(sp.points, dtype=np.float64)
        legs.append(p); reach.append(float(np.linalg.norm(p[-1] - b)))
        pieces.append(p if not pieces else p[1:])
    if per_leg:
        return legs, np.array(reach)
    return (np.vstack(pieces) if pieces else np.zeros((0, 3))), np.array(reach)


def judge(pf, pts):
    """점들이 navmesh 폴리곤 안인가. -> dict(n, nav, mask, bad, dxz_max, dy_max, bad_dxz_max)

    `snap_point`는 navigable이 아니면 NaN을 낼 수 있으므로 유한값만 집계한다.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    nan = float('nan')
    if not len(p):
        return dict(n=0, nav=0, mask=np.zeros(0, bool), bad=p, dxz_max=nan, dy_max=nan, bad_dxz_max=nan)
    nav = np.array([bool(pf.is_navigable(q, MAX_Y_DELTA)) for q in p])
    sn = np.array([np.asarray(pf.snap_point(q), dtype=np.float64) for q in p])
    ok = np.isfinite(sn).all(axis=1)
    dxz = np.full(len(p), nan)
    dy = np.full(len(p), nan)
    dxz[ok] = np.linalg.norm((sn - p)[ok][:, [0, 2]], axis=1)
    dy[ok] = np.abs((sn - p)[ok][:, 1])
    return dict(n=int(len(p)), nav=int(nav.sum()), mask=nav, bad=p[~nav],
                dxz_max=float(np.nanmax(dxz)) if ok.any() else nan,
                dy_max=float(np.nanmax(dy)) if ok.any() else nan,
                bad_dxz_max=float(np.nanmax(dxz[~nav])) if (~nav).any() and ok[~nav].any() else 0.0)


def off_depth(mask, org, mpp, xz):
    """래스터 밖으로 나간 점 수 + **정수 셀 깊이** 히스토그램. -> dict(n, off, hist)

    깊이는 Chebyshev(체스판) 거리 — 1이면 격자 반올림 한 칸, 2 이상이면 지도가 실제로 다르다.
    """
    from scipy.ndimage import distance_transform_cdt
    depth = distance_transform_cdt(~mask, metric='chessboard')
    q = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
    jx = np.floor((q[:, 0] - org[0]) / mpp).astype(int)
    jz = np.floor((q[:, 1] - org[1]) / mpp).astype(int)
    keep = ((jx >= 0) & (jx < mask.shape[1]) & (jz >= 0) & (jz < mask.shape[0]))
    d = depth[jz[keep], jx[keep]]
    return dict(n=int(keep.sum()), off=int((d > 0).sum()),
                hist={1: int((d == 1).sum()), 2: int((d == 2).sum()), 3: int((d >= 3).sum())})


def dense3(p, step_m):
    """3D 폴리라인을 등간격으로 채운다 — 2D 리샘플은 y(층)를 잃는다."""
    q = np.asarray(p, dtype=np.float64)
    seg = [a + (b - a) * np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / step_m) + 1))[:, None]
           for a, b in zip(q[:-1], q[1:])]
    return np.vstack(seg) if seg else q


def _arclen(xz):
    q = np.asarray(xz, dtype=np.float64)
    return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()) if len(q) > 1 else 0.0


def match_paths(a_xz, b_xz, n=200):
    """두 폴리라인의 대응점 거리. -> dict(mean, max, len_ratio). 호길이 등간격 리샘플 후 비교.

    점 분포가 다른 경로를 인덱스로 비교하면 안 된다(GT는 시간축 샘플이라 회전 구간이 촘촘하다).
    """
    if len(a_xz) < 2 or len(b_xz) < 2:
        return None
    ra, rb = resample_by_arclength(a_xz, n=n), resample_by_arclength(b_xz, n=n)
    d = np.linalg.norm(ra - rb, axis=1)
    lb = _arclen(b_xz)
    return dict(mean=float(d.mean()), max=float(d.max()),
                len_ratio=float(_arclen(a_xz) / lb) if lb > 0 else float('nan'))


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------
def episode_figure(out, name, mask, org, mpp, gt_xz, fp_legs, wps_xz, bad_xz):
    """배포 navmesh 위에 ⓑ주황 → ⓐ노랑 순으로 덮고 waypoint·start·goal·위반점을 찍는다.

    주황을 먼저 깔고 노랑으로 덮으므로 **주황이 보이는 곳이 곧 두 경로가 어긋난 곳**이다.
    """
    base, draw, emit = floorplan_canvas(mask, np.array([org[0], org[1]]), mpp, out)
    img = base.copy()
    for lg in fp_legs:
        if lg is not None:
            draw(img, resample_by_arclength(lg[:, [0, 2]], step_m=mpp), KNOT_COLOR, 2)
    draw(img, resample_by_arclength(gt_xz, step_m=mpp), GT_COLOR, 1)
    draw(img, wps_xz, WP_COLOR, 5)
    draw(img, bad_xz, BAD_COLOR, 3)
    draw(img, gt_xz[:1], START_COLOR, 4)
    draw(img, gt_xz[-1:], GOAL_COLOR, 4)
    emit(img, name)
    return name


# ---------------------------------------------------------------------------
def _fmt(v):
    return f'{v:.4f}' if isinstance(v, float) else str(v)


def sweep_navmesh(args, scene, r_bs, cache, shipped_area, wp_all, floor_y, out):
    """r_b별 `recompute_navmesh` -> area/islands/waypoint navigable + `.navmesh` 캐시 + topdown 그림.

    `agent_radius`만 바꾼다 — 배포본이 habitat 기본값으로 만들어졌으므로 그것이 "동일 조건"이다.
    """
    import habitat_sim
    from habitat_sim.nav import NavMeshSettings
    bc = habitat_sim.SimulatorConfiguration()
    bc.scene_id = f'{args.glb_root}/{scene}/{scene}.glb'
    bc.enable_physics = False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(bc, [habitat_sim.AgentConfiguration()]))
    rows = []
    try:
        for rb in r_bs:
            s = NavMeshSettings(); s.set_defaults(); s.agent_radius = float(rb)
            ok = bool(sim.recompute_navmesh(sim.pathfinder, s))
            pfr = sim.pathfinder
            nav = int(sum(bool(pfr.is_navigable(q, MAX_Y_DELTA)) for q in wp_all))
            f = cache / f'{scene}_rb{rb:.2f}.navmesh'
            pfr.save_nav_mesh(str(f))
            mask, org = topdown(pfr, args.mpp, floor_y)
            base, draw, emit = floorplan_canvas(mask, np.array([org[0], org[1]]), args.mpp, out)
            img = base.copy(); draw(img, wp_all[:, [0, 2]], WP_COLOR, 3)
            name = f'{scene}_sweep_rb{rb:.2f}.jpg'; emit(img, name)
            rows.append(dict(r_b=rb, ok=ok, area=float(pfr.navigable_area), islands=int(pfr.num_islands),
                             nav=nav, n=int(len(wp_all)), fig=name, cache=f.name))
            print(f'[S1] D r_b={rb:.2f} ok={ok} area {pfr.navigable_area:8.2f} m2 '
                  f'islands {pfr.num_islands} · waypoint navigable {nav}/{len(wp_all)} -> {f.name}')
    finally:
        sim.close()
    return rows


def run_scene(args, scene, raw_idx, h_rig, cache, out):
    """씬 하나를 처리한다. -> dict(에피소드 행 + 씬 합계 + 게이트)."""
    from habitat_sim.nav import PathFinder
    pf = PathFinder(); pf.load_nav_mesh(f'{args.glb_root}/{scene}/{scene}.navmesh')
    assert pf.is_loaded, f'{scene}: navmesh 로드 실패'
    st = pf.nav_mesh_settings
    shipped_area = float(pf.navigable_area)
    setting_bad = {k: getattr(st, k) for k, v in EXPECT.items() if abs(getattr(st, k) - v) > 1e-6}
    print(f'\n[S1] === {scene}: area {shipped_area:.2f} m2 · islands {pf.num_islands} · '
          f'설정 불일치 {setting_bad or "없음"}')

    raws = [v for (sc, _instr), v in raw_idx.items() if sc == scene]
    wp_all = np.vstack([np.asarray(r['reference_path'], dtype=np.float64) for r in raws])
    jw = judge(pf, wp_all)
    print(f'[S1] waypoint navigable {jw["nav"]}/{jw["n"]} · snap |dxz| max {jw["dxz_max"]:.4f} m')

    tasks = [json.loads(l) for l in open(Path(args.data_root) / scene / 'meta' / 'episodes.jsonl')]
    eps, skip = [], {'raw': 0, 'stairs': 0}
    acc = dict(a_n=0, a_nav=0, b_n=0, b_nav=0, j2_n=0, j2_off=0, j2_hist={1: 0, 2: 0, 3: 0},
               fz_mismatch=0, reach_ok=0, reach_n=0, mm=[], a_bad_dxz=0.0)
    for ep in range(min(args.episodes, len(tasks))):
        instr = tasks[ep]['tasks'][0].strip()
        raw = raw_idx.get((scene, instr))
        if raw is None:
            skip['raw'] += 1; print(f'[S1] ep{ep}: skip (raw 매칭 실패)'); continue
        gt_mesh = np.stack([(sf2mesh(raw) @ p)[:3, 3] for p in
                            load_poses(args.data_root, scene, args.rig, ep)])
        gt = mesh_to_habitat_floor(gt_mesh, h_rig)
        spread = float(gt[:, 1].max() - gt[:, 1].min())
        if spread > args.max_z_spread:
            skip['stairs'] += 1
            print(f'[S1] ep{ep}: skip (높이 변동 {spread:.2f} > {args.max_z_spread} = 계단)'); continue

        wps = np.asarray(raw['reference_path'], dtype=np.float64)
        fz = float(np.median(wps[:, 1]))
        fp_legs, reach = chain_find_path(pf, wps, per_leg=True)

        # B: 두 경로를 5 cm로 채워 점 단위 판정. ⓑ는 leg별로 재야 끊긴 구간을 직선으로 잇지 않는다.
        gt_d = dense3(gt, DENSE_M)
        ja = judge(pf, gt_d)
        fp_d = [dense3(lg, DENSE_M) for lg in fp_legs if lg is not None]
        jb = judge(pf, np.vstack(fp_d) if fp_d else np.zeros((0, 3)))

        # 진단 1: ⓑ가 ⓐ를 얼마나 재현하나. leg가 하나라도 실패하면 비교가 무의미하다.
        m = None
        if all(lg is not None for lg in fp_legs):
            chain = np.vstack([lg if i == 0 else lg[1:] for i, lg in enumerate(fp_legs)])
            m = match_paths(chain[:, [0, 2]], gt[:, [0, 2]])
            if m is not None:
                acc['mm'].append(m)

        # 진단 2: J1(진실) vs J2(래스터). 술어는 **J1 통과 점**에만 걸어야 하므로 먼저 걸러낸다.
        mask, org = topdown(pf, args.mpp, fz)
        j2 = off_depth(mask, org, args.mpp, gt_d[ja['mask']][:, [0, 2]])
        gt_fz = gt_d.copy(); gt_fz[:, 1] = fz
        j1_fz = judge(pf, gt_fz)['nav']

        bad = np.vstack([ja['bad'], jb['bad']]) if len(ja['bad']) or len(jb['bad']) \
            else np.zeros((0, 3))
        fig = episode_figure(out, f'{scene}_ep{ep:03d}.jpg', mask, org, args.mpp,
                             gt[:, [0, 2]], fp_legs, wps[:, [0, 2]], bad[:, [0, 2]])

        acc['a_n'] += ja['n']; acc['a_nav'] += ja['nav']
        acc['a_bad_dxz'] = max(acc['a_bad_dxz'], ja['bad_dxz_max'])
        acc['b_n'] += jb['n']; acc['b_nav'] += jb['nav']
        acc['j2_n'] += j2['n']; acc['j2_off'] += j2['off']
        for k in acc['j2_hist']:
            acc['j2_hist'][k] += j2['hist'][k]
        acc['fz_mismatch'] += abs(j1_fz - ja['nav'])
        acc['reach_ok'] += int((reach <= LEG_TOL_M).sum()); acc['reach_n'] += len(reach)
        eps.append(dict(ep=ep, fz=fz, ja=ja, jb=jb, j2=j2, m=m, reach=reach, fig=fig))
        print(f'[S1] ep{ep}: ⓐ {ja["nav"]}/{ja["n"]} · ⓑ {jb["nav"]}/{jb["n"]} navigable · '
              f'J2 밖 {j2["off"]}/{j2["n"]} 깊이 {j2["hist"]} · '
              + (f'재현 mean {m["mean"] * 100:.1f} cm 길이비 {m["len_ratio"]:.3f}' if m else '재현 실패')
              + f' · leg reach max {reach.max():.3f} m')

    sweep = sweep_navmesh(args, scene, [float(x) for x in args.r_bs.split(',')], cache,
                          shipped_area, wp_all, float(np.median(wp_all[:, 1])), out)
    a_frac = acc['a_nav'] / max(acc['a_n'], 1)
    b_frac = acc['b_nav'] / max(acc['b_n'], 1)
    gate = dict(A=not setting_bad,
                B=min(a_frac, b_frac) >= NAV_TOL,
                C=acc['reach_ok'] == acc['reach_n'] and acc['reach_n'] > 0,
                D=all(abs(s['area'] - shipped_area) / shipped_area < AREA_TOL
                      for s in sweep if abs(s['r_b'] - BASELINE_R_B) < 1e-9))
    print(f'[S1] B ⓐ {100 * a_frac:.2f}% · ⓑ {100 * b_frac:.2f}% navigable  '
          f'C leg {acc["reach_ok"]}/{acc["reach_n"]}  J2 밖 {acc["j2_off"]}/{acc["j2_n"]} '
          f'깊이 {acc["j2_hist"]}')
    # 실험결과 1의 행 = (점 수, navigable, snap |Δxz| max, snap Δy max). 여기서 뽑아 리포트를 단순하게 둔다.
    mx = lambda k, f: max((e[k][f] for e in eps), default=float('nan'))  # noqa: E731
    path1 = dict(a=(acc['a_n'], acc['a_nav'], mx('ja', 'dxz_max'), mx('ja', 'dy_max')),
                 b=(acc['b_n'], acc['b_nav'], mx('jb', 'dxz_max'), mx('jb', 'dy_max')),
                 w=(jw['n'], jw['nav'], jw['dxz_max'], jw['dy_max']))
    return dict(scene=scene, st=st, area=shipped_area, islands=int(pf.num_islands),
                setting_bad=setting_bad, jw=jw, eps=eps, skip=skip, acc=acc, sweep=sweep,
                a_frac=a_frac, b_frac=b_frac, gate=gate, path1=path1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='17DRP5sb8fy,s8pcmisQ38h')
    ap.add_argument('--rig', default='125cm_0deg')
    ap.add_argument('--r_bs', default='0.10,0.15,0.20,0.30,0.45')
    ap.add_argument('--episodes', type=int, default=14, help='씬당 검증할 에피소드 수')
    ap.add_argument('--mpp', type=float, default=0.05, help='topdown 래스터 해상도[m] — 우리 격자와 같게')
    ap.add_argument('--max_z_spread', type=float, default=MAX_Z_SPREAD_M,
                    help='GT 높이 변동이 이보다 크면 계단으로 보고 제외 — **#05와 같은 값**')
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r')
    ap.add_argument('--glb_root', default='data/scene_data/mp3d_ce/mp3d')
    ap.add_argument('--navmesh_cache', default='data/embodiment_aug/navmesh',
                    help='r_b별 navmesh 저장 위치 — #04/#05가 이걸 읽는다(없으면 assert)')
    ap.add_argument('--out_dir', default='logs/embodiment_augment/s1_navmesh')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.navmesh_cache); cache.mkdir(parents=True, exist_ok=True)
    h_rig = rig_height_m(args.rig)
    raw_idx = build_rawdata_index(Path(args.raw_root))
    print(f'[S1] raw index {len(raw_idx)} 에피소드 · rig {args.rig} (h={h_rig:.4f} m) · '
          f'r_b={BASELINE_R_B} 고정')

    rows = [run_scene(args, s, raw_idx, h_rig, cache, out) for s in args.scenes.split(',') if s]
    summary, body = summary_html(rows, args), body_html(rows)
    (out / 'summary.html').write_text(summary, encoding='utf-8')
    # body.html을 쓰면 발행기가 **본문이 참조한 그림만** 인라인한다(갤러리 모드는 옛 jpg까지 긁어온다).
    (out / 'body.html').write_text(body, encoding='utf-8')
    save_gallery(out, 'report.html', '#03 — 데이터셋이 쓴 지도에 두 경로를 올린다', summary, body,
                 eyebrow='embodiment augmentation · #03')
    print(f'\n[S1] report -> {out}/report.html')
    print('[S1] 게이트: ' + ' · '.join(
        r['scene'] + ' ' + ''.join(f'{k}{"o" if r["gate"][k] else "X"}' for k in 'ABCD')
        for r in rows))
    return 0 if all(all(r['gate'].values()) for r in rows) else 1


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------
def summary_html(rows, args):
    scope = ' · '.join(f'<b>{r["scene"]}</b> {len(r["eps"])}개'
                       + (f'(계단 {r["skip"]["stairs"]} · raw {r["skip"]["raw"]} 제외)'
                          if any(r['skip'].values()) else '') for r in rows)
    h = [f'<p>씬·에피소드: {scope} · rig {args.rig} · <b>r_b = {BASELINE_R_B} 고정</b>(배포 navmesh 직독값) '
         f'· 배경 = 그 에피소드 floor 높이에서 자른 <b>배포 navmesh</b>({args.mpp} m/px). '
         f'#04가 우리 맵을 이 지도에 맞추고, #05가 그 위에서 r_b를 흔든다.</p>']

    h.append('<h3>게이트</h3><table border=1 cellpadding=6><tr><th>게이트</th><th>묻는 것</th>'
             + ''.join(f'<th>{r["scene"]}</th>' for r in rows) + '</tr>')
    G = [('A', '배포 navmesh가 habitat 기본값인가',
          lambda r: f'불일치 {r["setting_bad"] or "없음"}'),
         ('B', f'ⓐ·ⓑ가 지도를 벗어나나 (≥{100 * NAV_TOL:.0f}%)',
          lambda r: f'ⓐ {100 * r["a_frac"]:.2f}% · ⓑ {100 * r["b_frac"]:.2f}%'),
         ('C', f'leg find_path가 ≤{LEG_TOL_M} m 안에 닿나 (100%)',
          lambda r: f'{r["acc"]["reach_ok"]}/{r["acc"]["reach_n"]}'),
         ('D', f'recompute가 배포본을 재현하나 (<{100 * AREA_TOL:.0f}%)',
          lambda r: f'{r["area"]:.2f} vs '
                    + ', '.join(f'{s["area"]:.2f}' for s in r['sweep']
                                if abs(s['r_b'] - BASELINE_R_B) < 1e-9))]
    for k, q, fn in G:
        h.append(f'<tr><td><b>{k}</b></td><td>{q}</td>'
                 + ''.join(f'<td>{fn(r)} — <b>{"PASS" if r["gate"][k] else "FAIL"}</b></td>'
                           for r in rows) + '</tr>')
    h.append('</table>')
    h.append('<p>A가 통과하면 <b>"VLN-CE와 동일 조건"의 자유 파라미터가 0개</b>다 — '
             '<code>NavMeshSettings().set_defaults()</code>에서 <code>agent_radius</code>만 바꾸면 된다.</p>')

    h.append('<h3>실험결과 1 — 두 경로가 배포 navmesh 위에 있나</h3>'
             '<table border=1 cellpadding=6><tr><th>경로</th><th>씬</th><th>점 수</th>'
             '<th>navigable</th><th>snap |Δxz| max</th><th>snap Δy max</th></tr>')
    for label, why, key in (('<b>ⓐ 원본 정답</b>(노랑)', '학습이 먹는 라벨', 'a'),
                            ('<b>ⓑ waypoint 연결</b>(주황)', 'leg별 find_path', 'b'),
                            ('waypoint 자체(하늘)', 'reference_path 4~7점', 'w')):
        for r in rows:
            n, nav, dxz, dy = r['path1'][key]
            h.append(f'<tr><td>{label}<br><small>{why}</small></td><td>{r["scene"]}</td><td>{n}</td>'
                     f'<td><b>{nav}/{n}</b> ({100 * nav / max(n, 1):.2f}%)</td>'
                     f'<td>{dxz:.4f} m</td><td>{dy:.4f} m</td></tr>')
    h.append('</table>')
    h.append(f'<p>경로를 <b>{DENSE_M * 100:.0f} cm 간격으로 채워</b> 점마다 <code>is_navigable</code>을 '
             f'묻는다(꼭짓점만 재면 코너를 놓친다). ⓑ는 <b>leg별로</b> 재므로 실패한 leg를 버리고 '
             '이어붙일 때 생기는 <b>가짜 직선</b>이 없다. '
             f'<code>is_navigable</code>은 수평 1 cm · 수직 {MAX_Y_DELTA} m 맹점이 있다.</p>')

    h.append('<h3>실험결과 2 — ⓑ는 ⓐ를 얼마나 재현하나 (게이트 아님)</h3>'
             '<table border=1 cellpadding=6><tr><th>씬</th><th>에피소드</th>'
             '<th>대응점 거리 mean</th><th>max</th><th>길이비 (ⓑ/ⓐ)</th></tr>')
    for r in rows:
        mm = r['acc']['mm']
        h.append(f'<tr><td><b>{r["scene"]}</b></td><td>{len(mm)}</td>'
                 + (f'<td><b>{np.median([m["mean"] for m in mm]) * 100:.1f} cm</b></td>'
                    f'<td>{np.max([m["max"] for m in mm]) * 100:.1f} cm</td>'
                    f'<td><b>{np.mean([m["len_ratio"] for m in mm]):.3f}</b></td>'
                    if mm else '<td>—</td><td>—</td><td>—</td>') + '</tr>')
    h.append('</table>')
    h.append('<p>⚠️ <b>길이비 ≈ 1 을 "같은 경로"의 증거로 읽으면 안 된다.</b> 그림을 보면 두 종류의 '
             '어긋남이 섞여 있고, <b>길이비는 둘을 구분하지 못한다</b>:</p><ul>'
             '<li><b>코너 절단·궤적 여유</b> — 원본 GT는 <b>0.25 m 전진 / 15° 회전 이산 액션</b>으로 '
             '저장돼 넓게 돌고, <code>find_path</code>는 측지 최단선이라 자른다. 17DRP가 이 경우다'
             '(median 12.9 cm).</li>'
             '<li><b>장애물의 반대편으로 갈라짐</b> — 두 우회로의 길이가 비슷하면 측지 최단선이 GT와 '
             '<b>다른 쪽</b>으로 돈다. s8의 큰 값(median 56.4 cm)이 이 경우고, '
             '<b>길이비는 그래도 1.002</b>다. 그림에서 노랑과 주황이 어두운 섬을 위/아래로 갈라져 '
             '지나가는 에피소드를 보라.</li></ul>'
             '<p>→ <b>회랑 중심선은 <code>find_path</code>가 아니라 GT 서브궤적이어야 한다</b>'
             '(현 <code>follow_waypoints</code>가 이미 그렇게 한다 — #03이 그 선택의 근거다). '
             '반대편으로 갈라지는 경우가 있으니 이건 선택이 아니라 <b>필수</b>다. '
             '<b>일치시키는 로직 자체는 이 리포트 범위 밖이다</b>(#05).</p>')

    h.append('<h3>실험결과 3 — "뚫는다"가 어디서 일어나나 (게이트 아님)</h3>'
             '<table border=1 cellpadding=6><tr><th>씬</th><th>ⓐ 점 수</th>'
             '<th>J1 밖 (진짜 뚫음)</th><th>그 점의 snap 거리 max</th>'
             '<th>J1 통과 중 J2 밖</th><th>깊이 1칸</th><th>깊이 2칸</th><th>깊이 ≥3칸</th>'
             '<th>술어</th></tr>')
    for r in rows:
        a = r['acc']; hs = a['j2_hist']
        verdict = ('✅ 전부 1칸 = 격자 반올림' if hs[2] == 0 and hs[3] == 0
                   else '❌ <b>반증</b> — 지도가 실제로 다르다')
        h.append(f'<tr><td><b>{r["scene"]}</b></td><td>{a["a_n"]}</td>'
                 f'<td><b>{a["a_n"] - a["a_nav"]}</b> '
                 f'({100 * (a["a_n"] - a["a_nav"]) / max(a["a_n"], 1):.2f}%)</td>'
                 f'<td>{a["a_bad_dxz"]:.4f} m</td>'
                 f'<td><b>{a["j2_off"]}</b>/{a["j2_n"]}</td>'
                 f'<td>{hs[1]}</td><td>{hs[2]}</td><td>{hs[3]}</td><td>{verdict}</td></tr>')
    h.append('</table>')
    h.append('<p><b>J1</b> = habitat에게 직접 물은 답(진실) — 여기서 밖이면 <b>원본 정답이 배포 navmesh를 '
             '실제로 벗어난 것</b>이고, 그 깊이는 <code>snap_point</code>까지의 수평 거리로 재 놓았다. '
             f'<b>J2</b> = <code>get_topdown_view</code> {args.mpp * 100:.0f} cm 래스터 — '
             '<b>우리가 그리고 계획할 때 보는 것</b>(<code>navmesh_grid</code>가 이걸 샘플링한다). '
             '깊이 = 마스크 밖 점에서 가장 가까운 navigable 칸까지의 <b>Chebyshev 칸 수</b>.</p>'
             '<p><b>술어: J1은 통과인데 J2가 막는 점이 전부 깊이 1칸이면 원인은 격자 반올림이다.</b> '
             '2칸 이상이 하나라도 나오면 반증되고, 그때는 지도 자체가 다른 것이다. '
             'J1 밖인 점은 술어에서 제외했다 — 그건 래스터 탓이 아니다.</p>'
             f'<p><b>J1 밖인 점도 한 칸({args.mpp * 100:.0f} cm)보다 얕은지 확인하라</b> — '
             '"진짜 뚫음"의 snap 거리 max가 한 칸 미만이면 그것도 경계에 걸친 이산화 오차이고, '
             '한 칸을 넘으면 원본 정답이 지도를 실제로 관통한다는 뜻이다.</p>')
    fzm = sum(r['acc']['fz_mismatch'] for r in rows)
    h.append(f'<p>높이 항은 <b>{fzm}점</b>이다 — 점의 자기 높이로 잰 J1과 floor 높이 <code>fz</code>로 '
             f'치환해 잰 J1의 차이. <code>is_navigable</code>의 수직 허용치가 {MAX_Y_DELTA} m이고 '
             f'계단 에피소드를 {args.max_z_spread} m로 걸러내므로 0이어야 한다 — 0이 아니면 계단 '
             '에피소드가 필터를 새어 나온 것이다.</p>')

    h.append('<h3>실험결과 4 — 크기별 지도 스윕과 캐시</h3>'
             '<table border=1 cellpadding=6><tr><th>씬</th><th>r_b</th><th>navigable area</th>'
             '<th>배포본 대비</th><th>islands</th><th>waypoint navigable</th><th>캐시</th></tr>')
    for r in rows:
        for s in r['sweep']:
            h.append(f'<tr><td>{r["scene"] if s is r["sweep"][0] else ""}</td>'
                     f'<td><b>{s["r_b"]:.2f}</b></td><td>{s["area"]:.2f} m²</td>'
                     f'<td>{100 * (s["area"] / r["area"] - 1):+.2f}%</td><td>{s["islands"]}</td>'
                     f'<td>{s["nav"]}/{s["n"]} ({100 * s["nav"] / max(s["n"], 1):.1f}%)</td>'
                     f'<td><code>{s["cache"]}</code></td></tr>')
    h.append('</table>')
    h.append(f'<p><code>agent_radius</code>만 바꿔 <code>recompute_navmesh</code>한 결과를 '
             f'<code>{args.navmesh_cache}/&lt;scene&gt;_rb&lt;r&gt;.navmesh</code>에 캐시한다 — '
             '#04의 default 맵이자 #05의 habitat 오라클이 이걸 읽는다(없으면 assert).</p>')

    h.append('<h3>범례</h3>'
             f'<p><span style="color:rgb{GT_COLOR}">■</span> <b>ⓐ 원본 정답</b> · '
             f'<span style="color:rgb{KNOT_COLOR}">■</span> <b>ⓑ waypoint 연결</b> · '
             f'<span style="color:rgb{WP_COLOR}">■</span> waypoint · '
             f'<span style="color:rgb{BAD_COLOR}">■</span> <b>벽 뚫은 점</b> · '
             f'<span style="color:rgb{START_COLOR}">■</span> start · '
             f'<span style="color:rgb{GOAL_COLOR}">■</span> goal. '
             '초록 배경 = 갈 수 있는 곳. <b>ⓑ를 먼저 깔고 ⓐ로 덮으므로 주황이 보이는 곳이 곧 두 경로가 '
             '어긋난 곳</b>이다 — 완전히 겹치면 주황은 하나도 안 보인다.</p>')

    h.append('<h3>한계</h3><ul>'
             f'<li>씬 2채 · 씬당 최대 {args.episodes} 에피소드. <b>계단 에피소드는 제외</b>했다'
             f'({" · ".join(f"{r['scene']} {r['skip']['stairs']}개" for r in rows)}) — '
             '<code>get_topdown_view</code>가 단일 높이 절단이라 다층 경로를 그릴 수 없다.</li>'
             f'<li><code>is_navigable</code>은 <b>수평 1 cm · 수직 {MAX_Y_DELTA} m 맹점</b>이 있다. '
             '게이트 B는 "지도를 벗어나지 않는다"가 아니라 "평면상 navmesh 폴리곤 안, ±1 cm"다.</li>'
             '<li><b>우리 플래너(<code>follow_waypoints</code>)와 우리 occ 맵은 이 리포트 범위 밖</b>'
             '이다(#04 / #05). #03은 <b>데이터셋이 쓴 지도</b>만 다룬다.</li>'
             '<li><b><code>clearance</code> 분석을 지웠다.</b> navmesh는 이미 <code>agent_radius</code> '
             f'{BASELINE_R_B}만큼 깎인 configuration space라 "칸 안이냐"가 올바른 질문이고, 거기에 '
             '<code>clearance ≥ r_b</code>를 또 요구하면 <b>반경을 두 번 센다</b>'
             '(<code>navmesh_grid.esdf_from_mask</code>의 <code>+ r_b</code> 트릭이 그 증거).</li>'
             '<li><b>이전 판의 결론을 철회했다.</b> "원본 정답이 navmesh 경계를 여유 0으로 스친다"는 '
             '<code>clearance min</code>(최악의 한 점)을 전형값으로 읽은 오류였다.</li>'
             '<li>게이트 D는 자기가 <b>쓰는</b> r_b=0.10 지도만 검증한다. r_b&gt;0.10 지도가 옳은지는 '
             '여기서 알 수 없다 — #05의 W6(habitat 오라클 교차검증)이 그 일을 한다.</li>'
             '</ul>')
    return ''.join(h)


def body_html(rows):
    h = []
    for r in rows:
        h.append(f'<h2>{r["scene"]}</h2>')
        h.append('<h3>배포 navmesh 설정 직독 (게이트 A)</h3>'
                 '<table border=1 cellpadding=6><tr><th>키</th><th>값</th></tr>'
                 + ''.join(f'<tr><td><code>{k}</code></td><td>{_fmt(getattr(r["st"], k))}</td></tr>'
                           for k in SETTING_KEYS) + '</table>')
        h.append('<h3>에피소드별</h3><table border=1 cellpadding=6><tr><th>ep</th><th>floor y</th>'
                 '<th>ⓐ navigable</th><th>ⓑ navigable</th><th>J2 밖 / 깊이</th>'
                 '<th>재현 mean/max</th><th>길이비</th><th>leg reach max</th></tr>')
        for e in r['eps']:
            m = e['m']
            h.append(f'<tr><td>{e["ep"]}</td><td>{e["fz"]:.3f}</td>'
                     f'<td>{e["ja"]["nav"]}/{e["ja"]["n"]}</td>'
                     f'<td>{e["jb"]["nav"]}/{e["jb"]["n"]}</td>'
                     f'<td>{e["j2"]["off"]}/{e["j2"]["n"]} · {e["j2"]["hist"]}</td>'
                     + (f'<td>{m["mean"] * 100:.1f} / {m["max"] * 100:.1f} cm</td>'
                        f'<td>{m["len_ratio"]:.3f}</td>' if m else '<td>—</td><td>—</td>')
                     + f'<td>{e["reach"].max():.3f} m</td></tr>')
        h.append('</table>')
        h.append('<h3>에피소드 그림</h3>'
                 '<p>노랑 <b>ⓐ 원본 정답</b> | 주황 <b>ⓑ waypoint 연결</b> | 하늘 waypoint | '
                 '<b>빨강 벽 뚫은 점</b> | 주황큰점 start | 자홍 goal — '
                 '주황을 먼저 깔고 노랑으로 덮으므로 <b>주황이 보이는 곳이 어긋난 곳</b></p>')
        for e in r['eps']:
            m = e['m']
            h.append(f'<figure><img src="{e["fig"]}" style="width:70%"><figcaption>'
                     f'ep{e["ep"]} · ⓐ {e["ja"]["nav"]}/{e["ja"]["n"]} · '
                     f'ⓑ {e["jb"]["nav"]}/{e["jb"]["n"]} navigable · J2 밖 {e["j2"]["off"]} '
                     f'깊이 {e["j2"]["hist"]}'
                     + (f' · 재현 mean {m["mean"] * 100:.1f} cm' if m else ' · 재현 실패')
                     + '</figcaption></figure>')
        h.append('<h3>크기별 지도 (r_b 스윕)</h3><div>' + ''.join(
            f'<figure style="display:inline-block;width:31%;margin:4px"><img src="{s["fig"]}" '
            f'style="width:100%"><figcaption>r_b={s["r_b"]:.2f} · {s["area"]:.1f} m²</figcaption>'
            '</figure>' for s in r['sweep']) + '</div>')
    return ''.join(h)


if __name__ == '__main__':
    sys.exit(main())
