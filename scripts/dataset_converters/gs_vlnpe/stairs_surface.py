"""층 이동 `--mf_stairs plan` 모드 — 계단 길까지 **새로 계획**한다 (GT 궤적 복사 없음).

`gt` 모드(`multifloor_utils.plan_multifloor_episode`)는 계단을 GT 점 그대로 쓴다. 이 모드는 GT 에서
**계단의 위치만**(계단 아래·위 GT 점, 계단 구간 GT 점의 범위 상자) 가져오고 길은 전부 새로 만든다.

    출발 층 지도에서 A*(03 plan_episode):  start  -> E (계단 앞, 새로 고른 점)
    높이 지도(2.5D)에서 A*:                E      -> X (계단 위, 새로 고른 점)
    도착 층 지도에서 A*:                   X      -> goal

높이 지도 (계단 주변 상자 안에서만)
-----------------------------------
`recast_like.py`(3dloader_vlnce)의 "칸마다 디딜 면 + 머리 공간 + 단차" 논리를 두 층에 걸치게 바꾼 것.
- 디딜 면: 점유 voxel k 이고 k+1 이 비어 있음(면의 윗면). 한 칸(x,y)에 여러 개 가능(층이 겹치는 곳).
- 머리 공간: 그 면 기준 `(h_nav, h_b]` 밴드에 점유가 없어야 함 — 03 의 층 지도와 같은 밴드(같은 h_nav·h_b).
- 연결: 이웃 칸(8방향 1칸 + 2칸)의 면과 높이 차 ≤ `step_m`(0.25 m; 계단 디딤 높이 ~0.17). 2칸 연결은 스캔 구멍 한 칸을 건넌다.
- 여유(clearance): 어느 방향으로든 1~2칸 안에 이어진 면이 없으면 "경계"(벽·난간·낭떠러지). 경계까지의
  그래프 거리 = 여유. A* 는 여유 ≥ `plan_r_b` 인 면만 쓰고, 판정은 `r_b` 로 한다(03 과 같은 짝).
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from esdf_utils import smooth_bezier, thin_waypoints, check_path_navigable, sample_esdf_at
import multifloor_utils as mfu

STEP_M = 0.25          # 이웃 면 높이 차 상한
BOX_MARGIN_M = 1.5     # 계단 GT 점 범위 상자 여유
FLOOR_Z_TOL_M = 0.15   # 계단 끝 노드가 층 바닥과 같은 높이로 볼 오차
VOID_DROP_M = 0.4      # 층 지도에서 막을 "구멍" = 바닥보다 이만큼(디딤 두 칸) 넘게 낮은 면만 있는 칸.
                       # 첫 디딤판(~0.2 m 아래)까지 막으면 계단 위 끝에 못 다가간다(실측: 한 씬 12/12 입구 막힘).
VOID_MIN_CELLS = 40    # 구멍 덩어리 최소 크기(0.1 m²). 바닥 스캔 구멍으로 생기는 점 같은 가짜 구멍은 버린다
                       # (실측: 점들이 plan_r_b 로 부풀려져 통로를 끊음 — 한 씬 A* 실패 0 -> 33).
_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
_LOOKUP = [(0, 0)] + _DIRS     # 판정용 칸 찾기: 자기 칸 먼저, 없으면 바로 옆 칸


def build_surface_graph(occ, origin, cell, xy_min, xy_max, z_lo, z_hi, h_nav, h_b, step_m=STEP_M):
    """상자 안 디딜 면 그래프. 반환 dict(nodes xyz, ix/iy/k, idx 볼륨, csr 그래프, clearance)."""
    nx_, ny_, nz_ = occ.shape
    ix0, iy0 = np.clip(np.floor((xy_min - origin[:2]) / cell).astype(int), 0, [nx_ - 1, ny_ - 1])
    ix1, iy1 = np.clip(np.ceil((xy_max - origin[:2]) / cell).astype(int) + 1, 1, [nx_, ny_])
    k0 = int(np.clip(np.floor((z_lo - origin[2]) / cell), 0, nz_ - 2))
    k1 = int(np.clip(np.ceil((z_hi - origin[2]) / cell), k0, nz_ - 2))
    sub = occ[ix0:ix1, iy0:iy1, :]
    gap, top = int(round(h_nav / cell)), int(round(h_b / cell))
    C = np.zeros(sub.shape[:2] + (nz_ + 1,), dtype=np.int32)
    C[:, :, 1:] = np.cumsum(sub, axis=2, dtype=np.int32)
    ks = np.arange(k0, k1 + 1)
    surf = sub[:, :, ks] & ~sub[:, :, ks + 1]
    a = np.clip(ks + gap + 1, 0, nz_); b = np.clip(ks + top + 1, 0, nz_)
    head_free = (C[:, :, b] - C[:, :, a]) == 0
    walk = surf & head_free                                   # (nx, ny, nk)
    ii, jj, kk = np.nonzero(walk)
    n = len(ii)
    idx = np.full(walk.shape, -1, dtype=np.int64)
    idx[ii, jj, kk] = np.arange(n)
    sk = int(round(step_m / cell))
    rows, cols, wts = [], [], []
    has = {}                                                  # 방향별 1~2칸 안에 이어진 이웃 존재
    shape = np.array(walk.shape)
    for dx, dy in _DIRS:
        ok_dir = np.zeros(n, dtype=bool)
        inside_dir = np.zeros(n, dtype=bool)
        for r in (1, 2):
            tx, ty = ii + dx * r, jj + dy * r
            inside = (tx >= 0) & (tx < shape[0]) & (ty >= 0) & (ty < shape[1])
            inside_dir |= inside if r == 1 else False
            best = np.full(n, -1, dtype=np.int64); bestdz = np.full(n, 99)
            for dz in range(-sk, sk + 1):
                tz = kk + dz
                m = inside & (tz >= 0) & (tz < shape[2])
                t = np.full(n, -1, dtype=np.int64)
                t[m] = idx[tx[m], ty[m], tz[m]]
                better = (t >= 0) & (abs(dz) < bestdz)
                best[better] = t[better]; bestdz[better] = abs(dz)
            e = best >= 0
            ok_dir |= e
            src = np.nonzero(e)[0]
            rows.append(src); cols.append(best[e])
            wts.append(cell * np.sqrt((dx * r) ** 2 + (dy * r) ** 2 + bestdz[e] ** 2))
        has[(dx, dy)] = ok_dir | ~inside_dir                  # 상자 밖 방향은 경계로 치지 않는다
    rows, cols, wts = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    G = csr_matrix((wts, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)
    boundary = ~np.logical_and.reduce([has[d] for d in _DIRS]) if n else np.zeros(0, bool)
    if boundary.any():
        clr = dijkstra(G, directed=False, indices=np.nonzero(boundary)[0], min_only=True)
    else:
        clr = np.full(n, np.inf)
    xyz = np.stack([(ii + ix0 + 0.5) * cell + origin[0], (jj + iy0 + 0.5) * cell + origin[1],
                    (kk + k0 + 1) * cell + origin[2]], axis=1)   # 면의 윗면 z
    return {'xyz': xyz, 'G': G, 'clr': clr, 'idx': idx, 'off': (ix0, iy0, k0), 'n': n, 'sk': sk}


def node_clearance_at(g, xyz, cell, origin, dz_max_m: float = None) -> np.ndarray:
    """궤적 점(xyz)마다: 그 칸에서 z 가 가장 가까운(±step) 면의 여유. 그 칸이 스캔 구멍이면 바로 옆 칸
    (그래프가 건너는 한 칸 구멍과 같은 기준)의 면을 쓴다. 그래도 없으면 0(관통 취급).
    `dz_max_m`: 면을 찾을 높이 창(기본 = 이웃 단차 상한). 이음매 판정은 z 가 층 바닥으로 고정돼 있어 넓게 준다."""
    sk = g['sk'] if dz_max_m is None else int(round(dz_max_m / cell))
    ix0, iy0, k0 = g['off']
    ijk = np.floor((xyz - origin) / cell).astype(int)
    ijk[:, 0] -= ix0; ijk[:, 1] -= iy0
    kz = np.rint((xyz[:, 2] - origin[2]) / cell).astype(int) - 1 - k0
    out = np.zeros(len(xyz))
    shp = g['idx'].shape
    for p in range(len(xyz)):
        best = None
        for di, dj in _LOOKUP:
            i, j = ijk[p, 0] + di, ijk[p, 1] + dj
            if not (0 <= i < shp[0] and 0 <= j < shp[1]):
                continue
            for dz in sorted(range(-sk, sk + 1), key=abs):
                k = kz[p] + dz
                if 0 <= k < shp[2] and g['idx'][i, j, k] >= 0:
                    best = g['idx'][i, j, k]; break
            if best is not None:
                break
        out[p] = g['clr'][best] if best is not None else 0.0
    return out


def plan_stairs(g, foot_xy, top_xy, z_lo, z_hi, plan_r_b, snap_max_m, nav_ok_lo, nav_ok_hi):
    """E(출발 층 바닥 높이 면) -> X(도착 층 바닥 높이 면) 최단 경로.

    후보 = 층 높이(±FLOOR_Z_TOL_M) · 여유 ≥ plan_r_b · 그 층 2D 지도(nav_coarse)에서도 통행 가능(평지 A* 가 닿아야 하므로).
    1차: GT 계단 끝에서 snap_max 이내 후보. 길이 없으면 2차: 상자(GT 계단 범위 ±1.5 m) 안 후보 전부.
    (계단 바로 앞 바닥은 층 지도에서 첫 디딤판 때문에 막혀 있어, 가까운 후보만 쓰면 끊긴 쪽만 남는다 — 표본 80 중 47.)
    도착은 그래프 거리 + GT 계단 끝까지 거리 최소인 후보. 반환 (status, node_ids)."""
    xyz, clr = g['xyz'], g['clr']
    keep = clr >= plan_r_b
    kid = np.nonzero(keep)[0]
    remap = np.full(g['n'], -1); remap[kid] = np.arange(len(kid))
    H = g['G'][kid][:, kid]
    lvl_lo = keep & (np.abs(xyz[:, 2] - z_lo) <= FLOOR_Z_TOL_M)
    lvl_hi = keep & (np.abs(xyz[:, 2] - z_hi) <= FLOOR_Z_TOL_M)
    ok_lo = np.array([nav_ok_lo(q) for q in xyz[lvl_lo, :2]], dtype=bool)
    ok_hi = np.array([nav_ok_hi(q) for q in xyz[lvl_hi, :2]], dtype=bool)
    all_src, all_dst = np.nonzero(lvl_lo)[0][ok_lo], np.nonzero(lvl_hi)[0][ok_hi]
    if len(all_src) == 0 or len(all_dst) == 0:
        return 'mf_entry_blocked', None
    status = 'mf_entry_blocked'
    for lim in (snap_max_m, np.inf):
        src = all_src[np.linalg.norm(xyz[all_src, :2] - foot_xy, axis=1) <= lim]
        dst = all_dst[np.linalg.norm(xyz[all_dst, :2] - top_xy, axis=1) <= lim]
        if len(src) == 0 or len(dst) == 0:
            continue
        dist, pred, _s = dijkstra(H, directed=False, indices=remap[src], min_only=True, return_predecessors=True)
        dd = dist[remap[dst]]
        if not np.isfinite(dd).any():
            status = 'mf_stairs_no_path'
            continue
        score = np.where(np.isfinite(dd), dd + np.linalg.norm(xyz[dst, :2] - top_xy, axis=1), np.inf)
        t = remap[dst[int(np.argmin(score))]]
        path = [t]
        while pred[path[-1]] >= 0:
            path.append(pred[path[-1]])
        return 'ok', kid[np.array(path[::-1])]
    return status, None


def smooth_stairs(node_xyz, spacing_m, step_m):
    """계단 노드 경로(칸 중심, 계단 모양) -> 솎기 + Bézier(xy). z 는 호길이 비율로 노드 z 를 보간."""
    xy = node_xyz[:, :2]
    if len(xy) < 3:
        return xy.copy(), node_xyz[:, 2].copy()
    sm = smooth_bezier(thin_waypoints(xy, spacing_m), step_m)
    s0 = mfu._arclen(xy); s1 = mfu._arclen(sm)
    t = s1 / max(s1[-1], 1e-9) * s0[-1]
    return sm, np.interp(t, s0, node_xyz[:, 2])


def plan_multifloor_episode_generated(occupancy, origin, scene_floor_z, gt_xyz, h_b, ep_floor, args, plan_episode) -> dict:
    """`--mf_stairs plan`. GT 에서는 층 분할(어디가 계단인가)과 계단 위치만 쓴다. 반환 형식은 gt 모드와 같다."""
    xy = np.asarray(gt_xyz[:, :2], dtype=np.float64)
    fz = mfu.point_floor_z(gt_xyz, ep_floor)
    floors = mfu.segment_levels(fz, xy, args.mf_level_tol_m, args.mf_plateau_min_m)
    cell = float(args.cell_m)
    coarse_cell = cell * args.downsample_factor
    h_nav = args.h_nav_ratio * h_b if args.h_nav_ratio else args.h_nav
    info = {'n_stairs': len(floors) - 1, 'levels': [round(f['level'], 3) for f in floors],
            'stair_dz_m': [round(floors[k + 1]['level'] - floors[k]['level'], 3) for k in range(len(floors) - 1)]}
    for f in floors:
        f['plan_z'] = f['level'] if abs(f['level'] - scene_floor_z) > args.floor_tol_m else scene_floor_z

    # 0) 계단마다 높이 지도 (계단 GT 점 범위 상자 안)
    graphs = []
    for k in range(len(floors) - 1):
        lo, hi = floors[k], floors[k + 1]
        span = xy[lo['i1']:hi['i0'] + 1]                     # GT 계단 위치(범위)만 쓴다
        zmin, zmax = min(lo['plan_z'], hi['plan_z']), max(lo['plan_z'], hi['plan_z'])
        graphs.append(build_surface_graph(occupancy, origin, cell, span.min(0) - BOX_MARGIN_M, span.max(0) + BOX_MARGIN_M,
                                          zmin - 0.3, zmax + 0.3, h_nav, h_b))

    # 0b) 층 지도에 "계단 구멍" 막기 — 층 바닥 높이엔 면이 없고 그보다 아래에만 면이 있는 칸(내려가는 계단·참 위 허공).
    #     층 지도(밴드)는 발밑 바닥을 안 봐서 이 칸을 통행 가능으로 본다(실측: plan 관통 130 중 82). 03 plan_episode 는
    #     그대로 두고, 그 층에 넘기는 occupancy 사본의 밴드 안에 점유를 찍어 벽으로 만든다.
    for k, f in enumerate(floors):
        f['occ'] = occupancy
        void = []
        for gk in (k - 1, k):
            if 0 <= gk < len(graphs) and graphs[gk]['n']:
                void.append(_void_columns(graphs[gk], f['plan_z'], origin, cell))
        void = np.unique(np.concatenate(void), axis=0) if void and sum(len(v) for v in void) else np.zeros((0, 2), int)
        if len(void):
            occ_k = occupancy.copy()
            kb = int(np.floor((f['plan_z'] + 0.5 * (h_nav + h_b) - origin[2]) / cell))
            if 0 <= kb < occ_k.shape[2]:
                occ_k[void[:, 0], void[:, 1], kb] = True
                f['occ'] = occ_k
        f['n_void'] = int(len(void))
        f['esdf'], f['nav'] = mfu.floor_nav_coarse(f['occ'], origin, f['plan_z'], h_b, args)
        f['nav_ok'] = (lambda p, f=f: mfu._nav_ok(f['nav'], origin, coarse_cell, p))
    info['n_void_cells'] = [f['n_void'] for f in floors]

    # 1) 계단마다 높이 지도 A*: E_k -> X_k. plan_r_b 로 안 되면 r_b 로 한 번 더(판정은 어차피 r_b — 좁은 계단, 표본 80 중 40).
    stairs = []
    for k in range(len(floors) - 1):
        lo, hi = floors[k], floors[k + 1]
        foot, top = xy[lo['i1']], xy[hi['i0']]
        g = graphs[k]
        if g['n'] == 0:
            return {'status': 'mf_stairs_no_path', 'where': f'stairs{k}', **info}
        for rr in dict.fromkeys((args.plan_r_b, args.r_b)):
            st, ids = plan_stairs(g, foot, top, lo['plan_z'], hi['plan_z'], rr, args.mf_snap_max_m,
                                  lo['nav_ok'], hi['nav_ok'])
            if st == 'ok':
                break
        if st != 'ok':
            return {'status': st, 'where': f'stairs{k}', **info}
        # 펴기(Bézier)가 U자 모서리를 깎아 벽에 붙으면 덜 펴서 다시 — 간격 spacing -> 0.2 -> 0(칸 중심 그대로)
        best = None
        for sp in (args.mf_stairs_spacing_m, 0.2, 0.0):
            sxy, sz = smooth_stairs(g['xyz'][ids], sp, args.smooth_step)
            cmin = float(node_clearance_at(g, np.column_stack([sxy, sz]), cell, origin).min())
            if best is None or cmin > best[0]:
                best = (cmin, sxy, sz, sp)
            if cmin >= args.r_b:
                break
        _, sxy, sz, used_sp = best
        # 층 높이인 점은 그 층 계획 바닥으로(이음매 z 튐 방지)
        sz = np.where(np.abs(sz - lo['plan_z']) <= FLOOR_Z_TOL_M, lo['plan_z'],
                      np.where(np.abs(sz - hi['plan_z']) <= FLOOR_Z_TOL_M, hi['plan_z'], sz))
        stairs.append({'k': k, 'xy': sxy, 'z': sz, 'g': g, 'E': g['xyz'][ids[0], :2], 'X': g['xyz'][ids[-1], :2],
                       'n_nodes': len(ids), 'spacing': used_sp})

    # 2) 층마다 A*: (GT start | X_{k-1}) -> (E_k | GT goal)
    parts = []
    for k, f in enumerate(floors):
        a = xy[0] if k == 0 else stairs[k - 1]['X']
        b = xy[-1] if k == len(floors) - 1 else stairs[k]['E']
        if np.linalg.norm(b - a) < args.mf_min_plan_m:
            parts.append({'kind': 'floor', 'k': k, 'xy': np.stack([a, b]), 'planned': None})
            continue
        planned = plan_episode(f['occ'], origin, f['plan_z'], h_b, a, b, args)
        if planned['status'] == 'start_or_goal_not_navigable':
            return {'status': 'mf_sg_blocked', 'where': f'floor{k}', **info}
        if planned['status'] == 'astar_failed':
            return {'status': 'mf_astar_failed', 'where': f'floor{k}', **info}
        assert planned['status'] == 'ok', f"unreachable status={planned['status']!r}"
        parts.append({'kind': 'floor', 'k': k, 'xy': np.asarray(planned['trajectory'], dtype=np.float64),
                      'planned': planned})

    pieces = []
    for k in range(len(floors)):
        pieces.append(parts[k])
        if k < len(stairs):
            s = stairs[k]
            pieces.append({'kind': 'stairs', 'k': k, 'xy': s['xy'], 'z': s['z']})

    # 3)~4) 이음매 + 판정. 이음매 깎기가 벽 쪽으로 들어가 여유를 잃는 경우가 많아(실측: 깎기 끄면 사용 가능 770 -> 993)
    #    폭을 mf_seam_m -> 절반 -> 0 순서로 시도해 판정(여유 ≥ r_b)을 통과하는 가장 부드러운 폭을 쓴다.
    def _assemble(seam_w):
        pieces = [dict(q) for q in pieces0]
        # 3) 이음매 — 평지 끝/계단 처음 (양쪽 다 새 경로). 계단 z 는 바뀐 xy 에 호길이 비율로 다시 맞춘다.
        for p in range(len(pieces) - 1):
            L, R = pieces[p], pieces[p + 1]
            if L['kind'] == 'floor' and R['kind'] == 'stairs':
                old = R['xy']; L['xy'], R['xy'] = mfu.seam_end(L['xy'], R['xy'], seam_w, args.smooth_step)
                R['z'] = _remap_z(old, R['z'], R['xy'])
            elif L['kind'] == 'stairs' and R['kind'] == 'floor':
                old = L['xy']; L['xy'], R['xy'] = mfu.seam_start(L['xy'], R['xy'], seam_w, args.smooth_step)
                L['z'] = _remap_z(old, L['z'], L['xy'])

        traj, tz, segments, n0 = [], [], [], 0
        for p in pieces:
            pxy = p['xy']; pz = p['z'] if p['kind'] == 'stairs' else np.full(len(pxy), floors[p['k']]['plan_z'])
            if traj and np.allclose(traj[-1][-1], pxy[0], atol=1e-9):
                pxy, pz = pxy[1:], pz[1:]
            if len(pxy) == 0:
                continue
            if p['kind'] == 'stairs':
                seg = {'kind': 'stairs', 'floor_z': [floors[p['k']]['plan_z'], floors[p['k'] + 1]['plan_z']], 'source': 'plan'}
            else:
                seg = {'kind': 'floor', 'floor_z': floors[p['k']]['plan_z'], 'planned': True}
            seg.update({'i0': n0, 'i1': n0 + len(pxy) - 1})
            n0 += len(pxy)
            traj.append(pxy); tz.append(pz); segments.append(seg)

        # 4) 판정 — 평지 = 그 층 ESDF, 계단 = 높이 지도 여유. 둘 다 새 경로라 둘 다 잰다.
        #    단, 평지 중 계단 끝(E/X)에서 mf_seam_m 안의 점은 높이 지도로 잰다 — 평지 지도는 첫 디딤판을 벽으로 보기
        #    때문이다(실측: 평지 r_b 미달 147 중 135 가 계단 끝 0.3 m 안).
        checks, clrs, refined = [], [], []
        for pi, p in enumerate(pieces):
            if p['kind'] == 'floor':
                esdf = floors[p['k']]['esdf']
                c = sample_esdf_at(esdf, p['xy'], origin, cell).astype(np.float64)
                near_any = np.zeros(len(c), dtype=bool)
                for nb in (pi - 1, pi + 1):
                    if 0 <= nb < len(pieces) and pieces[nb]['kind'] == 'stairs':
                        s = stairs[pieces[nb]['k']]
                        end = pieces[nb]['xy'][-1] if nb < pi else pieces[nb]['xy'][0]
                        near = np.linalg.norm(p['xy'] - end, axis=1) <= args.mf_seam_m + 1e-9
                        if near.any():
                            zz = np.full(near.sum(), floors[p['k']]['plan_z'])
                            c[near] = node_clearance_at(s['g'], np.column_stack([p['xy'][near], zz]), cell, origin,
                                                        dz_max_m=0.5)
                        near_any |= near
                chk = check_path_navigable(p['xy'][~near_any], esdf, origin, cell, args.r_b) if (~near_any).sum() >= 2 else None
                checks.append({'points_min_m': float(c.min()), 'points_median_m': float(np.median(c)),
                               'segments_min_m': float(min(c.min(), chk['segments_min_m'])) if chk else float(c.min()),
                               'length_m': float(np.linalg.norm(np.diff(p['xy'], axis=0), axis=1).sum()), 'r_b': args.r_b,
                               'points_ok': bool(c.min() >= args.r_b), 'segments_ok': bool(c.min() >= args.r_b),
                               'n_samples': int(chk['n_samples']) if chk else len(c),
                               'hard_n': int((c <= 1e-9).sum()), 'hard_ok': bool((c > 1e-9).all()),
                               'n_judged_by_heightmap': int(near_any.sum())})
                clrs.append(c)
                if p['planned'] is not None:
                    refined.append(p['planned']['check_refined'])
            else:
                g = stairs[p['k']]['g']
                c = node_clearance_at(g, np.column_stack([p['xy'], p['z']]), cell, origin)
                L = float(np.linalg.norm(np.diff(p['xy'], axis=0), axis=1).sum())
                checks.append({'points_min_m': float(c.min()), 'points_median_m': float(np.median(c)),
                               'segments_min_m': float(c.min()), 'length_m': L, 'r_b': args.r_b,
                               'points_ok': bool(c.min() >= args.r_b), 'segments_ok': bool(c.min() >= args.r_b),
                               'n_samples': int(len(c)), 'hard_n': int((c <= 1e-9).sum()), 'hard_ok': bool((c > 1e-9).all())})
                clrs.append(c)
        return pieces, traj, tz, segments, checks, clrs, refined

    pieces0 = pieces
    best = None
    for seam_w in dict.fromkeys((args.mf_seam_m, 0.5 * args.mf_seam_m, 0.0)):
        res = _assemble(seam_w)
        cmin = min(c['points_min_m'] for c in res[4])
        if best is None or cmin > best[0]:
            best = (cmin, seam_w, res)
        if cmin >= args.r_b:
            break
    _, seam_used, (pieces, traj, tz, segments, checks, clrs, refined) = best
    info['seam_used_m'] = seam_used
    plan_parts = [p for p in pieces if p['kind'] == 'floor' and p['planned'] is not None]
    agg = mfu._aggregate_checks(checks, clrs, args.r_b)
    agg['scope'] = 'floor_and_stairs'
    empty = np.zeros((0, 2))
    return {
        'status': 'ok', **info,
        'stair_xy_m': [round(float(mfu._arclen(s['xy'])[-1]), 3) for s in stairs],
        'snap_shift_m': [round(float(np.linalg.norm(s['E'] - xy[floors[s['k']]['i1']])), 3) for s in stairs]
                        + [round(float(np.linalg.norm(s['X'] - xy[floors[s['k'] + 1]['i0']])), 3) for s in stairs],
        'snap_reached_end': 0,
        'stairs_min_clearance_m': [float(c['points_min_m']) for c, p in zip(checks, pieces) if p['kind'] == 'stairs'],
        'stairs_spacing_used_m': [s['spacing'] for s in stairs],
        'trajectory': np.vstack(traj), 'trajectory_z': np.concatenate(tz), 'segments': segments,
        'floors': [{'i0': f['i0'], 'i1': f['i1'], 'level': f['level'], 'plan_z': f['plan_z']} for f in floors],
        'floor_z_start': floors[0]['plan_z'],
        'check_trajectory': agg,
        'check_refined': mfu._aggregate_checks(refined, [], args.plan_r_b) if refined else agg,
        'waypoints_astar': np.vstack([p['planned']['waypoints_astar'] for p in plan_parts]) if plan_parts else empty,
        'waypoints_refined': np.vstack([p['planned']['waypoints_refined'] for p in plan_parts]) if plan_parts else empty,
        'waypoints_thinned': np.vstack([p['planned']['waypoints_thinned'] for p in plan_parts]) if plan_parts else empty,
        'post_refine': [p['planned'].get('post_refine') for p in plan_parts],
    }


def _void_columns(g, plan_z, origin, cell) -> np.ndarray:
    """상자 안에서 층 바닥 높이(±FLOOR_Z_TOL_M)엔 면이 없고, 바닥보다 VOID_DROP_M 넘게 낮은 면만 있는 칸
    (전역 occupancy 의 ix, iy). 첫 디딤판 높이의 칸은 막지 않는다."""
    ix0, iy0, k0 = g['off']
    xyz = g['xyz']
    cols = np.floor((xyz[:, :2] - origin[:2]) / cell).astype(int)
    at = np.abs(xyz[:, 2] - plan_z) <= FLOOR_Z_TOL_M
    below = xyz[:, 2] < plan_z - VOID_DROP_M
    near = (xyz[:, 2] >= plan_z - VOID_DROP_M) & ~at          # 첫 디딤판 등 — 이런 면이 있는 칸은 구멍 아님
    key = cols[:, 0] * 1_000_000 + cols[:, 1]
    has_at = set(key[at | near].tolist())
    sel = below & ~np.isin(key, list(has_at))
    v = np.unique(cols[sel], axis=0)
    if len(v) == 0:
        return v
    from scipy.ndimage import binary_opening, label
    lo = v.min(0); m = np.zeros(tuple(v.max(0) - lo + 1), dtype=bool); m[tuple((v - lo).T)] = True
    m = binary_opening(m, structure=np.ones((3, 3), dtype=bool))
    lab, n = label(m, structure=np.ones((3, 3), dtype=int))
    if n:
        sizes = np.bincount(lab.ravel()); m &= (sizes >= VOID_MIN_CELLS)[lab] & (lab > 0)
    return np.argwhere(m) + lo


def _remap_z(old_xy, old_z, new_xy):
    """이음매로 xy 가 바뀐 계단 조각의 z — 새 점마다 옛 경로에서 가장 가까운 점의 z(순서 보존 위해 호길이 창 안에서)."""
    s_old = mfu._arclen(old_xy); s_new = mfu._arclen(new_xy)
    t = s_new / max(s_new[-1], 1e-9) * s_old[-1]
    return np.interp(t, s_old, old_z)
