"""층 이동 에피소드 — "층별 2D A* + 계단은 GT 그대로" 계산 모듈 (`--multifloor stairs` 전용).

기존 03/03u/04 의 계산은 건드리지 않는다. 03 의 `plan_episode` 를 층마다 그대로 불러 쓰고,
계단 구간은 GT 궤적(z 포함)을 그대로 이어 붙인다. 사족보행 로봇은 계단을 오른다고 보고
계단 구간에는 occupancy 를 쓰지 않는다.

    출발 층 지도에서 A*:  start      -> 계단 입구
    계단 구간 (GT 그대로): 계단 입구 -> 계단 출구
    도착 층 지도에서 A*:  계단 출구 -> goal

용어
----
- 점 바닥 z: GT 점마다 `body_z - (median(body_z) - ep_floor)`. vln_ce 는 cam_z - h_b,
  vln_pe 는 robot_z - torso 상수 와 같다(dataset_utils.load_gt_episode 의 floor_z 정의를 점별로 편 것).
- 평지(floor): 점 바닥 z 가 `level_tol` 안에서 이어지는 구간 중 xy 길이 >= `plateau_min_m` 인 것 + 첫/끝 구간.
- 계단(stairs): 이웃한 두 평지 사이의 GT 점 전부(짧은 참·작은 단차 포함).
- 입구/출구 물리기: 계단 칸이 그 층 지도에서 벽으로 나올 수 있다 — 입구를 GT 를 따라 평지 쪽으로
  물려 그 층 A* 격자에서 통행 가능한 첫 GT 점을 찾는다(최대 `snap_max_m`). 물린 구간은 GT 점 그대로
  계단 구간에 넣는다(지도를 비우면 계단 옆 벽까지 뚫릴 수 있어 이 방식을 택했다).

self-check: `python multifloor_utils.py`
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esdf_utils import (  # noqa: E402
    check_path_navigable,
    chamfer_distance,
    compute_esdf_2d,
    compute_scan_coverage_mask,
    derive_obstacle_2d,
    discrete_frechet_distance,
    downsample_navigable,
    path_smoothness,
    sample_esdf_at,
    truncate_navigable,
    world_to_cell,
)

# 03mf 인자 기본값 (근거는 03mf_plan_multifloor.py --help)
MF_DEFAULTS = {
    'dz_m': 0.5,            # 층 이동 판정 — 03u --dz_m 과 같다
    'level_tol_m': 0.1,     # 같은 평지로 볼 바닥 z 흔들림 (GT 실측 ±0.08)
    'plateau_min_m': 1.0,   # 이 길이 이상 이어져야 평지 (짧은 참은 계단에 흡수)
    'lead_m': 0.4,          # 입구/출구를 계단에서 최소 이만큼 평지 쪽으로 (이음매가 계단에 걸치지 않게)
    'snap_max_m': 2.0,      # 통행 가능한 GT 점을 찾아 물릴 최대 거리
    'seam_m': 0.3,          # 이음매 모서리 깎기 반폭
    'min_plan_m': 0.4,      # 평지 구간이 이보다 짧으면 A* 없이 GT 점 사용
}

FAIL_STATUSES = ('mf_entry_blocked', 'mf_sg_blocked', 'mf_astar_failed', 'mf_no_floor_plan', 'mf_stairs_no_path')


def is_multifloor(gt_xyz: np.ndarray, dz_m: float) -> bool:
    """03u `multifloor_episodes` 와 같은 규칙 — |z_끝 - z_시작| > dz_m (03 의 load_gt_episode body_xyz)."""
    return abs(float(gt_xyz[-1, 2] - gt_xyz[0, 2])) > dz_m


def point_floor_z(gt_xyz: np.ndarray, ep_floor: float) -> np.ndarray:
    """GT 점마다 바닥 z. ep_floor = median(cam_z) - h_b 이므로 body_z 에서 일정한 오프셋만 빼면 된다."""
    z = gt_xyz[:, 2]
    return z - (float(np.median(z)) - float(ep_floor))


def _arclen(xy: np.ndarray) -> np.ndarray:
    """누적 xy 호길이 (N,)"""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def segment_levels(fz: np.ndarray, xy: np.ndarray, level_tol: float, plateau_min_m: float) -> list:
    """점 바닥 z -> 평지 목록 [{'i0','i1','level'}] (인덱스 포함 구간, 순서대로).

    run: 앞에서부터 점을 붙이며 run 평균과 `level_tol` 이내면 같은 run. 평지 = 길이 >= plateau_min_m
    인 run + 첫/끝 run. 이웃 평지 사이(인덱스 i1 ~ 다음 i0)가 계단이다.
    """
    n = len(fz)
    runs, i = [], 0
    while i < n:
        j, total = i + 1, float(fz[i])
        while j < n and abs(float(fz[j]) - total / (j - i)) <= level_tol:
            total += float(fz[j])
            j += 1
        runs.append((i, j - 1))
        i = j
    s = _arclen(xy)
    floors = []
    for k, (a, b) in enumerate(runs):
        if k == 0 or k == len(runs) - 1 or (s[b] - s[a]) >= plateau_min_m:
            floors.append({'i0': a, 'i1': b, 'level': float(np.median(fz[a:b + 1]))})
    return floors


def floor_nav_coarse(occupancy, origin, floor_z, h_b, args):
    """`plan_episode` 와 같은 순서(obstacle->esdf->navigable->nav_coarse)로 그 층 지도를 만든다
    (`03.sample_random_start_goal` 과 같은 방식). 반환 (esdf, nav_coarse)."""
    cell = float(args.cell_m)
    h_nav = args.h_nav_ratio * h_b if args.h_nav_ratio else args.h_nav
    obstacle = derive_obstacle_2d(occupancy, origin, floor_z, h_b, cell, h_nav)
    esdf = compute_esdf_2d(obstacle, cell)
    navigable = truncate_navigable(esdf, args.plan_r_b) & compute_scan_coverage_mask(occupancy)
    return esdf, downsample_navigable(navigable, args.downsample_factor, args.downsample_mode)


def _nav_ok(nav_coarse, origin, coarse_cell, p) -> bool:
    ij = world_to_cell(np.asarray(p)[None, :], origin, coarse_cell)[0]
    h, w = nav_coarse.shape
    if not (0 <= ij[0] < w and 0 <= ij[1] < h):
        return False
    return bool(nav_coarse[ij[1], ij[0]])


def snap_index(xy, s, lo: int, hi: int, anchor: int, direction: int, nav_ok, lead_m, snap_max_m):
    """anchor(계단 쪽 평지 끝)에서 direction(-1=뒤로, +1=앞으로) 쪽 GT 점을 따라가며
    lead_m 이상 떨어지고 통행 가능한 첫 점을 찾는다. [lo, hi] 안에서만 찾는다.

    반환 (index, shift_m, how): how = 'ok' / 'reached_end'(평지 끝까지 와서 그 점 사용) / 'blocked'.
    """
    idx = anchor
    while True:
        d = abs(s[idx] - s[anchor])
        if d > snap_max_m:
            return None, d, 'blocked'
        if d >= lead_m and nav_ok(xy[idx]):
            return idx, d, 'ok'
        nxt = idx + direction
        if nxt < lo or nxt > hi:
            return idx, d, 'reached_end'
        idx = nxt


def _point_at_arclen(path: np.ndarray, target: float) -> tuple:
    """path 를 따라 호길이 target 위치의 점과 그 앞 꼭짓점 인덱스."""
    s = _arclen(path)
    target = float(np.clip(target, 0.0, s[-1]))
    k = int(np.searchsorted(s, target, side='right') - 1)
    k = min(max(k, 0), len(path) - 2)
    seg = s[k + 1] - s[k]
    u = 0.0 if seg <= 0 else (target - s[k]) / seg
    return path[k] + u * (path[k + 1] - path[k]), k


def _bezier(p0, p1, p2, step_m):
    n = max(3, int((np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1)) / step_m) + 1)
    u = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - u) ** 2) * p0 + 2 * u * (1 - u) * p1 + (u ** 2) * p2


def seam_end(floor_xy: np.ndarray, stair_xy: np.ndarray, seam_m: float, step_m: float):
    """평지 궤적 끝(= 계단 첫 점 J) 이음매 모서리를 2차 Bézier(P_a, J, P_b)로 깎는다.

    반환 (floor_xy', stair_xy') — floor 쪽은 P_a 까지 + Bézier 앞 절반, stair 쪽은 Bézier 뒤 절반 + P_b 이후.
    """
    lf, ls = _arclen(floor_xy)[-1], _arclen(stair_xy)[-1]
    w = min(seam_m, 0.5 * lf, 0.5 * ls)
    if w <= 1e-6 or len(floor_xy) < 2 or len(stair_xy) < 2:
        return floor_xy, stair_xy
    pa, ka = _point_at_arclen(floor_xy, lf - w)
    pb, kb = _point_at_arclen(stair_xy, w)
    j = floor_xy[-1]
    curve = _bezier(pa, j, pb, step_m)
    half = len(curve) // 2
    f_new = np.vstack([floor_xy[:ka + 1], curve[:half]])
    s_new = np.vstack([curve[half:], stair_xy[kb + 1:]])
    return f_new, s_new


def seam_start(stair_xy: np.ndarray, floor_xy: np.ndarray, seam_m: float, step_m: float):
    """계단 끝(= 평지 궤적 첫 점) 이음매. seam_end 를 뒤집어 쓴다. 반환 (stair_xy', floor_xy')."""
    f_rev, s_rev = seam_end(floor_xy[::-1], stair_xy[::-1], seam_m, step_m)
    return s_rev[::-1].copy(), f_rev[::-1].copy()


def _aggregate_checks(checks: list, clearances: list, r_b: float) -> dict:
    """평지 부분들의 check_path_navigable 결과를 한 dict 로 (03u 기존 판정식이 그대로 먹히도록 키를 맞춘다)."""
    allc = np.concatenate(clearances) if clearances else np.array([])
    return {
        'points_min_m': float(min(c['points_min_m'] for c in checks)),
        'points_median_m': float(np.median(allc)) if len(allc) else float('nan'),
        'segments_min_m': float(min(c['segments_min_m'] for c in checks)),
        'length_m': float(sum(c['length_m'] for c in checks)),
        'r_b': r_b,
        'points_ok': bool(all(c['points_ok'] for c in checks)),
        'segments_ok': bool(all(c['segments_ok'] for c in checks)),
        'n_samples': int(sum(c['n_samples'] for c in checks)),
        'hard_n': int(sum(c['hard_n'] for c in checks)),
        'hard_ok': bool(all(c['hard_ok'] for c in checks)),
        'scope': 'floor_parts_only',   # 계단 구간(GT 그대로)은 재지 않는다
    }


def plan_multifloor_episode(occupancy, origin, scene_floor_z, gt_xyz, h_b, ep_floor, args, plan_episode) -> dict:
    """층 이동 에피소드 하나. `plan_episode` 는 03 의 함수를 그대로 받는다(수정 없음).

    반환 dict: status ('ok' 또는 FAIL_STATUSES 중 하나) + 진단 정보, ok 면 궤적·segments.
    """
    xy = np.asarray(gt_xyz[:, :2], dtype=np.float64)
    fz = point_floor_z(gt_xyz, ep_floor)
    s = _arclen(xy)
    floors = segment_levels(fz, xy, args.mf_level_tol_m, args.mf_plateau_min_m)
    coarse_cell = float(args.cell_m) * args.downsample_factor
    info = {'n_stairs': len(floors) - 1,
            'levels': [round(f['level'], 3) for f in floors],
            'stair_dz_m': [round(floors[k + 1]['level'] - floors[k]['level'], 3) for k in range(len(floors) - 1)],
            'stair_xy_m': [round(float(s[floors[k + 1]['i0']] - s[floors[k]['i1']]), 3)
                           for k in range(len(floors) - 1)]}

    # 층별 계획 바닥 — 03 의 층 규칙 그대로(씬 대표 바닥과 floor_tol_m 이내면 씬 바닥)
    for f in floors:
        f['plan_z'] = f['level'] if abs(f['level'] - scene_floor_z) > args.floor_tol_m else scene_floor_z
        f['esdf'], f['nav'] = floor_nav_coarse(occupancy, origin, f['plan_z'], h_b, args)

    # 입구(평지 k 의 끝 쪽)·출구(평지 k 의 시작 쪽) 물리기
    shifts, hows = [], []
    for k, f in enumerate(floors):
        ok = lambda p, f=f: _nav_ok(f['nav'], origin, coarse_cell, p)  # noqa: E731
        f['a'], f['b'] = f['i0'], f['i1']
        if k > 0:      # 계단 출구 -> 앞으로
            idx, d, how = snap_index(xy, s, f['i0'], f['i1'], f['i0'], +1, ok, args.mf_lead_m, args.mf_snap_max_m)
            if how == 'blocked':
                return {'status': 'mf_entry_blocked', 'where': f'exit{k}', **info}
            f['a'] = idx; shifts.append(d); hows.append(how)
        if k < len(floors) - 1:   # 계단 입구 -> 뒤로
            idx, d, how = snap_index(xy, s, f['i0'], f['i1'], f['i1'], -1, ok, args.mf_lead_m, args.mf_snap_max_m)
            if how == 'blocked':
                return {'status': 'mf_entry_blocked', 'where': f'entry{k}', **info}
            f['b'] = idx; shifts.append(d); hows.append(how)
    info['snap_shift_m'] = [round(x, 3) for x in shifts]
    info['snap_reached_end'] = int(sum(h == 'reached_end' for h in hows))

    # 층별 A* — 평지 k: a -> b. 짧거나 물리기가 겹치면 GT 점을 그대로 쓴다.
    parts, n_planned = [], 0
    for k, f in enumerate(floors):
        if f['a'] >= f['b'] or (s[f['b']] - s[f['a']]) < args.mf_min_plan_m:
            lo, hi = min(f['a'], f['b']), max(f['a'], f['b'])
            f['a'], f['b'] = lo, hi
            parts.append({'kind': 'floor_gt', 'k': k, 'xy': xy[lo:hi + 1].copy()})
            continue
        planned = plan_episode(occupancy, origin, f['plan_z'], h_b, xy[f['a']], xy[f['b']], args)
        if planned['status'] == 'start_or_goal_not_navigable':
            return {'status': 'mf_sg_blocked', 'where': f'floor{k}', **info}
        if planned['status'] == 'astar_failed':
            return {'status': 'mf_astar_failed', 'where': f'floor{k}', **info}
        assert planned['status'] == 'ok', f"unreachable status={planned['status']!r}"
        n_planned += 1
        parts.append({'kind': 'floor', 'k': k, 'xy': np.asarray(planned['trajectory'], dtype=np.float64),
                      'planned': planned})
    if n_planned == 0:
        return {'status': 'mf_no_floor_plan', **info}

    # 계단 구간: 평지 k 의 b ~ 평지 k+1 의 a (GT 점 그대로)
    pieces = []
    for k, f in enumerate(floors):
        pieces.append(parts[k])
        if k < len(floors) - 1:
            a, b = f['b'], floors[k + 1]['a']
            pieces.append({'kind': 'stairs', 'k': k, 'xy': xy[a:b + 1].copy(), 'idx': np.arange(a, b + 1)})

    # 이음매 — 계획된 평지와 계단이 만나는 곳만 깎는다(GT 평지와 계단은 원래 GT 로 이어져 있다)
    for p in range(len(pieces) - 1):
        L, R = pieces[p], pieces[p + 1]
        if L['kind'] == 'floor' and R['kind'] == 'stairs':
            L['xy'], new = seam_end(L['xy'], R['xy'], args.mf_seam_m, args.smooth_step)
            R['xy'], R['idx'] = new, _reindex(R, new, keep='tail')
        elif L['kind'] == 'stairs' and R['kind'] == 'floor':
            new, R['xy'] = seam_start(L['xy'], R['xy'], args.mf_seam_m, args.smooth_step)
            L['xy'], L['idx'] = new, _reindex(L, new, keep='head')

    # z — 평지 = 그 층 계획 바닥, 계단 중 평지 run 에 속한 점 = 그 층 계획 바닥, 오르는 점 = GT 점 바닥
    traj, tz, segments = [], [], []
    n0 = 0
    for p in pieces:
        pxy = p['xy']
        if len(traj) and np.allclose(traj[-1][-1], pxy[0], atol=1e-9):
            pxy = pxy[1:]        # 이음 점 중복 제거
        if len(pxy) == 0:
            continue
        if p['kind'] == 'stairs':
            k = p['k']
            lo_f, hi_f = floors[k], floors[k + 1]
            z = []
            for gi in p['idx'][-len(pxy):]:
                if gi < 0:     # Bézier 로 새로 생긴 점(GT 인덱스 없음) — 이음매는 평지 쪽이다
                    z.append(lo_f['plan_z'] if -gi == 1 else hi_f['plan_z'])
                elif gi <= lo_f['i1']:
                    z.append(lo_f['plan_z'])
                elif gi >= hi_f['i0']:
                    z.append(hi_f['plan_z'])
                else:
                    z.append(float(fz[gi]))
            z = np.asarray(z)
            seg = {'kind': 'stairs', 'floor_z': [lo_f['plan_z'], hi_f['plan_z']]}
        else:
            z = np.full(len(pxy), floors[p['k']]['plan_z'])
            seg = {'kind': 'floor', 'floor_z': floors[p['k']]['plan_z'], 'planned': p['kind'] == 'floor'}
        seg.update({'i0': n0, 'i1': n0 + len(pxy) - 1})
        n0 += len(pxy)
        traj.append(pxy); tz.append(z); segments.append(seg)
    trajectory = np.vstack(traj)
    trajectory_z = np.concatenate(tz)

    # 판정용 — 계획된 평지 부분만 그 층 ESDF 로 다시 잰다(이음매 깎은 뒤)
    checks, clrs, refined = [], [], []
    for p in pieces:
        if p['kind'] != 'floor':
            continue
        esdf = floors[p['k']]['esdf']
        checks.append(check_path_navigable(p['xy'], esdf, origin, args.cell_m, args.r_b))
        clrs.append(sample_esdf_at(esdf, p['xy'], origin, args.cell_m))
        refined.append(p['planned']['check_refined'])
    plan_parts = [p for p in pieces if p['kind'] == 'floor']
    return {
        'status': 'ok', **info,
        'trajectory': trajectory, 'trajectory_z': trajectory_z, 'segments': segments,
        'floors': [{'i0': f['i0'], 'i1': f['i1'], 'level': f['level'], 'plan_z': f['plan_z'],
                    'a': int(f['a']), 'b': int(f['b'])} for f in floors],
        'floor_z_start': floors[0]['plan_z'],
        'check_trajectory': _aggregate_checks(checks, clrs, args.r_b),
        'check_refined': _aggregate_checks(refined, [], args.plan_r_b),
        'waypoints_astar': np.vstack([p['planned']['waypoints_astar'] for p in plan_parts]),
        'waypoints_refined': np.vstack([p['planned']['waypoints_refined'] for p in plan_parts]),
        'waypoints_thinned': np.vstack([p['planned']['waypoints_thinned'] for p in plan_parts]),
        'post_refine': [p['planned'].get('post_refine') for p in plan_parts],
        'floor_esdf': {p['k']: floors[p['k']]['esdf'] for p in pieces},   # 시각화용 (JSON 에는 안 나간다)
    }


def _reindex(piece, new_xy, keep: str) -> np.ndarray:
    """계단 조각의 점이 이음매에서 바뀌었을 때 GT 인덱스 배열을 맞춘다. 새로 생긴 점은 음수
    (-1 = 아래 평지 쪽, -2 = 위 평지 쪽)로 표시한다."""
    old_xy, old_idx = piece['xy'], piece['idx']
    n_new = len(new_xy)
    if keep == 'tail':   # 앞쪽이 Bézier 로 바뀜 — 뒤쪽 GT 점은 그대로
        n_keep = 0
        while n_keep < min(len(old_xy), n_new) and np.allclose(old_xy[-1 - n_keep], new_xy[-1 - n_keep]):
            n_keep += 1
        return np.concatenate([np.full(n_new - n_keep, -1), old_idx[len(old_idx) - n_keep:]])
    n_keep = 0
    while n_keep < min(len(old_xy), n_new) and np.allclose(old_xy[n_keep], new_xy[n_keep]):
        n_keep += 1
    return np.concatenate([old_idx[:n_keep], np.full(n_new - n_keep, -2)])


def gt_compare_xy(traj_xy: np.ndarray, gt_xy: np.ndarray, door_width_m: float) -> dict:
    """층 이동용 축약 GT 대조(ESDF 가 층마다 달라 clearance 지표는 뺀다). 03u 가 읽는 키 포함."""
    sm_o, sm_g = path_smoothness(traj_xy), path_smoothness(gt_xy)
    gl = float(_arclen(gt_xy)[-1]); ol = float(_arclen(traj_xy)[-1])
    fr = discrete_frechet_distance(traj_xy, gt_xy)
    return {'chamfer_m': chamfer_distance(traj_xy, gt_xy), 'frechet_m': fr,
            'same_route': bool(fr < door_width_m),
            'len_ratio': ol / gl if gl > 0 else float('nan'), 'gt_length_m': gl, 'our_length_m': ol,
            'turn_per_m_deg': sm_o['turn_per_m_deg'], 'gt_turn_per_m_deg': sm_g['turn_per_m_deg'],
            'scope': 'xy_only'}


# ---------------------------------------------------------------------------
# 04 용 — 점마다 z 가 있는 궤적
# ---------------------------------------------------------------------------

def resample_xyz_by_arclength(xy: np.ndarray, z: np.ndarray, step_m: float) -> tuple:
    """`esdf_utils.resample_by_arclength` 와 같은 xy 호길이 등간격 + z 도 같은 호길이로 보간."""
    from esdf_utils import resample_by_arclength
    p = np.asarray(xy, dtype=np.float64)
    out_xy = resample_by_arclength(p, step_m=step_m)
    s = _arclen(p)
    if s[-1] <= 0:
        return out_xy, np.full(len(out_xy), float(z[0]))
    t = np.linspace(0.0, s[-1], len(out_xy))
    return out_xy, np.interp(t, s, np.asarray(z, dtype=np.float64))


def synthesize_action_poses_z(xy: np.ndarray, floor_z_pts: np.ndarray, h_b: float, pitch_deg: float) -> np.ndarray:
    """`geometry_utils.synthesize_action_poses` 로 회전을 만든 뒤 카메라 z 만 점마다 바닥 z + h_b 로 바꾼다.
    action 행렬은 c2w_cv @ CAM_CV_TO_GL 이고 CAM_CV_TO_GL 은 translation 을 건드리지 않으므로 [:,2,3] 이 카메라 z 다."""
    from geometry_utils import synthesize_action_poses
    poses = synthesize_action_poses(xy, 0.0, h_b, pitch_deg)
    poses[:, 2, 3] = np.asarray(floor_z_pts, dtype=np.float64) + h_b
    return poses


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def _self_test() -> int:
    def prof(levels_and_lens, step=0.25, stair_step=0.18):
        """[(level, 길이 m), ...] 사이를 계단(스텝 0.18 m)으로 잇는 GT 모양 직선 궤적."""
        z, x = [], []
        cur_x = 0.0
        for i, (lv, ln) in enumerate(levels_and_lens):
            if i > 0:
                prev = levels_and_lens[i - 1][0]
                n = int(abs(lv - prev) / stair_step)
                for t in range(1, n + 1):
                    cur_x += step; x.append(cur_x); z.append(prev + (lv - prev) * t / (n + 1))
            for _ in range(max(1, int(ln / step))):
                cur_x += step; x.append(cur_x); z.append(lv + np.random.uniform(-0.05, 0.05))
        xy = np.stack([np.array(x), np.zeros(len(x))], axis=1)
        return np.array(z), xy

    np.random.seed(0)
    cases = {
        '한 번':        ([(0.0, 4.0), (3.0, 4.0)], 1),
        '두 번(중간 평지)': ([(3.9, 3.0), (3.2, 2.0), (1.4, 4.0)], 2),
        '짧은 참':      ([(0.0, 3.0), (1.5, 0.5), (3.0, 3.0)], 1),
        '시작이 계단':   ([(0.0, 0.25), (1.0, 5.0)], 1),
        '작은 단차':    ([(3.6, 2.0), (4.4, 3.0), (4.2, 2.0)], 2),
    }
    bad = 0
    for name, (spec, want) in cases.items():
        z, xy = prof(spec)
        fl = segment_levels(z, xy, 0.1, 1.0)
        got = len(fl) - 1
        ok = got == want and abs(fl[0]['level'] - spec[0][0]) < 0.1 and abs(fl[-1]['level'] - spec[-1][0]) < 0.1
        bad += not ok
        print(f'  [{"PASS" if ok else "FAIL"}] {name}: 계단 {got} (기대 {want}) · 층 {[round(f["level"], 2) for f in fl]}')

    # snap: 계단 쪽 2 점이 막힌 경우 -> lead 0.4 이후 첫 통행 가능 점
    xy = np.stack([np.arange(0, 3.01, 0.25), np.zeros(13)], axis=1)
    s = _arclen(xy)
    blocked = {12, 11, 10, 9}
    ok_fn = lambda p: int(round(p[0] / 0.25)) not in blocked  # noqa: E731
    idx, d, how = snap_index(xy, s, 0, 12, 12, -1, ok_fn, 0.4, 2.0)
    ok = idx == 8 and abs(d - 1.0) < 1e-9 and how == 'ok'
    bad += not ok
    print(f'  [{"PASS" if ok else "FAIL"}] snap: idx {idx} shift {d:.2f} {how} (기대 8 / 1.00 / ok)')
    idx, d, how = snap_index(xy, s, 0, 12, 12, -1, lambda p: False, 0.4, 2.0)
    ok = how == 'blocked'
    bad += not ok
    print(f'  [{"PASS" if ok else "FAIL"}] snap 전부 막힘 -> {how}')

    # seam: 직각 이음매가 깎이고 양 끝은 그대로
    f = np.stack([np.linspace(0, 1, 21), np.zeros(21)], axis=1)
    st = np.stack([np.ones(21), np.linspace(0, 1, 21)], axis=1)
    f2, s2 = seam_end(f, st, 0.3, 0.05)
    turn = np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(*np.diff(np.vstack([f2, s2[1:]]), axis=0).T[::-1])))).max())
    ok = np.allclose(f2[0], f[0]) and np.allclose(s2[-1], st[-1]) and turn < 30
    bad += not ok
    print(f'  [{"PASS" if ok else "FAIL"}] seam: 최대 꺾임 {turn:.1f}° (< 30°), 양 끝 유지')

    # 04 pose: z 가 점마다 바닥 + h_b
    from geometry_utils import synthesize_action_poses
    xy2 = np.stack([np.linspace(0, 2, 9), np.zeros(9)], axis=1)
    zz = np.linspace(0, 1, 9)
    p_new = synthesize_action_poses_z(xy2, zz, 1.25, 30.0)
    p_old = synthesize_action_poses(xy2, 0.0, 1.25, 30.0)
    ok = np.allclose(p_new[:, 2, 3], zz + 1.25) and np.allclose(p_new[:, :3, :3], p_old[:, :3, :3]) \
        and np.allclose(p_new[:, :2, 3], p_old[:, :2, 3])
    bad += not ok
    print(f'  [{"PASS" if ok else "FAIL"}] 04 pose: z = 바닥 + h_b, 회전·xy 는 기존 함수와 동일')
    print(f'=> {"PASS" if bad == 0 else f"FAIL ({bad})"}')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(_self_test())
