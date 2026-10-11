"""embodiment augmentation — `r_b`(로봇 반경) 때문에 못 쓰게 된 pixel goal을 옮긴다.

`pixel_goal_utils.py`에서 **`r_b`를 인자로 받는 부분만** 남긴 것이다(2026-10-11).
나머지(투영·가시성·rig intrinsics)는 vln_ce 포맷 기하라서
`gs_vlnpe/vlnce_rig.py`로 옮겼다 — 거긴 데이터 생성 쪽이고 여긴 augmentation 쪽이다.

- `clearance_at`   : mesh 좌표의 ESDF 값(장애물까지 거리[m])
- `adjust_goal`    : `clearance(G) < r_b`일 때 goal 이동 — 세 기준
  · `retreat` 경로 따라 후퇴 · `shift` **goal heading에 수직(lateral)으로만** 이동해 틈 중앙 정렬
    → 목적지 유지하며 **통과**. 통과 불가면 retreat 폴백 · `nearest` 최소 변위(방향 무관)

clearance(충돌) 판정·재계획은 occupancy가 mesh 좌표라 `T_sf2mesh`가 필요하다
(투영은 start-relative 프레임 안에서 끝나므로 정합이 필요 없다 — `vlnce_rig` 쪽 참고).

self-check: `/usr/bin/python scripts/dataset_converters/3dloader_vlnce/goal_adjust.py`
"""

import numpy as np

def clearance_at(esdf, origin, cell_m, xy):
    """mesh 좌표 xy의 ESDF(장애물까지 거리[m]). 격자 밖이면 0."""
    ij = np.floor((np.asarray(xy, dtype=np.float64)[:2] - np.asarray(origin)[:2]) / cell_m).astype(int)
    h, w = esdf.shape
    if not (0 <= ij[0] < w and 0 <= ij[1] < h):
        return 0.0
    return float(esdf[ij[1], ij[0]])


def _heading_at_end(path_xy, n_back=3):
    """경로 끝점에서의 진행 방향(단위벡터). 마지막 몇 구간을 평균해 노이즈를 줄인다."""
    p = np.asarray(path_xy, dtype=np.float64)[:, :2]
    if len(p) < 2:
        return None
    k = min(n_back, len(p) - 1)
    d = p[-1] - p[-1 - k]
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-9:
        return None
    return d / nrm


def adjust_goal(goal_xy_mesh, r_b, esdf, origin, cell_m, mode='retreat',
                path_xy_mesh=None, search_radius_m=1.0, robot_xy_mesh=None,
                max_lateral_m=1.0, lateral_step_m=0.05):
    """`clearance < r_b`인 goal을 안전한 위치로 옮긴다. -> (new_xy or None, status).

    세 기준:
    - `retreat` : `path_xy_mesh`(원본 경로, 시작→goal 순)를 **뒤에서부터** 훑어 `clearance ≥ r_b`인
      **가장 먼** 점. 방향·경로 유지, `r_b`에 **단조**(큰 로봇일수록 덜 감). **거리↓ 방위 유지**.
    - `shift`   : goal 지점의 **진행 방향(heading)에 수직인 lateral 방향으로만** 최소 이동
      (`±max_lateral_m`). 좁은 틈의 **중앙으로 정렬**해 **원래 목적지를 유지한 채 통과**하게 만든다.
      lateral로도 안 되면(=통과 불가) **경로를 따라 뒤로 물린다**(status `adjusted_retreat`).
      ⚠️ 로봇→goal 방위를 회전시키면(초기 구현) 통과가 아니라 **옆으로 돌아가** 목적지가 바뀐다 —
      pixel goal은 GT 경로상의 pose이므로 그 지점의 heading을 써야 한다.
    - `nearest` : goal 주변 `search_radius_m` 안 **최소 변위** 안전점(방향 무관).

    status: 'unchanged'(원래 안전) | 'adjusted' | 'no_safe_point'
    """
    g = np.asarray(goal_xy_mesh, dtype=np.float64)[:2]
    if clearance_at(esdf, origin, cell_m, g) >= r_b:
        return g, 'unchanged'

    if mode == 'retreat':
        assert path_xy_mesh is not None, "retreat 모드는 path_xy_mesh가 필요하다"
        path = np.asarray(path_xy_mesh, dtype=np.float64)[:, :2]
        for k in range(len(path) - 1, -1, -1):          # 먼 쪽부터
            if clearance_at(esdf, origin, cell_m, path[k]) >= r_b:
                return path[k], 'adjusted'
        return None, 'no_safe_point'

    elif mode == 'shift':
        # **goal 지점의 진행 방향(heading)에 수직인 lateral 방향으로만** 옮긴다.
        # 로봇→goal 방위를 회전시키면(이전 구현) 목적지를 유지한 채 통과하는 게 아니라 **옆으로 돌아가** 버린다.
        # pixel goal은 GT 경로상의 pose이므로 그 지점의 heading이 있고, 그 수직(lateral)으로 옮기면
        # 좁은 틈의 **중앙으로 정렬**돼 목적지를 유지하면서 통과할 수 있다.
        assert path_xy_mesh is not None, "shift 모드는 path_xy_mesh(heading 추정용)가 필요하다"
        path = np.asarray(path_xy_mesh, dtype=np.float64)[:, :2]
        head = _heading_at_end(path)
        if head is None:
            return None, 'no_safe_point'
        lat = np.array([-head[1], head[0]])                      # heading에 수직
        n = max(1, int(np.ceil(max_lateral_m / lateral_step_m)))
        for k in range(1, n + 1):
            for sgn in (+1, -1):
                cand = g + sgn * (k * lateral_step_m) * lat
                if clearance_at(esdf, origin, cell_m, cand) >= r_b:
                    return cand, 'adjusted'
        # lateral로 못 비키면 **통과 자체가 불가** → 경로를 따라 뒤로 물린다(retreat 폴백).
        for k in range(len(path) - 1, -1, -1):
            if clearance_at(esdf, origin, cell_m, path[k]) >= r_b:
                return path[k], 'adjusted_retreat'
        return None, 'no_safe_point'

    elif mode == 'nearest':
        rad = int(np.ceil(search_radius_m / cell_m))
        ij = np.floor((g - np.asarray(origin)[:2]) / cell_m).astype(int)
        h, w = esdf.shape
        best, best_d = None, np.inf
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                x, y = ij[0] + dx, ij[1] + dy
                if not (0 <= x < w and 0 <= y < h) or esdf[y, x] < r_b:
                    continue
                d = dx * dx + dy * dy
                if d < best_d:
                    best_d, best = d, np.array([origin[0] + (x + 0.5) * cell_m,
                                                origin[1] + (y + 0.5) * cell_m])
        return (best, 'adjusted') if best is not None else (None, 'no_safe_point')

    else:
        assert False, f'unreachable mode={mode!r}'

# ---------------------------------------------------------------------------
# self-check — adjust_goal 세 모드 단위 점검
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    esdf = np.full((40, 40), 1.0); esdf[20, 20] = 0.05      # (20,20) 셀만 좁음
    origin, cell = np.array([0.0, 0.0]), 0.1
    g = np.array([20.5 * cell, 20.5 * cell])
    out, st = adjust_goal(g, 0.3, esdf, origin, cell, mode='nearest')
    assert st == 'adjusted' and out is not None, (st, out)
    path = np.stack([np.linspace(0.5, g[0], 8), np.linspace(0.5, g[1], 8)], axis=1)
    out2, st2 = adjust_goal(g, 0.3, esdf, origin, cell, mode='retreat', path_xy_mesh=path)
    assert st2 == 'adjusted' and np.linalg.norm(out2 - g) > 0, (st2, out2)
    safe, st3 = adjust_goal(np.array([1.0, 1.0]), 0.3, esdf, origin, cell, mode='nearest')
    assert st3 == 'unchanged'
    # shift: goal의 **heading에 수직(lateral)** 으로만 이동해야 한다
    # 좁은 틈: y=20 행에서 x=18..22만 좁게 만들고, 경로는 +x 방향으로 진행 → lateral은 ±y
    esdf2 = np.full((40, 40), 1.0)
    esdf2[20, 18:23] = 0.05
    path2 = np.stack([np.arange(10, 21) * cell + 0.5 * cell,
                      np.full(11, 20.5 * cell)], axis=1)      # +x로 진행, y 고정
    g2 = path2[-1]
    out3, st4 = adjust_goal(g2, 0.3, esdf2, origin, cell, mode='shift', path_xy_mesh=path2)
    assert st4 == 'adjusted' and out3 is not None, (st4, out3)
    head = _heading_at_end(path2)
    along = abs(float(np.dot(out3 - g2, head)))               # 진행 방향 성분 ≈ 0이어야 한다
    lateral = abs(float(np.dot(out3 - g2, [-head[1], head[0]])))
    assert along < 1e-9 < lateral, f'shift가 lateral이 아니다: along={along:.4f} lat={lateral:.4f}'
    # lateral로 못 비키는 경우(전 구간이 좁음) → retreat 폴백
    esdf3 = np.full((40, 40), 1.0); esdf3[:, 18:23] = 0.05; esdf3[20, :] = 0.05
    out4, st5 = adjust_goal(g2, 0.3, esdf3, origin, cell, mode='shift', path_xy_mesh=path2)
    assert st5 in ('adjusted_retreat', 'no_safe_point'), st5
    print(f'[V1] adjust_goal (retreat/shift/nearest/unchanged) OK '
          f'— shift는 lateral {lateral:.2f} m만 이동(진행방향 성분 {along:.1e}), 막히면 {st5}')
