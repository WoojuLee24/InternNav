"""pixel goal 유틸 — 3D goal ↔ 픽셀 변환, 가시성 검사, `r_b` 충돌 시 goal 조정.

## pixel goal의 정의 (실측 확정 — self-check로 검증: MAE 1.2/0.9 px, >5px 0.00%, N=1701)
frame `i`의 pixel goal = **frame `i+goal_len+1`의 카메라 위치 아래 바닥점**을
frame `i`의 **룩다운(pitch_2) 카메라**에 투영한 것. 저장 형식은 `[u, v] = [col, row]`, **640×480 정수**.
바닥 높이는 `cam_z − rig_height`로 **프레임마다 유도**한다(고정 0은 계단·층이동에서 깨진다).
(주의: `habitat_vln_evaluator.py:777`의 `cv2.circle`은 u/v를 바꿔 그리는 버그가 있다 — 시각화만 영향.)

투영은 **에피소드 start-relative 프레임**(`pose.{rig}`) 안에서 끝나므로 mesh 정합이 필요 없다.
반면 clearance(충돌) 판정·재계획은 occupancy가 mesh 좌표라 `T_sf2mesh`가 필요하다.

## 이 모듈이 하는 일
- `goal_world_from_poses` : 샘플의 pose 시퀀스 → 3D goal `G`(바닥점)
- `project_to_pixel`      : `G` + 현재 pose → `(u, v, Z)`
- `check_visible`         : 화면 안 + 전방 + **비가림**(저장 depth와 투영 Z 비교)
- `adjust_goal`           : `clearance(G) < r_b`일 때 goal 이동 — 세 기준
  · `retreat` 경로 따라 후퇴 · `shift` **goal heading에 수직(lateral)으로만** 이동해 틈 중앙 정렬
    → 목적지 유지하며 **통과**. 통과 불가면 retreat 폴백 · `nearest` 최소 변위(방향 무관)

self-check: `/usr/bin/python scripts/dataset_converters/3dloader_vlnce/pixel_goal_utils.py`
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_GS = _HERE.parents[0] / 'gs_vlnpe'
sys.path.insert(0, str(_GS)); sys.path.insert(0, str(_HERE))

IMG_W, IMG_H = 640, 480
# rig별 intrinsics. fx=(W/2)/tan(hfov/2), cx=(W-1)/2, cy=(H-1)/2 (habitat get_intrinsic_matrix).
# hfov가 리그마다 달라 fx가 다르다 — 125cm는 79°(fx≈388.19), 60cm는 ≈68.7°(fx≈465.8, 실측 피팅).
_RIG_HFOV = {125: 79.0, 60: 68.7}


def intrinsics_for_rig(rig: str):
    """'125cm_30deg' -> (fx, fy, cx, cy). 높이별 hfov가 달라 **반드시 rig로 구해야** 한다."""
    h_cm = int(str(rig).split('cm')[0])
    hfov = _RIG_HFOV.get(h_cm)
    assert hfov is not None, f'알 수 없는 rig 높이: {rig} (알려진 값 {sorted(_RIG_HFOV)})'
    fx = (IMG_W / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    return fx, fx, (IMG_W - 1) / 2.0, (IMG_H - 1) / 2.0


def rig_height_m(rig: str) -> float:
    """'125cm_30deg' -> 1.25 (카메라의 바닥 기준 높이)."""
    return int(str(rig).split('cm')[0]) / 100.0


def goal_world_from_poses(poses, rig='125cm_30deg', floor_z=None):
    """샘플의 pose 시퀀스(`pose[start : start+goal_len+1]`) → 3D goal `G`(바닥점, start-relative).

    goal은 **마지막 pose의 카메라 위치 아래 바닥점**이다. 바닥 높이는 `floor_z`를 주지 않으면
    **그 프레임에서 유도**한다: `cam_z − rig_height`.

    ⚠️ `floor_z=0` 고정은 틀린다 — 계단/층 이동 에피소드에선 카메라 z가 음수(예 −0.61)까지 가고,
    바닥도 함께 내려간다. 실측: 고정 0으로 두면 전체의 22.7%가 5px 초과(최악 19798px), 유도하면 해소.
    """
    p = np.asarray(poses[-1], dtype=np.float64).reshape(4, 4)
    fz = (p[2, 3] - rig_height_m(rig)) if floor_z is None else float(floor_z)
    return np.array([p[0, 3], p[1, 3], fz])


def project_to_pixel(G, pose_cur, rig='125cm_30deg'):
    """3D 점 `G`(world, start-relative) + 현재 카메라 pose(T_cam→world, OpenCV) -> `(u, v, Z)`.

    `Z`는 카메라 전방 거리[m] (가시성 검사에 쓴다). 화면 밖/뒤여도 그대로 반환하니 호출부가 검사할 것.
    """
    fx, fy, cx, cy = intrinsics_for_rig(rig)
    T = np.asarray(pose_cur, dtype=np.float64).reshape(4, 4)
    Xc = np.linalg.inv(T) @ np.array([G[0], G[1], G[2], 1.0])
    X, Y, Z = Xc[0], Xc[1], Xc[2]
    if Z <= 1e-6:
        return float('nan'), float('nan'), float(Z)
    return float(cx + fx * X / Z), float(cy + fy * Y / Z), float(Z)


def check_visible(u, v, Z, depth_img=None, depth_tol_m=0.35, min_forward_m=0.3):
    """채택 조건: 전방 + 화면 안 + (depth 주면) **비가림**. -> (ok: bool, reason: str)."""
    if not np.isfinite(u) or not np.isfinite(v) or Z <= 0:
        return False, 'behind_camera'
    if Z < min_forward_m:
        return False, 'too_close'
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < IMG_W and 0 <= vi < IMG_H):
        return False, 'out_of_image'
    if depth_img is not None:
        d = float(depth_img[vi, ui])
        # 저장 depth가 투영 Z보다 유의하게 **작으면** 그 픽셀은 앞의 물체에 가려진 것.
        if d > 0.05 and d < Z - depth_tol_m:
            return False, 'occluded'
    return True, 'ok'


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
# self-check — V1(좌표 규약·투영식)을 실제 데이터로 검증
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import glob
    import json
    import pyarrow.parquet as pq

    DR = 'data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r'
    errs, n_ep = [], 0
    for scene in ['17DRP5sb8fy', 's8pcmisQ38h']:
        for f in sorted(glob.glob(f'{DR}/{scene}/data/chunk-000/episode_*.parquet'))[:8]:
            for rig in ['125cm_30deg', '125cm_45deg', '60cm_30deg']:
                t = pq.read_table(f)
                if f'goal.{rig}' not in t.column_names:
                    continue
                poses = [np.asarray(p).reshape(4, 4) for p in t[f'pose.{rig}'].to_pylist()]
                goals = np.array(t[f'goal.{rig}'].to_pylist())
                rels = np.array(t[f'relative_goal_frame_id.{rig}'].to_pylist())
                for i in range(len(poses)):
                    if rels[i] < 0:
                        continue
                    j = i + int(rels[i]) + 1          # 실측 확정: +1
                    if j >= len(poses):
                        continue
                    G = goal_world_from_poses(poses[i:j + 1], rig)
                    u, v, Z = project_to_pixel(G, poses[i], rig)
                    if np.isfinite(u):
                        errs.append([abs(u - goals[i][0]), abs(v - goals[i][1])])
                n_ep += 1
    errs = np.array(errs)
    mae_u, mae_v = errs[:, 0].mean(), errs[:, 1].mean()
    p95 = np.percentile(errs.max(axis=1), 95)
    frac_bad = float((errs.max(axis=1) > 5).mean())
    print(f'[V1] N={len(errs)} goals ({n_ep} ep×rig) — MAE u={mae_u:.2f}px v={mae_v:.2f}px, '
          f'p95={p95:.2f}px, >5px={100*frac_bad:.2f}%')
    assert mae_u < 2.0 and mae_v < 2.0, f'투영식/좌표 규약 불일치: MAE u={mae_u:.2f} v={mae_v:.2f}'
    print('[V1] PASS — goal=[u,v]=[col,row], 바닥점(cam_z-rig_h) @ frame i+rel+1, rig별 intrinsics 확인')

    # adjust_goal 단위 점검
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
