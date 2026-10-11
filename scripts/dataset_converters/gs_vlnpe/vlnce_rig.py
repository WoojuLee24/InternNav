"""vln_ce rig 기하 — 3D goal ↔ 픽셀 변환, 가시성 검사.

`3dloader_vlnce/pixel_goal_utils.py`에서 **포맷·투영 부분만** 떼어 온 것이다(2026-10-11).
가른 기준은 **`r_b`(로봇 반경)를 인자로 받는가** — 받으면 embodiment augmentation이라
`3dloader_vlnce/goal_adjust.py`에 남겼고, 안 받는 여기 함수들은 vln_ce 포맷을 읽고 쓰는 데
필요한 기하라서 데이터 생성 쪽(`gs_vlnpe`)이 가져야 한다. 이전에는 `gs_vlnpe`가 거꾸로
`3dloader_vlnce`를 sys.path로 끌어다 써서 양방향 의존이었다.

## pixel goal의 정의 (실측 확정 — self-check로 검증: MAE 1.2/0.9 px, >5px 0.00%, N=1701)
frame `i`의 pixel goal = **frame `i+goal_len+1`의 카메라 위치 아래 바닥점**을
frame `i`의 **룩다운(pitch_2) 카메라**에 투영한 것. 저장 형식은 `[u, v] = [col, row]`, **640×480 정수**.
바닥 높이는 `cam_z − rig_height`로 **프레임마다 유도**한다(고정 0은 계단·층이동에서 깨진다).
(주의: `habitat_vln_evaluator.py:777`의 `cv2.circle`은 u/v를 바꿔 그리는 버그가 있다 — 시각화만 영향.)

투영은 **에피소드 start-relative 프레임**(`pose.{rig}`) 안에서 끝나므로 mesh 정합이 필요 없다.

## 이 모듈이 하는 일
- `intrinsics_for_rig`     : rig 이름 → (fx, fy, cx, cy). **높이마다 hfov가 다르다**
- `rig_height_m`           : rig 이름 → 카메라 높이[m]
- `goal_world_from_poses`  : 샘플의 pose 시퀀스 → 3D goal `G`(바닥점)
- `project_to_pixel`       : `G` + 현재 pose → `(u, v, Z)`
- `check_visible`          : 화면 안 + 전방 + **비가림**(저장 depth와 투영 Z 비교)

self-check: `/usr/bin/python scripts/dataset_converters/gs_vlnpe/vlnce_rig.py`
"""

import numpy as np

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
