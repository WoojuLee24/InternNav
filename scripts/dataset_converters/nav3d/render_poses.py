"""경로·걷기 기록 -> 렌더 카메라 자세 (06 · 06p · 06m · 06m2 · 06v · 06n 이 같이 쓴다).

`06_render_noeun_vlnce.py` (body_floor_z · episode_poses · start_yaw_for), `06p_render_noeun_vlnpe.py`
(c2w_to_vlnpe_quat · ce_to_pe_actions), `06m_render_rbq.py` (D455 장착값 · quat_wxyz_to_R · trunk_to_cam_cv ·
frames_to_poses) 에서 옮겼다 (코드 그대로). 세 파일은 이 이름들을 다시 import 해서 그대로 노출한다.
"""

import json
import sys
from pathlib import Path

import numpy as np

GS = Path(__file__).resolve().parent.parent / 'gs_vlnpe'
for p in (str(GS), str(GS / 'apply_real')):
    if p not in sys.path:
        sys.path.insert(0, p)


def body_floor_z(xy: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """이산 몸 위치(순서대로) -> 04 경로의 바닥 z. 경로 앞쪽으로만 나아가며(겹친 층에서 다른 층으로 튀지 않게) xy 투영."""
    a, b = traj[:-1], traj[1:]
    ab = b[:, :2] - a[:, :2]
    L2 = np.maximum((ab * ab).sum(1), 1e-12)
    out, k = np.empty(len(xy)), 0
    for i, p in enumerate(xy):
        lo, hi = max(0, k - 5), min(len(a), k + 80)
        t = np.clip(((p - a[lo:hi, :2]) * ab[lo:hi]).sum(1) / L2[lo:hi], 0, 1)
        d = np.linalg.norm(a[lo:hi, :2] + t[:, None] * ab[lo:hi] - p, axis=1)
        j = int(np.argmin(d))
        k = lo + j
        out[i] = a[k, 2] + t[j] * (b[k, 2] - a[k, 2])
    return out


def episode_poses(traj: np.ndarray, prof, forward_m: float, waypoint_spacing_m: float, initial_yaw: float = None) -> tuple:
    """04 경로(점마다 바닥 z) -> (이산화 결과, 몸 바닥 z, 카메라 c2w OpenCV (N,4,4)). 06 렌더와 06v 검증이 같이 쓴다.
    initial_yaw = 시작 heading (nav3d yaw, rad). None 이면 첫 구간 방향 (제자리 회전 없이 출발 — 이전 동작)."""
    import vlnce_discretize as vd
    from geometry_utils import action_to_c2w
    disc = vd.discretize(traj[:, :2], initial_yaw=initial_yaw, waypoint_spacing_m=waypoint_spacing_m)
    zb = body_floor_z(disc['xy'], traj)
    cam_xy = disc['xy'] + forward_m * np.c_[np.cos(disc['yaw']), np.sin(disc['yaw'])]
    c2w = vd.poses_from_yaw(cam_xy, disc['yaw'], 0.0, prof.mount_height_m, prof.pitch_down_deg)
    c2w[:, 2, 3] += zb                                    # 몸 바닥 z 를 프레임마다
    poses_cv = np.stack([action_to_c2w(a, 'cam2world_gl') for a in c2w])
    return disc, zb, poses_cv


_R2R_INIT = None


def start_yaw_for(ep: dict, traj: np.ndarray, mode: str, seed: int, raw_json: str):
    """에피소드 시작 heading. ep['start_yaw'] (GT, 예: MP3D) 가 있으면 그것.
    mode=path -> None (첫 구간 방향, 이전 동작) / r2r_train -> R2R train 의 (시작 방향 − 첫 구간) 각 분포에서 뽑는다
    (실측 2026-10-18: 거의 균등, 중앙 80°; vln_pe 84 % 가 제자리 회전으로 시작)."""
    global _R2R_INIT
    if ep.get('start_yaw') is not None:
        return float(ep['start_yaw'])
    if mode == 'path':
        return None
    elif mode == 'r2r_train':
        if _R2R_INIT is None:
            import gzip
            import math
            E = json.load(gzip.open(raw_json))['episodes']
            out = []
            for e in E:   # route_words.route_geometry 의 initial 과 같은 정의 (나침반, + = 첫 구간이 시작 방향의 오른쪽)
                p, q = e['reference_path'][0], e['reference_path'][1]
                seg = math.degrees(math.atan2(q[0] - p[0], -(q[2] - p[2])))
                st = -math.degrees(2 * math.atan2(e['start_rotation'][1], e['start_rotation'][3]))
                out.append((seg - st + 180) % 360 - 180)
            _R2R_INIT = np.asarray(out)
        d = traj[min(len(traj) - 1, 5), :2] - traj[0, :2]
        import hashlib
        rng = np.random.default_rng(int(hashlib.sha1(f"{seed}:{ep['episode_id']}".encode()).hexdigest()[:8], 16))
        init = float(rng.choice(_R2R_INIT))
        return float(np.arctan2(d[1], d[0]) + np.radians(init))   # nav3d yaw(반시계 +) = 첫 구간 + initial
    else:
        assert False, f'unreachable start_heading={mode!r}'


def c2w_to_vlnpe_quat(R_cv: np.ndarray) -> list:
    """OpenCV c2w 회전 -> vln_pe camera_orientation (w, x, y, z). dataset_utils.vlnpe_orientation_to_c2w_rotation 의 역."""
    from scipy.spatial.transform import Rotation
    r = np.column_stack([R_cv[:, 2], -R_cv[:, 0], -R_cv[:, 1]])
    x, y, z, w = Rotation.from_matrix(r).as_quat()
    return [float(w), float(x), float(y), float(z)]


def ce_to_pe_actions(a_ce: np.ndarray) -> np.ndarray:
    """vln_ce action (a[0] = −1, a[t] = t 로 온 action) -> vln_pe (a[i] = i 에서 한 action, 마지막 0)."""
    a = np.asarray(a_ce, dtype=np.int64)
    return np.r_[a[1:], 0]


T_BASE_OS = np.array([0.3, 0.0, 0.05])
T_OS_CAM = np.array([0.065151, 0.002019, -0.015822])
Q_OS_CAM_XYZW = np.array([-0.000391, 0.128428, 0.001329, 0.991718])


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def trunk_to_cam_cv() -> np.ndarray:
    """trunk(base_link) → OpenCV 카메라 4×4."""
    x, y, z, w = Q_OS_CAM_XYZW
    R_link = quat_wxyz_to_R([w, x, y, z])
    R_link_cv = np.column_stack([-np.array([0, 1, 0.0]), -np.array([0, 0, 1.0]), np.array([1, 0, 0.0])])   # cv x=-y, y=-z, z=x
    T = np.eye(4)
    T[:3, :3] = R_link @ R_link_cv
    T[:3, 3] = T_BASE_OS + T_OS_CAM
    return T


def frames_to_poses(frames) -> np.ndarray:
    M = trunk_to_cam_cv()
    out = []
    for f in frames:
        T = np.eye(4)
        T[:3, :3] = quat_wxyz_to_R(f['quat_wxyz'])
        T[:3, 3] = f['pos']
        out.append(T @ M)
    return np.stack(out)
