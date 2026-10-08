"""vln_ce 절대정합 — start-relative `pose.{rig}`를 mesh 절대좌표로 올리는 rigid 변환을 구한다.

G1이 규명: vln_ce `pose.{rig}`는 에피소드 start-relative다(학습은 상대좌표만 쓴다). augmenter는
cached mesh/occ에서 depth·경로를 만들려면 절대좌표가 필요하다. 상대 pose는 내부 일관되므로,
에피소드 depth를 start-frame으로 누적한 point cloud를 mesh(Z-up)에 **강체 정합(yaw+평행이동)** 하면
`T_sf2mesh`가 나온다. raw_data `start_position`은 초기화용(규약 몰라도 fit이 흡수). residual이 곧 정합 지표.

ep0(17DRP5sb8fy) 실측: point-to-mesh median 0.0039 m, yaw 60°, t=axis-map('x,-z,y') of start_position.

offline converter(05) / augmenter / G1이 공용으로 쓴다. 순수 numpy/scipy + trimesh(거리) + gs 재사용.
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gs_vlnpe'))  # 기존 코드 import
from geometry_utils import compute_mesh_distance, unproject_to_world_frame  # noqa: E402

# dataloader와 동일 하드코딩 K (internvla_n1_lerobot_dataset.py:1026, 640x480 native)
K_VLNCE = np.array([[388.19, 0.0, 319.5], [0.0, 388.19, 239.5], [0.0, 0.0, 1.0]], dtype=np.float64)
POSE_CONVENTION = 'cam2world'  # pose.{rig}는 start-frame OpenCV c2w (G1: z-range=floor..ceil로 확인)


def build_rawdata_index(raw_root: Path) -> dict:
    """(scan, instruction_text) -> raw episode dict. 모든 split(train/val_*) 병합. start_position 보유."""
    idx = {}
    for fp in sorted(Path(raw_root).glob('*/*.json.gz')):
        d = json.load(gzip.open(fp))
        eps = d['episodes'] if isinstance(d, dict) and 'episodes' in d else d
        for e in eps:
            scan = Path(e['scene_id']).stem
            it = e['instruction']['instruction_text'].strip()
            idx.setdefault((scan, it), e)
    return idx


def _rigid(p):
    yaw, tx, ty, tz = p
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [tx, ty, tz]
    return T


def accumulate_startframe_cloud(frames, k=K_VLNCE, conv=POSE_CONVENTION, max_pts=4000, seed=0):
    """frames=[(f, pose_rel(4x4), depth_m, valid)] -> start-frame point cloud (N,3) Z-up."""
    pts = []
    for _f, pose, depth_m, valid in frames:
        pts.append(unproject_to_world_frame(depth_m, k, pose, conv)[valid])
    pts = np.concatenate(pts)
    rng = np.random.RandomState(seed)
    if len(pts) > max_pts:
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]
    return pts


# habitat start_position(Y-up) -> mesh(Z-up) 카메라-start translation의 해석적 affine.
#   t_x = start_x,  t_y = -start_z,  t_z = start_y - Z_OFFSET_M
# 다층 씬(예: s8pcmisQ38h, mesh z-range 12m)에서도 start_y가 에피소드 층을 직접 지정 → 올바른 층에 안착.
#
# 2026-08-19: **Z_OFFSET_M을 0.20 -> 0.0으로 정정.** 0.20은 "t_z가 mesh 바닥(17DRP bbox min z=-0.128)에
# 오도록" 역산한 fudge였고, 실제로는 offset이 없어야 한다(t_z = start_y 그대로).
# 근거 — 렌더 depth vs 저장 depth를 tz 스윕으로 비교(17DRP·s8, rig 125cm_0deg/30deg, ep 0/2/3/10/12 = 9케이스):
#   offset 0.20: median 5~61 mm, |err|>10cm 픽셀 22~47%
#   offset 0.00: median  0.5 mm, |err|>10cm 픽셀  0.0~0.6%   ← 9/9 전부 이쪽이 압도적
# 왜 여태 못 잡았나: G1a/G1b가 **median**만 봤다. z 오차는 수직 벽의 depth를 거의 바꾸지 않고
# **바닥·천장(시선에 스치는 면)에서만** 커지므로, 픽셀의 ~75%가 멀쩡해 median이 4~5 mm로 나왔다.
# → G1에 꼬리 지표(|err|>10cm 비율)를 추가해 같은 방식으로 숨지 못하게 했다.
Z_OFFSET_M = 0.0


def hmap_translation(start_position_yup):
    """habitat Y-up start_position -> mesh Z-up 카메라-start translation (해석적)."""
    sp = np.asarray(start_position_yup, dtype=np.float64)
    return np.array([sp[0], -sp[2], sp[1] - Z_OFFSET_M])


def hmap_yaw(start_rotation_xyzw):
    """habitat start_rotation(quaternion [x,y,z,w], up=Y 회전) -> mesh Z-up yaw[rad] (해석적).

    검증된 7개 에피소드에서 `yaw_mesh = (2*atan2(qy,qw) + 90°) mod 360°`가 exact로 성립
    (330→60, 270→0, 90→180, 120→210, ...). +90°는 habitat forward→mesh 축 규약 오프셋.
    → sweep 불필요, GT 회전을 직접 쓴다.
    """
    q = np.asarray(start_rotation_xyzw, dtype=np.float64)
    return (2.0 * np.arctan2(q[1], q[3]) + np.pi / 2.0) % (2.0 * np.pi)


def fit_sf2mesh(pts_sf, mesh, start_position_yup, start_rotation_xyzw, seed=0, refine=False,
                surface_samples=120000):
    """start-frame cloud를 mesh 절대좌표에 올리는 강체 변환. -> (T_sf2mesh 4x4, residual_median_m).

    **전부 GT 해석적**: translation=`hmap_translation`(start_position, 다층은 start_y가 층 지정),
    yaw=`hmap_yaw`(start_rotation). **sweep/opt 없음** — GT 회전을 직접 쓴다(검증 6/8<1cm, 나머지 ~1.2cm).
    `refine=True`(옵션)면 (yaw,tx,ty,tz) Nelder-Mead로 다듬으나 KDTree 근사 비용이라 오히려 나빠질 수
    있어 기본 off. 최종 residual은 exact `compute_mesh_distance`.
    """
    from scipy.spatial import cKDTree
    import trimesh
    rng = np.random.RandomState(seed)
    x0 = [hmap_yaw(start_rotation_xyzw), *hmap_translation(start_position_yup)]

    if refine:
        tree = cKDTree(np.asarray(trimesh.sample.sample_surface(mesh, surface_samples)[0]))
        sub = pts_sf[rng.choice(len(pts_sf), min(1200, len(pts_sf)), replace=False)]

        def cost(p):
            T = _rigid(p)
            return float(np.median(tree.query((T[:3, :3] @ sub.T).T + T[:3, 3])[0]))
        x0 = minimize(cost, x0, method='Nelder-Mead',
                      options={'xatol': 1e-3, 'fatol': 1e-4, 'maxiter': 400}).x

    T = _rigid(x0)
    q = (T[:3, :3] @ pts_sf.T).T + T[:3, 3]
    sel = rng.choice(len(q), min(1500, len(q)), replace=False)
    return T, float(np.median(compute_mesh_distance(q[sel], mesh)))


def align_episode(frames, mesh, raw_episode, seed=0, refine=False):
    """편의 래퍼: frames + matched raw episode -> (T_sf2mesh, residual_m, cloud).

    GT `start_position`+`start_rotation`으로 해석적 정합(sweep 없음). refine으로 ~cm 잔차만 다듬음.
    """
    pts = accumulate_startframe_cloud(frames, seed=seed)
    T, resid = fit_sf2mesh(pts, mesh, raw_episode['start_position'], raw_episode['start_rotation'],
                           seed=seed, refine=refine)
    return T, resid, pts
