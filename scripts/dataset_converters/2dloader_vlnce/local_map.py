"""depth 한 장 -> robot-centric planning 격자.

3dloader가 `mesh -> 3D occupancy -> derive_obstacle_2d`로 만들던 2D 장애물 맵을,
여기서는 **학습·평가가 실제로 쓰는 BEV 함수**로 만든다:
`internnav/model/utils/depth_rgb_to_bev_torch.py:319 depth_to_bev_occ_ros2`.
`z_min`/`z_max`가 그대로 `(h_nav, h_b]` slab이므로 embodiment 높이축이 공짜로 들어오고,
관측을 만드는 격자와 GT를 만드는 격자가 같아져 train==eval 불변식과 정보 누수 문제가 동시에 해결된다.

## 좌표계 3개 (전부 오른손잡이 아님 주의 — plan 좌표는 의도적 반사)

| 이름 | 정의 | 쓰는 곳 |
|---|---|---|
| **world** | 에피소드 시작 상대, **z-up**. `pose.<rig>`가 cam2world | pixel goal 투영(`pixel_goal_utils`) |
| **robot** | X=전방, Y=좌, Z=상, 원점 = 카메라 **바로 아래 바닥** | GT path, 장애물 배치, 결과 |
| **plan**  | `x_e = -Y_robot`, `y_e = -X_robot`, origin `(-R,-R)` | `esdf_utils`(A*/ESDF/spline) |

plan 좌표를 이렇게 잡은 이유: BEV 배열 `bev[i,j]`(i = 위=전방, j = 왼쪽=좌)를 **복사 없이 그대로**
`esdf_utils`의 `(Ny,Nx)` / `arr[iy,ix]` 규약에 꽂기 위해서다. `iy=i, ix=j`가 되고
`world_to_cell`/`cell_to_world`가 그대로 맞는다. `robot_to_plan`은 **자기역함수**라 왕복이 안전하다.

self-check: `/usr/bin/python scripts/dataset_converters/2dloader_vlnce/local_map.py`
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CONV = _HERE.parent
_REPO = _CONV.parents[1]
for _p in (str(_REPO), str(_CONV / 'gs_vlnpe'), str(_CONV / '3dloader_vlnce'), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esdf_utils import (  # noqa: E402  gs_vlnpe — 수정 금지, import만
    REFINERS,
    SMOOTHERS,
    astar,
    cell_to_world,
    compute_esdf_2d,
    downsample_navigable,
    thin_waypoints,
    truncate_navigable,
    world_to_cell,
)
from pixel_goal_utils import IMG_H, IMG_W, intrinsics_for_rig, rig_height_m  # noqa: E402  3dloader

BEV_RANGE_M = 5.0         # 학습 기본값 (internvla_n1_bev_provider.py)
BEV_SIZE = 224
DOWNSAMPLE_FACTOR = 4     # A* 격자 = 4.46cm × 4 ≈ 17.9 cm (03_sample_gt_paths와 같은 관례)
UNKNOWN_POLICY = ('nontraversable', 'free', 'block')
ROBOT_FREE_M = 0.50       # 하한. 실제로는 아래 `near_blind_radius`로 rig 기하에서 구한다


def limit_torch_threads(n: int = 1) -> None:
    """torch를 n스레드로 제한. **CPU에서 BEV를 만들기 전에 반드시.**

    ## 원인 (2026-08-17 재측정으로 확정 — 이전 설명은 틀렸다)
    "멀티스레드가 작은 텐서에 오버헤드"가 아니라, **intra-op 스레드 수가 코어 수와 같을 때
    (24 == `os.cpu_count()`)만** 생기는 절벽이다. 같은 입력·같은 함수
    (`depth_to_bev_occ_ros2`, 224², occupied 1665셀):

        threads  24 -> 396 ms      threads 23 -> 1.77 ms
        threads  22 -> 1.76 ms     threads 20 -> 1.66 ms
        threads  16 -> 1.67 ms     threads 12 -> 1.53 ms     threads 1 -> 2.14 ms

    23개만 돼도 정상이고 24에서만 220배 느려진다 — full subscription에서 OpenMP가 spin-wait로
    코어를 서로 뺏는 전형적 증상이다. 1스레드는 안전하면서 가장 빠른 축(1.5~2.4 ms)에 속한다.

    ## 3dloader R2의 "depth→BEV 165 ms/frame"도 같은 원인이다 (직접 재측정)
    `EmbodimentAugmenter.render_bev_along(T=12, with_bev=True)` (s8pcmisQ38h):

        threads 24: depth×12 341 ms | +BEV 2468 ms -> BEV만 2127 ms = 177.3 ms/frame
        threads  1: depth×12 340 ms | +BEV  370 ms -> BEV만   30 ms =   2.5 ms/frame   (71배)

    R2가 이 값을 잰 곳은 DataLoader worker가 아니라 **메인 프로세스**(bench_parallel.py:168-174)라
    torch 기본 24스레드였다. 따라서 "CPU BEV가 병목이므로 `with_bev=False`로 가야 한다"는 R2의
    결론은 근거가 무효다 — 스레드만 제한하면 된다.

    DataLoader worker는 원래 1스레드로 뜨므로 학습 경로에서는 no-op이고 스크립트/메인 프로세스에서만
    효과가 있다. 그래서 라이브러리 import 시점이 아니라 **worker 스코프**(`Augmenter2D.__init__`)와
    각 스크립트의 진입점에서만 호출한다.
    """
    import torch
    if torch.get_num_threads() > n:
        torch.set_num_threads(n)


def near_blind_radius(rig: str, pitch_deg: float, height: int, fy: float,
                      cam_height: float = None, margin: float = 1.15) -> float:
    """이 rig가 **원리적으로 볼 수 없는** 발밑 반경 [m].

    아래로 `pitch + vfov/2`만큼 기운 광선이 바닥에 닿는 거리가 관측 가능한 최소 거리다:
    `r_near = cam_height / tan(pitch + vfov/2)`, `vfov/2 = atan((H/2)/fy)`.
    실측(125cm_30deg, 224px, fy=181): 0.67 m — `ROBOT_FREE_M=0.5`로는 0.5~0.67 m 고리가 unknown으로
    남아 **로봇 바로 앞에서 경로가 미관측 칸을 지난다**(17DRP f0에서 경로의 20%가 이 고리였다).
    로봇이 서 있는 자리이므로 이 반경 안은 free로 본다.
    """
    ch = rig_height_m(rig) if cam_height is None else float(cam_height)
    ang = math.radians(pitch_deg) + math.atan((height / 2.0) / fy)
    if ang >= math.pi / 2 - 1e-6:
        return float('inf')
    return max(ROBOT_FREE_M, margin * ch / math.tan(ang))


def cell_size(bev_range: float = BEV_RANGE_M, bev_size: int = BEV_SIZE) -> float:
    return 2.0 * bev_range / bev_size


def plan_origin(bev_range: float = BEV_RANGE_M) -> np.ndarray:
    return np.array([-bev_range, -bev_range], dtype=np.float64)


# ---------------------------------------------------------------------------
# 좌표 변환
# ---------------------------------------------------------------------------

def robot_to_plan(xy: np.ndarray) -> np.ndarray:
    """robot (X=fwd, Y=left) <-> plan (x_e, y_e). **자기역함수** — 같은 함수로 되돌린다."""
    a = np.asarray(xy, dtype=np.float64)
    return np.stack([-a[..., 1], -a[..., 0]], axis=-1)


plan_to_robot = robot_to_plan


def rot_cam_to_robot(pitch_deg: float) -> np.ndarray:
    """`depth_rgb_to_bev_torch._build_R_c2w`의 numpy 복제 (카메라 x=우,y=하,z=전 -> robot X,Y,Z)."""
    c, s = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    return np.array([[0.0, -s,  c],
                     [-1.0, 0.0, 0.0],
                     [0.0, -c, -s]], dtype=np.float64)


def world_to_robot(pts_w, pose_cam2world, pitch_deg: float, cam_height: float) -> np.ndarray:
    """world 점 -> robot 프레임. `pts_w`는 (...,3)."""
    p = np.atleast_2d(np.asarray(pts_w, dtype=np.float64))
    T = np.asarray(pose_cam2world, dtype=np.float64).reshape(4, 4)
    xc = (np.linalg.inv(T) @ np.concatenate([p, np.ones((len(p), 1))], axis=1).T).T[:, :3]
    out = xc @ rot_cam_to_robot(pitch_deg).T
    out[:, 2] += cam_height
    return out.reshape(np.shape(pts_w))


def robot_to_world(pts_r, pose_cam2world, pitch_deg: float, cam_height: float) -> np.ndarray:
    """robot 프레임 -> world. `world_to_robot`의 정확한 역."""
    p = np.atleast_2d(np.asarray(pts_r, dtype=np.float64)).copy()
    p[:, 2] -= cam_height
    xc = p @ rot_cam_to_robot(pitch_deg)          # R^T @ p  == p @ R
    T = np.asarray(pose_cam2world, dtype=np.float64).reshape(4, 4)
    out = (T @ np.concatenate([xc, np.ones((len(xc), 1))], axis=1).T).T[:, :3]
    return out.reshape(np.shape(pts_r))


def gravity_alignment_error(pose_cam2world, pitch_deg: float) -> float:
    """world->robot 회전이 **world z축 둘레의 순수 yaw**인지. 0에 가까워야 한다.

    rig의 실제 pitch/roll이 `pitch_deg`와 다르면 BEV의 중력 정렬이 깨지고 slab(`h_nav`,`h_b`)이
    의미를 잃는다. W0의 첫 관문.
    """
    R_wc = np.asarray(pose_cam2world, dtype=np.float64)[:3, :3]
    M = rot_cam_to_robot(pitch_deg) @ R_wc.T      # world -> robot
    return float(max(np.abs(M[2, :] - [0, 0, 1]).max(), np.abs(M[:, 2] - [0, 0, 1]).max()))


def bev_ij_to_plan_xy(ij, bev_range: float = BEV_RANGE_M, bev_size: int = BEV_SIZE) -> np.ndarray:
    """BEV 픽셀 (i=row, j=col) -> plan (x_e, y_e) 셀 중심. `ij`는 (...,2) = (i, j)."""
    a = np.atleast_2d(np.asarray(ij))
    return cell_to_world(np.stack([a[:, 1], a[:, 0]], axis=-1),   # (ix, iy) = (j, i)
                         plan_origin(bev_range), cell_size(bev_range, bev_size)).reshape(np.shape(ij))


def plan_xy_to_bev_ij(xy, bev_range: float = BEV_RANGE_M, bev_size: int = BEV_SIZE) -> np.ndarray:
    """plan (x_e, y_e) -> BEV 픽셀 (i, j). 격자 밖도 그대로 반환하니 호출부가 검사할 것."""
    a = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    ixiy = world_to_cell(a, plan_origin(bev_range), cell_size(bev_range, bev_size))
    return np.stack([ixiy[:, 1], ixiy[:, 0]], axis=-1).reshape(np.shape(xy))


def robot_xy_to_bev_ij(xy, bev_range: float = BEV_RANGE_M, bev_size: int = BEV_SIZE) -> np.ndarray:
    return plan_xy_to_bev_ij(robot_to_plan(xy), bev_range, bev_size)


# ---------------------------------------------------------------------------
# BEV / local map
# ---------------------------------------------------------------------------

def intrinsics_for_res(rig: str, width: int, height: int):
    """rig intrinsics(640×480 기준)를 실제 입력 해상도로 스케일.

    `BEVProcessor._scaled_intrinsics`(visual_input_provider.py:184)와 같은 규칙.
    """
    fx, fy, cx, cy = intrinsics_for_rig(rig)
    return fx * width / IMG_W, fy * height / IMG_H, cx * width / IMG_W, cy * height / IMG_H


def build_bev(depth_m: np.ndarray, rig: str, pitch_deg: float, h_nav: float, h_b: float,
              bev_range: float = BEV_RANGE_M, bev_size: int = BEV_SIZE,
              cam_height: float = None) -> np.ndarray:
    """metric depth [H,W] -> 3-state BEV [S,S] (0=unknown, 0.5=free, 1=occupied).

    `depth_to_bev_occ_ros2`를 그대로 호출한다 — 학습(`internvla_n1_bev_provider`)·
    habitat eval(`habitat_vln_evaluator_unified`)과 **동일한 함수**.
    """
    import torch

    from internnav.model.utils.depth_rgb_to_bev_torch import depth_to_bev_occ_ros2

    assert h_b > h_nav, f'h_b({h_b})는 h_nav({h_nav})보다 커야 한다'
    H, W = depth_m.shape[-2:]
    fx, fy, cx, cy = intrinsics_for_res(rig, W, H)
    ch = rig_height_m(rig) if cam_height is None else float(cam_height)
    bev = depth_to_bev_occ_ros2(
        torch.from_numpy(np.ascontiguousarray(depth_m, dtype=np.float32))[None],
        cam_height=ch, cam_pitch_deg=float(pitch_deg),
        fx=fx, fy=fy, cx=cx, cy=cy,
        bev_range=bev_range, bev_size=bev_size, depth_scale=1.0,
        z_min=float(h_nav), z_max=float(h_b),
    )
    return bev[0].numpy()


def carve_free_from_floor(depth_m: np.ndarray, rig: str, pitch_deg: float, h_nav: float,
                          bev_range: float = BEV_RANGE_M, bev_size: int = BEV_SIZE,
                          cam_height: float = None, max_range_m: float = None) -> np.ndarray:
    """**같은 depth 한 장**에서 바닥 반환으로 free 공간을 판다. -> [S,S] bool.

    왜 필요한가 (실측으로 규명): `depth_to_bev_occ_ros2`의 raycast는 **점유 셀로 향하는 광선** 위에만
    free를 찍는다. 점유는 높이 밴드 `(h_nav, h_b]`에 반환이 있어야 생기므로, 그 방향에 밴드 안 물체가
    없으면(=바닥만 보이면) 광선 자체가 없어 **화면에 뻔히 보이는 바닥이 unknown으로 남는다**.
    s8pcmisQ38h ep0 f11/f14가 그 경우다 — pixel goal이 이미지 안(222,182)에 가려지지도 않고
    Z=3.34 m로 보이는데 BEV 셀은 unknown이었다.

    바닥 반환(`z <= h_nav`)은 "그 방향 그 거리까지 통행 가능"의 직접 증거다. 같은 depth 이미지 안의
    정보이므로 정보 누수가 아니다. **관측 BEV(`m['bev']`)는 건드리지 않고** planning용 free/observed만
    보강한다 — 학습이 보는 관측은 `depth_to_bev_occ_ros2` 출력 그대로여야 하기 때문이다.
    """
    import torch

    from internnav.model.utils.depth_rgb_to_bev_torch import _raycast_free_torch, unproject_depth

    H, W = depth_m.shape[-2:]
    fx, fy, cx, cy = intrinsics_for_res(rig, W, H)
    ch = rig_height_m(rig) if cam_height is None else float(cam_height)
    d = torch.from_numpy(np.ascontiguousarray(depth_m, dtype=np.float32))[None]
    xyz = unproject_depth(d, cam_height=ch, cam_pitch_deg=float(pitch_deg),
                          fx=fx, fy=fy, cx=cx, cy=cy, depth_scale=1.0)[0]
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    rng = max_range_m if max_range_m is not None else bev_range
    mask = (Z <= h_nav) & (d[0] > 0.1) & (d[0] < rng)     # 바닥/문턱 높이 반환만

    scale = bev_size / (2.0 * bev_range)
    i = torch.floor(bev_size / 2.0 - X * scale).long()
    j = torch.floor(bev_size / 2.0 - Y * scale).long()
    mask &= (i >= 0) & (i < bev_size) & (j >= 0) & (j < bev_size)
    ends = torch.zeros(bev_size, bev_size, dtype=torch.bool)
    ends[i[mask], j[mask]] = True
    if not bool(ends.any()):
        return np.zeros((bev_size, bev_size), bool)
    # 광선 위 중간 셀(0.5) + **끝점 자신(1.0)도 free** — 바닥은 통행 가능하므로 장애물이 아니다
    return (_raycast_free_torch(ends) > 0).numpy()


def build_local_map(depth_m: np.ndarray, rig: str, pitch_deg: float,
                    r_b: float, h_nav: float, h_b: float,
                    bev_range: float = BEV_RANGE_M, bev_size: int = BEV_SIZE,
                    unknown: str = 'nontraversable', cam_height: float = None,
                    bev: np.ndarray = None, robot_free_m: float = None,
                    free_from_floor: bool = True) -> dict:
    """depth(또는 이미 만든 BEV) -> planning에 바로 쓸 수 있는 격자 묶음.

    미관측(unknown) 셀에는 **두 역할이 있고 서로 다르게 다뤄야 한다** — 실측으로 확인한 결론:

    | 정책 | ESDF(장애물) | navigable | 결과 |
    |---|---|---|---|
    | `block` | occ ∪ unk | esdf ≥ r_b | 시야각 원뿔의 **측면 경계가 장애물**이 되어 원본 GT clearance가 0.00~0.10 m로 붕괴 → 어떤 r_b에서도 계획 불가 |
    | `free` | occ | esdf ≥ r_b | clearance는 정상(0.24~0.36 m)이지만 **A*가 미관측 영역을 가로질러** 우회한다(실측: r_b 0.35/0.50에서 벽 뒤로 우회) — GT로 못 쓴다 |
    | **`nontraversable`** (기본) | occ | (esdf ≥ r_b) ∧ free | clearance는 실제 물체까지, 경로는 **관측된 free 공간 안**에서만. 둘 다 만족 |

    `robot_free_m`: 발밑 blind 반경. None이면 `near_blind_radius`로 **rig 기하에서 계산**한다
    (상수로 두면 카메라가 볼 수 없는 고리가 unknown으로 남아 경로가 그 위를 지난다).

    `free_from_floor`: **바닥 반환으로 free를 추가로 판다**(`carve_free_from_floor` 참고).
    `m['bev']`(관측, 학습과 동일)는 그대로 두고 `m['free']`/`m['observed']`만 보강한다.
    """
    assert unknown in UNKNOWN_POLICY, f'unknown={unknown!r} not in {UNKNOWN_POLICY}'
    if bev is None:
        bev = build_bev(depth_m, rig, pitch_deg, h_nav, h_b, bev_range, bev_size, cam_height)
    cell = cell_size(bev_range, bev_size)
    if robot_free_m is None:
        H = bev_size if depth_m is None else depth_m.shape[-2]
        robot_free_m = near_blind_radius(rig, pitch_deg, H,
                                         intrinsics_for_res(rig, H, H)[1], cam_height)
    occ = bev >= 0.75
    unk = bev < 0.25
    if robot_free_m > 0:
        ii, jj = np.ogrid[:bev_size, :bev_size]
        near = ((ii - bev_size / 2.0) ** 2 + (jj - bev_size / 2.0) ** 2) <= (robot_free_m / cell) ** 2
        unk = unk & ~near
    if free_from_floor and depth_m is not None:
        unk = unk & ~carve_free_from_floor(depth_m, rig, pitch_deg, h_nav, bev_range, bev_size,
                                           cam_height)
    free = ~occ & ~unk

    if unknown == 'block':
        obstacle = occ | unk
        navigable = truncate_navigable(compute_esdf_2d(obstacle, cell), r_b)
    elif unknown == 'free':
        obstacle = occ
        navigable = truncate_navigable(compute_esdf_2d(obstacle, cell), r_b)
    elif unknown == 'nontraversable':
        obstacle = occ
        navigable = truncate_navigable(compute_esdf_2d(obstacle, cell), r_b) & free
    else:
        assert False, f'unreachable unknown={unknown!r}'
    esdf = compute_esdf_2d(obstacle, cell)
    # goal 조정용 ESDF: 통행 불가 셀(미관측)은 clearance 0으로 눌러 `adjust_goal`이 거기로 못 가게 한다.
    # `pixel_goal_utils.adjust_goal`은 `clearance >= r_b`만 보므로, 마스킹을 esdf 쪽에 넣는 게
    # 그 함수를 수정하지 않고 통행성을 반영하는 유일한 방법이다.
    esdf_nav = (np.where(free, esdf, 0.0).astype(np.float32)
                if unknown == 'nontraversable' else esdf)
    return {
        'bev': bev, 'obstacle': obstacle, 'observed': ~unk, 'occupied': occ, 'free': free,
        'esdf': esdf, 'esdf_nav': esdf_nav, 'navigable': navigable,
        'origin': plan_origin(bev_range), 'cell_m': cell,
        'bev_range': bev_range, 'bev_size': bev_size, 'r_b': r_b, 'unknown_policy': unknown,
    }


def path_violations(m: dict, traj: np.ndarray) -> dict:
    """경로가 격자 규약을 지키는지. -> {'occ': 점유칸 비율, 'unfree': 관측free 아닌 칸 비율, 'n': 표본수}

    **왜 필요한가**: A*가 통과한 뒤에도 refine/spline이 경로를 옮기고, coarse 격자는 fine 격자보다
    낙관적이라 최종 경로가 점유·미관측 칸을 지날 수 있다. 실측(수정 전, 17DRP ep0 f23):
    점유칸 18%, 미관측칸 39%. 그래서 **계획 결과를 fine 격자에서 반드시 다시 검사**한다.
    """
    S = m['bev_size']
    ij = plan_xy_to_bev_ij(robot_to_plan(np.atleast_2d(traj)), m['bev_range'], S)
    inb = (ij[:, 0] >= 0) & (ij[:, 0] < S) & (ij[:, 1] >= 0) & (ij[:, 1] < S)
    if not inb.any():
        return {'occ': 1.0, 'unfree': 1.0, 'n': 0}
    i, j = ij[inb, 0], ij[inb, 1]
    return {'occ': float(m['occupied'][i, j].mean()),
            'unfree': float((~m['free'][i, j]).mean()), 'n': int(inb.sum())}


def plan_path(m: dict, goal_xy_robot, downsample_factor: int = DOWNSAMPLE_FACTOR,
              refine_radius_m: float = 0.30, spacing_m: float = 0.4,
              refine_mode: str = 'argmax', smooth_mode: str = 'bezier',
              downsample_mode: str = 'any', max_unfree: float = 0.02) -> dict:
    """robot 원점 -> `goal_xy_robot` 경로. 반환 `trajectory`는 **robot XY**.

    체인은 `03_sample_gt_paths.plan_episode`와 동일: A*(coarse) -> refine -> thin -> smooth.
    refine/smooth는 `esdf_utils`의 `REFINERS`/`SMOOTHERS` **레지스트리**로 고른다(하드코딩 금지 규약).

    다만 2D(부분 관측)에서는 03의 기본값을 그대로 쓰면 **최종 경로가 장애물을 통과한다**.
    실측(두 씬 × 6프레임 × r_b 4값 = 43조합, 계획 성공분 중 위반 수 / 경로가 지나는 점유칸 최대비율):

        smooth  spacing  성공  위반  최대 점유칸
        cubic     0.8    36     4      18%     <- 03 기본값. 경로가 벽을 뚫는다
        bezier    0.8    36     3       3%
        cubic     0.4    36     3       0%
        bezier    0.4    36     3       0%     <- 채택
        (downsample 'majority'는 위반 0이지만 성공이 26/43으로 급감 — 관측 부채꼴이 좁아서다)

    원인은 downsample이 아니다(격자를 4.46 cm까지 낮춰도 위반이 남았다). **`thin_waypoints`가
    0.8 m 간격으로 솎아낸 뒤 spline이 코너를 가로지르는 것**이 원인이라, 간격을 0.4 m로 줄이고
    볼록껍질을 벗어나지 않는 `bezier`를 쓴다.

    그 밖에 03과 다른 점:
    - refine을 `esdf_nav`(미관측=0)로 한다 — 원래 `esdf`로 하면 미관측 영역의 clearance가 커서
      `greedy_refine`이 waypoint를 **미관측 쪽으로 밀어낸다**.
    - 마지막에 `path_violations`로 **fine 격자에서 재검사**하고, 점유칸을 하나라도 지나거나
      미관측칸 비율이 `max_unfree`를 넘으면 실패로 돌린다(→ 상위에서 fallback/기각).

    status: 'ok' | 'start_not_navigable' | 'goal_not_navigable' | 'astar_failed' | 'path_leaves_free'
    """
    cell, origin = m['cell_m'], m['origin']
    coarse_cell = cell * downsample_factor
    # coarse 격자는 **두 조건을 다르게** 축약한다:
    #  - navigable(=여유 ≥ r_b)은 `any` — 문틈 같은 좁은 통로를 살린다(03의 이유와 동일)
    #  - free(=관측됨)는 `all`   — 4×4 중 하나라도 미관측이면 그 coarse 칸을 쓰지 않는다
    # 실측(두 씬×6프레임×r_b 4값): free 조건 없음 위반 2·최대 unfree 17% / `majority` 1·15% /
    # **`all` 0·2%**, 도달률은 셋 다 29/48로 같다. A*의 clearance tie-break(가중치 0~2)는 효과 없었다.
    nav_coarse = (downsample_navigable(m['navigable'], downsample_factor, downsample_mode)
                  & downsample_navigable(m['free'], downsample_factor, 'all'))
    sg = np.stack([robot_to_plan(np.zeros(2)), robot_to_plan(np.asarray(goal_xy_robot, dtype=np.float64))])
    ij = world_to_cell(sg, origin, coarse_cell)
    h, w = nav_coarse.shape
    ij[:, 0] = np.clip(ij[:, 0], 0, w - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, h - 1)
    out = {'status': 'ok', 'nav_coarse': nav_coarse, 'trajectory': None}
    if not nav_coarse[ij[0, 1], ij[0, 0]]:
        out['status'] = 'start_not_navigable'; return out
    if not nav_coarse[ij[1, 1], ij[1, 0]]:
        out['status'] = 'goal_not_navigable'; return out
    path_ij = astar(nav_coarse, ij[0], ij[1])
    if path_ij is None:
        out['status'] = 'astar_failed'; return out
    wp = cell_to_world(path_ij, origin, coarse_cell)
    wp[0], wp[-1] = sg[0], sg[1]
    kw = {'fix_endpoints': True}
    if refine_mode == 'min_move':
        kw['r_b'] = m['r_b']          # 03의 호출부는 이걸 안 넘겨 기본 0.25가 쓰인다 — 여기선 명시
    # 미관측 쪽으로 밀리지 않도록 **통행 가능 영역으로 마스킹된 ESDF**를 쓴다
    wp = REFINERS[refine_mode](wp, m['esdf_nav'], origin, cell, refine_radius_m, **kw)
    wp = thin_waypoints(wp, spacing_m)
    traj_plan = SMOOTHERS[smooth_mode](wp, cell)
    traj = plan_to_robot(traj_plan)
    v = path_violations(m, traj)
    out['violations'] = v
    if v['occ'] > 0.0 or v['unfree'] > max_unfree:
        out['status'] = 'path_leaves_free'
        return out
    out['trajectory'] = traj
    out['waypoints'] = plan_to_robot(wp)
    return out


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    limit_torch_threads()
    R, S = BEV_RANGE_M, BEV_SIZE
    cell = cell_size(R, S)

    # 1) robot<->plan 자기역함수
    xy = np.array([[1.0, 2.0], [-0.5, 3.25], [0.0, 0.0]])
    assert np.allclose(robot_to_plan(robot_to_plan(xy)), xy), 'robot_to_plan이 자기역함수가 아니다'

    # 2) BEV 인덱스 <-> robot XY 왕복 (셀 중심 기준, 오차 <= cell/2)
    ij = np.array([[0, 0], [S // 2, S // 2], [10, 200], [S - 1, S - 1]])
    rxy = plan_to_robot(bev_ij_to_plan_xy(ij, R, S))
    back = robot_xy_to_bev_ij(rxy, R, S)
    assert np.array_equal(back, ij), f'BEV 왕복 실패 {ij.tolist()} -> {back.tolist()}'

    # 3) 알려진 지점 3개: 전방 = 위(row 작아짐), 좌 = 왼쪽(col 작아짐)
    c = robot_xy_to_bev_ij(np.array([[0.0, 0.0]]), R, S)[0]
    fwd = robot_xy_to_bev_ij(np.array([[2.0, 0.0]]), R, S)[0]
    left = robot_xy_to_bev_ij(np.array([[0.0, 2.0]]), R, S)[0]
    assert tuple(c) == (S // 2, S // 2), c
    assert fwd[0] < c[0] and fwd[1] == c[1], f'전방이 위로 안 간다: {fwd} vs {c}'
    assert left[1] < c[1] and left[0] == c[0], f'좌가 왼쪽으로 안 간다: {left} vs {c}'
    assert abs((c[0] - fwd[0]) * cell - 2.0) < cell, '전방 2 m 스케일 불일치'

    # 4) world<->robot 왕복 + 중력 정렬 (실제 pose로)
    from episode_io import load_episode
    import os
    DR = os.environ.get('VLNCE_ROOT', 'data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ep = load_episode(DR, '17DRP5sb8fy', 0, '125cm_0_30', n_frames=3)
    ch = rig_height_m(ep.rig_ld)
    errs, gerrs = [], []
    for f in ep.frames:
        gerrs.append(gravity_alignment_error(f.pose_ld, 30.0))
        pw = np.array([[1.0, 2.0, 0.3], [-2.0, 0.5, 1.1]])
        rt = robot_to_world(world_to_robot(pw, f.pose_ld, 30.0, ch), f.pose_ld, 30.0, ch)
        errs.append(np.abs(rt - pw).max())
        # 카메라 자신은 robot 원점 위 cam_height에 있어야 한다
        cam_r = world_to_robot(f.pose_ld[:3, 3][None], f.pose_ld, 30.0, ch)[0]
        assert np.allclose(cam_r, [0, 0, ch], atol=1e-9), f'카메라가 robot 원점 위가 아니다: {cam_r}'
    # 중력정렬 허용오차 1e-3: parquet의 pose 회전이 소수 4자리로 **양자화**되어 있어
    # 그 자체의 직교성 오차가 이미 4e-5~8e-5다 (det 0.99996). 5 m 거리에서 0.5 mm 수준.
    rot_err = max(np.abs(np.asarray(f.pose_ld)[:3, :3] @ np.asarray(f.pose_ld)[:3, :3].T
                         - np.eye(3)).max() for f in ep.frames)
    print(f'[map] world<->robot 왕복 오차 max={max(errs):.2e} m, 중력정렬 오차 max={max(gerrs):.2e} '
          f'(pose 자체 직교성 오차 {rot_err:.2e})')
    assert max(errs) < 1e-9 and max(gerrs) < 1e-3, '좌표 변환/중력 정렬 실패'

    # 5) BEV 생성 + local map + 전방 경로 계획
    from episode_io import to_224_depth
    d = to_224_depth(ep.frames[0].depth_m)
    m = build_local_map(d, ep.rig_ld, 30.0, r_b=0.10, h_nav=0.15, h_b=1.25)
    frac = {k: float(v.mean()) for k, v in [('occ', m['occupied']), ('obs', m['observed'])]}
    assert m['bev'].shape == (S, S) and set(np.unique(m['bev'])).issubset({0.0, 0.5, 1.0})
    print(f"[map] BEV occ={100*frac['occ']:.1f}% observed={100*frac['obs']:.1f}% "
          f"esdf[{m['esdf'][np.isfinite(m['esdf'])].min():.2f},"
          f"{m['esdf'][np.isfinite(m['esdf'])].max():.2f}]m")
    # observed가 10%대인 것은 정상: HFOV 79°·5 m clip의 부채꼴 면적이 ±5 m 정사각형의 ~17%다.
    assert 0.0 < frac['occ'] < 0.5 and frac['obs'] > 0.05, frac
    print('[map] PASS')
