"""합성 장애물 — robot frame에 3D 입체를 놓고 RGB/depth에 **동시에** 합성한다.

## 정합이 구조적으로 보장되는 이유
장애물은 robot frame의 3D 물체 하나로만 정의된다. 거기서
- **depth**: ray–입체 교차의 t(=카메라 Z-depth)로 `min(원본, 박스)` z-buffer 합성
- **RGB**: 같은 마스크에 면 법선 기준 flat Lambert 단색
- **BEV**: 따로 그리지 않고 **합성된 depth에서 다시 계산**(`local_map.build_bev`)
이 나오므로 셋이 어긋날 여지가 없다. rig가 둘(FPV pitch_1 / 룩다운 pitch_2)이어도 같은 3D 물체를
각자의 `Cam`으로 투영하므로 자동으로 맞는다.

## 확장 seam (v1은 flat-shaded box 하나)
`render_obstacle`의 `kind` 분기 **한 곳**만 늘리면 된다. 하류(`composite`, BEV 재계산, planning,
pixel goal, 검증 스크립트)는 kind를 전혀 모른다.
- 텍스처 -> `_render_box`의 색 계산만 교체 (depth/BEV/경로 무변경)
- 복잡한 형태 / dataset 유사 object -> `kind='mesh'` 추가 + 별도 파일(`obstacle_mesh.py`)에 래스터라이저,
  `footprint_xy`는 밑면 볼록껍질. 두 번째 kind가 생기는 시점에 dict registry로 승격한다(지금은 YAGNI).

self-check: `/usr/bin/python scripts/dataset_converters/2dloader_vlnce/obstacle_synth.py`
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent / '3dloader_vlnce'), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from local_map import rot_cam_to_robot  # noqa: E402

LIGHT_DIR = np.array([0.30, 0.50, 1.00])          # robot frame, 위-앞-왼쪽에서 비추는 고정 광원
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
AMBIENT = 0.40


class Cam(NamedTuple):
    """robot frame 기준 카메라 하나. 두 rig는 **중심이 같고 pitch만 다르다**(실측 확인)."""
    pitch_deg: float
    height: float
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height_px: int


@dataclass
class Obstacle:
    """robot frame(X=전방, Y=좌, Z=상), **바닥 접지**. v1은 `kind='box'`만."""
    center_xy: np.ndarray                  # (2,) 밑면 중심
    size: np.ndarray                       # (3,) 전체 크기 (w=X'방향, d=Y'방향, h=Z)
    yaw: float = 0.0                       # rad, Z축 회전
    kind: str = 'box'
    color: Tuple[float, float, float] = (0.55, 0.50, 0.45)
    payload: object = None                 # mesh/asset용 자리

    @property
    def center3(self) -> np.ndarray:
        return np.array([self.center_xy[0], self.center_xy[1], self.size[2] / 2.0])

    def rot(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class ObstacleCfg:
    """샘플링 파라미터 — 전부 config로 조절 가능(하드코딩 금지)."""
    path_frac: Tuple[float, float] = (0.25, 0.70)   # GT 경로 위 어디에 놓을지 (호길이 비율)
    lateral_m: float = 0.25                          # 경로에서 옆으로 흔드는 폭
    size_w: Tuple[float, float] = (0.30, 0.80)
    size_d: Tuple[float, float] = (0.30, 0.80)
    size_h: Tuple[float, float] = (0.40, 1.20)
    # 로봇에서의 거리. 하한이 너무 작으면(0.8 m) 박스가 화면을 가득 채우고 BEV에는 몇 셀만 남는다
    # (실측 f0: 화면 9%인데 신규 점유 42셀). 1.0 m부터가 "앞을 막는" 배치가 된다.
    range_m: Tuple[float, float] = (1.00, 3.50)
    max_tries: int = 40
    # **통과 가능한 틈을 남기는 배치**. None이면 경로 위에 그냥 놓는다(항상 막힘).
    # 값은 남길 통로의 **clearance**(= 통로 폭의 절반) 범위다 — `r_b < g`인 로봇만 지나갈 수 있으므로
    # sweep하는 r_b 범위(0.10~0.50)를 걸치도록 잡는다. 그래야 같은 장면이 embodiment에 따라
    # 통과/우회로 갈려 paired counterfactual이 선명해진다.
    gap_m: Optional[Tuple[float, float]] = (0.08, 0.55)


# ---------------------------------------------------------------------------
# 렌더 (kind 분기 = 확장 seam)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _ray_grid_robot(cam: Cam) -> Tuple[np.ndarray, np.ndarray]:
    """카메라 픽셀별 ray. -> (origin_robot(3,), dir_robot[H,W,3]).

    `dir`의 카메라 z성분이 1이므로 **ray 파라미터 t가 곧 Z-depth**다
    (`depth_rgb_to_bev_torch._build_ray_grid`와 같은 규약).

    카메라마다 한 번만 만들면 되므로 캐시한다(`Cam`은 NamedTuple이라 해시 가능). 640×480 격자를
    매 프레임 다시 만들면 합성 시간의 절반을 여기서 쓴다.
    """
    v, u = np.mgrid[0:cam.height_px, 0:cam.width].astype(np.float64)
    d_cam = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], axis=-1)
    d_rob = d_cam @ rot_cam_to_robot(cam.pitch_deg).T
    return np.array([0.0, 0.0, cam.height]), d_rob


def _silhouette_window(obs: Obstacle, cam: Cam):
    """박스가 이미지에서 차지할 수 있는 최소 사각창 (v0, v1, u0, u1).

    볼록체의 실루엣은 꼭짓점 투영의 볼록껍질 안에 들어간다. 꼭짓점 하나라도 카메라 뒤면 투영이
    발산하므로 전체 화면으로 물러선다(보수적). 이 창 밖은 계산하지 않는다 — 640×480 전체를 돌면
    합성이 프레임당 40 ms인데, 창만 돌면 한 자릿수 ms가 된다.
    """
    uvz = project(corners_xyz(obs), cam)
    if not np.isfinite(uvz[:, :2]).all() or (uvz[:, 2] <= 1e-6).any():
        return 0, cam.height_px, 0, cam.width
    u0 = max(int(np.floor(uvz[:, 0].min())) - 1, 0)
    u1 = min(int(np.ceil(uvz[:, 0].max())) + 2, cam.width)
    v0 = max(int(np.floor(uvz[:, 1].min())) - 1, 0)
    v1 = min(int(np.ceil(uvz[:, 1].max())) + 2, cam.height_px)
    return v0, max(v1, v0), u0, max(u1, u0)


def _render_box(obs: Obstacle, cam: Cam):
    """OBB slab 교차. -> (t_near[H,W], hit[H,W] bool, normal[H,W,3] robot frame)."""
    o_r, d_r = _ray_grid_robot(cam)
    H, W = cam.height_px, cam.width
    t_near = np.full((H, W), np.inf)
    hit = np.zeros((H, W), bool)
    normal = np.zeros((H, W, 3))
    v0, v1, u0, u1 = _silhouette_window(obs, cam)
    if v1 <= v0 or u1 <= u0:
        return t_near, hit, normal

    R = obs.rot()
    o = R.T @ (o_r - obs.center3)                       # 박스 로컬 좌표계의 ray 원점
    d = d_r[v0:v1, u0:u1] @ R                           # (R.T @ d) == d @ R
    h = obs.size / 2.0

    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / d
        t1 = (-h - o) * inv
        t2 = (h - o) * inv
    lo = np.minimum(t1, t2)
    hi = np.maximum(t1, t2)
    lo = np.where(np.isnan(lo), -np.inf, lo)
    hi = np.where(np.isnan(hi), np.inf, hi)
    tn = lo.max(axis=-1)
    tf = hi.min(axis=-1)
    hw = (tf >= np.maximum(tn, 0.0)) & (tn > 0.0)

    axis = np.argmax(lo, axis=-1)                       # tmin을 준 축 = 부딪힌 면
    sign = -np.sign(np.take_along_axis(d, axis[..., None], -1)[..., 0])
    n_local = np.zeros(d.shape)
    np.put_along_axis(n_local, axis[..., None], sign[..., None], -1)

    t_near[v0:v1, u0:u1] = np.where(hw, tn, np.inf)
    hit[v0:v1, u0:u1] = hw
    normal[v0:v1, u0:u1] = n_local @ R.T
    return t_near, hit, normal


def render_obstacle(obs: Obstacle, cam: Cam):
    """장애물 -> (t_near, hit, normal). **여기가 유일한 kind 분기점.**"""
    if obs.kind == 'box':
        return _render_box(obs, cam)
    else:
        assert False, f'unreachable kind={obs.kind!r}'


def shade(obs: Obstacle, normal: np.ndarray) -> np.ndarray:
    """면 법선 -> flat Lambert 단색 uint8 [H,W,3]. (텍스처를 넣을 때 교체할 지점)"""
    lam = np.clip(normal @ LIGHT_DIR, 0.0, 1.0)
    s = (AMBIENT + (1.0 - AMBIENT) * lam)[..., None]
    return np.clip(np.array(obs.color) * s * 255.0, 0, 255).astype(np.uint8)


def footprint_xy(obs: Obstacle) -> np.ndarray:
    """밑면 다각형 (robot XY). W2 정합 검증의 기준."""
    if obs.kind == 'box':
        w, d = obs.size[0] / 2.0, obs.size[1] / 2.0
        corners = np.array([[-w, -d], [w, -d], [w, d], [-w, d]])
        return corners @ obs.rot()[:2, :2].T + obs.center_xy
    else:
        assert False, f'unreachable kind={obs.kind!r}'


def corners_xyz(obs: Obstacle) -> np.ndarray:
    """8꼭짓점 (robot frame). 교차검증(실루엣 convex hull)용."""
    h = obs.size / 2.0
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
    return s @ obs.rot().T + obs.center3


def project(pts_robot: np.ndarray, cam: Cam) -> np.ndarray:
    """robot 점 -> 픽셀 (u, v, Z). Z<=0(뒤)이면 u,v는 nan."""
    p = np.atleast_2d(np.asarray(pts_robot, dtype=np.float64)).copy()
    p[:, 2] -= cam.height
    xc = p @ rot_cam_to_robot(cam.pitch_deg)            # R.T @ p
    Z = xc[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(Z > 1e-6, cam.cx + cam.fx * xc[:, 0] / Z, np.nan)
        v = np.where(Z > 1e-6, cam.cy + cam.fy * xc[:, 1] / Z, np.nan)
    return np.stack([u, v, Z], axis=-1)


# ---------------------------------------------------------------------------
# 합성
# ---------------------------------------------------------------------------

def composite(rgb: np.ndarray, depth_m: Optional[np.ndarray], obs: Obstacle, cam: Cam):
    """RGB(+있으면 depth)에 장애물을 합성한다. -> (rgb2, depth2 or None, mask).

    `depth_m`이 None이면(= RGB만 있는 rig) **원본 depth로 가림 판정을 못 하므로** 박스를 그대로 덮는다.
    두 rig가 카메라 중심을 공유하므로, 룩다운 depth로 판정한 가림과 실질적으로 같다
    (시야각 차이만큼만 다르다 — W2의 교차검증 지표로 확인한다).
    """
    t, hit, normal = render_obstacle(obs, cam)
    if depth_m is None:
        mask = hit
    else:
        # depth 0 = 무효(반환 없음)이므로 그 픽셀은 박스가 이긴다
        mask = hit & ((depth_m <= 0.05) | (t < depth_m))
    rgb2 = rgb.copy()
    rgb2[mask] = shade(obs, normal[mask])        # 마스크된 픽셀만 음영 계산
    depth2 = None
    if depth_m is not None:
        depth2 = depth_m.copy()
        depth2[mask] = t[mask].astype(depth_m.dtype)
    return rgb2, depth2, mask


# ---------------------------------------------------------------------------
# 샘플링
# ---------------------------------------------------------------------------

def sample_obstacle(gt_xy_robot: np.ndarray, rng: np.random.Generator, cam: Cam,
                    cfg: ObstacleCfg = None, m: dict = None) -> Optional[Obstacle]:
    """GT 경로 위에 장애물을 하나 놓는다.

    `cfg.gap_m`과 `m`(장애물 넣기 전 local map)이 둘 다 주어지면 **통과 가능한 틈을 남긴다**:
    경로점 p의 여유를 `c0 = esdf(p)`라 하면 자유 공간은 법선 n 방향으로 대략 `[-c0, +c0]`이다.
    박스(반폭 `hw`)를 `d = c0 - hw - 2g`만큼 밀면 +n 쪽에 **폭 2g**의 통로가 남고 그 한가운데의
    clearance가 `g`가 된다. 따라서 `r_b < g`인 로봇만 지나갈 수 있다.

    화면 안 + 전방 + 거리 범위를 만족할 때까지 재시도. 실패하면 None.
    """
    cfg = cfg or ObstacleCfg()
    p = np.asarray(gt_xy_robot, dtype=np.float64)[:, :2]
    if len(p) < 2:
        return None
    rr = np.linalg.norm(p, axis=1)

    # 거리 조건을 만족하는 경로 점들. 경로가 짧아 하나도 없으면 **가장 먼 점**으로 물러선다
    # (에피소드 끝 근처 프레임은 남은 경로가 1 m 미만이라 그냥 버리면 프레임을 통째로 잃는다).
    lo_i = int(cfg.path_frac[0] * (len(p) - 1))
    cand = np.where((rr >= cfg.range_m[0]) & (rr <= cfg.range_m[1]))[0]
    cand = cand[cand >= lo_i]
    if len(cand) == 0:
        cand = np.array([int(np.argmax(rr))])

    for _ in range(cfg.max_tries):
        k = int(rng.choice(cand))
        c = p[k].copy()
        tan = p[min(k + 1, len(p) - 1)] - p[max(k - 1, 0)]
        n = float(np.linalg.norm(tan))
        size = np.array([rng.uniform(*cfg.size_w), rng.uniform(*cfg.size_d),
                         rng.uniform(*cfg.size_h)])
        yaw = float(rng.uniform(0, np.pi))
        if n > 1e-6:
            nrm = np.array([-tan[1], tan[0]]) / n
            if cfg.gap_m is not None and m is not None:
                # 통로 폭 g가 남도록 옆으로 민다 (위 docstring의 d = c0 - hw - g)
                from local_map import robot_to_plan as _rp
                from esdf_utils import sample_esdf_at as _cl
                c0 = float(_cl(m['esdf'], _rp(c[None]), m['origin'], m['cell_m'])[0])
                g = float(rng.uniform(*cfg.gap_m))
                hw = float(np.hypot(size[0], size[1]) / 2.0)      # 회전 무관 보수적 반폭
                d = c0 - hw - 2.0 * g
                if not np.isfinite(d) or d < -cfg.lateral_m:
                    continue                                       # 통로를 낼 자리가 없다
                c = c + nrm * d
            else:
                c = c + nrm * rng.uniform(-cfg.lateral_m, cfg.lateral_m)
        obs = Obstacle(center_xy=c, size=size, yaw=yaw)
        uvz = project(obs.center3[None], cam)[0]
        if not (np.isfinite(uvz[0]) and 0 <= uvz[0] < cam.width and 0 <= uvz[1] < cam.height_px):
            continue
        return obs
    return None


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import cv2

    from local_map import BEV_SIZE, build_local_map, robot_xy_to_bev_ij

    W, H = 640, 480
    cam = Cam(pitch_deg=30.0, height=1.25, fx=388.19, fy=388.19, cx=319.5, cy=239.5,
              width=W, height_px=H)

    # 1) 정면 2 m, 축정렬 박스: 앞면까지의 Z-depth가 해석적으로 맞는가
    box = Obstacle(center_xy=np.array([2.0, 0.0]), size=np.array([0.6, 0.6, 1.0]), yaw=0.0)
    t, hit, normal = render_obstacle(box, cam)
    uvz = project(np.array([[2.0 - 0.3, 0.0, 0.5]]), cam)[0]        # 앞면 중앙
    ui, vi = int(round(uvz[0])), int(round(uvz[1]))
    assert hit[vi, ui], '앞면 중앙에 광선이 안 맞았다'
    err = abs(t[vi, ui] - uvz[2])
    assert err < 1e-3, f'Z-depth 불일치 {err:.4f} m'
    assert np.allclose(normal[vi, ui], [-1, 0, 0], atol=1e-6), normal[vi, ui]
    print(f'[obs] ray-box Z-depth 오차 {err:.2e} m, 법선 OK, hit {100*hit.mean():.2f}% 픽셀')

    # 2) 합성 -> BEV 재계산 -> footprint 정합
    depth0 = np.full((H, W), 4.5, np.float32)                        # 평평한 가상 벽
    rgb0 = np.full((H, W, 3), 120, np.uint8)
    rgb1, depth1, mask = composite(rgb0, depth0, box, cam)
    assert mask.sum() > 0 and np.array_equal(mask, (rgb1 != rgb0).any(-1)), 'RGB/depth 마스크 불일치'

    from episode_io import to_224_depth
    b0 = build_local_map(to_224_depth(depth0), '125cm_30deg', 30.0, 0.10, 0.15, 1.25)
    b1 = build_local_map(to_224_depth(depth1), '125cm_30deg', 30.0, 0.10, 0.15, 1.25)
    new = b1['occupied'] & ~b0['occupied']
    fp = np.zeros((BEV_SIZE, BEV_SIZE), np.uint8)
    poly = robot_xy_to_bev_ij(footprint_xy(box))[:, ::-1].astype(np.int32)   # (i,j)->(x=j,y=i)
    cv2.fillPoly(fp, [poly], 1)
    fp_d = cv2.dilate(fp, np.ones((3, 3), np.uint8)) > 0                     # 1셀 이산화 여유
    prec = float((new & fp_d).sum()) / max(int(new.sum()), 1)
    print(f'[obs] 신규 점유 {int(new.sum())}셀, footprint 안 비율 {100*prec:.1f}%')
    assert int(new.sum()) > 5 and prec > 0.95, (int(new.sum()), prec)

    # 3) 역투영한 depth가 박스 표면 위인가 (합성 depth <-> 기하 일치)
    o_r, d_r = _ray_grid_robot(cam)
    pts = o_r + d_r[mask] * depth1[mask][:, None]
    loc = (pts - box.center3) @ box.rot()
    surf = np.abs(np.abs(loc) - box.size / 2.0).min(axis=1)
    print(f'[obs] 역투영 점의 박스 표면 거리 median {np.median(surf):.2e} m, p99 {np.percentile(surf,99):.2e} m')
    assert np.median(surf) < 1e-6, np.median(surf)

    # 4) 실루엣 == 8꼭짓점 convex hull (가림 없는 장면이므로 정확히 일치해야)
    uv = project(corners_xyz(box), cam)[:, :2]
    hull = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2))
    hull_m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(hull_m, hull.astype(np.int32), 1)
    inter = float((mask & (hull_m > 0)).sum()); union = float((mask | (hull_m > 0)).sum())
    print(f'[obs] 실루엣 vs convex hull IoU {inter/union:.4f}')
    assert inter / union > 0.98, inter / union

    # 5) kind 분기 방어
    try:
        render_obstacle(Obstacle(np.zeros(2), np.ones(3), kind='mesh'), cam)
        raise SystemExit('unreachable 분기가 안 막혔다')
    except AssertionError as e:
        assert 'unreachable' in str(e)
    print('[obs] PASS')
