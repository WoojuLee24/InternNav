"""gs_vlnpe 파이프라인 공용 기하/projection 유틸리티.

04/06 등 이후 스크립트는 직접 재구현하지 말고 이 모듈을 import해서 쓸 것.

## 좌표 규약
- 이 모듈이 만드는 3D 점: **OpenCV** (x=right, y=down, z=forward)
- parquet `action[t]`의 회전: **OpenGL/USD** (y=up, z=backward)
  -> `pose_c2w = action[t] @ CAM_CV_TO_GL` = `action_to_c2w(action, 'cam2world_gl')`
- pose 해석은 전부 `action_to_c2w()` 하나를 거친다. `action`을 직접 행렬로 쓰지 말 것.
- `camera_extrinsic`은 에피소드마다 다르다 — `decompose_camera_extrinsic`으로 (h_b, pitch) 분해.

## 정합 확인 3종
- `check_alignment_fpv` / `check_alignment_bev`: 같은 상대 변환, 격자만 다름(정면 / top-down).
  앞 8개 인자와 반환 키가 동일하다. **둘 다 전역 강체변환은 검출하지 못한다.**
- `check_against_scene_mesh`: 관측 <-> 외부 GT mesh. **좌표계 자체를 검증하는 유일한 검사.**

## 회귀 게이트
`python geometry_utils.py` — identity / CAM_CV_TO_GL involution / camera_extrinsic 왕복 /
mesh 표면거리 <1mm / GT pitch 대조. pose 규약을 건드리면 여기서 먼저 깨진다.

상세 설명과 과거 오진 이력: `.claude/memory/understanding_gs_vlnpe_fpv_bev_geometry.md`
"""

from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

# 실측 근거: raw uint16을 mm(즉 /1000)로 해석하면 모든 프레임에서 최소값이 8.5~13m로 나와
# 실내 D435i 카메라 관측치로는 비현실적이다(카메라 바로 앞 바닥조차 8m 이상이 됨).
# 0.1mm 단위(/10000)로 해석하면 최소값 0.9~1.3m, 중앙값 1.3~2.8m로 실내 스케일에 부합한다.
# ground-truth 장애물 포인트클라우드(meta/pointcloud_obstacle.npy)와의 스케일 비교로도 교차검증됨
# (00_inspect_vln_n1.py의 pose-convention 조사 기록 참고).
DEPTH_SCALE_RAW_TO_M = 1.0 / 10000.0
DEPTH_INVALID_RAW = {0, 65535}

# OpenCV 카메라 프레임 -> OpenGL/USD 카메라 프레임 (parquet `action[t]`의 회전이 기대하는 것).
# 실측: GT depth를 world로 올려 씬 mesh 표면까지 재면 flip 적용 시 0.00003 m, 미적용 시 0.27~0.60 m.
# 이 flip이 빠져도 FPV는 멀쩡해 보인다(상대 pose라 상쇄) — 절대 좌표에서만 드러난다.
CAM_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)

# `action_to_c2w`가 받는 convention 문자열. 'cam2world_gl'이 실측으로 확정된 정답이고,
# 나머지 둘은 00_inspect_vln_n1.py가 판별 근거로 비교하는 (기각된) 후보다.
POSE_CONVENTIONS = ('cam2world_gl', 'cam2world', 'world2cam')

# `pointcloud.ply`에서 obstacle을 고르는 색 라벨. 값·허용치 모두 InternNav 데이터로더의
# `NavDP_Base_Datset.process_obstacle_points`와 동일하게 맞췄다(navdp_lerobot_dataset.py:319).
GT_OBSTACLE_COLOR = np.array([0.0, 0.0, 0.5])
GT_OBSTACLE_COLOR_TOL = 0.05




# ---------------------------------------------------------------------------
# IO 헬퍼
# ---------------------------------------------------------------------------

def load_rgb_frame(rgb_dir: Path, episode: int, frame_idx: int) -> np.ndarray:
    """(H,W,3) uint8 RGB"""
    path = rgb_dir / f'episode_{episode:06d}_{frame_idx:03d}.jpg'
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth_frame_m(depth_dir: Path, episode: int, frame_idx: int, max_depth_m: float):
    """uint16 PNG(raw/10000=meters) -> (depth_m with NaN at invalid pixels, valid_mask)"""
    path = depth_dir / f'episode_{episode:06d}_{frame_idx:03d}.png'
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    raw = raw.astype(np.uint16)
    depth_m = raw.astype(np.float32) * DEPTH_SCALE_RAW_TO_M
    valid = np.isin(raw, list(DEPTH_INVALID_RAW), invert=True) & (depth_m <= max_depth_m)
    depth_m = np.where(valid, depth_m, np.nan)
    return depth_m, valid


def save_jpg(rgb: np.ndarray, path: Path, quality: int = 90) -> Path:
    """(H,W,3) RGB uint8 -> JPG 저장 (base64 임베드 시 파일 크기를 줄이기 위해 PNG 대신 JPEG 사용)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


def colorize_depth(depth_m: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """(H,W) depth[m] -> (H,W,3) uint8 RGB 시각화(참고용, invalid=검정)."""
    if not valid_mask.any():
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    lo, hi = np.nanmin(depth_m[valid_mask]), np.nanmax(depth_m[valid_mask])
    norm = np.clip((np.nan_to_num(depth_m, nan=lo) - lo) / (hi - lo + 1e-6), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    colormap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
    color_bgr = cv2.applyColorMap(u8, colormap)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_rgb[~valid_mask] = 0
    return color_rgb


def mark_invalid_as_checkerboard(rgb: np.ndarray, valid_mask: np.ndarray, cell: int = 10) -> np.ndarray:
    """valid_mask가 False인 픽셀을 회색 체크무늬로 채운다.

    warp/rasterize 결과는 원본과 같은 크기지만, 데이터가 없는 픽셀을 검은색(0)으로 채우면
    "이미지가 작아진 것"처럼 보이는 착시가 생긴다(실측으로 확인). 체크무늬는 "데이터 없음"을
    명확히 표시하면서 크기는 그대로 유지한다.
    """
    h, w = valid_mask.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    checker = ((xx // cell + yy // cell) % 2 == 0)
    gray = np.where(checker, 150, 100).astype(np.uint8)
    pattern = np.stack([gray, gray, gray], axis=-1)
    out = rgb.copy()
    out[~valid_mask] = pattern[~valid_mask]
    return out


# ---------------------------------------------------------------------------
# Pinhole 기하 (일반 4x4 pose 기반)
# ---------------------------------------------------------------------------

def build_k_inv(k: np.ndarray) -> np.ndarray:
    """(3,3) intrinsic -> (3,3) K_inv, `internnav/model/utils/depth_rgb_to_bev_torch.py`의
    `_build_ray_grid`와 동일한 명시적 폐형식(K가 표준 pinhole이라 역행렬을 직접 풀 수 있음):
    K_inv = [[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]].
    """
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    return np.array([
        [1.0 / fx, 0.0, -cx / fx],
        [0.0, 1.0 / fy, -cy / fy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def build_ray_grid(h: int, w: int, k_inv: np.ndarray) -> np.ndarray:
    """(3,3) K_inv -> (H,W,3) 정규화된 카메라광선 방향(K_inv @ [u,v,1]).

    `depth_rgb_to_bev_torch.py`의 `_build_ray_grid`를 numpy로 이식(동일 pinhole 컨벤션:
    x=right, y=down, z=forward). depth로 스케일하면 카메라 프레임 3D 포인트가 된다.
    """
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=0)  # (3,H,W)
    return np.einsum('ij,jhw->hwi', k_inv, uv1)  # (H,W,3)


def unproject_to_camera_frame(depth_m: np.ndarray, k: np.ndarray) -> np.ndarray:
    """(H,W) depth[NaN=invalid], (3,3) K -> (H,W,3) camera-frame 포인트.

    `depth_rgb_to_bev_torch.py`의 `unproject_depth`와 동일한 2단계 구성
    (① K_inv @ [u,v,1]로 정규화 광선, ② depth로 스케일)을 그대로 따른다 — 다만 그 파일은
    정규화[0,1] depth * depth_scale(신경망 출력용)을 쓰고, 우리는 미터 단위로 이미 변환된
    `depth_m`(raw uint16 * DEPTH_SCALE_RAW_TO_M, `load_depth_frame_m` 참고)을 그대로 스케일로 쓴다.
    """
    h, w = depth_m.shape
    ray_grid = build_ray_grid(h, w, build_k_inv(k))  # (H,W,3), z성분은 전부 1
    return ray_grid * depth_m[..., None]


def transform_points(points_cam: np.ndarray, t_mat: np.ndarray) -> np.ndarray:
    """(H,W,3) -> 동차좌표 -> T(4,4) 적용 -> (H,W,3)"""
    h, w = points_cam.shape[:2]
    ones = np.ones((h, w, 1), dtype=points_cam.dtype)
    homog = np.concatenate([points_cam, ones], axis=-1)  # (H,W,4)
    transformed = np.einsum('ij,hwj->hwi', t_mat.astype(points_cam.dtype), homog)
    return transformed[..., :3]


def project_points_to_map(points_cam: np.ndarray, k: np.ndarray):
    """(H,W,3) camera-frame 포인트 -> map_x, map_y(float32 pixel 좌표), z(depth)"""
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    x, y, z = points_cam[..., 0], points_cam[..., 1], points_cam[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        map_x = (fx * x / z + cx).astype(np.float32)
        map_y = (fy * y / z + cy).astype(np.float32)
    return map_x, map_y, z


def in_bounds_mask(map_x: np.ndarray, map_y: np.ndarray, h: int, w: int) -> np.ndarray:
    return (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)


def action_to_c2w(pose: np.ndarray, convention: str) -> np.ndarray:
    """parquet의 `action[t]`(또는 그 후보 해석) -> OpenCV 카메라 프레임 기준 camera-to-world (4,4).

    pose 행렬의 해석은 **전부 이 함수 하나로 모았다** — 새 코드는 `action`을 직접 행렬 연산에
    쓰지 말고 반드시 여기를 거칠 것. 그래야 축 컨벤션(`CAM_CV_TO_GL`) 누락 같은 버그가
    한 군데에서만 관리된다.

    - 'cam2world_gl': `action[t]`가 OpenGL/USD 카메라 컨벤션(x=right, y=up, z=backward)의
      camera-to-world. **실측으로 확정된 정답** (mesh 표면거리 median 0.00003 m).
    - 'cam2world'   : `action[t]`를 OpenCV 컨벤션 camera-to-world로 그대로 해석 (기각된 후보,
      mesh 표면거리 0.271~0.601 m). 00의 판별 비교용으로만 남긴다.
    - 'world2cam'   : `action[t]`를 world-to-camera로 해석 (기각된 후보).
    """
    if convention == 'cam2world_gl':
        return np.asarray(pose, dtype=np.float64) @ CAM_CV_TO_GL
    elif convention == 'cam2world':
        return np.asarray(pose, dtype=np.float64)
    elif convention == 'world2cam':
        return np.linalg.inv(np.asarray(pose, dtype=np.float64))
    else:
        assert False, f'unreachable convention={convention!r}'


def compute_relative_transform(p_t0: np.ndarray, p_t1: np.ndarray, convention: str) -> np.ndarray:
    """camera(t0) -> camera(t1)로 포인트를 옮기는 4x4 변환.

    `T_c2w = action_to_c2w(pose, convention)`라 하면, world 경유로
    `X_cam1 = inv(T_c2w_t1) @ T_c2w_t0 @ X_cam0` — 즉 **`inv(t1) @ t0`**가 맞다.

    **주의**: 이 함수는 두 번 오진해 고쳐졌다(순서 -> 축 flip). 좁은 baseline(gap=3)에서는 틀린
    공식도 matchTemplate offset 0px로 나와 구분되지 않으므로, 고칠 때는 반드시 **넓은 gap**과
    **mesh 앵커**(self-check)로 검증할 것. 직접 재구현 금지.
    """
    tc2w_t0 = action_to_c2w(p_t0, convention)
    tc2w_t1 = action_to_c2w(p_t1, convention)
    return np.linalg.inv(tc2w_t1) @ tc2w_t0


def decompose_camera_extrinsic(e: np.ndarray):
    """(4,4) `observation.camera_extrinsic` -> `(h_b[m], pitch_down[deg])`.

    **"고정 마운트 오프셋"이 아니다** — 에피소드 내에서는 상수지만 **에피소드마다 달라지고**,
    NavDP가 랜덤화하는 로봇 파라미터를 담는다: `e[2,3]` = 로봇 키 `h_b`(실측 0.25~1.49),
    회전 = 카메라 하향 pitch(실측 0~30도). 0행이 `[1,0,0]`(순수 x축 회전)이라 분해된다.
    바닥 높이도 여기서 나온다: `floor_z = cam_z - h_b` (mesh 추정 불필요).
    """
    e = np.asarray(e, dtype=np.float64).reshape(4, 4)
    assert np.allclose(e[0, :3], [1.0, 0.0, 0.0], atol=1e-5), (
        f'camera_extrinsic의 0행이 [1,0,0]이 아니다 ({e[0, :3]}) — 순수 x축 회전 가정이 깨졌으므로 '
        f'이 분해식을 쓰면 안 된다.')
    h_b = float(e[2, 3])
    pitch_down_deg = float(90.0 - np.degrees(np.arctan2(e[2, 1], e[2, 2])))
    return h_b, pitch_down_deg


def compose_camera_extrinsic(h_b: float, pitch_down_deg: float) -> np.ndarray:
    """`(h_b, pitch_down[deg])` -> (4,4) camera_extrinsic. `decompose_camera_extrinsic`의 역."""
    theta = np.radians(90.0 - pitch_down_deg)
    c, s = np.cos(theta), np.sin(theta)
    e = np.eye(4, dtype=np.float64)
    e[1, 1], e[1, 2] = c, -s   # x축 회전
    e[2, 1], e[2, 2] = s, c
    e[2, 3] = h_b
    return e


def synthesize_action_poses(xy_world: np.ndarray, floor_z: float, h_b: float,
                            pitch_down_deg: float) -> np.ndarray:
    """2D world 경로(등간격 리샘플 권장) + `h_b`/pitch -> `action[t]` 포맷 (N,4,4). 04(2-B)가 03의
    `trajectory`를 실제 카메라 pose로 바꾸는 유일한 지점이다.

    위치는 각 점 그대로 `(x, y, floor_z + h_b)`, yaw는 프레임간 tangent(`atan2(dy,dx)`, 중앙차분)로
    정한다. 회전 합성 공식은 **4개 씬/에피소드 조합의 실제 `action[t]`를 역산해 실측으로 검증**했다
    (재구성 오차 최대 3e-8, 부동소수점 수준) — analytic하게 유도해 그대로 믿지 않고 반드시 실측으로
    검증하라는 원칙(`.claude/memory` 참고)을 따른 것이다:

        forward_world(yaw, pitch) = Rz(yaw - 90°) @ mount_forward(pitch)

    `mount_forward`는 `compose_camera_extrinsic(h_b, pitch)`가 만드는 "canonical(yaw 없음)" 마운트를
    `action_to_c2w`로 읽었을 때의 회전이다 — 이 canonical 마운트는 **world +y를 향한다**(yaw=90°에
    해당) — 그래서 `Rz`에 `yaw - 90°`를 넣는다. `Rz`는 world z축(위) 기준 표준 우수 회전이다.

    반환값은 `action_to_c2w(poses[t], 'cam2world_gl')`로 다시 읽으면 이 함수가 만든 cam2world와
    정확히 같아지도록 `CAM_CV_TO_GL`을 역으로 적용해 뒀다(그 자체가 involution이라 같은 행렬을 곱한다).
    """
    xy = np.asarray(xy_world, dtype=np.float64)
    n = len(xy)
    assert n >= 1, 'synthesize_action_poses: 빈 경로'

    if n == 1:
        yaw = np.zeros(1)
    elif n == 2:
        d = xy[1] - xy[0]
        yaw = np.full(2, np.arctan2(d[1], d[0]))
    else:
        yaw = np.empty(n)
        yaw[0] = np.arctan2(*(xy[1] - xy[0])[::-1])
        yaw[-1] = np.arctan2(*(xy[-1] - xy[-2])[::-1])
        d = xy[2:] - xy[:-2]
        yaw[1:-1] = np.arctan2(d[:, 1], d[:, 0])

    mount_cv_rot = action_to_c2w(compose_camera_extrinsic(h_b, pitch_down_deg), 'cam2world_gl')[:3, :3]
    delta = yaw - np.pi / 2.0
    c, s = np.cos(delta), np.sin(delta)
    rz = np.zeros((n, 3, 3), dtype=np.float64)
    rz[:, 0, 0], rz[:, 0, 1] = c, -s
    rz[:, 1, 0], rz[:, 1, 1] = s, c
    rz[:, 2, 2] = 1.0

    c2w_cv = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c2w_cv[:, :3, :3] = rz @ mount_cv_rot
    c2w_cv[:, :3, 3] = np.stack([xy[:, 0], xy[:, 1], np.full(n, floor_z + h_b)], axis=1)
    return c2w_cv @ CAM_CV_TO_GL   # action_to_c2w(action,'cam2world_gl') == action @ CAM_CV_TO_GL 역산


def unproject_to_world_frame(depth_m: np.ndarray, k: np.ndarray, pose: np.ndarray,
                              convention: str = 'cam2world_gl') -> np.ndarray:
    """(H,W) depth, (3,3) K, (4,4) `action[t]` pose -> (H,W,3) **절대 world-frame** 포인트.

    `depth_rgb_to_bev_torch.py`의 `unproject_depth`가 하는 일과 동일(카메라 포인트를 world로
    회전+이동)하지만, 그 파일은 pitch각+높이만 아는 단순 모델이라 `_build_R_c2w`로 회전을
    따로 만든다. 우리는 `action[t]`가 이미 완전한 4x4 pose이므로 `action_to_c2w` +
    `transform_points`면 된다 — 별도 회전 재구성이 필요 없다.

    **이 world 프레임은 씬 mesh 좌표계와 정확히 같다**(표면거리 median 0.00003 m). 따라서 서로
    다른 프레임을 각각 독립적으로 넣어 하나의 world로 합쳐도 된다 — anchor 우회는 필요 없다.
    """
    return transform_points(unproject_to_camera_frame(depth_m, k), action_to_c2w(pose, convention))


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def compute_photometric_error(warped: np.ndarray, ref: np.ndarray, valid_mask: np.ndarray):
    """mean |warped-ref| (valid 픽셀만, RGB 채널 평균). (nan, 0.0) if no valid pixels."""
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return float('nan'), 0.0
    diff = np.abs(warped.astype(np.float32) - ref.astype(np.float32)).mean(axis=-1)
    mean_err = float(diff[valid_mask].mean())
    valid_frac = n_valid / valid_mask.size
    return mean_err, valid_frac


# ---------------------------------------------------------------------------
# BEV 공유 grid rasterize (t0/t1을 같은 world 좌표계에 올려 비교하기 위함)
# ---------------------------------------------------------------------------

def compute_shared_grid(point_sets, cell_m: float = 0.05, pad_m: float = 0.5) -> dict:
    """여러 (유효 마스크 적용된) world-frame 포인트셋을 모두 포괄하는 공통 grid를 계산.

    Args:
        point_sets: [(N_i, 3) float array, ...] — 이미 valid_mask로 걸러진 포인트들
    Returns:
        {'x_min':, 'y_min':, 'w':, 'h':, 'cell_m':}
    """
    all_xy = np.concatenate([p[:, :2] for p in point_sets if len(p) > 0], axis=0)
    x_min, x_max = all_xy[:, 0].min() - pad_m, all_xy[:, 0].max() + pad_m
    y_min, y_max = all_xy[:, 1].min() - pad_m, all_xy[:, 1].max() + pad_m
    w = max(2, int((x_max - x_min) / cell_m) + 1)
    h = max(2, int((y_max - y_min) / cell_m) + 1)
    return {'x_min': float(x_min), 'y_min': float(y_min), 'w': w, 'h': h, 'cell_m': cell_m}


def _world_xy_to_grid_idx(x: np.ndarray, y: np.ndarray, grid: dict):
    cell_m, w, h = grid['cell_m'], grid['w'], grid['h']
    xi = np.clip(((x - grid['x_min']) / cell_m).astype(np.int64), 0, w - 1)
    # y가 클수록(왼쪽) 이미지 위쪽에 오도록: 이미지 행(row)은 위→아래로 증가하므로 y를 뒤집는다.
    yi = np.clip((h - 1 - (y - grid['y_min']) / cell_m).astype(np.int64), 0, h - 1)
    return xi, yi


def rasterize_rgb_to_grid(points_world: np.ndarray, valid_mask: np.ndarray, rgb: np.ndarray, grid: dict):
    """world-frame 포인트+색을 공유 grid 위에 **cell당 평균색**으로 스캐터.

    `depth_rgb_to_bev_torch.py`의 `depth_rgb_to_bev`(scatter_add로 색 합/카운트 누적 후
    나누기)를 numpy로 이식. z-buffer 없이 단순 평균이라 여러 층이 겹쳐도 섞이지만, 이 함수의
    목적은 "같은 world 위치에 t0/t1이 같은 색을 내는지" 비교이므로 평균으로 충분하다.

    Returns:
        bev_rgb: (H,W,3) uint8, 데이터 없는 cell은 0
        occ_mask: (H,W) bool — 이 cell에 점이 하나라도 있었는지
    """
    h, w = grid['h'], grid['w']
    pts = points_world[valid_mask]
    colors = rgb[valid_mask].astype(np.float32)
    if pts.shape[0] == 0:
        return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=bool)

    xi, yi = _world_xy_to_grid_idx(pts[:, 0], pts[:, 1], grid)
    flat_idx = yi * w + xi

    color_sum = np.zeros((h * w, 3), dtype=np.float32)
    count = np.zeros((h * w,), dtype=np.float32)
    np.add.at(color_sum, flat_idx, colors)
    np.add.at(count, flat_idx, 1.0)

    occ = count > 0
    bev = np.zeros((h * w, 3), dtype=np.float32)
    bev[occ] = color_sum[occ] / count[occ, None]
    return bev.reshape(h, w, 3).astype(np.uint8), occ.reshape(h, w)


# ---------------------------------------------------------------------------
# 씬 mesh (absolute 모드의 외부 ground truth)
# ---------------------------------------------------------------------------

def find_scene_mesh(mesh_root: Path, scene: str) -> Path:
    """`<mesh_root>/<scene>/matterport_mesh/<hash>/<hash>.obj` 경로를 찾아 반환.

    해시 폴더명이 씬마다 달라 하드코딩할 수 없으므로 glob으로 찾는다. `isaacsim_*`는 IsaacSim
    변환 사본이라 좌표가 손상됐을 수 있어 원본 `<hash>.obj`를 우선한다.
    """
    mesh_dir = Path(mesh_root) / scene / 'matterport_mesh'
    candidates = sorted(mesh_dir.glob('*/*.obj'))
    originals = [p for p in candidates if not p.name.startswith('isaacsim_')]
    picked = originals or candidates
    if not picked:
        raise FileNotFoundError(f'mesh(.obj) not found under {mesh_dir}')
    return picked[0]


def load_gt_obstacle_points(scene_dir: Path) -> np.ndarray:
    """`<scene>/meta/pointcloud.ply`(원본)에서 obstacle 점만 뽑아 (N,3)로 반환.

    필터는 InternNav 데이터로더의 `NavDP_Base_Datset.process_obstacle_points`
    (`internnav/dataset/navdp_lerobot_dataset.py:319`)와 동일하다 — obstacle은 색으로 라벨링돼 있고,
    이 함수는 그 후처리를 그대로 재현한다.

    **`pointcloud_obstacle.npy`를 읽지 않는 이유**: 그 파일은 릴리스 자산이 아니라
    `scripts/train/base_train/convert_ply_to_npy.py`가 이 필터로 만든 로컬 캐시다(없는 환경도 있다).
    원본에서 직접 뽑는 게 검증 스크립트로서 옳다. 내용은 동일함을 확인했다(array_equal).
    """
    import open3d as o3d  # 이 함수 전용 — 모듈 상단에 두면 open3d 없는 환경에서 import가 깨진다

    pcd = o3d.io.read_point_cloud(str(Path(scene_dir) / 'meta' / 'pointcloud.ply'))
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors, dtype=np.float64)
    is_obstacle = np.abs(colors - GT_OBSTACLE_COLOR).sum(axis=-1) < GT_OBSTACLE_COLOR_TOL
    return points[is_obstacle]


def load_scene_mesh(mesh_root: Path, scene: str):
    """씬 mesh를 trimesh 객체로 로드. absolute 모드/`check_against_scene_mesh`의 공용 진입점.

    호출부는 이걸 **한 번만** 호출해 재사용할 것 — 프레임마다 다시 로드하면 느리다.
    """
    import trimesh  # absolute 모드 전용 — 모듈 상단에 두면 trimesh 없는 환경에서 import가 깨진다
    return trimesh.load(str(find_scene_mesh(mesh_root, scene)), process=False, force='mesh')


def compute_mesh_distance(points_world: np.ndarray, mesh) -> np.ndarray:
    """(N,3) world 점 -> (N,) mesh 표면까지의 거리[m] (부호 무시).

    **이 파이프라인에서 "mesh 표면거리"를 계산하는 곳은 여기 하나뿐이다** — FPV/BEV의 absolute
    모드와 `check_against_scene_mesh`가 전부 이 함수를 거친다.
    """
    import trimesh
    if len(points_world) == 0:
        return np.empty((0,), dtype=np.float64)
    return np.abs(trimesh.proximity.ProximityQuery(mesh).signed_distance(points_world))


def check_alignment_fpv(rgb_t0: np.ndarray, depth_t0: np.ndarray, rgb_t1: np.ndarray, depth_t1: np.ndarray,
                         pose_t0: np.ndarray, pose_t1: np.ndarray, k: np.ndarray, convention: str,
                         visualize: bool = True, out_dir: Path = None, label_prefix: str = 'fpv') -> dict:
    """FPV backward-warp(t0->t1): t0의 depth로 만든 3D 점을 **상대 변환**으로 카메라(t1) 프레임에
    옮겨 재투영해, t0의 픽셀 격자에 t1의 색을 입힌다.

    출력이 t0와 같은 격자이므로 비교 대상은 반드시 `rgb_t0`다 — target인 `rgb_t1`과 비교하면
    서로 다른 시점을 비교하는 셈이라 의미가 없다.

    Returns:
        {'metric_name': 'photometric_error', 'metric_value': float, 'coverage_frac': float,
         'viz_paths': {'rgb_t0': Path, 'warped': Path} or None}
    """
    h, w = depth_t0.shape
    valid0 = ~np.isnan(depth_t0)
    pts_cam_t0 = unproject_to_camera_frame(depth_t0, k)
    t_rel = compute_relative_transform(pose_t0, pose_t1, convention)
    pts_cam_t1 = transform_points(pts_cam_t0, t_rel)
    map_x, map_y, z1 = project_points_to_map(pts_cam_t1, k)

    valid = valid0 & (z1 > 1e-4) & in_bounds_mask(map_x, map_y, h, w)
    warped = cv2.remap(rgb_t1, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    err, valid_frac = compute_photometric_error(warped, rgb_t0, valid)

    viz_paths = None
    if visualize and out_dir is not None:
        out_dir = Path(out_dir)
        viz_paths = {
            'rgb_t0': save_jpg(rgb_t0, out_dir / f'{label_prefix}_rgb_t0.jpg'),
            'warped': save_jpg(mark_invalid_as_checkerboard(warped, valid), out_dir / f'{label_prefix}_warped.jpg'),
        }
    return {'metric_name': 'photometric_error', 'metric_value': err,
            'coverage_frac': valid_frac, 'viz_paths': viz_paths}


def check_alignment_bev(rgb_t0: np.ndarray, depth_t0: np.ndarray, rgb_t1: np.ndarray, depth_t1: np.ndarray,
                         pose_t0: np.ndarray, pose_t1: np.ndarray, k: np.ndarray, convention: str,
                         cell_m: float = 0.05, visualize: bool = True, out_dir: Path = None,
                         label_prefix: str = 'bev') -> dict:
    """BEV(top-down) 정합 확인. **FPV와 완전히 같은 상대 변환**을 쓰고, 그리는 격자만 다르다.

    FPV가 t1의 색을 t0의 **카메라 픽셀 격자**에 재투영해 보여준다면, 이 함수는 같은 상대 변환으로
    t1의 점을 t0 프레임에 옮긴 뒤 **top-down 격자**에 rasterize해 위에서 내려다본 모습으로 보여준다.
    둘 다 데이터가 있는 cell(겹치는 영역)에서 photometric error를 계산한다.

    처리 순서 (**FPV와 인자 순서·변환 방향이 완전히 동일**):
      1. `compute_relative_transform(pose_t0, pose_t1, convention)`으로 t0의 점을 **cam1 프레임**에
         옮긴다 — FPV가 쓰는 것과 글자 그대로 같은 호출이다.
      2. t1을 앵커로 삼아 두 점군에 `action_to_c2w(pose_t1)`를 적용 — 격자를 **중력 정렬**시키기
         위한 것일 뿐이다(둘에 똑같이 적용되므로 비교 결과에는 영향 없음). 이걸 생략하면 카메라의
         고정 pitch tilt(약 6도) 때문에 바닥이 기울어져 보인다.
      3. 공통 grid에 scatter-mean으로 rasterize 후 겹치는 cell만 비교

    앵커를 t0로 잡든 t1로 잡든 결과는 같다(앵커 pose가 상대 변환의 역행렬과 상쇄). t1로 잡은 건
    FPV와 인자 순서를 맞추기 위해서다. **전역 강체변환은 검출하지 못한다** —
    절대 좌표계 검증은 `check_against_scene_mesh`가 전담한다.

    Returns:
        {'metric_name': 'bev_overlap_error', 'metric_value': float, 'coverage_frac': float,
         'viz_paths': {'bev_t0': Path, 'bev_t1': Path} or None}
    """
    valid0 = ~np.isnan(depth_t0)
    valid1 = ~np.isnan(depth_t1)

    # FPV와 **완전히 동일한** 상대 변환 호출: cam(t0) -> cam(t1).
    t_rel = compute_relative_transform(pose_t0, pose_t1, convention)
    pts_cam_t0_in_t1 = transform_points(unproject_to_camera_frame(depth_t0, k), t_rel)
    pts_cam_t1 = unproject_to_camera_frame(depth_t1, k)  # t1은 이미 자기 프레임

    # 격자를 중력 정렬하기 위해 앵커(t1)의 pose만 적용 — 두 점군에 동일하므로 비교 결과 불변.
    anchor_c2w = action_to_c2w(pose_t1, convention)
    world_pts_t0 = transform_points(pts_cam_t0_in_t1, anchor_c2w)
    world_pts_t1 = transform_points(pts_cam_t1, anchor_c2w)

    grid = compute_shared_grid([world_pts_t0[valid0], world_pts_t1[valid1]], cell_m=cell_m)
    bev_t0, occ_t0 = rasterize_rgb_to_grid(world_pts_t0, valid0, rgb_t0, grid)
    bev_t1, occ_t1 = rasterize_rgb_to_grid(world_pts_t1, valid1, rgb_t1, grid)

    overlap = occ_t0 & occ_t1
    err, overlap_frac = compute_photometric_error(bev_t1, bev_t0, overlap)

    viz_paths = None
    if visualize and out_dir is not None:
        out_dir = Path(out_dir)
        viz_paths = {
            'bev_t0': save_jpg(mark_invalid_as_checkerboard(bev_t0, occ_t0), out_dir / f'{label_prefix}_t0.jpg'),
            'bev_t1': save_jpg(mark_invalid_as_checkerboard(bev_t1, occ_t1), out_dir / f'{label_prefix}_t1.jpg'),
        }
    return {'metric_name': 'bev_overlap_error', 'metric_value': err,
            'coverage_frac': overlap_frac, 'viz_paths': viz_paths}


# ---------------------------------------------------------------------------
# self-check — 이 모듈을 고친 뒤 `python geometry_utils.py`로 핵심 불변식이 안 깨졌는지 확인
# ---------------------------------------------------------------------------

MESH_ANCHOR_TOL_M = 1e-3  # GT depth를 world로 올렸을 때 mesh 표면까지 허용 오차(실측치는 3e-5)


def check_against_scene_mesh(data_root: Path, mesh_root: Path, scene: str, episode: int = 0,
                             frames=None, n_frames: int = 4, n_points: int = 400,
                             convention: str = 'cam2world_gl', max_depth_m: float = 10.0) -> dict:
    """**절대 앵커 검증**: GT depth를 world로 올린 점들이 씬 mesh 표면 위에 놓이는지 잰다.

    photometric error 같은 상대 지표는 축 컨벤션이 틀려도 그럴듯한 값을 내므로 pose 규약 확정
    근거가 못 된다. `vln_n1`의 depth는 이 mesh를 렌더한 것이라 표면거리가 곧 정답 여부다
    (맞으면 0.03mm, 틀리면 27cm 이상).

    Returns: {'median_m', 'p90_m', 'per_frame': {frame: median_m}, 'passed'}
    """
    import trimesh  # self-check/01 전용 — 모듈 상단에서 import하면 trimesh 없는 환경에서 못 쓴다
    import pyarrow.parquet as pq

    scene_dir = Path(data_root) / scene
    table = pq.read_table(scene_dir / 'data' / 'chunk-000' / f'episode_{episode:06d}.parquet')
    poses = table['action'].to_pylist()
    k = np.asarray(table['observation.camera_intrinsic'].to_pylist()[0], dtype=np.float64).reshape(3, 3)
    depth_dir = scene_dir / 'videos' / 'chunk-000' / 'observation.images.depth'

    # 프레임은 에피소드 길이에서 균등하게 뽑는다 — 하드코딩하면 짧은 에피소드(예: s8pcmisQ38h는
    # 145프레임)에서 존재하지 않는 인덱스를 읽어 실패한다.
    if frames is None:
        frames = np.unique(np.linspace(0, len(poses) - 1, n_frames).astype(int)).tolist()
    else:
        frames = [f for f in frames if 0 <= f < len(poses)]
    assert frames, f'검사할 프레임이 없다 (episode 길이={len(poses)})'

    mesh = trimesh.load(str(find_scene_mesh(mesh_root, scene)), process=False, force='mesh')
    proximity = trimesh.proximity.ProximityQuery(mesh)

    per_frame, all_d = {}, []
    for frame_idx in frames:
        depth_m, valid = load_depth_frame_m(depth_dir, episode, frame_idx, max_depth_m)
        pose = np.asarray(poses[frame_idx], dtype=np.float64).reshape(4, 4)
        world_pts = unproject_to_world_frame(depth_m, k, pose, convention)[valid]
        sel = np.random.RandomState(0).choice(len(world_pts), min(n_points, len(world_pts)), replace=False)
        d = np.abs(proximity.signed_distance(world_pts[sel]))
        per_frame[int(frame_idx)] = float(np.median(d))
        all_d.append(d)

    all_d = np.concatenate(all_d)
    median_m, p90_m = float(np.median(all_d)), float(np.percentile(all_d, 90))
    return {'median_m': median_m, 'p90_m': p90_m, 'per_frame': per_frame,
            'passed': median_m < MESH_ANCHOR_TOL_M}


if __name__ == '__main__':
    k = np.array([[355.81463623, 0.0, 240.0], [0.0, 351.68701172, 135.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    depth_m = np.random.RandomState(0).rand(270, 480).astype(np.float32) * 3.0 + 0.5
    pts_cam = unproject_to_camera_frame(depth_m, k)

    # ① identity self-transform: t0->t0 여야 하므로 map_x==u, map_y==v가 정확히 나와야 한다.
    identity_pose = np.eye(4, dtype=np.float32)
    t_rel = compute_relative_transform(identity_pose, identity_pose, 'cam2world')
    pts_id = transform_points(pts_cam, t_rel)
    map_x, map_y, z = project_points_to_map(pts_id, k)
    u_grid, v_grid = np.meshgrid(np.arange(480, dtype=np.float32), np.arange(270, dtype=np.float32))
    max_dx = np.abs(map_x - u_grid).max()
    max_dy = np.abs(map_y - v_grid).max()
    assert max_dx < 1e-3 and max_dy < 1e-3, f'identity self-transform 회귀! max_dx={max_dx}, max_dy={max_dy}'
    print(f'[1/6] identity self-transform OK (max_dx={max_dx:.2e}, max_dy={max_dy:.2e})')

    # ② CAM_CV_TO_GL은 involution(자기 자신이 역행렬)이어야 한다 — 축 flip의 기본 성질.
    assert np.allclose(CAM_CV_TO_GL @ CAM_CV_TO_GL, np.eye(4)), 'CAM_CV_TO_GL이 involution이 아님'
    print('[2/6] CAM_CV_TO_GL involution OK')

    # ③ camera_extrinsic 분해/조립 왕복 — 04/05가 이 규약으로 필드를 채운다.
    for _h, _p in [(0.25, 0.0), (0.70, 6.07), (1.24, 30.0), (1.50, 15.5)]:
        _hb, _pd = decompose_camera_extrinsic(compose_camera_extrinsic(_h, _p))
        assert abs(_hb - _h) < 1e-9 and abs(_pd - _p) < 1e-9, f'왕복 실패 ({_h},{_p}) -> ({_hb},{_pd})'
    print('[3/6] camera_extrinsic 분해/조립 왕복 OK')

    # ④ **절대 앵커 회귀 게이트**: GT depth -> world가 씬 mesh 위에 놓이는지.
    #    이 파일의 pose 규약을 건드리면 여기가 가장 먼저 깨진다.
    _DATA_ROOT = Path('data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i')
    _MESH_ROOT = Path('data/scene_data/mp3d_n1')
    _SCENE = '17DRP5sb8fy'
    if (_DATA_ROOT / _SCENE).is_dir() and (_MESH_ROOT / _SCENE).is_dir():
        res = check_against_scene_mesh(_DATA_ROOT, _MESH_ROOT, _SCENE)
        assert res['passed'], (
            f"mesh 앵커 회귀! median={res['median_m']:.6f}m > {MESH_ANCHOR_TOL_M}m — "
            f"pose 규약(CAM_CV_TO_GL / action_to_c2w)을 확인할 것. per_frame={res['per_frame']}")
        print(f"[4/6] mesh anchor OK (median={res['median_m']:.2e} m, p90={res['p90_m']:.2e} m)")
    else:
        print(f'[4/6] mesh anchor SKIP (데이터 없음: {_DATA_ROOT / _SCENE} 또는 {_MESH_ROOT / _SCENE})')

    # ⑤ **GT 대조 게이트**: camera_extrinsic에서 뽑은 pitch가 action의 카메라 forward와 맞는지.
    #    분해식이 틀어지면 04가 잘못된 pitch로 카메라를 배치하게 되므로 여기서 먼저 걸린다.
    if (_DATA_ROOT / _SCENE).is_dir():
        import pyarrow.parquet as _pq
        _worst = 0.0
        for _ep in range(4):
            _t = _pq.read_table(_DATA_ROOT / _SCENE / 'data' / 'chunk-000' / f'episode_{_ep:06d}.parquet',
                                columns=['observation.camera_extrinsic', 'action'])
            _e = np.asarray(_t['observation.camera_extrinsic'].to_pylist()[0], dtype=np.float64).reshape(4, 4)
            _a = np.asarray(_t['action'].to_pylist()[0], dtype=np.float64).reshape(4, 4)
            _hb, _pd = decompose_camera_extrinsic(_e)
            _fwd_z = action_to_c2w(_a, 'cam2world_gl')[2, 2]   # OpenCV z=forward
            _worst = max(_worst, abs(_fwd_z - (-np.sin(np.radians(_pd)))))
        assert _worst < 1e-6, f'camera_extrinsic pitch 분해가 action forward와 불일치! max diff={_worst:.2e}'
        print(f'[5/6] GT camera_extrinsic pitch 대조 OK (max diff={_worst:.2e})')
    else:
        print('[5/6] GT camera_extrinsic pitch 대조 SKIP (데이터 없음)')

    # ⑥ **`synthesize_action_poses` 회귀 게이트**: 04(2-B)의 pose 합성 공식이 실제 GT 궤적을
    #    (위치+tangent yaw+h_b/pitch)만으로 재구성했을 때 진짜 `action[t]`와 일치하는지.
    #    이 공식은 analytic 유도만으로 확정하지 않고 실측(정확한 yaw 역산 -> 회전 재구성 오차 3e-8,
    #    부동소수점 수준)으로 검증했다 — 유도가 틀렸다면 여기서 가장 먼저 깨진다.
    #    게이트는 **yaw 각도[도]**로 잰다(원시 회전행렬 원소 diff가 아니다) — tangent(중앙차분)로
    #    추정한 yaw는 회전 구간에서 실제 순간 yaw와 최대 1도 안팎 어긋나는 게 정상이고(이산 샘플의
    #    본질적 한계, 공식 버그 아님), 그 각도가 실제로 작은지를 재는 게 의미 있는 검증이다.
    _SCENE_EPS = [('17DRP5sb8fy', 0), ('17DRP5sb8fy', 3), ('s8pcmisQ38h', 0), ('s8pcmisQ38h', 2)]
    _available = [(s, e) for s, e in _SCENE_EPS if (_DATA_ROOT / s).is_dir()]
    if _available:
        import pyarrow.parquet as _pq
        _worst_yaw_deg = 0.0
        for _scene, _ep in _available:
            _t = _pq.read_table(_DATA_ROOT / _scene / 'data' / 'chunk-000' / f'episode_{_ep:06d}.parquet',
                                columns=['observation.camera_extrinsic', 'action'])
            _e = np.asarray(_t['observation.camera_extrinsic'].to_pylist()[0], dtype=np.float64).reshape(4, 4)
            _hb, _pd = decompose_camera_extrinsic(_e)
            _actions = np.stack([np.asarray(a, dtype=np.float64).reshape(4, 4)
                                 for a in _t['action'].to_pylist()])
            _real_c2w = np.stack([action_to_c2w(a, 'cam2world_gl') for a in _actions])
            _floor_z = float(_real_c2w[0, 2, 3] - _hb)
            _synth_action = synthesize_action_poses(_real_c2w[:, :2, 3], _floor_z, _hb, _pd)
            _synth_c2w = np.stack([action_to_c2w(a, 'cam2world_gl') for a in _synth_action])
            # 경로 양끝은 tangent가 인접 세그먼트 하나뿐이라 실제 순간 yaw와 어긋날 수 있어 내부만 잰다.
            _real_yaw = np.degrees(np.arctan2(_real_c2w[2:-2, 1, 2], _real_c2w[2:-2, 0, 2]))
            _synth_yaw = np.degrees(np.arctan2(_synth_c2w[2:-2, 1, 2], _synth_c2w[2:-2, 0, 2]))
            _d = (_synth_yaw - _real_yaw + 180.0) % 360.0 - 180.0
            _worst_yaw_deg = max(_worst_yaw_deg, float(np.abs(_d).max()))
        assert _worst_yaw_deg < 2.0, f'synthesize_action_poses 회귀! max yaw diff={_worst_yaw_deg:.3f}deg'
        print(f'[6/6] synthesize_action_poses 회귀 OK (max yaw diff={_worst_yaw_deg:.3f}deg, tangent 추정 한계 이내)')
    else:
        print('[6/6] synthesize_action_poses 회귀 SKIP (데이터 없음)')

    print('[geometry_utils self-check] 전부 통과')
