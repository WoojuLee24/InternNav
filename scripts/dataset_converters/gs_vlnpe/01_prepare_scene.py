"""M1.1 — 씬 자산이 GT와 같은 world 프레임인지 판정하는 게이트.

    00: vln_n1 데이터 -> "어떻게 읽고 쓰나"  -> target_schema.json      (전체에 1개)
    01: 씬 mesh/USD    -> "이 씬 써도 되나"    -> scene_meta/<scene>.json (씬마다 1개)
둘 다 `check_against_scene_mesh`를 쓰지만 00은 규약 후보 3개 중 고르고(한 번), 01은 그 규약으로
씬을 검사한다(새 씬마다).

[목적]
원안은 "mesh를 Z-up·1m로 정규화 + `scene.usd` 생성"이었으나 실측 결과 둘 다 불필요했다 —
mesh가 이미 정합돼 있고 `fixed.usd`도 65개 씬 전부에 존재한다. 그래서 **게이트 역할만** 남았다:
02(집 전체 0.05m voxelize, 무거움)에 들어가기 전에 수십 초로 씬을 거르고 `scene_meta`를 남긴다.

[검증 4가지] (하나라도 실패하면 non-zero exit)
  1. obj bounds <-> 기존 `fixed.usd` bounds 일치
  2. up-axis Z + 씬 extent 자릿수 (100배/0.01배 스케일 오류만 잡는다 — 다층 건물이 섞여 있어서
     층고 같은 좁은 범위로 잡으면 정상 씬을 오판한다)
  3. **GT depth -> world -> mesh 표면거리 median < 1mm** (핵심)
  4. GT 장애물(pointcloud.ply)·궤적이 bounds 안 + 카메라 높이(= camera_extrinsic의 h_b)가 합리적

기하 계산은 `geometry_utils.py`를 재사용한다(직접 재구현 금지).

[필드 규약: `action` vs `camera_extrinsic`] (검증 ④에서 씀 — 자세한 설명은 00 참고)
  - `action[t]`는 매 프레임 카메라의 world-frame pose. `camera_extrinsic`(4,4, 에피소드 내 상수)은
    world pose가 아니라 로봇 base 기준 카메라 마운트 오프셋 — `e[2,3]` = h_b(로봇 키, m),
    회전 = 카메라 하향 pitch(deg). 둘 다 NavDP가 에피소드마다 랜덤화하는 값이라 에피소드마다 다르다.
  - `floor_z = cam_z(action) - h_b(camera_extrinsic)`로 바닥 world 높이를 얻는다(mesh 추정 아님,
    `floor_z_from_extrinsic` 참고) — 검증 ④가 이 높이가 사람/로봇 카메라로 합리적인지(0.2~2.5 m) 본다.

[입력]
  - `<mesh_root>/<scene>/matterport_mesh/*/*.obj` — 씬 mesh(trimesh로 로드, world 프레임 절대 앵커)
  - `<usd_root>/<scene>/matterport_mesh/*/fixed.usd` — 기존 USD(생성 안 함, up-axis/스케일/bounds만 읽음)
  - `<data_root>/<scene>/meta/pointcloud.ply` — GT 장애물 포인트클라우드 원본(색 라벨로 obstacle만 필터)
  - `<data_root>/<scene>/data/chunk-000/episode_XXXXXX.parquet` — `action`(float32,(T,4,4)),
    `observation.camera_intrinsic`(float32,(3,3)), `observation.camera_extrinsic`(float32,(4,4))
  - `<data_root>/<scene>/videos/chunk-000/observation.images.depth/episode_000000_000.png`
    (uint16, raw/10000 = m — mesh 표면거리·top-down 시각화용)

[출력]
  - `<out_dir>/scene_meta/<scene>.json` — 씬마다 1개(canonical). `bounds_min/max`, `usd_path`,
    `floor_z`, `gt_robot_params`(h_b/pitch_down_deg, episode 0 기준), `pose_convention`,
    `frame_alignment`(에피소드별 mesh 표면거리), `passed` 등. 02/03/04가 이 파일을 읽는다.
  - `<log_dir>/01_prepare_scene/<scene>/report.html` — floorplan blink + mesh 표면거리 히스토그램
  - `<log_dir>/01_prepare_scene/<scene>/*.jpg` — floorplan/히스토그램 이미지

[실행 예시]
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/01_prepare_scene.py \\
        --scene 17DRP5sb8fy

[negative test] (게이트가 항상 통과만 내지 않음을 보장)
    ... --scene 17DRP5sb8fy --pose_convention cam2world      # -> passed=false, exit != 0
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_utils import (  # noqa: E402
    MESH_ANCHOR_TOL_M,
    POSE_CONVENTIONS,
    _world_xy_to_grid_idx,
    action_to_c2w,
    check_against_scene_mesh,
    decompose_camera_extrinsic,
    find_scene_mesh,
    load_gt_obstacle_points,
    load_depth_frame_m,
    load_scene_mesh,
    save_jpg,
    unproject_to_world_frame,
)
from viz_utils import blink_widget_html, reference_button_html, save_gallery  # noqa: E402

# ---------------------------------------------------------------------------
# 1. 상수
# ---------------------------------------------------------------------------

DEFAULT_DATA_ROOT = 'data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i'
DEFAULT_MESH_ROOT = 'data/scene_data/mp3d_n1'
DEFAULT_USD_ROOT = 'data/scene_data/mp3d_pe'
DEFAULT_SCENE = '17DRP5sb8fy'
DEFAULT_OUT_DIR = 'scripts/dataset_converters/gs_vlnpe'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe'
SCRIPT_NAME = '01_prepare_scene'

BOUNDS_ATOL_M = 1e-3          # obj bounds ↔ USD bounds 허용 오차

# 스케일 sanity 범위. **좁게 잡으면 안 된다** — Matterport 씬에는 다층 건물이 섞여 있다
# (예: s8pcmisQ38h는 Z extent 12.24 m). 처음엔 "실내 층고 2~4m"로 잡았다가 다층 씬을 오판했다.
# 여기서 잡으려는 건 "미터 단위가 맞는가"(즉 100배/0.01배 스케일 오류)뿐이고, **진짜 스케일 근거는
# 검증 ③**이다 — depth(미터)를 world로 올려 mesh 표면에 0.03mm로 얹히는 것 자체가 스케일 증명이다.
SCENE_EXTENT_RANGE_M = (1.0, 100.0)

# floorplan 실루엣에 쓸 높이대 — 바닥/천장을 빼야 벽·가구 윤곽이 보인다.
FLOORPLAN_Z_BAND_M = (0.15, 2.0)
FLOORPLAN_CELL_M = 0.05
FLOORPLAN_MIN_PX = 700        # 작은 씬도 리포트에서 보이도록 최소 크기까지 확대


# ---------------------------------------------------------------------------
# 2. CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT, help='vln_n1 traj_data 루트 (씬 폴더들의 부모)')
    parser.add_argument('--mesh_root', default=DEFAULT_MESH_ROOT, help='obj mesh 루트 (<mesh_root>/<scene>/matterport_mesh/*/*.obj)')
    parser.add_argument('--usd_root', default=DEFAULT_USD_ROOT,
                        help='기존 USD 루트. 생성하지 않고 존재·bounds 확인만 한다 '
                             '(<usd_root>/<scene>/matterport_mesh/*/fixed.usd)')
    parser.add_argument('--scene', default=DEFAULT_SCENE, help='씬 ID (예: 17DRP5sb8fy)')
    parser.add_argument('--num_episodes', type=int, default=3, help='정합 검사할 에피소드 수 (0..N-1)')
    parser.add_argument('--frames_per_episode', type=int, default=4,
                        help='에피소드당 검사 프레임 수 (에피소드 길이에서 균등 추출)')
    parser.add_argument('--pose_convention', default='cam2world_gl', choices=list(POSE_CONVENTIONS),
                        help='pose 해석 규약. 기본값이 M1.0에서 확정된 정답이다. 일부러 틀린 값을 넣어 '
                             '게이트가 실제로 실패하는지 확인하는 negative test 용도로만 바꾼다.')
    parser.add_argument('--max_depth_m', type=float, default=10.0, help='유효 depth 최대값(m)')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR,
                        help='scene_meta/<scene>.json 출력 루트 (파이프라인 canonical 파일)')
    parser.add_argument('--log_dir', default=DEFAULT_LOG_DIR,
                        help='시각화 출력 루트. logs/gs-vlnpe/<script_name>/<scene>/ 아래에 저장된다')
    return parser


# ---------------------------------------------------------------------------
# 3. 검증 ①② — mesh / USD 자산
# ---------------------------------------------------------------------------

def probe_usd(usd_root: str, scene: str) -> dict:
    """기존 USD의 up-axis / metersPerUnit / bounds를 읽는다 (생성하지 않음).

    InternNav가 실제로 로드하는 파일 선택 규칙은
    `internnav/env/utils/episode_loader/generate_episode.py:15`와 동일하게 맞춘다
    (컨테이너면 fixed_docker.usd, 아니면 fixed.usd).
    """
    from pxr import Usd, UsdGeom

    usd_dir = Path(usd_root) / scene / 'matterport_mesh'
    candidates = sorted(usd_dir.glob('*/fixed.usd'))
    if not candidates:
        return {'found': False, 'searched': str(usd_dir / '*' / 'fixed.usd')}

    usd_path = candidates[0]
    docker_path = usd_path.parent / 'fixed_docker.usd'
    stage = Usd.Stage.Open(str(usd_path))
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
    aligned = bbox.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    return {
        'found': True,
        'usd_path': str(usd_path),
        'usd_docker_path': str(docker_path) if docker_path.exists() else None,
        'up_axis': str(UsdGeom.GetStageUpAxis(stage)),
        'meters_per_unit': float(UsdGeom.GetStageMetersPerUnit(stage)),
        'bounds_min': np.asarray(aligned.GetMin(), dtype=np.float64).tolist(),
        'bounds_max': np.asarray(aligned.GetMax(), dtype=np.float64).tolist(),
    }


def check_mesh_usd_bounds(mesh_bounds: np.ndarray, usd_info: dict) -> dict:
    """검증 ① obj bounds ↔ USD bounds 일치 + ② up-axis Z / 스케일 자릿수 sanity.

    ②는 **자릿수만** 본다(`SCENE_EXTENT_RANGE_M` 주석 참고) — 다층 건물이 섞여 있어서 층고 같은
    좁은 범위로 잡으면 정상 씬을 오판한다. 정밀한 스케일 근거는 검증 ③이다.
    """
    extents = (mesh_bounds[1] - mesh_bounds[0]).astype(float)
    lo, hi = SCENE_EXTENT_RANGE_M
    scale_ok = bool(np.all(extents >= lo) and np.all(extents <= hi))

    if not usd_info['found']:
        return {'usd_found': False, 'bounds_match': False, 'max_abs_diff_m': None,
                'extents_m': extents.tolist(), 'expected_extent_range_m': list(SCENE_EXTENT_RANGE_M),
                'scale_sane': scale_ok, 'usd_up_axis_z': False}

    diff = np.abs(np.array([usd_info['bounds_min'], usd_info['bounds_max']]) - mesh_bounds).max()
    return {
        'usd_found': True,
        'bounds_match': bool(diff <= BOUNDS_ATOL_M),
        'max_abs_diff_m': float(diff),
        'tolerance_m': BOUNDS_ATOL_M,
        'extents_m': extents.tolist(),
        'expected_extent_range_m': list(SCENE_EXTENT_RANGE_M),
        'scale_sane': scale_ok,
        'usd_up_axis_z': usd_info['up_axis'].upper().endswith('Z'),
    }


# ---------------------------------------------------------------------------
# 4. 검증 ③ — mesh 표면거리 (핵심 게이트). geometry_utils를 그대로 호출한다.
# ---------------------------------------------------------------------------

def count_episodes(data_root: str, scene: str) -> int:
    return len(sorted((Path(data_root) / scene / 'data' / 'chunk-000').glob('episode_*.parquet')))


def run_alignment_gate(args, n_available: int) -> dict:
    """에피소드마다 `check_against_scene_mesh`를 호출해 결과를 합친다.

    표면거리 median이 `MESH_ANCHOR_TOL_M`(1mm) 미만이면 통과. photometric error 같은 상대
    지표로는 좌표 규약을 확정할 수 없어서(축 오류가 상쇄됨) 절대 앵커인 mesh로 재는 것이다.
    """
    episodes = list(range(min(args.num_episodes, n_available)))
    per_episode, medians = {}, []
    for ep in episodes:
        res = check_against_scene_mesh(
            args.data_root, args.mesh_root, args.scene, episode=ep,
            n_frames=args.frames_per_episode, convention=args.pose_convention,
            max_depth_m=args.max_depth_m)
        per_episode[ep] = {'median_m': res['median_m'], 'p90_m': res['p90_m'],
                           'passed': res['passed'], 'per_frame': res['per_frame']}
        medians.append(res['median_m'])
        print(f'    episode {ep:>3}: median={res["median_m"]:.6f} m  p90={res["p90_m"]:.6f} m  '
              f'{"OK" if res["passed"] else "FAIL"}')

    worst = float(max(medians)) if medians else float('nan')
    return {
        'pose_convention': args.pose_convention,
        'episodes_tested': episodes,
        'frames_per_episode': args.frames_per_episode,
        'median_m': float(np.median(medians)) if medians else float('nan'),
        'worst_episode_median_m': worst,
        'tolerance_m': MESH_ANCHOR_TOL_M,
        'per_episode': per_episode,
        # 한 에피소드라도 넘으면 실패 — 게이트이므로 가장 나쁜 것을 기준으로 판정한다.
        'passed': bool(medians) and worst < MESH_ANCHOR_TOL_M,
    }


# ---------------------------------------------------------------------------
# 5. 검증 ④ — GT가 mesh bounds 안에 있는가
# ---------------------------------------------------------------------------

def load_gt_reference(data_root: str, scene: str, episode: int):
    """(장애물 점군 (N,3), action (T,4,4), intrinsic (3,3), camera_extrinsic (4,4))"""
    scene_dir = Path(data_root) / scene
    obstacle = load_gt_obstacle_points(scene_dir)

    table = pq.read_table(scene_dir / 'data' / 'chunk-000' / f'episode_{episode:06d}.parquet')
    action = np.stack([np.asarray(a, dtype=np.float64).reshape(4, 4) for a in table['action'].to_pylist()])
    k = np.asarray(table['observation.camera_intrinsic'].to_pylist()[0], dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(table['observation.camera_extrinsic'].to_pylist()[0], dtype=np.float64).reshape(4, 4)
    return obstacle, action, k, extrinsic


def floor_z_from_extrinsic(cam_z: float, h_b: float) -> float:
    """바닥 world z = `cam_z - h_b`. `h_b`는 `camera_extrinsic`에서 그대로 읽는다.

    **mesh에서 추정하지 않는다** — 이전 버전은 카메라 주변 mesh 정점에서 바닥을 추정했는데
    러그/문턱/가구를 주워 0.06~0.40 m 틀렸다.
    """
    return float(cam_z - h_b)


def check_gt_extent(obstacle: np.ndarray, cam_xyz: np.ndarray, mesh_bounds: np.ndarray,
                     floor_z: float, pad_m: float = 0.5) -> dict:
    """검증 ④ 장애물 점군·카메라 궤적이 mesh bounds 안에 있고, 카메라 높이가 합리적인가.

    `floor_z`는 mesh 추정이 아니라 `cam_z - camera_extrinsic[2,3]`(= h_b)에서 온다
    (`floor_z_from_extrinsic` 참고). 따라서 여기서 나오는 높이는 정의상 `h_b`와 같다.
    """
    lo, hi = mesh_bounds[0] - pad_m, mesh_bounds[1] + pad_m

    def frac_inside(pts):
        return float(np.all((pts >= lo) & (pts <= hi), axis=1).mean()) if len(pts) else 0.0

    obstacle_frac = frac_inside(obstacle)
    traj_frac = frac_inside(cam_xyz)
    cam_z = float(np.median(cam_xyz[:, 2]))
    cam_h = cam_z - floor_z if np.isfinite(floor_z) else float('nan')
    # 사람/로봇 카메라로 납득 가능한 높이 — 스케일이 100배 틀리면 여기서도 걸린다.
    height_sane = bool(np.isfinite(cam_h) and 0.2 < cam_h < 2.5)
    return {
        'obstacle_points': int(len(obstacle)),
        'obstacle_inside_frac': obstacle_frac,
        'trajectory_inside_frac': traj_frac,
        'floor_z_under_camera': floor_z,
        'gt_camera_z': cam_z,
        'gt_camera_height_m': cam_h,
        'camera_height_sane': height_sane,
        'pad_m': pad_m,
        'passed': bool(obstacle_frac > 0.99 and traj_frac > 0.99 and height_sane),
    }


# ---------------------------------------------------------------------------
# 6. 시각화 — mesh floorplan 위에 GT를 겹쳐 본다 (전부 같은 grid)
# ---------------------------------------------------------------------------

def build_floorplan_grid(mesh_bounds: np.ndarray, cell_m: float = FLOORPLAN_CELL_M) -> dict:
    x_min, y_min = float(mesh_bounds[0][0]), float(mesh_bounds[0][1])
    w = max(2, int((mesh_bounds[1][0] - x_min) / cell_m) + 1)
    h = max(2, int((mesh_bounds[1][1] - y_min) / cell_m) + 1)
    return {'x_min': x_min, 'y_min': y_min, 'w': w, 'h': h, 'cell_m': cell_m}


def _scatter(canvas: np.ndarray, pts: np.ndarray, grid: dict, color, radius: int = 0) -> np.ndarray:
    """world 점을 floorplan canvas에 찍는다 (grid 밖은 버림)."""
    out = canvas.copy()
    if len(pts) == 0:
        return out
    x_max = grid['x_min'] + grid['w'] * grid['cell_m']
    y_max = grid['y_min'] + grid['h'] * grid['cell_m']
    inside = ((pts[:, 0] >= grid['x_min']) & (pts[:, 0] < x_max) &
              (pts[:, 1] >= grid['y_min']) & (pts[:, 1] < y_max))
    if not inside.any():
        return out
    xi, yi = _world_xy_to_grid_idx(pts[inside, 0], pts[inside, 1], grid)
    if radius == 0:
        out[yi, xi] = color
    else:
        for x, y in zip(xi, yi):
            cv2.circle(out, (int(x), int(y)), radius, color, -1)
    return out


def render_floorplan_states(mesh, grid: dict, obstacle: np.ndarray, cam_xyz: np.ndarray,
                             depth_world: np.ndarray, out_dir: Path):
    """blink용 4개 state를 만든다 — 전부 **같은 grid**라 화살표를 넘겨도 화면이 안 튄다."""
    verts = np.asarray(mesh.vertices)
    band = verts[(verts[:, 2] > FLOORPLAN_Z_BAND_M[0]) & (verts[:, 2] < FLOORPLAN_Z_BAND_M[1])]

    base = np.full((grid['h'], grid['w'], 3), 38, np.uint8)
    base = _scatter(base, band, grid, (170, 170, 170))

    upscale = max(1, int(np.ceil(FLOORPLAN_MIN_PX / max(grid['w'], grid['h']))))

    def emit(img, name):
        # nearest 확대 — 작은 씬도 리포트에서 형태가 보여야 한다.
        big = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST)
        return save_jpg(big, out_dir / name)

    band_lo, band_hi = FLOORPLAN_Z_BAND_M
    dw = depth_world[(depth_world[:, 2] > band_lo) & (depth_world[:, 2] < band_hi)]
    return [
        ('① mesh 실루엣만 (회색, 높이 0.15~2.0m 단면)', emit(base, 'floorplan_mesh.jpg')),
        ('② mesh + GT 장애물 점군 (주황)', emit(_scatter(base, obstacle, grid, (255, 150, 40)), 'floorplan_obstacle.jpg')),
        ('③ mesh + GT 카메라 궤적 (노랑)', emit(_scatter(base, cam_xyz, grid, (255, 230, 60), radius=1), 'floorplan_traj.jpg')),
        ('④ mesh + depth→world 점 (초록) — 벽 위에 얹혀야 정합', emit(_scatter(base, dw, grid, (0, 255, 90)), 'floorplan_depth.jpg')),
    ]


def render_distance_histogram(distances: np.ndarray, out_dir: Path, width: int = 720, height: int = 300) -> Path:
    """mesh 표면거리 분포 (로그 스케일 x축) — 정답이면 전부 왼쪽 끝에 몰린다."""
    img = np.full((height, width, 3), 38, np.uint8)
    d = distances[np.isfinite(distances)]
    d = np.maximum(d, 1e-6)
    edges = np.logspace(-6, 1, 60)
    counts, _ = np.histogram(d, bins=edges)
    if counts.max() > 0:
        bar_w = max(1, width // len(counts))
        for i, c in enumerate(counts):
            bh = int((c / counts.max()) * (height - 40))
            x0 = i * bar_w
            color = (0, 255, 90) if edges[i] < MESH_ANCHOR_TOL_M else (226, 114, 91)
            cv2.rectangle(img, (x0, height - 20 - bh), (x0 + bar_w - 1, height - 20), color, -1)
    tol_x = int((np.log10(MESH_ANCHOR_TOL_M) + 6) / 7 * width)
    cv2.line(img, (tol_x, 0), (tol_x, height - 20), (224, 179, 77), 1)
    cv2.putText(img, '1um', (2, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(img, f'tol={MESH_ANCHOR_TOL_M}m', (max(0, tol_x - 60), 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (224, 179, 77), 1)
    cv2.putText(img, '10m', (width - 40, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return save_jpg(img, out_dir / 'mesh_distance_hist.jpg')


# ---------------------------------------------------------------------------
# 7. 리포트
# ---------------------------------------------------------------------------

def pill(ok: bool, label_ok: str = 'PASS', label_bad: str = 'FAIL') -> str:
    return f'<span class="pill {"good" if ok else "bad"}">{label_ok if ok else label_bad}</span>'


def build_summary_html(scene: str, checks: dict, meta: dict) -> str:
    a = checks['alignment']
    b = checks['bounds']
    g = checks['gt_extent']
    rows = ''.join(
        f'<tr><td>episode {ep}</td><td>{v["median_m"]:.6f}</td><td>{v["p90_m"]:.6f}</td>'
        f'<td>{pill(v["passed"], "OK", "FAIL")}</td></tr>'
        for ep, v in a['per_episode'].items())
    bounds_rows = ''.join(
        f'<tr><td>{lbl}</td><td>{v[0]:.3f} ~ {v2[0]:.3f}</td><td>{v[1]:.3f} ~ {v2[1]:.3f}</td>'
        f'<td>{v[2]:.3f} ~ {v2[2]:.3f}</td></tr>'
        for lbl, v, v2 in [('obj mesh', meta['bounds_min'], meta['bounds_max'])])
    return f'''
<div class="stat-row">
  <div class="stat"><b>전체</b>{pill(checks["passed"])}</div>
  <div class="stat"><b>mesh 표면거리 (worst)</b><span class="pill {"good" if a["passed"] else "bad"}">
      {a["worst_episode_median_m"]:.6f} m</span></div>
  <div class="stat"><b>허용치</b><span class="pill">{a["tolerance_m"]} m</span></div>
  <div class="stat"><b>pose 규약</b><span class="pill">{a["pose_convention"]}</span></div>
</div>
<p>이 스크립트는 <b>씬 정합 게이트</b>다 — mesh를 정규화하거나 USD를 만들지 않고,
이 씬의 mesh가 GT와 같은 world 프레임인지만 판정해 02로 넘길지 결정한다.
판정 근거는 photometric error(상대 지표, 축 오류가 상쇄됨)가 아니라
<b>GT depth를 world로 올렸을 때 mesh 표면까지의 거리</b>(절대 앵커)다.</p>
<table class="table">
  <tr><th>검증</th><th>결과</th><th>수치</th></tr>
  <tr><td>① obj bounds ↔ fixed.usd bounds</td><td>{pill(b["bounds_match"])}</td>
      <td>max diff {b["max_abs_diff_m"]:.2e} m (tol {b["tolerance_m"]})</td></tr>
  <tr><td>② up-axis Z + 스케일 자릿수</td><td>{pill(b["usd_up_axis_z"] and b["scale_sane"])}</td>
      <td>extent {b["extents_m"][0]:.2f} x {b["extents_m"][1]:.2f} x {b["extents_m"][2]:.2f} m
          (허용 {b["expected_extent_range_m"][0]}~{b["expected_extent_range_m"][1]} m; 정밀 스케일 근거는 ③)</td></tr>
  <tr><td>③ <b>mesh 표면거리 &lt; {a["tolerance_m"]} m</b></td><td>{pill(a["passed"])}</td>
      <td>median {a["median_m"]:.6f} m / worst {a["worst_episode_median_m"]:.6f} m</td></tr>
  <tr><td>④ GT가 mesh bounds 안 + 카메라 높이</td><td>{pill(g["passed"])}</td>
      <td>장애물 {g["obstacle_inside_frac"]*100:.1f}% · 궤적 {g["trajectory_inside_frac"]*100:.1f}% ·
          카메라 z={g["gt_camera_z"]:.3f} m, 바닥 z={g["floor_z_under_camera"]:.3f} m (= cam_z − h_b)
          → 높이 {g["gt_camera_height_m"]:.3f} m</td></tr>
</table>
<table class="table"><tr><th>에피소드</th><th>median (m)</th><th>p90 (m)</th><th></th></tr>{rows}</table>
<table class="table"><tr><th>bounds</th><th>X</th><th>Y</th><th>Z</th></tr>{bounds_rows}</table>
'''


# ---------------------------------------------------------------------------
# 8. main
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_argparser().parse_args()
    print(f'[{SCRIPT_NAME}] scene={args.scene} convention={args.pose_convention}')

    n_eps = count_episodes(args.data_root, args.scene)
    if n_eps == 0:
        print(f'  [ERROR] no episodes under {Path(args.data_root) / args.scene}')
        return 2
    print(f'  episodes available: {n_eps}')

    # --- 자산 로드 ---
    mesh_path = find_scene_mesh(args.mesh_root, args.scene)
    mesh = load_scene_mesh(args.mesh_root, args.scene)
    mesh_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    print(f'  mesh: {mesh_path.name}  verts={len(mesh.vertices)}  '
          f'bounds={np.round(mesh_bounds[0], 3).tolist()} ~ {np.round(mesh_bounds[1], 3).tolist()}')

    usd_info = probe_usd(args.usd_root, args.scene)
    if usd_info['found']:
        print(f'  usd : {Path(usd_info["usd_path"]).name}  upAxis={usd_info["up_axis"]}  '
              f'metersPerUnit={usd_info["meters_per_unit"]}')
    else:
        print(f'  [WARN] USD not found: {usd_info["searched"]}')

    # --- 검증 ①② ---
    bounds_check = check_mesh_usd_bounds(mesh_bounds, usd_info)

    # --- 검증 ③ (핵심 게이트) ---
    print(f'  [gate] mesh surface distance ({args.num_episodes} episodes x {args.frames_per_episode} frames) ...')
    alignment = run_alignment_gate(args, n_eps)

    # --- 검증 ④ ---
    obstacle, action, k, extrinsic = load_gt_reference(args.data_root, args.scene, 0)
    cam_xyz = np.stack([action_to_c2w(a, args.pose_convention)[:3, 3] for a in action])
    h_b, pitch_down_deg = decompose_camera_extrinsic(extrinsic)
    floor_z = floor_z_from_extrinsic(float(np.median(cam_xyz[:, 2])), h_b)
    gt_extent = check_gt_extent(obstacle, cam_xyz, mesh_bounds, floor_z)
    print(f'  robot: h_b={h_b:.4f} m  pitch_down={pitch_down_deg:.2f} deg  '
          f'floor_z={floor_z:+.4f} m  (camera_extrinsic에서 직접)')

    passed = bool(bounds_check['bounds_match'] and bounds_check['scale_sane']
                  and bounds_check['usd_up_axis_z'] and alignment['passed'] and gt_extent['passed'])
    checks = {'bounds': bounds_check, 'alignment': alignment, 'gt_extent': gt_extent, 'passed': passed}

    # --- scene_meta.json (canonical) ---
    meta = {
        'scene_id': args.scene,
        'mesh_path': str(mesh_path),
        'usd_path': usd_info.get('usd_path'),
        'usd_docker_path': usd_info.get('usd_docker_path'),
        'bounds_min': mesh_bounds[0].tolist(),
        'bounds_max': mesh_bounds[1].tolist(),
        'up_axis': 'Z',
        'scale': 1.0,
        'usd_stage': {
            'up_axis': usd_info.get('up_axis'),
            'meters_per_unit': usd_info.get('meters_per_unit'),
            'note': 'metersPerUnit 선언이 좌표값과 불일치하지만 InternNav가 scene_scale=(1,1,1)로 로드한다 '
                    '(generate_episode.py:15,76). 그 관례를 따른다 — 04에서 Isaac 렌더러를 쓸 경우 재확인 필요.',
        },
        'floor_z': gt_extent['floor_z_under_camera'],
        'gt_camera_z': gt_extent['gt_camera_z'],
        'gt_robot_params': {
            'h_b_m': h_b, 'pitch_down_deg': pitch_down_deg, 'episode': 0,
            'source': 'observation.camera_extrinsic (decompose_camera_extrinsic)',
            'note': 'h_b·pitch는 에피소드마다 다르다 — 여기 값은 episode 0 기준. '
                    '전체 분포는 target_schema.json의 robot_params 참고.',
        },
        'gt_camera_height_m': gt_extent['gt_camera_height_m'],
        'pose_convention': args.pose_convention,
        'frame_alignment': alignment,
        'gt_extent_check': gt_extent,
        'bounds_check': bounds_check,
        'passed': passed,
        'note': 'mesh 정규화/USD 생성은 하지 않는다 — 이미 Z-up·1.0=1m·GT와 동일 원점이고 USD도 존재한다. '
                '이 파일은 02(bounds/mesh_path)·03(floor_z)·04(usd_path/카메라 높이)가 읽는다.',
    }
    meta_path = Path(args.out_dir) / 'scene_meta' / f'{args.scene}.json'
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'  scene_meta -> {meta_path}')

    # --- 시각화 ---
    log_dir = Path(args.log_dir) / SCRIPT_NAME / args.scene
    grid = build_floorplan_grid(mesh_bounds)
    depth_dir = Path(args.data_root) / args.scene / 'videos' / 'chunk-000' / 'observation.images.depth'
    depth_m, valid = load_depth_frame_m(depth_dir, 0, 0, args.max_depth_m)
    depth_world = unproject_to_world_frame(depth_m, k, action[0], args.pose_convention)[valid]

    states = render_floorplan_states(mesh, grid, obstacle, cam_xyz, depth_world, log_dir)
    all_dists = np.concatenate([np.array(list(v['per_frame'].values()))
                                for v in alignment['per_episode'].values()]) if alignment['per_episode'] else np.array([])
    hist_path = render_distance_histogram(all_dists, log_dir)

    body = (f'<h3>mesh floorplan 위에 GT 겹쳐보기 (전부 같은 grid, cell={grid["cell_m"]} m)</h3>'
            '<p class="note">회색이 씬 mesh의 벽·가구 실루엣이다. ‹ ›로 넘기면서 장애물 점군·카메라 궤적·'
            'depth 복원점이 회색 위/사이에 제대로 놓이는지 본다. 네 state 모두 같은 격자라 화면이 튀지 않는다.</p>'
            + blink_widget_html('floorplan', states, title=f'{args.scene} floorplan')
            + reference_button_html('참고 — mesh 표면거리 분포', [('에피소드별 프레임 median 분포 (x축 로그)', hist_path)]))

    report = save_gallery(log_dir, 'report.html', f'01_prepare_scene — {args.scene}',
                          build_summary_html(args.scene, checks, meta), body)
    print(f'  report html -> {report}')

    print(f'  => {"PASS" if passed else "FAIL"} '
          f'(bounds={bounds_check["bounds_match"]} scale={bounds_check["scale_sane"]} '
          f'alignment={alignment["passed"]} gt_extent={gt_extent["passed"]})')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
