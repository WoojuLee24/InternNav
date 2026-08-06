"""M1.0 — 정답지 데이터를 읽어 파이프라인 규약을 확정한다.

    00: vln_n1 데이터 -> "어떻게 읽고 쓰나"  -> target_schema.json      (전체에 1개)
    01: 씬 mesh/USD    -> "이 씬 써도 되나"    -> scene_meta/<scene>.json (씬마다 1개)
둘 다 `check_against_scene_mesh`를 쓰지만 00은 규약 후보 3개 중 고르고(한 번), 01은 그 규약으로
씬을 검사한다(새 씬마다).

[하는 일 3가지]
  1. parquet 컬럼/메타의 dtype·shape을 `target_schema.json`으로 기록
  2. pose 규약 판별 — 후보 3개(cam2world_gl / cam2world / world2cam) 중 하나
  3. `camera_extrinsic`을 전 에피소드 스캔해 로봇 파라미터 분포(`robot_params`) 기록

[판별 근거: photometric error가 아니라 씬 mesh 표면거리]
photometric error는 상대 지표라 축 컨벤션이 틀려도 그럴듯한 값이 나온다(좁은 baseline에서는
정답/오답이 err 1.93 vs 1.91로 구분 불가). `vln_n1`의 depth는 로컬 mesh를 렌더한 것이므로
표면거리로 절대 판정한다 — 정답 0.00003 m vs 오답 0.42 m 이상. `geometry_utils.check_against_scene_mesh`.

주의: `observation.video.*.mp4`는 8bit 손실 재인코딩본이라 메트릭에 쓸 수 없다. 실제 depth는
`observation.images.depth/*.png` (uint16, raw/10000 = m).

기하 코드는 `geometry_utils.py`를 import해서 쓴다(직접 재구현 금지).

[필드 규약: `action` vs `camera_extrinsic`] (헷갈리기 쉬운 두 필드 정리)
  - `action[t]` (T,4,4): 매 프레임 카메라의 **world-frame pose**(camera-to-world). 프레임마다
    바뀌는 실제 궤적. OpenGL/USD 카메라 컨벤션이므로 재투영 전 `action_to_c2w`로 변환한다.
  - `observation.camera_extrinsic` (4,4): world pose가 **아니라** 로봇 base(바닥) 기준 카메라
    마운트 오프셋. 에피소드 내에서는 상수, **에피소드마다 다르다** — NavDP(논문 §3 Robot Model)가
    랜덤화하는 로봇 파라미터 2개를 인코딩한다:
      · `e[2,3]` = h_b (로봇 키 = 바닥~카메라 높이, m). 실측 0.26~1.47, 논문 표기 0.25~1.25.
      · 회전(0행이 `[1,0,0]`인 순수 x축 회전) = 카메라 하향 pitch(deg). 실측 0~30, 논문과 일치.
    `decompose_camera_extrinsic`/`compose_camera_extrinsic`으로 왕복.
  - `action`은 `camera_extrinsic`의 효과가 **이미 구워져(baked-in)** 있는 값이라, 재투영 시
    `camera_extrinsic`을 따로 곱하지 않는다(재투영엔 `action`만 사용). 절대 바닥 높이가 필요하면
    `floor_z = cam_z(action) - h_b(camera_extrinsic)`로 얻는다 — `geometry_utils.py` 셀프체크
    ⑤가 이 둘의 일치(pitch 분해 ↔ action forward 방향)를 assert로 검증한다.

[입력]
  - `<data_root>/<scene>/meta/info.json` — 씬 메타(codebase_version, fps 등)
  - `<data_root>/<scene>/data/chunk-000/episode_XXXXXX.parquet` — 컬럼: `index`(int64),
    `observation.camera_intrinsic`(float32,(3,3)), `observation.camera_extrinsic`(float32,(4,4)),
    `action`(float32,(4,4))
  - `<data_root>/<scene>/videos/chunk-000/observation.images.rgb/episode_XXXXXX_XXX.jpg`
    (uint8, 270x480x3)
  - `<data_root>/<scene>/videos/chunk-000/observation.images.depth/episode_XXXXXX_XXX.png`
    (uint16, raw/10000 = m; invalid 값은 `geometry_utils.DEPTH_INVALID_RAW`)
  - `<data_root>/<scene>/videos/chunk-000/observation.video.{rgb,depth}/episode_XXXXXX.mp4`
    (fps 확인용 참고 자료 — 8bit 손실 재인코딩본이라 메트릭 금지)
  - `--mesh_root/<scene>/matterport_mesh/*/*.obj` (선택) — pose convention 판별용 절대 앵커,
    없으면 mesh 검증을 건너뛰고 photometric error로만 판별

[출력]
  - `<out_dir>/target_schema.json` — 파이프라인 canonical 스키마(전체에 1개, 실행마다 덮어씀).
    `parquet_schema`, `image_streams`, `robot_params`, `pose_convention.detected_convention` 등
  - `<log_dir>/00_inspect_vln_n1/<scene>/episode_XXXXXX/report.html` — 시각 진단 리포트
  - `<log_dir>/00_inspect_vln_n1/<scene>/episode_XXXXXX/diagnostics/**` — blink/BEV 중간 이미지

[실행 예시]
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/00_inspect_vln_n1.py \\
        --scene 17DRP5sb8fy --episode 0
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
    DEPTH_INVALID_RAW,
    DEPTH_SCALE_RAW_TO_M,
    POSE_CONVENTIONS,
    MESH_ANCHOR_TOL_M,
    action_to_c2w,
    check_against_scene_mesh,
    check_alignment_bev,
    check_alignment_fpv,
    find_scene_mesh,
    colorize_depth,
    decompose_camera_extrinsic,
    load_depth_frame_m,
    load_rgb_frame,
    mark_invalid_as_checkerboard,
    save_jpg,
    unproject_to_world_frame,
)
from viz_utils import blink_widget_html, reference_button_html, save_gallery  # noqa: E402

# ---------------------------------------------------------------------------
# 1. 상수
# ---------------------------------------------------------------------------

DEFAULT_DATA_ROOT = 'data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i'
DEFAULT_SCENE = '17DRP5sb8fy'
DEFAULT_OUT_DIR = 'scripts/dataset_converters/gs_vlnpe'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe'
SCRIPT_NAME = '00_inspect_vln_n1'

FIELD_CONST_ATOL = 1e-4
# 'cam2world_gl'이 실측 확정된 정답 — 나머지 둘은 판별 근거로 수치만 남기는 기각 후보다
# (해석은 geometry_utils.action_to_c2w 참고).
CANDIDATES = POSE_CONVENTIONS
DEFAULT_MESH_ROOT = 'data/scene_data/mp3d_n1'
CONFIDENCE_MARGIN_WARN_THRESHOLD = 0.05
BEV_CELL_M = 0.05


# ---------------------------------------------------------------------------
# 2. CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT, help='vln_n1 traj_data 루트 (scene 폴더들의 부모)')
    parser.add_argument('--scene', default=DEFAULT_SCENE, help='씬 ID (예: 17DRP5sb8fy)')
    parser.add_argument('--mesh_root', default=DEFAULT_MESH_ROOT,
                        help='씬 mesh 루트 (<mesh_root>/<scene>/matterport_mesh/*/*.obj). pose convention을 '
                             'photometric error(상대 지표) 대신 mesh 표면거리(절대 앵커)로 확정하는 데 쓴다 — '
                             '없으면 mesh 검증을 건너뛰고 photometric error만으로 판별한다.')
    parser.add_argument('--episode', type=int, default=0, help='에피소드 인덱스')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR, help='target_schema.json 출력 루트 (파이프라인 canonical 파일)')
    parser.add_argument('--log_dir', default=DEFAULT_LOG_DIR,
                        help='시각화(진단 이미지 + report.html) 출력 루트. logs/gs-vlnpe/<script_name>/<scene>/episode_XXXXXX/ 아래에 저장됨')
    parser.add_argument('--num_pairs', type=int, default=5, help='재투영 sanity check에 쓸 프레임 쌍 개수')
    parser.add_argument('--frame_gap', type=int, default=15,
                        help='pose convention "판별"용 프레임 쌍의 간격(t0, t0+frame_gap). 통계적으로 두 candidate를 '
                             '가르는 데 쓰인다 — 인접 프레임(gap=1)은 baseline이 짧아 차이가 잘 안 보인다. 실측: '
                             'gap이 클수록 카메라 이동(반사면·가려짐 등 실제 요인)으로 오차 자체도 커지므로, '
                             '이 값은 "잘 정렬되어 보이는지"를 보여주는 용도가 아니다 — 그건 --visual_gap 참고.')
    parser.add_argument('--visual_gap', type=int, default=3,
                        help='리포트의 blink 비교(육안 정렬 확인)에 쓰는 프레임 쌍 간격. --frame_gap보다 훨씬 '
                             '작게 잡아야 한다 — 판별용 gap을 그대로 보여주면 카메라 이동(반사면/가려짐) 때문에 '
                             '정확한 convention도 안 맞아 보이는 오해를 준다.')
    parser.add_argument('--bev_cell_m', type=float, default=BEV_CELL_M, help='BEV grid의 셀 크기(m)')
    parser.add_argument('--num_robot_param_episodes', type=int, default=0,
                        help='camera_extrinsic(로봇 키/pitch) 스캔에 쓸 에피소드 수. 0이면 전체. '
                             '이 필드는 에피소드마다 달라서 한 개만 보면 "고정 오프셋"으로 오인한다.')
    parser.add_argument('--max_depth_m', type=float, default=10.0, help='유효 depth 최대값(m), 이보다 크면 invalid 취급')
    return parser


# ---------------------------------------------------------------------------
# 3. IO 헬퍼 (parquet/메타 — geometry_utils에 없는, 이 스크립트 전용 파싱)
# ---------------------------------------------------------------------------

def load_info_json(scene_dir: Path) -> dict:
    with open(scene_dir / 'meta' / 'info.json') as f:
        return json.load(f)


def load_episode_parquet(scene_dir: Path, episode: int) -> dict:
    """parquet 1개 에피소드를 읽어 컬럼별 배열로 반환.

    Returns:
        {'index': (T,) int64, 'camera_intrinsic': (T,3,3) float32,
         'camera_extrinsic': (T,4,4) float32, 'action': (T,4,4) float32,
         'num_rows': T}
    """
    path = scene_dir / 'data' / 'chunk-000' / f'episode_{episode:06d}.parquet'
    table = pq.read_table(str(path))
    cols = table.to_pydict()
    index = np.asarray(cols['index'], dtype=np.int64)
    intrinsic = np.asarray(cols['observation.camera_intrinsic'], dtype=np.float32).reshape(-1, 3, 3)
    extrinsic = np.asarray(cols['observation.camera_extrinsic'], dtype=np.float32).reshape(-1, 4, 4)
    action = np.asarray(cols['action'], dtype=np.float32).reshape(-1, 4, 4)
    return {
        'index': index,
        'camera_intrinsic': intrinsic,
        'camera_extrinsic': extrinsic,
        'action': action,
        'num_rows': len(index),
    }


def count_episode_frame_files(dir_path: Path, episode: int, ext: str) -> int:
    if not dir_path.exists():
        return 0
    return len(list(dir_path.glob(f'episode_{episode:06d}_*{ext}')))


def probe_video_fps(mp4_path: Path):
    """컨테이너 메타데이터만 조회. 실패해도 스크립트를 죽이지 않는다(순수 정보용)."""
    if not mp4_path.exists():
        return None
    try:
        cap = cv2.VideoCapture(str(mp4_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps <= 0:
            return None
        return {'fps': float(fps), 'frame_count': int(frame_count)}
    except Exception:
        return None


def render_topdown(points_world: np.ndarray, valid_mask: np.ndarray, cam_pos_world: np.ndarray = None,
                    cell_m: float = 0.05, pad_m: float = 0.5, min_render_px: int = 320) -> np.ndarray:
    """world-frame 포인트를 top-down(X-Y 평면, Z=높이로 색칠)으로 스캐터 렌더링 (단일 프레임, 참고용).

    `geometry_utils.rasterize_rgb_to_grid`(t0/t1 비교용, 실제 색으로 rasterize)와는 목적이
    다르다 — 이 함수는 **높이(Z)로 색칠**해서 카메라 파라미터·depth scale이 그럴듯한지
    한 프레임만으로 빠르게 육안 확인하는 참고용 시각화다(reference 모달에 표시).

    `cam_pos_world`(3,)를 주면 카메라 위치에 십자 마커를 그려 방향 참조점으로 삼는다.
    배경은 중간 회색(값이 낮은/어두운 색의 점도 배경과 구분되도록) — 순수 검정이면 TURBO
    colormap 저채도(어두운 남색) 영역의 점이 배경과 거의 안 보였다(실제 확인된 문제).
    작은 캔버스는 `min_render_px`까지 nearest-neighbor로 확대해 리포트에서 잘 보이게 한다.
    """
    pts = points_world[valid_mask]
    if pts.shape[0] == 0:
        return np.full((10, 10, 3), 60, dtype=np.uint8)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x_min, x_max = x.min() - pad_m, x.max() + pad_m
    y_min, y_max = y.min() - pad_m, y.max() + pad_m
    w = max(2, int((x_max - x_min) / cell_m) + 1)
    h = max(2, int((y_max - y_min) / cell_m) + 1)

    z_lo, z_hi = np.percentile(z, 1), np.percentile(z, 99)
    z_norm = np.clip((z - z_lo) / (z_hi - z_lo + 1e-6), 0, 1)
    colormap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
    colors = cv2.applyColorMap((z_norm * 255).astype(np.uint8), colormap).reshape(-1, 3)

    def world_to_px(px, py):
        xi = np.clip(((px - x_min) / cell_m).astype(np.int64), 0, w - 1)
        yi = np.clip((h - 1 - (py - y_min) / cell_m).astype(np.int64), 0, h - 1)
        return xi, yi

    canvas = np.full((h, w, 3), 60, dtype=np.uint8)  # 중간 회색 배경
    xi, yi = world_to_px(x, y)
    order = np.argsort(z)
    canvas[yi[order], xi[order]] = colors[order]

    if cam_pos_world is not None:
        cxi, cyi = world_to_px(np.array([cam_pos_world[0]]), np.array([cam_pos_world[1]]))
        cx_px, cy_px = int(cxi[0]), int(cyi[0])
        cv2.drawMarker(canvas, (cx_px, cy_px), (255, 255, 255), cv2.MARKER_CROSS, 12, 2)

    long_side = max(h, w)
    if long_side < min_render_px:
        scale = min_render_px / long_side
        canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    return canvas


# ---------------------------------------------------------------------------
# 4. 검증 헬퍼
# ---------------------------------------------------------------------------

def verify_constant_field(arr_per_row: np.ndarray, name: str, atol: float = FIELD_CONST_ATOL):
    ref = arr_per_row[0]
    diffs = np.abs(arr_per_row - ref[None, ...]).reshape(len(arr_per_row), -1).max(axis=1)
    is_constant = bool((diffs <= atol).all())
    num_differing = int((diffs > atol).sum())
    if not is_constant:
        print(f'[WARN] {name} is NOT constant across episode: '
              f'{num_differing}/{len(arr_per_row)} rows differ (max_abs_diff={diffs.max():.6f})')
    return ref, {
        'constant': is_constant,
        'max_abs_diff': float(diffs.max()),
        'num_rows_differing': num_differing,
        'atol': atol,
    }


def choose_frame_pairs(t_eff: int, num_pairs: int, frame_gap: int = 1):
    """등간격으로 (t0, t0+frame_gap) 쌍을 뽑는다.

    frame_gap=1(인접 프레임)은 baseline이 너무 짧아 두 pose convention의 재투영 오차 차이가
    거의 나지 않는다(실측: margin<3%) — 두 후보를 실제로 구분하려면 더 넓은 baseline이 필요하다.
    """
    max_t0 = t_eff - 1 - frame_gap
    if max_t0 < 0:
        raise ValueError(f'episode too short for frame_gap={frame_gap} reprojection check (effective length={t_eff})')
    n = max(1, min(num_pairs, max_t0 + 1))
    t0s = sorted(set(np.linspace(0, max_t0, n).round().astype(int).tolist()))
    return [(t0, t0 + frame_gap) for t0 in t0s]


# ---------------------------------------------------------------------------
# 5. Pose 컨벤션 판별 (핵심 로직) — geometry_utils.check_alignment_fpv를 그대로 재사용
# ---------------------------------------------------------------------------

def run_pose_convention_check(action: np.ndarray, k: np.ndarray, rgb_dir: Path, depth_dir: Path,
                               episode: int, frame_pairs, max_depth_m: float, diag_dir: Path):
    """각 프레임 쌍마다 rgb_t0/depth_t0(참고용)를 한 번씩 저장하고, 후보 컨벤션별로
    `check_alignment_fpv`(**backward-warp vs rgb(t0)**)로 진단 이미지를 만든다.

    Returns:
        per_candidate: {'cam2world': {'pair_errors': [...], 'pair_valid_fracs': [...]}, 'world2cam': {...}}
        pair_records: [{'t0':.., 't1':.., 'rgb_t0_path':.., 'depth_t0_path':..,
                         'candidates': {'cam2world': {'warped_path':.., 'err':.., 'valid_frac':..}, ...}}, ...]
    """
    per_candidate = {c: {'pair_errors': [], 'pair_valid_fracs': []} for c in CANDIDATES}
    pair_records = []

    for t0, t1 in frame_pairs:
        depth_t0_m, valid0 = load_depth_frame_m(depth_dir, episode, t0, max_depth_m)
        rgb_t0 = load_rgb_frame(rgb_dir, episode, t0)
        rgb_t1 = load_rgb_frame(rgb_dir, episode, t1)

        pair_dir = diag_dir / f'pair_{t0:04d}_{t1:04d}'
        rgb_t0_path = save_jpg(rgb_t0, pair_dir / 'rgb_t0.jpg')
        depth_t0_path = save_jpg(colorize_depth(depth_t0_m, valid0), pair_dir / 'depth_t0.jpg')

        pair_record = {
            't0': t0, 't1': t1,
            'rgb_t0_path': rgb_t0_path, 'depth_t0_path': depth_t0_path,
            'candidates': {},
        }

        for convention in CANDIDATES:
            result = check_alignment_fpv(
                rgb_t0, depth_t0_m, rgb_t1, depth_t0_m,  # depth_t1 미사용(backward-warp는 depth_t0만 필요)
                action[t0], action[t1], k, convention,
                visualize=True, out_dir=pair_dir, label_prefix=f'warped_{convention}',
            )
            err, valid_frac = result['metric_value'], result['coverage_frac']
            per_candidate[convention]['pair_errors'].append(err)
            per_candidate[convention]['pair_valid_fracs'].append(valid_frac)
            pair_record['candidates'][convention] = {
                'warped_path': result['viz_paths']['warped'], 'err': err, 'valid_frac': valid_frac,
            }

        pair_records.append(pair_record)

    return per_candidate, pair_records


def scan_robot_params(data_root: str, scene: str, max_episodes: int = 0) -> dict:
    """전 에피소드의 `camera_extrinsic`을 스캔해 로봇 키/카메라 pitch 분포를 기록한다.

    이 필드는 에피소드마다 달라서 한 개만 보면 "고정 마운트 오프셋"으로 오인한다(실제 이력 있음).
    03/04가 이 분포에서 샘플링하므로 스키마에 남긴다.
    """
    files = sorted((Path(data_root) / scene / 'data' / 'chunk-000').glob('episode_*.parquet'))
    if max_episodes > 0:
        files = files[:max_episodes]

    h_bs, pitches, floors, bad = [], [], [], []
    for path in files:
        table = pq.read_table(path, columns=['observation.camera_extrinsic', 'action'])
        e = np.asarray(table['observation.camera_extrinsic'].to_pylist()[0], dtype=np.float64).reshape(4, 4)
        a = np.asarray(table['action'].to_pylist()[0], dtype=np.float64).reshape(4, 4)
        try:
            h_b, pitch = decompose_camera_extrinsic(e)
        except AssertionError:
            bad.append(path.name)
            continue
        h_bs.append(h_b)
        pitches.append(pitch)
        floors.append(action_to_c2w(a, 'cam2world_gl')[2, 3] - h_b)

    if not h_bs:
        return {'episodes_scanned': 0, 'undecomposable_episodes': bad}

    h_bs, pitches, floors = np.array(h_bs), np.array(pitches), np.array(floors)
    return {
        'source': 'observation.camera_extrinsic (per-episode)',
        'episodes_scanned': len(h_bs),
        'undecomposable_episodes': bad,
        'h_b_m': {'min': float(h_bs.min()), 'max': float(h_bs.max()), 'mean': float(h_bs.mean()),
                  'std': float(h_bs.std()), 'paper_range': [0.25, 1.25]},
        'pitch_down_deg': {'min': float(pitches.min()), 'max': float(pitches.max()),
                           'mean': float(pitches.mean()), 'paper_range': [0.0, 30.0]},
        'floor_z': {'median': float(np.median(floors)), 'std': float(floors.std()),
                    'constant': bool(floors.std() < 1e-3), 'derivation': 'cam_z - h_b'},
        'note': 'NavDP가 랜덤화하는 로봇 파라미터(논문 §3 Robot Model). 03은 h_b로 h_obs를 정하고, '
                '04는 이 분포에서 샘플링해 카메라를 배치하며, 05는 compose_camera_extrinsic으로 '
                '같은 규약으로 기록한다. h_b 상한은 실측(≈1.5)이 논문 표기(1.25)보다 크다.',
    }


def run_mesh_anchor_check(mesh_root: Path, data_root: Path, scene: str, episode: int, max_depth_m: float):
    """후보별로 mesh 표면거리를 재서 **절대 기준**으로 pose convention을 판별한다.

    photometric error는 두 관측을 서로 비교하는 상대 지표라 축 컨벤션이 틀려도 그럴듯한 값이
    나온다(좁은 baseline에서는 정답/오답이 err 1.93 vs 1.91로 구분 불가). 반면 mesh는 이 depth를
    실제로 렌더한 원본이므로, 표면거리는 정답이면 ~0, 오답이면 수십 cm로 자릿수가 갈린다.

    Returns: {convention: {'median_m','p90_m','per_frame','passed'}} — mesh가 없으면 None
    """
    if not (Path(mesh_root) / scene).is_dir():
        print(f'[WARN] mesh not found under {Path(mesh_root) / scene} — mesh 앵커 검증을 건너뛰고 '
              f'photometric error만으로 판별한다(권장하지 않음).')
        return None
    results = {}
    for convention in CANDIDATES:
        results[convention] = check_against_scene_mesh(
            data_root, mesh_root, scene, episode=episode,
            convention=convention, max_depth_m=max_depth_m)
    return results


def pick_best_convention(per_candidate: dict, mesh_results: dict = None):
    """pose convention 확정. mesh 앵커 결과가 있으면 **그게 우선**이고, 없을 때만 photometric fallback."""
    summary = {}
    for convention, stats in per_candidate.items():
        errors = [e for e in stats['pair_errors'] if not np.isnan(e)]
        fracs = stats['pair_valid_fracs']
        mean_err = float(np.mean(errors)) if errors else float('nan')
        summary[convention] = {
            'mean_abs_photometric_error': mean_err,
            'mean_valid_pixel_fraction': float(np.mean(fracs)) if fracs else 0.0,
            'per_pair_errors': stats['pair_errors'],
        }
        if mesh_results is not None:
            summary[convention]['mesh_surface_distance_m'] = {
                'median': mesh_results[convention]['median_m'],
                'p90': mesh_results[convention]['p90_m'],
                'passed': mesh_results[convention]['passed'],
            }

    if mesh_results is not None:
        # 절대 앵커: 표면거리가 가장 작은 후보. margin은 2등과의 자릿수 차이로 계산한다
        # (photometric margin과 달리 수십~수만 배로 벌어지므로 1.0에 수렴하는 게 정상).
        ranked = sorted(CANDIDATES, key=lambda c: mesh_results[c]['median_m'])
        detected = ranked[0]
        best, second = mesh_results[ranked[0]]['median_m'], mesh_results[ranked[1]]['median_m']
        margin = float(abs(second - best) / max(second, 1e-12))
        if not mesh_results[detected]['passed']:
            print(f'[WARN] 최우수 후보 {detected}조차 mesh 표면거리 {best:.6f}m로 허용치를 넘는다 '
                  f'— pose 규약이나 depth scale을 다시 볼 것')
        return summary, detected, margin

    valid_conventions = {c: s for c, s in summary.items() if not np.isnan(s['mean_abs_photometric_error'])}
    if not valid_conventions:
        print('[WARN] no candidate produced valid pixels — cannot determine pose convention')
        return summary, None, 0.0

    detected = min(valid_conventions, key=lambda c: valid_conventions[c]['mean_abs_photometric_error'])
    errs = sorted(s['mean_abs_photometric_error'] for s in valid_conventions.values())
    margin = abs(errs[1] - errs[0]) / max(errs[1], 1e-8) if len(errs) >= 2 else 1.0
    if margin < CONFIDENCE_MARGIN_WARN_THRESHOLD:
        print(f'[WARN] pose convention candidates are nearly tied (margin={margin:.3f}) '
              f'— manually inspect diagnostics/ before trusting the result')
    return summary, detected, margin


# ---------------------------------------------------------------------------
# 6. 육안 확인용: pair 구간의 모든 프레임을 보여주는 FPV 시퀀스 + BEV(t0 vs t1) 확인
# ---------------------------------------------------------------------------

def build_pair_step_sequence(t0: int, t1: int, action: np.ndarray, k: np.ndarray, rgb_dir: Path,
                              depth_dir: Path, episode: int, convention: str, max_depth_m: float, pair_dir: Path):
    """t0부터 t1까지 모든 프레임에 대해 두 개의 **독립된** blink 시퀀스를 만든다
    (`check_alignment_fpv`를 스텝마다 호출 — run_pose_convention_check와 동일 함수 재사용).

    - `reconstruction_states`: `[실제 rgb(t0), warp(t0→t0+1), warp(t0→t0+2), ..., warp(t0→t1)]`
      — **전부 t0 그리드**라 화면 위치가 항상 고정된다. 이게 진짜 "정렬 확인"이다.
    - `actual_states`: `[실제 rgb(t0), 실제 rgb(t0+1), ..., 실제 rgb(t1)]`
      — 각 프레임 고유 시점(실제 카메라 영상 그대로), 참고용.

    Returns:
        reconstruction_states, actual_states: [(label, path), ...] 각각
        last_err, last_valid_frac: 마지막 스텝(t0->t1)의 오차/valid 비율
        depth_t0_m, valid0: BEV 비교·top-down 렌더링 등에 재사용
    """
    depth_t0_m, valid0 = load_depth_frame_m(depth_dir, episode, t0, max_depth_m)
    rgb_t0 = load_rgb_frame(rgb_dir, episode, t0)

    rgb_t0_path = save_jpg(rgb_t0, pair_dir / f'rgb_t{t0:04d}.jpg')
    reconstruction_states = [(f'실제 rgb(t{t0}) — 기준', rgb_t0_path)]
    actual_states = [(f'실제 rgb(t{t0})', rgb_t0_path)]

    last_err, last_valid_frac = float('nan'), 0.0
    for tk in range(t0 + 1, t1 + 1):
        rgb_tk = load_rgb_frame(rgb_dir, episode, tk)
        rgb_tk_path = save_jpg(rgb_tk, pair_dir / f'rgb_t{tk:04d}.jpg')

        result = check_alignment_fpv(
            rgb_t0, depth_t0_m, rgb_tk, depth_t0_m,  # depth_t1 미사용(backward-warp는 depth_t0만 필요)
            action[t0], action[tk], k, convention,
            visualize=True, out_dir=pair_dir, label_prefix=f'warped_t{t0:04d}_to_t{tk:04d}_{convention}',
        )
        err, valid_frac = result['metric_value'], result['coverage_frac']

        reconstruction_states.append(
            (f'{convention} warp t{t0}→t{tk} (t{t0} 그리드) — err={err:.2f} valid_frac={valid_frac:.2f}',
             result['viz_paths']['warped']))
        actual_states.append((f'실제 rgb(t{tk})', rgb_tk_path))
        last_err, last_valid_frac = err, valid_frac

    return reconstruction_states, actual_states, last_err, last_valid_frac, depth_t0_m, valid0


# ---------------------------------------------------------------------------
# 7. target_schema.json 빌드
# ---------------------------------------------------------------------------

def build_target_schema(data_root: str, scene: str, episode: int, info: dict, parquet_data: dict,
                         intrinsic_stats: dict, extrinsic_stats: dict, frame_counts: dict,
                         rgb_fps_probe, depth_fps_probe, pose_summary: dict, detected_convention,
                         confidence_margin: float, frame_pairs, mesh_results: dict = None,
                         mesh_root: str = DEFAULT_MESH_ROOT, robot_params: dict = None) -> dict:
    return {
        'source': {
            'data_root': data_root,
            'scene': scene,
            'episode': episode,
            'codebase_version': info.get('codebase_version'),
        },
        'fps': {
            'declared_in_info_json': info.get('fps'),
            'measured_from_rgb_video': rgb_fps_probe,
            'measured_from_depth_video': depth_fps_probe,
            'note': 'mp4는 손실 재인코딩본이며 fps가 info.json 선언값과 다를 수 있다; '
                    '실제 데이터 rate는 declared fps를 신뢰한다.',
        },
        'frame_counts': frame_counts,
        'parquet_schema': {
            'index': {'dtype': 'int64', 'shape': 'scalar'},
            'observation.camera_intrinsic': {
                'dtype': 'float32', 'shape': [3, 3],
                'constant_across_episode': intrinsic_stats['constant'],
                'max_abs_diff': intrinsic_stats['max_abs_diff'],
                'row0_value': parquet_data['camera_intrinsic'][0].tolist(),
            },
            'observation.camera_extrinsic': {
                'dtype': 'float32', 'shape': [4, 4],
                'constant_across_episode': extrinsic_stats['constant'],
                'constant_across_dataset': False,
                'max_abs_diff': extrinsic_stats['max_abs_diff'],
                'row0_value': parquet_data['camera_extrinsic'][0].tolist(),
                'note': '에피소드 내에서는 상수지만 **에피소드마다 다르다**(robot_params 필드 참고). '
                        'translation z = 로봇 키 h_b, 회전 = 카메라 하향 pitch — NavDP가 랜덤화하는 '
                        '로봇 파라미터 2개를 담고 있다. `geometry_utils.decompose_camera_extrinsic`/'
                        '`compose_camera_extrinsic`으로 다룰 것. per-frame world pose는 `action`이고 '
                        '이 필드는 재투영에 쓰지 않는다. 바닥 높이는 floor_z = cam_z - h_b로 얻는다.',
            },
            'action': {
                'dtype': 'float32', 'shape': [4, 4],
                'constant_across_episode': False, 'per_frame': True,
                'note': '매 프레임 카메라 pose(camera-to-world). 정확한 convention은 pose_convention 필드 참고.',
            },
        },
        'image_streams': {
            'rgb': {
                'path_pattern': 'videos/chunk-000/observation.images.rgb/episode_{episode:06d}_{frame:03d}.jpg',
                'dtype': 'uint8', 'shape': [270, 480, 3],
            },
            'depth': {
                'path_pattern': 'videos/chunk-000/observation.images.depth/episode_{episode:06d}_{frame:03d}.png',
                'dtype': 'uint16', 'shape': [270, 480], 'units': '0.1mm (raw uint16 / 10000 = meters)',
                'scale_to_meters': DEPTH_SCALE_RAW_TO_M,
                'invalid_raw_values': sorted(DEPTH_INVALID_RAW),
            },
            'rgb_video': {
                'path_pattern': 'videos/chunk-000/observation.video.rgb/episode_{episode:06d}.mp4',
                'note': '시각화 전용 손실 재인코딩본(해상도 패딩됨) — 메트릭 용도로 사용 금지, rgb 원본은 observation.images.rgb 사용.',
            },
            'depth_video': {
                'path_pattern': 'videos/chunk-000/observation.video.depth/episode_{episode:06d}.mp4',
                'note': '8bit 손실 시각화본 — 재투영/메트릭 depth에 사용 금지, 실제 depth는 observation.images.depth 사용.',
            },
        },
        'robot_params': robot_params,
        'pose_convention': {
            'candidates_tested': list(CANDIDATES),
            'frame_pairs_tested': [list(p) for p in frame_pairs],
            'per_candidate': pose_summary,
            'detected_convention': detected_convention,
            'confidence_margin': confidence_margin,
            'decided_by': 'scene_mesh_surface_distance' if mesh_results is not None else 'photometric_error',
            'mesh_anchor': None if mesh_results is None else {
                'mesh_path': str(find_scene_mesh(mesh_root, scene)),
                'tolerance_m': MESH_ANCHOR_TOL_M,
                'per_candidate_median_m': {c: mesh_results[c]['median_m'] for c in CANDIDATES},
                'per_frame_median_m': mesh_results[detected_convention]['per_frame'],
            },
            'evidence_note':
                'action[t]는 per-frame 카메라 pose이고, camera_extrinsic(상수)은 고정 마운트 오프셋이라 '
                '재투영에서 제외한다. **축 컨벤션**: action[t]의 회전은 OpenGL/USD 카메라 프레임'
                '(x=right, y=up, z=backward) 기준이라, OpenCV(x=right, y=down, z=forward)로 unproject한 '
                '점에 쓰려면 `action[t] @ CAM_CV_TO_GL`이 필요하다(= convention "cam2world_gl", '
                '`geometry_utils.action_to_c2w`). 이 flip을 빼면 씬 mesh 표면까지 0.27m 어긋난다. '
                'relative transform은 `geometry_utils.compute_relative_transform`(inv(t1) @ t0 순서)을 사용. '
                '**판별 근거**: photometric error는 상대 지표라 축 컨벤션 오류를 상쇄해 버린다'
                '(좁은 baseline에서 정답/오답이 err 1.93 vs 1.91로 구분 불가) — 그래서 정답지 mesh를 '
                '절대 앵커로 삼아 확정했다(mesh_anchor 필드).',
        },
        'notes': [
            'depth 재투영은 프레임별 PNG(observation.images.depth)만 사용한다; video 스트림(mp4)은 손실 '
            '시각화본이라 메트릭 계산에서 제외한다.',
            'FPV warp 방향: rgb(t1)을 depth(t0) 기반 좌표로 backward-warp(bilinear)해 rgb(t0)와 비교했다 '
            '(forward-splat 대신 — 홀/z-buffer 로직 불필요, 동일한 판별 근거).',
            'BEV 교차검증: t0/t1을 각자 독립적으로 절대 world 프레임에 unproject해 같은 grid에 올려 겹치는 '
            '영역을 비교 (`geometry_utils.check_alignment_bev`). FPV는 상대 pose만 쓰므로 축 컨벤션 오류를 '
            '상쇄하지만 BEV는 절대 pose를 쓰므로 그대로 드러난다 — 두 검증은 서로 대체 불가.',
            'world 프레임은 씬 mesh(data/scene_data/mp3d_n1/<scene>/matterport_mesh/*.obj)의 좌표계와 '
            '정확히 같다(표면거리 median 0.00003 m). 02/03이 mesh 좌표계에서 만든 경로와 04가 기록할 '
            'pose가 같은 프레임이라는 뜻 — 별도 정합(registration) 불필요.',
        ],
    }


# ---------------------------------------------------------------------------
# 8. main
# ---------------------------------------------------------------------------

def main():
    args = build_argparser().parse_args()

    scene_dir = Path(args.data_root) / args.scene
    rgb_dir = scene_dir / 'videos' / 'chunk-000' / 'observation.images.rgb'
    depth_dir = scene_dir / 'videos' / 'chunk-000' / 'observation.images.depth'
    rgb_video_path = scene_dir / 'videos' / 'chunk-000' / 'observation.video.rgb' / f'episode_{args.episode:06d}.mp4'
    depth_video_path = scene_dir / 'videos' / 'chunk-000' / 'observation.video.depth' / f'episode_{args.episode:06d}.mp4'

    out_dir = Path(args.out_dir)
    run_log_dir = Path(args.log_dir) / SCRIPT_NAME / args.scene / f'episode_{args.episode:06d}'
    diag_dir = run_log_dir / 'diagnostics'

    print(f'[00_inspect_vln_n1] scene={args.scene} episode={args.episode}')
    print(f'  scene_dir = {scene_dir}')

    info = load_info_json(scene_dir)
    parquet_data = load_episode_parquet(scene_dir, args.episode)

    n_rgb = count_episode_frame_files(rgb_dir, args.episode, '.jpg')
    n_depth = count_episode_frame_files(depth_dir, args.episode, '.png')
    n_parquet = parquet_data['num_rows']
    consistent = n_rgb == n_depth == n_parquet
    t_eff = min(n_parquet, n_rgb, n_depth)
    frame_counts = {
        'parquet_rows': n_parquet, 'rgb_image_files': n_rgb, 'depth_image_files': n_depth,
        'consistent': consistent, 'effective_length_used': t_eff,
    }
    print(f'  frame counts: parquet={n_parquet} rgb={n_rgb} depth={n_depth} '
          f'({"OK" if consistent else "MISMATCH -> using min=" + str(t_eff)})')

    intrinsic_ref, intrinsic_stats = verify_constant_field(parquet_data['camera_intrinsic'], 'camera_intrinsic')
    _, extrinsic_stats = verify_constant_field(parquet_data['camera_extrinsic'], 'camera_extrinsic')

    # 1) 판별용: 넓은 gap으로 두 candidate를 통계적으로 비교(margin 확보용, 시각화용 아님)
    frame_pairs = choose_frame_pairs(t_eff, args.num_pairs, args.frame_gap)
    print(f'  [detection] testing {len(frame_pairs)} frame pairs (gap={args.frame_gap}): {frame_pairs}')

    per_candidate_raw, _detection_pair_records = run_pose_convention_check(
        parquet_data['action'], intrinsic_ref, rgb_dir, depth_dir, args.episode,
        frame_pairs, args.max_depth_m, diag_dir / f'detection_gap{args.frame_gap}',
    )
    # 0-b) camera_extrinsic 스캔 — 에피소드마다 다른 로봇 키/pitch를 기록한다.
    robot_params = scan_robot_params(args.data_root, args.scene, args.num_robot_param_episodes)
    if robot_params.get('episodes_scanned'):
        rp_h, rp_p = robot_params['h_b_m'], robot_params['pitch_down_deg']
        print(f'  [robot params] {robot_params["episodes_scanned"]} episodes: '
              f'h_b {rp_h["min"]:.3f}~{rp_h["max"]:.3f} (mean {rp_h["mean"]:.3f}), '
              f'pitch_down {rp_p["min"]:.2f}~{rp_p["max"]:.2f} deg, '
              f'floor_z {robot_params["floor_z"]["median"]:+.3f} '
              f'(constant={robot_params["floor_z"]["constant"]})')
        if robot_params['undecomposable_episodes']:
            print(f'  [WARN] camera_extrinsic 분해 실패: {robot_params["undecomposable_episodes"]}')

    # 1-b) 절대 앵커: 씬 mesh 표면거리로 확정한다. photometric error(상대 지표)는 축 컨벤션 오류를
    #      상쇄해 버려서 판별 근거로 부족하다 — 실제로 이걸로 한 차례 오판했다(docstring ③ 참고).
    print(f'  [mesh anchor] measuring depth->world distance to scene mesh ({args.mesh_root}/{args.scene}) ...')
    mesh_results = run_mesh_anchor_check(args.mesh_root, args.data_root, args.scene, args.episode, args.max_depth_m)

    pose_summary, detected_convention, confidence_margin = pick_best_convention(per_candidate_raw, mesh_results)

    for convention, stats in pose_summary.items():
        mesh_txt = ''
        if 'mesh_surface_distance_m' in stats:
            mesh_txt = f' mesh_dist_median={stats["mesh_surface_distance_m"]["median"]:.6f}m'
        print(f'  [{convention}] mean_err={stats["mean_abs_photometric_error"]:.3f} '
              f'valid_frac={stats["mean_valid_pixel_fraction"]:.3f}{mesh_txt}')
    decided_by = 'mesh surface distance' if mesh_results is not None else 'photometric error (mesh 없음)'
    print(f'  => detected_convention = {detected_convention} '
          f'(confidence_margin={confidence_margin:.3f}, decided by {decided_by})')

    # 2) 육안 확인용: 좁은 gap, pair 구간의 "모든 프레임"을 하나도 건너뛰지 않고 순서대로 보여준다
    #    + 각 pair의 (t0, t1) 양끝에 대해 BEV(월드 프레임) 교차검증도 함께 수행.
    visual_pairs = choose_frame_pairs(t_eff, args.num_pairs, args.visual_gap)
    print(f'  [visual] testing {len(visual_pairs)} frame pairs (gap={args.visual_gap}, 모든 중간 프레임 표시): '
          f'{visual_pairs}')

    visual_dir = diag_dir / f'visual_gap{args.visual_gap}'
    pair_sections = []
    visual_errs, visual_vfs = [], []
    bev_errs, bev_coverages = [], []
    for t0, t1 in visual_pairs:
        pair_dir = visual_dir / f'pair_{t0:04d}_{t1:04d}'
        reconstruction_states, actual_states, last_err, last_vf, depth_t0_m, valid0 = build_pair_step_sequence(
            t0, t1, parquet_data['action'], intrinsic_ref, rgb_dir, depth_dir, args.episode,
            detected_convention, args.max_depth_m, pair_dir,
        )
        visual_errs.append(last_err)
        visual_vfs.append(last_vf)

        # depth scale·camera parameter 참고용 top-down 뷰(높이로 색칠, 단일 프레임).
        depth_t0_path = save_jpg(colorize_depth(depth_t0_m, valid0), pair_dir / 'depth_t0.jpg')
        world_pts_t0 = unproject_to_world_frame(depth_t0_m, intrinsic_ref, parquet_data['action'][t0])
        cam_pos_world = parquet_data['action'][t0][:3, 3]
        topdown_path = save_jpg(render_topdown(world_pts_t0, valid0, cam_pos_world), pair_dir / 'topdown_t0.jpg')

        # BEV(월드 프레임) 교차검증: t0/t1을 각자 world로 unproject해 같은 grid에서 비교.
        # FPV(카메라 프레임 재투영)와 입력/출력 형식이 통일된 별도 함수(check_alignment_bev) 사용.
        rgb_t0 = load_rgb_frame(rgb_dir, args.episode, t0)
        rgb_t1 = load_rgb_frame(rgb_dir, args.episode, t1)
        depth_t1_m, valid1 = load_depth_frame_m(depth_dir, args.episode, t1, args.max_depth_m)
        bev_result = check_alignment_bev(
            rgb_t0, depth_t0_m, rgb_t1, depth_t1_m,
            parquet_data['action'][t0], parquet_data['action'][t1], intrinsic_ref, detected_convention,
            cell_m=args.bev_cell_m, visualize=True, out_dir=pair_dir, label_prefix='bev',
        )
        bev_errs.append(bev_result['metric_value'])
        bev_coverages.append(bev_result['coverage_frac'])

        ref_entries = [
            ('depth(t0), colorized', depth_t0_path),
            ('top-down point cloud (높이로 색칠, 참고용)', topdown_path),
        ]
        ref_btn = reference_button_html('reference ↗', ref_entries)

        # blink 세 개: ① 재구성(t0 그리드) ② 실제 촬영(참고) ③ BEV(월드 그리드, t0 vs t1)
        recon_widget = blink_widget_html(
            f'blink_recon_{t0}_{t1}', reconstruction_states,
            title=f'① 재구성 vs 원본 (전부 t{t0} 그리드) — {detected_convention} '
                  f'<span class="pill good">selected</span>')
        actual_widget = blink_widget_html(
            f'blink_actual_{t0}_{t1}', actual_states,
            title='② 실제 촬영 프레임 (참고용 — 프레임마다 고유 시점, 위 재구성과 그리드가 다름)')
        bev_states = [
            (f'BEV(t{t0}) — world grid', bev_result['viz_paths']['bev_t0']),
            (f'BEV(t{t1}) — world grid, overlap_err={bev_result["metric_value"]:.2f} '
             f'coverage={bev_result["coverage_frac"]:.2f}', bev_result['viz_paths']['bev_t1']),
        ]
        bev_widget = blink_widget_html(
            f'blink_bev_{t0}_{t1}', bev_states,
            title='③ BEV 교차검증 (t0 vs t1, 같은 world grid — 카메라 프레임이 아니라 월드 프레임 일치성)')

        pair_sections.append(f'''
        <div class="pair-section">
          <div class="pair-head"><h2>t{t0} &rarr; t{t1}</h2>{ref_btn}</div>
          {recon_widget}
          <div style="height:12px"></div>
          {actual_widget}
          <div style="height:12px"></div>
          {bev_widget}
        </div>''')

    visual_err = float(np.mean(visual_errs))
    visual_vf = float(np.mean(visual_vfs))
    bev_err = float(np.nanmean(bev_errs))
    bev_coverage = float(np.mean(bev_coverages))
    print(f'  [visual@{detected_convention}] mean_err={visual_err:.3f} valid_frac={visual_vf:.3f}')
    print(f'  [bev@{detected_convention}] mean_overlap_err={bev_err:.3f} mean_coverage={bev_coverage:.3f}')

    rgb_fps_probe = probe_video_fps(rgb_video_path)
    depth_fps_probe = probe_video_fps(depth_video_path)

    schema = build_target_schema(
        args.data_root, args.scene, args.episode, info, parquet_data,
        intrinsic_stats, extrinsic_stats, frame_counts, rgb_fps_probe, depth_fps_probe,
        pose_summary, detected_convention, confidence_margin, frame_pairs,
        mesh_results=mesh_results, mesh_root=args.mesh_root, robot_params=robot_params,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    schema_path = out_dir / 'target_schema.json'
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f'  target_schema.json -> {schema_path}')

    margin_pill_cls = 'good' if confidence_margin >= CONFIDENCE_MARGIN_WARN_THRESHOLD else 'warn'
    consistent_pill_cls = 'good' if consistent else 'bad'
    selected_badge = '<span class="pill good">selected</span>'
    summary_rows = ''.join(
        f'<tr><td>{c}</td><td>{s["mean_abs_photometric_error"]:.2f}</td>'
        f'<td>{s["mean_valid_pixel_fraction"]:.3f}</td>'
        f'<td>{selected_badge if c == detected_convention else ""}</td></tr>'
        for c, s in pose_summary.items()
    )
    summary_html = (
        f'<div class="stat-row">'
        f'<div class="stat"><div class="label">scene / episode</div>'
        f'<div class="value">{args.scene} / {args.episode}</div></div>'
        f'<div class="stat"><div class="label">detection gap (판별용)</div><div class="value">{args.frame_gap}</div></div>'
        f'<div class="stat"><div class="label">visual gap (아래 blink용)</div><div class="value">{args.visual_gap}</div></div>'
        f'<div class="stat"><div class="label">frame counts</div>'
        f'<div class="value"><span class="pill {consistent_pill_cls}">'
        f'{"consistent" if consistent else "MISMATCH"}</span></div></div>'
        f'<div class="stat"><div class="label">detected pose convention</div>'
        f'<div class="value"><span class="pill good">{detected_convention}</span></div></div>'
        f'<div class="stat"><div class="label">confidence margin</div>'
        f'<div class="value"><span class="pill {margin_pill_cls}">{confidence_margin:.1%}</span></div></div>'
        f'<div class="stat"><div class="label">BEV mean overlap err</div>'
        f'<div class="value">{bev_err:.2f} <span style="color:var(--text-dim)">(coverage {bev_coverage:.2f})</span></div></div>'
        f'</div>'
        f'<table><tr><th>candidate</th><th>mean abs photometric error (gap={args.frame_gap})</th>'
        f'<th>mean valid-pixel fraction</th><th></th></tr>{summary_rows}</table>'
        f'<p style="color:var(--text-dim)">두 후보를 가르는 통계는 baseline이 넓어야(gap={args.frame_gap}) '
        f'뚜렷이 갈린다. 하지만 gap이 클수록 카메라가 실제로 많이 움직여 반사면·가려짐 등으로 <b>정답 convention도 '
        f'오차가 커진다</b>(gap에 비례해 매끄럽게 증가 — 버그 아님). '
        f'그래서 아래 blink 비교는 훨씬 좁은 gap={args.visual_gap}로 따로 만들었다 — '
        f'{detected_convention} 기준(마지막 스텝) 평균 err={visual_err:.2f}, valid_frac={visual_vf:.2f}.</p>'
        f'<p>표에서 판별된 <b>{detected_convention}</b>만 아래 blink 비교에 보여준다(margin이 확보돼 '
        f'{"world2cam" if detected_convention == "cam2world" else "cam2world"}은 이미 기각됨). 각 pair마다 '
        f'blink가 <b>세 개</b> 있다 — ① <b>재구성 vs 원본</b>(카메라 프레임, t0 그리드 고정, `geometry_utils.'
        f'check_alignment_fpv`) ② <b>실제 촬영 프레임</b>(참고용) ③ <b>BEV 교차검증</b>(월드 프레임, t0/t1을 '
        f'각자 world로 unproject해 같은 grid에서 비교 — `geometry_utils.check_alignment_bev`, '
        f'<code>depth_rgb_to_bev_torch.py</code>의 scatter-mean 패턴 참고). FPV/BEV 두 함수는 입력(rgb_t0, '
        f'depth_t0, rgb_t1, depth_t1, pose_t0, pose_t1, k, ..., visualize, out_dir) 및 출력(metric_name/'
        f'metric_value/coverage_frac/viz_paths) 형식을 통일해뒀다. <b>reference</b> 버튼을 누르면 t0의 원본 '
        f'depth와 top-down 포인트클라우드(참고용)를 별도 창(모달)에서 볼 수 있다.</p>'
        f'<p style="color:var(--text-dim)"><b>이전에 "align 안 됨" 리포트가 있었던 이유(수정 완료)</b>: '
        f'실제로 <code>compute_relative_transform</code>에 진짜 버그가 있었다 — t0/t1 순서가 뒤바뀌어 '
        f'있었는데, `cv2.matchTemplate`으로 정량 측정해서 발견·수정했다(자세한 내용은 '
        f'<code>geometry_utils.compute_relative_transform</code> docstring 참고). 수정 후 재투영 오차가 '
        f'5~10배 줄고 pose convention 판별 margin도 크게 개선됐다.</p>'
        f'<p style="color:var(--text-dim)">참고: 체크무늬는 "이 픽셀/셀은 데이터가 없다"는 뜻이다 — 실제 이미지 '
        f'크기는 항상 원본과 동일하며, 체크무늬가 넓을수록 두 시점 사이 겹치는 시야(FPV)/영역(BEV)이 적다는 '
        f'뜻이다.</p>'
    )

    body_html = f'<div class="grid">{"".join(pair_sections)}</div>'

    report_path = save_gallery(
        out_dir=run_log_dir,
        filename='report.html',
        title=f'00_inspect_vln_n1 — {args.scene} ep{args.episode}',
        summary_html=summary_html,
        body_html=body_html,
    )

    print(f'  diagnostics -> {diag_dir}')
    print(f'  report html -> {report_path}')


if __name__ == '__main__':
    main()
