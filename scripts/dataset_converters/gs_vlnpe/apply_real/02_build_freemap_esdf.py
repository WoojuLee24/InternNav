"""M1.2 — 씬 mesh를 occupancy로 만들어 저장하고, 그 맵이 정상인지 검증한다.

    00: vln_n1 데이터 -> "어떻게 읽고 쓰나"  -> target_schema.json      (전체에 1개)
    01: 씬 mesh/USD    -> "이 씬 써도 되나"    -> scene_meta/<scene>.json (씬마다 1개)
    02: 씬 mesh        -> "어디로 갈 수 있나"  -> esdf/<scene>.npz        (씬마다 1개)
    03: occupancy      -> "어떤 경로로 가나"   -> paths/<scene>.json      (씬마다 1개)

계산은 전부 `esdf_utils.py`가 한다. 이 스크립트는 **파일 입출력 + 검증 + 리포트**만 담당한다
(01과 같은 구조 — 중복 구현 금지).

## 처리 단계별 데이터

```
scene_meta/<scene>.json          <- 01 출력: mesh_path / usd_path / floor_z
  | trimesh 로드
mesh (삼각형)                     obj 191,625 verts / 215,757 faces
  | sample_surface(3M) -> 격자 스캐터            esdf_utils.voxelize_surface()
(1) occupancy   (Nx,Ny,Nz) bool   17DRP5sb8fy: 328x166x57, 점유 7.4%   ** npz로 저장 **
  |                                                 --- 여기부터는 h_b에 의존 ---
  | 밴드 (floor_z+h_nav, floor_z+h_b]           esdf_utils.derive_obstacle_2d()
(2) obstacle_2d (Ny,Nx) bool
  | distance_transform_edt * cell               esdf_utils.compute_esdf_2d()
(3) esdf        (Ny,Nx) float32 [m]
  | esdf >= r_b                                 esdf_utils.truncate_navigable()
(4) navigable   (Ny,Nx) bool
```

(2)~(4)는 로봇 키 `h_b`마다 달라지므로 **저장하지 않는다** — 03이 (1)에서 매번 도출한다
(수 ms). npz에는 `--ref_episode`의 `h_b`로 만든 참조본만 시각화·검증용으로 함께 넣는다.

## 입력을 obj로 하는 이유

논문이 raw scene mesh를 썼고(BlenderProc + Matterport3D) GT도 그 mesh에서 렌더됐다.
USD는 표면이 동일하고 collision Plane 2장만 더 있는데 최종 navigable 차이가 0.4%(103셀)뿐이다.
Stage 2(Isaac)로 넘어갈 때 `--geometry usd`로 바꾼다.

## 검증 (하나라도 실패하면 non-zero exit)

  1. **GT 궤적 clearance** — 여러 에피소드에서 각자의 `h_b`로 맵을 만들어 궤적의 clearance를 잰다.
     median이 허용 범위 안이고 최소값이 하한을 넘는지 본다. **유일한 독립 검증.**
     범위로 잡는 이유는 씬마다 값이 다르기 때문이다(6개 씬 실측 0.46~0.68) —
     `esdf_utils.GT_CLEARANCE_MEDIAN_RANGE_M` 주석 참고.
  2. sanity — 장애물 셀 esdf==0, 자유공간 esdf>r_b, occupancy가 bounds 안
  3. `.ply` obstacle 겹침 — **기하 검증이 아니다**(그 점들은 mesh 표면 위 0.0000 m라 순환논증).
     mesh->voxel 변환이 GT 라벨과 같은 곳을 obstacle로 보는지의 정황 확인일 뿐이다.

실행 예시:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/02_build_freemap_esdf.py \\
        --scene 17DRP5sb8fy

negative test (게이트가 항상 통과만 내지 않음을 보장):
    ... --scene 17DRP5sb8fy --h_nav -0.1     # 바닥을 장애물에 포함 -> clearance 0으로 붕괴, exit 1
    ... --scene 17DRP5sb8fy --h_nav 1.0      # 로봇 키보다 큼 -> 설정 불가로 거름, exit 2
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esdf_utils import (  # noqa: E402
    ASTAR_CELL_M,
    GT_CLEARANCE_MEDIAN_RANGE_M,
    GT_CLEARANCE_MIN_M,
    GT_CLEARANCE_TYPICAL_RANGE_M,
    H_NAV_M,
    ROBOT_RADIUS_M,
    SURFACE_SAMPLES,
    VOXEL_SIZE_M,
    compute_esdf_2d,
    derive_obstacle_2d,
    load_scene_usd,
    sample_esdf_at,
    truncate_navigable,
    voxelize_surface,
    world_to_cell,
)
from geometry_utils import (  # noqa: E402
    action_to_c2w,
    decompose_camera_extrinsic,
    load_gt_obstacle_points,
    load_scene_mesh,
    save_jpg,
)
from viz_utils import blink_widget_html, reference_button_html, save_gallery  # noqa: E402
import dataset_utils  # noqa: E402

DEFAULT_DATA_ROOT = 'data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i'
DEFAULT_MESH_ROOT = 'data/scene_data/mp3d_n1'
DEFAULT_OUT_DIR = 'scripts/dataset_converters/gs_vlnpe/apply_real'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe/apply_real'
DEFAULT_SCENE = '17DRP5sb8fy'
SCRIPT_NAME = '02_build_freemap_esdf'

ESDF_VIZ_MAX_M = 1.5          # 히트맵 고정 범위 — 실행 간 색 기준이 같아야 비교가 된다
FLOORPLAN_MIN_PX = 700


_USD_APP = None


def load_usd_mesh_with_app(usd_path: str):
    """Standalone Isaac Python에서 USD API를 활성화하고 프로세스가 끝날 때까지 유지한다."""
    global _USD_APP
    from isaacsim import SimulationApp

    if _USD_APP is None:
        _USD_APP = SimulationApp({'headless': True})
    return load_scene_usd(usd_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', default=DEFAULT_SCENE)
    parser.add_argument('--dataset', default='vln_n1', choices=['vln_n1', 'vln_pe'],
                        help='GT 궤적을 읽을 데이터셋. vln_pe면 출력(npz/json/log)에 _vlnpe 태그가 붙는다')
    parser.add_argument('--geometry', default='obj', choices=['obj', 'usd', 'both'],
                        help='occupancy를 만들 mesh. Stage 1은 obj(논문과 동일), Stage 2(Isaac)는 usd.')
    parser.add_argument('--data_root', default=None,
                        help='traj_data 루트. 생략 시 --dataset의 기본 경로')
    parser.add_argument('--mesh_root', default=DEFAULT_MESH_ROOT)
    parser.add_argument('--cell_m', type=float, default=VOXEL_SIZE_M)
    parser.add_argument('--h_nav', type=float, default=H_NAV_M,
                        help='지면 여유고. 이 아래 점유는 밟고 넘는 지면으로 본다.')
    parser.add_argument('--r_b', type=float, default=ROBOT_RADIUS_M, help='로봇 반경')
    parser.add_argument('--h_nav_ratio', type=float, default=None,
                        help='주면 h_nav = ratio * h_b 로 쓴다(--h_nav 무시). 논문의 "h_nav는 로봇 키에 '
                             '의존" 서술 대응 — 고정 0.10과 재현 정확도가 같아 기본은 고정값이다.')
    parser.add_argument('--grid_align', type=float, default=ASTAR_CELL_M,
                        help='격자 원점을 이 간격의 월드 격자에 스냅한다(내림). 0을 주면 스냅하지 않고 '
                             'mesh.bounds[0]을 그대로 쓴다(이전 동작). 논문은 원점을 언급하지 않는데 '
                             'bounds를 그대로 쓰면 격자 위상이 씬마다 제멋대로가 되고(실측 y로 0.087/'
                             '0.082 m 어긋남) A*가 내는 셀중심 좌표가 그만큼 옮겨진다. 월드 정렬은 '
                             '자유 파라미터가 0개인 씬 독립 규칙이며, s8pcmisQ38h ep17의 루트 불일치가 '
                             '이것으로 해결됐다(같은 루트 39/40 -> 40/40).')
    parser.add_argument('--surface_samples', type=int, default=SURFACE_SAMPLES)
    parser.add_argument('--ref_episode', type=int, default=0,
                        help='npz에 넣을 참조 2D 산출물의 h_b를 가져올 에피소드')
    parser.add_argument('--ref_h_b', type=float, default=None,
                        help='GT가 없는 scene의 참조 2D 산출물에 사용할 고정 h_b [m]')
    parser.add_argument('--num_check_episodes', type=int, default=5,
                        help='GT 궤적 clearance 게이트에 쓸 에피소드 수')
    parser.add_argument('--scene_meta_dir', default=None,
                        help='scene_meta/<scene>.json 을 읽을 루트. 기본값은 --out_dir '
                             '(negative test처럼 출력만 다른 곳에 쓰고 싶을 때 분리해 쓴다)')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    return parser


# ---------------------------------------------------------------------------
# GT 읽기
# ---------------------------------------------------------------------------

def load_gt_episode(data_root: str, scene: str, episode: int, dataset: str = 'vln_n1'):
    """(로봇 몸통 궤적 (T,3) world, h_b, pitch_down, floor_z) — dataset_utils.load_gt_episode의
    body_xyz를 쓴다(vln_pe 카메라는 몸통보다 0.2m 앞이라 회전 시 가짜 경로가 생김 — 그쪽 주석 참고)."""
    ep = dataset_utils.load_gt_episode(data_root, scene, episode, dataset)
    return ep['body_xyz'], ep['h_b'], ep['pitch_deg'], ep['floor_z']


def count_episodes(data_root: str, scene: str) -> int:
    return dataset_utils.count_episodes(data_root, scene)


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

def check_gt_clearance(occupancy, origin, args, n_available: int) -> dict:
    """검증 ① 에피소드마다 그 로봇 키로 맵을 만들어 GT 궤적의 clearance를 잰다.

    맵 계산(높이대 분류·거리 변환·좌표 변환) 어디가 틀어져도 이 숫자가 먼저 흔들린다.
    """
    episodes = list(range(min(args.num_check_episodes, n_available)))
    per_episode, medians, mins = {}, [], []
    for ep in episodes:
        cam_xyz, h_b, _, floor_z = load_gt_episode(args.data_root, args.scene, ep, args.dataset)
        h_nav = args.h_nav_ratio * h_b if args.h_nav_ratio else args.h_nav
        esdf = compute_esdf_2d(derive_obstacle_2d(occupancy, origin, floor_z, h_b,
                                                  args.cell_m, h_nav), args.cell_m)
        clearance = sample_esdf_at(esdf, cam_xyz[:, :2], origin, args.cell_m)
        med, mn = float(np.median(clearance)), float(clearance.min())
        zero_frac = float((clearance < 0.05).mean())
        per_episode[ep] = {'h_b': h_b, 'floor_z': floor_z, 'median_m': med, 'min_m': mn,
                           'zero_frac': zero_frac}
        medians.append(med)
        mins.append(mn)
        print(f'    episode {ep:>3}: h_b={h_b:.3f}  clearance median={med:.3f} min={mn:.3f} m '
              f'zero_frac={zero_frac*100:.1f}%')

    overall_median = float(np.median(medians)) if medians else float('nan')
    worst_min = float(min(mins)) if mins else float('nan')
    worst_zero_frac = float(max(v['zero_frac'] for v in per_episode.values())) if per_episode else float('nan')
    lo, hi = GT_CLEARANCE_MEDIAN_RANGE_M
    t_lo, t_hi = GT_CLEARANCE_TYPICAL_RANGE_M
    if medians and not (t_lo <= overall_median <= t_hi):
        print(f'    [WARN] clearance median {overall_median:.3f} m가 6개 씬 실측 범위 '
              f'{GT_CLEARANCE_TYPICAL_RANGE_M} 밖이다 (판정 기준은 {GT_CLEARANCE_MEDIAN_RANGE_M})')
    if args.dataset == 'vln_pe':
        # vln_n1용 범위 게이트는 vln_pe에 이전 불가 — 실측으로 확인한 두 가지 정당한 차이:
        # ① H1은 h_b~1.66m라 장애물 밴드가 넓어 median 자체가 낮아진다(0.14~0.45 vs vln_n1
        #    0.46~0.68), ② 물리 보행이라 13cm 문턱을 밟고 넘고, 꼭대기층 경사 천장/문틀 아래
        #    (바닥+1.66m 안)를 지나며 zero clearance가 정상적으로 발생한다(씬2 다락 에피소드
        #    실측 zero_frac 5~19%). 이 게이트의 본질 목적(좌표계/맵 파손 검출 — 파손이면 궤적
        #    대부분이 벽 안=zero_frac 수십%%)에 맞는 약한 기준으로 판정한다.
        passed = bool(medians) and overall_median > 0.10 and worst_zero_frac < 0.30
    else:
        passed = bool(medians) and lo < overall_median < hi and worst_min > GT_CLEARANCE_MIN_M
    return {
        'episodes_tested': episodes,
        'median_m': overall_median,
        'worst_min_m': worst_min,
        'worst_zero_frac': worst_zero_frac,
        'median_range_m': list(GT_CLEARANCE_MEDIAN_RANGE_M),
        'typical_range_m': list(GT_CLEARANCE_TYPICAL_RANGE_M),
        'min_floor_m': GT_CLEARANCE_MIN_M,
        'criterion': ('vln_pe: median>0.10 & max zero_frac<30%' if args.dataset == 'vln_pe'
                      else f'vln_n1: median in {list(GT_CLEARANCE_MEDIAN_RANGE_M)} & min>{GT_CLEARANCE_MIN_M}'),
        'per_episode': per_episode,
        'passed': passed,
    }


def check_map_sanity(occupancy, origin, esdf_ref, obstacle_ref, mesh_bounds, args) -> dict:
    """검증 ② 맵 자체의 자명한 성질."""
    shape = np.array(occupancy.shape)
    grid_max = origin + shape * args.cell_m
    obstacle_zero = bool(esdf_ref[obstacle_ref].max() == 0.0) if obstacle_ref.any() else True
    free = ~obstacle_ref
    free_positive = bool(esdf_ref[free].min() > 0.0) if free.any() else True
    return {
        'occupancy_within_bounds': bool(np.all(origin <= mesh_bounds[0] + 1e-9)
                                        and np.all(grid_max >= mesh_bounds[1] - 1e-9)),
        'obstacle_cells_zero_esdf': obstacle_zero,
        'free_cells_positive_esdf': free_positive,
        'max_esdf_m': float(esdf_ref[np.isfinite(esdf_ref)].max()) if np.isfinite(esdf_ref).any() else 0.0,
        'obstacle_frac': float(obstacle_ref.mean()),
        'navigable_frac': float(truncate_navigable(esdf_ref, args.r_b).mean()),
        'passed': obstacle_zero and free_positive,
    }


def check_ply_overlap(scene_dir: Path, obstacle_ref, origin, floor_z, args) -> dict:
    """검증 ③ GT가 obstacle로 라벨한 점들이 우리 obstacle 셀에 들어가는가 (정황 확인).

    **기하 검증이 아니다** — `.ply` 점들은 mesh 표면 위 0.0000 m라 같은 mesh에서 만든 우리 맵과
    비교하면 순환논증이다. 여기서 보는 건 "mesh->voxel 변환 + 높이대 분류가 GT 라벨과 같은 곳을
    장애물로 보는가"라는 정황뿐이다. 기준값을 모르므로 임계 판정 없이 수치만 남긴다.
    """
    points = load_gt_obstacle_points(scene_dir)
    h, w = obstacle_ref.shape
    ij = world_to_cell(points[:, :2], origin, args.cell_m)
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    hit = obstacle_ref[ij[inside, 1], ij[inside, 0]]
    return {
        'ply_obstacle_points': int(len(points)),
        'inside_grid_frac': float(inside.mean()),
        'in_our_obstacle_frac': float(hit.mean()) if len(hit) else 0.0,
        'ply_z_range': [float(points[:, 2].min()), float(points[:, 2].max())],
        'band_z_range': [floor_z + args.h_nav, None],
        'note': '.ply obstacle은 바닥 슬라이스(z 중앙값 ~0)이고 우리 밴드는 (floor+h_nav, floor+h_b]라 '
                '높이 범위가 다르다 — 겹침률이 낮은 것이 정상이다. 순환논증이라 판정에 쓰지 않는다.',
    }


# ---------------------------------------------------------------------------
# 시각화 — 01의 floorplan 컨벤션. y를 뒤집는 것은 여기서만.
# ---------------------------------------------------------------------------

def _to_image(arr2d: np.ndarray) -> np.ndarray:
    """(Ny,Nx) 계산용 배열 -> 화면용 (위가 +y). esdf_utils는 y를 안 뒤집으므로 여기서 뒤집는다."""
    return arr2d[::-1]


def _upscale(img: np.ndarray) -> np.ndarray:
    factor = max(1, int(np.ceil(FLOORPLAN_MIN_PX / max(img.shape[:2]))))
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def render_states(occupancy, origin, obstacle_ref, esdf_ref, cam_xyz, args, out_dir: Path):
    """blink 4-state — 전부 같은 grid라 화살표를 넘겨도 화면이 안 튄다."""
    h, w = obstacle_ref.shape

    # ① mesh 실루엣 (참고용 높이대, 리포트 가독성 목적)
    k_lo = int((0.15) / args.cell_m)
    k_hi = min(int((2.0) / args.cell_m), occupancy.shape[2] - 1)
    sil = occupancy[:, :, k_lo:k_hi + 1].any(axis=2).T
    img_sil = np.where(_to_image(sil)[..., None], np.uint8([170, 170, 170]), np.uint8([38, 38, 38]))

    # ② obstacle_2d (ref h_b 밴드)
    img_obs = np.where(_to_image(obstacle_ref)[..., None], np.uint8([255, 120, 60]), np.uint8([38, 38, 38]))

    # ③ ESDF 히트맵 (고정 범위 — 실행 간 색 기준 통일)
    norm = np.clip(np.nan_to_num(_to_image(esdf_ref), posinf=ESDF_VIZ_MAX_M) / ESDF_VIZ_MAX_M, 0, 1)
    img_esdf = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8),
                                              getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)),
                            cv2.COLOR_BGR2RGB)

    # ④ navigable + GT 궤적
    nav = truncate_navigable(esdf_ref, args.r_b)
    img_nav = np.where(_to_image(nav)[..., None], np.uint8([60, 90, 60]), np.uint8([38, 38, 38]))
    ij = world_to_cell(cam_xyz[:, :2], origin, args.cell_m)
    keep = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    for x, y in ij[keep]:
        cv2.circle(img_nav, (int(x), int(h - 1 - y)), 1, (255, 230, 60), -1)

    return [
        ('① mesh 실루엣 (높이 0.15~2.0 m 단면)', save_jpg(_upscale(img_sil), out_dir / 'floorplan_mesh.jpg')),
        (f'② obstacle — 밴드 (floor+{args.h_nav:.2f}, floor+h_b]', save_jpg(_upscale(img_obs), out_dir / 'obstacle.jpg')),
        (f'③ ESDF 히트맵 (0~{ESDF_VIZ_MAX_M} m 고정 범위)', save_jpg(_upscale(img_esdf), out_dir / 'esdf.jpg')),
        ((f'④ navigable (r_b={args.r_b} truncate) + GT 궤적'
          if len(cam_xyz) else
          f'④ navigable (r_b={args.r_b} truncate) — GT 없음'),
         save_jpg(_upscale(img_nav), out_dir / 'navigable.jpg')),
    ]


def pill(ok: bool) -> str:
    return f'<span class="pill {"good" if ok else "bad"}">{"PASS" if ok else "FAIL"}</span>'


def build_summary(scene, checks, meta) -> str:
    c, s, p = checks['gt_clearance'], checks['sanity'], checks['ply_overlap']

    if c.get('skipped'):
        return f'''
<div class="stat-row">
  <div class="stat"><b>전체</b><span class="pill warn">UNVERIFIED</span></div>
  <div class="stat"><b>GT clearance</b><span class="pill warn">SKIPPED</span></div>
  <div class="stat"><b>맵 sanity</b>{pill(s["passed"])}</div>
  <div class="stat"><b>grid</b><span class="pill">{meta["grid_shape"]}</span></div>
  <div class="stat"><b>입력</b><span class="pill">{meta["geometry"]}</span></div>
</div>
<p>GT episode가 없는 scene이다. 독립 GT clearance 검증은 수행하지 않았으며,
아래 obstacle/ESDF/navigable은 <code>h_b={meta["ref_h_b"]:.3f} m</code>의
시각화용 reference다. 실제 03 random planning은 occupancy에서 episode별 h_b로 다시 계산한다.</p>
<table class="table">
  <tr><th>검증</th><th>결과</th><th>수치</th></tr>
  <tr><td>① GT 궤적 clearance</td>
      <td><span class="pill warn">SKIPPED</span></td>
      <td>{c["reason"]}</td></tr>
  <tr><td>② 맵 sanity</td><td>{pill(s["passed"])}</td>
      <td>장애물 {s["obstacle_frac"]*100:.1f}% ·
          navigable {s["navigable_frac"]*100:.1f}% ·
          max ESDF {s["max_esdf_m"]:.2f} m</td></tr>
  <tr><td>③ .ply obstacle 겹침</td>
      <td><span class="pill warn">SKIPPED</span></td>
      <td>{p["skipped"]}</td></tr>
</table>
'''

    rows = ''.join(
        f'<tr><td>ep {ep}</td><td>{v["h_b"]:.3f}</td><td>{v["median_m"]:.3f}</td>'
        f'<td>{v["min_m"]:.3f}</td></tr>' for ep, v in c['per_episode'].items())
    if 'skipped' in p:
        ply_row = (f'<tr><td>③ .ply obstacle 겹침 <i>(정황 확인, 판정 아님)</i></td>'
                   f'<td><span class="pill warn">생략</span></td><td>{p["skipped"]}</td></tr>')
        ply_note = ''
    else:
        ply_row = (f'<tr><td>③ .ply obstacle 겹침 <i>(정황 확인, 판정 아님)</i></td>'
                   f'<td><span class="pill warn">참고</span></td>'
                   f'<td>{p["ply_obstacle_points"]:,}점 중 격자 안 {p["inside_grid_frac"]*100:.1f}%, '
                   f'우리 obstacle과 겹침 {p["in_our_obstacle_frac"]*100:.1f}%</td></tr>')
        ply_note = (f'<p class="note">③은 <b>기하 검증이 아니다</b> — .ply 점들은 mesh 표면 위 0.0000 m라 '
                    f'같은 mesh에서 만든 맵과 비교하면 순환논증이다. 게다가 .ply obstacle은 바닥 슬라이스'
                    f'(z {p["ply_z_range"][0]:.2f}~{p["ply_z_range"][1]:.2f} m)이고 우리 밴드는 그 위쪽이라 '
                    f'겹침률이 낮은 게 정상이다.</p>')
    return f'''
<div class="stat-row">
  <div class="stat"><b>전체</b>{pill(checks["passed"])}</div>
  <div class="stat"><b>GT clearance median</b><span class="pill {"good" if c["passed"] else "bad"}">
      {c["median_m"]:.3f} m</span></div>
  <div class="stat"><b>허용 범위</b><span class="pill">{c["median_range_m"][0]}~{c["median_range_m"][1]} m</span></div>
  <div class="stat"><b>grid</b><span class="pill">{meta["grid_shape"]}</span></div>
  <div class="stat"><b>입력</b><span class="pill">{meta["geometry"]}</span></div>
</div>
<p>씬 mesh를 {meta["voxel_size"]} m occupancy로 만들고 저장한다. 2D ESDF는 로봇 키 <code>h_b</code>에
의존하므로 저장하지 않고 03이 매번 도출한다 — 아래 그림은 ep {meta["ref_episode"]}의
<code>h_b={meta["ref_h_b"]:.3f}</code> 기준 참조본이다.</p>
<table class="table">
  <tr><th>검증</th><th>결과</th><th>수치</th></tr>
  <tr><td>① <b>GT 궤적 clearance</b> (유일한 독립 검증)</td><td>{pill(c["passed"])}</td>
      <td>median {c["median_m"]:.3f} m (허용 {c["median_range_m"]}, 6개 씬 실측 {c["typical_range_m"]}),
          최소 {c["worst_min_m"]:.3f} m (&gt; {c["min_floor_m"]})</td></tr>
  <tr><td>② 맵 sanity</td><td>{pill(s["passed"])}</td>
      <td>장애물 {s["obstacle_frac"]*100:.1f}% · navigable {s["navigable_frac"]*100:.1f}% ·
          max ESDF {s["max_esdf_m"]:.2f} m</td></tr>
  {ply_row}
</table>
{ply_note}
<table class="table"><tr><th>에피소드</th><th>h_b</th><th>clearance median</th><th>min</th></tr>{rows}</table>
'''


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_argparser().parse_args()
    if args.data_root is None:
        args.data_root = dataset_utils.default_data_root(args.dataset)
    print(f'[{SCRIPT_NAME}] scene={args.scene} dataset={args.dataset} geometry={args.geometry}')

    meta_path = Path(args.scene_meta_dir or args.out_dir) / 'scene_meta' / f'{args.scene}.json'
    if not meta_path.is_file():
        print(f'  [ERROR] scene_meta 없음: {meta_path} — 01_prepare_scene.py를 먼저 돌릴 것')
        return 2
    with open(meta_path) as f:
        scene_meta = json.load(f)
    if not scene_meta.get('passed'):
        print(f'  [ERROR] 01 게이트를 통과하지 못한 씬이다 (scene_meta.passed=false)')
        return 2

    n_eps = count_episodes(args.data_root, args.scene)
    floor_z = float(scene_meta['floor_z'])
    print(f'  scene_meta: floor_z={floor_z:+.4f} m, episodes={n_eps}')

    # --- occupancy ---
    sources = {'obj': lambda: load_scene_mesh(args.mesh_root, args.scene),
               'usd': lambda: load_usd_mesh_with_app(scene_meta['usd_path'])}
    primary = 'obj' if args.geometry in ('obj', 'both') else 'usd'
    if scene_meta.get('scene_type') == 'new_scene' and primary != 'usd':
        print('  [ERROR] new_scene은 --geometry usd를 사용할 것')
        return 2

    mesh = sources[primary]()

    voxel_bounds = None
    if scene_meta.get('scene_type') == 'new_scene':
        voxel_bounds = np.asarray(
            [scene_meta['bounds_min'], scene_meta['bounds_max']],
            dtype=np.float64,
        )

    occupancy, origin = voxelize_surface(
        mesh, args.cell_m, args.surface_samples,
        align_to_m=args.grid_align,
        bounds=voxel_bounds,
    )
    map_bounds = voxel_bounds if voxel_bounds is not None else np.asarray(mesh.bounds)

    print(f'  occupancy[{primary}]: grid={occupancy.shape} 점유 {100 * occupancy.mean():.1f}% '
          f'(mesh {len(mesh.vertices)} verts)')

    occupancy_alt = None
    if args.geometry == 'both':
        occupancy_alt, origin_alt = voxelize_surface(sources['usd'](), args.cell_m,
                                                    args.surface_samples,
                                                    align_to_m=args.grid_align)
        assert np.abs(origin_alt - origin).max() < 1e-6, 'obj/usd origin 불일치'
        print(f'  occupancy[usd] : 점유 {100 * occupancy_alt.mean():.1f}% '
              f'(XOR {int((occupancy_alt ^ occupancy).sum())} voxels)')

    # --- 참조 2D (시각화·검증용) ---
    if n_eps > 0:
        cam_xyz, ref_h_b, _, _ = load_gt_episode(
            args.data_root, args.scene, args.ref_episode, args.dataset
        )
        ref_episode = args.ref_episode
    else:
        if scene_meta.get('scene_type') != 'new_scene':
            print('  [ERROR] GT episode가 없고 new_scene도 아니다')
            return 2
        if args.ref_h_b is None or args.ref_h_b <= 0:
            print('  [ERROR] GT가 없는 new_scene은 --ref_h_b > 0 을 지정할 것')
            return 2
        ref_h_b = float(args.ref_h_b)
        ref_episode = None
        cam_xyz = np.empty((0, 3), dtype=np.float64)

    # h_nav가 로봇 키보다 크면 밴드가 비어 맵이 무의미해진다 — assert 트레이스백 대신 명확히 거른다.
    h_b_list = [load_gt_episode(args.data_root, args.scene, ep, args.dataset)[1]
                for ep in range(min(args.num_check_episodes, n_eps))]
    _min_hb = min(h_b_list + [ref_h_b])
    _eff_h_nav = args.h_nav_ratio * _min_hb if args.h_nav_ratio else args.h_nav
    if _eff_h_nav >= _min_hb:
        print(f'  [ERROR] 실효 h_nav {_eff_h_nav:.3f} 가 최소 로봇 키 {_min_hb:.3f} m 이상이다 — '
              f'밴드 (floor+h_nav, floor+h_b] 가 비어 맵이 무의미해진다.')
        return 2
    ref_h_nav = args.h_nav_ratio * ref_h_b if args.h_nav_ratio else args.h_nav
    obstacle_ref = derive_obstacle_2d(occupancy, origin, floor_z, ref_h_b, args.cell_m, ref_h_nav)
    esdf_ref = compute_esdf_2d(obstacle_ref, args.cell_m)
    nav_ref = truncate_navigable(esdf_ref, args.r_b)

    # --- 검증 ---
    sanity = check_map_sanity(
        occupancy, origin, esdf_ref, obstacle_ref, map_bounds, args
    )

    if n_eps > 0:
        print(f'  [gate] GT 궤적 clearance ({min(args.num_check_episodes, n_eps)} episodes) ...')
        gt_clearance = check_gt_clearance(occupancy, origin, args, n_eps)
        if args.dataset == 'vln_pe':
            ply_overlap = {'skipped': 'vln_pe has no meta/pointcloud.ply'}
        else:
            ply_overlap = check_ply_overlap(
                Path(args.data_root) / args.scene,
                obstacle_ref, origin, floor_z, args,
            )
        passed = bool(gt_clearance['passed'] and sanity['passed'])
    else:
        print('  [gate] GT 궤적 clearance: SKIPPED (GT episode 없음)')
        gt_clearance = {
            'skipped': True,
            'passed': None,
            'reason': 'GT episode가 없어 독립 clearance 검증을 수행하지 않음',
        }
        ply_overlap = {'skipped': 'GT pointcloud 없음'}
        passed = None

    checks = {
        'gt_clearance': gt_clearance,
        'sanity': sanity,
        'ply_overlap': ply_overlap,
        'passed': passed,
    }

    # --- 저장 ---
    esdf_dir = Path(args.out_dir) / 'esdf'
    esdf_dir.mkdir(parents=True, exist_ok=True)
    arrays = {'occupancy': occupancy, 'origin': origin,
              'voxel_size': np.float32(args.cell_m), 'h_nav': np.float32(args.h_nav),
              'h_nav_ratio': np.float32(args.h_nav_ratio if args.h_nav_ratio else 0.0),
              'r_b': np.float32(args.r_b), 'floor_z': np.float32(floor_z),
              'ref_h_b': np.float32(ref_h_b), 'nav_mask_ref': nav_ref, 'esdf_ref': esdf_ref}
    if occupancy_alt is not None:
        arrays['occupancy_alt'] = occupancy_alt
    npz_path = esdf_dir / f'{args.scene}{dataset_utils.dataset_tag(args.dataset)}.npz'
    np.savez_compressed(npz_path, **arrays)

    meta = {
        'scene_id': args.scene, 'geometry': primary,
        'source_mesh': str(scene_meta['mesh_path'] if primary == 'obj' else scene_meta['usd_path']),
        'has_occupancy_alt': occupancy_alt is not None,
        'grid_shape': list(occupancy.shape), 'origin': origin.tolist(),
        'voxel_size': args.cell_m, 'h_nav': args.h_nav, 'h_nav_ratio': args.h_nav_ratio, 'r_b': args.r_b, 'floor_z': floor_z,
        'ref_episode': ref_episode, 'ref_h_b': ref_h_b,
        'occupied_frac': float(occupancy.mean()),
        'checks': checks, 'passed': passed,
        'note': '2D ESDF는 h_b에 의존하므로 저장하지 않는다 — 03이 occupancy에서 매번 도출한다. '
                'nav_mask_ref/esdf_ref는 ref_h_b 기준 참조본(시각화·검증용).',
    }
    json_path = esdf_dir / f'{args.scene}{dataset_utils.dataset_tag(args.dataset)}.json'
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'  esdf -> {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB) + {json_path.name}')

    # --- 리포트 ---
    log_dir = Path(args.log_dir) / SCRIPT_NAME / f'{args.scene}{dataset_utils.dataset_tag(args.dataset)}'
    states = render_states(occupancy, origin, obstacle_ref, esdf_ref, cam_xyz, args, log_dir)
    gt_note = (
        '④의 노란 점이 GT 궤적 — 초록(navigable) 안에 있어야 한다.'
        if n_eps > 0 else
        'GT episode가 없어 ④에는 navigable map만 표시한다.'
    )
    body = (f'<h3>맵 단계별 (전부 같은 grid, cell={args.cell_m} m)</h3>'
            '<p class="note">‹ ›로 넘기며 본다. ① mesh 실루엣은 참고용 높이대이고, ②가 실제로 쓰는 '
            f'obstacle이다(로봇 키 밴드). {gt_note}</p>'
            + blink_widget_html('mapstages', states, title=f'{args.scene} map stages')
            + reference_button_html('참고 — 입력/파라미터',
                                    [(f'mesh 실루엣 ({primary})', states[0][1])]))
    report = save_gallery(log_dir, 'report.html', f'02_build_freemap_esdf — {args.scene}',
                          build_summary(args.scene, checks, meta), body)
    print(f'  report html -> {report}')
    if passed is None:
        print(f'  => UNVERIFIED (GT clearance=SKIPPED, sanity={sanity["passed"]})')
        return 0 if sanity['passed'] else 1

    print(f'  => {"PASS" if passed else "FAIL"} '
          f'(clearance={gt_clearance["passed"]} sanity={sanity["passed"]})')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
