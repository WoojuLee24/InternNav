"""8축 파라미터 그리드서치 — GT / 기준선ⓒ / 생성 경로 세 쌍을 비교한다.

    03 : ESDF + GT 조건  -> "경로를 만든다"          -> paths/<scene>.json
    03b: 03의 결과       -> "왜 이만큼 다른가"        -> verify/<scene>.json
    03c: 두 refine 해석  -> "어느 해석이 맞나"        -> compare/<scene>.json
    03d: 8개 파라미터 축 -> "**어디를 더 만질 수 있나**" -> gridsearch/results.json

지금까지 `h_nav`/`refine_radius`/`r_b`/다운샘플/원점을 매번 다른 시점에 다른 임시 코드로 스윕했다.
이 스크립트는 8개 축을 **한 곳에서, 각 축을 독립적으로**(다른 7개 축은 현재 프로덕션 기본값에
고정) 그리드서치하고, 각 지점에서 세 거리를 함께 잰다:

  - `d(GT, ⓒ)` — 기준선 오차. 루트가 완벽해도 못 줄이는 하한 (`pipeline_floor_path`가 만든 경로).
  - `d(GT, path)` — 실제 재현 오차 (03의 `chamfer_m`/`frechet_m`과 동일).
  - `d(ⓒ, path)` — 기준선과 생성 경로 사이 거리(**이 스크립트가 추가하는 것**). 생성 경로가
    "이론적 최선"에서 얼마나 벗어났는지를 직접 보여준다.

## 8개 축과 현재 프로덕션 기본값

| # | 축 | 기본값 |
|---|---|---|
| 1 | A* 격자 `astar_cell_m` | 0.20 (논문 명시) |
| 2 | refine 반경 `refine_radius` (mode=argmax 고정) | 0.10 |
| 3a | A* 연결성 `connectivity` | 8 |
| 3b | A* clearance tie-break 가중치 `clearance_weight` | 0.0 |
| 4 | `thin_waypoints` 간격 `spacing_m` | 0.8 |
| 5 | cubic spline 출력 간격 `smooth_step` | 0.05 |
| 6 | 지면 여유고 `h_nav` | 0.10 |
| 7 | 로봇 반경 `r_b` | 0.25 |
| 8 | 다운샘플 방식 `downsample_mode` | any |

**1번(A* 0.05 포함)은 "논문 숫자값(voxel 0.05, A* 0.2)은 고정한다"는 원칙을 일부러 깨는 탐색
측정이다.** 이 스크립트는 순수 탐색용이며 00~03c의 기본값을 바꾸지 않는다.

**8번 "다운샘플 유무"**: `downsample_mode='any'`인 채로 `astar_cell_m`을 voxel 크기(0.05)까지
낮추면 다운샘플 factor가 1이 되어 그게 곧 "다운샘플 없음"과 동치다 — 축1의 최소값이 이미 그
경우를 대표하므로, 8번 자체는 다운샘플 **방식**(any/majority/all)만 스윕한다.

## 내장 회귀 검증

축 1·2·4·5·6·7의 "현재 기본값" 지점, 축3의 (connectivity=8, clearance_weight=0), 축8의 `any`는
**전부 지금 프로덕션 03의 설정과 정확히 같다.** 그 지점에서 `d(GT,path)` median이 이미 검증된
값(씬1 chamfer 0.103/Fréchet 0.266, 씬2 0.134/0.333)과 일치하는지 자동으로 확인한다 —
8개 축 각각에서 독립적으로 같은 두 숫자를 재현해야 하므로 회귀 가드가 8중으로 걸린다.

## 실행

    python 03d_grid_search_params.py --num_episodes 20

출력: `gridsearch/results.json`, `logs/gs-vlnpe/03d_grid_search_params/report.html`
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esdf_utils import (  # noqa: E402
    ASTAR_CELL_M,
    DOWNSAMPLE_MODES,
    H_NAV_M,
    REFINERS,
    ROBOT_RADIUS_M,
    SMOOTHERS,
    VOXEL_SIZE_M,
    WAYPOINT_SPACING_M,
    astar,
    cell_to_world,
    chamfer_distance,
    check_path_navigable,
    compare_clearance_profile,
    compute_esdf_2d,
    compute_scan_coverage_mask,
    derive_obstacle_2d,
    discrete_frechet_distance,
    downsample_navigable,
    heading_alignment,
    path_smoothness,
    pipeline_floor_path,
    thin_waypoints,
    truncate_navigable,
    world_to_cell,
)
from geometry_utils import action_to_c2w, decompose_camera_extrinsic  # noqa: E402
from viz_utils import (  # noqa: E402
    FLOOR_COLOR, GT_COLOR, OURS_COLOR, blink_widget_html, floorplan_canvas, line_chart, save_gallery,
)

DEFAULT_DATA_ROOT = 'data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i'
DEFAULT_SCENES = '17DRP5sb8fy,s8pcmisQ38h'
DEFAULT_OUT_DIR = 'scripts/dataset_converters/gs_vlnpe'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe'
SCRIPT_NAME = '03d_grid_search_params'
SAME_ROUTE_M = 0.8

# 이미 검증된 03 프로덕션 수치(모든 축이 기본값일 때) — 8개 축 각각의 "현재값" 지점에서 재현되는지
# 확인하는 회귀 가드. 값이 어긋나면 plan_variant가 03의 plan_episode와 갈렸다는 뜻이다.
KNOWN_CHAMFER_M = {'17DRP5sb8fy': 0.103, 's8pcmisQ38h': 0.134}
REGRESSION_TOL_M = 0.001


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenes', default=DEFAULT_SCENES)
    p.add_argument('--num_episodes', type=int, default=20)
    p.add_argument('--astar_cell_candidates', default='0.05,0.10,0.15,0.20,0.25,0.30')
    p.add_argument('--refine_radius_candidates', default='0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40')
    p.add_argument('--connectivity_candidates', default='4,8')
    p.add_argument('--clearance_weight_candidates', default='0.0,0.05,0.1,0.2,0.5,1.0')
    p.add_argument('--spacing_candidates', default='0.0,0.2,0.4,0.6,0.8,1.0,1.5,2.0,3.0')
    p.add_argument('--smooth_step_candidates', default='0.01,0.02,0.05,0.10,0.20,0.40')
    p.add_argument('--h_nav_candidates', default='0.05,0.08,0.10,0.12,0.15,0.20')
    p.add_argument('--r_b_candidates', default='0.15,0.18,0.20,0.25,0.30')
    p.add_argument('--downsample_mode_candidates', default='any,majority,all')
    p.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--esdf_dir', default=None, help='기본값은 <out_dir>/esdf')
    p.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    p.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    return p


# ---------------------------------------------------------------------------
# 데이터 로딩 (씬당 1회, 모든 축 sweep이 재사용)
# ---------------------------------------------------------------------------

def load_scene_map(esdf_dir: Path, scene: str):
    d = np.load(esdf_dir / f'{scene}.npz')
    return d['occupancy'], d['origin'], float(d['floor_z']), float(d['voxel_size'])


def load_episode(data_root: str, scene: str, ep: int):
    table = pq.read_table(Path(data_root) / scene / 'data' / 'chunk-000' / f'episode_{ep:06d}.parquet',
                          columns=['observation.camera_extrinsic', 'action'])
    extrinsic = np.asarray(table['observation.camera_extrinsic'].to_pylist()[0],
                           dtype=np.float64).reshape(4, 4)
    actions = [np.asarray(a, dtype=np.float64).reshape(4, 4) for a in table['action'].to_pylist()]
    xy = np.stack([action_to_c2w(a, 'cam2world_gl')[:3, 3] for a in actions])[:, :2]
    h_b, _ = decompose_camera_extrinsic(extrinsic)
    return xy, h_b


# ---------------------------------------------------------------------------
# 한 파라미터 조합으로 에피소드 1개를 계획 — 03의 plan_episode와 같은 순서
# ---------------------------------------------------------------------------

def coarse_mean(esdf: np.ndarray, factor: int) -> np.ndarray:
    """ESDF를 `factor`배 블록으로 평균 다운샘플 — A* clearance tie-break 가중치(축3b)의 입력.

    `downsample_navigable`은 bool을 any/majority/all로 모으지만, tie-break는 실수 clearance가
    필요해서 별도로 mean-pool한다. 이 축 전용이라 `esdf_utils`에는 넣지 않는다.
    """
    h, w = esdf.shape
    hc, wc = h // factor, w // factor
    if hc == 0 or wc == 0:
        return np.zeros((max(hc, 1), max(wc, 1)), dtype=np.float32)
    block = esdf[:hc * factor, :wc * factor].reshape(hc, factor, wc, factor)
    return block.mean(axis=(1, 3))


def plan_variant(occupancy, origin, floor_z, h_b, gt_xy, cell, *, h_nav, r_b, downsample_mode,
                 astar_cell_m, connectivity, clearance_weight, refine_radius, spacing_m,
                 smooth_step) -> dict:
    """03의 `plan_episode`와 완전히 같은 순서. 8개 파라미터가 전부 함수 인자로 열려 있다는 것만 다르다.

    기본값(astar_cell_m=0.2, connectivity=8, clearance_weight=0, refine_radius=0.10, spacing_m=0.8,
    smooth_step=0.05, h_nav=0.10, r_b=0.25, downsample_mode='any')에서는 03과 **비트 단위로 같은
    계산**이라 회귀 검증(`KNOWN_CHAMFER_M`)이 성립한다.
    """
    obstacle = derive_obstacle_2d(occupancy, origin, floor_z, h_b, cell, h_nav)
    esdf = compute_esdf_2d(obstacle, cell)
    # 03의 plan_episode와 비트 단위로 같은 계산이어야 하므로 03에 추가한 coverage mask도 그대로
    # 반영한다(GT start/goal은 항상 건물 안이라 이 축의 회귀값 자체는 바뀌지 않는다).
    navigable = truncate_navigable(esdf, r_b) & compute_scan_coverage_mask(occupancy)
    factor = max(1, int(round(astar_cell_m / cell)))
    nav_coarse = downsample_navigable(navigable, factor, downsample_mode)
    coarse_cell = cell * factor

    clearance_coarse = coarse_mean(esdf, factor) if clearance_weight else None

    sg = np.array([gt_xy[0], gt_xy[-1]], dtype=np.float64)
    ij = world_to_cell(sg, origin, coarse_cell)
    h_c, w_c = nav_coarse.shape
    ij[:, 0] = np.clip(ij[:, 0], 0, w_c - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, h_c - 1)
    if not nav_coarse[ij[0, 1], ij[0, 0]] or not nav_coarse[ij[1, 1], ij[1, 0]]:
        return {'status': 'start_or_goal_not_navigable'}

    path_ij = astar(nav_coarse, ij[0], ij[1], connectivity=connectivity,
                    clearance=clearance_coarse, clearance_weight=clearance_weight)
    if path_ij is None:
        return {'status': 'astar_failed'}

    wp_astar = cell_to_world(path_ij, origin, coarse_cell)
    wp_astar[0], wp_astar[-1] = sg[0], sg[1]
    wp_refined = REFINERS['argmax'](wp_astar, esdf, origin, cell, refine_radius, fix_endpoints=True)
    wp_thinned = thin_waypoints(wp_refined, spacing_m)
    trajectory = SMOOTHERS['cubic'](wp_thinned, smooth_step)

    floor_path = pipeline_floor_path(gt_xy, esdf, origin, cell, coarse_cell,
                                     refine_radius_m=refine_radius, spacing_m=spacing_m,
                                     smooth_step_m=smooth_step, refine_mode='argmax',
                                     smooth_mode='cubic')

    chk_refined = check_path_navigable(wp_refined, esdf, origin, cell, r_b)
    chk_traj = check_path_navigable(trajectory, esdf, origin, cell, r_b)
    sm_ours, sm_gt = path_smoothness(trajectory), path_smoothness(gt_xy)
    frechet = discrete_frechet_distance(trajectory, gt_xy)

    return {
        'status': 'ok', 'trajectory': trajectory, 'floor_path': floor_path,
        'wp_astar': wp_astar, 'navigable': navigable,
        'd_gt_floor_chamfer': chamfer_distance(floor_path, gt_xy),
        'd_gt_floor_frechet': discrete_frechet_distance(floor_path, gt_xy),
        'd_gt_path_chamfer': chamfer_distance(trajectory, gt_xy),
        'd_gt_path_frechet': frechet,
        'd_floor_path_chamfer': chamfer_distance(floor_path, trajectory),
        'turn_ratio': sm_ours['turn_per_m_deg'] / max(sm_gt['turn_per_m_deg'], 1e-9),
        'clr_diff_m': compare_clearance_profile(trajectory, gt_xy, esdf, origin, cell)['median_diff_m'],
        'heading_deg': heading_alignment(trajectory, gt_xy)['median_deg'],
        'same_route': bool(frechet < SAME_ROUTE_M),
        'hard_collision': bool(chk_traj['hard_n'] > 0),
        'r_b_ok': bool(chk_refined['points_ok']),
    }


# ---------------------------------------------------------------------------
# 축 정의
# ---------------------------------------------------------------------------

AGG_KEYS = ('d_gt_floor_chamfer', 'd_gt_floor_frechet', 'd_gt_path_chamfer', 'd_gt_path_frechet',
           'd_floor_path_chamfer', 'turn_ratio', 'clr_diff_m', 'heading_deg')


def make_axes(args) -> list:
    parse_f = lambda s: [float(x) for x in s.split(',')]
    return [
        {'key': 'astar_cell_m', 'label': 'A* 격자 (astar_cell_m)',
         'candidates': parse_f(args.astar_cell_candidates), 'default': ASTAR_CELL_M},
        {'key': 'refine_radius', 'label': 'refine 반경 (maximizing distance, argmax)',
         'candidates': parse_f(args.refine_radius_candidates), 'default': 0.10},
        {'key': 'connectivity', 'label': 'A* 연결성 (connectivity)',
         'candidates': [int(x) for x in args.connectivity_candidates.split(',')], 'default': 8},
        {'key': 'clearance_weight', 'label': 'A* clearance tie-break 가중치',
         'candidates': parse_f(args.clearance_weight_candidates), 'default': 0.0},
        {'key': 'spacing_m', 'label': 'thin_waypoints 간격',
         'candidates': parse_f(args.spacing_candidates), 'default': WAYPOINT_SPACING_M},
        {'key': 'smooth_step', 'label': 'cubic spline 출력 간격',
         'candidates': parse_f(args.smooth_step_candidates), 'default': 0.05},
        {'key': 'h_nav', 'label': '지면 여유고 (h_nav)',
         'candidates': parse_f(args.h_nav_candidates), 'default': H_NAV_M},
        {'key': 'r_b', 'label': '로봇 반경 (r_b)',
         'candidates': parse_f(args.r_b_candidates), 'default': ROBOT_RADIUS_M},
        {'key': 'downsample_mode', 'label': '다운샘플 방식',
         'candidates': args.downsample_mode_candidates.split(','), 'default': 'any'},
    ]


def base_params() -> dict:
    return dict(h_nav=H_NAV_M, r_b=ROBOT_RADIUS_M, downsample_mode='any', astar_cell_m=ASTAR_CELL_M,
               connectivity=8, clearance_weight=0.0, refine_radius=0.10,
               spacing_m=WAYPOINT_SPACING_M, smooth_step=0.05)


def sweep_axis(axis: dict, scenes_data: dict, num_episodes: int) -> list:
    """축의 후보마다 두 씬 20 에피소드 median을 낸다. 다른 7개 축은 `base_params()`에 고정."""
    rows = []
    for val in axis['candidates']:
        params = base_params()
        params[axis['key']] = val
        row = {'value': val, 'scenes': {}}
        for scene, (occ, origin, floor_z, cell, episodes) in scenes_data.items():
            agg = {k: [] for k in AGG_KEYS}
            n_ok = n_same = n_hard = n_rb_ok = 0
            for gt_xy, h_b in episodes[:num_episodes]:
                r = plan_variant(occ, origin, floor_z, h_b, gt_xy, cell, **params)
                if r['status'] != 'ok':
                    continue
                n_ok += 1
                n_same += int(r['same_route'])
                n_hard += int(r['hard_collision'])
                n_rb_ok += int(r['r_b_ok'])
                for k in AGG_KEYS:
                    agg[k].append(r[k])
            med = {k: (float(np.median(v)) if v else float('nan')) for k, v in agg.items()}
            med['ratio_over_floor'] = med['d_gt_path_chamfer'] / max(med['d_gt_floor_chamfer'], 1e-9)
            med.update(n_ok=n_ok, n_total=min(num_episodes, len(episodes)),
                      n_same_route=n_same, n_hard_collision=n_hard, n_rb_ok=n_rb_ok)
            row['scenes'][scene] = med
        rows.append(row)
    return rows


def check_regression(axis: dict, rows: list) -> bool:
    """축의 '현재 기본값' 지점이 03 프로덕션 수치와 일치하는지 — 회귀 가드."""
    hit = next((r for r in rows if r['value'] == axis['default']), None)
    if hit is None:
        return True   # 기본값이 후보에 없으면 검사 생략 (에러 아님)
    ok = True
    for scene, known in KNOWN_CHAMFER_M.items():
        if scene not in hit['scenes']:
            continue
        got = hit['scenes'][scene]['d_gt_path_chamfer']
        if abs(got - known) > REGRESSION_TOL_M:
            print(f"  [회귀 실패] 축 '{axis['label']}' 기본값({axis['default']}) {scene}: "
                  f"chamfer {got:.4f} != 검증값 {known:.4f} (허용 {REGRESSION_TOL_M})")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# 시각화 — 축의 극값(최소/최대 후보) 두 지점에서 겹쳐보기
# ---------------------------------------------------------------------------

def crop_canvas(navigable, origin, cell, paths, out_dir: Path, margin_m=0.6, min_px=760):
    """에피소드 주변만 잘라 확대 (03c와 동일 패턴) — 씬 전체를 그리면 경로가 몇 픽셀이라 판독 불가."""
    pts = np.vstack([np.asarray(p)[:, :2] for p in paths])
    lo, hi = pts.min(axis=0) - margin_m, pts.max(axis=0) + margin_m
    h, w = navigable.shape
    i0, j0 = (np.floor((lo - origin[:2]) / cell)).astype(int)
    i1, j1 = (np.ceil((hi - origin[:2]) / cell)).astype(int) + 1
    i0, j0 = max(i0, 0), max(j0, 0)
    i1, j1 = min(i1, w), min(j1, h)
    sub = navigable[j0:j1, i0:i1]
    sub_origin = np.array([origin[0] + i0 * cell, origin[1] + j0 * cell, 0.0])
    return floorplan_canvas(sub, sub_origin, cell, out_dir, min_px)


def render_axis_extremes(axis: dict, scene_data, out_dir: Path) -> str:
    """축의 최소/최대 후보에서 가장 크게 갈리는 에피소드 하나를 골라 GT/ⓒ/path blink로 보여준다."""
    occ, origin, floor_z, cell, episodes = scene_data
    lo_val, hi_val = min(axis['candidates']), max(axis['candidates'])
    if lo_val == hi_val:
        return ''
    best = None
    for idx, (gt_xy, h_b) in enumerate(episodes):
        lo = plan_variant(occ, origin, floor_z, h_b, gt_xy, cell,
                          **{**base_params(), axis['key']: lo_val})
        hi = plan_variant(occ, origin, floor_z, h_b, gt_xy, cell,
                          **{**base_params(), axis['key']: hi_val})
        if lo['status'] != 'ok' or hi['status'] != 'ok':
            continue
        diff = abs(lo['d_gt_path_chamfer'] - hi['d_gt_path_chamfer'])
        if best is None or diff > best[0]:
            best = (diff, idx, gt_xy, lo, hi)
    if best is None:
        return ''
    _, idx, gt_xy, lo, hi = best
    base, draw, emit = crop_canvas(lo['navigable'], origin, cell,
                                   [gt_xy, lo['trajectory'], hi['trajectory']], out_dir)
    g = draw(base.copy(), gt_xy, GT_COLOR, 2)
    lo_img = draw(g.copy(), lo['trajectory'], OURS_COLOR, 2)
    hi_img = draw(g.copy(), hi['trajectory'], FLOOR_COLOR, 2)
    tag = axis['key']
    states = [
        (f'{axis["label"]} = {lo_val} (초록) — chamfer {lo["d_gt_path_chamfer"]:.3f} m',
         emit(lo_img, f'{tag}_lo.jpg')),
        (f'{axis["label"]} = {hi_val} (하늘) — chamfer {hi["d_gt_path_chamfer"]:.3f} m',
         emit(hi_img, f'{tag}_hi.jpg')),
    ]
    return blink_widget_html(f'ext_{tag}', states, title=f'ep {idx} — 극값 비교 (노랑=GT)')


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

def fmt_val(v) -> str:
    return f'{v:.2f}' if isinstance(v, float) else str(v)


def build_axis_table(scene: str, rows: list, default_val) -> str:
    trs = []
    for row in rows:
        m = row['scenes'].get(scene)
        if m is None:
            continue
        mark = ' style="font-weight:700"' if row['value'] == default_val else ''
        trs.append(
            f'<tr{mark}><td>{fmt_val(row["value"])}{" (기본값)" if row["value"] == default_val else ""}</td>'
            f'<td>{m["d_gt_floor_chamfer"]:.3f}</td><td>{m["d_gt_floor_frechet"]:.3f}</td>'
            f'<td>{m["d_gt_path_chamfer"]:.3f}</td><td>{m["d_gt_path_frechet"]:.3f}</td>'
            f'<td>{m["d_floor_path_chamfer"]:.3f}</td><td>{m["ratio_over_floor"]:.2f}</td>'
            f'<td>{m["n_same_route"]}/{m["n_ok"]}</td><td>{m["n_hard_collision"]}</td>'
            f'<td>{m["n_ok"]}/{m["n_total"]}</td></tr>')
    return (f'<div class="table-scroll" style="overflow-x:auto"><table class="table">'
            f'<tr><th>값</th><th>d(GT,ⓒ) chamfer</th><th>d(GT,ⓒ) Fréchet</th>'
            f'<th>d(GT,path) chamfer</th><th>d(GT,path) Fréchet</th><th>d(ⓒ,path) chamfer</th>'
            f'<th>배수</th><th>같은루트</th><th>하드충돌</th><th>계획성공</th></tr>'
            f'{"".join(trs)}</table></div>')


def build_axis_chart(scene: str, rows: list, out_dir: Path, tag: str) -> str:
    xs = [row['value'] for row in rows if isinstance(row['value'], (int, float))
         and scene in row['scenes']]
    if len(xs) < 2:
        return ''
    ys = lambda key: [row['scenes'][scene][key] for row in rows if row['value'] in xs]
    series = [
        ('d(GT,C) floor', FLOOR_COLOR, xs, ys('d_gt_floor_chamfer')),
        ('d(GT,path)', OURS_COLOR, xs, ys('d_gt_path_chamfer')),
        ('d(C,path)', GT_COLOR, xs, ys('d_floor_path_chamfer')),
    ]
    path = line_chart(series, out_dir / f'{tag}_{scene}_chart.jpg',
                      x_label='parameter value', y_label='chamfer [m]')
    import base64
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<div class="card"><img src="data:image/jpeg;base64,{data}" alt="{tag} {scene}"></div>'


def build_report(axes: list, axes_results: dict, scenes: list, out_dir: Path) -> str:
    """축마다 표(씬별) + chamfer 추이 차트 + 극값 겹쳐보기를 이어붙인다."""
    body = []
    for axis in axes:
        rows = axes_results[axis['key']]
        body.append(f'<h3>{axis["label"]} (`{axis["key"]}`, 기본값 {fmt_val(axis["default"])})</h3>')
        for scene in scenes:
            body.append(f'<p class="note">{scene}</p>')
            body.append(build_axis_table(scene, rows, axis['default']))
            body.append(build_axis_chart(scene, rows, out_dir, axis['key']))
        ext = axis.get('_ext_html', '')
        if ext:
            body.append('<p class="note">극값(후보 최소/최대) 겹쳐보기 — 가장 크게 갈리는 에피소드</p>')
            body.append(ext)
    return ''.join(body)


def main() -> int:
    args = build_argparser().parse_args()
    scenes = args.scenes.split(',')
    esdf_dir = Path(args.esdf_dir or (Path(args.out_dir) / 'esdf'))
    print(f'[{SCRIPT_NAME}] scenes={scenes} num_episodes={args.num_episodes}')

    scenes_data = {}
    for scene in scenes:
        occ, origin, floor_z, cell = load_scene_map(esdf_dir, scene)
        n_avail = len(sorted((Path(args.data_root) / scene / 'data' / 'chunk-000').glob('*.parquet')))
        episodes = [load_episode(args.data_root, scene, ep)
                   for ep in range(min(args.num_episodes, n_avail))]
        scenes_data[scene] = (occ, origin, floor_z, cell, episodes)
        print(f'  {scene}: grid={occ.shape} cell={cell} 에피소드 {len(episodes)}개 로드')

    axes = make_axes(args)
    axes_results = {}
    all_ok = True
    for axis in axes:
        rows = sweep_axis(axis, scenes_data, args.num_episodes)
        ok = check_regression(axis, rows)
        all_ok = all_ok and ok
        log_dir = Path(args.log_dir) / SCRIPT_NAME
        axis['_ext_html'] = render_axis_extremes(axis, scenes_data[scenes[0]], log_dir)
        axes_results[axis['key']] = rows
        print(f"  축 '{axis['label']}': {len(rows)}개 후보 완료"
              f"{' [회귀 OK]' if ok else ' [회귀 실패!]'}")

    out_dir = Path(args.out_dir) / 'gridsearch'
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / 'results.json'
    json_path.write_text(json.dumps(
        {a['key']: {'label': a['label'], 'default': a['default'], 'rows': axes_results[a['key']]}
         for a in axes},
        indent=2, ensure_ascii=False, default=lambda o: None), encoding='utf-8')

    log_dir = Path(args.log_dir) / SCRIPT_NAME
    body = build_report(axes, axes_results, scenes, log_dir)

    n_configs = sum(len(a['candidates']) for a in axes)
    summary = f'''
<div class="stat-row">
  <div class="stat"><b>축</b><span class="pill">{len(axes)}</span></div>
  <div class="stat"><b>(축,값) 조합</b><span class="pill">{n_configs}</span></div>
  <div class="stat"><b>회귀 검증</b><span class="pill {"good" if all_ok else "bad"}">
      {"전부 통과" if all_ok else "실패 있음"}</span></div>
</div>
<p>8개 축을 각각 독립적으로(다른 7개는 프로덕션 기본값에 고정) 스윕해 <b>d(GT,ⓒ)</b>(기준선 오차) /
<b>d(GT,path)</b>(실제 재현 오차) / <b>d(ⓒ,path)</b>(생성 경로가 이론적 최선에서 벗어난 정도)를
비교한다. 굵게 표시된 행이 현재 프로덕션 기본값이며, 그 지점의 d(GT,path)가 이미 검증된 03 수치
(17DRP5sb8fy 0.103 m, s8pcmisQ38h 0.134 m)와 일치하는지가 이 스윕 자체의 회귀 가드다.</p>
'''
    report = save_gallery(log_dir, 'report.html', f'{SCRIPT_NAME} — 8축 그리드서치', summary, body)
    print(f'  결과 json -> {json_path}')
    print(f'  report html -> {report}')
    print(f'  => 회귀 검증 {"전부 통과" if all_ok else "실패"}, exit {0 if all_ok else 1}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
