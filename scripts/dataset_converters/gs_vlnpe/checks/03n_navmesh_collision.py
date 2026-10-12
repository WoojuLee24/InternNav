"""GT 실행 궤적과 우리 경로의 충돌을 **occupancy 와 navmesh 두 자로 나란히** 잰다 (vln_ce 전용).

왜 필요한가
-----------
`03` 은 우리 occupancy(밴드 투영)로 충돌을 잰다. 그 자로 재면 **GT 실행 궤적 자신이** 일부 에피소드에서
장애물을 지난다. vln_ce 의 GT 는 habitat navmesh 위에서 만들어졌으므로 navmesh 가 GT 의 자다 —
같은 궤적을 navmesh 로 다시 재서 그 위반이 GT 결함인지 우리 지도의 오차인지 가른다.

충돌 정의는 `03` 의 `hard_collision_eps` 와 같다: `check_path_navigable(...)['hard_ok'] == False`
(궤적을 5분의 1 셀 간격으로 찍었을 때 clearance 0 인 점이 하나라도 있으면 충돌).

navmesh 는 **이미 `r_b` 로 깎인 configuration space** 다. `navmesh_grid.esdf_from_mask` 가
`esdf = (마스크 안) ? 경계까지거리 + r_b : 0` 로 규약을 맞추므로 여기서 `r_b` 를 **다시** 적용하지 않는다.

깊이(depth)는 충돌 점이 자유 공간에서 얼마나 떨어져 있는지 [m] — 1 셀(0.05) 이하는 격자 반올림이다.

floor 규약
- occupancy: `03` 과 똑같이 고른다(씬 대표 floor_z, 0.3 m 넘게 다르면 에피소드 자기 floor).
  우리 경로도 같은 규칙 → `03` 의 `hard_collision_eps` 와 정확히 일치(`ours_occ_matches_03`).
  ⚠️ paths json 의 `episodes[].floor_z` 는 씬 대표 바닥이라 다층 씬에서 실제 계획 바닥과 다르다 — 쓰지 말 것.
- navmesh: 에피소드 자기 floor(`median(cam_z) - h_b` = habitat y)로 `get_topdown_view` 를 자른다.
- 계단 에피소드(cam z 변동 > 0.3 m)는 단일 높이 절단이 무의미하므로 따로 센다.

출력: `<log_dir>/03n_navmesh_collision/<label>/summary.json`, 충돌 에피소드 PNG,
      `--grid_out_dir/<scene>_rb<r_b>.npz` (habitat 없이 재사용할 navmesh 격자).
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent.parent / '3dloader_vlnce'), str(HERE.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataset_utils  # noqa: E402
from esdf_utils import (  # noqa: E402
    check_path_navigable,
    compute_esdf_2d,
    derive_obstacle_2d,
    world_to_cell,
)

SCRIPT_NAME = '03n_navmesh_collision'
FLOOR_TOL_M = 0.3          # `03_sample_gt_paths.py --floor_tol_m` 기본값과 같게
STAIRS_SPREAD_M = 0.3      # `3dloader_vlnce/03_verify_navmesh_gt.py` MAX_Z_SPREAD_M 과 같게


def depth_outside(free: np.ndarray, xy: np.ndarray, origin, cell: float, step: float) -> float:
    """궤적을 촘촘히 찍었을 때 자유 공간 밖 점의 최대 깊이 [m] (자유 공간 최근접 셀까지 거리)."""
    pts = [xy[0][None]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    pts = np.vstack(pts)
    dist = ndimage.distance_transform_edt(~free) * cell
    ij = world_to_cell(pts, origin, cell)
    h, w = free.shape
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    d = np.full(len(pts), np.inf)
    d[inside] = dist[ij[inside, 1], ij[inside, 0]]
    return float(d.max())


def check(xy, esdf, origin, cell, r_b) -> dict:
    c = check_path_navigable(xy, esdf, origin, cell, r_b)
    c['depth_m'] = depth_outside(esdf > 1e-9, xy, origin, cell, cell / 5.0) if not c['hard_ok'] else 0.0
    return c


def render(path_png, occ_free, nav_free, origin, cell, gt_xy, our_xy, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ext = [origin[0], origin[0] + occ_free.shape[1] * cell, origin[1], origin[1] + occ_free.shape[0] * cell]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, free, name in ((axes[0], occ_free, 'occupancy free'), (axes[1], nav_free, 'navmesh mask')):
        ax.imshow(free, origin='lower', extent=ext, cmap='gray', vmin=0, vmax=1)
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], '-', color='tab:green', lw=1.5, label='GT exec')
        if our_xy is not None:
            ax.plot(our_xy[:, 0], our_xy[:, 1], '-', color='tab:orange', lw=1.2, label='ours')
        ax.plot(*gt_xy[0], 'o', color='tab:blue'); ax.plot(*gt_xy[-1], '*', color='tab:red', ms=12)
        pad = 1.5
        allxy = gt_xy if our_xy is None else np.vstack([gt_xy, our_xy])
        ax.set_xlim(allxy[:, 0].min() - pad, allxy[:, 0].max() + pad)
        ax.set_ylim(allxy[:, 1].min() - pad, allxy[:, 1].max() + pad)
        ax.set_title(name); ax.set_aspect('equal'); ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path_png, dpi=90); plt.close(fig)


def run_scene(args, scene, out_dir, navmesh_grid):
    tag = dataset_utils.dataset_tag(args.dataset)
    data = np.load(Path(args.esdf_dir) / f'{scene}{tag}.npz')
    occupancy, origin = data['occupancy'], data['origin']
    scene_floor, cell = float(data['floor_z']), float(data['voxel_size'])
    # `03.load_esdf`/`plan_episode` 와 같은 규칙 — 02 가 비례형(h_nav_ratio)으로 만들었으면 h_nav = ratio * h_b
    ratio = float(data['h_nav_ratio']) if 'h_nav_ratio' in data and float(data['h_nav_ratio']) > 0 else None

    def h_nav(h_b):
        return ratio * h_b if ratio else float(data['h_nav'])
    paths = json.load(open(Path(args.paths_dir) / f'{scene}{tag}.json'))
    ours = {e['episode_id']: e for e in paths['episodes']}

    nav_cache = {}          # floor_y(2자리 반올림) -> (mask, esdf)

    def nav_grid(floor_y, shape):
        key = round(float(floor_y), 2)
        if key not in nav_cache:
            mask = navmesh_grid.load_mask(args.navmesh_dir, scene, args.r_b, key, origin, cell, shape)
            nav_cache[key] = (mask, navmesh_grid.esdf_from_mask(mask, cell, args.r_b))
        return nav_cache[key]

    rows = []
    for ep in range(args.num_episodes):
        gt = dataset_utils.load_gt_episode(args.data_root, scene, ep, args.dataset, rig=args.rig)
        gt_xyz, h_b, ep_floor = gt['body_xyz'], gt['h_b'], gt['floor_z']
        gt_xy = gt_xyz[:, :2]
        stairs = float(gt_xyz[:, 2].max() - gt_xyz[:, 2].min()) > STAIRS_SPREAD_M
        occ_floor = ep_floor if abs(ep_floor - scene_floor) > FLOOR_TOL_M else scene_floor

        occ_esdf = compute_esdf_2d(derive_obstacle_2d(occupancy, origin, occ_floor, h_b, cell, h_nav(h_b)), cell)
        nav_mask, nav_esdf = nav_grid(ep_floor, occ_esdf.shape)
        row = {'scene': scene, 'episode': ep, 'stairs': stairs, 'floor_occ': occ_floor, 'floor_nav': ep_floor,
               'gt_occ': check(gt_xy, occ_esdf, origin, cell, args.r_b),
               'gt_nav': check(gt_xy, nav_esdf, origin, cell, args.r_b)}

        our_xy = None
        if ep in ours:
            e = ours[ep]
            our_xy = np.asarray(e['trajectory'], dtype=np.float64)[:, :2]
            # json 의 `floor_z` 는 씬 대표 바닥이 기록돼 있어(다층 씬에서 실제 계획 바닥과 다르다) 쓰지 않는다.
            # `03` main 루프와 같은 규칙(occ_floor)·같은 h_b 로 계획했으므로 occ_esdf 가 곧 계획 지도다.
            row['ours_occ'] = check(our_xy, occ_esdf, origin, cell, args.r_b)
            row['ours_nav'] = check(our_xy, nav_esdf, origin, cell, args.r_b)
            row['ours_occ_stored_hard_ok'] = bool(e['check_trajectory']['hard_ok'])
        rows.append(row)

        flags = [k for k in ('gt_occ', 'gt_nav', 'ours_occ', 'ours_nav') if k in row and not row[k]['hard_ok']]
        if flags:
            render(out_dir / f'{scene}_ep{ep:03d}.png', occ_esdf > 1e-9, nav_mask, origin, cell, gt_xy, our_xy,
                   f'{scene} ep{ep}  collide: {", ".join(flags)}' + ('  [stairs]' if stairs else '') +
                   '  depth ' + ' '.join(f'{k}={row[k]["depth_m"]:.2f}' for k in flags))

    if args.grid_out_dir:
        keys = sorted(nav_cache)
        Path(args.grid_out_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(args.grid_out_dir) / f'{scene}_rb{args.r_b:.2f}.npz',
            mask=np.stack([nav_cache[k][0] for k in keys]), esdf=np.stack([nav_cache[k][1] for k in keys]),
            floor_y=np.asarray(keys, dtype=np.float64), origin=np.asarray(origin, dtype=np.float64),
            cell=np.float64(cell), r_b=np.float64(args.r_b), scene=scene,
            note='mask/esdf: (K,Ny,Nx), K = floor_y 별 get_topdown_view 절단. esdf 는 navmesh_grid.esdf_from_mask '
                 '규약(마스크 안 = 경계거리 + r_b) — r_b 를 다시 적용하지 말 것. 격자 = 02 esdf(vln_ce) origin/cell.')
    return rows


def summarize(rows, r_b) -> dict:
    def agg(key, subset):
        rs = [r for r in subset if key in r]
        bad = [r for r in rs if not r[key]['hard_ok']]
        deep = [r for r in bad if r[key]['depth_m'] > 0.05 + 1e-6]
        return {'n': len(rs), 'collide': len(bad), 'deep_gt_1cell': len(deep),
                'max_depth_m': max([r[key]['depth_m'] for r in bad], default=0.0)}

    def cross(a, b, rs):
        return {'a_only': sum((not r[a]['hard_ok']) and r[b]['hard_ok'] for r in rs),
                'b_only': sum(r[a]['hard_ok'] and (not r[b]['hard_ok']) for r in rs),
                'both': sum((not r[a]['hard_ok']) and (not r[b]['hard_ok']) for r in rs)}

    non_stairs = [r for r in rows if not r['stairs']]
    both = [r for r in rows if 'ours_occ' in r]
    out = {'r_b': r_b, 'n_rows': len(rows), 'n_stairs': len(rows) - len(non_stairs)}
    for key in ('gt_occ', 'gt_nav', 'ours_occ', 'ours_nav'):
        out[key] = {'all': agg(key, rows), 'non_stairs': agg(key, non_stairs)}
    out['gt_cross_occ_vs_nav'] = cross('gt_occ', 'gt_nav', rows)
    out['ours_cross_occ_vs_nav'] = cross('ours_occ', 'ours_nav', both)
    out['ours_occ_matches_03'] = all(r['ours_occ']['hard_ok'] == r['ours_occ_stored_hard_ok'] for r in both)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenes', required=True, help='콤마 구분')
    ap.add_argument('--dataset', default='vln_ce', choices=['vln_ce'])
    ap.add_argument('--rig', default='125cm_30deg')
    ap.add_argument('--r_b', type=float, default=0.10)
    ap.add_argument('--num_episodes', type=int, default=20)
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--esdf_dir', default='scripts/dataset_converters/gs_vlnpe/logs/esdf')
    ap.add_argument('--paths_dir', default='scripts/dataset_converters/gs_vlnpe/logs/paths')
    ap.add_argument('--navmesh_dir', default='data/embodiment_aug/navmesh')
    ap.add_argument('--grid_out_dir', default='data/embodiment_aug/navmesh_grid', help='빈 문자열이면 저장 안 함')
    ap.add_argument('--label', default='vln_ce')
    ap.add_argument('--log_dir', default='logs/gs-vlnpe')
    args = ap.parse_args()
    if args.data_root is None:
        args.data_root = dataset_utils.default_data_root(args.dataset)

    navmesh_grid = importlib.import_module('navmesh_grid')
    out_dir = Path(args.log_dir) / SCRIPT_NAME / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in args.scenes.split(','):
        r = run_scene(args, scene, out_dir, navmesh_grid)
        s = summarize(r, args.r_b)
        print(f'[{SCRIPT_NAME}] {scene}: GT occ {s["gt_occ"]["all"]["collide"]}/{s["gt_occ"]["all"]["n"]} '
              f'nav {s["gt_nav"]["all"]["collide"]}/{s["gt_nav"]["all"]["n"]} · ours occ '
              f'{s["ours_occ"]["all"]["collide"]}/{s["ours_occ"]["all"]["n"]} nav '
              f'{s["ours_nav"]["all"]["collide"]}/{s["ours_nav"]["all"]["n"]} · stairs {s["n_stairs"]}', flush=True)
        rows += r

    summary = summarize(rows, args.r_b)
    (out_dir / 'summary.json').write_text(json.dumps({'summary': summary, 'rows': rows}, indent=1, default=float))
    print(json.dumps(summary, indent=1))
    print(f'[{SCRIPT_NAME}] -> {out_dir}/summary.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
