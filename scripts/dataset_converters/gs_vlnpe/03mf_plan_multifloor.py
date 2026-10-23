"""03mf — 층 이동 에피소드를 "층별 2D A* + 계단은 GT 그대로" 로 계획한다 (선택 단계).

    03:   occupancy -> paths/<scene>.json       (평지 계획 — 그대로, 이 스크립트는 03 을 고치지 않는다)
    03mf: occupancy -> paths_mf/<scene>.json    (층 이동 에피소드만)
    03u --multifloor stairs: paths + paths_mf -> usable/   (안 주면 03u 는 지금 그대로 층 이동을 뺀다)

    출발 층 지도에서 A*:  start      -> 계단 입구
    계단 구간 (GT 그대로): 계단 입구 -> 계단 출구   (occupancy 안 씀 — 사족보행 로봇은 계단을 오른다)
    도착 층 지도에서 A*:  계단 출구 -> goal

인자는 **03 과 똑같이** 받는다(03 의 build_argparser 를 그대로 씀) — 확정 03 명령에 `--mf_*` 만 붙이면
같은 설정(r_b, plan_r_b, ②③④, floor_tol_m …)으로 층마다 계획한다. 층 이동 판정은 03u 와 같은 함수·규칙
(`|z_끝 - z_시작| > --mf_dz_m`). 계산은 `multifloor_utils.py`.

실패 사유 (서로 겹치지 않음, 03u 칸 이름)
    mf_entry_blocked  계단 입구/출구를 GT 따라 --mf_snap_max_m 물려도 그 층 지도에서 통행 가능한 점이 없음
                      (plan 모드: 계단 아래·위 GT 점 --mf_snap_max_m 안에 층 높이·여유 충분·층 지도 통행 가능한 면이 없음)
    mf_stairs_no_path (plan 모드) 높이 지도에서 계단 아래 -> 위 경로 없음
    mf_sg_blocked     층 지도에서 A* 의 시작/끝(GT start·goal 또는 물린 입구·출구)이 통행 불가
    mf_astar_failed   층 지도에서 연결 경로 없음
    mf_no_floor_plan  평지 구간이 전부 너무 짧아 계획할 곳이 없음(=GT 복사본이라 새 경로가 아님)
    degenerate        GT 5 프레임 미만

출력 `<out_dir>/paths_mf/<scene><sfx>.json` — 03 json 과 같은 키 + 추가 키
(`trajectory_z` 점마다 바닥 z, `segments` [{kind: floor|stairs, floor_z, i0, i1}], `multifloor` 진단).
`check_trajectory` 는 **계획한 평지 부분만** 그 층 ESDF 로 잰 값(계단은 GT 라 재지 않음).

사용 (vln_ce 확정 설정):
    python 03mf_plan_multifloor.py --dataset vln_ce --rig 125cm_30deg --scene <S> --r_b 0.10 --plan_r_b 0.15 --downsample_mode majority --floor_tol_m 0.1 --post_refine --post_refine_margin 0.05 --thin_safe --smooth bezier --num_episodes 999 --esdf_dir <esdf> --out_dir <03 과 같은 out_dir>
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dataset_utils  # noqa: E402
import multifloor_utils as mfu  # noqa: E402
import stairs_surface  # noqa: E402

_spec = importlib.util.spec_from_file_location('m03', HERE / '03_sample_gt_paths.py')
m03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m03)

SCRIPT_NAME = '03mf_plan_multifloor'


def build_argparser():
    p = m03.build_argparser()
    p.description = __doc__
    d = mfu.MF_DEFAULTS
    p.add_argument('--mf_dz_m', type=float, default=d['dz_m'],
                   help='층 이동 판정 |z끝-z시작| [m]. 03u --dz_m 과 같은 값이어야 목록이 일치한다')
    p.add_argument('--mf_level_tol_m', type=float, default=d['level_tol_m'],
                   help='같은 평지로 볼 바닥 z 흔들림 [m]. GT 평지 흔들림 실측 ±0.08')
    p.add_argument('--mf_plateau_min_m', type=float, default=d['plateau_min_m'],
                   help='이 xy 길이 이상 같은 높이면 평지(그 층에서 A*). 짧은 참은 계단에 흡수')
    p.add_argument('--mf_lead_m', type=float, default=d['lead_m'],
                   help='계단 입구/출구를 평지 쪽으로 최소 이만큼 물린다(이음매가 계단에 걸치지 않게)')
    p.add_argument('--mf_snap_max_m', type=float, default=d['snap_max_m'],
                   help='계단 칸이 지도에서 벽일 때 GT 를 따라 물릴 최대 거리. 넘으면 mf_entry_blocked')
    p.add_argument('--mf_seam_m', type=float, default=d['seam_m'],
                   help='평지 경로와 계단이 만나는 모서리를 이 반폭으로 Bézier 로 깎는다')
    p.add_argument('--mf_min_plan_m', type=float, default=d['min_plan_m'],
                   help='평지 구간이 이보다 짧으면 A* 없이 GT 점을 쓴다(plan 모드: 두 점 직선)')
    p.add_argument('--mf_stairs', default='gt', choices=['gt', 'plan'],
                   help="계단 구간 만드는 법. gt(기본)=GT 궤적 그대로(평지 일부도 GT). "
                        "plan=GT 에서 계단 위치만 쓰고 계단·평지 길을 전부 새로 계획(stairs_surface.py — 높이 지도 A*)")
    p.add_argument('--mf_stairs_spacing_m', type=float, default=0.4,
                   help='plan 모드: 계단 경로(칸 중심 계단 모양)를 Bézier 로 펼 때 waypoint 솎기 간격')
    return p


def episode_entry(ep, h_b, pitch, gt_xy, r, args) -> dict:
    """03 의 episodes[] 항목과 같은 키 + 층 이동 추가 키."""
    traj = r['trajectory']
    return {
        'episode_id': ep, 'mode': args.mode, 'smooth': args.smooth,
        'h_b': h_b, 'pitch_deg': pitch, 'floor_z': r['floor_z_start'],
        'start': gt_xy[0].tolist(), 'goal': gt_xy[-1].tolist(),
        'waypoints_astar': r['waypoints_astar'].tolist(),
        'waypoints_refined': r['waypoints_refined'].tolist(),
        'waypoints_thinned': r['waypoints_thinned'].tolist(),
        'post_refine': r['post_refine'],
        'trajectory': traj.tolist(),
        'path_length_m': float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()),
        'check_refined': r['check_refined'],
        'check_trajectory': r['check_trajectory'],
        'gt_compare': mfu.gt_compare_xy(traj, gt_xy, m03.DOOR_WIDTH_M),
        # --- 층 이동 추가 키 ---
        'trajectory_z': r['trajectory_z'].tolist(),
        'segments': r['segments'],
        'multifloor': {k: r[k] for k in ('n_stairs', 'levels', 'stair_dz_m', 'stair_xy_m',
                                         'snap_shift_m', 'snap_reached_end', 'floors')}
                      | ({'stairs': 'plan', 'stairs_min_clearance_m': r['stairs_min_clearance_m'],
                          'stairs_spacing_used_m': r['stairs_spacing_used_m'], 'seam_used_m': r['seam_used_m']}
                         if args.mf_stairs == 'plan' else {}),
    }


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    for name, val in m03.apply_path_profile(parser, args):
        print(f'[{SCRIPT_NAME}] --path_profile {args.path_profile}: {name} = {val}', flush=True)
    assert args.mode == 'reproduce', '03mf 는 --mode reproduce 만 (GT start/goal 재현)'
    if args.data_root is None:
        args.data_root = dataset_utils.default_data_root(args.dataset)
    print(f'[{SCRIPT_NAME}] scene={args.scene} dataset={args.dataset}')
    loaded = m03.load_esdf(args)
    if loaded is None:
        return 2
    occupancy, origin, floor_z = loaded

    n_avail = len(sorted((Path(args.data_root) / args.scene / 'data' / 'chunk-000').glob('*.parquet')))
    n_total = min(args.num_episodes, n_avail)
    episodes, failures, diag, n_mf = [], {}, {}, 0
    for ep in range(n_total):
        gt_xyz, h_b, pitch, ep_floor = m03.load_gt_episode(args.data_root, args.scene, ep, args.dataset, args.rig)
        if not mfu.is_multifloor(gt_xyz, args.mf_dz_m):
            continue
        n_mf += 1
        if len(gt_xyz) < 5:
            failures[ep] = f'degenerate ({len(gt_xyz)} frames)'
            continue
        if args.mf_stairs == 'plan':
            r = stairs_surface.plan_multifloor_episode_generated(occupancy, origin, floor_z, gt_xyz, h_b, ep_floor,
                                                                 args, m03.plan_episode)
        else:
            r = mfu.plan_multifloor_episode(occupancy, origin, floor_z, gt_xyz, h_b, ep_floor, args, m03.plan_episode)
        if r['status'] != 'ok':
            failures[ep] = r['status']
            diag[ep] = {k: r[k] for k in r if k != 'status'}
            print(f'    ep {ep:>3}: {r["status"]} ({r.get("where", "")}) 층 {r["levels"]}')
            continue
        e = episode_entry(ep, h_b, pitch, gt_xyz[:, :2], r, args)
        episodes.append(e)
        c = e['check_trajectory']
        print(f'    ep {ep:>3}: ok 계단 {r["n_stairs"]} 층 {r["levels"]} 물림 {r["snap_shift_m"]} '
              f'평지 최소여유 {c["points_min_m"]:.3f} frechet {e["gt_compare"]["frechet_m"]:.3f}')

    mins = [e['check_trajectory']['points_min_m'] for e in episodes]
    stats = {
        'n_total': n_total, 'n_multifloor': n_mf, 'n_ok': len(episodes), 'failures': failures,
        'failure_diag': diag,
        'n_stairs_hist': {str(k): int(sum(e['multifloor']['n_stairs'] == k for e in episodes))
                          for k in sorted({e['multifloor']['n_stairs'] for e in episodes})},
        'n_snapped': int(sum(any(x > args.mf_lead_m + 1e-9 for x in e['multifloor']['snap_shift_m'])
                             for e in episodes)),
        'traj_min_m': float(min(mins)) if mins else None,
    }
    out_dir = Path(args.out_dir) / 'paths_mf'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        'scene_id': args.scene, 'mode': args.mode, 'smooth': args.smooth,
        'params': {'r_b': args.r_b, 'plan_r_b': args.plan_r_b, 'refine_radius': args.refine_radius,
                   'cell_m': args.cell_m, 'astar_cell_m': args.cell_m * args.downsample_factor, 'h_nav': args.h_nav,
                   'refine_mode': args.refine_mode, 'downsample_mode': args.downsample_mode,
                   'smooth_step': args.smooth_step, 'floor_z': floor_z, 'h_nav_ratio': args.h_nav_ratio,
                   'waypoint_spacing_m': args.waypoint_spacing_m, 'floor_tol_m': args.floor_tol_m,
                   'post_refine': args.post_refine, 'thin_safe': args.thin_safe,
                   'astar_edge_check': args.astar_edge_check,
                   'multifloor': {k: getattr(args, f'mf_{k}') for k in mfu.MF_DEFAULTS}
                   | ({'stairs': 'plan', 'stairs_spacing_m': args.mf_stairs_spacing_m}
                      if args.mf_stairs == 'plan' else {})},
        'stats': stats,
        'episodes': episodes,
    }
    json_path = out_dir / f'{args.scene}{dataset_utils.dataset_tag(args.dataset)}.json'
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'  paths_mf -> {json_path}')
    print(f'  => 층 이동 {n_mf} · 계획 성공 {len(episodes)} · 실패 사유: {failures}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
