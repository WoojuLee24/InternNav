"""논문 6단계(refine)의 두 해석을 끝까지 돌려 비교한다 — `argmax` vs `min_move`.

    03 : ESDF + GT 조건  -> "경로를 만든다"        -> paths/<scene>.json
    03b: 03의 결과       -> "왜 이만큼 다른가"      -> verify/<scene>.json
    03c: 두 refine 해석  -> "**어느 해석이 맞나**"  -> compare/<scene>.json

논문 6단계는 *"refine the position by maximizing the distance to nearby obstacles"*다.
문자 그대로 읽으면 `argmax`(창 안 ESDF 최대점으로 이동)이고, GT가 약하게 refine된 것으로 측정되어
(03b의 ⑤ R* = 0.05) `min_move`(`r_b`를 만족하는 만큼만 이동)도 후보가 된다.

## 이 스크립트가 답하는 것 — 기준선과 실제 경로가 **반대 방향**으로 움직인다

    항목                       argmax        min_move
    기준선 ⓒ (GT 루트 가정)     0.080/0.109   0.058/0.062   <- min_move가 좋다
    실제 경로 (우리 A* 루트)     0.107/0.154   0.142/0.165   <- argmax가 좋다

`argmax`는 어떤 루트든 **복도 중심선(ESDF 능선)으로 정규화**한다. GT도 refine을 거쳤으므로 능선
근처에 있어서, 우리 A\\*의 계단 모양이 씻겨나가며 GT와 수렴한다. `min_move`는 계단을 그대로 남기므로
격자에서 온 임의의 요철이 GT와의 차이로 남는다.

**따라서 "기준선이 낮다"가 좋은 게 아니다.** 기준선은 파이프라인의 표현 변위를 재는 진단값이고,
변위가 작은 파이프라인이 *다른 루트*를 받았을 때 GT에 더 가까워지는 것은 아니다.

## 실행

    python 03c_compare_refine.py --scene 17DRP5sb8fy
    ... --modes argmax,min_move --refine_radius 0.10 --min_move_radius 0.30

출력: `compare/<scene>.json`, `logs/gs-vlnpe/03c_compare_refine/<scene>/report.html`
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esdf_utils import (  # noqa: E402
    DOWNSAMPLE_FACTOR,
    REFINERS,
    SMOOTHERS,
    ROBOT_RADIUS_M,
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
from geometry_utils import action_to_c2w, decompose_camera_extrinsic, save_jpg  # noqa: E402
from viz_utils import (  # noqa: E402
    FLOOR_COLOR, GT_COLOR, OURS_COLOR, blink_widget_html, floorplan_canvas, save_gallery,
)

DEFAULT_DATA_ROOT = 'data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i'
DEFAULT_OUT_DIR = 'scripts/dataset_converters/gs_vlnpe/apply_real'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe/apply_real'
SCRIPT_NAME = '03c_compare_refine'

ARGMAX_COLOR = (235, 90, 70)      # 빨강
MINMOVE_COLOR = OURS_COLOR        # 초록
WP_COLOR = (255, 255, 255)        # 흰 점 = A* waypoint (refine 전)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene', default='17DRP5sb8fy')
    p.add_argument('--num_episodes', type=int, default=20)
    p.add_argument('--modes', default='argmax,min_move')
    p.add_argument('--refine_radius', type=float, default=0.10, help='argmax 반경 (03 기본값)')
    p.add_argument('--min_move_radius', type=float, default=0.30,
                   help='min_move 반경 — 최소 이동이라 넓게 줘도 멀리 안 간다')
    p.add_argument('--smooth', default='cubic', choices=list(SMOOTHERS))
    p.add_argument('--r_b', type=float, default=ROBOT_RADIUS_M)
    p.add_argument('--waypoint_spacing_m', type=float, default=WAYPOINT_SPACING_M)
    p.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    p.add_argument('--esdf_dir', default=None, help='기본값은 <out_dir>/esdf')
    p.add_argument('--paths_dir', default=None, help='random 모드 경로 GT 디렉터리')
    p.add_argument('--mode', default='reproduce', choices=['reproduce', 'random'])
    p.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    return p


def load_gt(data_root: str, scene: str, episode: int):
    t = pq.read_table(Path(data_root) / scene / 'data' / 'chunk-000' / f'episode_{episode:06d}.parquet',
                      columns=['observation.camera_extrinsic', 'action'])
    e = np.asarray(t['observation.camera_extrinsic'].to_pylist()[0], dtype=np.float64).reshape(4, 4)
    xy = np.stack([action_to_c2w(np.asarray(a, dtype=np.float64).reshape(4, 4), 'cam2world_gl')[:3, 3]
                   for a in t['action'].to_pylist()])[:, :2]
    return xy, decompose_camera_extrinsic(e)[0]


def plan_modes(occ, origin, floor_z, h_b, gt_xy, cell, h_nav, args):
    """모드별로 03과 같은 파이프라인을 돌려 (경로, refined wp, 기준선)을 만든다.

    `wp_astar`는 모드와 무관하게 같다 — 그래야 refine이 각각 어디로 옮기는지 비교할 수 있다.
    """
    coarse = cell * DOWNSAMPLE_FACTOR
    esdf = compute_esdf_2d(derive_obstacle_2d(occ, origin, floor_z, h_b, cell, h_nav), cell)
    navigable = truncate_navigable(esdf, args.r_b) & compute_scan_coverage_mask(occ)
    nav_coarse = downsample_navigable(navigable, DOWNSAMPLE_FACTOR)
    sg = np.array([gt_xy[0], gt_xy[-1]], dtype=np.float64)
    ij = world_to_cell(sg, origin, coarse)
    h_c, w_c = nav_coarse.shape
    ij[:, 0] = np.clip(ij[:, 0], 0, w_c - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, h_c - 1)
    path_ij = astar(nav_coarse, ij[0], ij[1])
    if path_ij is None:
        return None
    wp_astar = cell_to_world(path_ij, origin, coarse)
    wp_astar[0], wp_astar[-1] = sg[0], sg[1]

    out = {'esdf': esdf, 'wp_astar': wp_astar, 'gt': gt_xy, 'h_b': h_b, 'modes': {}}
    for mode in args.modes.split(','):
        radius = args.min_move_radius if mode == 'min_move' else args.refine_radius
        wp = REFINERS[mode](wp_astar, esdf, origin, cell, radius, fix_endpoints=True)
        traj = SMOOTHERS[args.smooth](thin_waypoints(wp, args.waypoint_spacing_m), cell)
        floor = pipeline_floor_path(gt_xy, esdf, origin, cell, coarse, refine_radius_m=radius,
                                   spacing_m=args.waypoint_spacing_m, smooth_step_m=cell,
                                   refine_mode=mode, smooth_mode=args.smooth)
        chk = check_path_navigable(wp, esdf, origin, cell, args.r_b)
        sm_o, sm_g = path_smoothness(traj), path_smoothness(gt_xy)
        out['modes'][mode] = {
            'radius': radius, 'wp_refined': wp, 'traj': traj, 'floor': floor,
            'chamfer_m': chamfer_distance(traj, gt_xy),
            'frechet_m': discrete_frechet_distance(traj, gt_xy),
            'floor_chamfer_m': chamfer_distance(floor, gt_xy),
            'floor_frechet_m': discrete_frechet_distance(floor, gt_xy),
            'wp_move_median_m': float(np.median(np.linalg.norm(wp - wp_astar, axis=1))),
            'refined_min_m': chk['points_min_m'], 'refined_ok': chk['points_ok'],
            'clr_diff_m': compare_clearance_profile(traj, gt_xy, esdf, origin, cell)['median_diff_m'],
            'heading_deg': heading_alignment(traj, gt_xy)['median_deg'],
            'turn_ratio': sm_o['turn_per_m_deg'] / max(sm_g['turn_per_m_deg'], 1e-9),
        }
    return out


def crop_canvas(navigable, origin, cell, paths, out_dir: Path, margin_m=0.6, min_px=760):
    """에피소드 주변만 잘라 확대한 캔버스 — 씬 전체를 그리면 경로가 몇 픽셀이라 판독이 안 된다.

    `navigable` 배열을 먼저 자르고 원점을 옮기면 `floorplan_canvas`를 그대로 쓸 수 있다
    (y 뒤집기가 함수 안에 있으므로 자를 때 뒤집지 않은 배열을 넘겨야 한다).
    """
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


def build_report(args, rows, stats, navigable, origin, cell, out_dir: Path) -> str:
    modes = args.modes.split(',')
    colors = {'argmax': ARGMAX_COLOR, 'min_move': MINMOVE_COLOR}

    # 배수 차이가 가장 큰 에피소드 + 중앙값 에피소드 (겹쳐 그리면 뒤에 그린 것이 덮는다)
    key = lambda r: r['modes']['min_move']['chamfer_m'] - r['modes']['argmax']['chamfer_m'] \
        if 'min_move' in r['modes'] else 0.0
    order = sorted(rows, key=lambda r: -key(r))
    picks = [('차이 최대', order[0]), ('중앙값', order[len(order) // 2])]

    body = ''
    for tag, r in picks:
        ep = r['episode']
        base, draw, emit = crop_canvas(
            navigable, origin, cell,
            [r['gt'], r['wp_astar']] + [r['modes'][m]['traj'] for m in modes]
            + [r['modes'][m]['floor'] for m in modes], out_dir)
        # ① 실제 경로 비교
        g = draw(base.copy(), r['gt'], GT_COLOR, 2)
        states = [('① GT (노랑)', emit(g, f'ep{ep}_gt.jpg'))]
        cur = g.copy()
        for m in modes:
            cur = draw(cur.copy(), r['modes'][m]['traj'], colors[m], 2)
            states.append((f'{"②" if m == modes[0] else "③"} + {m} 경로 '
                           f'({"빨강" if m == "argmax" else "초록"}) — chamfer '
                           f'{r["modes"][m]["chamfer_m"]:.3f} m', emit(cur, f'ep{ep}_{m}.jpg')))
        wp_img = draw(cur.copy(), r['wp_astar'], WP_COLOR, 2)
        states.append(('④ + A* waypoint (흰 점) — refine 전 위치',
                       emit(wp_img, f'ep{ep}_wp.jpg')))
        body += (f'<h4>ep {ep} — {tag} (chamfer argmax {r["modes"][modes[0]]["chamfer_m"]:.3f} vs '
                 f'min_move {r["modes"][modes[-1]]["chamfer_m"]:.3f} m)</h4>'
                 + blink_widget_html(f'traj{ep}', states, title=f'ep {ep} 실제 경로'))

        # ② 기준선 비교 (GT 루트를 가정했을 때)
        fs = [('① GT (노랑)', emit(g, f'ep{ep}_fgt.jpg'))]
        cur = g.copy()
        for m in modes:
            cur = draw(cur.copy(), r['modes'][m]['floor'], colors[m], 2)
            fs.append((f'+ {m} 기준선 — chamfer {r["modes"][m]["floor_chamfer_m"]:.3f} m',
                       emit(cur, f'ep{ep}_f{m}.jpg')))
        body += ('<p class="note">같은 에피소드의 <b>기준선</b>(GT 루트를 가정) — 여기서는 min_move가 '
                 'GT에 더 붙는다. 위 실제 경로와 <b>순서가 반대</b>인 것이 이 스크립트의 요점이다.</p>'
                 + blink_widget_html(f'floor{ep}', fs, title=f'ep {ep} 기준선'))

    head = ''.join(f'<th>{m}</th>' for m in modes)
    def row(label, fmt, pick):
        return (f'<tr><td>{label}</td>'
                + ''.join(f'<td>{fmt.format(pick(stats[m]))}</td>' for m in modes) + '</tr>')
    table = f'''
<table class="table">
  <tr><th>지표 (median)</th>{head}</tr>
  {row('<b>실제 경로 chamfer</b> [m]', '{:.4f}', lambda d: d['chamfer_m'])}
  {row('<b>실제 경로 Fréchet</b> [m]', '{:.4f}', lambda d: d['frechet_m'])}
  {row('기준선 ⓒ chamfer [m]', '{:.4f}', lambda d: d['floor_chamfer_m'])}
  {row('기준선 ⓒ Fréchet [m]', '{:.4f}', lambda d: d['floor_frechet_m'])}
  {row('실제/기준선 배수', '{:.2f}×', lambda d: d['ratio'])}
  {row('refine 이동거리 [m]', '{:.4f}', lambda d: d['wp_move_median_m'])}
  {row('refined wp 최소 clearance [m]', '{:.3f}', lambda d: d['refined_min_m'])}
  {row('refined wp가 r_b 이상인 에피소드', '{}', lambda d: d['n_refined_ok'])}
  {row('② clearance 차 [m]', '{:+.3f}', lambda d: d['clr_diff_m'])}
  {row('③ 헤딩 [도]', '{:.1f}', lambda d: d['heading_deg'])}
  {row('① 회전배율', '{:.2f}×', lambda d: d['turn_ratio'])}
</table>'''

    a, b = stats[modes[0]], stats[modes[-1]]
    verdict = ('argmax' if a['chamfer_m'] <= b['chamfer_m'] else 'min_move')
    summary = f'''
<div class="stat-row">
  <div class="stat"><b>실제 경로 우세</b><span class="pill good">{verdict}</span></div>
  <div class="stat"><b>기준선 우세</b><span class="pill warn">
      {'argmax' if a['floor_chamfer_m'] <= b['floor_chamfer_m'] else 'min_move'}</span></div>
  <div class="stat"><b>에피소드</b><span class="pill">{stats[modes[0]]['n_ok']}</span></div>
</div>
<p>논문 6단계 <i>"maximizing the distance to nearby obstacles"</i>의 두 해석을 비교한다.
<b>기준선과 실제 경로가 반대 방향으로 움직인다</b> — 기준선은 min_move가 GT에 더 붙는데
실제 경로는 argmax가 더 붙는다.</p>
<p><b>이유</b>: <code>argmax</code>는 어떤 루트든 <b>복도 중심선(ESDF 능선)으로 정규화</b>한다.
GT도 refine을 거쳐 능선 근처에 있으므로, 우리 A*의 계단 모양이 씻겨나가며 GT와 수렴한다.
<code>min_move</code>는 계단을 그대로 남기므로 격자에서 온 임의의 요철이 GT와의 차이로 남는다.
기준선은 <b>GT 자신의 셀 시퀀스</b>에서 출발하므로 안 움직이는 쪽이 유리한데, 그 이점은 우리가
GT 루트를 모를 때 쓸 수 없다.</p>
<p><b>따라서 "기준선이 낮다"가 좋은 것이 아니다.</b> 기준선은 파이프라인의 표현 변위를 재는
진단값이고, 변위가 작은 파이프라인이 <i>다른 루트</i>를 받았을 때 GT에 더 가까워지는 것은 아니다.</p>
{table}
<p class="note">게이트 참고: <code>min_move</code>는 이미 <code>r_b</code>를 만족하는 waypoint를
움직이지 않으므로 03의 게이트 ②("refine이 clearance를 올렸는가")를 통과하지 못한다.
반경에는 둔감하다(0.15/0.30/0.60에서 수치 동일) — <code>r_b</code>를 채울 셀이 늘 근처에 있기 때문이다.</p>
'''
    return summary, body


def main() -> int:
    args = build_argparser().parse_args()
    modes = args.modes.split(',')
    print(f'[{SCRIPT_NAME}] scene={args.scene} modes={modes}')

    npz_path = Path(args.esdf_dir or (Path(args.out_dir) / 'esdf')) / f'{args.scene}.npz'
    if not npz_path.is_file():
        print(f'  [ERROR] {npz_path} 없음 — 02를 먼저 돌릴 것')
        return 2
    data = np.load(npz_path)
    occ, origin, floor_z = data['occupancy'], data['origin'], float(data['floor_z'])
    cell, h_nav = float(data['voxel_size']), float(data['h_nav'])

    random_eps = None
    if args.mode == 'random':
        paths_dir = Path(args.paths_dir or (Path(args.out_dir) / 'paths'))
        random_eps = json.load(open(paths_dir / f'{args.scene}_random.json'))['episodes']
        n_avail = len(random_eps)
    else:
        n_avail = len(sorted((Path(args.data_root) / args.scene / 'data' / 'chunk-000').glob('*.parquet')))
    rows, nav_ref = [], None
    for ep in range(min(args.num_episodes, n_avail)):
        if random_eps is not None:
            gt_xy = np.asarray(random_eps[ep]['trajectory'], dtype=np.float64)
            h_b = float(random_eps[ep]['h_b'])
        else:
            gt_xy, h_b = load_gt(args.data_root, args.scene, ep)
        r = plan_modes(occ, origin, floor_z, h_b, gt_xy, cell, h_nav, args)
        if r is None:
            print(f'    ep {ep:>3}: astar_failed')
            continue
        if nav_ref is None:
            nav_ref = truncate_navigable(r['esdf'], args.r_b)
        r['episode'] = ep
        rows.append(r)
        print(f'    ep {ep:>3}: ' + '  '.join(
            f'{m}: 경로 {r["modes"][m]["chamfer_m"]:.3f} 기준선 {r["modes"][m]["floor_chamfer_m"]:.3f} '
            f'이동 {r["modes"][m]["wp_move_median_m"]:.3f}' for m in modes))

    if not rows:
        print('  [ERROR] 성공 0건')
        return 1

    med = lambda m, k: float(np.median([r['modes'][m][k] for r in rows]))
    stats = {m: {
        'n_ok': len(rows), 'radius': rows[0]['modes'][m]['radius'],
        'chamfer_m': med(m, 'chamfer_m'), 'frechet_m': med(m, 'frechet_m'),
        'floor_chamfer_m': med(m, 'floor_chamfer_m'), 'floor_frechet_m': med(m, 'floor_frechet_m'),
        'wp_move_median_m': med(m, 'wp_move_median_m'),
        'refined_min_m': float(min(r['modes'][m]['refined_min_m'] for r in rows)),
        'n_refined_ok': f"{sum(r['modes'][m]['refined_ok'] for r in rows)}/{len(rows)}",
        'clr_diff_m': med(m, 'clr_diff_m'), 'heading_deg': med(m, 'heading_deg'),
        'turn_ratio': med(m, 'turn_ratio'),
    } for m in modes}
    for m in modes:
        stats[m]['ratio'] = stats[m]['chamfer_m'] / max(stats[m]['floor_chamfer_m'], 1e-9)

    log_dir = Path(args.log_dir) / SCRIPT_NAME / args.scene
    summary, body = build_report(args, rows, stats, nav_ref, origin, cell, log_dir)
    out_dir = Path(args.out_dir) / 'compare'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{args.scene}.json').write_text(json.dumps({
        'scene_id': args.scene, 'stats': stats, 'params': vars(args),
        'episodes': [{'episode': r['episode'], 'h_b': r['h_b'],
                      'modes': {m: {k: v for k, v in r['modes'][m].items()
                                    if k not in ('wp_refined', 'traj', 'floor')}
                                for m in modes}} for r in rows]},
        indent=2, ensure_ascii=False), encoding='utf-8')
    report = save_gallery(log_dir, 'report.html',
                          f'03c_compare_refine — {args.scene}', summary, body)
    print(f'  compare -> {out_dir / f"{args.scene}.json"}')
    print(f'  report html -> {report}')
    for m in modes:
        s = stats[m]
        print(f'  {m:>9}: 경로 chamfer {s["chamfer_m"]:.4f} Fréchet {s["frechet_m"]:.4f} | '
              f'기준선 {s["floor_chamfer_m"]:.4f} ({s["ratio"]:.2f}x) | '
              f'refine 이동 {s["wp_move_median_m"]:.4f} | r_b 통과 {s["n_refined_ok"]}')
    best = min(modes, key=lambda m: stats[m]['chamfer_m'])
    print(f'  => 실제 경로 기준 우세: {best}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
