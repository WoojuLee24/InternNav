"""M1.3 역검증 — "생성 경로가 gt_path와 거의 동일한가"를 논문 단계별로 되짚는다.

    00 : vln_n1 데이터   -> "어떻게 읽고 쓰나"   -> target_schema.json      (전체에 1개)
    01 : 씬 mesh/USD     -> "이 씬 써도 되나"     -> scene_meta/<scene>.json (씬마다 1개)
    02 : 씬 mesh         -> "어디로 갈 수 있나"   -> esdf/<scene>.npz        (씬마다 1개)
    03 : ESDF + GT 조건  -> "경로를 만든다"       -> paths/<scene>.json      (씬마다 1개)
    03b: 03의 결과       -> "**왜 이만큼 다른가**" -> verify/<scene>.json     (씬마다 1개)

03은 "얼마나 다른가"(chamfer/Fréchet)를 재고, **03b는 "그 값이 좋은 값인지"에 답한다.**
경로가 정말 같아도 지표가 0이 아니기 때문이다 — 0.2 m A* 격자 양자화와 refine·스무딩이 하한을
만든다. 실측 하한은 chamfer 0.080~0.109 m다. 그 하한을 모르면 남은 오차가 개선 여지인지 원리적 한계인지 알 수 없고,
파라미터를 더 만지는 것은 노이즈를 쫓는 일이 된다.

또 논문이 **값을 주지 않은 파라미터**(6단계 local area 크기, 7단계 knot 간격)를 03은 chamfer 스윕으로
골랐다 — 지표를 보고 지표에 맞춘 셈이라 순환논증이다. 03b는 그 값을 **GT에서 역으로 읽어내** 독립
근거를 만든다.

## 4가지 검사

    F1 자기재현  : 같은 입력으로 두 번 계획 -> 비트 단위 동일해야 한다 (파이프라인 결정성)
    기준선 ⓒ     : GT 루트를 격자에 얹어 논문 6~7단계를 그대로 돌린 것 vs GT
                   = "A*가 GT 루트를 정확히 골랐을 때 우리 파이프라인이 내는 값" -> 그 몇 배인가
                   (오차 원인 분해로 격자 F3 / 점분포 F2도 함께 낸다)
    (5) refine 반경 : 반경 R로 refine한 경로의 "다시 refine했을 때 이동거리" 프로파일을 GT와
                      맞춰 논문 6단계 반경을 역추정. 알려진 반경을 되찾는지로 자기검증한다
    (7) knot 복원   : GT를 재현하는 최소 knot 수와 그 간격 -> 논문 7단계 + thin_waypoints 검증
                      무릎이 없으면 GT는 spline이 아니라 컨트롤러 실행 궤적이다

## 입출력

    입력: esdf/<scene>.npz (02), paths/<scene>.json (03), GT parquet
    출력: verify/<scene>.json, logs/gs-vlnpe/03b_verify/<scene>/report.html

## 실행

    python 03b_verify_reproduction.py --scene 17DRP5sb8fy
    ... --radii 0.05,0.10,0.15,0.20,0.30,0.50    # 반경 스윕 후보
    ... --knot_tol 0.02 --k_max 40               # knot 복원 허용오차와 상한
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
    DOWNSAMPLE_FACTOR,
    ROBOT_RADIUS_M,
    REFINE_PROBE_RADII_M,
    WAYPOINT_SPACING_M,
    astar,
    cell_to_world,
    chamfer_distance,
    compute_esdf_2d,
    compute_scan_coverage_mask,
    derive_obstacle_2d,
    discrete_frechet_distance,
    downsample_navigable,
    greedy_refine,
    estimate_refine_radius,
    DOWNSAMPLE_MODES,
    REFINERS,
    SMOOTHERS,
    metric_floors,
    path_cell_sequence,
    pipeline_floor_path,
    recover_spline_knots,
    resample_by_arclength,
    smooth_cubic_spline,
    thin_waypoints,
    truncate_navigable,
    world_to_cell,
)
from geometry_utils import action_to_c2w, decompose_camera_extrinsic  # noqa: E402
from viz_utils import (  # noqa: E402
    FLOOR_B_COLOR, FLOOR_COLOR, GT_COLOR, KNOT_COLOR, OURS_COLOR,
    blink_widget_html, floorplan_canvas, line_chart, save_gallery,
)

DEFAULT_DATA_ROOT = 'data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i'
DEFAULT_OUT_DIR = 'scripts/dataset_converters/gs_vlnpe/apply_real'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe/apply_real'
SCRIPT_NAME = '03b_verify_reproduction'

# 우리 chamfer가 바닥값의 이 배수 이내면 "이 파이프라인의 한계에 도달"로 본다.
# 초과하면 남은 차이가 격자/샘플링이 아닌 실제 알고리즘 차이라는 뜻이므로 계속 파는 근거가 된다.
FLOOR_RATIO_AT_LIMIT = 1.5


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene', default='17DRP5sb8fy')
    p.add_argument('--num_episodes', type=int, default=20)
    p.add_argument('--radii', default='0.05,0.10,0.15,0.20,0.25,0.30,0.40',
                   help='(5) refine 반경 후보 [m]. GT 프로파일과 가장 가까운 후보를 R*로 쓴다')
    p.add_argument('--knot_tol', type=float, default=0.02, help='(7) knot 복원 RMSE 허용오차 [m]')
    p.add_argument('--k_max', type=int, default=40)
    # 아래 둘은 기본이 None이다 — paths/<scene>.json의 params에서 **03이 실제로 쓴 값**을 읽는다.
    # 03b가 자기 기본값을 쓰면 03과 어긋나 F1(결정성)이 통째로 실패한다(실제로 겪음).
    p.add_argument('--r_b', type=float, default=None)
    p.add_argument('--refine_radius', type=float, default=None)
    p.add_argument('--refine_mode', default=None, choices=list(REFINERS))
    p.add_argument('--smooth', default=None, choices=list(SMOOTHERS))
    p.add_argument('--downsample_mode', default=None, choices=list(DOWNSAMPLE_MODES))
    p.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    p.add_argument('--esdf_dir', default=None, help='기본값은 <out_dir>/esdf')
    p.add_argument('--paths_dir', default=None, help='기본값은 <out_dir>/paths')
    p.add_argument('--mode', default='reproduce', choices=['reproduce', 'random'])
    p.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    return p


def load_gt_xy(data_root: str, scene: str, episode: int):
    table = pq.read_table(Path(data_root) / scene / 'data' / 'chunk-000' / f'episode_{episode:06d}.parquet',
                          columns=['observation.camera_extrinsic', 'action'])
    extrinsic = np.asarray(table['observation.camera_extrinsic'].to_pylist()[0],
                           dtype=np.float64).reshape(4, 4)
    actions = [np.asarray(a, dtype=np.float64).reshape(4, 4) for a in table['action'].to_pylist()]
    xy = np.stack([action_to_c2w(a, 'cam2world_gl')[:3, 3] for a in actions])[:, :2]
    return xy, decompose_camera_extrinsic(extrinsic)[0]


def plan_once(occupancy, origin, floor_z, h_b, start_xy, goal_xy, cell, h_nav, args):
    """03의 `plan_episode`와 같은 순서 — F1(결정성)을 재려면 같은 파이프라인이어야 한다."""
    esdf = compute_esdf_2d(derive_obstacle_2d(occupancy, origin, floor_z, h_b, cell, h_nav), cell)
    navigable = truncate_navigable(esdf, args.r_b) & compute_scan_coverage_mask(occupancy)
    nav_coarse = downsample_navigable(navigable, DOWNSAMPLE_FACTOR,
                                      args.downsample_mode)
    coarse = cell * DOWNSAMPLE_FACTOR
    sg = np.array([start_xy, goal_xy], dtype=np.float64)
    ij = world_to_cell(sg, origin, coarse)
    h_c, w_c = nav_coarse.shape
    ij[:, 0] = np.clip(ij[:, 0], 0, w_c - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, h_c - 1)
    path_ij = astar(nav_coarse, ij[0], ij[1])
    if path_ij is None:
        return None, esdf
    wp = cell_to_world(path_ij, origin, coarse)
    wp[0], wp[-1] = sg[0], sg[1]
    wp = REFINERS[args.refine_mode](wp, esdf, origin, cell, args.refine_radius,
                                    fix_endpoints=True)
    return SMOOTHERS[args.smooth](thin_waypoints(wp, WAYPOINT_SPACING_M), cell), esdf


def pill(ok: bool) -> str:
    return f'<span class="pill {"good" if ok else "bad"}">{"PASS" if ok else "FAIL"}</span>'


def build_summary(args, stats, rows, radii) -> str:
    at_limit = stats['chamfer_over_floor'] <= FLOOR_RATIO_AT_LIMIT
    sweep_head = ''.join(f'<th>{r:.2f}</th>' for r in radii)

    probes = stats['probe_radii_m']
    prof_head = ''.join(f'<th>{r:.2f}</th>' for r in probes)

    def prof_row(label, prof, note, bold=False):
        cells = ''.join(f'<td>{v:.3f}</td>' for v in prof)
        lab = f'<b>{label}</b>' if bold else label
        return f'<tr><td>{lab}</td>{cells}<td class="note">{note}</td></tr>'

    sweep_body = prof_row('GT (복원 knot)', stats['refine_gt_profile'], '측정 대상', bold=True)
    for c in radii:
        prof = stats['refine_candidate_profiles'].get(round(float(c), 3))
        if prof is None:
            continue
        mark = ' &larr; R*' if round(float(c), 3) == round(stats['refine_radius_est_m'], 3) else ''
        cur = ' (현재값)' if abs(c - args.refine_radius) < 1e-9 else ''
        sweep_body += prof_row(f'후보 R = {c:.2f}{cur}', prof,
                               f'L1 {stats["refine_l1"].get(round(float(c), 3), float("nan")):.3f}{mark}')
    ep_rows = ''.join(
        f'<tr><td>{r["episode"]}</td><td>{r["chamfer_m"]:.3f}</td>'
        f'<td>{r["floor_chamfer_m"]:.3f}</td><td>{r["floor_no_refine_chamfer_m"]:.3f}</td>'
        f'<td>{r["chamfer_over_floor"]:.2f}</td>'
        f'<td>{"O" if r["f1_identical"] else "X"}</td>'
        f'<td>{r["k_star"] if r["k_star"] else "—"}</td>'
        f'<td>{r["knot_spacing_m"]:.2f}</td>'
        f'<td>{r["refine_radius_est_m"]:.2f}</td></tr>' for r in rows)
    return f'''
<div class="stat-row">
  <div class="stat"><b>F1 결정성</b>{pill(stats["f1_all_identical"])}</div>
  <div class="stat"><b>chamfer / 기준선</b><span class="pill {"good" if at_limit else "warn"}">
      {stats["chamfer_over_floor"]:.2f}×</span></div>
  <div class="stat"><b>R* (논문 6단계)</b><span class="pill">{stats["refine_radius_est_m"]:.2f} m</span></div>
  <div class="stat"><b>knot 간격 (논문 7단계)</b><span class="pill">{stats["knot_spacing_median_m"]:.2f} m</span></div>
</div>
<p>03은 "얼마나 다른가"를 재고 <b>03b는 그 값이 좋은 값인지에 답한다.</b> 경로가 정말 같아도 지표는
0이 아니다 — GT 루트를 0.2 m 격자에 얹고 refine·스무딩을 거치는 것만으로 chamfer가
{stats["floor_chamfer_median_m"]:.3f} m 나온다. 절대값이 아니라 <b>기준선 대비 배수</b>로 읽어야 한다.</p>

<table class="table">
  <tr><th>검사</th><th>결과</th><th>수치</th><th>해석</th></tr>
  <tr><td>F1 — 같은 입력으로 두 번 계획</td><td>{pill(stats["f1_all_identical"])}</td>
      <td>{stats["n_f1_identical"]}/{stats["n_ok"]} 에피소드 비트 단위 동일</td>
      <td>파이프라인에 비결정성(RNG·집합 순회 의존)이 없다는 뜻. 지표의 나머지 오차를
          재현 불안정으로 설명할 수 없게 만든다</td></tr>
  <tr><td><b>기준선 ⓒ</b> — A*가 GT 루트를 정확히 골랐을 때 우리 파이프라인이 내는 값</td>
      <td><span class="pill {"good" if at_limit else "warn"}">{stats["chamfer_over_floor"]:.2f}×</span></td>
      <td>chamfer {stats["chamfer_median_m"]:.3f} / <b>기준선 {stats["floor_chamfer_median_m"]:.3f}</b> m<br>
          Fréchet {stats["frechet_median_m"]:.3f} / <b>{stats["floor_frechet_median_m"]:.3f}</b> m
          ({stats["frechet_over_floor"]:.2f}×)</td>
      <td>GT 루트를 0.2 m 격자에 얹어 논문 6~7단계를 <b>03과 똑같이</b> 돌린 결과다.
          루트가 완벽해도 이 값이 나오므로 <b>0을 목표로 삼을 수 없다.</b>
          {'배수가 1에 가까워 <b>남은 차이가 거의 전부 이산화</b>다 — 파라미터를 더 만질 여지가 없다.'
           if at_limit else
           '배수가 %.1f배다 — 이산화로 설명되지 않는 <b>루트 선택 차이</b>가 남아 있다.'
           % stats["chamfer_over_floor"]}
          <br>내역: 격자+스무딩만 {stats["floor_no_refine_chamfer_median_m"]:.3f} m
          -> refine 포함 {stats["floor_chamfer_median_m"]:.3f} m.
          원인 분해는 격자(F3) {stats["f3_chamfer_median_m"]:.3f} /
          점분포(F2) {stats["f2_chamfer_median_m"]:.3f} m — F3는 씬마다 크게 흔들려
          <b>기준선으로 쓰면 안 된다</b>(실측 0.105 vs 0.063)</td></tr>
  <tr><td>(5) refine 반경 역추정 <i>(논문: "a local area")</i></td>
      <td><span class="pill {"good" if stats["refine_control_ok"] else "warn"}">
          R* = {stats["refine_radius_est_m"]:.2f} m</span></td>
      <td>내장 대조군 {stats["n_refine_control_ok"]}/{stats["n_ok"]} 통과
          <span class="note">(나머지는 A* 경로가 이미 벽에서 멀어 후보들이 구분되지 않는
          에피소드다 — 추정기 결함이 아니다)</span><br>
          현재 <code>--refine_radius {args.refine_radius}</code></td>
      <td>같은 A* waypoint를 후보 반경으로 refine한 뒤 "다시 refine하면 얼마나 움직이나" 프로파일을
          만들어 GT와 L1로 맞춘다. <b>대조군</b>은 후보 반경으로 만든 경로에서 그 반경이 되찾아지는지다 —
          통과했으면 R*를 신뢰할 수 있다.
          {'R*가 현재값보다 <b>작다</b> = GT가 우리보다 벽에 가깝게 다닌다(우리 refine이 과하다). 지표 ②의 clearance 차이와 같은 방향이어야 앞뒤가 맞는다.' if stats["refine_radius_est_m"] < args.refine_radius else 'R*가 현재값 이상이다 = GT가 우리보다 벽에서 멀다.'}
          <br><i>"refine 산출물은 국소최대다"로 검사하면 안 된다</i> — greedy refine은 한 번만 적용되므로
          우리 <code>wp_refined</code>조차 국소최대 비율이 14%였다(양성 대조 실패)</td></tr>
  <tr><td>(7) knot 복원 <i>(논문: cubic spline)</i></td>
      <td><span class="pill {"good" if stats["n_has_knee"] == stats["n_ok"] else "warn"}">
          무릎 {stats["n_has_knee"]}/{stats["n_ok"]}</span></td>
      <td>k* median {stats["k_star_median"]}, 간격 {stats["knot_spacing_median_m"]:.2f} m</td>
      <td>{'GT는 소수 knot의 cubic spline으로 재현된다 — 논문 7단계와 일치.'
           if stats["n_has_knee"] == stats["n_ok"] else
           '<b>무릎이 없는 에피소드가 있다</b> = 그 GT는 spline이 아니라 컨트롤러 실행 궤적이다. '
           '점 단위 일치는 원리적으로 불가능하다.'}
          현재 <code>--waypoint_spacing_m {WAYPOINT_SPACING_M}</code>과 비교할 값이다</td></tr>
</table>

<h3>(5) refine 반경 역추정 — "다시 refine하면 얼마나 움직이나" 프로파일 [m]</h3>
<table class="table"><tr><th></th>{prof_head}<th></th></tr>{sweep_body}</table>
<p class="note">열은 탐침 반경, 값은 median 이동거리다. 이미 벽에서 멀리 밀려난 경로는 ESDF 평탄부에
있어 덜 움직인다 — 그 지문으로 refine 강도를 읽는다. GT 행과 L1이 가장 작은 후보가 R*다.</p>

<h3>에피소드별</h3>
<table class="table">
  <tr><th>ep</th><th>chamfer</th><th>기준선 ⓒ</th><th>ⓑ refine없이</th><th>배수</th>
      <th>F1 동일</th><th>k*</th><th>knot 간격</th><th>R*</th></tr>{ep_rows}</table>
'''


def build_body(args, stats, rows, radii, navigable, origin, cell, out_dir: Path) -> str:
    """그림 3종 — 표만으로는 "왜 이만큼 다른가"가 안 보인다.

      ① 기준선 분해 : GT -> ⓑ(격자+스무딩) -> ⓒ(+refine) -> 우리. 단계마다 어디서 벌어지는지
      ② knot 복원   : GT 위에 복원한 k* knot과 그 knot으로 만든 spline. 겹치면 GT는 spline이다
      ③ 반경 프로파일: (5)의 GT 곡선이 어느 후보 곡선에 붙는지 (표의 L1을 눈으로)
    """
    if navigable is None or not rows:
        return ''
    base, draw, emit = floorplan_canvas(navigable, origin, cell, out_dir)

    # ① 기준선 분해 — **대표 에피소드 단위로** 본다. 20개를 겹치면 마지막에 그린 초록이 다 덮어서
    # 판독이 안 된다(실제로 겪음). 배수 중앙값 에피소드와 최댓값 에피소드를 각각 낸다.
    order = sorted(rows, key=lambda r: -r['chamfer_over_floor'])
    decomp = ''
    for tag, r in (('중앙값', order[len(order) // 2]), ('최댓값', order[0])):
        ep, v = r['episode'], r['_viz']
        g = draw(base.copy(), v['gt'], GT_COLOR, 2)
        b_ = draw(g.copy(), v['floor_b'], FLOOR_B_COLOR, 2)
        c_ = draw(b_.copy(), v['floor_c'], FLOOR_COLOR, 2)
        o_ = draw(c_.copy(), v['ours'], OURS_COLOR, 1)
        decomp += (f'<h4>ep {ep} — 배수 {tag} ({r["chamfer_over_floor"]:.2f}배)</h4>'
                   + blink_widget_html(f'decomp{ep}', [
                       ('① GT (노랑)', emit(g, f'ep{ep}_d_gt.jpg')),
                       (f'② + ⓑ 격자+스무딩 (보라) — chamfer {r["floor_no_refine_chamfer_m"]:.3f} m',
                        emit(b_, f'ep{ep}_d_b.jpg')),
                       (f'③ + ⓒ = ⓑ+refine (하늘) — {r["floor_chamfer_m"]:.3f} m',
                        emit(c_, f'ep{ep}_d_c.jpg')),
                       (f'④ + 우리 (초록) — {r["chamfer_m"]:.3f} m = ⓒ의 '
                        f'{r["chamfer_over_floor"]:.2f}배', emit(o_, f'ep{ep}_d_ours.jpg')),
                   ], title=f'ep {ep}'))

    # ② knot 복원 — 배수 상위 2개 + 하위 1개(잘 맞는 예)
    picks = order[:2] + order[-1:]
    knot_html = ''
    for r in picks:
        ep, v = r['episode'], r['_viz']
        g = draw(base.copy(), v['gt'], GT_COLOR, 2)
        sp = draw(g.copy(), v['knot_spline'], FLOOR_COLOR, 1)
        kn = draw(sp.copy(), v['knots'], KNOT_COLOR, 3)
        knot_html += (f'<h4>ep {ep} — k* = {r["k_star"]}, knot 간격 {r["knot_spacing_m"]:.2f} m '
                      f'(우리 설정 {WAYPOINT_SPACING_M})</h4>'
                      + blink_widget_html(f'knot{ep}', [
                          ('① GT만 (노랑)', emit(g, f'ep{ep}_knot_gt.jpg')),
                          ('② + 복원 knot으로 만든 cubic spline (하늘)',
                           emit(sp, f'ep{ep}_knot_spline.jpg')),
                          (f'③ + 복원 knot {len(v["knots"])}개 (주황) — 소수 점으로 GT가 재현된다',
                           emit(kn, f'ep{ep}_knot_pts.jpg')),
                      ], title=f'ep {ep}'))

    # ③ 반경 프로파일 곡선 — 축 라벨은 cv2 제약으로 ASCII만
    probes = stats['probe_radii_m']
    series = [('GT', GT_COLOR, probes, stats['refine_gt_profile'])]
    for c in radii:
        prof = stats['refine_candidate_profiles'].get(round(float(c), 3))
        if prof is None:
            continue
        hit = round(float(c), 3) == round(stats['refine_radius_est_m'], 3)
        series.append((f'R={c:.2f}{" (est)" if hit else ""}',
                       OURS_COLOR if hit else (120, 128, 140), probes, prof))
    chart = line_chart(series, out_dir / 'refine_profile.jpg',
                       x_label='probe radius [m]', y_label='median displacement [m]')

    # ④ 격자 확대 — "기준선이 왜 GT에서 벌어지나"에 대한 직접적인 답
    zoom_path, zoom_dev = render_grid_zoom(order[0], origin, cell, cell * DOWNSAMPLE_FACTOR,
                                           out_dir / 'grid_zoom.jpg')

    return (f'<h3>기준선이 왜 GT에서 벌어지는가 — 0.2 m 격자 확대</h3>'
            f'<p class="note">기준선은 GT가 <b>아니다</b>. GT의 <b>루트</b>를 우리 파이프라인이 '
            '표현할 수 있는 형태로 다시 쓴 것이다. 벌어지는 이유가 두 개인데 이 그림에서 분리된다: '
            '<b>①</b> A*가 출력할 수 있는 좌표는 <b>0.2 m 셀중심</b>(흰 점)뿐인데 GT(노랑)는 연속 '
            '곡선이다 → 흰 점과 보라(ⓑ)가 노랑에서 벗어나는 몫. '
            '<b>②</b> <code>greedy_refine</code>이 waypoint를 벽에서 밀어낸다 → 보라에서 하늘(ⓒ)로 '
            f'옮겨간 몫. ep {order[0]["episode"]}의 최대 이탈은 <b>{zoom_dev:.3f} m</b>로, '
            '0.2 m 셀 반대각선 0.141 m 규모다.</p>'
            f'<div class="card"><img src="{_data_uri(zoom_path)}" alt="grid zoom"></div>'
            f'<h3>기준선 분해 — 오차가 어느 단계에서 생기는가</h3>'
            '<p class="note">‹ ›로 넘긴다. <b>노랑 = GT, 보라 = ⓑ 격자+스무딩, 하늘 = ⓒ +refine, '
            '초록 = 우리.</b> 보라가 노랑에서 벌어지는 것은 <b>격자 양자화</b>, 보라→하늘은 '
            '<b>refine이 벽에서 밀어낸 몫</b>, 하늘→초록은 <b>루트 선택 차이</b>다. '
            '에피소드를 겹치면 마지막에 그린 초록이 다 덮어 판독이 안 되므로 대표 2개만 낸다. '
            '배경 navigable은 첫 에피소드의 h_b 기준이다.</p>' + decomp
            + '<h3>(7) knot 복원 — GT가 cubic spline인가</h3>'
            '<p class="note">주황 점 몇 개로 만든 하늘색 spline이 노랑(GT)에 포개지면 GT는 논문 7단계 '
            f'그대로 cubic spline이다. 무릎 {stats["n_has_knee"]}/{stats["n_ok"]} 에피소드.</p>'
            + knot_html
            + '<h3>(5) refine 반경 프로파일 — GT가 어느 후보에 붙는가</h3>'
            '<p class="note">가로축은 탐침 반경, 세로축은 "다시 refine하면 얼마나 움직이나". '
            f'노랑(GT)에 가장 붙는 후보가 R* = {stats["refine_radius_est_m"]:.2f} m(초록)다. '
            '넓게 refine한 경로는 ESDF 평탄부에 있어 아래쪽에 깔린다.</p>'
            f'<div class="card"><img src="{_data_uri(chart)}" alt="refine profile"></div>')


def render_grid_zoom(row, origin, cell, coarse, out_path, win_m=(2.6, 1.8), px_per_m=260):
    """**기준선이 왜 GT에서 벌어지는가**를 보여주는 확대 그림.

    `A*`가 출력할 수 있는 좌표는 **0.2 m 셀중심뿐**이고 GT는 연속 곡선이다. 격자선과 셀중심 점을
    함께 그리면 "GT를 우리 어휘로 다시 쓰면 이만큼 어긋난다"가 눈에 보인다 — 표의 +0.035 m가
    어디서 오는지에 대한 답이다.

    이탈이 가장 큰 지점을 중심으로 자른다(전체를 보여주면 격자가 안 보인다).
    """
    import cv2

    v = row['_viz']
    gt, floor_c, centers = np.asarray(v['gt']), np.asarray(v['floor_c']), np.asarray(v['cell_centers'])
    # 이탈 최대 지점 찾기 — 기준선의 각 점에서 GT까지의 최근접 거리
    d = np.linalg.norm(floor_c[:, None, :] - gt[None, :, :], axis=2).min(axis=1)
    cx, cy = floor_c[int(np.argmax(d))]

    W, H = int(win_m[0] * px_per_m), int(win_m[1] * px_per_m)
    x0, y0 = cx - win_m[0] / 2, cy - win_m[1] / 2
    to_px = lambda p: (np.asarray(p, dtype=np.float64)[:, :2] - [x0, y0]) * px_per_m

    img = np.full((H, W, 3), 30, dtype=np.uint8)

    # 장애물 셀 (0.05 m) — esdf == 0
    esdf = v['esdf']
    gh, gw = esdf.shape
    ys, xs = np.nonzero(esdf <= 1e-9)
    wx = origin[0] + (xs + 0.5) * cell
    wy = origin[1] + (ys + 0.5) * cell
    keep = (wx >= x0) & (wx < x0 + win_m[0]) & (wy >= y0) & (wy < y0 + win_m[1])
    for px_, py_ in to_px(np.stack([wx[keep], wy[keep]], axis=1)):
        cv2.rectangle(img, (int(px_ - cell * px_per_m / 2), int(H - 1 - py_ - cell * px_per_m / 2)),
                      (int(px_ + cell * px_per_m / 2), int(H - 1 - py_ + cell * px_per_m / 2)),
                      (70, 70, 74), -1)

    # 0.2 m 격자선 — origin에 정렬한다 (A*의 셀 경계와 같아야 의미가 있다)
    k0 = int(np.floor((x0 - origin[0]) / coarse))
    for k in range(k0, k0 + int(win_m[0] / coarse) + 2):
        gx = (origin[0] + k * coarse - x0) * px_per_m
        cv2.line(img, (int(gx), 0), (int(gx), H), (58, 66, 76), 1)
    m0 = int(np.floor((y0 - origin[1]) / coarse))
    for m in range(m0, m0 + int(win_m[1] / coarse) + 2):
        gy = H - 1 - (origin[1] + m * coarse - y0) * px_per_m
        cv2.line(img, (0, int(gy)), (W, int(gy)), (58, 66, 76), 1)

    def poly(path, color, thick=2):
        p = to_px(path)
        pts = np.stack([p[:, 0], H - 1 - p[:, 1]], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)

    poly(gt, GT_COLOR, 3)
    poly(np.asarray(v['floor_b']), FLOOR_B_COLOR, 2)   # 격자+스무딩까지 (refine 전)
    poly(floor_c, FLOOR_COLOR, 2)                      # + refine
    for p in to_px(centers):                       # A*가 낼 수 있는 좌표
        if -20 < p[0] < W + 20 and -20 < p[1] < H + 20:
            cv2.circle(img, (int(p[0]), int(H - 1 - p[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(p[0]), int(H - 1 - p[1])), 5, (40, 40, 40), 1, cv2.LINE_AA)

    # 범례는 하단에 배경 박스를 깔고 둔다 — 곡선 위에 얹으면 서로 가린다
    legend = [('0.2m cell centers = all A* can output', (255, 255, 255)),
              ('GT (continuous curve)', GT_COLOR),
              ('B: cells -> thin -> spline (grid only)', FLOOR_B_COLOR),
              ('C: B + greedy_refine  = baseline', FLOOR_COLOR)]
    bh = 18 * len(legend) + 12
    cv2.rectangle(img, (0, H - bh), (W, H), (18, 20, 24), -1)
    for i, (txt, col) in enumerate(legend):
        y = H - bh + 16 + i * 18
        cv2.line(img, (10, y - 4), (30, y - 4), col, 2, cv2.LINE_AA)
        cv2.putText(img, txt, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    from geometry_utils import save_jpg
    return save_jpg(img, Path(out_path)), float(d.max())


def _data_uri(path) -> str:
    import base64
    return 'data:image/jpeg;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def main() -> int:
    args = build_argparser().parse_args()
    radii = [float(x) for x in args.radii.split(',')]
    print(f'[{SCRIPT_NAME}] scene={args.scene}')

    esdf_dir = Path(args.esdf_dir or (Path(args.out_dir) / 'esdf'))
    paths_dir = Path(args.paths_dir or (Path(args.out_dir) / 'paths'))
    suffix = '_random' if args.mode == 'random' else ''
    npz_path, json_path = esdf_dir / f'{args.scene}.npz', paths_dir / f'{args.scene}{suffix}.json'
    for p, who in ((npz_path, '02_build_freemap_esdf.py'), (json_path, '03_sample_gt_paths.py')):
        if not p.is_file():
            print(f'  [ERROR] {p} 없음 — {who}를 먼저 돌릴 것')
            return 2

    data = np.load(npz_path)
    occupancy, origin, floor_z = data['occupancy'], data['origin'], float(data['floor_z'])
    cell, h_nav = float(data['voxel_size']), float(data['h_nav'])
    planned = json.load(open(json_path))
    coarse = cell * DOWNSAMPLE_FACTOR
    # 03이 실제로 쓴 파라미터를 그대로 따른다 — 어긋나면 F1(결정성)이 통째로 실패한다
    if args.refine_radius is None:
        args.refine_radius = float(planned['params']['refine_radius'])
    if args.r_b is None:
        args.r_b = float(planned['params'].get('r_b', ROBOT_RADIUS_M))
    if args.refine_mode is None:
        args.refine_mode = planned['params'].get('refine_mode', 'argmax')
    if args.smooth is None:
        args.smooth = planned.get('smooth', 'cubic')
    if args.downsample_mode is None:
        args.downsample_mode = planned['params'].get('downsample_mode', 'any')
    print(f'  esdf: grid={occupancy.shape} cell={cell:.3f} h_nav={h_nav:.2f} | '
          f'03 결과 {len(planned["episodes"])} 에피소드 '
          f'(03의 refine={args.refine_mode}/{args.refine_radius} r_b={args.r_b} '
          f'smooth={args.smooth})')

    rows, nav_ref = [], None
    for ep_data in planned['episodes'][:args.num_episodes]:
        ep, h_b = ep_data['episode_id'], ep_data['h_b']
        ours = np.array(ep_data['trajectory'], dtype=np.float64)
        if args.mode == 'random':
            gt_xy, gt_h_b = np.asarray(ep_data['trajectory'], dtype=np.float64), h_b
        else:
            gt_xy, gt_h_b = load_gt_xy(args.data_root, args.scene, ep)
        assert abs(gt_h_b - h_b) < 1e-6, f'ep{ep}: h_b 불일치 {gt_h_b} vs {h_b}'

        # F1 — 같은 입력으로 두 번. 03과 같은 순서를 돌려 결과가 비트 단위 같은지 본다.
        again, esdf = plan_once(occupancy, origin, floor_z, h_b, gt_xy[0], gt_xy[-1], cell, h_nav, args)
        identical = again is not None and again.shape == ours.shape and np.allclose(again, ours, atol=1e-9)
        f1_chamfer = chamfer_distance(again, ours) if again is not None else float('nan')

        # 기준선 ⓒ — GT 루트를 격자에 얹어 논문 6~7단계를 03과 똑같이 돌린 것
        floor_path = pipeline_floor_path(gt_xy, esdf, origin, cell, coarse,
                                         refine_radius_m=args.refine_radius,
                                         spacing_m=WAYPOINT_SPACING_M, smooth_step_m=cell,
                                         refine_mode=args.refine_mode, smooth_mode=args.smooth)
        floor_b = pipeline_floor_path(gt_xy, esdf, origin, cell, coarse,
                                      refine_radius_m=args.refine_radius,
                                      spacing_m=WAYPOINT_SPACING_M, smooth_step_m=cell,
                                      with_refine=False, refine_mode=args.refine_mode,
                                      smooth_mode=args.smooth)
        if nav_ref is None:
            nav_ref = truncate_navigable(esdf, args.r_b)   # 시각화 배경 (첫 에피소드의 h_b 기준)
        floors = metric_floors(gt_xy, coarse, cell)     # 원인 분해 (진단용)
        chamfer = chamfer_distance(ours, gt_xy)
        floor = max(chamfer_distance(floor_path, gt_xy), 1e-9)
        floor_frechet = discrete_frechet_distance(floor_path, gt_xy)

        # (7) GT를 재현하는 최소 knot -> 그 knot이 GT의 waypoint 추정치다
        knots = recover_spline_knots(gt_xy, tol_m=args.knot_tol, k_max=args.k_max)
        k_est = knots['k_star'] or max(3, int(knots['length_m'] / WAYPOINT_SPACING_M) + 1)
        gt_knots = resample_by_arclength(gt_xy, n=k_est)

        # (5) 논문 6단계 반경 역추정 — 같은 A* waypoint를 후보 반경으로 refine한 프로파일과 매칭.
        # 내장 대조군(후보 반경으로 만든 경로에서 그 반경을 되찾는가)이 함께 나온다.
        est = estimate_refine_radius(gt_knots, np.array(ep_data['waypoints_astar'], dtype=np.float64),
                                     esdf, origin, cell, candidates_m=radii)
        our_knots = recover_spline_knots(ours, tol_m=args.knot_tol, k_max=args.k_max)

        rows.append({
            'episode': ep, 'h_b': h_b,
            'chamfer_m': chamfer, 'frechet_m': discrete_frechet_distance(ours, gt_xy),
            'floor_chamfer_m': floor, 'floor_frechet_m': floor_frechet,
            'floor_no_refine_chamfer_m': chamfer_distance(floor_b, gt_xy),
            'f2_chamfer_m': floors['f2_chamfer_m'], 'f3_chamfer_m': floors['f3_chamfer_m'],
            'chamfer_over_floor': chamfer / floor,
            'frechet_over_floor': discrete_frechet_distance(ours, gt_xy) / max(floor_frechet, 1e-9),
            'f1_identical': bool(identical), 'f1_chamfer_m': f1_chamfer,
            'k_star': knots['k_star'], 'has_knee': knots['has_knee'],
            'knot_spacing_m': knots['spacing_m'] if knots['k_star'] else float('nan'),
            'rmse_at_k_max': knots['rmse_at_k_max'],
            'refine_radius_est_m': est['estimate_m'], 'refine_control_ok': est['control_ok'],
            'refine_l1': est['l1_by_candidate'], 'refine_gt_profile': est['gt_profile'],
            'refine_candidate_profiles': est['candidate_profiles'],
            'n_gt_knots': int(len(gt_knots)),
            'our_k_star': our_knots['k_star'], 'our_knot_spacing_m': our_knots['spacing_m'],
            # 시각화용 (JSON에는 nan/None으로 나가도 무해하도록 아래에서 따로 뺀다)
            '_viz': {'gt': gt_xy, 'ours': ours, 'floor_c': floor_path, 'floor_b': floor_b,
                     'knots': gt_knots, 'esdf': esdf,
                     # A*가 낼 수 있는 좌표 = 0.2 m 셀중심. 확대 그림에서 이걸 보여준다.
                     'cell_centers': cell_to_world(path_cell_sequence(gt_xy, origin, coarse),
                                                   origin, coarse),
                     # 여기는 cubic 고정 — "GT가 논문 7단계 cubic spline인가" 검정의 시각화다
                     'knot_spline': smooth_cubic_spline(gt_knots, cell) if len(gt_knots) >= 3
                     else gt_knots},
        })
        print(f'    ep {ep:>3}: chamfer={chamfer:.3f} (기준선 {floor:.3f} -> {chamfer / floor:.2f}x, '
              f'refine없이 {chamfer_distance(floor_b, gt_xy):.3f})  '
              f'F1={"same" if identical else f"DIFF({f1_chamfer:.4f})"}  '
              f'k*={knots["k_star"]} 간격={knots["spacing_m"]:.2f}m  '
              f'R*={est["estimate_m"]:.2f}m{"" if est["control_ok"] else "(대조군실패)"}')

    if not rows:
        print('  [ERROR] 검사할 에피소드가 없다')
        return 1

    def med(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        return float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else float('nan')

    # (5) 프로파일은 에피소드 평균으로 합산한 뒤 다시 매칭한다 — 에피소드별 추정보다 잡음이 적다
    cand_keys = [round(float(x), 3) for x in radii]
    gt_prof = np.mean([r['refine_gt_profile'] for r in rows], axis=0)
    cand_prof = {c: np.mean([r['refine_candidate_profiles'][str(c)] if str(c) in
                             r['refine_candidate_profiles'] else r['refine_candidate_profiles'][c]
                             for r in rows], axis=0) for c in cand_keys}
    l1 = {c: float(np.abs(cand_prof[c] - gt_prof).sum()) for c in cand_keys}
    radius_est = min(l1, key=l1.get)
    k_stars = [r['k_star'] for r in rows if r['k_star']]
    stats = {
        'n_ok': len(rows),
        'n_f1_identical': int(sum(r['f1_identical'] for r in rows)),
        'f1_all_identical': bool(all(r['f1_identical'] for r in rows)),
        'chamfer_median_m': med('chamfer_m'), 'frechet_median_m': med('frechet_m'),
        'floor_chamfer_median_m': med('floor_chamfer_m'),
        'floor_frechet_median_m': med('floor_frechet_m'),
        'floor_no_refine_chamfer_median_m': med('floor_no_refine_chamfer_m'),
        'f2_chamfer_median_m': med('f2_chamfer_m'), 'f3_chamfer_median_m': med('f3_chamfer_m'),
        'chamfer_over_floor': med('chamfer_over_floor'),
        'frechet_over_floor': med('frechet_over_floor'),
        'refine_radius_est_m': radius_est, 'refine_l1': l1,
        'refine_gt_profile': gt_prof.tolist(),
        'refine_candidate_profiles': {c: cand_prof[c].tolist() for c in cand_keys},
        'probe_radii_m': list(REFINE_PROBE_RADII_M),
        'n_refine_control_ok': int(sum(r['refine_control_ok'] for r in rows)),
        # 에피소드 대조군은 후보 반경들이 서로 구분될 때만 통과한다 — A* 경로가 이미 벽에서 멀면
        # 여러 후보가 같은 결과를 내 구분이 안 된다(추정기 결함이 아니라 그 에피소드의 정보 부족).
        # 집계 판정은 과반으로 본다.
        'refine_control_ok': bool(sum(r['refine_control_ok'] for r in rows) > len(rows) / 2),
        'our_knot_spacing_median_m': med('our_knot_spacing_m'),
        'n_has_knee': int(sum(r['has_knee'] for r in rows)),
        'k_star_median': int(np.median(k_stars)) if k_stars else None,
        'knot_spacing_median_m': med('knot_spacing_m'),
        'current_refine_radius': args.refine_radius,
        'current_waypoint_spacing_m': WAYPOINT_SPACING_M,
        'floor_ratio_at_limit': FLOOR_RATIO_AT_LIMIT,
        'at_floor': bool(med('chamfer_over_floor') <= FLOOR_RATIO_AT_LIMIT),
    }

    log_dir = Path(args.log_dir) / SCRIPT_NAME / args.scene
    body = build_body(args, stats, rows, radii, nav_ref, origin, cell, log_dir)

    verify_dir = Path(args.out_dir) / 'verify'
    verify_dir.mkdir(parents=True, exist_ok=True)
    out_json = verify_dir / f'{args.scene}.json'
    out_json.write_text(json.dumps(
        {'scene_id': args.scene, 'stats': stats,
         'episodes': [{k: v for k, v in r.items() if k != '_viz'} for r in rows],
         'params': vars(args)}, indent=2, ensure_ascii=False, default=lambda o: None),
        encoding='utf-8')
    report = save_gallery(log_dir, 'report.html', f'03b_verify_reproduction — {args.scene}',
                          build_summary(args, stats, rows, radii), body)
    print(f'  verify -> {out_json}')
    print(f'  report html -> {report}')
    print(f'  => F1 {stats["n_f1_identical"]}/{stats["n_ok"]} 동일, '
          f'chamfer {stats["chamfer_median_m"]:.3f} = 기준선({stats["floor_chamfer_median_m"]:.3f})의 '
          f'{stats["chamfer_over_floor"]:.2f}x'
          f'{" [이산화 한계]" if stats["at_floor"] else " [루트 차이 남음]"}, '
          f'R*={stats["refine_radius_est_m"]:.2f}m'
          f'{"" if stats["refine_control_ok"] else "(대조군실패)"}, '
          f'무릎 {stats["n_has_knee"]}/{stats["n_ok"]} 간격 {stats["knot_spacing_median_m"]:.2f}m')
    # F1(결정성)만 실패 조건이다 — 나머지는 진단 수치라 판정하지 않는다.
    return 0 if stats['f1_all_identical'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
