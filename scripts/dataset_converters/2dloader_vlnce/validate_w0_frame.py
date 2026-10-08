"""W0 — 좌표 규약·항등 게이트. **여기가 틀리면 이후 전부 무의미하다.**

depth 한 장에서 만든 robot-centric 격자가 (a) 중력 정렬돼 있고, (b) 학습이 쓰는 GT 궤적 프레임과
같은 좌표계이며, (c) 저장된 pixel goal을 재현하는지 확인한다.

| 항목 | 무엇 | 통과 |
|---|---|---|
| A 중력 정렬 | world->robot 회전이 world z축 둘레 순수 yaw인가 | < 1e-3 (pose 자체 양자화 오차 ~8e-5) |
| B 학습 프레임 일치 | `world_to_robot`로 만든 GT xy == `get_trajectory_relative_to_frame`(학습 코드)의 xy | max < 1e-3 m |
| C pixel goal 재현 | `goal_world_from_poses`+`project_to_pixel` == parquet `goal.<rig>` | max < 3 px |
| D GT가 free 위 | 원본 GT 경로가 BEV 점유 셀을 밟지 않는가 / clearance | 점유셀 밟음 0, clearance 보고 |

실행: /usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w0_frame.py --scene 17DRP5sb8fy --episode 0 --preset 125cm_0_30 --n_frames 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CONV = _HERE.parent
for _p in (str(_CONV / 'gs_vlnpe'), str(_CONV / '3dloader_vlnce'), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esdf_utils import sample_esdf_at  # noqa: E402
from geometry_utils import colorize_depth, save_jpg  # noqa: E402
from viz_utils import save_gallery  # noqa: E402
from pixel_goal_utils import goal_world_from_poses, project_to_pixel, rig_height_m  # noqa: E402

from episode_io import load_episode, to_224_depth  # noqa: E402
from local_map import (  # noqa: E402
    limit_torch_threads,
    build_local_map, gravity_alignment_error, robot_to_plan, robot_xy_to_bev_ij, world_to_robot,
)
from report_common import (  # noqa: E402
    bev_rows, gate_note, goal_rows, legend_table, param_glossary, path_rows, pipeline_note,
)
from viz2d import (  # noqa: E402
    ADJ_COLOR, GOAL_COLOR, GT_COLOR, START_COLOR, bev_rgb, draw_marker, draw_pts, hstrip, label,
)

TOL_GRAVITY = 1e-3
TOL_FRAME_M = 1e-3
TOL_GOAL_PX = 3.0


def pill(ok: bool) -> str:
    c = '#2e7d32' if ok else '#c62828'
    return (f'<span style="background:{c};color:#fff;border-radius:10px;padding:2px 10px;'
            f'font-weight:600">{"PASS" if ok else "FAIL"}</span>')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='17DRP5sb8fy')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--preset', default='125cm_0_30', help='<H>cm_<pitch1>_<pitch2>')
    ap.add_argument('--n_frames', type=int, default=6)
    ap.add_argument('--r_b', type=float, default=0.10, help='baseline = habitat 기본 agent radius')
    ap.add_argument('--h_nav', type=float, default=0.15)
    ap.add_argument('--h_b', type=float, default=1.25)
    ap.add_argument('--unknown', default='nontraversable',
                    choices=('nontraversable', 'free', 'block'))
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--out_dir', default='logs/embodiment_augment2d/w0')
    args = ap.parse_args()
    limit_torch_threads()

    from internnav.dataset.internvla_n1_lerobot_dataset import get_trajectory_relative_to_frame

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ep = load_episode(args.data_root, args.scene, args.episode, args.preset, n_frames=args.n_frames)
    pitch_2 = float(ep.rig_ld.split('_')[1].replace('deg', ''))
    cam_h = rig_height_m(ep.rig_ld)

    rows, body = [], []
    for f in ep.frames:
        j = f.idx + f.rel_goal + 1
        fut = ep.poses_ld[f.idx:j + 1]

        # A 중력 정렬
        g_err = gravity_alignment_error(f.pose_ld, pitch_2)

        # B 학습 프레임 일치 — 학습 코드가 쓰는 상대 궤적과 같은 xy가 나와야 한다
        ref = get_trajectory_relative_to_frame(fut, camera_deg=pitch_2)[:, :2]
        mine = world_to_robot(fut[:, :3, 3], f.pose_ld, pitch_2, cam_h)[:, :2]
        frame_err = float(np.abs(ref - mine).max())

        # C pixel goal 재현
        G = goal_world_from_poses(fut, ep.rig_ld)
        u, v, Z = project_to_pixel(G, f.pose_ld, ep.rig_ld)
        goal_err = float(np.max(np.abs(np.array([u, v]) - f.goal_uv)))

        # D GT가 free 위인가
        d224 = to_224_depth(f.depth_m)
        m = build_local_map(d224, ep.rig_ld, pitch_2, args.r_b, args.h_nav, args.h_b,
                            unknown=args.unknown)
        ij = robot_xy_to_bev_ij(mine)
        inb = ((ij[:, 0] >= 0) & (ij[:, 0] < m['bev_size']) & (ij[:, 1] >= 0) & (ij[:, 1] < m['bev_size']))
        hits = int(m['occupied'][ij[inb, 0], ij[inb, 1]].sum())
        cl = sample_esdf_at(m['esdf'], robot_to_plan(mine[inb]), m['origin'], m['cell_m'])
        goal_r = world_to_robot(G[None], f.pose_ld, pitch_2, cam_h)[0]

        # 그림: FPV(pitch_1) | lookdown(pitch_2)+goal | depth | BEV+GT
        fpv = f.rgb_fpv.copy()
        ld = f.rgb_ld.copy()
        draw_marker(ld, f.goal_uv, GOAL_COLOR, 9, filled=False)
        draw_marker(ld, (u, v), ADJ_COLOR, 4, filled=True)
        dep = colorize_depth(d224, d224 > 0.05)
        bev = bev_rgb(m['bev'])
        draw_pts(bev, ij[inb], GT_COLOR, 1)
        draw_pts(bev, robot_xy_to_bev_ij(goal_r[None][:, :2]), GOAL_COLOR, 3)
        draw_pts(bev, np.array([[m['bev_size'] // 2, m['bev_size'] // 2]]), START_COLOR, 3)
        strip = hstrip([label(fpv, f'FPV {ep.rig_fpv}'), label(ld, f'lookdown {ep.rig_ld} + goal'),
                        label(dep, 'depth 224'), label(bev, 'BEV + GT path')])
        save_jpg(strip, out / f'w0_f{f.idx:04d}.jpg')

        ok = (g_err < TOL_GRAVITY and frame_err < TOL_FRAME_M and goal_err < TOL_GOAL_PX and hits == 0)
        rows.append(dict(idx=f.idx, g=g_err, fr=frame_err, gp=goal_err, hits=hits,
                         n=int(inb.sum()), cmin=float(cl.min()), cmed=float(np.median(cl)),
                         Z=Z, ok=ok))
        body.append(f'<h3>frame {f.idx} {pill(ok)}</h3>'
                    f'<p>중력정렬 {g_err:.2e} · 학습프레임 일치 {frame_err:.2e} m · '
                    f'pixel goal {goal_err:.2f} px · GT {int(inb.sum())}점 중 점유셀 {hits}개 · '
                    f'clearance min {cl.min():.2f} / median {np.median(cl):.2f} m</p>'
                    f'<img src="w0_f{f.idx:04d}.jpg" style="width:100%">')
        print(f'[W0] f{f.idx}: gravity={g_err:.2e} frame={frame_err:.2e}m goal={goal_err:.2f}px '
              f'hits={hits}/{int(inb.sum())} clearance min={cl.min():.2f} med={np.median(cl):.2f}')

    a_ok = max(r['g'] for r in rows) < TOL_GRAVITY
    b_ok = max(r['fr'] for r in rows) < TOL_FRAME_M
    c_ok = max(r['gp'] for r in rows) < TOL_GOAL_PX
    d_ok = sum(r['hits'] for r in rows) == 0
    legend = legend_table(
        path_rows(with_gt=True) + goal_rows(with_orig=True, with_adj=True) + bev_rows())
    summary = (
        f'<p><b>{args.scene}</b> ep{args.episode} · preset {args.preset} '
        f'(FPV {ep.rig_fpv} / lookdown {ep.rig_ld})</p>'
        + gate_note(
            what='depth 한 장에서 만든 robot-centric 격자가 (A) 중력 정렬돼 있고, (B) <b>학습이 실제로 '
                 '쓰는 GT 궤적 좌표계와 같으며</b>, (C) 저장된 pixel goal 라벨을 재현하고, '
                 '(D) 원본 GT가 점유칸을 밟지 않는지 확인한다.',
            why='여기가 틀리면 이후 모든 게이트가 <b>잘못된 좌표계 위에서</b> 측정된다. 특히 (B)는 '
                '"여기서 재계획한 경로를 그대로 학습 GT로 쓸 수 있는가"의 근거이고, (C)는 "라벨을 '
                '우리가 다시 만들 수 있는가"의 근거다.',
            criterion=f'중력정렬 &lt; {TOL_GRAVITY} · 학습 프레임 일치 &lt; {TOL_FRAME_M} m · '
                      f'pixel goal &lt; {TOL_GOAL_PX} px · 점유칸 밟음 0')
        + pipeline_note()
        + param_glossary(preset=f'{args.preset} (FPV {ep.rig_fpv} / lookdown {ep.rig_ld})',
                         r_b=args.r_b, h_nav=args.h_nav, h_b=args.h_b, unknown=args.unknown)
        + legend +
        f'<h4>게이트</h4>'
        f'<table border=1 cellpadding=6><tr><th>게이트</th><th>측정(최댓값)</th><th>기준</th><th></th></tr>'
        f'<tr><td>A 중력 정렬</td><td>{max(r["g"] for r in rows):.2e}</td><td>&lt; {TOL_GRAVITY}</td>'
        f'<td>{pill(a_ok)}</td></tr>'
        f'<tr><td>B 학습 GT 프레임 일치</td><td>{max(r["fr"] for r in rows):.2e} m</td>'
        f'<td>&lt; {TOL_FRAME_M} m</td><td>{pill(b_ok)}</td></tr>'
        f'<tr><td>C pixel goal 재현</td><td>{max(r["gp"] for r in rows):.2f} px</td>'
        f'<td>&lt; {TOL_GOAL_PX} px</td><td>{pill(c_ok)}</td></tr>'
        f'<tr><td>D GT가 점유셀 밟음</td><td>{sum(r["hits"] for r in rows)} / '
        f'{sum(r["n"] for r in rows)} 점</td><td>0</td><td>{pill(d_ok)}</td></tr></table>'
        f'<h4>이미지 4열의 뜻</h4><p><b>FPV</b>(pitch_1) = System 2가 보는 정면 RGB · '
        f'<b>lookdown</b>(pitch_2) = depth·pose·pixel goal이 모두 이 카메라 기준. 여기에 저장 goal(빈 원)과 '
        f'우리가 재투영한 goal(채운 원)을 겹쳐 그렸다 · <b>depth 224</b> = 학습에 들어가는 해상도로 줄인 '
        f'metric depth(파랑=가까움, 빨강=멂) · <b>BEV</b> = 그 depth로 만든 robot-centric 격자, 전방이 위쪽</p>'
        f'<h4>프레임별 측정</h4>'
        f'<table border=1 cellpadding=6><tr><th>frame</th><th>중력</th><th>프레임(m)</th>'
        f'<th>goal(px)</th><th>goal Z(m)</th><th>점유셀</th><th>clearance min/med(m)</th></tr>'
        + ''.join(f'<tr><td>{r["idx"]}</td><td>{r["g"]:.1e}</td><td>{r["fr"]:.1e}</td>'
                  f'<td>{r["gp"]:.2f}</td><td>{r["Z"]:.2f}</td><td>{r["hits"]}/{r["n"]}</td>'
                  f'<td>{r["cmin"]:.2f} / {r["cmed"]:.2f}</td></tr>' for r in rows) + '</table>'
        f'<p><b>읽는 법</b> — B가 통과하면 이 모듈의 robot 프레임이 학습의 <code>traj_poses</code>와 '
        f'같은 좌표계라는 뜻이고, 따라서 여기서 재계획한 경로를 그대로 GT로 쓸 수 있다. '
        f'C는 pixel goal 라벨을 우리가 다시 만들 수 있다는 뜻(=<code>e</code>가 바뀔 때 라벨도 갱신 가능). '
        f'D의 clearance가 곧 <code>r_b</code> sweep의 여유 — 이 값보다 큰 <code>r_b</code>는 원본 경로를 '
        f'통행 불가로 만든다.</p>'
        f'<p><b>unknown 정책</b>: <code>free</code>(기본). <code>block</code>으로 두면 시야각 원뿔의 '
        f'측면 경계가 장애물이 되어 GT clearance가 0.00~0.10 m로 붕괴한다(실측). '
        f'대신 <code>free</code>는 미관측 영역을 낙관적으로 본다 — 그 대가는 W1에서 정량화한다.</p>')
    (out / 'summary.html').write_text(summary, encoding='utf-8')
    (out / 'body.html').write_text(''.join(body), encoding='utf-8')
    save_gallery(out, 'report.html', 'W0 — 좌표 규약·항등 게이트', summary, ''.join(body))
    print(f'[W0] A={a_ok} B={b_ok} C={c_ok} D={d_ok} -> {out}/report.html')
    return 0 if (a_ok and b_ok and c_ok and d_ok) else 1


if __name__ == '__main__':
    raise SystemExit(main())
