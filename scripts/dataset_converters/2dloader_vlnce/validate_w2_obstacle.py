"""W2 — 장애물 합성의 RGB/depth/BEV 정합 검증.

실제 vln_ce 프레임에 3D 박스를 놓고, **RGB·depth·BEV가 같은 물체를 같은 자리에서 보고 있는지**를
네 지표로 확인한다. BEV는 따로 그리지 않고 **합성된 depth에서 다시 계산**하므로, 이 검증은
"합성 depth가 기하와 맞는가"와 "두 rig 투영이 맞는가"를 보는 것이다.

| 게이트 | 무엇 | 통과 |
|---|---|---|
| V 가시성 | 룩다운 이미지에서 박스가 실제로 보이는가 | 마스크 ≥ 0.2% 픽셀 |
| P footprint | 새로 점유된 BEV 셀이 박스 밑면 안인가 | ≥ 95% |
| S 표면 | 합성 depth를 역투영한 점이 박스 표면 위인가 | median ≤ 1 cm |
| X 교차 rig | FPV 실루엣이 8꼭짓점 convex hull 안인가 | precision ≥ 0.99 (recall은 가림 때문에 보고만) |

`--kind`를 늘려도 이 스크립트는 그대로 재사용된다(`footprint_xy`/`corners_xyz`만 kind를 안다).

실행: /usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w2_obstacle.py --scene 17DRP5sb8fy --episode 0 --preset 125cm_0_30 --n_frames 4 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_CONV = _HERE.parent
for _p in (str(_CONV / 'gs_vlnpe'), str(_CONV / '3dloader_vlnce'), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from geometry_utils import colorize_depth, save_jpg  # noqa: E402
from viz_utils import save_gallery  # noqa: E402
from pixel_goal_utils import intrinsics_for_rig, rig_height_m  # noqa: E402

from episode_io import IMG_H, IMG_W, load_episode, to_224_depth  # noqa: E402
from local_map import (  # noqa: E402
    build_local_map, limit_torch_threads, robot_xy_to_bev_ij, world_to_robot,
)
from obstacle_synth import (  # noqa: E402
    Cam, ObstacleCfg, _ray_grid_robot, composite, corners_xyz, footprint_xy, project, sample_obstacle,
)
from report_common import (  # noqa: E402
    bev_rows, gate_note, legend_table, param_glossary, path_rows, pipeline_note,
)
from augment2d import Augmenter2D, Aug2DCfg, Embodiment  # noqa: E402
from viz2d import (  # noqa: E402
    GT_COLOR, NEW_COLOR, OBST_COLOR, START_COLOR, bev_rgb, draw_polyline, draw_pts, hstrip, label,
)

TOL_VISIBLE = 0.002
TOL_PRECISION = 0.95
TOL_SURFACE_M = 0.01
TOL_XRIG = 0.99


def pill(ok: bool) -> str:
    c = '#2e7d32' if ok else '#c62828'
    return (f'<span style="background:{c};color:#fff;border-radius:10px;padding:2px 10px;'
            f'font-weight:600">{"PASS" if ok else "FAIL"}</span>')


def make_cam(rig: str, pitch: float) -> Cam:
    fx, fy, cx, cy = intrinsics_for_rig(rig)
    return Cam(pitch, rig_height_m(rig), fx, fy, cx, cy, IMG_W, IMG_H)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='17DRP5sb8fy')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--preset', default='125cm_0_30')
    ap.add_argument('--n_frames', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--h_nav', type=float, default=0.15)
    ap.add_argument('--h_b', type=float, default=1.25)
    ap.add_argument('--r_b', type=float, default=0.10)
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--out_dir', default='logs/embodiment_augment2d/w2')
    args = ap.parse_args()
    limit_torch_threads()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ep = load_episode(args.data_root, args.scene, args.episode, args.preset, n_frames=args.n_frames)
    p1 = float(ep.rig_fpv.split('_')[1].replace('deg', ''))
    p2 = float(ep.rig_ld.split('_')[1].replace('deg', ''))
    cam_ld, cam_fpv = make_cam(ep.rig_ld, p2), make_cam(ep.rig_fpv, p1)
    cam_h = rig_height_m(ep.rig_ld)
    rng = np.random.default_rng(args.seed)
    aug = Augmenter2D(ep.rig_fpv, ep.rig_ld, Aug2DCfg(obstacle=True))

    rows, body = [], []
    for f in ep.frames:
        j = f.idx + f.rel_goal + 1
        gt = world_to_robot(ep.poses_ld[f.idx:j + 1, :3, 3], f.pose_ld, p2, cam_h)[:, :2]
        obs = sample_obstacle(gt, rng, cam_ld, ObstacleCfg())
        if obs is None:
            print(f'[W2] f{f.idx}: 장애물 배치 실패(시야/거리 조건) — 건너뜀'); continue

        rgb_ld2, depth2, mask_ld = composite(f.rgb_ld, f.depth_m, obs, cam_ld)
        rgb_fpv2, _, mask_fpv = composite(f.rgb_fpv, f.depth_fpv_m, obs, cam_fpv)

        d0, d1 = to_224_depth(f.depth_m), to_224_depth(depth2)
        m0 = build_local_map(d0, ep.rig_ld, p2, args.r_b, args.h_nav, args.h_b)
        m1 = build_local_map(d1, ep.rig_ld, p2, args.r_b, args.h_nav, args.h_b)
        S = m1['bev_size']

        # P — 새로 점유된 셀이 밑면 안인가
        new = m1['occupied'] & ~m0['occupied']
        fp = np.zeros((S, S), np.uint8)
        poly = robot_xy_to_bev_ij(footprint_xy(obs))[:, ::-1].astype(np.int32)
        cv2.fillPoly(fp, [poly], 1)
        fp_d = cv2.dilate(fp, np.ones((3, 3), np.uint8)) > 0
        prec = float((new & fp_d).sum()) / max(int(new.sum()), 1)

        # S — 합성 depth 역투영이 박스 표면 위인가
        o_r, d_r = _ray_grid_robot(cam_ld)
        pts = o_r + d_r[mask_ld] * depth2[mask_ld][:, None]
        loc = (pts - obs.center3) @ obs.rot()
        surf = float(np.median(np.abs(np.abs(loc) - obs.size / 2.0).min(axis=1))) if mask_ld.any() else np.nan

        # X — FPV 실루엣 vs 8꼭짓점 convex hull
        uv = project(corners_xyz(obs), cam_fpv)[:, :2]
        hull_m = np.zeros((IMG_H, IMG_W), np.uint8)
        if np.isfinite(uv).all():
            cv2.fillConvexPoly(hull_m, cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2)).astype(np.int32), 1)
        hb = hull_m > 0
        # 바닥에 붙은 낮은 박스가 가까이 있으면 **수평 FPV(pitch_1=0)의 화각 아래**로 내려가 아예
        # 안 보인다(실측 s8pcmisQ38h f21). 마스크가 비면 precision은 정의되지 않는다 — 0으로 세면
        # 게이트가 거짓 실패한다. S2는 룩다운 이미지도 함께 받으므로 이 경우도 학습에 문제는 없다.
        n_fpv = int(mask_fpv.sum())
        xprec = (float((mask_fpv & hb).sum()) / n_fpv) if n_fpv else float('nan')
        xrec = (float((mask_fpv & hb).sum()) / max(int(hb.sum()), 1)) if n_fpv else float('nan')

        # 장애물만 넣고 경로를 그대로 두면 충돌이다 — 합성 뒤 **재계획**을 여기서 미리 돌려 둔다
        rr = aug.augment(f, ep.poses_ld, Embodiment(r_b=args.r_b, h_nav=args.h_nav, h_b=args.h_b),
                         rng=np.random.default_rng(args.seed), with_obstacle=True)
        newt = rr.get('trajectory')

        vis = float(mask_ld.mean())
        ok = (vis >= TOL_VISIBLE and prec >= TOL_PRECISION and surf <= TOL_SURFACE_M
              and (np.isnan(xprec) or xprec >= TOL_XRIG))
        rows.append(dict(idx=f.idx, vis=vis, prec=prec, surf=surf, xp=xprec, xr=xrec,
                         new=int(new.sum()), size=obs.size.round(2).tolist(),
                         c=obs.center_xy.round(2).tolist(), ok=ok, replan=rr['status']))

        # 그림 (before | after) × (FPV, lookdown, depth, BEV)
        gt_ij = robot_xy_to_bev_ij(gt)
        b0, b1 = bev_rgb(m0['bev']), bev_rgb(m1['bev'])
        for im in (b0, b1):
            draw_pts(im, gt_ij, GT_COLOR, 1)
            draw_pts(im, np.array([[S // 2, S // 2]]), START_COLOR, 3)
        cv2.polylines(b1, [poly.reshape(-1, 1, 2)], True, tuple(int(c) for c in OBST_COLOR), 1)
        if newt is not None:
            draw_polyline(b1, robot_xy_to_bev_ij(newt), NEW_COLOR, 1)
        top = hstrip([label(f.rgb_fpv.copy(), 'FPV before'), label(f.rgb_ld.copy(), 'lookdown before'),
                      label(colorize_depth(d0, d0 > 0.05), 'depth before'), label(b0, 'BEV before')],
                     height=360)
        bot = hstrip([label(rgb_fpv2, 'FPV after'), label(rgb_ld2, 'lookdown after'),
                      label(colorize_depth(d1, d1 > 0.05), 'depth after'), label(b1, 'BEV after')],
                     height=360)
        save_jpg(np.vstack([top, np.full((6, top.shape[1], 3), 255, np.uint8), bot]),
                 out / f'w2_f{f.idx:04d}.jpg')
        body.append(
            f'<h3>frame {f.idx} {pill(ok)}</h3>'
            f'<p>박스 {obs.size.round(2).tolist()} m @ robot XY {obs.center_xy.round(2).tolist()} '
            f'yaw {np.degrees(obs.yaw):.0f}° · 가시 {100*vis:.2f}% 픽셀 · 신규 점유 {int(new.sum())}셀 · '
            f'footprint 안 {100*prec:.1f}% · 표면오차 {1000*surf:.2f} mm · '
            f'교차 rig precision {100*xprec:.1f}% / recall {100*xrec:.1f}%</p>'
            f'<img src="w2_f{f.idx:04d}.jpg" style="width:100%">')
        print(f'[W2] f{f.idx}: vis={100*vis:.2f}% prec={100*prec:.1f}% surf={1000*surf:.2f}mm '
              f'xrig p={100*xprec:.1f}%/r={100*xrec:.1f}% new={int(new.sum())}cells')

    assert rows, '유효한 프레임이 없다'
    v_ok = min(r['vis'] for r in rows) >= TOL_VISIBLE
    p_ok = min(r['prec'] for r in rows) >= TOL_PRECISION
    s_ok = max(r['surf'] for r in rows) <= TOL_SURFACE_M
    xps = [r['xp'] for r in rows if np.isfinite(r['xp'])]
    n_fpv_blind = sum(1 for r in rows if not np.isfinite(r['xp']))
    x_ok = (min(xps) >= TOL_XRIG) if xps else False
    legend = legend_table(path_rows(with_gt=True, with_new=True, with_obst=True) + bev_rows())
    summary = (
        f'<p><b>{args.scene}</b> ep{args.episode} · preset {args.preset} · '
        f'장애물 kind=box (flat-shaded 단색 직육면체)</p>'
        + gate_note(
            what='합성한 3D 박스 하나가 <b>FPV RGB · 룩다운 RGB · depth · BEV 네 곳에 같은 자리로</b> '
                 '들어가는지를 네 지표로 잰다.',
            why='이 네 가지가 어긋나면 뒤따르는 모든 것(재계획·라벨)이 틀린 관측 위에서 만들어진다. '
                'BEV를 따로 그리지 않고 <b>합성된 depth에서 다시 계산</b>하기 때문에, 이 검증은 '
                '"합성 depth가 기하와 맞는가"와 "두 rig 투영이 맞는가"로 환원된다.',
            criterion=f'가시성 ≥ {100*TOL_VISIBLE:.1f}% · footprint 안 ≥ {100*TOL_PRECISION:.0f}% · '
                      f'표면오차 ≤ {1000*TOL_SURFACE_M:.0f} mm · 교차 rig precision ≥ {100*TOL_XRIG:.0f}%')
        + pipeline_note()
        + param_glossary(preset=args.preset, r_b=args.r_b, h_nav=args.h_nav, h_b=args.h_b,
                         seed=args.seed, obstacle='O (box)')
        + '<h4>그림 8칸의 뜻</h4><p>위 줄 = <b>합성 전</b>, 아래 줄 = <b>합성 후</b>. '
        '왼쪽부터 FPV(pitch_1, System 2가 보는 RGB) / lookdown(pitch_2, depth·goal 기준 RGB) / '
        'depth(파랑=가까움) / BEV. 아래 줄 BEV에는 <b>합성 뒤 다시 계획한 경로</b>(초록)를 함께 그렸다 — '
        '장애물만 넣고 노란 원본 GT를 그대로 두면 그것이 곧 충돌이기 때문이다. '
        '<b>네 칸이 모두 같은 3D 박스 하나에서 나온다</b> — depth는 ray-box '
        '교차로 z-buffer 합성, RGB는 같은 마스크에 단색 음영, BEV는 따로 그리지 않고 '
        '<b>합성된 depth에서 다시 계산</b>한다. 그래서 정합이 구조적으로 보장되고, 아래 네 게이트로 '
        '그것을 수치로 확인한다.</p>'
        + legend
        + f'<h4>게이트</h4>'
        f'<table border=1 cellpadding=6><tr><th>게이트</th><th>측정</th><th>기준</th><th></th></tr>'
        f'<tr><td>V 가시성</td><td>min {100*min(r["vis"] for r in rows):.2f}% 픽셀</td>'
        f'<td>≥ {100*TOL_VISIBLE:.1f}%</td><td>{pill(v_ok)}</td></tr>'
        f'<tr><td>P footprint 정합</td><td>min {100*min(r["prec"] for r in rows):.1f}%</td>'
        f'<td>≥ {100*TOL_PRECISION:.0f}%</td><td>{pill(p_ok)}</td></tr>'
        f'<tr><td>S 표면 일치</td><td>max {1000*max(r["surf"] for r in rows):.2f} mm</td>'
        f'<td>≤ {1000*TOL_SURFACE_M:.0f} mm</td><td>{pill(s_ok)}</td></tr>'
        f'<tr><td>X 교차 rig(FPV)</td><td>precision min {100*min(xps):.1f}% '
        f'(FPV 화각 밖 {n_fpv_blind}/{len(rows)} 프레임은 제외)</td>'
        f'<td>≥ {100*TOL_XRIG:.0f}%</td><td>{pill(x_ok)}</td></tr></table>'
        f'<table border=1 cellpadding=6><tr><th>frame</th><th>박스 크기(m)</th><th>중심 XY(m)</th>'
        f'<th>가시(%)</th><th>신규 셀</th><th>footprint 안</th><th>표면(mm)</th>'
        f'<th>교차 rig p/r</th><th>합성 후 재계획</th></tr>'
        + ''.join(f'<tr><td>{r["idx"]}</td><td>{r["size"]}</td><td>{r["c"]}</td>'
                  f'<td>{100*r["vis"]:.2f}</td><td>{r["new"]}</td><td>{100*r["prec"]:.1f}%</td>'
                  f'<td>{1000*r["surf"]:.2f}</td><td>{100*r["xp"]:.0f}/{100*r["xr"]:.0f}%</td>'
                  f'<td>{r["replan"]}</td></tr>' for r in rows) + '</table>'
        f'<p><b>읽는 법</b> — P는 "BEV가 박스를 엉뚱한 자리에 그리지 않는다", S는 "합성 depth가 박스 '
        f'기하와 정확히 같다", X는 "FPV(pitch_1)와 룩다운(pitch_2)이 같은 물체를 본다"를 뜻한다. '
        f'X의 recall이 100%가 아닌 것은 정상 — 장면 물체가 박스를 <b>가리면</b> 그만큼 실루엣이 '
        f'줄어들기 때문이다(가림은 각 rig의 원본 depth로 판정한다).</p>'
        f'<p><b>확장</b> — 텍스처·복잡한 형태·dataset 유사 object는 <code>render_obstacle</code>의 '
        f'<code>kind</code> 분기 한 곳만 늘리면 되고, BEV는 합성된 depth에서 다시 계산되므로 '
        f'이 검증 스크립트와 하류 파이프라인은 그대로 재사용된다.</p>')
    (out / 'summary.html').write_text(summary, encoding='utf-8')
    (out / 'body.html').write_text(''.join(body), encoding='utf-8')
    save_gallery(out, 'report.html', 'W2 — 장애물 합성 RGB/depth/BEV 정합', summary, ''.join(body))
    print(f'[W2] V={v_ok} P={p_ok} S={s_ok} X={x_ok} -> {out}/report.html')
    return 0 if (v_ok and p_ok and s_ok and x_ok) else 1


if __name__ == '__main__':
    raise SystemExit(main())
