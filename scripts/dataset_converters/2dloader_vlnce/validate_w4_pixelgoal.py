"""W4 — `e`가 바뀔 때 **pixel goal 라벨**도 따라가는가.

pixel goal은 System 2의 학습 라벨이므로, GT path만 바꾸고 goal을 그대로 두면 라벨이 거짓이 된다.
3dloader `validate_pixel_goal.py`의 게이트 규약(V0/V2/V3/V5)을 그대로 승계하되, ESDF를
**mesh가 아니라 그 프레임 depth**에서 만든다는 점만 다르다.

| 게이트 | 무엇 | 통과 |
|---|---|---|
| V0 항등 | baseline `r_b`(=0.10, habitat 기본)에서 저장된 라벨이 그대로 살아남는가 | 전부 `unchanged` & 픽셀 변화 < 3 px |
| V2 단조 | `r_b`↑ ⇒ goal까지 거리 비증가 | 모든 인접쌍 |
| V3 도달성 | 새 경로 끝점 vs 조정된 goal | < 0.10 m |
| V5 분포 | `unchanged / adjusted / rejected` 표 | 서술 |

3dloader 규약 승계: `status=='unchanged'`면 **저장된 원본 라벨을 그대로 쓰고 재투영하지 않는다**
(재투영은 정수 절삭·hfov 잔차로 1~2 px를 흘리는데 그건 embodiment 효과가 아니다).

실행: /usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w4_pixelgoal.py --scene 17DRP5sb8fy --episode 0 --r_bs 0.10,0.20,0.35,0.50 --goal_adjust shift
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

from geometry_utils import save_jpg  # noqa: E402
from viz_utils import save_gallery  # noqa: E402

from augment2d import Aug2DCfg, Augmenter2D, Embodiment  # noqa: E402
from episode_io import load_episode  # noqa: E402
from validate_w3_embodiment import _augment_with, pill  # noqa: E402
from local_map import robot_xy_to_bev_ij  # noqa: E402
from report_common import (  # noqa: E402
    bev_rows, gate_note, goal_rows, legend_table, param_glossary, pipeline_note, status_glossary,
)
from viz2d import (  # noqa: E402
    GOAL_COLOR, GT_COLOR, RB_COLORS, START_COLOR, bev_rgb_planning, content_box, crop, draw_marker,
    draw_polyline, draw_pts, hstrip, label,
)

TOL_V0_PX = 3.0
TOL_V3_M = 0.10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='17DRP5sb8fy')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--preset', default='125cm_0_30')
    ap.add_argument('--n_frames', type=int, default=6)
    ap.add_argument('--r_bs', default='0.10,0.20,0.35,0.50')
    ap.add_argument('--h_nav', type=float, default=0.15)
    ap.add_argument('--h_b', type=float, default=1.25)
    ap.add_argument('--goal_adjust', default='retreat', choices=('retreat', 'shift', 'nearest'))
    ap.add_argument('--obstacle', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--unknown', default='nontraversable',
                    choices=('nontraversable', 'free', 'block'))
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--out_dir', default='logs/embodiment_augment2d/w4')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    r_bs = [float(x) for x in args.r_bs.split(',')]
    assert len(r_bs) <= len(RB_COLORS)
    ep = load_episode(args.data_root, args.scene, args.episode, args.preset, n_frames=args.n_frames)
    aug = Augmenter2D(ep.rig_fpv, ep.rig_ld,
                      Aug2DCfg(unknown=args.unknown, goal_adjust=args.goal_adjust,
                               obstacle=bool(args.obstacle)))

    rows, body = [], []
    for f in ep.frames:
        base = aug.augment(f, ep.poses_ld, Embodiment(h_nav=args.h_nav, h_b=args.h_b),
                           rng=np.random.default_rng(args.seed), with_obstacle=bool(args.obstacle))
        obs = base['obstacle']
        img = base['rgb_ld'].copy()
        draw_marker(img, f.goal_uv, GOAL_COLOR, 10, filled=False)   # 저장된 원본 라벨 = 빈 원
        box = content_box(base['map']['observed'])
        panels = []

        frows, drawn = [], []
        for i, r_b in enumerate(r_bs):
            e = Embodiment(r_b=r_b, h_nav=args.h_nav, h_b=args.h_b)
            r = _augment_with(aug, f, ep.poses_ld, e, obs, base)
            col = RB_COLORS[i % len(RB_COLORS)]
            gstat, traj = r['goal_status'], r.get('trajectory')
            goal = r.get('goal')

            label_goal = r.get('label_goal')
            if gstat in ('unchanged', 'unobserved'):
                # 3dloader 규약: 라벨이 안 바뀌면 저장된 값을 그대로 쓰고 재투영하지 않는다
                # (재투영은 정수 절삭·hfov 잔차로 1~2 px를 흘리는데 embodiment 효과가 아니다).
                uv, dpx, vis_why = tuple(f.goal_uv.astype(float)), 0.0, 'ok'
            elif label_goal is None or goal is None:
                uv, dpx, vis_why = (np.nan, np.nan), np.nan, 'no_safe_goal'
            else:
                u, v, Z, ok, why = aug.pixel_goal(label_goal, ep.poses_ld[f.idx], base['depth'])
                uv, dpx, vis_why = (u, v), float(np.max(np.abs(np.array([u, v]) - f.goal_uv))), why
                if ok:
                    draw_marker(img, uv, col, 6, filled=True)
                    drawn.append((f'r_b={r_b}', col))
            # 이 판정이 **어느 지도에서** 나왔는지 함께 보인다 (mesh가 아니라 depth 1장에서 나온 지도)
            pim = bev_rgb_planning(r['map'], r.get('nav_coarse'))
            draw_pts(pim, robot_xy_to_bev_ij(base['gt']), GT_COLOR, 1)
            if traj is not None:
                draw_polyline(pim, robot_xy_to_bev_ij(traj), col, 1)
            if label_goal is not None:
                draw_pts(pim, robot_xy_to_bev_ij(np.atleast_2d(label_goal)), GOAL_COLOR, 3)
            draw_pts(pim, np.array([[pim.shape[0] // 2, pim.shape[1] // 2]]), START_COLOR, 3)
            panels.append((r_b, pim, gstat))

            # V2는 **라벨** goal의 거리로 잰다 (계획 목표는 미관측 시 항상 물러나므로 자명해진다).
            # 기각된 조합은 학습에서 버려지므로 단조성 판정에서 제외한다(nan).
            dist = (float(np.linalg.norm(label_goal))
                    if (label_goal is not None and gstat != 'no_safe_point') else np.nan)
            reach = (float(np.linalg.norm(traj[-1] - goal))
                     if (traj is not None and goal is not None) else np.nan)
            frows.append(dict(idx=f.idx, r_b=r_b, gstat=gstat, status=r['status'], dpx=dpx,
                              dist=dist, reach=reach, vis=vis_why, uv=uv))

        rows.extend(frows)
        top = hstrip([label(img, f'f{f.idx}: lookdown + goal labels')], height=300)
        bot = hstrip([label(crop(pim, box).copy(), f'r_b={rb} {st}') for rb, pim, st in panels],
                     height=300)
        w = max(top.shape[1], bot.shape[1])
        pad = lambda a: np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)))  # noqa: E731
        save_jpg(np.vstack([pad(top), np.full((6, w, 3), 255, np.uint8), pad(bot)]),
                 out / f'w4_f{f.idx:04d}.jpg')
        legend = legend_table(
            goal_rows(with_orig=True)
            + [(c, 'dot', f'갱신된 goal — {n}', '이 r_b에서 라벨이 바뀐 위치(위 이미지의 채운 원, '
                                               '아래 지도의 같은 색 경로)') for n, c in drawn]
            + [(GT_COLOR, 'dot', '원본 GT path (노랑)', '아래 지도에 겹쳐 그린 원본 경로'),
               (GOAL_COLOR, 'dot', '라벨 goal 위치 (자홍 점)', '아래 지도에서 그 라벨이 가리키는 지점')]
            + bev_rows(with_cspace=True),
            title='범례 — 이 프레임의 이미지·지도')
        body.append(
            f'<h3>frame {f.idx}</h3><p>{legend}</p>'
            f'<table border=1 cellpadding=5><tr><th>r_b</th><th>goal</th><th>status</th>'
            f'<th>Δpx</th><th>거리(m)</th><th>도달오차(m)</th><th>가시성</th></tr>'
            + ''.join(f'<tr><td>{r["r_b"]}</td><td>{r["gstat"]}</td><td>{r["status"]}</td>'
                      f'<td>{r["dpx"]:.1f}</td><td>{r["dist"]:.2f}</td><td>{r["reach"]:.3f}</td>'
                      f'<td>{r["vis"]}</td></tr>' for r in frows) + '</table>'
            f'<img src="w4_f{f.idx:04d}.jpg" style="width:65%">')
        print(f'[W4] f{f.idx}: goal {[r["gstat"] for r in frows]} '
              f'dpx {[round(r["dpx"],1) for r in frows]} '
              f'dist {[round(r["dist"],2) for r in frows]}')

    base_rows = [r for r in rows if abs(r['r_b'] - aug.cfg.baseline_r_b) < 1e-9]
    # V0: baseline에서 라벨이 바뀌면 안 된다. `unobserved`는 라벨을 그대로 두므로 통과로 친다
    # (못 본 곳을 위험하다고 말할 근거가 없다 — `Augmenter2D.decide_goal` 참고).
    v0 = all(r['gstat'] in ('unchanged', 'unobserved') and r['dpx'] < TOL_V0_PX for r in base_rows)
    mono = []
    for idx in sorted({r['idx'] for r in rows}):
        d = [r['dist'] for r in rows if r['idx'] == idx]
        mono.append(all(np.isnan(d[k]) or np.isnan(d[k + 1]) or d[k] >= d[k + 1] - 1e-6
                        for k in range(len(d) - 1)))
    v2 = all(mono)
    reach = [r['reach'] for r in rows if np.isfinite(r['reach'])]
    v3 = (max(reach) <= TOL_V3_M) if reach else False
    cnt = {k: sum(1 for r in rows if r['gstat'] == k)
           for k in ('unchanged', 'unobserved', 'adjusted', 'no_safe_point')}
    rej = sum(1 for r in rows if str(r['status']).startswith('rejected'))

    # 세 모드 비교 (이미지는 --goal_adjust 것만, 표는 셋 다). 3dloader와 결론이 다르므로 근거를 남긴다.
    mode_rows = []
    for mode in ('retreat', 'shift', 'nearest'):
        c, mo, rj = _mode_counts(ep, args, r_bs, mode)
        mode_rows.append((mode, c, mo, rj))

    summary = (
        f'<p><b>{args.scene}</b> ep{args.episode} · preset {args.preset}</p>'
        + gate_note(
            what='<code>e</code>가 바뀔 때 System 2의 학습 라벨인 <b>pixel goal</b>이 함께 갱신되는지, '
                 '그리고 <b>바뀌면 안 될 때는 안 바뀌는지</b>를 본다.',
            why='GT path만 바꾸고 라벨을 그대로 두면 S2가 도달 불가능한 지점을 계속 배운다. '
                '반대로 근거 없이 라벨을 옮기면 baseline(원래 embodiment)에서도 데이터가 망가진다.',
            criterion='V0 baseline에서 라벨 유지 · V2 r_b↑에 goal 거리 비증가 · V3 경로 끝점 오차 ≤ 0.10 m')
        + '<h4>3dloader(3D mesh) 리포트와 무엇이 다른가</h4>'
        '<p>그림 구성은 3dloader의 pixel-goal 리포트와 같은 규약(빈 원=원본, 채운 원=갱신)을 쓰지만, '
        '<b>판정의 근거가 완전히 다르다</b>: 여기서는 씬 mesh도 3D occupancy도 정합(<code>T_sf2mesh</code>)도 '
        '쓰지 않는다. goal이 갈 수 있는 곳인지, 어디까지 물러나야 하는지를 전부 '
        '<b>그 프레임 depth 한 장에서 만든 지도</b>로 판정한다 — 그래서 아래 줄에 그 지도를 함께 그렸다. '
        '(mesh를 쓰는 곳은 W1 하나뿐이고, 거기서도 <i>비교 대상 오라클</i>로만 쓴다.)</p>'
        + pipeline_note()
        + param_glossary(preset=args.preset, r_b=args.r_bs, h_nav=args.h_nav, h_b=args.h_b,
                         unknown=args.unknown, goal_adjust=args.goal_adjust,
                         obstacle='O' if args.obstacle else 'X')
        + '<h4>pixel goal이란</h4><p>System 2의 학습 라벨이다. parquet <code>goal.&lt;rig&gt;</code>에 '
        '<b>640×480 룩다운(pitch_2) 이미지의 [u,v] 정수</b>로 저장돼 있고, 의미는 '
        '<b>몇 프레임 뒤 카메라 위치 아래의 바닥점</b>을 지금 프레임에 투영한 것이다. '
        'GT path만 바꾸고 이 라벨을 그대로 두면 거짓 라벨이 되므로 같이 갱신해야 한다.</p>'
        + status_glossary()
        + '<h4>게이트</h4>'
        f'<table border=1 cellpadding=6><tr><th>게이트</th><th>측정</th><th>기준</th><th></th></tr>'
        f'<tr><td>V0 항등 (baseline r_b={aug.cfg.baseline_r_b})</td>'
        f'<td>라벨 유지 '
        f'{sum(1 for r in base_rows if r["gstat"] in ("unchanged","unobserved"))}/{len(base_rows)} · '
        f'Δpx max {max((r["dpx"] for r in base_rows if np.isfinite(r["dpx"])), default=float("nan")):.1f}</td>'
        f'<td>전부 유지, &lt; {TOL_V0_PX} px</td><td>{pill(v0)}</td></tr>'
        f'<tr><td>V2 단조 (r_b↑ ⇒ goal 거리 비증가)</td><td>{sum(mono)}/{len(mono)} 프레임</td>'
        f'<td>전부</td><td>{pill(v2)}</td></tr>'
        f'<tr><td>V3 도달성 (경로 끝점 vs goal)</td>'
        f'<td>max {max(reach) if reach else float("nan"):.3f} m</td>'
        f'<td>≤ {TOL_V3_M} m</td><td>{pill(v3)}</td></tr>'
        f'<tr><td>V5 분포</td><td>unchanged {cnt["unchanged"]} / unobserved {cnt["unobserved"]} / '
        f'adjusted {cnt["adjusted"]} / no_safe_point {cnt["no_safe_point"]} · '
        f'최종 rejected {rej} (총 {len(rows)})</td>'
        f'<td>서술</td><td>—</td></tr></table>'
        f'<h4>goal 조정 모드 비교 (같은 프레임·같은 r_b)</h4>'
        f'<table border=1 cellpadding=6><tr><th>mode</th><th>unchanged</th><th>adjusted</th>'
        f'<th>unobserved</th><th>no_safe_point</th><th>최종 rejected</th><th>V2 단조</th></tr>'
        + ''.join(f'<tr><td>{"<b>"+m+"</b>" if m == args.goal_adjust else m}</td>'
                  f'<td>{c["unchanged"]}</td><td>{c["adjusted"]}</td><td>{c.get("unobserved",0)}</td>'
                  f'<td>{c["no_safe_point"]}</td>'
                  f'<td>{rj}</td><td>{"O" if mo else "X"}</td></tr>' for m, c, mo, rj in mode_rows)
        + '</table>'
        f'<p><b>3dloader(3D mesh)에서는 <code>shift</code>가 최선이었는데 2D에서는 뒤집힌다.</b> '
        f'<code>shift</code>는 거리를 유지한 채 방위만 트는데, 관측 부채꼴이 좁아 그 방위가 '
        f'미관측 영역으로 나가 안전점을 못 찾는다. <code>retreat</code>는 원본 경로를 따라 물러나므로 '
        f'항상 관측된 free 공간 안에 머문다.</p>'
        + '<h4>표의 열</h4><p><b>Δpx</b> = 저장된 원본 라벨과의 픽셀 거리(라벨을 안 바꾸면 0) · '
        '<b>거리(m)</b> = 로봇에서 라벨 goal까지 · <b>도달오차(m)</b> = 새 경로 끝점과 계획 목표의 차이 · '
        '<b>가시성</b> = 새 라벨 픽셀이 이미지 안이고 가려지지 않았는가</p>'
        f'<table border=1 cellpadding=6><tr><th>frame</th><th>r_b</th><th>goal</th><th>status</th>'
        f'<th>Δpx</th><th>거리(m)</th><th>도달오차(m)</th><th>가시성</th></tr>'
        + ''.join(f'<tr><td>{r["idx"]}</td><td>{r["r_b"]}</td><td>{r["gstat"]}</td>'
                  f'<td>{r["status"]}</td><td>{r["dpx"]:.1f}</td><td>{r["dist"]:.2f}</td>'
                  f'<td>{r["reach"]:.3f}</td><td>{r["vis"]}</td></tr>' for r in rows) + '</table>'
        f'<p><b><code>unobserved</code>가 왜 따로 있는가</b> — 원본 pixel goal은 여러 프레임 뒤의 '
        f'바닥점이라 <b>지금 시야각 안에 없을 때가 많다</b>. 그런 goal을 "clearance 0"으로 보면 '
        f'baseline에서도 라벨이 수백 px 튄다(실측 s8pcmisQ38h f11: 371 px). 못 본 곳을 위험하다고 '
        f'말할 근거가 없으므로 <b>라벨은 그대로 두고 계획 목표만</b> 관측 범위 안으로 물린다.</p>'
        f'<p><b>읽는 법</b> — V0가 통과해야 "augmentation을 꺼도(=원래 embodiment) 라벨이 그대로"가 '
        f'보장된다. V2는 큰 로봇일수록 goal이 더 가까워진다는 뜻(<code>shift</code> 모드는 거리를 '
        f'유지하고 방위만 트니, 거리 변화는 goal이 <b>기각</b>되어 경로가 짧아질 때만 생긴다). '
        f'V3는 새 GT 경로가 실제로 그 goal에 도착한다는 뜻이다.</p>'
        f'<p><b>가시성 기각</b>: 합성 장애물이 goal을 가리면 <code>occluded</code>로 기각된다 — '
        f'라벨이 "보이지 않는 픽셀"을 가리키면 안 되기 때문이다. 기각률이 높으면 "새 경로 위에서 '
        f'가장 먼 보이는 점"으로 라벨을 물리는 방식을 검토한다(현재는 기각).</p>')
    (out / 'summary.html').write_text(summary, encoding='utf-8')
    (out / 'body.html').write_text(''.join(body), encoding='utf-8')
    save_gallery(out, 'report.html', 'W4 — pixel goal 라벨 갱신', summary, ''.join(body))
    print(f'[W4] V0={v0} V2={v2} V3={v3} · {cnt} rejected={rej} -> {out}/report.html')
    return 0 if (v0 and v2 and v3) else 1


def _mode_counts(ep, args, r_bs, mode: str):
    """모드별 goal 조정 결과 카운트만 (이미지 없이). -> (counts, V2단조, rejected수)"""
    a = Augmenter2D(ep.rig_fpv, ep.rig_ld,
                    Aug2DCfg(unknown=args.unknown, goal_adjust=mode, obstacle=bool(args.obstacle)))
    cnt = {'unchanged': 0, 'unobserved': 0, 'adjusted': 0, 'no_safe_point': 0}
    rej, mono = 0, []
    for f in ep.frames:
        base = a.augment(f, ep.poses_ld, Embodiment(h_nav=args.h_nav, h_b=args.h_b),
                         rng=np.random.default_rng(args.seed), with_obstacle=bool(args.obstacle))
        d = []
        for r_b in r_bs:
            r = _augment_with(a, f, ep.poses_ld,
                              Embodiment(r_b=r_b, h_nav=args.h_nav, h_b=args.h_b),
                              base['obstacle'], base)
            cnt[r['goal_status']] = cnt.get(r['goal_status'], 0) + 1
            rej += int(str(r['status']).startswith('rejected'))
            g = r.get('label_goal')
            d.append(float(np.linalg.norm(g))
                     if (g is not None and r['goal_status'] != 'no_safe_point') else np.nan)
        mono.append(all(np.isnan(d[k]) or np.isnan(d[k + 1]) or d[k] >= d[k + 1] - 1e-6
                        for k in range(len(d) - 1)))
    return cnt, all(mono), rej


if __name__ == '__main__':
    raise SystemExit(main())
