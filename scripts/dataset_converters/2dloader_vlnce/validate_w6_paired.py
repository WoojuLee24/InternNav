"""W6 — paired counterfactual 종합 그림.

**같은 base 이미지 한 장이 여러 `e`를 서비스한다**(문서 L157)를 한 장으로 보인다.
위: 모든 `e`가 공유하는 관측(FPV / 룩다운 / depth, 합성 장애물 포함).
아래: `e`별 BEV + 재계획 경로 + goal.

논문 §제안 C3(paired counterfactual batching)의 그림 후보이자, W2~W4가 각각 본 것을
한 프레임 안에서 동시에 확인하는 통합 점검이다.

실행: /usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w6_paired.py --scene 17DRP5sb8fy --episode 0 --r_bs 0.10,0.20,0.35,0.50 --obstacle 1
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

from augment2d import Aug2DCfg, Augmenter2D, Embodiment  # noqa: E402
from episode_io import load_episode  # noqa: E402
from local_map import limit_torch_threads, robot_xy_to_bev_ij  # noqa: E402
from obstacle_synth import ObstacleCfg, footprint_xy  # noqa: E402
from validate_w3_embodiment import _augment_with, max_deviation, plen  # noqa: E402
from report_common import (  # noqa: E402
    bev_rows, gate_note, legend_table, param_glossary, path_rows, pipeline_note, status_glossary,
)
from viz2d import (  # noqa: E402
    GOAL_COLOR, GT_COLOR, OBST_COLOR, RB_COLORS, START_COLOR, bev_rgb_planning, content_box, crop,
    draw_polyline, draw_pts, hstrip, label,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='17DRP5sb8fy')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--preset', default='125cm_0_30')
    ap.add_argument('--n_frames', type=int, default=4)
    ap.add_argument('--r_bs', default='0.10,0.20,0.35,0.50')
    ap.add_argument('--h_nav', type=float, default=0.15)
    ap.add_argument('--h_b', type=float, default=1.25)
    ap.add_argument('--obstacle', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--goal_adjust', default='retreat', choices=('retreat', 'shift', 'nearest'))
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--out_dir', default='logs/embodiment_augment2d/w6')
    args = ap.parse_args()
    limit_torch_threads()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    r_bs = [float(x) for x in args.r_bs.split(',')]
    assert len(r_bs) <= len(RB_COLORS)
    ep = load_episode(args.data_root, args.scene, args.episode, args.preset, n_frames=args.n_frames)
    aug = Augmenter2D(ep.rig_fpv, ep.rig_ld,
                      Aug2DCfg(goal_adjust=args.goal_adjust, obstacle=bool(args.obstacle)))

    rows, body = [], []
    for f in ep.frames:
        base = aug.augment(f, ep.poses_ld, Embodiment(h_nav=args.h_nav, h_b=args.h_b),
                           rng=np.random.default_rng(args.seed), with_obstacle=bool(args.obstacle))
        obs, gt = base['obstacle'], base['gt']
        d = base['depth224']
        top = hstrip([label(base['rgb_fpv'].copy(), f'shared FPV {ep.rig_fpv}'),
                      label(base['rgb_ld'].copy(), f'shared lookdown {ep.rig_ld}'),
                      label(colorize_depth(d, d > 0.05), 'shared depth')], height=300)

        box = content_box(base['map']['observed'])
        base_traj = _augment_with(aug, f, ep.poses_ld,
                                  Embodiment(r_b=aug.cfg.baseline_r_b, h_nav=args.h_nav, h_b=args.h_b),
                                  obs, base).get('trajectory')
        panels, frows = [], []
        for i, r_b in enumerate(r_bs):
            e = Embodiment(r_b=r_b, h_nav=args.h_nav, h_b=args.h_b)
            r = _augment_with(aug, f, ep.poses_ld, e, obs, base)
            m, traj, goal = r['map'], r.get('trajectory'), r.get('goal')
            S = m['bev_size']
            img = bev_rgb_planning(m, r.get('nav_coarse'))
            draw_pts(img, robot_xy_to_bev_ij(gt), GT_COLOR, 1)
            if obs is not None:
                cv2.polylines(img, [robot_xy_to_bev_ij(footprint_xy(obs))[:, ::-1]
                                    .astype(np.int32).reshape(-1, 1, 2)], True,
                              tuple(int(c) for c in OBST_COLOR), 1)
            col = RB_COLORS[i % len(RB_COLORS)]
            if traj is not None:
                draw_polyline(img, robot_xy_to_bev_ij(traj), col, 1)
            if goal is not None:
                draw_pts(img, robot_xy_to_bev_ij(goal[None]), GOAL_COLOR, 3)
            draw_pts(img, np.array([[S // 2, S // 2]]), START_COLOR, 3)
            # 모든 패널에 **같은** 창을 써야 e별 비교가 성립한다 (baseline map의 관측 영역 기준)
            kind = ('기각' if traj is None else
                    ('통과' if (base_traj is not None and max_deviation(traj, base_traj) < 0.10)
                     else '우회'))
            # cv2.putText는 한글을 못 그린다 — 이미지 안 라벨만 ASCII로
            ascii_kind = {'통과': 'pass', '우회': 'detour', '기각': 'reject'}[kind]
            panels.append(label(crop(img, box).copy(), f'r_b={r_b} {ascii_kind}'))
            frows.append(dict(idx=f.idx, r_b=r_b, kind=kind, status=r['status'], goal=r['goal_status'],
                              length=plen(traj) if traj is not None else float('nan'),
                              dev=(max_deviation(traj, base_traj)
                                   if traj is not None and base_traj is not None else float('nan'))))
        bottom = hstrip(panels, height=340)
        w = max(top.shape[1], bottom.shape[1])
        pad = lambda a: np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)))  # noqa: E731
        save_jpg(np.vstack([pad(top), np.full((6, w, 3), 255, np.uint8), pad(bottom)]),
                 out / f'w6_f{f.idx:04d}.jpg')

        rows.extend(frows)
        drawn = [(f'r_b={rb}', RB_COLORS[i % len(RB_COLORS)]) for i, rb in enumerate(r_bs)
                 if frows[i]['status'] in ('ok', 'fallback')]
        legend = legend_table(
            [(c, 'line', f'재계획 path — {n}', '아래 줄 각 패널에 그려진 그 r_b의 새 GT 경로')
             for n, c in drawn]
            + path_rows(with_gt=True, with_obst=obs is not None)
            + [(GOAL_COLOR, 'dot', 'goal (자홍 점)', '그 r_b에서 쓰는 목표 위치')]
            + bev_rows(with_cspace=True))
        body.append(
            f'<h3>frame {f.idx}</h3><p>{legend}</p>'
            f'<table border=1 cellpadding=5><tr><th>r_b</th><th>결과</th><th>status</th><th>goal</th>'
            f'<th>길이(m)</th><th>baseline 대비 이탈(m)</th></tr>'
            + ''.join(f'<tr><td>{r["r_b"]}</td><td><b>{r["kind"]}</b></td><td>{r["status"]}</td>'
                      f'<td>{r["goal"]}</td><td>{r["length"]:.2f}</td><td>{r["dev"]:.2f}</td></tr>'
                      for r in frows)
            + '</table><img src="w6_f{:04d}.jpg" style="width:100%">'.format(f.idx))
        print(f'[W6] f{f.idx}: ' + ' | '.join(f'r_b={r["r_b"]} {r["status"]} dev={r["dev"]:.2f}'
                                              for r in frows))

    ok_n = sum(1 for r in rows if r['kind'] != '기각')
    summary = (
        f'<p><b>{args.scene}</b> ep{args.episode} · preset {args.preset}</p>'
        + pipeline_note()
        + param_glossary(preset=args.preset, r_b=args.r_bs, h_nav=args.h_nav, h_b=args.h_b,
                         goal_adjust=args.goal_adjust, seed=args.seed,
                         obstacle='O' if args.obstacle else 'X')
        + gate_note(
            what='<b>같은 관측 한 장</b>이 여러 <code>e</code>에 대해 서로 다른 GT를 만들어내는지를 '
                 '한 장의 그림으로 확인한다.',
            why='논문 §제안 C3(paired counterfactual batching)의 재료가 실제로 만들어지는지 보는 게이트다. '
                '같은 입력에 정답이 하나뿐이면 모델이 <code>e</code>를 무시해도 loss가 최소가 되지만, '
                '같은 장면에 <code>e</code>별로 다른 정답이 들어오면 그 차이를 설명할 변수가 '
                '<code>e</code>밖에 없다.',
            criterion='정성 — r_b가 커질수록 통과 → 우회 → 기각으로 갈리는지')
        + '<h4>그림 읽는 법</h4><p><b>위 줄</b>은 모든 r_b가 공유하는 관측 3장(FPV / lookdown / depth) — '
        '합성 장애물이 세 장 모두에 정합돼 들어가 있다. <b>아래 줄</b>은 그 관측 하나에서 r_b만 바꿔 만든 '
        '지도 + 새 GT 경로다(관측 영역만 잘라 확대, 전방이 위쪽). 패널 제목의 '
        '<b>통과/우회/기각</b>은 baseline(r_b=0.10) 경로 대비 10 cm 이내면 통과, 그보다 벗어나면 우회, '
        '경로 자체가 안 나오면 기각이다.</p>'
        '<h4>장애물은 "통과할 수 있게" 놓는다</h4>'
        '<p>경로 위에 그냥 놓으면 모든 embodiment가 똑같이 막혀 counterfactual이 되지 않는다. '
        '그래서 장애물을 넣기 전 지도의 여유 <code>c0</code>를 읽어, 박스를 옆으로 '
        '<code>d = c0 − 반폭 − 2g</code>만큼 밀어 <b>clearance가 g인 통로</b>를 남긴다'
        '(<code>ObstacleCfg.gap_m</code>). 그러면 <code>r_b &lt; g</code>인 로봇만 지나간다. '
        f'이번 실행의 g 범위는 {ObstacleCfg().gap_m} m로, sweep하는 r_b 범위를 걸치도록 잡았다.</p>'
        + status_glossary()
        + f'<p>한 프레임의 <b>같은 관측</b>(위 3장)이 <b>{len(r_bs)}개의 서로 다른 GT</b>(아래 BEV)를 '
        f'만든다 — 이것이 논문 §제안 C3 paired counterfactual batching의 재료다. '
        f'같은 입력에 정답이 하나뿐이면 모델이 <code>e</code>를 무시해도 loss가 최소가 되지만, '
        f'같은 장면에 <code>e</code>별로 다른 정답이 들어오면 그 차이를 설명하는 변수가 '
        f'<code>e</code>밖에 없다.</p>'
        f'<table border=1 cellpadding=6><tr><th>항목</th><th>값</th></tr>'
        f'<tr><td>통과 / 우회 / 기각</td>'
        f'<td>{sum(1 for r in rows if r["kind"] == "통과")} / '
        f'{sum(1 for r in rows if r["kind"] == "우회")} / '
        f'{sum(1 for r in rows if r["kind"] == "기각")}</td></tr>'
        f'<tr><td>유효 샘플 (경로 생성됨)</td><td>{ok_n} / {len(rows)}</td></tr>'
        f'<tr><td>baseline 대비 이탈 &gt; 10 cm</td>'
        f'<td>{sum(1 for r in rows if np.isfinite(r["dev"]) and r["dev"] > 0.10)} / {len(rows)}</td></tr>'
        f'</table>'
        f'<table border=1 cellpadding=6><tr><th>frame</th><th>r_b</th><th>결과</th><th>status</th>'
        f'<th>goal</th><th>길이(m)</th><th>baseline 대비 이탈(m)</th></tr>'
        + ''.join(f'<tr><td>{r["idx"]}</td><td>{r["r_b"]}</td><td><b>{r["kind"]}</b></td>'
                  f'<td>{r["status"]}</td><td>{r["goal"]}</td><td>{r["length"]:.2f}</td>'
                  f'<td>{r["dev"]:.2f}</td></tr>' for r in rows) + '</table>'
        f'<p><b>기각/fallback이 큰 r_b에 몰리는 것은 버그가 아니라 이 방법의 한계다</b> — '
        f'시야각(HFOV 79°, 5 m) 안에서만 경로를 바꿀 수 있으므로, 큰 로봇이 지나갈 틈이 관측 범위 '
        f'안에 없으면 만들 GT가 없다(문서 L158). 학습에서는 이런 샘플을 버리거나 baseline 경로로 '
        f'되돌린다.</p>')
    (out / 'summary.html').write_text(summary, encoding='utf-8')
    (out / 'body.html').write_text(''.join(body), encoding='utf-8')
    save_gallery(out, 'report.html', 'W6 — paired counterfactual', summary, ''.join(body))
    print(f'[W6] 유효 {ok_n}/{len(rows)} -> {out}/report.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
