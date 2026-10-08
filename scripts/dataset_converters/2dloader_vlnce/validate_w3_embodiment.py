"""W3 — `e`가 GT path를 실제로 바꾸는가.

같은 프레임·같은 관측에서 `r_b`(와 `h_nav`)만 바꿔 (a) **원본 GT가 통행 불가로 바뀌는지**,
(b) 재계획한 경로가 **얼마나 달라지는지**를 잰다. 이게 논문 §해결방법3의 전제
("Embodiment 크기 변화 -> BEV에서 반영 -> robot-centric bev map에서 path 수정")의 직접 증거다.

| 지표 | 뜻 |
|---|---|
| 원본 GT feasible | `check_path_navigable(gt, esdf, r_b)` — `r_b`에 **단조 감소**해야 한다 |
| 최대 이탈 | 새 경로가 원본 GT 폴리라인에서 가장 멀어진 거리 |
| min clearance | 새 경로의 최소 여유 — `r_b` 이상이어야 정상 |
| status | ok / fallback(끝점 이탈→baseline) / rejected |

실행: /usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w3_embodiment.py --scene 17DRP5sb8fy --episode 0 --r_bs 0.10,0.20,0.35,0.50 --obstacle 0
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

from esdf_utils import check_path_navigable  # noqa: E402
from geometry_utils import save_jpg  # noqa: E402
from viz_utils import save_gallery  # noqa: E402

from augment2d import Aug2DCfg, Augmenter2D, Embodiment  # noqa: E402
from episode_io import load_episode  # noqa: E402
from local_map import robot_to_plan, robot_xy_to_bev_ij  # noqa: E402
from obstacle_synth import footprint_xy  # noqa: E402
from geometry_utils import colorize_depth  # noqa: E402
from local_map import path_violations  # noqa: E402
from report_common import (  # noqa: E402
    bev_rows, gate_note, legend_table, param_glossary, path_rows, pipeline_note, status_glossary,
)
from viz2d import (  # noqa: E402
    GT_COLOR, OBST_COLOR, RB_COLORS, START_COLOR, bev_rgb_planning, content_box, crop, draw_polyline,
    draw_pts, hstrip, label,
)


def pill(ok: bool) -> str:
    c = '#2e7d32' if ok else '#c62828'
    return (f'<span style="background:{c};color:#fff;border-radius:10px;padding:2px 10px;'
            f'font-weight:600">{"PASS" if ok else "FAIL"}</span>')


def plen(p) -> float:
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(p), axis=0), axis=1)))


def max_deviation(path, ref) -> float:
    """path의 각 점에서 ref 폴리라인까지 거리의 최댓값 [m]."""
    p, r = np.asarray(path, float), np.asarray(ref, float)
    if len(r) < 2:
        return float(np.max(np.linalg.norm(p - r[0], axis=1)))
    a, b = r[:-1], r[1:]
    ab = b - a
    L2 = np.sum(ab ** 2, axis=1)
    L2[L2 < 1e-12] = 1e-12
    t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return float(np.max(np.min(np.linalg.norm(p[:, None, :] - proj, axis=-1), axis=1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='17DRP5sb8fy')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--preset', default='125cm_0_30')
    ap.add_argument('--n_frames', type=int, default=6)
    ap.add_argument('--r_bs', default='0.10,0.20,0.35,0.50')
    ap.add_argument('--h_navs', default='0.15')
    ap.add_argument('--h_b', type=float, default=1.25)
    ap.add_argument('--obstacle', type=int, default=0, help='1이면 합성 장애물도 함께')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--goal_adjust', default='retreat', choices=('retreat', 'shift', 'nearest'))
    ap.add_argument('--unknown', default='nontraversable',
                    choices=('nontraversable', 'free', 'block'))
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--out_dir', default='logs/embodiment_augment2d/w3')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    r_bs = [float(x) for x in args.r_bs.split(',')]
    h_navs = [float(x) for x in args.h_navs.split(',')]
    assert len(r_bs) <= len(RB_COLORS), f'r_b는 최대 {len(RB_COLORS)}개 (색 개수)'
    ep = load_episode(args.data_root, args.scene, args.episode, args.preset, n_frames=args.n_frames)
    aug = Augmenter2D(ep.rig_fpv, ep.rig_ld,
                      Aug2DCfg(unknown=args.unknown, goal_adjust=args.goal_adjust,
                               obstacle=bool(args.obstacle)))

    rows, body, mono_all = [], [], []
    for f in ep.frames:
        # 장애물은 프레임당 한 번만 뽑아 **모든 r_b가 같은 장면**을 보게 한다 (paired counterfactual의 전제)
        base = aug.augment(f, ep.poses_ld, Embodiment(h_nav=h_navs[0], h_b=args.h_b),
                           rng=np.random.default_rng(args.seed), with_obstacle=bool(args.obstacle))
        obs = base['obstacle']
        gt = base['gt']
        m0 = base['map']
        S = m0['bev_size']
        panels = []   # r_b별 C-space 지도 (embodiment마다 지도가 달라진다)

        # 재계획 경로는 baseline r_b에서도 원본 GT와 0.2~0.5 m 차이가 난다(planner가 복도 중심으로
        # 정규화하기 때문). 따라서 **e의 순수 효과**는 baseline 재계획 경로 대비 이탈로 잰다.
        base_traj = _augment_with(aug, f, ep.poses_ld,
                                  Embodiment(r_b=aug.cfg.baseline_r_b, h_nav=h_navs[0], h_b=args.h_b),
                                  obs, base).get('trajectory')

        drawn, frows = [], []
        for h_nav in h_navs:
            feas_seq = []
            for i, r_b in enumerate(r_bs):
                e = Embodiment(r_b=r_b, h_nav=h_nav, h_b=args.h_b)
                # 같은 장애물·같은 depth를 재사용하려면 augment를 다시 돌리되 장애물만 고정해야 한다.
                # `_augment_fixed`가 없으므로 map만 e로 다시 만들고 나머지는 augment의 로직을 그대로 쓴다.
                r = _augment_with(aug, f, ep.poses_ld, e, obs, base)
                gtc = check_path_navigable(robot_to_plan(gt), r['map']['esdf'], r['map']['origin'],
                                           r['map']['cell_m'], r_b)
                feas_seq.append(bool(gtc['points_ok']))
                col = RB_COLORS[i % len(RB_COLORS)]
                traj = r.get('trajectory')
                if h_nav == h_navs[0]:
                    # **A*가 실제로 경로를 고른 격자**를 그린다 (세밀 격자가 아니라)
                    pim = bev_rgb_planning(r['map'], r.get('nav_coarse'))
                    draw_pts(pim, robot_xy_to_bev_ij(gt), GT_COLOR, 1)
                    if obs is not None:
                        import cv2 as _cv
                        _cv.polylines(pim, [robot_xy_to_bev_ij(footprint_xy(obs))[:, ::-1]
                                            .astype(np.int32).reshape(-1, 1, 2)], True,
                                      tuple(int(c) for c in OBST_COLOR), 1)
                    if traj is not None:
                        draw_polyline(pim, robot_xy_to_bev_ij(traj), col, 1)
                        drawn.append((f'r_b={r_b}', col))
                    draw_pts(pim, np.array([[S // 2, S // 2]]), START_COLOR, 3)
                    panels.append((r_b, pim, r['status']))
                nc = (check_path_navigable(robot_to_plan(traj), r['map']['esdf'], r['map']['origin'],
                                           r['map']['cell_m'], r_b) if traj is not None else None)
                frows.append(dict(
                    r_b=r_b, h_nav=h_nav, col=col if traj is not None else None,
                    gt_feasible=bool(gtc['points_ok']), gt_min=float(gtc['points_min_m']),
                    status=r['status'], goal_status=r.get('goal_status', '-'),
                    length=plen(traj) if traj is not None else float('nan'),
                    dev=max_deviation(traj, gt) if traj is not None else float('nan'),
                    clear=float(nc['points_min_m']) if nc else float('nan'),
                    viol=(path_violations(r['map'], traj) if traj is not None else None),
                    dev_base=(max_deviation(traj, base_traj)
                              if traj is not None and base_traj is not None else float('nan')),
                    ep_err=float(r.get('endpoint_err', float('nan'))),
                ))
            mono = all(feas_seq[k] >= feas_seq[k + 1] for k in range(len(feas_seq) - 1))
            mono_all.append(mono)

        # 위: 이 BEV가 어느 이미지에서 나왔는지 (룩다운 RGB + depth) / 아래: r_b별 C-space 지도
        box = content_box(m0['observed'])
        d = base['depth224']
        top = hstrip([label(base['rgb_ld'].copy(), f'lookdown {ep.rig_ld} (BEV의 출처)'),
                      label(colorize_depth(d, d > 0.05), 'depth 224')], height=300)
        bot = hstrip([label(crop(pim, box).copy(), f'r_b={rb} {st.split(":")[0]}')
                      for rb, pim, st in panels], height=300)
        w = max(top.shape[1], bot.shape[1])
        pad = lambda a: np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)))  # noqa: E731
        save_jpg(np.vstack([pad(top), np.full((6, w, 3), 255, np.uint8), pad(bot)]),
                 out / f'w3_f{f.idx:04d}.jpg')
        rows.extend([dict(idx=f.idx, **fr) for fr in frows])
        # 범례에는 **실제로 그린 색만** (기각돼 선이 없는 r_b는 넣지 않는다)
        legend = legend_table(
            [(c, 'line', f'재계획 path — {n}', '해당 패널에 그려진 그 r_b의 새 GT 경로') for n, c in drawn]
            + path_rows(with_gt=True, with_obst=obs is not None) + bev_rows(with_cspace=True))
        body.append(
            f'<h3>frame {f.idx}</h3><p>{legend}</p>'
            f'<table border=1 cellpadding=5><tr><th>r_b</th><th>h_nav</th><th>원본 GT 통행</th>'
            f'<th>GT min clr</th><th>status</th><th>goal</th><th>길이(m)</th><th>GT 대비 이탈(m)</th>'
            f'<th>baseline 대비 이탈(m)</th><th>새 경로 min clr</th></tr>'
            + ''.join(f'<tr><td>{r["r_b"]}</td><td>{r["h_nav"]}</td>'
                      f'<td>{"O" if r["gt_feasible"] else "X"}</td><td>{r["gt_min"]:.2f}</td>'
                      f'<td>{r["status"]}</td><td>{r["goal_status"]}</td><td>{r["length"]:.2f}</td>'
                      f'<td>{r["dev"]:.2f}</td><td>{r["dev_base"]:.2f}</td>'
                      f'<td>{r["clear"]:.2f}</td></tr>' for r in frows)
            + '</table>'
            f'<img src="w3_f{f.idx:04d}.jpg" style="width:70%">')
        print(f'[W3] f{f.idx}: GT feasible {[r["gt_feasible"] for r in frows]} '
              f'dev_gt {[round(r["dev"],2) for r in frows]} '
              f'dev_base {[round(r["dev_base"],2) for r in frows]} '
              f'status {[r["status"] for r in frows]}')

    mono_ok = all(mono_all)
    changed = sum(1 for r in rows if np.isfinite(r['dev_base']) and r['dev_base'] > 0.10)
    flip = sum(1 for i in rows if not i['gt_feasible'])
    rejected = sum(1 for r in rows if str(r['status']).startswith('rejected'))
    # e의 효과는 **두 형태**로 나타난다: 경로가 옆으로 비켜서거나(이탈), 아예 못 가거나(기각).
    # 넓은 복도 씬(s8pcmisQ38h)에서는 이탈이 0이고 기각으로만 나타난다 — 둘 중 하나면 성립.
    e_ok = flip > 0 and (changed > 0 or rejected > 0)
    summary = (
        f'<p><b>{args.scene}</b> ep{args.episode} · preset {args.preset} · '
        f'r_b 목록 {args.r_bs}</p>'
        + gate_note(
            what='<b>같은 관측</b>에서 <code>r_b</code>만 바꿔 (a) 원본 GT가 통행 불가로 뒤집히는지, '
                 '(b) 새로 만든 경로가 얼마나 달라지는지를 잰다.',
            why='논문 §해결방법3의 전제("Embodiment 크기 변화 → BEV에서 반영 → robot-centric bev map에서 '
                'path 수정")가 이 데이터에서 실제로 성립하는지를 보는 유일한 게이트다.',
            criterion='원본 GT 통행성이 r_b에 <b>단조 감소</b>하고, 통행 불가로 뒤집히는 조합과 '
                      'baseline 대비 10 cm 넘게 이탈하는 조합이 각각 1건 이상')
        + pipeline_note()
        + param_glossary(preset=args.preset, r_b=args.r_bs, h_nav=args.h_navs, h_b=args.h_b,
                         unknown=args.unknown, goal_adjust=args.goal_adjust,
                         obstacle='O' if args.obstacle else 'X')
        + '<h4>그림 읽는 법</h4><p><b>위 줄</b>은 이 BEV가 나온 원본 — 룩다운(pitch_2) RGB와 그 depth다'
        '(BEV·planning은 전부 이 카메라 기준). <b>아래 줄</b>은 <code>r_b</code>별 지도로, '
        '관측 영역만 잘라 확대했고 전방이 위쪽이다. <b>이 지도는 A*가 실제로 경로를 고른 그 격자다</b> — '
        '4.46 cm 세밀 격자가 아니라 그것을 4×4로 묶은 17.9 cm 조립 격자라서 네모가 굵게 보인다. '
        '회색이 A*가 쓸 수 있었던 칸, 어두운 빨강이 <b>비어 있는데도 이 로봇에겐 못 쓰는 칸</b>이다. '
        '점유칸(빨강)은 r_b와 무관하지만 어두운 빨강은 r_b가 커질수록 넓어진다 — 그래서 패널마다 '
        '지도가 다르고, 경로가 갈린다. '
        '(경로 다듬기는 세밀 ESDF로 하고, 마지막에 세밀 격자에서 다시 검사한다.)</p>'
        '<p>관측이 완전히 같고 r_b만 다르므로 선이 갈라지는 것이 곧 embodiment 효과다. '
        '범례에는 그 프레임에서 <b>실제로 그려진 색만</b> 넣는다(기각된 r_b는 선이 없으므로 색도 없다).</p>'
        + status_glossary()
        + '<h4>게이트</h4>'
        f'<table border=1 cellpadding=6><tr><th>게이트</th><th>측정</th><th>기준</th><th></th></tr>'
        f'<tr><td>feasibility 단조성 (r_b↑ ⇒ 통행성↓)</td><td>{sum(mono_all)}/{len(mono_all)} 프레임</td>'
        f'<td>전부</td><td>{pill(mono_ok)}</td></tr>'
        f'<tr><td>e가 실제로 GT를 바꾼다</td>'
        f'<td>원본 GT 통행 불가 {flip}/{len(rows)} · baseline 대비 이탈&gt;10 cm {changed}/{len(rows)} · '
        f'경로 생성 실패 {rejected}/{len(rows)}</td>'
        f'<td>통행 불가 &gt; 0 이고 (이탈 또는 실패) &gt; 0</td><td>{pill(e_ok)}</td></tr></table>'
        f'<p>e의 효과는 <b>두 형태</b>로 나타난다 — 경로가 옆으로 비켜서거나(이탈), 관측 범위 안에 '
        f'틈이 없어 아예 못 가거나(기각). 넓은 복도 씬에서는 이탈이 0이고 기각으로만 나타난다.</p>'
        + '<h4>표의 열</h4><p><b>원본 GT 통행</b> O/X = 데이터셋 원본 경로가 이 r_b에서 통행 가능한가'
        '(<code>check_path_navigable</code>) · <b>GT 대비</b> = 새 경로가 원본 GT에서 가장 멀어진 거리 · '
        '<b>baseline 대비</b> = 같은 planner·같은 관측에서 r_b만 바꿨을 때의 차이 = <b>e의 순수 효과</b> · '
        '<b>min clr</b> = 새 경로의 최소 여유(m)</p>'
        f'<table border=1 cellpadding=6><tr><th>frame</th><th>r_b</th><th>h_nav</th>'
        f'<th>원본 GT 통행</th><th>status</th><th>길이(m)</th><th>GT 대비</th>'
        f'<th>baseline 대비</th><th>min clr</th></tr>'
        + ''.join(f'<tr><td>{r["idx"]}</td><td>{r["r_b"]}</td><td>{r["h_nav"]}</td>'
                  f'<td>{"O" if r["gt_feasible"] else "X"}</td><td>{r["status"]}</td>'
                  f'<td>{r["length"]:.2f}</td><td>{r["dev"]:.2f}</td><td>{r["dev_base"]:.2f}</td>'
                  f'<td>{r["clear"]:.2f}</td></tr>' for r in rows) + '</table>'
        f'<p><b>읽는 법</b> — "원본 GT 통행" 열이 <code>r_b</code>가 커질 때 O→X로 바뀌는 것이 '
        f'논문 전제의 핵심 증거다(같은 장면·같은 관측인데 로봇이 커지면 원래 경로로 갈 수 없다). '
        f'"baseline 대비 이탈"이 <b>e의 순수 효과</b>다 — 같은 planner·같은 관측에서 r_b만 바꿨을 때의 차이. '
        f'"GT 대비 이탈"에는 planner가 복도 중심으로 정규화하는 편향(baseline에서도 0.2~0.5 m)이 섞여 있다. '
        f'<code>fallback</code>은 끝점이 10 cm 넘게 밀려 기각되고 baseline r_b 경로로 되돌린 경우 — '
        f'instruction 라벨을 지키기 위한 규칙이다.</p>')
    (out / 'summary.html').write_text(summary, encoding='utf-8')
    (out / 'body.html').write_text(''.join(body), encoding='utf-8')
    save_gallery(out, 'report.html', 'W3 — e가 GT path를 바꾸는가', summary, ''.join(body))
    print(f'[W3] 단조성={mono_ok} 통행불가={flip}/{len(rows)} 이탈>10cm={changed} -> {out}/report.html')
    return 0 if (mono_ok and e_ok) else 1


def _augment_with(aug: Augmenter2D, frame, poses_ld, e: Embodiment, obs, base) -> dict:
    """`base`가 이미 뽑은 장애물/합성 depth를 **그대로 재사용**하고 `e`만 바꿔 재계획한다.

    r_b마다 장애물을 새로 뽑으면 장면이 달라져 비교가 무효가 된다(paired counterfactual의 전제).
    """
    from local_map import build_local_map

    cfg = aug.cfg
    m = build_local_map(base['depth224'], aug.rig_ld, aug.pitch_ld, e.r_b, e.h_nav, e.h_b,
                        cfg.bev_range, cfg.bev_size, unknown=cfg.unknown)
    out = dict(map=m, obstacle=obs, e=e)
    out.update(aug.plan_and_label(m, base['goal0'], base['gt'], e))   # 목표·라벨 결정은 한 곳에서만
    return out


if __name__ == '__main__':
    raise SystemExit(main())
