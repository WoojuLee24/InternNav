"""경로 결과의 충돌을 **관통 / r_b 미달**로 갈라 보고하고, 위반의 출처를 가린다.

왜 필요한가
----------
`03` 의 `stats` 는 씬 단위 요약이라 "왜 위반했는가"가 안 보인다. 특히 두 가지를 구분해야
파라미터를 고칠지 알고리즘을 고칠지 갈린다:

- **waypoint 부터 나쁨** — A* + refine 이 좁은 길을 골랐다. 삽입(`--post_refine`)으로 못 고친다
- **waypoint 는 멀쩡, spline 만 나쁨** — 보간이 모서리를 질러갔다. 삽입으로 고칠 수 있다

또 "관통(clearance <= 0)"과 "r_b 미달(clearance < r_b)"을 섞어 쓰면 안 된다. 관통은 벽을
통과한 것이고, r_b 미달은 로봇 몸이 스칠 수 있다는 뜻이다.

사용:
    python 03h_violation_breakdown.py --paths_dir /tmp/pe45_best/paths --r_b 0.20 --plan_r_b 0.28
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths_dir', required=True, help='03 의 paths/ 디렉터리')
    ap.add_argument('--r_b', type=float, required=True, help='검사 반경 — 로봇 몸 반지름')
    ap.add_argument('--plan_r_b', type=float, default=None, help='계획 반경 (생략 시 r_b)')
    args = ap.parse_args()
    plan_r_b = args.r_b if args.plan_r_b is None else args.plan_r_b

    files = sorted(glob.glob(os.path.join(args.paths_dir, '*.json')))
    assert files, f'{args.paths_dir} 에 json 이 없다'

    n_ep = n_ok = 0
    hard = rb_bad = 0
    wp_bad = wp_plan_bad = 0
    src_wp = src_spline = 0
    src_wp_seg = 0     # 'spline 만 나쁨' 중 waypoint **선분**은 이미 r_b 미달인 것
    hard_wp_seg = 0    # 관통 중 waypoint 선분이 이미 벽을 지난 것
    bite = []          # spline 이 깎아먹은 여유 (refine 후 - spline 후)
    traj_min, wp_min = [], []
    fired = reached = 0
    stops = Counter()

    for f in files:
        j = json.load(open(f))
        n_ep += j['stats']['n_total']
        for e in j['episodes']:
            ct = e.get('check_trajectory') or {}
            cr = e.get('check_refined') or {}
            tm, wm = ct.get('points_min_m'), cr.get('points_min_m')
            if tm is None:
                continue
            n_ok += 1
            traj_min.append(tm)
            if tm <= 0:
                hard += 1
                if (cr.get('segments_min_m') or 0.0) <= 1e-9 and 'segments_min_m' in cr:
                    hard_wp_seg += 1
            if tm < args.r_b - 1e-9:
                rb_bad += 1
            if wm is not None:
                wp_min.append(wm)
                bite.append(wm - tm)
                if wm < args.r_b - 1e-9:
                    wp_bad += 1
                if wm < plan_r_b - 1e-9:
                    wp_plan_bad += 1
                if tm < args.r_b - 1e-9:
                    if wm < args.r_b - 1e-9:
                        src_wp += 1
                    else:
                        src_spline += 1
                        if cr.get('segments_min_m', np.inf) < args.r_b - 1e-9:
                            src_wp_seg += 1
            pr = e.get('post_refine') or {}
            if pr.get('n_bad_start'):
                fired += 1
                if (pr.get('min_end') or 0.0) >= args.r_b - 1e-9:
                    reached += 1
                else:
                    stops[pr.get('stop', '?')] += 1

    traj_min, wp_min, bite = map(np.asarray, (traj_min, wp_min, bite))
    print(f'{len(files)}씬 · 에피소드 {n_ep} 중 경로 생성 {n_ok}'
          f'  (r_b {args.r_b} · plan_r_b {plan_r_b})\n')
    print(f'  관통 (여유 <= 0)        : {hard}')
    print(f'  r_b 미달 (여유 < {args.r_b}) : {rb_bad}\n')
    print(f'  ② refine 후 waypoint : 중앙 {np.median(wp_min):.3f}'
          f' · plan_r_b 미달 {wp_plan_bad} · r_b 미달 {wp_bad}')
    print(f'  ③ spline 후 궤적      : 중앙 {np.median(traj_min):.3f} · r_b 미달 {rb_bad}')
    print(f'\n  r_b 미달 {rb_bad}건의 출처:')
    print(f'    waypoint 부터 나쁨   : {src_wp:>4}   <- 계획 문제. 삽입으로 못 고침')
    print(f'    spline 만 나쁨       : {src_spline:>4}   <- post_refine 이 고칠 수 있는 것')
    # 위 분류는 waypoint **점**만 본다. 점은 멀쩡해도 점 사이 선분이 벽을 지나면(A* 가 얇은 벽을
    # 건너뜀) spline 탓이 아니다 — 그 몫을 따로 보인다(2026-10-13 추가).
    print(f'      └ 그중 waypoint 선분부터 나쁨 : {src_wp_seg:>4}   <- 계획 문제 (--astar_edge_check)')
    print(f'    관통 {hard}건 중 waypoint 선분이 이미 벽 관통 : {hard_wp_seg}')
    if len(bite):
        print(f'\n  spline 이 깎아먹은 양: 중앙 {np.median(bite):.3f} m'
              f' · p90 {np.percentile(bite, 90):.3f} · 최대 {bite.max():.3f}')
        print(f'    -> plan_r_b 는 r_b + (이 값) 이상이어야 한다')
    if fired:
        print(f'\n  post_refine 발동 {fired} · r_b 통과시킨 것 {reached}'
              f' · 못 넘긴 것 {fired - reached} {dict(stops) if stops else ""}')


if __name__ == '__main__':
    main()
