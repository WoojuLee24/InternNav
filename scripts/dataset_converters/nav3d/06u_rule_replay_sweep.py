"""06u — 릴리스 vln_ce action 을 가장 잘 재현하는 '멈춰서 회전' 규칙(문턱 각도)을 찾는다.

규칙 후보 (운동학: 전진 0.25 m, 회전 15°, navmesh 위)
  목표 = reference_path[1:] 를 차례로 (몸이 r 안에 들어오면 다음 점).
  목표 방향 = 그 점까지 navmesh 최단 경로의 첫 `--look_m` 지점 방향.
  |목표 방향 − yaw| > X → 그쪽으로 15° 회전, 아니면 전진.
  X ∈ `--thr` · r ∈ `--reach`. 같은 시작 위치·시작 yaw (raw start_position / start_rotation) 에서 돌린다.

비교 (ep 마다, 릴리스 action[1:] 과): 완전 일치 · 편집 거리 / 릴리스 길이 · 회전 수 차 · 전진 수 차.

## 판정 (결과를 보기 전에 정함)
| | 기준 |
|---|---|
| U1 최적 X | 편집거리/길이 중앙이 가장 작은 (X, r) |
| U2 확정 | 그 (X, r) 의 완전 일치 ≥ 50 % 이면 '릴리스 규칙 = 문턱 X' 확정. 아니면 '가장 가까운 근사' 로만 보고 |
| 참고 | 라벨링 P1-A: habitat GreedyGeodesicFollower 완전 일치 0.3 %, 편집거리/길이 0.27 |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/06u_rule_replay_sweep.py
"""

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'gs_vlnpe'))
import nav3d_utils as nu  # noqa: E402

D = 'data/InternData-N1-v0.5-mini'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_scenes', type=int, default=15)
    ap.add_argument('--eps_per_scene', type=int, default=20)
    ap.add_argument('--thr', default='5,7.5,10,12.5,15,17.5,20,22.5,25,30')
    ap.add_argument('--reach', default='0.25,0.5')
    ap.add_argument('--look_m', type=float, default=0.25, help='최단 경로에서 방향을 잴 앞 거리 (전진 1 걸음)')
    ap.add_argument('--scene_root', default='data/scene_data/mp3d_ce/mp3d')
    ap.add_argument('--out', default='scripts/dataset_converters/nav3d/logs/06u_rule_replay/summary.json')
    return ap.parse_args()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def geo_heading(pf, p, g, look_m):
    import habitat_sim
    sp = habitat_sim.ShortestPath()
    sp.requested_start = nu.ours_to_hab(p).astype(np.float32)
    sp.requested_end = nu.ours_to_hab(g).astype(np.float32)
    if not pf.find_path(sp) or len(sp.points) < 2:
        d = g[:2] - p[:2]
        return math.atan2(d[1], d[0])
    P = nu.hab_to_ours(np.asarray(sp.points))[:, :2]
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = min(look_m, L[-1])
    q = np.array([np.interp(s, L, P[:, 0]), np.interp(s, L, P[:, 1])])
    d = q - p[:2]
    return math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else 0.0


def replay(pf, ref, yaw0, thr_deg, reach_m, look_m, max_steps=500):
    pos, yaw, acts = ref[0].copy(), float(yaw0), []
    thr = math.radians(thr_deg)
    for wp in ref[1:]:
        while len(acts) < max_steps and np.linalg.norm(wp[:2] - pos[:2]) >= reach_m:
            h = geo_heading(pf, pos, wp, look_m)
            e = (h - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(e) > thr:
                acts.append(2 if e > 0 else 3)
                yaw += math.radians(15) * (1 if e > 0 else -1)
            else:
                nxt = pos + np.array([0.25 * math.cos(yaw), 0.25 * math.sin(yaw), 0.0])
                q = np.asarray(pf.snap_point(nu.ours_to_hab(nxt).astype(np.float32)))
                pos = nu.hab_to_ours(q) if np.isfinite(q).all() else nxt     # navmesh 위로 (미끄러짐 근사)
                acts.append(1)
    return acts


def main():
    import pandas as pd
    import vlnce_gt
    args = parse()
    thrs = [float(x) for x in args.thr.split(',')]
    reaches = [float(x) for x in args.reach.split(',')]
    raw = vlnce_gt.raw_index(vlnce_gt.DEFAULT_RAW_ROOT)
    res = {(t, r): [] for t in thrs for r in reaches}
    n_ep = 0
    for sc in sorted(glob.glob(f'{D}/vln_ce/traj_data/r2r/*'))[:args.n_scenes]:
        s = Path(sc).name
        pf = nu.load_pathfinder(f'{args.scene_root}/{s}/{s}.navmesh')
        n = len(glob.glob(f'{sc}/data/chunk-000/*.parquet'))
        for ep in range(min(n, args.eps_per_scene)):
            try:
                e = raw[(s, vlnce_gt.episode_instruction(f'{D}/vln_ce/traj_data/r2r', s, ep))]
            except KeyError:
                continue
            rel = [int(a) for a in pd.read_parquet(f'{sc}/data/chunk-000/episode_{ep:06d}.parquet', columns=['action'])['action'] if a >= 0]
            ref = nu.hab_to_ours(np.asarray(e['reference_path']))
            q = e['start_rotation']                                   # habitat (x, y, z, w) — y 축 회전
            yaw0 = 2 * math.atan2(q[1], q[3]) + math.pi / 2           # habitat 앞(−z) → z-up yaw
            for t in thrs:
                for r in reaches:
                    a = replay(pf, ref, yaw0, t, r, args.look_m)
                    res[(t, r)].append({'exact': a == rel, 'ed': edit_distance(a, rel) / max(len(rel), 1),
                                        'dturn': sum(x in (2, 3) for x in a) - sum(x in (2, 3) for x in rel),
                                        'dfwd': a.count(1) - rel.count(1)})
            n_ep += 1
    R = {}
    for (t, r), v in res.items():
        R[f'thr{t}_r{r}'] = {'exact': float(np.mean([x['exact'] for x in v])), 'ed_med': float(np.median([x['ed'] for x in v])),
                             'dturn_med': float(np.median([x['dturn'] for x in v])), 'dturn_abs_med': float(np.median(np.abs([x['dturn'] for x in v]))),
                             'dfwd_abs_med': float(np.median(np.abs([x['dfwd'] for x in v])))}
    best = min(R, key=lambda k: R[k]['ed_med'])
    R['_best'] = best
    R['_episodes'] = n_ep
    R['_U2_confirmed'] = R[best]['exact'] >= 0.5
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(R, indent=1))
    for k, v in R.items():
        if not k.startswith('_'):
            print(f'{k:16s} exact {v["exact"]:.3f} · ed/len {v["ed_med"]:.3f} · Δturn {v["dturn_med"]:+.0f} (|·| {v["dturn_abs_med"]:.0f}) · |Δfwd| {v["dfwd_abs_med"]:.0f}')
    print('best', best, '· ep', n_ep, '· U2', R['_U2_confirmed'], '->', args.out)


if __name__ == '__main__':
    main()
