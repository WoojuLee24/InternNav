"""06t — 릴리스 vln_ce · vln_pe 의 '멈춰서 회전' 문턱을 **navmesh 최단 경로 방향**을 목표로 두고 잰다 (06s S1 의 개선).

06s S1 에서 R2R 경유점 직선 방향을 목표로 두니 전진 프레임 오차가 중앙 34~45° 였다 — 벽을 돌아가는 길을 못 본다.
여기서는 프레임마다 '지금 목표 경유점' 까지 navmesh 최단 경로(`find_path`)를 구해, 그 경로의 첫 방향을 목표 방향으로 쓴다
(habitat GreedyGeodesicFollower · 릴리스 플래너 모두 navmesh 최단 경로를 따른다고 보는 가정).

- 목표 경유점: reference_path[1:] 를 차례로. 몸이 `--reach_m` 안에 들어오면 다음 점 (03 이전 이탈 대비: 다음 점이 더 가까우면 건너뜀).
- err = 목표 방향 − 몸(카메라) yaw. 프레임에서 한 action 별로 |err| 분포.
- 첫 방향은 경로의 `--look_m` 앞 점 방향 (경로 첫 꼭짓점이 몸 바로 옆이면 방향이 튐).

## 판정 (결과를 보기 전에 정함)
| | 기준 |
|---|---|
| T0 yaw 규약 | 전진 프레임에서 실제 이동 방향 − yaw 중앙 |·| ≤ 2° (vln_ce) / ≤ 5° (vln_pe) |
| T1 문턱 | X̂ = 전진 |err| ≤ X 와 회전 |err| > X 를 가장 잘 가르는 각 · 그때 오분류율 |
| T2 선명함 | 오분류율 ≤ 10 % 이면 'X̂ 규칙이 맞다', 아니면 '단순 문턱 규칙이 아니다' |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/06t_turn_rule_geodesic.py
"""

import argparse
import glob
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/ (이 파일은 nav3d/checks/ 에 있다)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'gs_vlnpe'))
import nav3d_utils as nu  # noqa: E402

D = 'data/InternData-N1-v0.5-mini'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_scenes', type=int, default=15)
    ap.add_argument('--eps_per_scene', type=int, default=20)
    ap.add_argument('--reach_m', type=float, default=0.25, help='경유점 도달 반경 (InternNav ShortestPathFollower 0.25)')
    ap.add_argument('--look_m', type=float, default=0.5)
    ap.add_argument('--scene_root', default='data/scene_data/mp3d_ce/mp3d')
    ap.add_argument('--out', default='scripts/dataset_converters/nav3d/logs/06t_turn_rule/summary.json')
    return ap.parse_args()


def heading_along(pf, p_ours, g_ours, look_m):
    """navmesh 최단 경로의 look_m 앞 점 방향 (rad, z-up). 경로 없으면 None."""
    import habitat_sim
    sp = habitat_sim.ShortestPath()
    sp.requested_start = pf.snap_point(nu.ours_to_hab(p_ours).astype(np.float32))
    sp.requested_end = pf.snap_point(nu.ours_to_hab(g_ours).astype(np.float32))
    if not pf.find_path(sp) or len(sp.points) < 2:
        return None
    P = nu.hab_to_ours(np.asarray(sp.points))[:, :2]
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = min(look_m, L[-1])
    q = np.array([np.interp(s, L, P[:, 0]), np.interp(s, L, P[:, 1])])
    d = q - p_ours[:2]
    if np.linalg.norm(d) < 1e-6:
        return None
    return math.atan2(d[1], d[0])


def episode_rows(pf, X, yaw, A, ref, reach_m, look_m):
    """X (T,3) · yaw (T,) · A[t] = t 에서 한 action · ref (M,3) -> [(a, err_deg)] 와 T0 용 [(이동 방향 − yaw)]."""
    rows, t0, wi = [], [], 1
    for t in range(len(X) - 1):
        while wi < len(ref) and (np.linalg.norm(ref[wi, :2] - X[t, :2]) < reach_m
                                 or (wi + 1 < len(ref) and np.linalg.norm(ref[wi + 1, :2] - X[t, :2]) < np.linalg.norm(ref[wi, :2] - X[t, :2])
                                     and np.linalg.norm(ref[wi, :2] - X[t, :2]) < 1.0)):
            wi += 1
        if wi >= len(ref):
            break
        h = heading_along(pf, X[t], ref[wi], look_m)
        if h is None:
            continue
        e = (h - yaw[t] + math.pi) % (2 * math.pi) - math.pi
        rows.append((int(A[t]), math.degrees(e)))
        if A[t] == 1:
            mv = X[t + 1, :2] - X[t, :2]
            if np.linalg.norm(mv) > 0.05:
                t0.append(math.degrees((math.atan2(mv[1], mv[0]) - yaw[t] + math.pi) % (2 * math.pi) - math.pi))
    return rows, t0


def summarize(rows, t0):
    a = np.array([r[0] for r in rows])
    e = np.abs(np.array([r[1] for r in rows]))
    fw, tn = e[a == 1], e[(a == 2) | (a == 3)]
    xs = np.arange(0.5, 45.01, 0.5)
    mis = np.array([((fw > x).sum() + (tn <= x).sum()) / (len(fw) + len(tn)) for x in xs])
    k = int(np.argmin(mis))
    return {'T0_move_minus_yaw_median_deg': float(np.median(np.abs(t0))) if t0 else None,
            'n_forward': int(len(fw)), 'n_turn': int(len(tn)),
            'forward_err_pct_50_90_95_99': np.percentile(fw, [50, 90, 95, 99]).round(2).tolist(),
            'turn_err_pct_1_5_10_25_50': np.percentile(tn, [1, 5, 10, 25, 50]).round(2).tolist(),
            'misclass_at': {str(x): float(mis[int(round((x - 0.5) / 0.5))]) for x in (7.5, 15.0, 22.5, 30.0)},
            'T1_best_deg': float(xs[k]), 'T1_best_misclass': float(mis[k]), 'T2_sharp': bool(mis[k] <= 0.10)}


def main():
    import pandas as pd
    import vlnce_gt
    args = parse()
    R = {}
    raw_pe = {}
    for f in glob.glob(f'{D}/vln_pe/raw_data/r2r/*/*.json.gz'):
        for e in json.load(gzip.open(f))['episodes']:
            raw_pe[(e['scene_id'].split('/')[-2], e['instruction']['instruction_text'].strip())] = e
    raw_ce = vlnce_gt.raw_index(vlnce_gt.DEFAULT_RAW_ROOT)
    for fmt in ('vln_ce', 'vln_pe'):
        rows, t0, n_ep = [], [], 0
        for sc in sorted(glob.glob(f'{D}/{fmt}/traj_data/r2r/*'))[:args.n_scenes]:
            s = Path(sc).name
            pf = nu.load_pathfinder(f'{args.scene_root}/{s}/{s}.navmesh')
            meta = [json.loads(l) for l in open(f'{sc}/meta/episodes.jsonl')]
            for m in meta[:args.eps_per_scene]:
                ep = m['episode_index']
                if fmt == 'vln_ce':
                    try:
                        g = vlnce_gt.load_gt_episode_vlnce(f'{D}/vln_ce/traj_data/r2r', s, ep, '125cm_0deg', vlnce_gt.DEFAULT_RAW_ROOT)
                        e = raw_ce[(s, vlnce_gt.episode_instruction(f'{D}/vln_ce/traj_data/r2r', s, ep))]
                    except KeyError:
                        continue
                    X = g['cam_xyz'] - [0, 0, g['h_b']]
                    fwd = g['poses_c2w'][:, :3, 2]
                    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
                    a = pd.read_parquet(f'{sc}/data/chunk-000/episode_{ep:06d}.parquet', columns=['action'])['action'].to_numpy()
                    A = np.r_[a[1:], 0]
                else:
                    e = next((raw_pe[(s, t.strip())] for t in m['tasks'] if (s, t.strip()) in raw_pe), None)
                    if e is None:
                        continue
                    d = pd.read_parquet(f'{sc}/data/chunk-000/episode_{ep:06d}.parquet',
                                        columns=['observation.action', 'observation.robot_position', 'observation.robot_yaw'])
                    X = np.stack(d['observation.robot_position'].to_numpy())
                    yaw = d['observation.robot_yaw'].to_numpy()
                    A = d['observation.action'].to_numpy()
                ref = nu.hab_to_ours(np.asarray(e['reference_path']))
                r_, t_ = episode_rows(pf, X, yaw, A, ref, args.reach_m, args.look_m)
                rows += r_
                t0 += t_
                n_ep += 1
        S = summarize(rows, t0)
        S['episodes'] = n_ep
        R[fmt] = S
        print(fmt, json.dumps(S, ensure_ascii=False), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(R, indent=1))
    print('->', args.out)


if __name__ == '__main__':
    main()
