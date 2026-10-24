"""06w — 릴리스의 '멈춰서 회전' 문턱 X 를 **출발 직후 회전**으로 범위를 좁혀 정한다.

원리: 출발 직후 연속 회전 수 k 는 (시작 yaw, 첫 목표 방향, X) 만으로 정해진다.
  e0 = 첫 목표 방향 − 시작 yaw. 규칙 "|err| > X 이면 15° 회전" 이면
    회전 k 번 뒤 남은 오차 r_k = |e0 − 15°·k·sign| ≤ X      (그래서 멈춤)
    회전 k−1 번 뒤 남은 오차 r_{k−1} > X                      (그래서 k 번째 회전을 함)
  → 모든 ep 에서 max(r_k) ≤ X < min(r_{k−1}).  두 값이 겹치지 않으면 X 범위가 정해진다.
첫 목표 방향 = 시작점 → reference_path[1] 의 navmesh 최단 경로의 첫 `--look_m` 지점 방향.
회전 각은 vln_ce 는 정확히 15°, vln_pe 는 실제 yaw 변화를 쓴다 (H1 물리).

## 판정 (결과를 보기 전에 정함)
| | 기준 |
|---|---|
| W1 범위 | 하한 = 출발 회전 뒤 남은 오차의 p99, 상한 = 마지막 회전 직전 오차의 p1. 하한 < 상한이면 X ∈ [하한, 상한] |
| W2 7.5° | 7.5 가 그 범위 안이면 '7.5° 규칙과 맞다' |
p99 / p1 을 쓰는 이유: 짝짓기·경로 차이로 생기는 소수 이상치를 빼려고. max/min 도 같이 적는다.

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/turn_threshold_bound.py
"""

import argparse
import glob
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'gs_vlnpe'))
import nav3d_utils as nu  # noqa: E402

D = 'data/InternData-N1-v0.5-mini'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_scenes', type=int, default=0, help='0 = 전부')
    ap.add_argument('--look_m', type=float, default=0.25)
    ap.add_argument('--scene_root', default='data/scene_data/mp3d_ce/mp3d')
    ap.add_argument('--out', default='scripts/dataset_converters/nav3d/logs/checks/turn_threshold_bound/summary.json')
    return ap.parse_args()


def geo_heading(pf, p, g, look_m):
    import habitat_sim
    sp = habitat_sim.ShortestPath()
    sp.requested_start = nu.ours_to_hab(p).astype(np.float32)
    sp.requested_end = nu.ours_to_hab(g).astype(np.float32)
    if not pf.find_path(sp) or len(sp.points) < 2:
        return None
    P = nu.hab_to_ours(np.asarray(sp.points))[:, :2]
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = min(look_m, L[-1])
    q = np.array([np.interp(s, L, P[:, 0]), np.interp(s, L, P[:, 1])])
    d = q - p[:2]
    return math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else None


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


def bounds(after, before):
    after, before = np.abs(after), np.abs(before)
    lo, hi = float(np.percentile(after, 99)), float(np.percentile(before, 1)) if len(before) else float('inf')
    return {'n_after': int(len(after)), 'n_before': int(len(before)),
            'after_pct_50_90_99_max': np.percentile(after, [50, 90, 99, 100]).round(2).tolist(),
            'before_pct_min_1_10_50': ([float(before.min())] + np.percentile(before, [1, 10, 50]).round(2).tolist()) if len(before) else None,
            'W1_lo': lo, 'W1_hi': hi, 'W1_ok': lo < hi, 'W2_7.5_in': lo <= 7.5 < hi}


def main():
    import pandas as pd
    import vlnce_gt
    args = parse()
    raw_ce = vlnce_gt.raw_index(vlnce_gt.DEFAULT_RAW_ROOT)
    raw_pe = {}
    for f in glob.glob(f'{D}/vln_pe/raw_data/r2r/*/*.json.gz'):
        for e in json.load(gzip.open(f))['episodes']:
            raw_pe[(e['scene_id'].split('/')[-2], e['instruction']['instruction_text'].strip())] = e
    R = {}
    for fmt in ('vln_ce', 'vln_pe'):
        after, before, n0, n_ep = [], [], 0, 0
        scenes = sorted(glob.glob(f'{D}/{fmt}/traj_data/r2r/*'))
        for sc in scenes[:args.n_scenes] if args.n_scenes else scenes:
            s = Path(sc).name
            pf = nu.load_pathfinder(f'{args.scene_root}/{s}/{s}.navmesh')
            for m in [json.loads(l) for l in open(f'{sc}/meta/episodes.jsonl')]:
                ep = m['episode_index']
                if fmt == 'vln_ce':
                    try:
                        e = raw_ce[(s, vlnce_gt.episode_instruction(f'{D}/vln_ce/traj_data/r2r', s, ep))]
                    except KeyError:
                        continue
                    a = [int(x) for x in pd.read_parquet(f'{sc}/data/chunk-000/episode_{ep:06d}.parquet', columns=['action'])['action'] if x >= 0]
                    q = e['start_rotation']
                    yaw0 = math.degrees(2 * math.atan2(q[1], q[3])) + 90.0
                    p0 = nu.hab_to_ours(np.asarray(e['start_position']))
                    k = 0
                    while k < len(a) and a[k] in (2, 3) and a[k] == a[0]:
                        k += 1
                    yaws = [yaw0 + 15.0 * i * (1 if (a[0] == 2) else -1) for i in range(k + 1)]
                else:
                    e = next((raw_pe[(s, t.strip())] for t in m['tasks'] if (s, t.strip()) in raw_pe), None)
                    if e is None:
                        continue
                    d = pd.read_parquet(f'{sc}/data/chunk-000/episode_{ep:06d}.parquet',
                                        columns=['observation.action', 'observation.robot_position', 'observation.robot_yaw'])
                    a = d['observation.action'].to_numpy().tolist()
                    X = np.stack(d['observation.robot_position'].to_numpy())
                    Y = np.degrees(d['observation.robot_yaw'].to_numpy())
                    p0 = X[0]
                    k = 0
                    while k < len(a) and a[k] in (2, 3) and a[k] == a[0]:
                        k += 1
                    yaws = Y[:k + 1].tolist()
                if k >= len(yaws) or k >= len(a):                 # ep 전체가 회전뿐 (멈춘 순간이 없다)
                    continue
                ref = nu.hab_to_ours(np.asarray(e['reference_path']))
                h = geo_heading(pf, p0, ref[1], args.look_m)
                if h is None:
                    continue
                h = math.degrees(h)
                n_ep += 1
                after.append(wrapd(h - yaws[k]))                     # k 번 돈 뒤 (멈춘 순간) 남은 오차
                if k >= 1:
                    before.append(wrapd(h - yaws[k - 1]))            # 마지막 회전 직전 오차
                else:
                    n0 += 1
        B = bounds(np.array(after), np.array(before))
        B.update(episodes=n_ep, no_initial_turn=n0)
        R[fmt] = B
        print(fmt, json.dumps(B, ensure_ascii=False), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(R, indent=1))
    print('->', args.out)


if __name__ == '__main__':
    main()
