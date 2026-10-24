"""06s — 릴리스 vln_ce · vln_pe 의 '멈춰서 회전' 문턱을 데이터에서 잰다.

질문: 릴리스 생성기가 "목표 방향과 X° 넘게 어긋나면 제자리 15° 회전, 아니면 전진" 규칙인가, X 는 얼마인가?
(저장소의 `scripts/visualization/eval_gt_collect.py` 는 X = 7.5° — 팀이 만든 재현 스크립트이고 원 생성기 코드는 없다.)

방법
- 릴리스 ep 를 지시문으로 raw R2R (reference_path) 와 짝짓는다.
- 프레임마다 그 순간 몸(vln_pe: robot_position/yaw) · 카메라(vln_ce: pose, 시작 기준 → raw start 로 world 복원) 에서
  '지금 목표 경유점' (eval_gt_collect 와 같이: 0.15 m 안에 들어오면 다음 점) 방향과의 차 err 를 잰다.
- 그 프레임에서 한 action 별로 |err| 분포를 본다. 규칙이 X 라면: 전진 프레임 |err| ≤ X, 회전 프레임 |err| > X 로 갈린다.

## 판정 (결과를 보기 전에 정함)
| | 기준 |
|---|---|
| S0 짝짓기 | 짝지은 ep 의 첫 프레임 위치 ↔ reference_path[0] 중앙 ≤ 0.3 m (좌표 복원이 맞는지) |
| S1 문턱 추정 | X̂ = 전진 |err| 와 회전 |err| 를 가장 잘 가르는 각 (오분류 최소). 7.5° 규칙이 맞으면 X̂ ∈ [5°, 10°] |
| S2 횟수 재현 | 같은 reference_path · 같은 시작 자세에서 규칙(문턱 X ∈ 7.5/15/22.5/30°)을 운동학으로 돌려 ep 마다 회전·전진 action 수를 릴리스와 비교. 가장 가까운 X 를 고른다 (|Δ회전 수| 중앙 최소) | 

2026-10-20 실측: S1 은 R2R 경유점을 목표로 두면 전진 프레임 |err| 중앙 34~45° — 릴리스 생성기는 R2R 경유점을 바로 목표로 하지 않는다 (자 설계 실패, S2 로 대신).

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/turn_rule_check.py
"""

import argparse
import glob
import gzip
import json
import math
from pathlib import Path

import numpy as np

D = 'data/InternData-N1-v0.5-mini'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_scenes', type=int, default=15)
    ap.add_argument('--eps_per_scene', type=int, default=20)
    ap.add_argument('--close_m', type=float, default=0.15, help='목표 경유점 도달 = eval_gt_collect CLOSE_ENOUGH (0.25 × 0.6)')
    ap.add_argument('--out', default='scripts/dataset_converters/nav3d/logs/checks/turn_rule_check/summary.json')
    return ap.parse_args()


def hab_xy(p):
    """habitat (x, y-up, z) -> nav3d/Isaac z-up xy (x, -z)."""
    return np.array([p[0], -p[2]])


def raw_index(fmt):
    idx = {}
    for split in ('train', 'val_seen', 'val_unseen'):
        for f in glob.glob(f'{D}/{fmt}/raw_data/r2r/{split}/{split}.json.gz'):
            for e in json.load(gzip.open(f))['episodes']:
                idx.setdefault((e['scene_id'].split('/')[-2] if '/' in e['scene_id'] else e['scene_id'], e['instruction']['instruction_text'].strip()), e)
    return idx


def errs_along(X, yaw, A, ref, close_m):
    """X (T,2) 위치 · yaw (T,) rad · A (T,) 그 프레임에서 한 action · ref (M,2) -> [(action, err_deg)]."""
    out, wi = [], 1
    for t in range(len(X) - 1):
        while wi < len(ref) and np.linalg.norm(ref[wi] - X[t]) < close_m:
            wi += 1
        if wi >= len(ref):
            break
        dp = ref[wi] - X[t]
        e = (math.atan2(dp[1], dp[0]) - yaw[t] + math.pi) % (2 * math.pi) - math.pi
        out.append((int(A[t]), math.degrees(e)))
    return out


def replay_counts(ref, x0, yaw0, thr_deg, close_m, max_steps=2000):
    """eval_gt_collect 규칙 (문턱 thr) 운동학 재생 -> (회전 수, 전진 수)."""
    pos, yaw, n_t, n_f = np.array(x0, float), float(yaw0), 0, 0
    for wp in ref[1:]:
        for _ in range(max_steps):
            dp = wp - pos
            if np.linalg.norm(dp) < close_m:
                break
            e = (math.atan2(dp[1], dp[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(e) > math.radians(thr_deg):
                yaw += math.radians(15) * (1 if e > 0 else -1)
                n_t += 1
            else:
                pos = pos + 0.25 * np.array([math.cos(yaw), math.sin(yaw)])
                n_f += 1
    return n_t, n_f


def best_split(fw, tn):
    xs = np.arange(0.5, 45.01, 0.5)
    err = [(np.mean(fw > x) * len(fw) + np.mean(tn <= x) * len(tn)) / (len(fw) + len(tn)) for x in xs]
    k = int(np.argmin(err))
    return float(xs[k]), float(err[k])


def summarize(rows):
    a = np.array([r[0] for r in rows])
    e = np.abs(np.array([r[1] for r in rows]))
    fw, tn = e[a == 1], e[(a == 2) | (a == 3)]
    x, mis = best_split(fw, tn)
    return {'n_forward': int(len(fw)), 'n_turn': int(len(tn)),
            'forward_abs_err_pct_50_90_95_99': np.percentile(fw, [50, 90, 95, 99]).round(2).tolist(),
            'turn_abs_err_pct_1_5_10_50': np.percentile(tn, [1, 5, 10, 50]).round(2).tolist(),
            'forward_gt_7.5_frac': float(np.mean(fw > 7.5)), 'turn_le_7.5_frac': float(np.mean(tn <= 7.5)),
            'best_threshold_deg': x, 'best_misclass': mis, 'S1_pass': 5.0 <= x <= 10.0}


def main():
    import pandas as pd
    args = parse()
    R = {}
    for fmt in ('vln_pe', 'vln_ce'):
        raw = raw_index(fmt)
        rows, s0, n_ep, n_miss, s2 = [], [], 0, 0, {}
        for sc in sorted(glob.glob(f'{D}/{fmt}/traj_data/r2r/*'))[:args.n_scenes]:
            scene = Path(sc).name
            meta = [json.loads(l) for l in open(f'{sc}/meta/episodes.jsonl')]
            for m in meta[:args.eps_per_scene]:
                e = next((raw.get((scene, t.strip())) for t in m['tasks'] if (scene, t.strip()) in raw), None)
                if e is None:
                    n_miss += 1
                    continue
                ref = np.array([hab_xy(p) for p in e['reference_path']])
                p = f"{sc}/data/chunk-000/episode_{m['episode_index']:06d}.parquet"
                if fmt == 'vln_pe':
                    d = pd.read_parquet(p, columns=['observation.action', 'observation.robot_position', 'observation.robot_yaw'])
                    X = np.stack(d['observation.robot_position'].to_numpy())[:, :2]
                    yaw = d['observation.robot_yaw'].to_numpy()
                    A = d['observation.action'].to_numpy()                       # A[t] = t 에서 한 action
                else:
                    d = pd.read_parquet(p, columns=['action', 'pose.125cm_0deg'])
                    P = np.stack([np.stack(x).astype(np.float64) for x in d['pose.125cm_0deg']])
                    q = e['start_rotation']                                      # habitat (x, y, z, w), y-up
                    y0 = 2 * math.atan2(q[1], q[3])                              # habitat yaw (+y 축 회전)
                    # 시작 기준 상대 -> world: 상대 프레임 x 축 = 시작 카메라 앞. habitat 앞 = −z → z-up yaw = y0 + 90°
                    th = y0 + math.pi / 2
                    c, s = math.cos(th), math.sin(th)
                    Rz = np.array([[c, -s], [s, c]])
                    fw = P[:, :3, 2]
                    X = (Rz @ P[:, :2, 3].T).T + ref[0]
                    yaw = np.arctan2(fw[:, 1], fw[:, 0]) + th
                    a = d['action'].to_numpy()
                    A = np.r_[a[1:], 0]                                          # vln_ce a[t] = t-1 -> t 로 온 action -> t 에서 한 action
                s0.append(float(np.linalg.norm(X[0] - ref[0])))
                rel_t, rel_f = int(np.isin(A, (2, 3)).sum()), int((A == 1).sum())
                for thr in (7.5, 15.0, 22.5, 30.0):
                    t_, f_ = replay_counts(ref, X[0], yaw[0], thr, args.close_m)
                    s2.setdefault(thr, []).append((t_ - rel_t, f_ - rel_f))
                rows += errs_along(X, yaw, A, ref, args.close_m)
                n_ep += 1
        S = summarize(rows)
        S.update(episodes=n_ep, unmatched=n_miss, S0_start_dist_median=float(np.median(s0)), S0_pass=float(np.median(s0)) <= 0.3,
                 path_end_dist_note='vln_ce 는 시작 기준 상대 pose 를 raw start 로 world 복원')
        S['S2'] = {str(k): {'d_turn_abs_median': float(np.median(np.abs([a for a, _ in v]))), 'd_turn_median': float(np.median([a for a, _ in v])),
                            'd_fwd_abs_median': float(np.median(np.abs([b for _, b in v])))} for k, v in s2.items()}
        S['S2_best'] = min(S['S2'], key=lambda k: S['S2'][k]['d_turn_abs_median'])
        R[fmt] = S
        print(fmt, json.dumps(S, ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(R, indent=1))
    print('->', args.out)


if __name__ == '__main__':
    main()
