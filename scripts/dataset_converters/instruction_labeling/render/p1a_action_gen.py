"""P1-A: action 생성기 후보가 릴리스 train action 을 얼마나 재현하는지 잰다 (렌더 없음).

trajectory 당 1회 (같은 trajectory 의 episode 는 action 이 같다 — 실측).
지표:
  exact      : action 시퀀스 완전일치
  edit/len   : 편집거리 / 릴리스 길이
  len_diff   : 프레임 수 차 (ours - release)
  path_mean  : 우리 경로 각 점에서 릴리스 경로까지 최단거리 평균 (m), 그 반대 방향과의 평균 (대칭 chamfer)
  path_max   : 대칭 최대 (Hausdorff, m)
  end_diff   : 끝점 차 (m)
  turn_frac  : 회전 action 비율 (ours, release)

    /usr/bin/python scripts/dataset_converters/instruction_labeling/render/p1a_action_gen.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --scenes all --out logs/instruction_labeling/p1a
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_gen import generate_actions  # noqa: E402
from habitat_rig_sim import ACTION_NAMES, RigSim  # noqa: E402
from release_io import (list_release_scenes, load_cfg, match_raw, release_episode_data,  # noqa: E402
                        release_episodes, scene_glb)

CANDIDATES = [  # (이름, method, goal_radius)
    ('greedy_wp_r0.25', 'greedy_wp', 0.25),
    ('greedy_wp_r0.5', 'greedy_wp', 0.5),
    ('greedy_goal_r0.25', 'greedy_goal', 0.25),
    ('greedy_dense0.5_r0.25', 'greedy_dense', 0.25),
]


def edit_distance(a, b) -> int:
    D = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        prev = D.copy()
        D[0] = i
        for j, y in enumerate(b, 1):
            D[j] = min(prev[j] + 1, D[j - 1] + 1, prev[j - 1] + (x != y))
    return int(D[-1])


def replay_positions(rs, ep, actions) -> np.ndarray:
    rs.set_start(ep['start_position'], ep['start_rotation'])
    P = [rs.agent_position()]
    for a in actions[1:]:
        rs.agent.act(ACTION_NAMES[int(a)])
        P.append(rs.agent_position())
    return np.array(P)


def path_dist(A: np.ndarray, B: np.ndarray):
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    ab, ba = d.min(1), d.min(0)
    return float((ab.mean() + ba.mean()) / 2), float(max(ab.max(), ba.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--scenes', default='all', help="'all' 또는 쉼표 구분")
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max_traj', type=int, default=0, help='scene 당 trajectory 상한 (0=전부)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cfg['camera']['rigs'] = {next(iter(cfg['camera']['rigs'])): 0.0}  # 렌더 안 함 -> 센서 1개면 충분
    scenes = list_release_scenes(cfg) if args.scenes == 'all' else args.scenes.split(',')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for si, scene in enumerate(scenes):
        rs = RigSim(cfg, scene_glb(cfg, scene), gpu=args.gpu)
        seen = set()
        for ep in release_episodes(cfg, scene):
            raw = match_raw(cfg, scene, ep['task'])
            if raw is None or raw['trajectory_id'] in seen:
                continue
            seen.add(raw['trajectory_id'])
            if args.max_traj and len(seen) > args.max_traj:
                break
            rel = [int(a) for a in release_episode_data(cfg, scene, ep['episode_index'], [])['action']]
            P_rel = replay_positions(rs, raw, rel)
            for name, method, r in CANDIDATES:
                g = generate_actions(rs, raw, method, r)
                ours = g['actions']
                P = replay_positions(rs, raw, ours)
                pm, px = path_dist(P[:, [0, 2]], P_rel[:, [0, 2]])
                rows.append({
                    'scene': scene, 'trajectory_id': raw['trajectory_id'], 'candidate': name,
                    'exact': ours == rel, 'edit_per_len': edit_distance(ours, rel) / len(rel),
                    'len_diff': len(ours) - len(rel), 'path_mean_m': pm, 'path_max_m': px,
                    'end_diff_m': float(np.linalg.norm(P[-1] - P_rel[-1])),
                    'turn_frac_ours': float(np.mean(np.isin(ours[1:], [2, 3]))) if len(ours) > 1 else 0.0,
                    'turn_frac_rel': float(np.mean(np.isin(rel[1:], [2, 3]))),
                    'follower_errors': g['follower_errors'], 'truncated': g['truncated'],
                })
        rs.close()
        print(f'[{si + 1}/{len(scenes)}] {scene}: {len(seen)} traj, {time.time() - t0:.0f}s', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / 'per_trajectory.csv', index=False)
    agg = df.groupby('candidate').agg(
        n=('exact', 'size'), exact_rate=('exact', 'mean'), edit_per_len=('edit_per_len', 'mean'),
        abs_len_diff=('len_diff', lambda x: x.abs().mean()), path_mean_m=('path_mean_m', 'mean'),
        path_max_p95_m=('path_max_m', lambda x: x.quantile(0.95)), end_diff_p95_m=('end_diff_m', lambda x: x.quantile(0.95)),
        turn_frac_ours=('turn_frac_ours', 'mean'), turn_frac_rel=('turn_frac_rel', 'mean'),
        follower_errors=('follower_errors', 'sum'), truncated=('truncated', 'sum'),
    ).sort_values('edit_per_len')
    agg.to_csv(out / 'summary.csv')
    (out / 'summary.json').write_text(json.dumps({'args': vars(args), 'elapsed_s': round(time.time() - t0, 1),
                                                  'summary': agg.reset_index().to_dict('records')}, indent=2))
    print(agg.to_string())


if __name__ == '__main__':
    main()
