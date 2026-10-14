"""P1-V: 생성한 경로가 GT waypoint(reference_path)를 얼마나 따르는지 split 별로 잰다 (렌더 없음).

val 에는 릴리스 이미지·pose 가 없어 "릴리스와 비교"는 할 수 없다. 대신 모든 split 에 공통으로 있는
reference_path(GT waypoint) 를 자로 삼아 같은 지표로 잰다:
  train: 릴리스 경로(release) vs 우리 생성기(ours)   -> 두 경로가 GT waypoint 를 얼마나 따르는지 기준선
  val  : 우리 생성기(ours)                           -> train 의 ours 와 분포가 같으면 val 에서도 같은 품질

지표 (모두 수평면 x,z, m)
  dev_mean / dev_max : 경로의 각 점에서 reference_path 선분까지 거리의 평균 / 최대
  wp_miss_max        : 각 GT waypoint 에서 경로까지 최단거리 중 최대 (waypoint 를 지나쳤나)
  end_to_goal        : 끝점 ~ goal 거리 (R2R 성공 반경 3 m)
  frames, follower_errors

    /usr/bin/python scripts/dataset_converters/instruction_labeling/render/p1v_path_stats.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --split val_unseen --gpu 2 --out logs/instruction_labeling/p1v
"""

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_gen import generate_actions  # noqa: E402
from habitat_rig_sim import ACTION_NAMES, RigSim  # noqa: E402
from release_io import load_cfg, match_raw, release_episode_data, release_episodes, scene_glb  # noqa: E402


def seg_dist(P: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """P (N,2) 각 점에서 polyline ref (M,2) 까지 최단거리."""
    A, B = ref[:-1], ref[1:]
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(-1), 1e-12)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(-1) / L2[None], 0, 1)
    proj = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(P[:, None, :] - proj, axis=-1).min(1)


def stats(P: np.ndarray, ep: dict) -> dict:
    ref = np.asarray(ep['reference_path'])[:, [0, 2]]
    Pxz = P[:, [0, 2]]
    d = seg_dist(Pxz, ref)
    wp = np.linalg.norm(ref[:, None, :] - Pxz[None], axis=-1).min(1)
    return {'dev_mean': float(d.mean()), 'dev_max': float(d.max()), 'wp_miss_max': float(wp.max()),
            'end_to_goal': float(np.linalg.norm(P[-1] - np.asarray(ep['goals'][0]['position']))),
            'frames': len(P)}


def walk(rs, ep, actions) -> np.ndarray:
    rs.set_start(ep['start_position'], ep['start_rotation'])
    P = [rs.agent_position()]
    for a in actions[1:]:
        rs.agent.act(ACTION_NAMES[int(a)])
        P.append(rs.agent_position())
    return np.array(P)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', required=True, help='train 이면 릴리스 경로도 함께 잰다')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    cfg['camera']['rigs'] = {next(iter(cfg['camera']['rigs'])): 0.0}
    ag = cfg['action_gen']
    raw = json.load(gzip.open(cfg['paths']['raw_root'] / args.split / f'{args.split}.json.gz'))['episodes']
    by_scene = defaultdict(dict)
    for e in raw:
        by_scene[Path(e['scene_id']).stem].setdefault(e['trajectory_id'], e)
    if args.split == 'train':  # 릴리스에 있는 trajectory 만 (mini)
        rel_scenes = {p.name for p in cfg['paths']['release_root'].iterdir()}
        by_scene = {s: t for s, t in by_scene.items() if s in rel_scenes}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for si, (scene, trajs) in enumerate(sorted(by_scene.items())):
        rs = RigSim(cfg, scene_glb(cfg, scene), gpu=args.gpu)
        rel_actions = {}
        if args.split == 'train':
            for ep in release_episodes(cfg, scene):
                r = match_raw(cfg, scene, ep['task'])
                if r is not None and r['trajectory_id'] not in rel_actions:
                    rel_actions[r['trajectory_id']] = [int(a) for a in
                                                       release_episode_data(cfg, scene, ep['episode_index'], [])['action']]
        for tid, ep in trajs.items():
            if args.split == 'train' and tid not in rel_actions:
                continue
            g = generate_actions(rs, ep, ag['method'], float(ag['goal_radius']), int(ag['max_steps']))
            rows.append({'scene': scene, 'trajectory_id': tid, 'source': 'ours', 'follower_errors': g['follower_errors'],
                         **stats(walk(rs, ep, g['actions']), ep)})
            if tid in rel_actions:
                rows.append({'scene': scene, 'trajectory_id': tid, 'source': 'release', 'follower_errors': 0,
                             **stats(walk(rs, ep, rel_actions[tid]), ep)})
        rs.close()
        print(f'[{si + 1}/{len(by_scene)}] {scene} {len(trajs)} traj {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f'{args.split}_per_trajectory.csv', index=False)
    q = lambda p: (lambda x: x.quantile(p))  # noqa: E731
    agg = df.groupby('source').agg(
        n=('frames', 'size'), dev_mean=('dev_mean', 'mean'), dev_mean_p50=('dev_mean', 'median'),
        dev_max_p95=('dev_max', q(0.95)), wp_miss_max_p50=('wp_miss_max', 'median'), wp_miss_max_p95=('wp_miss_max', q(0.95)),
        end_to_goal_p50=('end_to_goal', 'median'), end_to_goal_p95=('end_to_goal', q(0.95)),
        within_3m=('end_to_goal', lambda x: (x < 3.0).mean()), frames_mean=('frames', 'mean'),
        dev_max_gt1m=('dev_max', lambda x: (x > 1.0).mean()), follower_errors=('follower_errors', 'sum'))
    agg.to_csv(out / f'{args.split}_summary.csv')
    print(agg.round(3).T.to_string())


if __name__ == '__main__':
    main()
