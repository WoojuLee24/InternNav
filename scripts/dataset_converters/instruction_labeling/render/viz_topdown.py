"""경로 top-down 시각화 — navmesh 평면도 위에 reference_path, 실제로 걸은 경로, 카메라 방향(화살표)을 그린다.

heading 부호 버그(vln-s2-pipeline 의 거울상 방향) 같은 문제를 눈으로 잡기 위한 그림이다.
화살표 = 그 프레임에서 카메라가 보는 방향. 경로가 화살표 방향으로 진행해야 정상이다.

모드
  --split val_unseen : render_split 캐시(meta.json 의 positions/actions)를 그린다. 우리 경로(파랑).
  --split train      : 릴리스 action 재생 경로(초록)와 action 생성기 경로(파랑)를 겹쳐 그린다 (P1-A 비교).

    /usr/bin/python scripts/dataset_converters/instruction_labeling/render/viz_topdown.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --split val_unseen --n_per_scene 2 --out logs/instruction_labeling/topdown
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_io import load_cfg, list_release_scenes, match_raw, release_episode_data, release_episodes, scene_glb  # noqa: E402


def yaw_series(start_rotation_xyzw, actions, turn_deg):
    """habitat 평면 (x, z) 에서의 전방 벡터 각도 [rad] 시퀀스. 전방 = R @ (0,0,-1)."""
    x, y, z, w = start_rotation_xyzw
    yaw = 2.0 * np.arctan2(y, w)  # y축 회전각
    out = []
    for i, a in enumerate(actions):
        if i > 0 and a == 2:
            yaw += np.radians(turn_deg)
        elif i > 0 and a == 3:
            yaw -= np.radians(turn_deg)
        out.append(yaw)
    return np.array(out)


def fwd_xz(yaw):
    return np.stack([-np.sin(yaw), -np.cos(yaw)], -1)  # R_y(yaw) @ (0,0,-1) 의 (x, z)


def draw(ax, pf, height, ref, paths, title, mpp=0.05):
    import matplotlib.pyplot as plt  # noqa: F401
    lo, hi = pf.get_bounds()
    td = np.asarray(pf.get_topdown_view(mpp, height))
    ax.imshow(td, cmap='gray', extent=[lo[0], lo[0] + td.shape[1] * mpp, lo[2] + td.shape[0] * mpp, lo[2]],
              alpha=0.5)
    ref = np.asarray(ref)
    ax.plot(ref[:, 0], ref[:, 2], 'r.-', lw=2, ms=10, label='reference_path')
    for (P, yaw, color, label) in paths:
        P = np.asarray(P)
        ax.plot(P[:, 0], P[:, 2], '-', color=color, lw=1.2, label=label)
        k = max(1, len(P) // 15)
        F = fwd_xz(yaw)
        ax.quiver(P[::k, 0], P[::k, 2], 0.35 * F[::k, 0], 0.35 * F[::k, 1], color=color, angles='xy', scale_units='xy', scale=1, width=0.005)  # angles='xy': y축 반전을 따름
    ax.plot(*ref[0, [0, 2]], 'k*', ms=14)
    pad = 1.5
    allp = np.concatenate([ref[:, [0, 2]]] + [np.asarray(p)[:, [0, 2]] for p, *_ in paths])
    ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ax.set_ylim(allp[:, 1].max() + pad, allp[:, 1].min() - pad)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc='best')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--scenes', default='all')
    ap.add_argument('--n_per_scene', type=int, default=2)
    ap.add_argument('--traj', default='', help='쉼표 구분 trajectory_id 만 그림 (train 모드)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import habitat_sim

    cfg = load_cfg(args.config)
    turn = float(cfg['agent']['turn_deg'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.split == 'train':
        from action_gen import generate_actions
        from habitat_rig_sim import ACTION_NAMES, RigSim
        cfg['camera']['rigs'] = {next(iter(cfg['camera']['rigs'])): 0.0}
        ag = cfg['action_gen']
        want = {int(t) for t in args.traj.split(',') if t}
        scenes = list_release_scenes(cfg) if args.scenes == 'all' else args.scenes.split(',')
        for scene in scenes:
            rs = RigSim(cfg, scene_glb(cfg, scene), gpu=args.gpu)
            seen = []
            for ep in release_episodes(cfg, scene):
                raw = match_raw(cfg, scene, ep['task'])
                if raw is None or raw['trajectory_id'] in seen:
                    continue
                if want and raw['trajectory_id'] not in want:
                    continue
                seen.append(raw['trajectory_id'])
                if not want and len(seen) > args.n_per_scene:
                    break
                rel = [int(a) for a in release_episode_data(cfg, scene, ep['episode_index'], [])['action']]
                ours = generate_actions(rs, raw, ag['method'], float(ag['goal_radius']), int(ag['max_steps']))['actions']
                paths = []
                for acts, color, label in ((rel, 'g', f'release ({len(rel)} fr)'),
                                           (ours, 'b', f'{ag["method"]} ({len(ours)} fr)')):
                    rs.set_start(raw['start_position'], raw['start_rotation'])
                    P = [rs.agent_position()]
                    for a in acts[1:]:
                        rs.agent.act(ACTION_NAMES[a])
                        P.append(rs.agent_position())
                    paths.append((P, yaw_series(raw['start_rotation'], acts, turn), color, label))
                fig, ax = plt.subplots(figsize=(7, 7))
                draw(ax, rs.sim.pathfinder, raw['start_position'][1], raw['reference_path'], paths,
                     f'train {scene} traj {raw["trajectory_id"]}')
                fig.savefig(out / f'train_{scene}_traj{raw["trajectory_id"]}.png', dpi=110)
                plt.close(fig)
            rs.close()
            print(scene, 'done', flush=True)
        return

    raw = json.load(gzip.open(cfg['paths']['raw_root'] / args.split / f'{args.split}.json.gz'))['episodes']
    by_traj = {}
    for e in raw:
        by_traj.setdefault(e['trajectory_id'], e)
    cache = cfg['paths']['cache_root'] / args.split
    scenes = sorted(p.name for p in cache.iterdir()) if args.scenes == 'all' else args.scenes.split(',')
    for scene in scenes:
        pf = habitat_sim.PathFinder()
        pf.load_nav_mesh(str(scene_glb(cfg, scene).with_suffix('.navmesh')))
        tdirs = sorted(p for p in (cache / scene).iterdir() if (p / 'meta.json').is_file())[:args.n_per_scene]
        for tdir in tdirs:
            m = json.loads((tdir / 'meta.json').read_text())
            ep = by_traj[m['trajectory_id']]
            fig, ax = plt.subplots(figsize=(7, 7))
            draw(ax, pf, ep['start_position'][1], ep['reference_path'],
                 [(m['positions'], yaw_series(ep['start_rotation'], m['actions'], turn), 'b',
                   f'rendered ({m["frames"]} fr)')],
                 f'{args.split} {scene} traj {m["trajectory_id"]} end->goal {m["end_to_goal_m"]:.2f} m')
            fig.savefig(out / f'{args.split}_{scene}_traj{m["trajectory_id"]}.png', dpi=110)
            plt.close(fig)
        print(scene, len(tdirs), flush=True)


if __name__ == '__main__':
    main()
