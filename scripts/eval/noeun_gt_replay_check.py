"""노은역 GT reference_path 를 Isaac H1 flash(바닥 따라가기)로 재생해 habitat 과 이동이 같은지 확인한다.

GT reference_path = habitat navmesh 로 만든 경로. Isaac 에서 같은 경로를 같은 이산 동작(0.25 m 전진 · 15 deg 회전)으로
따라갔을 때 도착하는지, 어디서 막히는지, 높이가 navmesh 와 같은지를 에피소드별로 남긴다.
waypoint 추종 규칙은 scripts/visualization/eval_gt_collect.py navigate_to_waypoint 와 같다.

    CUDA_VISIBLE_DEVICES=0 NOEUN_DATA=<raw_data/r2r> /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/noeun_gt_replay_check.py --config scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py --out logs/noeun_gt_replay_v6

출력: <out>/episodes.jsonl (에피소드당 1 줄), <out>/steps/<path_key>.jsonl (step 별 위치 · 동작 · 막힘)
"""
import sys

sys.path.append('.')

import argparse
import importlib.util
import json
import math
import os

import numpy as np

FORWARD_DIST = 0.25
TURN_ANGLE_RAD = math.radians(15.0)
CLOSE_ENOUGH = FORWARD_DIST * 0.6
MAX_STEPS_PER_WP = 400


def load_eval_cfg(config_path):
    spec = importlib.util.spec_from_file_location('eval_config_module', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


def step_until_finish(env, action, robot_name, max_iter=3000):
    for i in range(max_iter):
        obs_list, _, terminated, _, _ = env.step(action)
        obs = obs_list[0][robot_name]
        if obs['finish_action'] or terminated[0]:
            return obs, terminated[0]
        if i % 500 == 499:
            print(f'[GT REPLAY] waiting finish_action: action={action} iter={i + 1} '
                  f'pos={np.round(np.array(obs.get("globalgps", [0, 0, 0])), 3).tolist()} keys={sorted(obs)[:12]}', flush=True)
    raise RuntimeError(f'finish_action not reached in {max_iter} steps: action={action}')


def get_yaw(quat):
    from omni.isaac.core.utils.rotations import quat_to_euler_angles

    return float(quat_to_euler_angles(np.array(quat))[2])


def block_reason(env):
    """flash floor 컨트롤러의 마지막 막힘 이유 ('' = 안 막힘)."""
    for task in env.env.runner.current_tasks.values():
        for robot in task.robots.values():
            ctrl = robot.controllers.get('move_by_flash')
            if ctrl is not None:
                return getattr(ctrl, 'last_block_reason', '')
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py')
    ap.add_argument('--out', default='logs/noeun_gt_replay_v6')
    ap.add_argument('--max_episodes', type=int, default=0, help='0 = 전부')
    ap.add_argument('--print_every', type=int, default=50, help='이 step 마다 위치 · 동작 출력')
    ap.add_argument('--stack_dump_s', type=int, default=0, help='>0 이면 이 초마다 Python 스택을 stderr 로 (멈춤 진단)')
    ap.add_argument('--warmup_renders', type=int, default=4, help='첫 step 전 렌더 step 수 (pointcloud annotator 가 비어 있으면 죽는다)')
    args = ap.parse_args()
    if args.stack_dump_s > 0:
        import faulthandler

        faulthandler.dump_traceback_later(args.stack_dump_s, repeat=True)

    eval_cfg = load_eval_cfg(args.config)
    from internnav.configs.evaluator.vln_default_config import get_config

    eval_cfg = get_config(eval_cfg)
    eval_cfg.env.env_settings.update({'rank': 0, 'local_rank': 0, 'world_size': 1, 'dataset': eval_cfg.dataset})
    robot_name = eval_cfg.task.robot_name

    from internnav.env import Env

    env = Env.init(eval_cfg.env, eval_cfg.task)
    os.makedirs(f'{args.out}/steps', exist_ok=True)
    ep_f = open(f'{args.out}/episodes.jsonl', 'w')

    obs_list, reset_info_list = env.reset()
    env.env.warm_up(steps=args.warmup_renders, render=True, physics=True)

    n = 0
    while reset_info_list and reset_info_list[0] is not None:
        info = reset_info_list[0]
        if args.max_episodes and n >= args.max_episodes:
            break
        ref = np.array(info.data['reference_path'], dtype=float)
        key = info.data['path_key']
        obs, terminated = step_until_finish(env, [{robot_name: {'stand_still': []}}], robot_name)
        start_z = float(obs['globalgps'][2])

        rows, n_block, wp_reached = [], 0, 0
        for wp_idx, wp in enumerate(ref[1:], start=1):
            for _ in range(MAX_STEPS_PER_WP):
                cur = np.array(obs['globalgps'])
                if np.linalg.norm(wp[:2] - cur[:2]) < CLOSE_ENOUGH:
                    wp_reached = wp_idx
                    break
                dp = wp[:2] - cur[:2]
                delta = math.atan2(dp[1], dp[0]) - get_yaw(obs['globalrotation'])
                delta = ((delta + math.pi) % (2 * math.pi)) - math.pi
                a = (2 if delta > 0 else 3) if abs(delta) > TURN_ANGLE_RAD * 0.5 else 1
                obs, terminated = step_until_finish(env, [{robot_name: {'move_by_flash': [a]}}], robot_name)
                reason = block_reason(env) if a == 1 else ''
                n_block += bool(reason)
                p = np.array(obs['globalgps'])
                if len(rows) % args.print_every == 0:
                    print(f'[GT REPLAY] {key} step {len(rows)} wp {wp_idx}/{len(ref) - 1} a {a} pos {np.round(p, 2).tolist()} '
                          f'quat {np.round(np.array(obs["globalrotation"]), 3).tolist()} wp_xyz {np.round(wp, 2).tolist()} yaw {math.degrees(get_yaw(obs["globalrotation"])):.0f} block {reason!r}', flush=True)
                rows.append({'wp': wp_idx, 'a': a, 'pos': np.round(p, 3).tolist(), 'block': reason,
                             'wp_xyz': np.round(wp, 3).tolist()})
                if terminated:
                    break
            else:
                break  # 이 waypoint 에 MAX_STEPS 안에 못 감 -> 포기
            if terminated:
                break
        if not terminated:
            obs, terminated = step_until_finish(env, [{robot_name: {'stop': []}}], robot_name)

        m = {}
        if obs.get('metrics'):
            m = obs['metrics'][list(obs['metrics'].keys())[0]][0]
        final = np.array(obs['globalgps'])
        # 경로 이탈: 각 step 위치에서 reference_path 선분까지 최소 xy 거리
        dev = []
        for r in rows:
            q = np.array(r['pos'][:2])
            dmin = min(np.linalg.norm(q - (a0[:2] + np.clip(np.dot(q - a0[:2], b0[:2] - a0[:2]) / max(np.dot(b0[:2] - a0[:2], b0[:2] - a0[:2]), 1e-9), 0, 1) * (b0[:2] - a0[:2]))) for a0, b0 in zip(ref[:-1], ref[1:]))
            dev.append(dmin)
        rec = {
            'path_key': key, 'n_wp': len(ref) - 1, 'wp_reached': wp_reached, 'steps': len(rows), 'n_block': n_block,
            'success': bool(m.get('success', 0)), 'NE': m.get('NE'), 'fail_reason': m.get('fail_reason', ''),
            'final_xyz': np.round(final, 3).tolist(), 'goal_xyz': np.round(ref[-1], 3).tolist(),
            'final_dz_base_minus_goal': round(float(final[2] - ref[-1][2]), 3), 'start_base_z': round(start_z, 3),
            'max_dev_xy': round(float(max(dev)) if dev else 0.0, 3), 'mean_dev_xy': round(float(np.mean(dev)) if dev else 0.0, 3),
        }
        print('[GT REPLAY]', json.dumps(rec), flush=True)
        ep_f.write(json.dumps(rec) + '\n')
        ep_f.flush()
        with open(f'{args.out}/steps/{key}.jsonl', 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        n += 1
        obs_list, reset_info_list = env.reset([0])
    ep_f.close()
    env.close()


if __name__ == '__main__':
    main()
