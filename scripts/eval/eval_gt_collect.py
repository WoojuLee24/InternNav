"""
GT episode data collection for VLN-PE.

GT reference_path를 move_by_flash로 재현하고, NE/success를 측정한다.
success 에피소드만 rgb/depth/pose를 저장하며, 재현 성공 여부로 구현을 검증한다.

Usage:
    python scripts/eval/eval_gt_collect.py \
        --config scripts/eval/configs/h1_internvla_n1_async_cfg.py \
        --ne_threshold 3.0 \
        --save_dir logs/gt_collect
"""
import sys

sys.path.append('.')
sys.path.append('./third_party/diffusion-policy')

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np


def load_eval_cfg(config_path):
    spec = importlib.util.spec_from_file_location('eval_config_module', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


def step_until_finish(env, action, robot_name):
    """finish_action=True 또는 terminated=True 까지 env.step을 반복한다."""
    while True:
        obs_list, _, terminated, _, _ = env.step(action)
        obs = obs_list[0][robot_name]
        if obs['finish_action'] or terminated[0]:
            return obs, terminated[0]


def get_yaw(globalrotation):
    """Isaac Sim quaternion [w,x,y,z] 에서 yaw(rad)를 추출한다."""
    from omni.isaac.core.utils.rotations import quat_to_euler_angles
    _, _, yaw = quat_to_euler_angles(np.array(globalrotation))
    return yaw


def navigate_to_waypoint(env, robot_name, target_wp, obs):
    """
    현재 obs에서 target_wp (3D)로 이동하기 위한 discrete flash 액션 시퀀스를 실행한다.
    move_by_flash: 1=forward 0.25m, 2=left 15°, 3=right 15°
    """
    FORWARD_DIST = 0.25       # meters per forward step
    TURN_ANGLE_DEG = 15.0     # degrees per turn step
    TURN_ANGLE_RAD = math.radians(TURN_ANGLE_DEG)
    CLOSE_ENOUGH = FORWARD_DIST * 0.6   # 0.15m
    MAX_STEPS = 400

    terminated = False
    for _ in range(MAX_STEPS):
        curr_pos = np.array(obs['globalgps'])[:2]
        tgt_pos = np.array(target_wp)[:2]
        dist = np.linalg.norm(tgt_pos - curr_pos)

        if dist < CLOSE_ENOUGH:
            break

        # 목표 방향 계산
        dp = tgt_pos - curr_pos
        target_yaw = math.atan2(dp[1], dp[0])
        curr_yaw = get_yaw(obs['globalrotation'])

        # heading 오차 [-pi, pi]
        delta = target_yaw - curr_yaw
        delta = ((delta + math.pi) % (2 * math.pi)) - math.pi

        if abs(delta) > TURN_ANGLE_RAD * 0.5:
            action_idx = 2 if delta > 0 else 3   # left or right
        else:
            action_idx = 1  # forward

        obs, terminated = step_until_finish(env, [{robot_name: {'move_by_flash': [action_idx]}}], robot_name)
        if terminated:
            break

    return obs, terminated


def save_episode(save_dir, path_key, episode_data, step_obs_list):
    ep_dir = Path(save_dir) / path_key.replace('/', '_')
    ep_dir.mkdir(parents=True, exist_ok=True)

    with open(ep_dir / 'metadata.json', 'w') as f:
        json.dump(
            {
                'path_key': path_key,
                'instruction': episode_data.get('instruction', {}).get('instruction_text', ''),
                'reference_path': episode_data['reference_path'],
                'start_position': episode_data['start_position'],
                'goals': episode_data['goals'],
            },
            f,
            indent=2,
            default=str,
        )

    rgbs = [s['rgb'] for s in step_obs_list if s.get('rgb') is not None]
    depths = [s['depth'] for s in step_obs_list if s.get('depth') is not None]
    poses = [s['globalgps'] for s in step_obs_list if s.get('globalgps') is not None]

    if rgbs:
        np.save(ep_dir / 'rgb.npy', np.array(rgbs))
    if depths:
        np.save(ep_dir / 'depth.npy', np.array(depths))
    if poses:
        np.save(ep_dir / 'poses.npy', np.array(poses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/eval/configs/h1_internvla_n1_async_cfg.py')
    parser.add_argument('--task_name', default=None, help='task name override (default: gt_<config task_name>)')
    parser.add_argument('--ne_threshold', type=float, default=3.0, help='NE filter threshold (m)')
    parser.add_argument('--save_dir', default='logs/gt_collect')
    parser.add_argument('--no_save', action='store_true', help='metric 측정만 하고 obs 저장 생략')
    args = parser.parse_args()

    # 1. config 로드 + get_config() 적용 (eval.py 와 동일한 경로)
    eval_cfg = load_eval_cfg(args.config)
    from internnav.configs.evaluator.vln_default_config import get_config

    if args.task_name:
        eval_cfg.task.task_name = args.task_name
    else:
        eval_cfg.task.task_name = f'gt_{eval_cfg.task.task_name}'

    eval_cfg = get_config(eval_cfg)

    # 2. env_settings 보완 (DistributedEvaluator + VLNDistributedEvaluator 가 하는 부분)
    eval_cfg.env.env_settings['rank'] = 0
    eval_cfg.env.env_settings['local_rank'] = 0
    eval_cfg.env.env_settings['world_size'] = 1
    eval_cfg.env.env_settings['dataset'] = eval_cfg.dataset

    robot_name = eval_cfg.task.robot_name  # 'h1'

    # 3. env 초기화 (agent 없음)
    from internnav.env import Env

    env = Env.init(eval_cfg.env, eval_cfg.task)

    os.makedirs(args.save_dir, exist_ok=True)
    results = []

    # 4. 첫 번째 에피소드 reset
    obs_list, reset_info_list = env.reset()

    while True:
        info = reset_info_list[0] if reset_info_list else None
        if info is None:
            break

        ref_path = info.data['reference_path']  # robot_offset 이미 적용됨
        path_key = info.data['path_key']
        instruction = info.data['instruction']['instruction_text']
        print(f'\n[{path_key}] waypoints={len(ref_path)} | {instruction[:60]}...')

        # 5. warm up: stand_still 완료까지 대기
        obs, _ = step_until_finish(env, [{robot_name: {'stand_still': []}}], robot_name)

        # 6. reference_path 순서대로 navigate (discrete flash: forward/turn)
        # ref_path[0] = start position → skip
        step_obs_list = []
        terminated = False
        for wp in ref_path[1:]:
            obs, terminated = navigate_to_waypoint(env, robot_name, wp, obs)
            step_obs_list.append({
                'globalgps': np.array(obs['globalgps']).tolist() if 'globalgps' in obs else None,
                'rgb': obs.get('rgb'),
                'depth': obs.get('depth'),
            })
            if terminated:
                break

        # 7. stop 전송 → metric 계산 트리거
        if not terminated:
            obs, terminated = step_until_finish(env, [{robot_name: {'stop': []}}], robot_name)

        # 8. metric 추출 (VLNPEMetrics가 계산한 값 재사용)
        metrics_raw = obs.get('metrics', {})
        if metrics_raw:
            m = metrics_raw[list(metrics_raw.keys())[0]][0]
            ne = float(m.get('NE', float('inf')))
            success = bool(m.get('success', 0))
            fail_reason = m.get('fail_reason', '')
        else:
            # fallback: globalgps로 직접 계산
            final_pos = np.array(obs.get('globalgps', ref_path[-1]))
            ne = float(np.linalg.norm(final_pos[:2] - np.array(ref_path[-1])[:2]))
            success = ne < args.ne_threshold
            fail_reason = 'no_metrics'

        print(f'  NE={ne:.2f}m  success={success}  steps={len(step_obs_list)}  reason={fail_reason}')
        results.append({'path_key': path_key, 'ne': ne, 'success': success, 'fail_reason': fail_reason})

        # 9. success 에피소드 저장
        if success and not args.no_save:
            save_episode(args.save_dir, path_key, info.data, step_obs_list)

        # 10. 다음 에피소드로 reset
        obs_list, reset_info_list = env.reset([0])

    env.close()

    # 결과 요약
    total = len(results)
    n_success = sum(r['success'] for r in results)
    avg_ne = float(np.mean([r['ne'] for r in results])) if results else 0.0
    print(f'\n=== Summary ===')
    print(f'Total: {total}  SR: {n_success}/{total} ({n_success/total:.1%})  Avg NE: {avg_ne:.2f}m')

    result_path = Path(args.save_dir) / 'results.json'
    with open(result_path, 'w') as f:
        json.dump(
            {'summary': {'total': total, 'n_success': n_success, 'avg_ne': avg_ne}, 'episodes': results},
            f,
            indent=2,
        )
    print(f'Results saved to {result_path}')


if __name__ == '__main__':
    main()
