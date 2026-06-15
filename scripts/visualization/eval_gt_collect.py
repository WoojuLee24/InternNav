"""
GT episode data collection for VLN-PE.

GT reference_path를 move_by_flash로 재현하고, NE/success를 측정한다.
success 에피소드는 vln_pe/traj_data/r2r 포맷과 동일하게 r2r_h1에 저장한다.

Usage:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/eval_gt_collect.py \
        --config scripts/eval/configs/h1_internvla_n1_async_cfg_orig.py \
        --save_dir data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r_h1
"""
import sys

sys.path.append('.')
sys.path.append('./third_party/diffusion-policy')

import argparse
import importlib.util
import json
import math
import os
from collections import defaultdict
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


def get_yaw(quat):
    """Isaac Sim quaternion [w,x,y,z] → yaw(rad)."""
    from omni.isaac.core.utils.rotations import quat_to_euler_angles
    _, _, yaw = quat_to_euler_angles(np.array(quat))
    return float(yaw)


def navigate_to_waypoint(env, robot_name, target_wp, obs, step_buffer=None, wp_idx=0, total_wps=1):
    """
    target_wp(3D)까지 discrete flash 액션으로 이동한다.
    step_buffer가 주어지면 각 flash 스텝의 데이터를 수집한다.
    """
    FORWARD_DIST = 0.25
    TURN_ANGLE_RAD = math.radians(15.0)
    CLOSE_ENOUGH = FORWARD_DIST * 0.6
    MAX_STEPS = 400

    progress = wp_idx / max(total_wps, 1)
    terminated = False

    for _ in range(MAX_STEPS):
        curr_pos = np.array(obs['globalgps'])[:2]
        tgt_pos = np.array(target_wp)[:2]
        if np.linalg.norm(tgt_pos - curr_pos) < CLOSE_ENOUGH:
            break

        dp = tgt_pos - curr_pos
        delta = math.atan2(dp[1], dp[0]) - get_yaw(obs['globalrotation'])
        delta = ((delta + math.pi) % (2 * math.pi)) - math.pi
        action_idx = (2 if delta > 0 else 3) if abs(delta) > TURN_ANGLE_RAD * 0.5 else 1

        obs, terminated = step_until_finish(env, [{robot_name: {'move_by_flash': [action_idx]}}], robot_name)

        if step_buffer is not None:
            r_pos = np.array(obs['globalgps']).copy()
            r_ori = np.array(obs['globalrotation']).copy()
            c_pos = np.array(obs.get('camera_position', r_pos)).copy()
            c_ori = np.array(obs.get('camera_orientation', r_ori)).copy()
            rgb   = obs.get('rgb')
            depth = obs.get('depth')
            step_buffer.append({
                'camera_position':    c_pos,
                'camera_orientation': c_ori,
                'camera_yaw':         get_yaw(c_ori),
                'robot_position':     r_pos,
                'robot_orientation':  r_ori,
                'robot_yaw':          get_yaw(r_ori),
                'action':             action_idx,
                'progress':           progress,
                'rgb':   rgb.copy()   if rgb   is not None else None,
                'depth': depth.copy() if depth is not None else None,
            })

        if terminated:
            break

    return obs, terminated


# ---------------------------------------------------------------------------
# vln_pe/traj_data/r2r 포맷 저장
# ---------------------------------------------------------------------------

def _stats(vals):
    """episodes_stats.jsonl용 min/max/mean/std 딕셔너리."""
    arr = np.array(vals)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    return {
        'min':   [arr.min(axis=0).tolist()],
        'max':   [arr.max(axis=0).tolist()],
        'mean':  [arr.mean(axis=0).tolist()],
        'std':   [arr.std(axis=0).tolist()],
        'count': [int(len(arr))],
    }


def save_episode(save_dir, scan, episode_index, task_index, episode_data,
                 step_buffer, success, fail_reason, global_frame_offset):
    """
    step_buffer를 vln_pe/traj_data/r2r 와 동일한 포맷으로 저장한다.

    저장 경로: <save_dir>/<scan>/
        data/chunk-000/episode_XXXXXX.parquet
        videos/chunk-000/observation.images.rgb/episode_XXXXXX.npy   (T,256,256,3) uint8
        videos/chunk-000/observation.images.depth/episode_XXXXXX.npy (T,256,256)   float32
        meta/episodes.jsonl, tasks.jsonl, episodes_stats.jsonl
    """
    import cv2
    import pyarrow as pa
    import pyarrow.parquet as pq

    scene_dir = Path(save_dir) / scan
    data_dir  = scene_dir / 'data'   / 'chunk-000'
    rgb_dir   = scene_dir / 'videos' / 'chunk-000' / 'observation.images.rgb'
    depth_dir = scene_dir / 'videos' / 'chunk-000' / 'observation.images.depth'
    meta_dir  = scene_dir / 'meta'
    for d in [data_dir, rgb_dir, depth_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ep_fname = f'episode_{episode_index:06d}'
    T = len(step_buffer)

    cam_pos, cam_ori, cam_yaw = [], [], []
    rob_pos, rob_ori, rob_yaw = [], [], []
    actions, progresses = [], []
    timestamps, frame_idxs, ep_idxs, global_idxs, task_idxs = [], [], [], [], []
    rgbs, depths = [], []

    for fi, step in enumerate(step_buffer):
        cam_pos.append(step['camera_position'].tolist())
        cam_ori.append(step['camera_orientation'].tolist())
        cam_yaw.append(step['camera_yaw'])
        rob_pos.append(step['robot_position'].tolist())
        rob_ori.append(step['robot_orientation'].tolist())
        rob_yaw.append(step['robot_yaw'])
        actions.append(int(step['action']))
        progresses.append(float(step['progress']))
        timestamps.append(float(fi) / 30.0)
        frame_idxs.append(fi)
        ep_idxs.append(episode_index)
        global_idxs.append(global_frame_offset + fi)
        task_idxs.append(task_index)

        rgb = step['rgb']
        if rgb is not None:
            if rgb.shape[:2] != (256, 256):
                rgb = cv2.resize(rgb, (256, 256))
            rgbs.append(rgb.astype(np.uint8))

        depth = step['depth']
        if depth is not None:
            if depth.ndim == 3:
                depth = depth[..., 0]
            if depth.shape != (256, 256):
                depth = cv2.resize(depth, (256, 256))
            depths.append(depth.astype(np.float32))

    # parquet
    table = pa.table({
        'observation.camera_position':    pa.array(cam_pos,    type=pa.list_(pa.float64(), 3)),
        'observation.camera_orientation': pa.array(cam_ori,    type=pa.list_(pa.float64(), 4)),
        'observation.camera_yaw':         pa.array(cam_yaw,    type=pa.float64()),
        'observation.robot_position':     pa.array(rob_pos,    type=pa.list_(pa.float64(), 3)),
        'observation.robot_orientation':  pa.array(rob_ori,    type=pa.list_(pa.float64(), 4)),
        'observation.robot_yaw':          pa.array(rob_yaw,    type=pa.float64()),
        'observation.progress':           pa.array(progresses, type=pa.float64()),
        'observation.step':               pa.array(frame_idxs, type=pa.int64()),
        'observation.action':             pa.array(actions,    type=pa.int64()),
        'timestamp':                      pa.array(timestamps, type=pa.float32()),
        'frame_index':                    pa.array(frame_idxs, type=pa.int64()),
        'episode_index':                  pa.array(ep_idxs,    type=pa.int64()),
        'index':                          pa.array(global_idxs, type=pa.int64()),
        'task_index':                     pa.array(task_idxs,  type=pa.int64()),
    })
    pq.write_table(table, data_dir / f'{ep_fname}.parquet')

    # 이미지 npy
    if rgbs:
        np.save(rgb_dir   / f'{ep_fname}.npy', np.array(rgbs,   dtype=np.uint8))
    if depths:
        np.save(depth_dir / f'{ep_fname}.npy', np.array(depths, dtype=np.float32))

    # meta/episodes.jsonl
    with open(meta_dir / 'episodes.jsonl', 'a') as f:
        f.write(json.dumps({
            'episode_index': episode_index,
            'tasks': [episode_data['instruction']['instruction_text']],
        }) + '\n')

    # meta/tasks.jsonl
    with open(meta_dir / 'tasks.jsonl', 'a') as f:
        f.write(json.dumps({
            'task_index':         task_index,
            'task':               episode_data['instruction']['instruction_text'],
            'instruction_tokens': episode_data['instruction'].get('instruction_tokens', []),
            'finish_status':      'success' if success else 'fail',
            'fail_reason':        'success' if success else fail_reason,
        }) + '\n')

    # meta/episodes_stats.jsonl
    with open(meta_dir / 'episodes_stats.jsonl', 'a') as f:
        f.write(json.dumps({
            'episode_index': episode_index,
            'stats': {
                'observation.camera_position':    _stats(cam_pos),
                'observation.camera_orientation': _stats(cam_ori),
                'observation.camera_yaw':         _stats(cam_yaw),
                'observation.robot_position':     _stats(rob_pos),
                'observation.robot_orientation':  _stats(rob_ori),
                'observation.robot_yaw':          _stats(rob_yaw),
                'observation.progress':           _stats(progresses),
                'observation.action':             _stats(actions),
            },
            'task_index': {'min': task_index, 'max': task_index, 'count': T},
        }) + '\n')

    return T


def finalize_info_json(save_dir, scene_states):
    """각 scene의 meta/info.json 작성 (r2r/info.json 포맷과 동일)."""
    features = {
        'observation.camera_position':    {'dtype': 'float64', 'shape': [3], 'names': None},
        'observation.camera_orientation': {'dtype': 'float64', 'shape': [4], 'names': None},
        'observation.camera_yaw':         {'dtype': 'float64', 'shape': [1], 'names': None},
        'observation.robot_position':     {'dtype': 'float64', 'shape': [3], 'names': None},
        'observation.robot_orientation':  {'dtype': 'float64', 'shape': [4], 'names': None},
        'observation.robot_yaw':          {'dtype': 'float64', 'shape': [1], 'names': None},
        'observation.progress':           {'dtype': 'float64', 'shape': [1], 'names': None},
        'observation.step':               {'dtype': 'int64',   'shape': [1], 'names': None},
        'observation.action':             {'dtype': 'int64',   'shape': [1], 'names': None},
        'timestamp':                      {'dtype': 'float32', 'shape': [1], 'names': None},
        'frame_index':                    {'dtype': 'int64',   'shape': [1], 'names': None},
        'episode_index':                  {'dtype': 'int64',   'shape': [1], 'names': None},
        'index':                          {'dtype': 'int64',   'shape': [1], 'names': None},
        'task_index':                     {'dtype': 'int64',   'shape': [1], 'names': None},
    }
    for scan, state in scene_states.items():
        meta_dir = Path(save_dir) / scan / 'meta'
        meta_dir.mkdir(parents=True, exist_ok=True)
        info = {
            'codebase_version': 'v2.1',
            'robot_type':       'h1',
            'total_episodes':   state['episode_count'],
            'total_frames':     state['frame_count'],
            'total_tasks':      state['task_count'],
            'total_videos':     state['episode_count'],
            'total_chunks':     1,
            'chunks_size':      1000,
            'fps':              30,
            'splits':           {'train': f"0:{state['episode_count']}"},
            'data_path':        'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path':       'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.npy',
            'features':         features,
        }
        with open(meta_dir / 'info.json', 'w') as f:
            json.dump(info, f, indent=2)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       default='scripts/eval/configs/h1_internvla_n1_async_cfg_orig.py')
    parser.add_argument('--task_name',    default=None)
    parser.add_argument('--ne_threshold', type=float, default=3.0)
    parser.add_argument('--save_dir',     default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r_h1')
    parser.add_argument('--no_save',      action='store_true', help='metric만 측정, 저장 생략')
    parser.add_argument('--max_episodes', type=int, default=None, help='수집할 최대 episode 수 (테스트용)')
    args = parser.parse_args()

    eval_cfg = load_eval_cfg(args.config)
    from internnav.configs.evaluator.vln_default_config import get_config

    eval_cfg.task.task_name = args.task_name if args.task_name else f'gt_{eval_cfg.task.task_name}'
    eval_cfg = get_config(eval_cfg)
    eval_cfg.env.env_settings['rank']       = 0
    eval_cfg.env.env_settings['local_rank'] = 0
    eval_cfg.env.env_settings['world_size'] = 1
    eval_cfg.env.env_settings['dataset']    = eval_cfg.dataset

    robot_name = eval_cfg.task.robot_name

    from internnav.env import Env
    env = Env.init(eval_cfg.env, eval_cfg.task)

    os.makedirs(args.save_dir, exist_ok=True)
    results = []
    scene_states = defaultdict(lambda: {'episode_count': 0, 'task_count': 0, 'frame_count': 0})

    obs_list, reset_info_list = env.reset()

    while True:
        info = reset_info_list[0] if reset_info_list else None
        if info is None:
            break

        ref_path    = info.data['reference_path']
        path_key    = info.data['path_key']
        scene_id    = info.data.get('scene_id', '')
        scan        = scene_id.split('/')[-2] if '/' in scene_id else scene_id
        total_wps   = len(ref_path)
        instruction = info.data['instruction']['instruction_text']

        if args.max_episodes and len(results) >= args.max_episodes:
            break

        print(f'\n[{path_key}] scan={scan} wps={total_wps} | {instruction[:60]}...')

        obs, _ = step_until_finish(env, [{robot_name: {'stand_still': []}}], robot_name)

        step_buffer = [] if not args.no_save else None
        terminated = False
        for wp_idx, wp in enumerate(ref_path[1:], start=1):
            obs, terminated = navigate_to_waypoint(
                env, robot_name, wp, obs,
                step_buffer=step_buffer,
                wp_idx=wp_idx,
                total_wps=total_wps,
            )
            if terminated:
                break

        if not terminated:
            obs, terminated = step_until_finish(env, [{robot_name: {'stop': []}}], robot_name)

        metrics_raw = obs.get('metrics', {})
        if metrics_raw:
            m           = metrics_raw[list(metrics_raw.keys())[0]][0]
            ne          = float(m.get('NE', float('inf')))
            success     = bool(m.get('success', 0))
            fail_reason = m.get('fail_reason', '')
        else:
            final_pos   = np.array(obs.get('globalgps', ref_path[-1]))
            ne          = float(np.linalg.norm(final_pos[:2] - np.array(ref_path[-1])[:2]))
            success     = ne < args.ne_threshold
            fail_reason = 'no_metrics'

        n_steps = len(step_buffer) if step_buffer is not None else 0
        print(f'  NE={ne:.2f}m  success={success}  steps={n_steps}  reason={fail_reason}')
        results.append({'path_key': path_key, 'ne': ne, 'success': success, 'fail_reason': fail_reason})

        if success and not args.no_save and step_buffer:
            state = scene_states[scan]
            frames_written = save_episode(
                args.save_dir, scan,
                episode_index=state['episode_count'],
                task_index=state['task_count'],
                episode_data=info.data,
                step_buffer=step_buffer,
                success=success,
                fail_reason=fail_reason,
                global_frame_offset=state['frame_count'],
            )
            state['episode_count'] += 1
            state['task_count']    += 1
            state['frame_count']   += frames_written

        obs_list, reset_info_list = env.reset([0])

    env.close()

    if not args.no_save:
        finalize_info_json(args.save_dir, scene_states)

    total     = len(results)
    n_success = sum(r['success'] for r in results)
    avg_ne    = float(np.mean([r['ne'] for r in results])) if results else 0.0
    print(f'\n=== Summary ===')
    print(f'Total: {total}  SR: {n_success}/{total} ({n_success/total:.1%})  Avg NE: {avg_ne:.2f}m')

    with open(Path(args.save_dir) / 'results.json', 'w') as f:
        json.dump({'summary': {'total': total, 'n_success': n_success, 'avg_ne': avg_ne}, 'episodes': results}, f, indent=2)
    print(f'Results saved to {args.save_dir}/results.json')


if __name__ == '__main__':
    main()
