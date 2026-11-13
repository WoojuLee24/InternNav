"""노은역 **목적지 한 장만** 빠르게 찍는다 — 로봇을 목적지로 순간이동시켜 렌더.

왜: `noeun_vlnpe_snap.py` 는 정답 경로를 **처음부터 끝까지 재생**하며 찍는다 (ep 당 약 7 분).
    stop_cue 생성 · 목적지 거르기는 **목적지 프레임 한 장**만 있으면 되므로, 경로를 걷지 않고
    flash 컨트롤러(`reset_robot_state`)로 목적지에 바로 놓고 한 장 찍는다.

자세: 위치 = 정답 경로 끝점, 방향 = 경로 마지막 구간의 진행 방향 (= 로봇이 목적지에 도착했을 때의 방향).
      카메라는 task 가 매 관측마다 바닥 + camera_floor_height, camera_pitch_deg 로 다시 놓는다
      (`noeun_vln_eval_task._place_camera_on_floor`) — `noeun_vlnpe_snap.py` 와 같은 자세가 된다.

c2w 규약·출력 형식은 `noeun_vlnpe_snap.py` 와 같다 (snaps.jsonl: ep · step · c2w · rgb · robot_pos)
-> `68_stop_cues.py` · `noeun_landmark_sam.py` 가 그대로 읽는다.

    (렌더는 GPU 0 에서만 된다 — 다른 GPU 는 "No device could be created", 2026-11-09 메모와 같음)
    CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0 NOEUN_AGENT_SERVER=0 NOEUN_DATA=<raw_data/r2r> \
      /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/noeun_goal_snap_fast.py --out logs/goal_fast
"""
import sys

sys.path.append('.')

import argparse
import importlib.util
import json
import math
import os

import numpy as np

_spec = importlib.util.spec_from_file_location('gt_replay', 'scripts/eval/noeun_gt_replay_check.py')
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
_s2 = importlib.util.spec_from_file_location('snap', 'scripts/eval/noeun_vlnpe_snap.py')


def c2w_cv(cam_pos, yaw, pitch_deg):
    """noeun_vlnpe_snap.c2w_cv 와 같은 식 (OpenCV: x 오른 · y 아래 · z 앞)."""
    p = math.radians(pitch_deg)
    f = np.array([math.cos(yaw) * math.cos(p), math.sin(yaw) * math.cos(p), -math.sin(p)])
    r = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([r, -u, f])
    T[:3, 3] = cam_pos
    return T


def yaw_to_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])  # wxyz, z축 회전


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py')
    ap.add_argument('--episodes', default='', help='episode_id 목록 (쉼표). 비우면 전부')
    ap.add_argument('--out', required=True)
    ap.add_argument('--warmup_renders', type=int, default=4)
    ap.add_argument('--settle_steps', type=int, default=2, help='순간이동 뒤 관측이 안정될 때까지 stand_still 수')
    ap.add_argument('--back_m', type=float, default=0.0,
                    help='목적지에서 이만큼 뒤로 물러나 찍는다 (0 = 목적지 바로 위). 도착 직전 시야를 원하면 0.5~1.0')
    return ap.parse_args()


def main():
    a = parse()
    want = {int(x) for x in a.episodes.split(',')} if a.episodes.strip() else None

    eval_cfg = G.load_eval_cfg(a.config)
    from internnav.configs.evaluator.vln_default_config import get_config

    eval_cfg = get_config(eval_cfg)
    eval_cfg.env.env_settings.update({'rank': 0, 'local_rank': 0, 'world_size': 1, 'dataset': eval_cfg.dataset})
    robot_name = eval_cfg.task.robot_name
    pitch_deg = eval_cfg.task.task_settings['camera_pitch_deg']
    base_h = eval_cfg.task.flash_controller_settings['base_height']
    from PIL import Image

    from internnav.env import Env

    env = Env.init(eval_cfg.env, eval_cfg.task)
    os.makedirs(f'{a.out}/rgb', exist_ok=True)
    f_out = open(f'{a.out}/snaps.jsonl', 'w')
    obs_list, reset_info_list = env.reset()
    env.env.warm_up(steps=a.warmup_renders, render=True, physics=True)

    n_done = 0
    while reset_info_list and reset_info_list[0] is not None:
        info = reset_info_list[0]
        eid = int(info.data['episode_id'])
        if want is not None and eid not in want:
            obs_list, reset_info_list = env.reset([0])
            continue
        ref = np.array(info.data['reference_path'], dtype=float)
        goal = ref[-1]
        # 도착 방향 = 마지막 구간 진행 방향
        i = max(0, len(ref) - 2)
        yaw = math.atan2(goal[1] - ref[i][1], goal[0] - ref[i][0])
        pos = goal.copy()
        if a.back_m > 0:
            pos[0] -= math.cos(yaw) * a.back_m
            pos[1] -= math.sin(yaw) * a.back_m

        # 로봇을 목적지로 순간이동 (flash 컨트롤러가 쓰는 것과 같은 길)
        # noeun_gt_replay_check.py:56 과 같은 길로 로봇을 찾는다
        flash = None
        for task in env.env.runner.current_tasks.values():
            rb = task.robots[list(task.robots.keys())[0]]
            for c in rb.controllers.values():
                if hasattr(c, 'reset_robot_state'):
                    flash = c
                    break
            if flash is not None:
                break
        assert flash is not None, 'reset_robot_state 를 가진 컨트롤러를 못 찾았다'
        flash.reset_robot_state(np.array([pos[0], pos[1], goal[2] + base_h]), yaw_to_quat(yaw))

        obs = None
        for _ in range(a.settle_steps):
            obs, _ = G.step_until_finish(env, [{robot_name: {'stand_still': []}}], robot_name)

        T = c2w_cv(np.asarray(obs['camera_position'], dtype=float), G.get_yaw(obs['globalrotation']), pitch_deg)
        fn = f'rgb/ep{eid:02d}_goal.jpg'
        Image.fromarray(np.asarray(obs['rgb'], dtype=np.uint8)).save(f'{a.out}/{fn}', quality=95)
        f_out.write(json.dumps({'ep': eid, 'step': 0, 'c2w': T.tolist(), 'rgb': fn,
                                'robot_pos': np.round(obs['globalgps'], 3).tolist(),
                                'goal': np.round(goal, 3).tolist()}) + '\n')
        f_out.flush()
        n_done += 1
        d = math.hypot(obs['globalgps'][0] - goal[0], obs['globalgps'][1] - goal[1])
        print(f'[FAST] ep {eid} 목적지까지 {d:.2f} m  ({n_done})', flush=True)
        obs_list, reset_info_list = env.reset([0])

    f_out.close()
    env.close()
    print(f'[FAST] 끝 — {n_done} 장')


if __name__ == '__main__':
    main()
