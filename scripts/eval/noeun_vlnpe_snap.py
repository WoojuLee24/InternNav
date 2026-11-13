"""VLN-PE(Isaac H1 flash) 에서 GT reference_path 를 재생하며 평가 카메라 사진과 카메라 자세를 저장한다 — 학습 렌더러와 사진 비교용.

noeun_gt_replay_check.py 와 같은 환경 · 같은 waypoint 추종. 다른 점: 고른 ep 만, --every step 마다 obs rgb 와
카메라 c2w(OpenCV: x 오른 · y 아래 · z 앞) 를 저장한다. c2w 위치 = obs camera_position (task 가 바닥 + 0.6 m 에 놓은 값),
방향 = 로봇 yaw + 아래 camera_pitch_deg (task._place_camera_on_floor 와 같은 식).
짝 = scripts/dataset_converters/nav3d/noeun/86_render_compare.py (같은 c2w 를 학습 렌더러로 찍어 나란히).

    CUDA_VISIBLE_DEVICES=0 NOEUN_AGENT_SERVER=0 NOEUN_DATA=<raw_data/r2r> /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/noeun_vlnpe_snap.py --episodes 1,8,20,27,34 --out logs/render_compare/vlnpe
출력: <out>/snaps.jsonl (사진 1 장 = 1 줄: ep · step · c2w · rgb 파일), <out>/rgb/ep<ep>_<step>.jpg
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


def c2w_cv(cam_pos, yaw, pitch_deg):
    p = math.radians(pitch_deg)
    f = np.array([math.cos(yaw) * math.cos(p), math.sin(yaw) * math.cos(p), -math.sin(p)])
    r = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([r, -u, f])
    T[:3, 3] = cam_pos
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py')
    ap.add_argument('--episodes', required=True, help='val_seen episode_id 목록 (쉼표)')
    ap.add_argument('--every', type=int, default=12, help='이 step 마다 저장')
    ap.add_argument('--max_snaps', type=int, default=12, help='ep 당 최대 장수')
    ap.add_argument('--out', required=True)
    ap.add_argument('--warmup_renders', type=int, default=4)
    ap.add_argument('--save_depth', action='store_true',
                    help='obs 의 depth[m] 를 npz 로 같이 저장 (랜드마크 3D 위치 계산용). 기본 off = 기존 동작')
    args = ap.parse_args()
    want = {int(x) for x in args.episodes.split(',')}

    eval_cfg = G.load_eval_cfg(args.config)
    from internnav.configs.evaluator.vln_default_config import get_config

    eval_cfg = get_config(eval_cfg)
    eval_cfg.env.env_settings.update({'rank': 0, 'local_rank': 0, 'world_size': 1, 'dataset': eval_cfg.dataset})
    robot_name = eval_cfg.task.robot_name
    pitch_deg = eval_cfg.task.task_settings['camera_pitch_deg']
    from PIL import Image

    from internnav.env import Env

    env = Env.init(eval_cfg.env, eval_cfg.task)
    os.makedirs(f'{args.out}/rgb', exist_ok=True)
    if args.save_depth:
        os.makedirs(f'{args.out}/depth', exist_ok=True)
    f_out = open(f'{args.out}/snaps.jsonl', 'w')
    obs_list, reset_info_list = env.reset()
    env.env.warm_up(steps=args.warmup_renders, render=True, physics=True)

    while reset_info_list and reset_info_list[0] is not None and want:
        info = reset_info_list[0]
        eid = int(info.data['episode_id'])
        if eid not in want:
            obs_list, reset_info_list = env.reset([0])
            continue
        want.discard(eid)
        ref = np.array(info.data['reference_path'], dtype=float)
        obs, terminated = G.step_until_finish(env, [{robot_name: {'stand_still': []}}], robot_name)
        step, n_snap = 0, 0

        def snap(o):
            yaw = G.get_yaw(o['globalrotation'])
            T = c2w_cv(np.asarray(o['camera_position'], dtype=float), yaw, pitch_deg)
            fn = f'rgb/ep{eid:02d}_{step:04d}.jpg'
            Image.fromarray(np.asarray(o['rgb'], dtype=np.uint8)).save(f'{args.out}/{fn}', quality=95)
            rec = {'ep': eid, 'step': step, 'c2w': T.tolist(), 'rgb': fn, 'robot_pos': np.round(o['globalgps'], 3).tolist()}
            if args.save_depth:
                if not hasattr(snap, '_keys_shown'):
                    print('[SNAP] obs keys:', sorted(o.keys()), flush=True)
                    snap._keys_shown = True
                d = o.get('depth')
                if d is None:
                    print('[SNAP] 경고: obs 에 depth 가 없다 — depth 저장 생략', flush=True)
                else:
                    dn = f'depth/ep{eid:02d}_{step:04d}.npz'
                    np.savez_compressed(f'{args.out}/{dn}', depth=np.asarray(d, dtype=np.float32))
                    rec['depth'] = dn
            f_out.write(json.dumps(rec) + '\n')
            f_out.flush()

        snap(obs)
        n_snap = 1
        for wp in ref[1:]:
            for _ in range(G.MAX_STEPS_PER_WP):
                cur = np.array(obs['globalgps'])
                if np.linalg.norm(wp[:2] - cur[:2]) < G.CLOSE_ENOUGH:
                    break
                dp = wp[:2] - cur[:2]
                delta = math.atan2(dp[1], dp[0]) - G.get_yaw(obs['globalrotation'])
                delta = ((delta + math.pi) % (2 * math.pi)) - math.pi
                a = (2 if delta > 0 else 3) if abs(delta) > G.TURN_ANGLE_RAD * 0.5 else 1
                obs, terminated = G.step_until_finish(env, [{robot_name: {'move_by_flash': [a]}}], robot_name)
                step += 1
                if step % args.every == 0 and n_snap < args.max_snaps:
                    snap(obs)
                    n_snap += 1
                if terminated or n_snap >= args.max_snaps:
                    break
            if terminated or n_snap >= args.max_snaps:
                break
        print(f'[SNAP] ep {eid} step {step} snaps {n_snap}', flush=True)
        obs_list, reset_info_list = env.reset([0])
    f_out.close()
    env.close()


if __name__ == '__main__':
    main()
