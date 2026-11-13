"""노은역 씬 전체를 훑어 **랜드마크 후보 사진**을 찍는다 (목적지를 랜드마크 근처로 고르기 위해).

왜: 지금은 목적지를 navmesh 에서 무작위로 뽑고(22_select_paths) 거기에 쓸 만한 랜드마크가 있는지 **나중에** 본다.
    그래서 "2 m 안에 아무것도 없는" 목적지가 많이 나오고 대부분 버려진다.
    -> 순서를 뒤집어, **씬에 무엇이 어디 있는지 먼저 알아내고** 그 근처를 목적지로 삼는다.

방법: navmesh 를 격자로 훑어 설 수 있는 점을 고르고, 각 점에서 네 방향으로 한 장씩 찍는다.
      (렌더는 noeun_goal_snap_fast.py 와 같은 길 — flash 컨트롤러로 로봇을 놓고 관측 한 번)
      사진 + c2w 를 남기면 68_stop_cues 의 검출·광선이 그대로 붙는다.

출력: <out>/snaps.jsonl (사진 1 장 = 1 줄: id · c2w · rgb · robot_pos · yaw) · <out>/rgb/*.jpg

    (렌더는 GPU 0 에서만 된다)
    CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0 NOEUN_AGENT_SERVER=0 NOEUN_DATA=<raw_data/r2r> \
      /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/noeun_scene_scan.py --grid_m 5 --out logs/scene_scan
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
_fast = importlib.util.spec_from_file_location('fast', 'scripts/eval/noeun_goal_snap_fast.py')
F = importlib.util.module_from_spec(_fast)
_fast.loader.exec_module(F)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py')
    ap.add_argument('--navmesh', default='data/data/noeun_eval/navmesh/noeun_station/noeun_station.navmesh')
    ap.add_argument('--spots_json', default='', help='설 자리 목록(json). 주면 navmesh 를 읽지 않는다 — '
                    'habitat_sim 은 habitat 이미지에만 있고 렌더는 isaac 에서 하므로 미리 만들어 넘긴다')
    ap.add_argument('--make_spots', action='store_true', help='설 자리만 계산해 --spots_json 에 저장하고 끝낸다 (habitat 에서)')
    ap.add_argument('--grid_m', type=float, default=5.0, help='설 자리 격자 간격')
    ap.add_argument('--yaws', type=int, default=4, help='한 자리에서 찍을 방향 수')
    ap.add_argument('--max_spots', type=int, default=0, help='0 = 전부')
    ap.add_argument('--out', required=True)
    ap.add_argument('--warmup_renders', type=int, default=4)
    return ap.parse_args()


def spots(navmesh, grid_m):
    """navmesh 에서 격자로 설 수 있는 점 뽑기 (habitat_sim PathFinder)."""
    import importlib.util as iu
    sp = iu.spec_from_file_location('ns', 'scripts/eval/noeun_navmesh_server.py')
    NS = iu.module_from_spec(sp); sp.loader.exec_module(NS)
    import habitat_sim
    pf = habitat_sim.PathFinder()
    assert pf.load_nav_mesh(navmesh), navmesh
    lo, hi = pf.get_bounds()
    out, seen = [], set()
    xs = np.arange(lo[0], hi[0] + grid_m, grid_m)
    zs = np.arange(lo[2], hi[2] + grid_m, grid_m)
    for x in xs:
        for z in zs:
            for y in np.arange(lo[1], hi[1] + 1.0, 1.0):   # 층마다 시도
                p = pf.snap_point(np.array([x, y, z], dtype=np.float32))
                if not np.all(np.isfinite(p)):
                    continue
                w = NS.h2w(np.asarray(p, float))
                key = (round(w[0] / grid_m), round(w[1] / grid_m), round(w[2] / 1.5))
                if key in seen:
                    continue
                seen.add(key)
                out.append([float(t) for t in w])
    return out


def main():
    a = parse()
    if a.make_spots:
        pts = spots(a.navmesh, a.grid_m)
        os.makedirs(os.path.dirname(a.spots_json) or '.', exist_ok=True)
        json.dump(pts, open(a.spots_json, 'w'))
        print(f'[scan] 설 자리 {len(pts)} 개 -> {a.spots_json}')
        return
    pts = json.load(open(a.spots_json)) if a.spots_json else spots(a.navmesh, a.grid_m)
    if a.max_spots:
        pts = pts[:a.max_spots]
    print(f'[scan] 설 자리 {len(pts)} · 방향 {a.yaws} -> 사진 {len(pts) * a.yaws} 장', flush=True)

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
    env.reset()
    env.env.warm_up(steps=a.warmup_renders, render=True, physics=True)

    flash = None
    for task in env.env.runner.current_tasks.values():
        rb = task.robots[list(task.robots.keys())[0]]
        for c in rb.controllers.values():
            if hasattr(c, 'reset_robot_state'):
                flash = c; break
        if flash is not None:
            break
    assert flash is not None

    n = 0
    for si, p in enumerate(pts):
        for k in range(a.yaws):
            yaw = 2 * math.pi * k / a.yaws
            flash.reset_robot_state(np.array([p[0], p[1], p[2] + base_h]), F.yaw_to_quat(yaw))
            obs, _ = G.step_until_finish(env, [{robot_name: {'stand_still': []}}], robot_name)
            T = F.c2w_cv(np.asarray(obs['camera_position'], dtype=float), G.get_yaw(obs['globalrotation']), pitch_deg)
            fn = f'rgb/s{si:05d}_{k}.jpg'
            Image.fromarray(np.asarray(obs['rgb'], dtype=np.uint8)).save(f'{a.out}/{fn}', quality=92)
            f_out.write(json.dumps({'id': f's{si:05d}_{k}', 'spot': si, 'yaw_i': k,
                                    'c2w': T.tolist(), 'rgb': fn,
                                    'robot_pos': [round(float(t), 3) for t in obs['globalgps']]}) + '\n')
            n += 1
        if (si + 1) % 20 == 0:
            f_out.flush()
            print(f'[scan] 자리 {si + 1}/{len(pts)} · 사진 {n}', flush=True)
    f_out.close()
    env.close()
    print(f'[scan] 끝 — 사진 {n} 장')


if __name__ == '__main__':
    main()
