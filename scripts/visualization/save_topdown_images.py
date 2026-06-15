"""Save top-down camera images to a separate debug directory.

Runs a Habitat oracle follower and saves the top-down camera view per frame.
Images are stored in a directory completely separate from training data so that
the dataset loader can safely try to load them (present = debug mode, absent = no-op).

Camera setup (mirrors validate_depth_to_bev_habitat_cam.py):
  - Position:    [0, bev_cam_height, 0]  relative to agent base
  - Orientation: [-pi/2, 0, 0]  (looks straight down)
  - HFOV:        2 * atan(bev_range / bev_cam_height)
  - Resolution:  args.size x args.size

Output:
  {data_root}/traj_data/r2r/{scene}/videos/chunk-XXX/observation.images.rgb.topdown/episode_{ep_id:06d}_{frame_id}.jpg

Usage:
  python scripts/eval/save_topdown_images.py \
      --config scripts/eval/configs/habitat_dual_system_mini_5090_cfg.py \
      --data_root data/InternData-N1-v0.5-mini-debug/vln-ce \
      --split train --skip_existing
"""

import argparse
import importlib.util
import math
import os
import sys

sys.path.append('.')

import cv2
import pyarrow.parquet as pq
from omegaconf import OmegaConf

import habitat
from habitat.config.default_structured_configs import HabitatSimRGBSensorConfig
from habitat_baselines.config.default import get_config as get_habitat_config

import internnav.habitat_extensions.vln.measures  # noqa: F401


def load_eval_cfg(config_path):
    spec = importlib.util.spec_from_file_location('eval_config_module', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.eval_cfg


def get_parquet_path(data_root, scene_id, ep_idx):
    chunk = f'chunk-{ep_idx // 1000:03d}'
    return os.path.join(data_root, 'traj_data', 'r2r', scene_id,
                        'data', chunk, f'episode_{ep_idx:06d}.parquet')


def build_instr_to_ep_idx(data_root, scene_id):
    """instruction text → LeRobot episode_index (from episodes.jsonl)."""
    import json
    jsonl_path = os.path.join(data_root, 'traj_data', 'r2r', scene_id, 'meta', 'episodes.jsonl')
    if not os.path.exists(jsonl_path):
        return {}
    mapping = {}
    with open(jsonl_path) as f:
        for line in f:
            ep = json.loads(line)
            for task in ep.get('tasks', []):
                for instr in task.split('<INSTRUCTION_SEP>'):
                    mapping[instr.strip()] = ep['episode_index']
    return mapping


def run(args):
    eval_cfg = load_eval_cfg(args.config)
    config = get_habitat_config(eval_cfg.env.env_settings['config_path'])

    with habitat.config.read_write(config):
        config.habitat.dataset.split = args.split

    bev_cam_h = args.bev_cam_height if args.bev_cam_height > 0 else args.bev_range * 2.0
    bev_hfov_deg = math.degrees(2.0 * math.atan(args.bev_range / bev_cam_h))

    print(f'[topdown camera]  height={bev_cam_h:.2f}m  hfov={bev_hfov_deg:.1f}deg  '
          f'covers=+-{args.bev_range}m  size={args.size}px')
    print(f'[output]  {args.data_root}/traj_data/r2r/{{scene}}/videos/chunk-XXX/observation.images.rgb.topdown/')

    with habitat.config.read_write(config):
        cam_cfg = HabitatSimRGBSensorConfig(
            height=args.size,
            width=args.size,
            hfov=int(round(bev_hfov_deg)),
            position=[0.0, bev_cam_h, 0.0],
            orientation=[-math.pi / 2, 0.0, 0.0],
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.update(
            {'topdown_rgb': cam_cfg}
        )
        sensor_node = config.habitat.simulator.agents.main_agent.sim_sensors.topdown_rgb
        OmegaConf.set_struct(sensor_node, False)
        sensor_node.uuid = 'topdown_rgb'

    env = habitat.Env(config)
    scene_filter = set(args.scenes) if args.scenes else None

    instr_maps = {}  # scene_id → {instruction: episode_index}

    ep_count = 0
    frame_total = 0
    skip_count = 0

    while True:
        obs = env.reset()
        if obs is None:
            break

        episode  = env.current_episode
        scene_id = episode.scene_id.split('/')[-2]

        if scene_filter and scene_id not in scene_filter:
            continue

        # Map Habitat episode → LeRobot episode_index via instruction text
        if scene_id not in instr_maps:
            instr_maps[scene_id] = build_instr_to_ep_idx(args.data_root, scene_id)
        instruction = getattr(getattr(episode, 'instruction', None), 'instruction_text', '').strip()
        ep_idx = instr_maps[scene_id].get(instruction)
        if ep_idx is None:
            print(f'  skip  {scene_id}  hab_ep={episode.episode_id}  (no matching episode_index)')
            continue

        parquet_path = get_parquet_path(args.data_root, scene_id, ep_idx)
        if not os.path.exists(parquet_path):
            continue

        chunk   = f'chunk-{ep_idx // 1000:03d}'
        out_dir = os.path.join(
            args.data_root, 'traj_data', 'r2r', scene_id,
            'videos', chunk, 'observation.images.rgb.topdown',
        )
        first_path = os.path.join(out_dir, f'episode_{ep_idx:06d}_0.jpg')

        if args.skip_existing and os.path.exists(first_path):
            print(f'  skip  {scene_id}  ep={ep_idx:06d}')
            skip_count += 1
            ep_count += 1
            if args.max_episodes and ep_count >= args.max_episodes:
                break
            continue

        os.makedirs(out_dir, exist_ok=True)

        # Replay original R2R trajectory actions from parquet
        actions = pq.read_table(parquet_path).to_pandas()['action'].tolist()
        # actions[0] == -1 (initial state, no action); actions[1:] are discrete Habitat actions

        frame_id = 0

        cv2.imwrite(os.path.join(out_dir, f'episode_{ep_idx:06d}_0.jpg'),
                    cv2.cvtColor(obs['topdown_rgb'], cv2.COLOR_RGB2BGR))

        for i in range(1, len(actions)):
            action = actions[i]
            if action <= 0:
                break
            obs = env.step(action)
            cv2.imwrite(os.path.join(out_dir, f'episode_{ep_idx:06d}_{i}.jpg'),
                        cv2.cvtColor(obs['topdown_rgb'], cv2.COLOR_RGB2BGR))
            frame_id = i
            if env.episode_over:
                break

        frame_total += frame_id + 1
        ep_count    += 1
        print(f'  saved  {scene_id}  ep={ep_idx:06d} (hab_id={episode.episode_id})  frames={frame_id}')

        if args.max_episodes and ep_count >= args.max_episodes:
            break

    env.close()
    print(f'\nDone. {ep_count} episodes ({skip_count} skipped), {frame_total} frames.')
    print(f'Saved to: {args.data_root}/traj_data/r2r/{{scene}}/videos/chunk-000/observation.images.rgb.topdown/')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='scripts/eval/configs/habitat_dual_system_mini_5090_cfg.py')
    parser.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce',
                        help='Data root; topdown images saved to {data_root}/traj_data/r2r/{scene}/videos/chunk-XXX/observation.images.rgb.topdown/')
    parser.add_argument('--split', default='train')
    parser.add_argument('--scenes', nargs='*', default=None)
    parser.add_argument('--max_episodes', type=int, default=0, help='0 = all')
    parser.add_argument('--max_steps',    type=int, default=500)
    parser.add_argument('--bev_range',    type=float, default=5.0)
    parser.add_argument('--bev_cam_height', type=float, default=0.0,
                        help='Camera height above agent root (0 = bev_range * 2)')
    parser.add_argument('--size',         type=int, default=224)
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
