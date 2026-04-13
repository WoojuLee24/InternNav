"""Pre-generate navdp dataset JSON for use with preload=True during training.

Usage:
    python scripts/train/base_train/generate_navdp_json.py \
        --root_dir /home/irteam/data-vol1/InternData-N1/vln_n1/traj_data \
        --output /home/irteam/data-vol1/datasets/navdp_dataset_lerobot_n1.json
"""
import argparse
import json
import os

import jsonlines
import numpy as np
from tqdm import tqdm


def generate(root_dir: str, output: str):
    trajectory_data_dir = []
    trajectory_rgb_path = []
    trajectory_depth_path = []
    trajectory_afford_path = []
    trajectory_scene_key = []
    # Each entry: {"scene": scene_id, "reason": str, "episode_idx": int or None}
    skip_log = []

    def skip_scene(scene_id, reason):
        print(f"  Skipping ({reason}): {scene_id}")
        skip_log.append({"scene": scene_id, "reason": reason, "episode_idx": None})

    def skip_episode(scene_id, episode_idx, reason):
        print(f"  Skipping episode {episode_idx} in {scene_id}: {reason}")
        skip_log.append({"scene": scene_id, "reason": reason, "episode_idx": episode_idx})

    group_dirs = sorted(os.listdir(root_dir))
    for group_dir in group_dirs:
        scene_dirs = sorted(os.listdir(os.path.join(root_dir, group_dir)))
        for scene_dir in tqdm(scene_dirs, desc=group_dir):
            scene_path = os.path.join(root_dir, group_dir, scene_dir)
            scene_id = f"{group_dir}/{scene_dir}"

            # Check data/ exists and is non-empty
            data_path = os.path.join(scene_path, 'data')
            data_entries = os.listdir(data_path) if os.path.isdir(data_path) else []
            if not data_entries:
                skip_scene(scene_id, "missing/empty data/")
                continue

            # Check meta files exist
            episodes_path = os.path.join(scene_path, 'meta/episodes_stats.jsonl')
            afford_dir = os.path.join(scene_path, 'meta/pointcloud.ply')
            if not os.path.isfile(episodes_path) or not os.path.isfile(afford_dir):
                skip_scene(scene_id, "missing meta files")
                continue

            chunk_name = data_entries[0]
            data_dir = os.path.join(scene_path, f'data/{chunk_name}')

            # Check video dirs exist
            rgb_dir = os.path.join(scene_path, f'videos/{chunk_name}/observation.images.rgb/')
            depth_dir = os.path.join(scene_path, f'videos/{chunk_name}/observation.images.depth/')
            if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
                skip_scene(scene_id, "missing video dirs")
                continue

            try:
                with jsonlines.open(episodes_path, 'r') as reader:
                    episode_info = list(reader)

                rgb_paths = [os.path.join(rgb_dir, p) for p in sorted(os.listdir(rgb_dir))]
                depth_paths = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
                data_paths = [os.path.join(data_dir, p) for p in sorted(os.listdir(data_dir))]
            except Exception as e:
                skip_scene(scene_id, f"error reading scene data: {e}")
                continue

            for episode_idx, episode in enumerate(episode_info):
                try:
                    image_start_index = episode['image_index']['min']
                    image_end_index = episode['image_index']['max']
                    episode_rgb_path = np.array(rgb_paths)[image_start_index: image_end_index + 1].tolist()
                    episode_depth_path = np.array(depth_paths)[image_start_index: image_end_index + 1].tolist()
                    trajectory_data_dir.append(data_paths[episode_idx])
                    trajectory_rgb_path.append(episode_rgb_path)
                    trajectory_depth_path.append(episode_depth_path)
                    trajectory_afford_path.append(afford_dir)
                    trajectory_scene_key.append(scene_id)
                except Exception as e:
                    skip_episode(scene_id, episode_idx, str(e))
                    continue

    os.makedirs(os.path.dirname(output), exist_ok=True)

    save_dict = {
        'trajectory_data_dir': trajectory_data_dir,
        'trajectory_rgb_path': trajectory_rgb_path,
        'trajectory_depth_path': trajectory_depth_path,
        'trajectory_afford_path': trajectory_afford_path,
        'trajectory_scene_key': trajectory_scene_key,
    }
    with open(output, 'w') as f:
        json.dump(save_dict, f, indent=4)
    print(f"\nSaved {len(trajectory_data_dir)} episodes -> {output}")

    if skip_log:
        log_path = output.replace('.json', '_skipped.json')
        with open(log_path, 'w') as f:
            json.dump(skip_log, f, indent=4)
        n_scenes = sum(1 for e in skip_log if e['episode_idx'] is None)
        n_episodes = sum(1 for e in skip_log if e['episode_idx'] is not None)
        print(f"Skipped {n_scenes} scenes + {n_episodes} episodes -> {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    generate(args.root_dir, args.output)
