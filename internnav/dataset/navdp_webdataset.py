"""
NavDP_WebDataset — episode-packed shard format for fast training.

Shard directory layout:
    {shard_dir}/
        shard_index.json           — {scene: [[ep_idx, length, orig_H, orig_W], ...]}
        {scene}/
            {ep_idx:06d}.rgb.npy   — (N, 224, 224, 3) float32, all frames pre-processed
            {ep_idx:06d}.depth.npy — (N, 224, 224, 1) float32
            {ep_idx:06d}.meta.npz  — intrinsic (3,3), extrinsic (4,4), trajectory (N,4,4)
        {scene}_obstacle.npy       — (M, 3) float32, obstacle points

Speed gain vs npy_cache:
    npy_cache  : ~13 file opens per __getitem__ (per-frame individual files)
    WebDataset : 3 file opens per episode (then numpy indexing from in-memory array)
                 Worker-level LRU cache means repeated episodes → 0 file opens
"""

import io
import json
import os

import cv2
import numpy as np
import torch

from internnav.dataset.navdp_lerobot_dataset import NavDP_Base_Datset


class NavDP_WebDataset(NavDP_Base_Datset):
    """
    Drop-in replacement for NavDP_Base_Datset that loads from episode-packed shards.
    Inherits all computation methods (process_actions, xyz_to_xyt, relative_pose, etc.)
    Overrides __init__, __len__, __getitem__.
    """

    def __init__(
        self,
        root_dirs,
        shard_dir,
        preload_path=False,
        memory_size=8,
        predict_size=24,
        batch_size=64,
        image_size=224,
        scene_data_scale=1.0,
        trajectory_data_scale=1.0,
        pixel_channel=7,
        action_dim=3,
        debug=False,
        preload=False,           # accepted but unused (shard_index replaces preload JSON)
        random_digit=False,
        prior_sample=False,
        is_train=True,
        val_ratio=0.0,
        use_npy_cache=True,      # accepted but unused (shards are already pre-processed)
    ):
        # Minimal attribute init — bypass NavDP_Base_Datset.__init__ entirely
        self.shard_dir = shard_dir
        self.memory_size = memory_size
        self.image_size = image_size
        self.scene_scale_size = scene_data_scale
        self.predict_size = predict_size
        self.action_dim = action_dim
        self.pixel_channel = pixel_channel
        self.debug = debug
        self.random_digit = random_digit
        self.prior_sample = prior_sample
        self.item_cnt = 0
        self.batch_size = batch_size
        self.batch_time_sum = 0.0
        self._last_time = None
        # Worker-local LRU cache: (scene, ep_idx) → episode data dict
        self._episode_cache: dict = {}
        self._cache_max = 256

        # Load shard index
        index_path = os.path.join(shard_dir, 'shard_index.json')
        with open(index_path) as f:
            shard_index = json.load(f)

        # Build episode lists (mirrors the scene-scale selection in parent __init__)
        self.trajectory_keys = []     # (scene, ep_idx)
        self.trajectory_lengths = []  # int
        self.trajectory_orig_hw = []  # (orig_H, orig_W)

        for scene in sorted(shard_index.keys()):
            episodes = shard_index[scene]  # list of [ep_idx, length, orig_H, orig_W]
            all_ep = np.array(episodes)    # shape (E, 4)
            select_idx = np.arange(0, all_ep.shape[0], 1 / self.scene_scale_size).astype(np.int32)
            selected = all_ep[select_idx]
            for ep_info in selected:
                ep_idx, ep_length, orig_H, orig_W = int(ep_info[0]), int(ep_info[1]), int(ep_info[2]), int(ep_info[3])
                self.trajectory_keys.append((scene, ep_idx))
                self.trajectory_lengths.append(ep_length)
                self.trajectory_orig_hw.append((orig_H, orig_W))

        # val split
        if val_ratio and val_ratio > 0:
            n_base = len(self.trajectory_keys)
            n_val = max(1, int(n_base * val_ratio))
            if is_train:
                self.trajectory_keys = self.trajectory_keys[:-n_val]
                self.trajectory_lengths = self.trajectory_lengths[:-n_val]
                self.trajectory_orig_hw = self.trajectory_orig_hw[:-n_val]
            else:
                self.trajectory_keys = self.trajectory_keys[-n_val:]
                self.trajectory_lengths = self.trajectory_lengths[-n_val:]
                self.trajectory_orig_hw = self.trajectory_orig_hw[-n_val:]

        # 50x replication (train only)
        if is_train:
            self.trajectory_keys = self.trajectory_keys * 50
            self.trajectory_lengths = self.trajectory_lengths * 50
            self.trajectory_orig_hw = self.trajectory_orig_hw * 50

        print(
            f'[NavDP_WebDataset] shard_dir={shard_dir} '
            f'episodes(×50)={len(self.trajectory_keys)} '
            f'scale={scene_data_scale}'
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.trajectory_keys)

    # ------------------------------------------------------------------
    # Episode loading (worker-local LRU cache)
    # ------------------------------------------------------------------

    def _load_episode(self, scene: str, ep_idx: int) -> dict:
        cache_key = (scene, ep_idx)
        if cache_key in self._episode_cache:
            return self._episode_cache[cache_key]

        ep_str = f'{ep_idx:06d}'
        scene_dir = os.path.join(self.shard_dir, scene)

        # mmap_mode='r': 파일을 메모리 맵으로 열기 → 접근한 프레임 페이지만 실제 disk read
        # episode 전체(~90MB)를 한 번에 적재하지 않고, rgb[memory_index] 등 인덱싱 시 필요한 오프셋만 읽음
        rgb = np.load(os.path.join(scene_dir, f'{ep_str}.rgb.npy'), mmap_mode='r')
        depth = np.load(os.path.join(scene_dir, f'{ep_str}.depth.npy'), mmap_mode='r')
        meta = np.load(os.path.join(scene_dir, f'{ep_str}.meta.npz'))

        obstacle_path = os.path.join(self.shard_dir, f'{scene}_obstacle.npy')
        obstacle = np.load(obstacle_path)

        data = {
            'rgb': rgb,
            'depth': depth,
            'intrinsic': meta['intrinsic'],     # (3,3)
            'extrinsic': meta['extrinsic'],     # (4,4)
            'trajectory': meta['trajectory'],   # (N,4,4)
            'orig_H': int(meta['orig_H']),
            'orig_W': int(meta['orig_W']),
            'obstacle': obstacle,
        }

        # Simple LRU eviction (drop oldest)
        if len(self._episode_cache) >= self._cache_max:
            self._episode_cache.pop(next(iter(self._episode_cache)))
        self._episode_cache[cache_key] = data
        return data

    # ------------------------------------------------------------------
    # pixel_goal helper using pre-loaded array (same logic as parent
    # process_pixel_goal but takes float32 array instead of file path)
    # ------------------------------------------------------------------

    def _process_pixel_goal_array(
        self, rgb_float32, target_point, camera_intrinsic, camera_extrinsic, orig_H, orig_W
    ):
        """
        Identical to NavDP_Base_Datset.process_pixel_goal() except:
          - rgb_float32 : pre-processed (224,224,3) float32 (replaces process_image call)
          - orig_H/W    : original image dims (replaces image.shape from PIL.open)
        """
        resize_image = rgb_float32  # already 224x224 float32

        coordinate = np.array([-target_point[1], target_point[0], camera_extrinsic[2, 3] * 0.8])
        camera_coordinate = np.matmul(camera_extrinsic[0:3, 0:3], coordinate[:, None])
        pixel_coord_x = (
            camera_intrinsic[0, 2]
            + (camera_coordinate[0] / camera_coordinate[2]) * camera_intrinsic[0, 0]
        )
        pixel_coord_y = (
            camera_intrinsic[1, 2]
            + (-camera_coordinate[1] / camera_coordinate[2]) * camera_intrinsic[1, 1]
        )

        pixel_mask = np.zeros((orig_H, orig_W, 3), dtype=np.uint8)
        visible_flag = False

        if (pixel_coord_x > 0 and pixel_coord_x < orig_W
                and pixel_coord_y > 0 and pixel_coord_y < orig_H):
            pixel_mask = cv2.rectangle(
                pixel_mask,
                (int(pixel_coord_x - np.random.randint(6, 12)),
                 int(pixel_coord_y - np.random.randint(6, 12))),
                (int(pixel_coord_x + np.random.randint(6, 12)),
                 int(pixel_coord_y + np.random.randint(6, 12))),
                (255, 255, 255),
                -1,
            )
            visible_flag = True

        H, W, C = pixel_mask.shape
        prop = self.image_size / max(H, W)
        pixel_mask = cv2.resize(pixel_mask, (-1, -1), fx=prop, fy=prop)
        pad_width = max((self.image_size - pixel_mask.shape[1]) // 2, 0)
        pad_height = max((self.image_size - pixel_mask.shape[0]) // 2, 0)
        pad_mask = np.pad(
            pixel_mask,
            ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
            mode='constant',
            constant_values=0,
        )
        mask = cv2.resize(pad_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = np.array(mask, np.float32) / 255.0
        mask = mask.mean(axis=-1)[:, :, None]
        return np.concatenate((resize_image, mask), axis=-1), visible_flag

    # ------------------------------------------------------------------
    # __getitem__ — identical logic to parent, image I/O replaced with
    # in-memory numpy indexing from episode arrays
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        import time

        if self._last_time is None:
            self._last_time = time.time()
        start_time = time.time()

        scene, ep_idx = self.trajectory_keys[index]
        ep_data = self._load_episode(scene, ep_idx)

        trajectory_length = len(ep_data['trajectory'])
        camera_intrinsic = ep_data['intrinsic']
        trajectory_base_extrinsic = ep_data['extrinsic']
        trajectory_extrinsics = ep_data['trajectory']
        trajectory_obstacle_points = ep_data['obstacle']
        orig_H = ep_data['orig_H']
        orig_W = ep_data['orig_W']

        # ---- same random sampling as parent ----
        if self.prior_sample:
            pixel_start_choice, target_choice = self.rank_steps(
                trajectory_extrinsics, trajectory_obstacle_points
            )
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)
        else:
            pixel_start_choice = np.random.randint(0, trajectory_length // 2)
            target_choice = np.random.randint(pixel_start_choice + 1, trajectory_length - 1)
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)

        if self.random_digit:
            memory_digit = np.random.randint(2, 8)
            pred_digit = memory_digit
        else:
            memory_digit = 4
            pred_digit = 4

        # ---- process_memory from pre-loaded array (no file I/O) ----
        memory_index = np.arange(
            memory_start_choice - (self.memory_size - 1) * memory_digit,
            memory_start_choice + 1,
            memory_digit,
        )
        outrange_sum = int((memory_index < 0).sum())
        memory_index = memory_index[outrange_sum:]
        context_image = np.zeros(
            (self.memory_size, self.image_size, self.image_size, 3), np.float32
        )
        context_image[outrange_sum:] = ep_data['rgb'][memory_index]  # (N,H,W,3) float32
        depth_image = ep_data['depth'][memory_start_choice]           # (H,W,1) float32

        # ---- process_actions (inherited, unchanged) ----
        (
            target_local_points,
            augment_local_points,
            target_world_points,
            augment_world_points,
            action_indexes,
        ) = self.process_actions(
            trajectory_extrinsics,
            trajectory_base_extrinsic,
            memory_start_choice,
            target_choice,
            pred_digit=pred_digit,
        )

        init_vector = target_local_points[1] - target_local_points[0]
        target_xyt_actions = self.xyz_to_xyt(target_local_points, init_vector)
        augment_xyt_actions = self.xyz_to_xyt(augment_local_points, init_vector)
        pred_actions = target_xyt_actions[action_indexes]
        augment_actions = augment_xyt_actions[action_indexes]

        if trajectory_obstacle_points.shape[0] != 0:
            pred_distance = (
                np.abs(
                    target_world_points[:, np.newaxis, 0:2]
                    - trajectory_obstacle_points[np.newaxis, :, 0:2]
                )
                .sum(axis=-1)
                .min(axis=-1)
            )
            augment_distance = (
                np.abs(
                    augment_world_points[:, np.newaxis, 0:2]
                    - trajectory_obstacle_points[np.newaxis, :, 0:2]
                )
                .sum(axis=-1)
                .min(axis=-1)
            )
            pred_critic = (
                -5.0 * (pred_distance[action_indexes[:-1]] < 0.1).mean()
                + 0.5 * (pred_distance[action_indexes][1:] - pred_distance[action_indexes][:-1]).sum()
            )
            augment_critic = (
                -5.0 * (augment_distance[action_indexes[:-1]] < 0.1).mean()
                + 0.5 * (augment_distance[action_indexes][1:] - augment_distance[action_indexes][:-1]).sum()
            )
        else:
            pred_distance = np.ones(pred_actions.shape[0], dtype=np.float32)
            augment_distance = np.ones(pred_actions.shape[0], dtype=np.float32)
            pred_critic = 2.0
            augment_critic = 2.0

        point_goal = target_xyt_actions[-1]
        image_goal = np.concatenate(
            (ep_data['rgb'][target_choice], ep_data['rgb'][memory_start_choice]),
            axis=-1,
        )

        # ---- pixel goal (process_actions called second time, same as parent) ----
        pixel_target_local_points, _, _, _, _ = self.process_actions(
            trajectory_extrinsics,
            trajectory_base_extrinsic,
            pixel_start_choice,
            target_choice,
            pred_digit=pred_digit,
        )
        pixel_init_vector = pixel_target_local_points[1] - pixel_target_local_points[0]
        pixel_xyt_actions = self.xyz_to_xyt(pixel_target_local_points, pixel_init_vector)
        pixel_goal, pixel_flag = self._process_pixel_goal_array(
            ep_data['rgb'][pixel_start_choice],
            pixel_xyt_actions[-1],
            camera_intrinsic,
            trajectory_base_extrinsic,
            orig_H,
            orig_W,
        )

        if self.pixel_channel == 7:
            pixel_goal = np.concatenate((pixel_goal, context_image[-1]), axis=-1)

        pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0
        augment_actions = (augment_actions[1:] - augment_actions[:-1]) * 4.0

        pred_actions = np.pad(
            pred_actions,
            ((0, 0), (0, self.action_dim - pred_actions.shape[-1])),
            mode='constant',
            constant_values=0,
        )
        augment_actions = np.pad(
            augment_actions,
            ((0, 0), (0, self.action_dim - augment_actions.shape[-1])),
            mode='constant',
            constant_values=0,
        )

        # ---- timing log (same as parent) ----
        end_time = time.time()
        self.item_cnt += 1
        self.batch_time_sum += end_time - start_time
        epoch_items = len(self)
        log_interval = 100 * epoch_items
        if self.item_cnt % log_interval == 0:
            avg_time = self.batch_time_sum / log_interval
            print(
                f'[WebDS] __getitem__ pid={os.getpid()}, '
                f'avg_time(last {log_interval})={avg_time:.2f}s, cnt={self.item_cnt}'
            )
            self.batch_time_sum = 0.0

        return (
            torch.tensor(point_goal, dtype=torch.float32),
            torch.tensor(image_goal, dtype=torch.float32),
            torch.tensor(pixel_goal, dtype=torch.float32),
            torch.tensor(context_image, dtype=torch.float32),
            torch.tensor(depth_image, dtype=torch.float32),
            torch.tensor(pred_actions, dtype=torch.float32),
            torch.tensor(augment_actions, dtype=torch.float32),
            torch.tensor(pred_critic, dtype=torch.float32),
            torch.tensor(augment_critic, dtype=torch.float32),
            float(pixel_flag),
        )
