"""04 `--multifloor_mode stairs` 전용 — 점마다 바닥 z(`trajectory_z`)가 있는 층 이동 궤적을 렌더한다.

04 의 `generate_episode` 는 그대로 두고, 그 맨 앞 guard 가 `trajectory_z` 가 있는 에피소드만 여기로 넘긴다.
다른 점은 pose 합성 한 곳뿐이다: 카메라 z = 점마다 바닥 z + h_b (`multifloor_utils.synthesize_action_poses_z`).
렌더·저장·검사 코드는 04 모듈(`mod`)의 함수·상수를 그대로 부른다(04 의 `generate_episode` 와 같은 순서).
"""

from pathlib import Path

import cv2
import numpy as np

import multifloor_utils as mfu


def generate_episode_mf(mod, camera_and_sim, k: np.ndarray, ep_data: dict, ep_dir: Path,
                        frame_step_m: float) -> dict:
    xy, fz = mfu.resample_xyz_by_arclength(np.asarray(ep_data['trajectory'], dtype=np.float64),
                                           np.asarray(ep_data['trajectory_z'], dtype=np.float64), frame_step_m)
    action_poses = mfu.synthesize_action_poses_z(xy, fz, ep_data['h_b'], ep_data['pitch_deg'])
    poses_c2w = np.stack([mod.action_to_c2w(a, 'cam2world_gl') for a in action_poses])
    rgb, depth_m = mod.render_along(camera_and_sim, poses_c2w)
    n = len(action_poses)

    (ep_dir / 'rgb').mkdir(parents=True, exist_ok=True)
    (ep_dir / 'depth').mkdir(parents=True, exist_ok=True)
    (ep_dir / 'extrinsic').mkdir(parents=True, exist_ok=True)
    valid_all = np.zeros((n, mod.RENDER_H, mod.RENDER_W), dtype=bool)
    for i in range(n):
        mod.save_jpg(rgb[i], ep_dir / 'rgb' / f'frame_{i:04d}.jpg')
        valid = (np.isfinite(depth_m[i]) & (depth_m[i] < mod.RENDER_FAR_M * 0.99)
                 & (depth_m[i] >= mod.DEPTH_CLIP_MIN_M) & (depth_m[i] <= mod.DEPTH_CLIP_MAX_M))
        valid_all[i] = valid
        cv2.imwrite(str(ep_dir / 'depth' / f'frame_{i:04d}.png'), mod.depth_m_to_raw16(depth_m[i], valid))
        np.save(ep_dir / 'extrinsic' / f'frame_{i:04d}.npy', action_poses[i].astype(np.float32))
    np.save(ep_dir / 'intrinsic.npy', np.asarray(k, dtype=np.float32))

    step = np.linalg.norm(np.diff(poses_c2w[:, :2, 3], axis=0), axis=1)
    step_bad = int((step > frame_step_m * mod.GEN_STEP_TOL_RATIO).sum())

    rng = np.random.RandomState(0)
    anchor_d = []
    for i in range(0, n, max(1, n // 8)):
        pts_cam = np.stack(np.where(valid_all[i]), axis=1)
        if len(pts_cam) == 0:
            continue
        sel = pts_cam[rng.choice(len(pts_cam), min(300, len(pts_cam)), replace=False)]
        v, u = sel[:, 0], sel[:, 1]
        z = depth_m[i][v, u].astype(np.float64)
        x_cam = (u - k[0, 2]) * z / k[0, 0]
        y_cam = (v - k[1, 2]) * z / k[1, 1]
        pts_cam3 = np.stack([x_cam, y_cam, z], axis=1)
        pts_world = (poses_c2w[i][:3, :3] @ pts_cam3.T).T + poses_c2w[i][:3, 3]
        anchor_d.append(pts_world)
    return {
        'n_frames': n, 'step_max_m': float(step.max()) if n > 1 else 0.0, 'step_bad_n': step_bad,
        'frame_step_m': float(frame_step_m), 'step_tol_m': float(frame_step_m * mod.GEN_STEP_TOL_RATIO),
        'anchor_points_world': np.concatenate(anchor_d, axis=0) if anchor_d else np.empty((0, 3)),
        'rgb_first': rgb[0], 'depth_first': depth_m[0], 'valid_first': valid_all[0],
        'rgb_last': rgb[-1], 'depth_last': depth_m[-1], 'valid_last': valid_all[-1],
        'camera_z': poses_c2w[:, 2, 3].astype(float).tolist(),   # 층 이동 확인용 — 프레임별 카메라 z
    }
