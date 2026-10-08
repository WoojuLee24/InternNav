"""vln_ce 에피소드 프레임 로더 — parquet(두 rig pose/goal) + 640×480 jpg/png.

`internnav/dataset/internvla_n1_lerobot_dataset.py`를 import하지 않고 **같은 규약만 재현**한다
(학습 코드와의 결합 회피 — 이 폴더의 코드는 기존 코드를 수정하지 않는다).

## 재현하는 규약 (실측 확인)
- rig 이름 `<H>cm_<pitch>deg`. 학습 preset은 `(height, pitch_1, pitch_2)` 3인자이고
  **pose/goal/depth는 전부 pitch_2(룩다운) rig 기준**이다 (`..._lerobot_dataset.py:891`
  `setting = f'{height}cm_{pitch_2}deg'`). pitch_1은 S2가 보는 FPV RGB에만 쓰인다.
- parquet: `pose.<rig>`(4×4 **cam2world**, 에피소드 시작 상대, world **z-up**),
  `goal.<rig>`([u,v] = [col,row], **640×480 정수**), `relative_goal_frame_id.<rig>`(-1 = goal 없음).
- depth PNG는 uint16 **mm**. 학습은 `PIL.NEAREST`로 224로 줄인 **뒤** `/1000`, 5.0 m clip
  (`..._lerobot_dataset.py:1052-1065`) — `to_224_depth()`가 그 순서를 그대로 따른다.

self-check: `/usr/bin/python scripts/dataset_converters/2dloader_vlnce/episode_io.py`
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

IMG_W, IMG_H = 640, 480
DEPTH_SCALE = 1000.0      # uint16 mm -> m
DEPTH_CLIP_M = 5.0        # 학습과 동일
BEV_RES = 224             # 학습이 모델에 넣는 해상도


# ---------------------------------------------------------------------------
# rig / preset
# ---------------------------------------------------------------------------

def parse_preset(preset: str):
    """'125cm_0_30' -> (rig_fpv, rig_ld) = ('125cm_0deg', '125cm_30deg').

    학습 preset 이름(`r2r_125cm_0_30` 등)의 뒷부분과 같은 형식이다.
    """
    parts = str(preset).split('_')
    assert len(parts) == 3 and parts[0].endswith('cm'), \
        f"preset 형식은 '<H>cm_<pitch1>_<pitch2>' (받은 값: {preset!r})"
    h = parts[0]
    return f'{h}_{int(parts[1])}deg', f'{h}_{int(parts[2])}deg'


# ---------------------------------------------------------------------------
# 프레임 / 에피소드
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """한 스텝. 이미지는 **640×480 원본 해상도**로 들고 있는다 (장애물 합성이 여기서 일어난다)."""
    idx: int
    rgb_fpv: np.ndarray     # (480,640,3) uint8 — pitch_1, S2가 보는 이미지
    rgb_ld: np.ndarray      # (480,640,3) uint8 — pitch_2, traj_images의 원본
    depth_m: np.ndarray     # (480,640) float32 metres, 0 = invalid (clip 전) — pitch_2
    depth_fpv_m: np.ndarray  # (480,640) float32 metres — pitch_1. FPV에 장애물 합성할 때 가림 판정용
    pose_fpv: np.ndarray    # (4,4) cam2world
    pose_ld: np.ndarray     # (4,4) cam2world
    goal_uv: np.ndarray     # (2,) int — [-1,-1]이면 goal 없음
    rel_goal: int           # -1이면 goal 없음

    @property
    def has_goal(self) -> bool:
        return self.rel_goal >= 0


@dataclass
class Episode:
    scene: str
    ep_id: int
    instruction: str
    rig_fpv: str
    rig_ld: str
    poses_ld: np.ndarray    # (N,4,4) 전체 시퀀스 — goal 계산에 뒤쪽 프레임이 필요하다
    frames: List[Frame]


def _video_dir(data_root: str, scene: str, ep_id: int) -> Path:
    return Path(data_root) / scene / 'videos' / f'chunk-{ep_id // 1000:03d}'


def _read_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_episode(data_root: str, scene: str, episode: int, preset: str = '125cm_0_30',
                 frame_ids: Optional[Sequence[int]] = None, n_frames: int = 6) -> Episode:
    """에피소드 하나를 읽는다.

    `frame_ids`를 주지 않으면 **goal이 있는 프레임 중** 균등하게 `n_frames`개를 고른다
    (goal이 없는 프레임은 pixel goal 검증에 쓸 수 없다).
    """
    import pyarrow.parquet as pq

    rig_fpv, rig_ld = parse_preset(preset)
    scene_dir = Path(data_root) / scene
    pq_path = scene_dir / 'data' / f'chunk-{episode // 1000:03d}' / f'episode_{episode:06d}.parquet'
    assert pq_path.exists(), f'parquet 없음: {pq_path}'
    t = pq.read_table(pq_path)
    for key in (f'pose.{rig_fpv}', f'pose.{rig_ld}', f'goal.{rig_ld}'):
        assert key in t.column_names, f'{pq_path.name}에 컬럼 {key} 없음 (있는 것: {t.column_names})'

    poses_fpv = np.array([np.asarray(p).reshape(4, 4) for p in t[f'pose.{rig_fpv}'].to_pylist()], dtype=np.float64)
    poses_ld = np.array([np.asarray(p).reshape(4, 4) for p in t[f'pose.{rig_ld}'].to_pylist()], dtype=np.float64)
    goals = np.array(t[f'goal.{rig_ld}'].to_pylist(), dtype=np.int64)
    rels = np.array(t[f'relative_goal_frame_id.{rig_ld}'].to_pylist(), dtype=np.int64)
    n = len(poses_ld)

    eps_meta = _read_jsonl(scene_dir / 'meta' / 'episodes.jsonl')
    instruction = ''
    for ep in eps_meta:
        if ep['episode_index'] == episode:
            instruction = ep['tasks'][0].split('<INSTRUCTION_SEP>')[0]
            break

    if frame_ids is None:
        # goal이 있고, 그 goal 프레임(i+rel+1)이 시퀀스 안에 있는 프레임만
        ok = [i for i in range(n) if rels[i] >= 0 and i + int(rels[i]) + 1 < n]
        assert ok, f'{scene} ep{episode}: goal 있는 프레임이 없다'
        frame_ids = [ok[k] for k in np.unique(np.linspace(0, len(ok) - 1, n_frames).astype(int))]

    vdir = _video_dir(data_root, scene, episode)
    frames = []
    for i in frame_ids:
        i = int(i)
        rgb_fpv = np.asarray(Image.open(vdir / f'observation.images.rgb.{rig_fpv}' /
                                        f'episode_{episode:06d}_{i}.jpg').convert('RGB'))
        rgb_ld = np.asarray(Image.open(vdir / f'observation.images.rgb.{rig_ld}' /
                                       f'episode_{episode:06d}_{i}.jpg').convert('RGB'))
        depth_raw = np.asarray(Image.open(vdir / f'observation.images.depth.{rig_ld}' /
                                          f'episode_{episode:06d}_{i}.png'))
        depth_raw_fpv = np.asarray(Image.open(vdir / f'observation.images.depth.{rig_fpv}' /
                                              f'episode_{episode:06d}_{i}.png'))
        assert rgb_ld.shape[:2] == (IMG_H, IMG_W) and depth_raw.shape == (IMG_H, IMG_W), \
            f'해상도 불일치: rgb{rgb_ld.shape} depth{depth_raw.shape}'
        frames.append(Frame(
            idx=i, rgb_fpv=rgb_fpv, rgb_ld=rgb_ld,
            depth_m=(depth_raw.astype(np.float32) / DEPTH_SCALE),
            depth_fpv_m=(depth_raw_fpv.astype(np.float32) / DEPTH_SCALE),
            pose_fpv=poses_fpv[i], pose_ld=poses_ld[i],
            goal_uv=goals[i], rel_goal=int(rels[i]),
        ))

    return Episode(scene=scene, ep_id=episode, instruction=instruction,
                   rig_fpv=rig_fpv, rig_ld=rig_ld, poses_ld=poses_ld, frames=frames)


def to_224_depth(depth_m: np.ndarray, res: int = BEV_RES, clip_m: float = DEPTH_CLIP_M) -> np.ndarray:
    """640×480 metric depth -> 학습이 모델에 넣는 것과 **동일한** [res,res] depth.

    학습(`..._lerobot_dataset.py:1052-1065`)은 uint16 PNG를 PIL NEAREST로 줄인 뒤 /1000, clip한다.
    여기서는 이미 metres이므로 mm로 되돌려 같은 경로를 태운다 — 합성으로 바뀐 depth도 같은 처리를 받게 하려는 것.
    """
    mm = np.clip(np.round(depth_m * DEPTH_SCALE), 0, 65535).astype(np.uint16)
    small = np.asarray(Image.fromarray(mm).resize((res, res), Image.NEAREST)).astype(np.float32) / DEPTH_SCALE
    small[small > clip_m] = clip_m
    return small


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    DR = os.environ.get('VLNCE_ROOT', 'data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')

    assert parse_preset('125cm_0_30') == ('125cm_0deg', '125cm_30deg')
    assert parse_preset('60cm_15_15') == ('60cm_15deg', '60cm_15deg')

    ep = load_episode(DR, '17DRP5sb8fy', 0, '125cm_0_30', n_frames=4)
    assert len(ep.frames) == 4 and ep.instruction, ep
    f = ep.frames[0]
    assert f.rgb_fpv.shape == (IMG_H, IMG_W, 3) and f.depth_m.shape == (IMG_H, IMG_W)
    assert f.has_goal and 0 <= f.goal_uv[0] < IMG_W and 0 <= f.goal_uv[1] < IMG_H, f.goal_uv
    # 두 rig는 카메라 중심이 같고 pitch만 다르다 (합성 박스를 양쪽에 투영할 때의 전제)
    dt = np.linalg.norm(f.pose_fpv[:3, 3] - f.pose_ld[:3, 3])
    assert dt < 1e-4, f'두 rig의 카메라 중심이 다르다: {dt:.4f} m'

    d224 = to_224_depth(f.depth_m)
    assert d224.shape == (224, 224) and d224.max() <= DEPTH_CLIP_M + 1e-6
    print(f'[io] {ep.scene} ep{ep.ep_id} frames={[fr.idx for fr in ep.frames]} '
          f'instr="{ep.instruction[:50]}..." depth[{f.depth_m.min():.2f},{f.depth_m.max():.2f}]m '
          f'goal={f.goal_uv.tolist()} rel={f.rel_goal} rig_dt={dt:.6f}m')
    print('[io] PASS')
