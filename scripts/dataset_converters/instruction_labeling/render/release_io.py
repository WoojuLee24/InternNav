"""InternData vln_ce 릴리스(train traj_data) 읽기 + lerobot episode <-> raw R2R episode 매핑.

매핑 키는 (scene, 지시문 strip) — lerobot `episode_index` 순서와 R2R `episode_id` 순서가 다르다.
`gs_vlnpe/vlnce_align.build_rawdata_index` 를 재사용한다 (모든 split 병합, setdefault).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'scripts/dataset_converters/gs_vlnpe'))
from vlnce_align import build_rawdata_index  # noqa: E402


def load_cfg(path) -> dict:
    cfg = yaml.safe_load(open(path))
    for k, v in cfg['paths'].items():
        p = Path(v)
        cfg['paths'][k] = p if p.is_absolute() else REPO / p
    return cfg


def scene_glb(cfg, scene: str) -> Path:
    return cfg['paths']['scene_root'] / scene / f'{scene}.glb'


def list_release_scenes(cfg) -> list:
    return sorted(p.name for p in cfg['paths']['release_root'].iterdir() if (p / 'meta').is_dir())


def release_episodes(cfg, scene: str) -> list:
    """[{episode_index, task, length}] — meta/episodes.jsonl."""
    meta = cfg['paths']['release_root'] / scene / 'meta' / 'episodes.jsonl'
    out = []
    for line in open(meta):
        d = json.loads(line)
        out.append({'episode_index': d['episode_index'], 'task': d['tasks'][0], 'length': d['length']})
    return out


def release_episode_data(cfg, scene: str, ep_index: int, rigs) -> dict:
    """parquet 의 action, pose.<rig> (T,4,4), goal.<rig>, relative_goal_frame_id.<rig>."""
    pq = cfg['paths']['release_root'] / scene / 'data' / 'chunk-000' / f'episode_{ep_index:06d}.parquet'
    cols = ['action'] + [f'{k}.{r}' for r in rigs for k in ('pose', 'goal', 'relative_goal_frame_id')]
    df = pd.read_parquet(pq, columns=cols)
    out = {'action': np.asarray(df['action'].tolist(), dtype=np.int64)}
    for r in rigs:
        out[f'pose.{r}'] = np.stack([np.stack(x) for x in df[f'pose.{r}']]).astype(np.float64)
        out[f'goal.{r}'] = np.stack([np.asarray(x) for x in df[f'goal.{r}']])
        out[f'rel.{r}'] = np.asarray(df[f'relative_goal_frame_id.{r}'].tolist())
    return out


def release_image_path(cfg, scene: str, kind: str, rig: str, ep_index: int, frame: int) -> Path:
    ext = 'jpg' if kind == 'rgb' else 'png'
    return (cfg['paths']['release_root'] / scene / 'videos' / 'chunk-000' / f'observation.images.{kind}.{rig}'
            / f'episode_{ep_index:06d}_{frame}.{ext}')


_RAW_INDEX = None


def raw_index(cfg) -> dict:
    global _RAW_INDEX
    if _RAW_INDEX is None:
        _RAW_INDEX = build_rawdata_index(cfg['paths']['raw_root'])
    return _RAW_INDEX


def match_raw(cfg, scene: str, task: str):
    """릴리스 지시문 -> raw episode (없으면 None). `<INSTRUCTION_SEP>` 가 있으면 첫 문장으로 매칭."""
    idx = raw_index(cfg)
    t = task.strip()
    e = idx.get((scene, t))
    if e is None and '<INSTRUCTION_SEP>' in t:
        e = idx.get((scene, t.split('<INSTRUCTION_SEP>')[0].strip()))
    return e
