"""labeling 입력 소스 — split 별로 (raw episode, LeRobot 이미지 위치) 를 같은 형태로 돌려준다.

  train      : InternData 릴리스 traj_data (학습 이미지 그대로). lerobot episode <-> raw 는 지시문 일치로 매핑.
  val_unseen : render_split 산출물 (`vln_ce_render/.../val_unseen`), 매핑은 `_index/<scene>.jsonl`.

반환 item: {'ep': raw episode (instruction 필드 **제거**), 'scene', 'root', 'episode_index'}
GT 누출 차단: 생성 코드로 넘기는 ep 에는 instruction 이 없다. GT 문장은 평가(P3)에서만 읽는다.
"""

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'render'))
from release_io import REPO, load_cfg, release_episodes  # noqa: E402

RENDER_CFG = HERE.parent / 'configs' / 'render.yaml'
GT_KEYS = ('episode_id', 'trajectory_id', 'scene_id', 'start_position', 'start_rotation', 'goals',
           'reference_path', 'info')


def strip_gt(e: dict) -> dict:
    return {k: e[k] for k in GT_KEYS if k in e}


def load_raw(split: str) -> list:
    rc = load_cfg(RENDER_CFG)
    return json.load(gzip.open(rc['paths']['raw_root'] / split / f'{split}.json.gz'))['episodes']


def items(split: str) -> list:
    rc = load_cfg(RENDER_CFG)
    raw = load_raw(split)
    out = []
    if split == 'train':
        # lerobot episode -> raw episode: scene 안에서 지시문 완전일치, 같은 문장이 여러 개면 episode_index 순으로
        # 아직 안 쓴 후보를 배정한다 (relabel_vlnce/build_relabel_dataset.build_episode_map 과 같은 규칙).
        # 예: jh4fc5c5qoQ 의 lerobot 82/85 는 문장이 같지만 다른 trajectory(raw 5258/7415)다.
        by_scene = defaultdict(lambda: defaultdict(list))
        for e in raw:
            by_scene[Path(e['scene_id']).stem][e['instruction']['instruction_text'].strip()].append(e)
        root = rc['paths']['release_root']
        for scene in sorted(p.name for p in root.iterdir() if (p / 'meta').is_dir()):
            used = set()
            for le in release_episodes(rc, scene):
                cand = by_scene[scene].get(le['task'].strip(), [])
                if not cand:
                    continue
                e = next((c for c in cand if c['episode_id'] not in used), cand[0])
                used.add(e['episode_id'])
                out.append({'ep': strip_gt(e), 'scene': scene, 'root': root,
                            'episode_index': le['episode_index'], 'split': split})
    else:
        by_id = {e['episode_id']: e for e in raw}
        root = rc['paths']['out_root'] / split
        for fp in sorted((root / '_index').glob('*.jsonl')):
            for line in open(fp):
                r = json.loads(line)
                out.append({'ep': strip_gt(by_id[r['episode_id']]), 'scene': r['scene'], 'root': root,
                            'episode_index': r['episode_index'], 'split': split})
    out.sort(key=lambda x: x['ep']['episode_id'])
    rank = defaultdict(int)  # 같은 trajectory 안에서 episode_id 순서 (형제 번호, 0부터)
    for it in out:
        it['sib_idx'] = rank[it['ep']['trajectory_id']]
        rank[it['ep']['trajectory_id']] += 1
    return out


def gt_instructions(split: str) -> dict:
    """episode_id -> GT 문장. **평가 전용** (생성 코드에서 부르지 않는다)."""
    return {e['episode_id']: e['instruction']['instruction_text'] for e in load_raw(split)}


def episode_track(item: dict, rig: str) -> dict:
    """parquet action + pose.<rig> (시작 기준 로컬 OpenCV c2w) + 이미지 경로 함수."""
    root, scene, idx = Path(item['root']), item['scene'], item['episode_index']
    df = pd.read_parquet(root / scene / 'data' / 'chunk-000' / f'episode_{idx:06d}.parquet',
                         columns=['action', f'pose.{rig}'])
    pose = np.stack([np.stack(x) for x in df[f'pose.{rig}']]).astype(np.float64)
    vid = root / scene / 'videos' / 'chunk-000'

    def img(f, r=rig):
        return vid / f'observation.images.rgb.{r}' / f'episode_{idx:06d}_{f}.jpg'

    return {'action': np.asarray(df['action'].tolist()), 'pose': pose, 'img': img, 'item': item}


def house_path(ep: dict) -> Path:
    """MP3D .house (방 annotation). scene_id = 'mp3d/<scene>/<scene>.glb'."""
    scene = Path(ep['scene_id']).stem
    return load_cfg(RENDER_CFG)['paths']['scene_root'] / scene / f'{scene}.house'


def view_path(item: dict, at: str, turn_deg: float, rig: str) -> Path:
    """render/extra_views.py 산출물 위치 (trajectory 단위)."""
    rc = load_cfg(RENDER_CFG)
    return (rc['paths']['cache_root'] / 'views' / item['split'] / item['scene'] / str(item['ep']['trajectory_id'])
            / f'{at}_{int(turn_deg)}.{rig}.jpg')


def group_by_trajectory(its: list) -> dict:
    g = defaultdict(list)
    for it in its:
        g[it['ep']['trajectory_id']].append(it)
    return g
