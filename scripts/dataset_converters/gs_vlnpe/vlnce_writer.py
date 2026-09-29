"""vln_ce 레이아웃 라이터 — parquet 21컬럼 + `meta/` 4종.

GT가 없는 씬(노은역)에서 쓴다. `05_export_vlnce.py`(mp3d)는 GT parquet·meta를 **바이트 복사**하면
끝이지만, 노은역은 원본이 없어 **직접 써야** 한다.

`vlnpe_writer.py`의 vln_ce 판이고 구조가 같다 — 릴리스 스키마를 원본에서 읽어 쓰고,
릴리스 에피소드를 읽어 다시 쓰는 자기검사를 붙인다.

스키마를 손으로 적지 않는다
---------------------------
`pose.<rig>`는 `list<list<float>>` 위에 HuggingFace `datasets`의 확장 타입 메타데이터가 붙어 있다:

```
ARROW:extension:name     = datasets.features.features.Array2DExtensionType
ARROW:extension:metadata = [[4, 4], "float32"]
```

그리고 스키마 전체에 `huggingface` 키가 하나 더 붙는다. **`datasets` 패키지는 이 환경에 없지만**
(Isaac python에 미설치), 이건 전부 **메타데이터**라서 `pyarrow`만으로 붙일 수 있다 —
릴리스 parquet에서 스키마를 통째로 읽어 그대로 쓰면 된다. 실측으로 확인했다:
스키마·필드메타·huggingface메타·값이 전부 일치.

이 방식이라 **릴리스가 바뀌어도 따라간다.** 타입을 손으로 옮겨 적으면 어긋난다.

21컬럼 (릴리스 실측)
--------------------
```
action                            int32     (프레임 0은 -1 센티널, 그 뒤 1=전진 2=좌 3=우)
pose.<rig>                        4x4 float32, 에피소드 **시작 기준 상대**, world Z-up   × 5 rig
goal.<rig>                        int32[2] = [u, v], 없으면 [-1,-1]                      × 5 rig
relative_goal_frame_id.<rig>      int32, -1 = goal 없음                                  × 5 rig
timestamp float32 (= frame_index / 30) · frame_index · episode_index · index · task_index  int64
```

meta 4종 (vln_pe보다 단순하다)
------------------------------
```
episodes.jsonl    {"episode_index", "tasks": [문장 1개], "length"}
tasks.jsonl       {"task_index", "task"}              ← instruction_tokens 없음, 어휘 불필요
info.json         robot_type: null · total_videos: 0 · fps: 30 · features 21개
episodes_stats.jsonl  {"episode_index", "stats": {21컬럼: {min,max,mean,std,count}}}
```

**지시문 토큰이 필요 없다** — InternVLA-N1이 Qwen 토크나이저로 문장을 직접 읽는다.
vln_pe(200개 정수 + 2504 어휘)와 완전히 다르다.

self-check: `/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/vlnce_writer.py --self_test`
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SCRIPT_NAME = 'vlnce_writer'

FPS = 30.0
CHUNK_SIZE = 1000
ACTION_START = -1
NO_GOAL = -1
# 스키마 원본 — 릴리스 parquet 하나. 타입·확장메타를 여기서 통째로 가져온다.
DEFAULT_SCHEMA_REF = ('data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/'
                      '17DRP5sb8fy/data/chunk-000/episode_000000.parquet')
INDEX_COLUMNS = ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index')


def load_schema(schema_ref):
    """릴리스 parquet에서 스키마를 그대로 가져온다 (확장 타입 메타 포함)."""
    import pyarrow.parquet as pq

    ref = Path(schema_ref)
    if not ref.is_file():
        raise FileNotFoundError(f'스키마 원본 parquet이 없다: {ref}')
    return pq.read_schema(ref)


def rigs_in_schema(schema) -> list:
    """스키마의 `pose.<rig>` 컬럼에서 rig 목록."""
    return [n[len('pose.'):] for n in schema.names if n.startswith('pose.')]


def scene_paths(save_dir, scene: str) -> dict:
    base = Path(save_dir) / scene
    return {'base': base, 'parquet_dir': base / 'data' / 'chunk-000',
            'videos_dir': base / 'videos' / 'chunk-000', 'meta_dir': base / 'meta'}


def stream_dir(videos_dir: Path, kind: str, rig: str) -> Path:
    return Path(videos_dir) / f'observation.images.{kind}.{rig}'


# ---------------------------------------------------------------------------
# parquet
# ---------------------------------------------------------------------------

def build_episode_table(schema, episode_index: int, task_index: int, action,
                        per_rig: dict, global_index_offset: int):
    """21컬럼 테이블. `per_rig[rig] = {'pose': (T,4,4), 'goal': (T,2), 'rel': (T,)}`.

    스키마에 있는 rig 중 `per_rig`에 없는 것은 **센티널로 채운다**
    (pose=단위행렬, goal=[-1,-1], rel=-1) — 학습은 preset이 고른 rig만 읽으므로 무해하지만,
    컬럼을 빼면 스키마가 달라져 로더가 깨진다.
    """
    import pyarrow as pa

    action = np.asarray(action, dtype=np.int32)
    T = len(action)
    cols = {'action': action.tolist()}

    for rig in rigs_in_schema(schema):
        d = per_rig.get(rig)
        if d is None:
            pose = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
            goal = np.full((T, 2), NO_GOAL, dtype=np.int32)
            rel = np.full(T, NO_GOAL, dtype=np.int32)
        else:
            pose = np.asarray(d['pose'], dtype=np.float32).reshape(T, 4, 4)
            goal = np.asarray(d['goal'], dtype=np.int32).reshape(T, 2)
            rel = np.asarray(d['rel'], dtype=np.int32).reshape(T)
        cols[f'pose.{rig}'] = pose.tolist()
        cols[f'goal.{rig}'] = goal.tolist()
        cols[f'relative_goal_frame_id.{rig}'] = rel.tolist()

    fi = np.arange(T, dtype=np.int64)
    cols['timestamp'] = (fi / FPS).astype(np.float32).tolist()
    cols['frame_index'] = fi.tolist()
    cols['episode_index'] = np.full(T, episode_index, dtype=np.int64).tolist()
    cols['index'] = (fi + global_index_offset).astype(np.int64).tolist()
    cols['task_index'] = np.full(T, task_index, dtype=np.int64).tolist()

    missing = [n for n in schema.names if n not in cols]
    if missing:
        raise KeyError(f'{SCRIPT_NAME}: 스키마 컬럼이 빠졌다 — {missing}')
    return pa.Table.from_pydict({n: cols[n] for n in schema.names}, schema=schema)


def _stats(vals) -> dict:
    """`episodes_stats.jsonl`의 컬럼 통계. 릴리스는 스칼라도 중첩 리스트로 적는다."""
    a = np.asarray(vals)
    flat = a.reshape(len(a), -1) if a.ndim > 1 else a.reshape(-1, 1)
    shape = a.shape[1:] if a.ndim > 1 else ()

    def pack(v):
        return np.asarray(v).reshape(shape).tolist() if shape else float(v[0])

    return {'min': pack(flat.min(axis=0)), 'max': pack(flat.max(axis=0)),
            'mean': pack(flat.mean(axis=0)), 'std': pack(flat.std(axis=0)),
            'count': [int(len(a))]}


def save_episode_vlnce(save_dir, scene: str, episode_index: int, action, per_rig: dict,
                       instruction: str, *, schema, global_index_offset: int = 0,
                       task_index: int = None) -> dict:
    """에피소드 하나의 parquet + meta 3종(append)을 쓴다. 이미지는 호출자가 이미 저장한 것으로 본다.

    vln_ce는 **에피소드당 지시문 1개**다(릴리스 1,938 에피소드 전수 확인). `task_index`를
    생략하면 `episode_index`와 같게 둔다 — 릴리스도 그렇다(`total_tasks == total_episodes`).
    """
    import pyarrow.parquet as pq

    P = scene_paths(save_dir, scene)
    for k in ('parquet_dir', 'meta_dir'):
        P[k].mkdir(parents=True, exist_ok=True)

    if task_index is None:
        task_index = episode_index
    action = np.asarray(action, dtype=np.int32)
    T = len(action)
    if T == 0:
        raise ValueError(f'{scene} ep{episode_index}: 프레임 0개')
    if action[0] != ACTION_START:
        raise ValueError(f'{scene} ep{episode_index}: action[0]={action[0]} (릴리스 규약은 -1)')

    table = build_episode_table(schema, episode_index, task_index, action, per_rig,
                                global_index_offset)
    pq.write_table(table, P['parquet_dir'] / f'episode_{episode_index:06d}.parquet',
                   compression='snappy')

    with open(P['meta_dir'] / 'episodes.jsonl', 'a') as f:
        f.write(json.dumps({'episode_index': episode_index, 'tasks': [instruction],
                            'length': T}, ensure_ascii=False) + '\n')
    with open(P['meta_dir'] / 'tasks.jsonl', 'a') as f:
        f.write(json.dumps({'task_index': task_index, 'task': instruction},
                           ensure_ascii=False) + '\n')

    d = table.to_pydict()
    stats = {n: _stats(np.asarray(d[n])) for n in schema.names}
    with open(P['meta_dir'] / 'episodes_stats.jsonl', 'a') as f:
        f.write(json.dumps({'episode_index': episode_index, 'stats': stats},
                           ensure_ascii=False) + '\n')
    return {'frames': T, 'next_index_offset': global_index_offset + T}


def finalize_info_json(save_dir, scene: str, schema, n_episodes: int, n_frames: int,
                       n_tasks: int = None) -> Path:
    """`meta/info.json`. features는 **스키마에서 유도**해 parquet과 어긋나지 않게 한다.

    (릴리스 vln_pe의 `info.json`은 features가 stale해서 실제 컬럼과 안 맞는다 —
    같은 실수를 하지 않으려고 스키마에서 만든다.)
    """
    import pyarrow as pa

    def feat(field):
        t = field.type
        if field.name.startswith('pose.'):
            return {'dtype': 'float32', 'shape': [4, 4], 'names': None}
        if field.name.startswith('goal.'):
            return {'dtype': 'int32', 'shape': [2], 'names': None}
        if pa.types.is_int32(t):
            return {'dtype': 'int32', 'shape': [1],
                    'names': ['action_index'] if field.name == 'action' else None}
        if pa.types.is_float32(t):
            return {'dtype': 'float32', 'shape': [1], 'names': None}
        return {'dtype': 'int64', 'shape': [1], 'names': None}

    P = scene_paths(save_dir, scene)
    P['meta_dir'].mkdir(parents=True, exist_ok=True)
    info = {
        'codebase_version': 'v2.1',
        'robot_type': None,                 # 릴리스 vln_ce는 null (vln_pe는 "unknown")
        'total_episodes': int(n_episodes),
        'total_frames': int(n_frames),
        'total_tasks': int(n_tasks if n_tasks is not None else n_episodes),
        'total_videos': 0,                  # 릴리스도 0 — 이미지가 mp4가 아니라 낱개 파일이다
        'total_chunks': 1,
        'chunks_size': CHUNK_SIZE,
        'fps': int(FPS),
        'splits': {'train': f'0:{int(n_episodes)}'},
        'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
        'features': {f.name: feat(f) for f in schema},
    }
    p = P['meta_dir'] / 'info.json'
    p.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return p


# ---------------------------------------------------------------------------
# 자기검사 — 릴리스를 읽어 다시 쓰고 원본과 대조 (GPU 불필요)
# ---------------------------------------------------------------------------

def self_test(ce_root: Path, scene: str, episode: int, schema_ref: str) -> int:
    import tempfile

    import pyarrow.parquet as pq

    schema = load_schema(schema_ref)
    rigs = rigs_in_schema(schema)
    G = scene_paths(ce_root, scene)
    src = G['parquet_dir'] / f'episode_{episode:06d}.parquet'
    if not src.is_file():
        print(f'[{SCRIPT_NAME}] 릴리스 에피소드가 없다: {src}', file=sys.stderr)
        return 2
    gt = pq.read_table(src).to_pydict()
    T = len(gt['action'])

    per_rig = {r: {'pose': np.asarray(gt[f'pose.{r}'], dtype=np.float32),
                   'goal': np.asarray(gt[f'goal.{r}'], dtype=np.int32),
                   'rel': np.asarray(gt[f'relative_goal_frame_id.{r}'], dtype=np.int32)}
               for r in rigs}
    eps_meta = {}
    for line in (G['meta_dir'] / 'episodes.jsonl').read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            eps_meta[d['episode_index']] = d
    instr = (eps_meta.get(episode, {}).get('tasks') or [''])[0]

    print(f'[{SCRIPT_NAME}] 릴리스 {scene} ep{episode}: {T}프레임 · rig {len(rigs)}개')
    fails = []
    with tempfile.TemporaryDirectory(prefix='vlnce_writer_') as td:
        out = Path(td)
        res = save_episode_vlnce(out, scene, episode, gt['action'], per_rig, instr,
                                 schema=schema, global_index_offset=int(gt['index'][0]),
                                 task_index=int(gt['task_index'][0]))
        finalize_info_json(out, scene, schema, 1, res['frames'])
        O = scene_paths(out, scene)
        dst = O['parquet_dir'] / f'episode_{episode:06d}.parquet'

        # ① 스키마 (필드 타입 + 확장 메타 + huggingface 메타)
        gs, os_ = pq.read_schema(src), pq.read_schema(dst)
        if not gs.equals(os_):
            diffs = [f'{n}: {gs.field(n).type} vs {os_.field(n).type}'
                     for n in gs.names if gs.field(n).type != os_.field(n).type]
            fails.append(f'스키마 다름 {diffs[:3] or "(타입은 같으나 equals 실패)"}')
        bad_meta = [n for n in gs.names
                    if (gs.field(n).metadata or {}) != (os_.field(n).metadata or {})]
        if bad_meta:
            fails.append(f'필드 확장 메타 다름: {bad_meta[:3]}')
        if (gs.metadata or {}).get(b'huggingface') != (os_.metadata or {}).get(b'huggingface'):
            fails.append('huggingface 스키마 메타 다름')
        if not fails:
            print(f'[{SCRIPT_NAME}] ① 스키마 21컬럼 · Array2D 확장 메타 · huggingface 메타 동일')

        # ② 컬럼 값
        od = pq.read_table(dst).to_pydict()
        for n in gs.names:
            a, b = np.asarray(gt[n]), np.asarray(od[n])
            if a.shape != b.shape:
                fails.append(f'{n}: shape {a.shape} vs {b.shape}')
                continue
            if not np.allclose(a.astype(np.float64), b.astype(np.float64), rtol=0, atol=1e-6):
                fails.append(f'{n}: 값 다름')
        if not any('값 다름' in f or 'shape' in f for f in fails):
            print(f'[{SCRIPT_NAME}] ② 21컬럼 값 전부 릴리스와 일치 '
                  f'(timestamp=fi/{FPS:g}, action[0]=-1 포함)')

        # ③ meta
        e = json.loads((O['meta_dir'] / 'episodes.jsonl').read_text().strip())
        t = json.loads((O['meta_dir'] / 'tasks.jsonl').read_text().strip())
        st = json.loads((O['meta_dir'] / 'episodes_stats.jsonl').read_text().strip())
        info = json.loads((O['meta_dir'] / 'info.json').read_text())
        if set(e) != {'episode_index', 'tasks', 'length'}:
            fails.append(f'episodes.jsonl 키 {sorted(e)} (기대 episode_index/tasks/length)')
        if e['length'] != T:
            fails.append(f'episodes.jsonl length {e["length"]} != {T}')
        if set(t) != {'task_index', 'task'}:
            fails.append(f'tasks.jsonl 키 {sorted(t)} (기대 task_index/task — 토큰은 없어야)')
        if len(e['tasks']) != 1:
            fails.append(f'지시문 {len(e["tasks"])}개 (vln_ce는 1개)')
        if set(st['stats']) != set(gs.names):
            fails.append(f'episodes_stats 컬럼 {len(st["stats"])}개 != {len(gs.names)}')
        if info['robot_type'] is not None or info['total_videos'] != 0:
            fails.append(f'info.json robot_type={info["robot_type"]} total_videos={info["total_videos"]}')
        if set(info['features']) != set(gs.names):
            fails.append('info.json features가 parquet 컬럼과 다르다')
        if not any(('jsonl' in f or 'info.json' in f or '지시문' in f) for f in fails):
            print(f'[{SCRIPT_NAME}] ③ meta 4종 — episodes/tasks 키·지시문 1개·stats 21컬럼·'
                  f'info(robot_type null, total_videos 0)')

        # ④ 릴리스 stats 형식과 대조
        g_st = None
        for line in (G['meta_dir'] / 'episodes_stats.jsonl').read_text().splitlines():
            if line.strip() and json.loads(line)['episode_index'] == episode:
                g_st = json.loads(line)
                break
        if g_st:
            for n in ('action', 'timestamp'):
                if n not in g_st['stats']:
                    continue
                if set(g_st['stats'][n]) != set(st['stats'][n]):
                    fails.append(f'stats[{n}] 키 {sorted(st["stats"][n])} vs '
                                 f'릴리스 {sorted(g_st["stats"][n])}')
                    break
            if not any('stats[' in f for f in fails):
                print(f'[{SCRIPT_NAME}] ④ episodes_stats 키 형식이 릴리스와 일치')

    if fails:
        print(f'\n[{SCRIPT_NAME}] 자기검사 FAIL {len(fails)}건')
        for f in fails[:12]:
            print(f'  - {f}')
        return 1
    print(f'\n[{SCRIPT_NAME}] 자기검사 전부 PASS — 이 라이터로 쓴 것은 릴리스 vln_ce와 구조가 같다')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self_test', action='store_true')
    ap.add_argument('--ce_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--scene', default='17DRP5sb8fy')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--schema_ref', default=DEFAULT_SCHEMA_REF)
    args = ap.parse_args()
    if not args.self_test:
        ap.print_help()
        return 0
    return self_test(Path(args.ce_root), args.scene, args.episode, args.schema_ref)


if __name__ == '__main__':
    sys.exit(main())
