"""vln_pe 레이아웃 라이터 — 릴리스 파일과 **바이트 구조가 같은** parquet·npy·meta를 쓴다.

왜 필요한가
-----------
Phase A(GT pose 재렌더)는 parquet·meta를 GT에서 복사하면 끝이지만, Phase C(새로 뽑은 경로)는
GT가 없어서 **직접 써야** 한다. 그런데 저장소 전체에 vln_pe 레이아웃을 쓰는 코드가 없다
(읽기만 있다). `scripts/visualization/eval_gt_collect.py:save_episode`가 가장 가깝지만
릴리스와 **4곳이 다르고** parquet 스키마 메타데이터가 빠진다.

이 모듈이 메우는 차이 (전부 릴리스 데이터를 직접 읽어 확인한 것)

| | `eval_gt_collect.save_episode` | 릴리스 `vln_pe/traj_data/r2r` |
|---|---|---|
| `timestamp` | `fi / 30.0` | **`fi / 6.0`** |
| `observation.step` | `fi` | **`fi * 50`** (sim substep 카운터) |
| `episodes_stats.stats` 키 | 8개 (`observation.step` 없음) | **9개** |
| 에피소드당 지시문 | 1개 | **3개** (R2R 3인 주석) |
| parquet 스키마 메타 | 없음 | **`huggingface` 키 1,014 B** |

마지막 항목이 특히 조용히 틀린다 — `pa.table({...})`만으로는 그 메타데이터가 안 붙고,
`pyarrow`가 만든 파일은 열리기는 하지만 릴리스와 스키마가 **같지 않다.**
그래서 이 모듈은 **릴리스 parquet 하나를 스키마 원본으로 읽어** 그 스키마로 테이블을 만든다.
필드 타입과 메타데이터를 손으로 옮겨 적지 않으므로 릴리스가 바뀌어도 따라간다.

`_stats()`는 `eval_gt_collect`에서 그대로 재사용한다 (이중 중첩 리스트 형식이 릴리스와 같다).

자기검사
--------
GPU가 필요 없다. GT 에피소드를 읽어 `step_buffer`로 되돌린 뒤 이 라이터로 다시 쓰고,
스키마·메타데이터·컬럼 값이 GT와 같은지 비교한다.

/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/vlnpe_writer.py --self_test
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 릴리스 규약 (실측)
TIMESTAMP_FPS = 6.0        # timestamp = frame_index / 6.0  (info.json의 fps 30과 어긋나지만 실제 값이 이렇다)
STEP_STRIDE = 50           # observation.step = frame_index * 50
INSTRUCTIONS_PER_EPISODE = 3
DEFAULT_SCHEMA_REF = ('data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r/'
                      '17DRP5sb8fy/data/chunk-000/episode_000000.parquet')

# parquet 14컬럼 (릴리스 순서 그대로)
COLUMNS = [
    'observation.camera_position', 'observation.camera_orientation', 'observation.camera_yaw',
    'observation.robot_position', 'observation.robot_orientation', 'observation.robot_yaw',
    'observation.progress', 'observation.step', 'observation.action',
    'timestamp', 'frame_index', 'episode_index', 'index', 'task_index',
]
# episodes_stats.jsonl의 stats 9키 (릴리스와 같은 집합·순서)
STAT_COLUMNS = [
    'observation.camera_position', 'observation.camera_orientation', 'observation.camera_yaw',
    'observation.robot_position', 'observation.robot_orientation', 'observation.robot_yaw',
    'observation.progress', 'observation.step', 'observation.action',
]


def load_stats_fn():
    """`eval_gt_collect._stats`를 그대로 쓴다 — 새로 구현하면 형식이 갈린다."""
    import importlib.util

    path = REPO / 'scripts' / 'visualization' / 'eval_gt_collect.py'
    spec = importlib.util.spec_from_file_location('eval_gt_collect_for_writer', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._stats


def load_schema(schema_ref) -> 'pyarrow.Schema':  # noqa: F821
    """릴리스 parquet에서 스키마(필드 타입 + `huggingface` 메타)를 그대로 가져온다."""
    import pyarrow.parquet as pq

    ref = Path(schema_ref)
    if not ref.is_file():
        raise FileNotFoundError(f'스키마 원본 parquet이 없다: {ref}')
    schema = pq.read_schema(ref)
    if schema.names != COLUMNS:
        raise ValueError(f'스키마 원본의 컬럼이 예상과 다르다:\n  기대 {COLUMNS}\n  실제 {schema.names}')
    return schema


def scene_paths(save_dir, scan: str) -> dict:
    base = Path(save_dir) / scan
    return {
        'base': base,
        'parquet_dir': base / 'data' / 'chunk-000',
        'rgb_dir': base / 'videos' / 'chunk-000' / 'observation.images.rgb',
        'depth_dir': base / 'videos' / 'chunk-000' / 'observation.images.depth',
        'meta_dir': base / 'meta',
    }


def save_episode_vlnpe(save_dir, scan: str, episode_index: int, step_buffer: list,
                       instructions: list, *, schema, stats_fn,
                       success: bool = True, fail_reason: str = 'success',
                       faithful_index: bool = False, write_instruction_text: bool = False) -> int:
    """에피소드 하나를 vln_pe 레이아웃으로 쓴다. 반환: 프레임 수 T.

    `step_buffer`의 각 원소는 `eval_gt_collect`와 같은 dict다 —
    `camera_position`(3,) · `camera_orientation`(4, wxyz) · `camera_yaw` ·
    `robot_position`(3,) · `robot_orientation`(4,) · `robot_yaw` ·
    `progress` · `action` · `rgb`(H,W,3 uint8) · `depth`(H,W float32, 0<v<=1).

    `instructions`는 `[{'text': str, 'tokens': [200개 int]}, ...]` 3개다.
    3개가 아니면 마지막 것을 복제해 채운다(릴리스가 항상 3개이고 로더가 범위로 읽기 때문).

    `faithful_index`
        기본 False = **릴리스 규약을 그대로 따른다** — `episode_index` 컬럼은 0,
        `index`는 프레임 인덱스와 같고, `task_index` 컬럼도 0이다. 릴리스가 실제로 그렇고
        (stale이지만) 로더는 이 컬럼들을 읽지 않는다(`internnav/utils/loader.py:184-219`가
        `meta/episodes_stats.jsonl`의 범위를 쓴다). True면 실제 값을 쓴다 — 구조는 같지만
        릴리스와 값이 달라진다.

    `write_instruction_text`
        기본 False. 릴리스 `tasks.jsonl`에는 `instruction_text` 키가 **없다**(`task_index`,
        `task`, `instruction_tokens`, `finish_status`, `fail_reason` 5개). True면 키를 하나
        더 얹어 `CMA_CLIP_Policy`(`cma_lerobot_dataset.py:134`)도 바로 돌게 한다 — 구조가
        릴리스의 상위집합이 된다.
    """
    import cv2  # noqa: F401  (해상도 보정이 필요할 때만 쓴다)
    import pyarrow as pa
    import pyarrow.parquet as pq

    P = scene_paths(save_dir, scan)
    for key in ('parquet_dir', 'rgb_dir', 'depth_dir', 'meta_dir'):
        P[key].mkdir(parents=True, exist_ok=True)

    T = len(step_buffer)
    if T == 0:
        raise ValueError(f'{scan} ep{episode_index}: step_buffer가 비었다')

    ins = list(instructions)
    if not ins:
        raise ValueError(f'{scan} ep{episode_index}: 지시문이 없다')
    while len(ins) < INSTRUCTIONS_PER_EPISODE:
        ins.append(dict(ins[-1]))
    ins = ins[:INSTRUCTIONS_PER_EPISODE]
    task_base = episode_index * INSTRUCTIONS_PER_EPISODE

    cols = {c: [] for c in COLUMNS}
    rgbs, depths = [], []
    for fi, step in enumerate(step_buffer):
        cols['observation.camera_position'].append(np.asarray(step['camera_position'], dtype=np.float64).tolist())
        cols['observation.camera_orientation'].append(np.asarray(step['camera_orientation'], dtype=np.float64).tolist())
        cols['observation.camera_yaw'].append(float(step['camera_yaw']))
        cols['observation.robot_position'].append(np.asarray(step['robot_position'], dtype=np.float64).tolist())
        cols['observation.robot_orientation'].append(np.asarray(step['robot_orientation'], dtype=np.float64).tolist())
        cols['observation.robot_yaw'].append(float(step['robot_yaw']))
        cols['observation.progress'].append(float(step['progress']))
        # **릴리스 규약** — 아래 두 줄이 eval_gt_collect와 다른 지점이다
        cols['observation.step'].append(int(fi * STEP_STRIDE))
        cols['timestamp'].append(float(fi) / TIMESTAMP_FPS)
        cols['observation.action'].append(int(step['action']))
        cols['frame_index'].append(int(fi))
        cols['episode_index'].append(int(episode_index) if faithful_index else 0)
        cols['index'].append(int(fi))
        cols['task_index'].append(int(task_base) if faithful_index else 0)
        rgbs.append(np.asarray(step['rgb'], dtype=np.uint8))
        depths.append(np.asarray(step['depth'], dtype=np.float32))

    # 스키마를 **릴리스 파일에서 읽어온 것 그대로** 써서 필드 타입과 huggingface 메타를 맞춘다
    table = pa.table({c: pa.array(cols[c], type=schema.field(c).type) for c in COLUMNS},
                     schema=schema)
    ep_name = f'episode_{episode_index:06d}'
    pq.write_table(table, P['parquet_dir'] / f'{ep_name}.parquet', compression='snappy')
    np.save(P['rgb_dir'] / f'{ep_name}.npy', np.stack(rgbs).astype(np.uint8))
    np.save(P['depth_dir'] / f'{ep_name}.npy', np.stack(depths).astype(np.float32))

    # --- meta/episodes.jsonl : 지시문 3개 ---
    with open(P['meta_dir'] / 'episodes.jsonl', 'a') as f:
        f.write(json.dumps({'episode_index': episode_index,
                            'tasks': [i['text'] for i in ins]}, ensure_ascii=False) + '\n')

    # --- meta/tasks.jsonl : 3행 (task_index = 3i, 3i+1, 3i+2) ---
    with open(P['meta_dir'] / 'tasks.jsonl', 'a') as f:
        for j, i in enumerate(ins):
            row = {'task_index': task_base + j,
                   'task': i['text'],
                   'instruction_tokens': list(i['tokens']),
                   'finish_status': 'success' if success else 'fail',
                   'fail_reason': 'success' if success else fail_reason}
            if write_instruction_text:
                row['instruction_text'] = i['text']
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    # --- meta/episodes_stats.jsonl : stats 9키 + task_index 범위 ---
    stats = {c: stats_fn(cols[c]) for c in STAT_COLUMNS}
    with open(P['meta_dir'] / 'episodes_stats.jsonl', 'a') as f:
        f.write(json.dumps({
            'episode_index': episode_index,
            'stats': stats,
            # **로더가 에피소드->지시문을 찾는 유일한 경로다** (loader.py:184-219)
            'task_index': {'min': task_base, 'max': task_base + len(ins) - 1, 'count': T},
        }, ensure_ascii=False) + '\n')
    return T


def finalize_info_json_vlnpe(save_dir, scene_states: dict, robot_type: str = 'camera_only') -> None:
    """씬마다 `meta/info.json`을 쓴다.

    **릴리스 `info.json`의 `features`는 stale하다** — `vln_n1` 스키마를 복붙해 놓아서 실제
    parquet에 없는 `observation.camera_intrinsic`/`_extrinsic`/`action`을 선언하고 실제 9개
    `observation.*` 컬럼은 빠져 있다(`dataset_utils.py:17-18`이 이미 경고한다).
    `LerobotAsLmdb`는 `info.json`을 열지 않으므로, 여기서는 **실제 14컬럼**을 적는다.
    """
    features = {
        'observation.camera_position': {'dtype': 'float64', 'shape': [3], 'names': None},
        'observation.camera_orientation': {'dtype': 'float64', 'shape': [4], 'names': None},
        'observation.camera_yaw': {'dtype': 'float64', 'shape': [1], 'names': None},
        'observation.robot_position': {'dtype': 'float64', 'shape': [3], 'names': None},
        'observation.robot_orientation': {'dtype': 'float64', 'shape': [4], 'names': None},
        'observation.robot_yaw': {'dtype': 'float64', 'shape': [1], 'names': None},
        'observation.progress': {'dtype': 'float64', 'shape': [1], 'names': None},
        'observation.step': {'dtype': 'int64', 'shape': [1], 'names': None},
        'observation.action': {'dtype': 'int64', 'shape': [1], 'names': None},
        'timestamp': {'dtype': 'float32', 'shape': [1], 'names': None},
        'frame_index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'episode_index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'task_index': {'dtype': 'int64', 'shape': [1], 'names': None},
    }
    for scan, st in scene_states.items():
        meta_dir = Path(save_dir) / scan / 'meta'
        meta_dir.mkdir(parents=True, exist_ok=True)
        n_ep = int(st['episode_count'])
        info = {
            'codebase_version': 'v2.1',
            'robot_type': robot_type,
            'total_episodes': n_ep,
            'total_frames': int(st['frame_count']),
            'total_tasks': int(st.get('task_count', n_ep * INSTRUCTIONS_PER_EPISODE)),
            'total_videos': n_ep,
            'total_chunks': 1,
            'chunks_size': 1000,
            'fps': 30,
            'splits': {'train': f'0:{n_ep}'},
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.npy',
            'features': features,
        }
        (meta_dir / 'info.json').write_text(json.dumps(info, indent=2))


# ---------------------------------------------------------------------------
# 자기검사 — GT를 읽어 step_buffer로 되돌린 뒤 다시 쓰고 GT와 비교한다 (GPU 불필요)
# ---------------------------------------------------------------------------

def gt_to_step_buffer(gt_scene_dir: Path, episode: int) -> tuple:
    """GT 에피소드 -> (step_buffer, instructions). 라이터를 왕복 검증하기 위한 것."""
    import pyarrow.parquet as pq

    P = scene_paths(gt_scene_dir.parent, gt_scene_dir.name)
    t = pq.read_table(P['parquet_dir'] / f'episode_{episode:06d}.parquet').to_pydict()
    rgb = np.load(P['rgb_dir'] / f'episode_{episode:06d}.npy', mmap_mode='r')
    dep = np.load(P['depth_dir'] / f'episode_{episode:06d}.npy', mmap_mode='r')
    T = len(t['frame_index'])
    buf = [{
        'camera_position': np.asarray(t['observation.camera_position'][i]),
        'camera_orientation': np.asarray(t['observation.camera_orientation'][i]),
        'camera_yaw': t['observation.camera_yaw'][i],
        'robot_position': np.asarray(t['observation.robot_position'][i]),
        'robot_orientation': np.asarray(t['observation.robot_orientation'][i]),
        'robot_yaw': t['observation.robot_yaw'][i],
        'progress': t['observation.progress'][i],
        'action': t['observation.action'][i],
        'rgb': np.asarray(rgb[i]),
        'depth': np.asarray(dep[i]),
    } for i in range(T)]

    stats = [json.loads(l) for l in (P['meta_dir'] / 'episodes_stats.jsonl').read_text().splitlines() if l.strip()]
    tasks = {j['task_index']: j for j in
             (json.loads(l) for l in (P['meta_dir'] / 'tasks.jsonl').read_text().splitlines() if l.strip())}
    ti = next(s['task_index'] for s in stats if s['episode_index'] == episode)
    ins = [{'text': tasks[k]['task'], 'tokens': tasks[k]['instruction_tokens']}
           for k in range(ti['min'], ti['max'] + 1) if k in tasks]
    return buf, ins


def self_test(gt_root: Path, scene: str, episode: int, schema_ref: str) -> int:
    """GT를 다시 써서 스키마·메타·컬럼 값이 GT와 같은지 확인한다."""
    import tempfile

    import pyarrow.parquet as pq

    gt_scene = gt_root / scene
    schema = load_schema(schema_ref)
    stats_fn = load_stats_fn()
    buf, ins = gt_to_step_buffer(gt_scene, episode)
    print(f'[writer] GT {scene} ep{episode}: {len(buf)}프레임 · 지시문 {len(ins)}개')

    fails = []
    with tempfile.TemporaryDirectory(prefix='vlnpe_writer_') as td:
        out = Path(td)
        T = save_episode_vlnpe(out, scene, episode, buf, ins, schema=schema, stats_fn=stats_fn)
        if T != len(buf):
            fails.append(f'프레임 수 {T} != {len(buf)}')

        G = scene_paths(gt_root, scene)
        O = scene_paths(out, scene)
        gp = G['parquet_dir'] / f'episode_{episode:06d}.parquet'
        op = O['parquet_dir'] / f'episode_{episode:06d}.parquet'

        # ① 스키마 (필드 타입 + huggingface 메타)
        gs, os_ = pq.read_schema(gp), pq.read_schema(op)
        if gs.names != os_.names:
            fails.append(f'컬럼명 다름: {os_.names}')
        elif not gs.equals(os_):
            diffs = [f'{n}: {gs.field(n).type} vs {os_.field(n).type}'
                     for n in gs.names if gs.field(n).type != os_.field(n).type]
            fails.append(f'필드 타입 다름 {diffs}')
        g_hf = (gs.metadata or {}).get(b'huggingface')
        o_hf = (os_.metadata or {}).get(b'huggingface')
        if g_hf != o_hf:
            fails.append(f'huggingface 메타 다름 (GT {len(g_hf or b"")}B vs 우리 {len(o_hf or b"")}B)')
        else:
            print(f'[writer] ① 스키마 동일 · huggingface 메타 {len(g_hf or b"")}B 동일')

        # ② 컬럼 값 왕복
        gt_d, ot_d = pq.read_table(gp).to_pydict(), pq.read_table(op).to_pydict()
        for c in COLUMNS:
            a, b = gt_d[c], ot_d[c]
            same = (np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                                rtol=0, atol=1e-9)
                    if not isinstance(a[0], (list, tuple))
                    else np.allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-9))
            if not same:
                fails.append(f'컬럼 값 다름: {c} (GT {a[:2]} vs 우리 {b[:2]})')
        if not any('컬럼 값' in f for f in fails):
            print(f'[writer] ② 14컬럼 값 전부 GT와 일치 '
                  f'(timestamp=fi/{TIMESTAMP_FPS:g}, observation.step=fi*{STEP_STRIDE} 포함)')

        # ③ npy
        for key, dt in (('rgb_dir', 'uint8'), ('depth_dir', 'float32')):
            g = np.load(G[key] / f'episode_{episode:06d}.npy', mmap_mode='r')
            o = np.load(O[key] / f'episode_{episode:06d}.npy', mmap_mode='r')
            if g.shape != o.shape or str(g.dtype) != str(o.dtype) != dt:
                fails.append(f'{key}: GT {g.shape}{g.dtype} vs 우리 {o.shape}{o.dtype}')
            elif not np.array_equal(np.asarray(g), np.asarray(o)):
                fails.append(f'{key}: 픽셀 값 다름')
        if not any('_dir' in f for f in fails):
            print('[writer] ③ rgb·depth npy dtype/shape/값 동일')

        # ④ meta
        st = [json.loads(l) for l in (O['meta_dir'] / 'episodes_stats.jsonl').read_text().splitlines() if l.strip()]
        tk = [json.loads(l) for l in (O['meta_dir'] / 'tasks.jsonl').read_text().splitlines() if l.strip()]
        eps = [json.loads(l) for l in (O['meta_dir'] / 'episodes.jsonl').read_text().splitlines() if l.strip()]
        if len(tk) != INSTRUCTIONS_PER_EPISODE:
            fails.append(f'tasks.jsonl {len(tk)}행 (에피소드당 {INSTRUCTIONS_PER_EPISODE} 이어야)')
        if list(st[0]['stats'].keys()) != STAT_COLUMNS:
            fails.append(f'episodes_stats 키 {len(st[0]["stats"])}개 (9개 이어야): '
                         f'{list(st[0]["stats"].keys())}')
        want = {'min': episode * 3, 'max': episode * 3 + 2, 'count': T}
        if st[0]['task_index'] != want:
            fails.append(f'task_index 범위 {st[0]["task_index"]} != {want}')
        if eps[0]['tasks'] != [i['text'] for i in ins]:
            fails.append('episodes.jsonl tasks가 입력 지시문과 다름')
        for row in tk:
            if len(row['instruction_tokens']) != 200:
                fails.append(f'task{row["task_index"]}: 토큰 {len(row["instruction_tokens"])}개 (200 이어야)')
                break
        if not any(('tasks.jsonl' in f or 'episodes_stats' in f or 'task_index' in f or '토큰' in f)
                   for f in fails):
            print(f'[writer] ④ meta: tasks 3행 · stats 9키 · task_index {want} · 토큰 200개')

        # ⑤ 릴리스 stats 값과 대조 (형식이 같은지)
        g_st = [json.loads(l) for l in (G['meta_dir'] / 'episodes_stats.jsonl').read_text().splitlines() if l.strip()]
        g_row = next(s for s in g_st if s['episode_index'] == episode)
        for c in STAT_COLUMNS:
            if c not in g_row['stats']:
                fails.append(f'GT stats에 {c}가 없다 (기대와 다름)')
                continue
            ga, oa = g_row['stats'][c], st[0]['stats'][c]
            if set(ga.keys()) != set(oa.keys()):
                fails.append(f'stats[{c}] 키 집합 다름: {sorted(oa)} vs GT {sorted(ga)}')
                break
            if not np.allclose(np.asarray(ga['mean'], dtype=float),
                              np.asarray(oa['mean'], dtype=float), rtol=0, atol=1e-6):
                fails.append(f'stats[{c}].mean 값 다름')
        if not any('stats[' in f for f in fails):
            print('[writer] ⑤ episodes_stats 값이 GT와 일치 (min/max/mean/std/count 형식 포함)')

    if fails:
        print(f'\n[writer] 자기검사 FAIL {len(fails)}건')
        for f in fails:
            print(f'  - {f}')
        return 1
    print('\n[writer] 자기검사 전부 PASS — 이 라이터로 쓴 것은 릴리스 vln_pe와 구조가 같다')
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self_test', action='store_true', help='GT 왕복 자기검사를 실행한다')
    ap.add_argument('--gt_root', default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r')
    ap.add_argument('--scene', default='17DRP5sb8fy')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--schema_ref', default=DEFAULT_SCHEMA_REF,
                    help='스키마(필드 타입 + huggingface 메타)를 가져올 릴리스 parquet')
    args = ap.parse_args()
    if not args.self_test:
        ap.print_help()
        return 0
    return self_test(Path(args.gt_root), args.scene, args.episode, args.schema_ref)


if __name__ == '__main__':
    sys.exit(main())
