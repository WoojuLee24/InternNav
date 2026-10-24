"""07 — instruction_labeling(b0, e2e) 를 GT 데이터 · nav3d 생성 데이터 · 노은역 데이터에 같은 설정으로 붙인다.

labeling 코드(`instruction_labeling/label`)는 고치지 않는다. 여기서는 **입력 item** 만 만든다:
  item = {'ep': raw R2R 형식 episode (instruction 없음), 'scene', 'root', 'episode_index', 'split', 'sib_idx'}
  -> `sources.episode_track` (parquet action + pose.<rig> + 이미지) -> `e2e.generate` (route words + key frame + VLM 1회)

| source | 이미지 (root) | 경로 단어·key frame 의 경로 (ep['reference_path']) |
|---|---|---|
| gt       | 릴리스 vln_ce (사람 지시문이 붙은 원본)       | R2R reference_path (GT waypoint, 그대로) |
| nav3d_wp | 05b vln_ce_nav3d/nav3d_wp (GT 경유점을 이어 계획) | nav3d 03 경로를 RDP 로 꺾인 점만 남김 |
| nav3d_sg | 05b vln_ce_nav3d/nav3d_sg (시작·끝만 주고 계획) | 〃 |
| noeun    | 06 vln_ce_noeun_nav3d (Isaac GS, rig 60cm_15deg) | 04 경로(nav3d 좌표) -> habitat 좌표 -> RDP |

## 용어
- **RDP 단순화**: 촘촘한 경로에서 직선에서 `--simplify_m` 이상 벗어나는 점만 남긴다 -> R2R waypoint 처럼 꺾이는 곳·계단 시작/끝만 남는다.
  (촘촘한 점을 그대로 쓰면 구간마다 높이 변화가 작아 `stairs_dy_m`(0.3 m) 를 못 넘어 "Go up the stairs" 가 안 나온다.)
- 같은 GT episode 는 source 가 달라도 **같은 episode_id** 를 써서 VLM seed 가 같다 (차이는 이미지·경로에서만 온다).

출력: `<out_root>/<label name>/<source>/meta/ep_<id>.json` (생성 기록) + `<source>_<name>.json.gz` (R2R 형식 + instruction).

사용 (habitat 컨테이너, 저장소 루트, vlm_server.py 가 떠 있을 때):
    /usr/bin/python scripts/dataset_converters/nav3d/common/label_paths.py --sources gt,nav3d_wp,nav3d_sg
    /usr/bin/python scripts/dataset_converters/nav3d/common/label_paths.py --sources noeun --limit 10
"""

import argparse
import copy
import gzip
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
LABEL = HERE.parent / 'instruction_labeling' / 'label'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LABEL))
sys.path.insert(0, str(HERE.parent / 'instruction_labeling' / 'render'))
import config as LC  # noqa: E402
import e2e  # noqa: E402
import nav3d_utils as nu  # noqa: E402
from route_words import CODE_VERSION  # noqa: E402
from sources import REPO, episode_track, strip_gt  # noqa: E402

D = 'data/InternData-N1-v0.5-mini'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sources', default='gt,nav3d_wp,nav3d_sg', help='gt | nav3d_wp | nav3d_sg | noeun (쉼표)')
    ap.add_argument('--method', default='b0')
    ap.add_argument('--set', action='append', default=[], help='annotate.py 와 같음. model 은 --model 로')
    ap.add_argument('--model', default='qwen3vl8b-local', help='configs/model/<name>.yaml')
    ap.add_argument('--seed', type=int, default=0, help='cfg seed (0 = b0 기본). 다른 값이면 이름에 _seed<N>')
    ap.add_argument('--episodes_json', default='scripts/dataset_converters/nav3d/logs/common/episodes.json', help='gt·nav3d 의 (scene, 릴리스 episode_index)')
    ap.add_argument('--nav03_dir', default='scripts/dataset_converters/nav3d/logs/common/refine_paths/scenes')
    ap.add_argument('--nav3d_root', default=f'{D}/vln_ce_nav3d/traj_data/r2r')
    ap.add_argument('--noeun_root', default=f'{D}/vln_ce_noeun_nav3d/traj_data/r2r')
    ap.add_argument('--noeun_scene', default='noeunstation')
    ap.add_argument('--noeun_rig', default='60cm_15deg')
    ap.add_argument('--noeun_log', default='scripts/dataset_converters/nav3d/logs/noeun/tools/render_vlnce_camera', help='06 frames_meta.json')
    ap.add_argument('--noeun_paths', default='scripts/dataset_converters/nav3d/logs/noeun/paths/noeun_station_paths.json')
    ap.add_argument('--simplify_m', type=float, default=0.3, help='RDP 허용 오차 (생성 경로 -> waypoint)')
    ap.add_argument('--vertices', default='rdp', choices=['rdp', 'rdp_merge'],
                    help='rdp = 꺾인 점 그대로 (기본, 이전 동작) / rdp_merge = merge_m 안의 같은 방향 회전을 하나로 (모퉁이 하나 = "turn left" 한 번; route_vertices.py)')
    ap.add_argument('--merge_m', type=float, default=1.5, help='rdp_merge 거리 (07r: 1.5 m 가 sg 큰 회전 일치 57 %, wp 43 %)')
    ap.add_argument('--limit', type=int, default=0, help='source 당 앞에서 N 개 (0 = 전부)')
    ap.add_argument('--pick', default='spread', help="noeun 에서 --limit 일 때: 'spread' (층 이동·평지 고르게) | 'head'")
    ap.add_argument('--workers', type=int, default=2, help='서버가 요청을 하나씩 처리하므로 2 면 충분 (이미지 인코딩만 겹침)')
    ap.add_argument('--out_root', default=f'{D}/instruction_labels_nav3d')
    return ap.parse_args()


def make_vertices(P: np.ndarray, args) -> np.ndarray:
    if args.vertices == 'rdp':
        return rdp(P, args.simplify_m)
    elif args.vertices == 'rdp_merge':
        from route_vertices import merge_turns
        return merge_turns(rdp(P, args.simplify_m), args.merge_m)
    else:
        assert False, f'unreachable vertices={args.vertices!r}'


def rdp(P: np.ndarray, eps: float) -> np.ndarray:
    """Ramer–Douglas–Peucker (3D). 끝점 유지."""
    keep = np.zeros(len(P), bool)
    keep[[0, -1]] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L2 = max(float(ab @ ab), 1e-12)
        t = np.clip(((P[i + 1:j] - a) @ ab) / L2, 0, 1)
        d = np.linalg.norm(P[i + 1:j] - (a + t[:, None] * ab), axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return P[keep]


def quat_from_nav3d_yaw(theta: float) -> list:
    """nav3d yaw θ (전방 = (cosθ, sinθ), z-up) -> habitat start_rotation [x, y, z, w] (y 축 회전 φ = θ - 90°)."""
    phi = theta - math.pi / 2
    return [0.0, math.sin(phi / 2), 0.0, math.cos(phi / 2)]


def items_gt_nav3d(args, source: str) -> list:
    from release_io import load_cfg, match_raw, release_episodes
    rc = load_cfg(HERE.parent / 'instruction_labeling' / 'configs' / 'render.yaml')
    sel = json.loads(Path(args.episodes_json).read_text())
    out = []
    for scene, eps in sel.items():
        rel = {e['episode_index']: e for e in release_episodes(rc, scene)}
        if source == 'gt':
            root, idx = rc['paths']['release_root'], {ep: ep for ep in eps}
        elif source in ('nav3d_wp', 'nav3d_sg'):
            root = REPO / args.nav3d_root / source
            idx = {r['episode_id']: r['episode_index'] for r in map(json.loads, open(root.parent.parent / '_index' / source / f'{scene}.jsonl'))}
            Z = np.load(Path(args.nav03_dir) / f'{scene}_traj.npz')
        else:
            assert False, f'unreachable source={source!r}'
        for ep in eps:
            raw = match_raw(rc, scene, rel[ep]['task'])
            e = strip_gt(raw)
            if source != 'gt':
                key = f'{ep}_{source.split("_")[1]}_ref'
                if key not in Z or ep not in idx:
                    continue
                Wh = nu.ours_to_hab(make_vertices(Z[key].astype(np.float64), args))
                e = dict(e, reference_path=Wh.tolist(), start_position=Wh[0].tolist(), goals=[{'position': Wh[-1].tolist(), 'radius': 3.0}])
            out.append({'ep': e, 'scene': scene, 'root': root, 'episode_index': idx[ep], 'split': 'train', 'sib_idx': 0,
                        'src': {'source': source, 'release_episode_index': ep}})
    return out


def items_noeun(args) -> list:
    meta = json.loads((Path(args.noeun_log) / 'frames_meta.json').read_text())
    paths = {e['episode_id']: e for e in json.loads(Path(args.noeun_paths).read_text())['episodes']}
    out = []
    for m in meta['episodes']:
        p = paths[m['src_episode_id']]
        traj = np.asarray(p['trajectory'], dtype=np.float64)
        Wh = nu.ours_to_hab(make_vertices(traj, args))
        cam0 = np.asarray(m['cam_c2w_cv'][0])[:3, 3]                       # pose.<rig> 의 원점 = 시작 프레임 카메라 xy
        start = nu.ours_to_hab(np.array([cam0[0], cam0[1], m['body_z'][0]]))
        e = {'episode_id': int(m['episode']), 'trajectory_id': int(m['episode']), 'scene_id': 'noeun_station/noeun_station_collision.usdz',
             'start_position': start.tolist(), 'start_rotation': quat_from_nav3d_yaw(m['yaw'][0]),
             'goals': [{'position': Wh[-1].tolist(), 'radius': 3.0}], 'reference_path': Wh.tolist(),
             'info': {'geodesic_distance': float(p['path_length_m'])}}
        out.append({'ep': e, 'scene': args.noeun_scene, 'root': REPO / args.noeun_root, 'episode_index': int(m['episode']),
                    'split': 'noeun', 'sib_idx': 0, 'src': {'source': 'noeun', 'src_episode_id': m['src_episode_id'], 'kind': m['kind'],
                                                            'storeys': [p['from_storey'], p['to_storey']]}})
    if args.limit and args.pick == 'spread':          # 층 이동 · S0 평지 · S1 평지 를 번갈아
        groups = {}
        for it in out:
            groups.setdefault((it['src']['kind'], it['src']['storeys'][0]), []).append(it)
        rr, i = [], 0
        while len(rr) < min(args.limit, len(out)):
            for g in groups.values():
                if i < len(g) and len(rr) < args.limit:
                    rr.append(g[i])
            i += 1
        out = rr
    return out


def run_source(args, source: str):
    sets = list(args.set) + [f'model={args.model}']
    if source == 'noeun':
        sets.append(f'frames.rig={args.noeun_rig}')
    cfg = LC.load(args.method, sets)
    cfg['seed'] = args.seed
    name = cfg['name'] + (f'_seed{args.seed}' if args.seed else '') + (f'_merge{args.merge_m:g}' if args.vertices == 'rdp_merge' else '')
    ghash = LC.gen_hash(cfg, CODE_VERSION)
    its = items_noeun(args) if source == 'noeun' else items_gt_nav3d(args, source)
    if args.limit and not (source == 'noeun' and args.pick == 'spread'):
        its = its[:args.limit]
    out_dir = REPO / args.out_root / name / source
    md = out_dir / 'meta'
    md.mkdir(parents=True, exist_ok=True)
    todo = []
    for it in its:
        mp = md / f'ep_{it["ep"]["episode_id"]:06d}.json'
        if mp.is_file():
            old = json.loads(mp.read_text())
            assert old['gen_hash'] == ghash, f'{mp}: gen_hash {old["gen_hash"]} != {ghash} — 설정이 바뀌었다'
            continue
        todo.append(it)
    print(f'[07 {source}] {name} gen_hash={ghash} episodes={len(its)} todo={len(todo)}', flush=True)
    from openai import OpenAI
    client = OpenAI(base_url=cfg['vllm']['base_url'], api_key=cfg['vllm']['api_key'], timeout=600)
    t0 = time.time()

    def work(it):
        track = episode_track(it, cfg['frames']['rigs'][0])
        r = e2e.generate(client, cfg, it['ep'], track)
        m = {'episode_id': it['ep']['episode_id'], 'scene': it['scene'], 'episode_index': it['episode_index'], 'gen_hash': ghash,
             'name': name, 'src': it['src'], 'reference_path_n': len(it['ep']['reference_path']), 'frames_total': len(track['action']),
             'generated': r}
        (md / f'ep_{it["ep"]["episode_id"]:06d}.json').write_text(json.dumps(m, ensure_ascii=False))
        return r

    n_err = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(work, todo), 1):
            n_err += r['error'] is not None
            print(f'  {k}/{len(todo)} err={n_err} {time.time() - t0:.0f}s | {r["text"][:110]}', flush=True)
    eps = []
    for it in its:
        m = json.loads((md / f'ep_{it["ep"]["episode_id"]:06d}.json').read_text())
        e = copy.deepcopy(it['ep'])
        e['instruction'] = {'instruction_text': m['generated']['text'].strip() + ' ', 'instruction_tokens': []}
        eps.append(e)
    dst = out_dir / f'{source}_{name}.json.gz'
    with gzip.open(dst, 'wt') as f:
        json.dump({'episodes': eps, '_generation_meta': {'name': name, 'gen_hash': ghash, 'source': source, 'sets': sets,
                                                         'seed': args.seed, 'simplify_m': args.simplify_m, 'vertices': args.vertices,
                                                         'merge_m': args.merge_m if args.vertices == 'rdp_merge' else None, 'n': len(eps)}}, f)
    print(f'[07 {source}] -> {dst} ({time.time() - t0:.0f}s, 오류 {n_err})', flush=True)


def main():
    args = parse()
    for s in [x for x in args.sources.split(',') if x]:
        run_source(args, s)


if __name__ == '__main__':
    main()
