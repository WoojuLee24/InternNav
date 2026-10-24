"""04d — RBQ 가 출발 자리에서 넘어진 노은역 ep 를 **같은 경로 종류(평지/층 이동 · 같은 층 쌍)의 새 출발·도착·경로**로 바꾼다.

왜: 06k v3 (200 ep) 실패 6 건이 모두 출발 직후 넘어짐이었다 (noeun_walk_start_check.py). 출발점은 navmesh 반지름 r_b = 0.25 m 로만
벽에서 떨어져 있는데 RBQ 몸은 1.0 × 0.5 m (중심에서 몸 끝까지 ≈ 0.56 m) 이고 출발 yaw 는 R2R 분포로 무작위라,
몸 끝이 벽·턱 너머(바닥 없음)나 물체 위에 놓였다.

새 경로 규칙 = 04 와 같음 (같은 표본기·다듬기·U1~U5 판정) + **U6 (출발 자리)**:
| | 묻는 것 | 기준 | 근거 |
|---|---|---|---|
| U6 | 출발점 둘레에 몸을 어느 방향으로 놓아도 바닥이 평평한가 | 출발점 중심 반지름 `--start_disk_m` 원 안 0.1 m 격자 점마다 위에서 아래로 쏜 광선이 모두 닿고, 높이가 출발 바닥 ± `--start_dz_m` 안 | 실패 6 건은 몸 아래 꺼진 점 2~9 개. yaw 와 무관하게 재야 R2R 출발 yaw 분포를 안 건드린다 |

단계 (모두 저장소 루트):
  1. gen   (habitat): 바꿀 ep 마다 후보 `--n_cand` 개 → 04 형식 후보 json
  2. 04b · 04c 를 후보 json 에 그대로 (--paths_json/--out)
  3. eps   (habitat): 후보 → 06k 입력 json (0.5 m 경유점 · 출발 yaw = 06 start_yaw_for r2r_train 과 같은 규칙, 같은 episode_id seed)
  4. 06k_parallel 로 후보를 걷기
  5. merge (아무 python): 바꿀 ep 마다 성공한 첫 후보로 paths json · eps json · walk json 을 갈아 끼운 새 파일

사용:
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/43_replace_fallen.py gen
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/43_replace_fallen.py eps
    python3 scripts/dataset_converters/nav3d/noeun/43_replace_fallen.py merge --cand_walk <06k walk json>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
sys.path.insert(0, str(HERE))
import nav3d_path as npth  # noqa: E402
import nav3d_scenes as scenes  # noqa: E402
import nav3d_utils as nu  # noqa: E402
from mesh_probe import start_flat  # noqa: E402,F401
from path_quality import judge, load_holes, one, segments  # noqa: E402
from walk_inputs import resample, r2r_initial, start_yaw  # noqa: E402,F401
D04 = 'scripts/dataset_converters/nav3d/logs/noeun/paths'
D06K = 'scripts/dataset_converters/nav3d/logs/noeun/walk'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['gen', 'eps', 'merge'])
    ap.add_argument('--paths_json', default=f'{D04}/noeun_station_paths_meshfloor_straight.json', help='바꿀 대상이 든 현재 paths json')
    ap.add_argument('--walk_json', default=f'{D06K}/walk_noeun_v3.json', help='실패 ep 를 고를 06k 결과')
    ap.add_argument('--eps_json', default=f'{D06K}/eps_noeun_all_v2.json', help='현재 06k 입력 (merge 에서 갈아 끼움)')
    ap.add_argument('--n_cand', type=int, default=3, help='바꿀 ep 마다 후보 수 (걸어서 넘어지면 다음 후보)')
    ap.add_argument('--start_disk_m', type=float, default=0.6, help='U6 원 반지름 — RBQ 몸 끝 0.56 m + 여유')
    ap.add_argument('--start_dz_m', type=float, default=0.1, help='U6 바닥 높이 허용 폭')
    ap.add_argument('--seed', type=int, default=100, help='04 의 seed 0 과 다른 표본')
    ap.add_argument('--cand_json', default=f'{D04}/noeun_station_paths_cand.json', help='gen 출력 (04 형식)')
    ap.add_argument('--cand_straight_json', default=f'{D04}/noeun_station_paths_cand_meshfloor_straight.json', help='04b·04c 를 거친 후보')
    ap.add_argument('--cand_eps_json', default=f'{D06K}/eps_noeun_cand.json')
    ap.add_argument('--cand_walk', default=f'{D06K}/walk_noeun_cand.json')
    ap.add_argument('--start_seed', type=int, default=0, help='06 --start_seed 와 같게 (eps_noeun_all_v2 는 0)')
    ap.add_argument('--r2r_raw', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r/train/train.json.gz')
    ap.add_argument('--out_tag', default='v4', help='merge 출력 이름')
    return ap.parse_args()


def failed_keys(walk_json):
    return [r['key'] for r in json.loads(Path(walk_json).read_text())['results'] if not r['success']]


def stage_gen(args):
    import trimesh
    spec = scenes.SCENES['noeun_station']
    d01 = HERE / 'logs' / 'noeun' / '12_build_navmesh' / 'noeun_station'
    pf = nu.load_pathfinder(str(d01 / 'noeun_station.navmesh'))
    sc = npth.NavScene(pf)
    mesh = nu.add_box_blockers(nu.load_mesh_zup(spec['mesh']), spec.get('blockers', []))
    md = nu.MeshDistance(mesh)
    floor = trimesh.load('data/scene_data/noeun_station/noeun_station_mid.ply', force='mesh')   # 04b·04c 와 같은 바닥 mesh
    cfg = npth.RefineCfg()
    r_b = sc.agent_radius
    holes = load_holes('scripts/dataset_converters/nav3d/logs/noeun/11_check_assets/noeun_station/holes.csv')
    a04 = argparse.Namespace(flat_min_m=3.0, flat_max_m=40.0)
    rng = np.random.default_rng(args.seed)
    pts = scenes.sample_storey_points(pf, sc.levels, sc.lv2st, per_bucket=400, seed=args.seed + 1)
    P = {str(e['episode_id']): e for e in json.loads(Path(args.paths_json).read_text())['episodes']}
    keys = failed_keys(args.walk_json)
    print(f'[04d gen] 바꿀 ep {len(keys)}: {keys}', flush=True)
    out, stat = [], {}
    for key in keys:
        old = P[key.split(':')[1]]
        kind, sa, sb = old['kind'], old['from_storey'], old['to_storey']
        isl = sorted({k[1] for k in pts if (sa, k[1]) in pts and (sb, k[1]) in pts},
                     key=lambda i: -(len(pts[(sa, i)]) + len(pts[(sb, i)])))
        got, why_n = 0, {}
        for t in range(20000):
            if got >= args.n_cand:
                break
            il = isl[0] if kind == 'flat' else isl[rng.integers(len(isl))]
            A, B = pts[(sa, il)], pts[(sb, il)]
            a, b = nu.hab_to_ours(A[rng.integers(len(A))]), nu.hab_to_ours(B[rng.integers(len(B))])
            if kind == 'flat' and not (a04.flat_min_m <= np.linalg.norm(a[:2] - b[:2]) <= a04.flat_max_m):
                continue
            ok6, bad6, n6 = start_flat(floor, a, args.start_disk_m, args.start_dz_m)
            if not ok6:                                   # 경로를 찾기 전에 출발 자리부터 (싸다)
                why_n['U6 출발 자리'] = why_n.get('U6 출발 자리', 0) + 1
                continue
            Pth = sc.find_path(a, b)
            if Pth is None:
                why_n['U1 경로 없음'] = why_n.get('U1 경로 없음', 0) + 1
                continue
            row, r = one(sc, Pth, md, r_b, cfg)
            X, st = r['traj'].astype(np.float64), np.asarray(r['stairs'], bool)
            why = judge(row, X, kind, holes, r_b, a04)
            if kind == 'multifloor' and not (abs(X[0, 2] - old['trajectory_navmesh_z'][0]) < 0.5 and abs(X[-1, 2] - old['trajectory_navmesh_z'][-1]) < 0.5):
                why.append('층 쌍 다름')
            if why:
                for w in why:
                    why_n[w] = why_n.get(w, 0) + 1
                continue
            out.append({'episode_id': old['episode_id'], 'replaces': key, 'cand': got, 'kind': kind, 'from_storey': sa, 'to_storey': sb,
                        'source': '04d', 'start': X[0].tolist(), 'goal': X[-1].tolist(), 'trajectory': X.tolist(),
                        'trajectory_z': X[:, 2].tolist(), 'floor_z': float(X[0, 2]), 'segments': segments(st, X[:, 2]),
                        'path_length_m': float(row['ref_length_m']), 'mesh_clear_min_m': float(row['ref_mesh_clear_min_m']),
                        'margin_min_m': float(row['ref_margin_min_m']), 'stairs_frac': float(np.mean(st))})
            got += 1
        stat[key] = {'got': got, 'tries': t, 'rejected': why_n}
        print(f'  {key}: 후보 {got}/{args.n_cand} · 시도 {t} · 탈락 {why_n}', flush=True)
    d = json.loads(Path(args.paths_json).read_text())
    doc = {k: v for k, v in d.items() if k not in ('episodes', 'floor_source', 'narrow_source')}
    doc.update(episodes=out, replace_source={'args': vars(args), 'stat': stat, 'criteria': 'U1~U5 (04) + U6 (04d)'})
    Path(args.cand_json).write_text(json.dumps(doc))
    print(f'-> {args.cand_json} ({len(out)} 후보)')


def stage_eps(args):
    d = json.loads(Path(args.cand_straight_json).read_text())
    base = json.loads(Path(args.eps_json).read_text())
    init = r2r_initial(args.r2r_raw)
    E = []
    for ep in d['episodes']:
        T = np.asarray(ep['trajectory'], np.float64)
        E.append({'key': f"{ep['replaces']}:c{ep['cand']}", 'waypoints': resample(T).tolist(), 'start_yaw': start_yaw(ep, T, args.start_seed, init)})
    Path(args.cand_eps_json).write_text(json.dumps({k: v for k, v in base.items() if k != 'episodes'} | {'episodes': E}))
    print(f'-> {args.cand_eps_json} ({len(E)} ep)')


def stage_merge(args):
    W = json.loads(Path(args.walk_json).read_text())
    C = json.loads(Path(args.cand_walk).read_text())['results']
    cand_eps = {e['key']: e for e in json.loads(Path(args.cand_eps_json).read_text())['episodes']}
    cand_paths = {(e['replaces'], e['cand']): e for e in json.loads(Path(args.cand_straight_json).read_text())['episodes']}
    pick = {}
    for r in sorted(C, key=lambda r: r['key']):
        old, c = r['key'].rsplit(':', 1)
        if r['success'] and old not in pick:
            pick[old] = (int(c[1:]), r)
    keys = failed_keys(args.walk_json)
    print(f'[04d merge] 바꾼 ep {len(pick)}/{len(keys)} · 후보 걷기 성공 {sum(r["success"] for r in C)}/{len(C)}')
    for k in keys:
        print(f'  {k}: ' + (f'후보 c{pick[k][0]}' if k in pick else '성공한 후보 없음 — 그대로 둠'))
    # paths json
    d = json.loads(Path(args.paths_json).read_text())
    for i, e in enumerate(d['episodes']):
        key = next((k for k in pick if k.split(':')[1] == str(e['episode_id'])), None)
        if key is not None:
            d['episodes'][i] = dict(cand_paths[(key, pick[key][0])], replaced_from=e)
    d['replace_source'] = {'cand_walk': args.cand_walk, 'picked': {k: v[0] for k, v in pick.items()}}
    p_out = Path(args.paths_json).with_name(Path(args.paths_json).stem + f'_{args.out_tag}.json')
    p_out.write_text(json.dumps(d))
    # eps json · walk json (키는 원래 키로)
    E = json.loads(Path(args.eps_json).read_text())
    for i, e in enumerate(E['episodes']):
        if e['key'] in pick:
            E['episodes'][i] = dict(cand_eps[f"{e['key']}:c{pick[e['key']][0]}"], key=e['key'])
    e_out = Path(args.eps_json).with_name(f'eps_noeun_{args.out_tag}.json')
    e_out.write_text(json.dumps(E))
    W['results'] = [dict(pick[r['key']][1], key=r['key']) if r['key'] in pick else r for r in W['results']]
    W['replace_source'] = {'from_walk': args.walk_json, 'cand_walk': args.cand_walk}
    w_out = Path(args.walk_json).with_name(f'walk_noeun_{args.out_tag}.json')
    w_out.write_text(json.dumps(W))
    print(f'  성공 {sum(r["success"] for r in W["results"])}/{len(W["results"])} -> {p_out} · {e_out} · {w_out}')


def main():
    args = parse()
    if args.stage == 'gen':
        stage_gen(args)
    elif args.stage == 'eps':
        stage_eps(args)
    elif args.stage == 'merge':
        stage_merge(args)
    else:
        assert False, f'unreachable stage={args.stage!r}'


if __name__ == '__main__':
    main()
