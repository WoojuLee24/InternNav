"""04g — 계단 구간 경로를 곧게 펴서 RBQ 가 계단 위에서 제자리 회전하지 않게 한다 (paths json → 새 json, 지정한 ep 만).

왜: 06k ep 86 (S1→S2) 이 지상 출구 계단 맨 위에서 매번 넘어졌다 (4/4). 로봇은 경로를 ±0.06 m 로 따라갔지만
**경로가 계단 안에서 좌우로 0.6 m 꺾여** (navmesh 최단 경로) 계단 위에서 제자리 회전을 3 번 했고, 마지막 회전을 계단 끝에서 하다 넘어졌다.
04c 는 계단과 겹치는 구간을 일부러 건드리지 않았다.

규칙
- 계단 구간(segments kind=stairs) 마다: 계단 시작 `--pre_m` 앞 점 → 계단 끝 `--post_m` 뒤 점을 잇는 직선으로 바꾼다
  (회전은 계단 밖에서). z 는 원래 경로의 가장 가까운 점 z (계단 경사 유지).
  직선 하나가 검사를 못 넘으면 원래 경로에서 가장 먼 점에서 나눠 직선 조각들로 (RDP 방식, 조각 ≥ `--min_piece_m`).
- 새 직선의 모든 점이 다리 높이(바닥 + 0.15 / 0.4 m)에서 **좌우로** mesh 와 `--min_clear_m` 이상 떨어져 있고 (앞뒤 광선은 계단 디딤판에 닿아 쓰지 않음), 원래 경로에서 `--max_offset_m` 안일 때만 바꾼다.
  아니면 원래 경로를 두고 이유를 기록한다.
- 출력 episode 에 `stairs_straight` = [{i0, i1, applied, why|clear_min}] 를 남긴다. 06k 입력 json 도 같은 ep 만 갈아 끼운다 (출발 yaw 규칙 04d 와 같음).

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/45_straighten_stairs.py --episode_ids 86
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
sys.path.insert(0, str(HERE))
from mesh_probe import side_dists  # noqa: E402
from walk_inputs import r2r_initial, resample, start_yaw  # noqa: E402
D04 = 'scripts/dataset_converters/nav3d/logs/noeun/paths'
D06K = 'scripts/dataset_converters/nav3d/logs/noeun/walk'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths_json', default=f'{D04}/noeun_station_paths_meshfloor_straight_v5.json')
    ap.add_argument('--eps_json', default=f'{D06K}/eps_noeun_v5.json')
    ap.add_argument('--episode_ids', required=True, help='쉼표로 (예: 86). all = 전부')
    ap.add_argument('--mesh', default='data/scene_data/noeun_station/noeun_station_mid.ply')
    ap.add_argument('--pre_m', type=float, default=1.0, help='계단 앞 직선 길이 (그 전에 돌아서 들어가게)')
    ap.add_argument('--post_m', type=float, default=1.0, help='계단 끝 뒤 직선 길이 (다 올라간 뒤 돌게)')
    ap.add_argument('--min_clear_m', type=float, default=0.3)
    ap.add_argument('--max_offset_m', type=float, default=1.0, help='새 직선이 원래 경로에서 벗어나도 되는 최대 거리 (다른 계단으로 새지 않게)')
    ap.add_argument('--min_piece_m', type=float, default=1.5, help='직선 조각 최소 길이 (더 짧으면 펴지 않는다)')
    ap.add_argument('--start_seed', type=int, default=0)
    ap.add_argument('--r2r_raw', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r/train/train.json.gz')
    ap.add_argument('--out_tag', default='v6')
    return ap.parse_args()


def straight(T, ja, jb):
    m = max(2, int(np.ceil(np.linalg.norm(T[jb, :2] - T[ja, :2]) / 0.1)))
    xy = T[ja, :2] + (T[jb, :2] - T[ja, :2]) * (np.arange(m + 1)[:, None] / m)
    d = np.linalg.norm(T[None, ja:jb + 1, :2] - xy[:, None], axis=2)
    return np.c_[xy, T[np.argmin(d, axis=1) + ja, 2]], float(d.min(1).max())


def side_clear(mesh, seg):
    """직선 조각의 좌우 여유 (다리 높이 0.15 / 0.4 m). 앞뒤 광선은 계단 디딤판에 닿으므로 쓰지 않는다."""
    Lw, Rw, _ = side_dists(mesh, seg, seg[:, 2], [0.15, 0.4], 3.0)
    return np.minimum(Lw, Rw)


def fit_pieces(T, L, ja, jb, mesh, args):
    """ja~jb 를 직선 조각으로 (RDP 처럼 원래 경로에서 가장 먼 점에서 나눔). 조각마다 여유·벗어남 검사. 못 하면 None."""
    seg, off = straight(T, ja, jb)
    if off <= args.max_offset_m and side_clear(mesh, seg).min() >= args.min_clear_m:
        return [seg]
    if L[jb] - L[ja] < 2 * args.min_piece_m:
        return None
    a, b = T[ja, :2], T[jb, :2]
    ab = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    v = T[ja:jb + 1, :2] - a
    dist = np.abs(v[:, 0] * ab[1] - v[:, 1] * ab[0])
    lo, hi = int(np.searchsorted(L, L[ja] + args.min_piece_m)), int(np.searchsorted(L, L[jb] - args.min_piece_m))
    if hi <= lo:
        return None
    jm = lo + int(np.argmax(dist[lo - ja:hi - ja + 1]))
    left = fit_pieces(T, L, ja, jm, mesh, args)
    right = fit_pieces(T, L, jm, jb, mesh, args)
    return None if left is None or right is None else left + right


def main():
    import trimesh
    args = parse()
    mesh = trimesh.load(args.mesh, force='mesh')
    d = json.loads(Path(args.paths_json).read_text())
    ids = None if args.episode_ids == 'all' else {int(x) for x in args.episode_ids.split(',')}
    changed = []
    for e in d['episodes']:
        if ids is not None and e['episode_id'] not in ids:
            continue
        T = np.asarray(e['trajectory'], np.float64)
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T[:, :2], axis=0), axis=1))]
        rec, pieces, k = [], [], 0
        for s in sorted([s for s in e['segments'] if s['kind'] == 'stairs'], key=lambda s: s['i0']):
            i0, i1 = int(s['i0']), int(s['i1'])
            ja = int(np.searchsorted(L, L[i0] - args.pre_m))
            jb = min(len(T) - 1, int(np.searchsorted(L, L[i1] + args.post_m)))
            if ja < k:
                rec.append({'i0': i0, 'i1': i1, 'applied': False, 'why': '앞 계단 구간과 겹침'})
                continue
            parts = fit_pieces(T, L, ja, jb, mesh, args)
            if parts is None:
                rec.append({'i0': i0, 'i1': i1, 'applied': False, 'why': f'{args.min_piece_m} m 보다 짧게 나눠도 여유 부족'})
                continue
            seg = np.concatenate([q if n == 0 else q[1:] for n, q in enumerate(parts)])
            c = side_clear(mesh, seg)
            pieces.append((ja, jb, seg))
            k = jb + 1
            rec.append({'i0': i0, 'i1': i1, 'ja': ja, 'jb': jb, 'applied': True, 'pieces': len(parts), 'clear_min': float(c.min())})
        e['stairs_straight'] = rec
        if not pieces:
            continue
        out, last = [], 0
        for ja, jb, seg in pieces:                         # 원래 점 [last, ja) + 직선 seg (T[ja] ~ T[jb]) 를 차례로
            out += T[last:ja].tolist() + seg.tolist()
            last = jb + 1
        out += T[last:].tolist()
        Tn = np.asarray(out)
        e['trajectory_before_04g'] = T.round(5).tolist()
        e['trajectory'] = Tn.tolist()
        e['trajectory_z'] = Tn[:, 2].tolist()
        for key in ('trajectory_navmesh_z', 'floor_offset'):
            e.pop(key, None)
        # 계단 구간 다시: 원래 계단 점들에 가장 가까운 새 점 범위
        new_segs = []
        for s in e['segments']:
            if s['kind'] != 'stairs':
                continue
            a = int(np.argmin(np.linalg.norm(Tn[:, :2] - T[int(s['i0']), :2], axis=1)))
            b = int(np.argmin(np.linalg.norm(Tn[:, :2] - T[int(s['i1']), :2], axis=1)))
            new_segs.append({'kind': 'stairs', 'i0': min(a, b), 'i1': max(a, b), 'floor_z': s.get('floor_z')})
        segs, prev = [], 0
        for s in sorted(new_segs, key=lambda s: s['i0']):
            if s['i0'] > prev:
                segs.append({'kind': 'floor', 'i0': prev, 'i1': s['i0'] - 1, 'floor_z': float(Tn[prev, 2])})
            segs.append(s)
            prev = s['i1'] + 1
        if prev < len(Tn):
            segs.append({'kind': 'floor', 'i0': prev, 'i1': len(Tn) - 1, 'floor_z': float(Tn[prev, 2])})
        e['segments'] = segs
        e['path_length_m'] = float(np.linalg.norm(np.diff(Tn[:, :2], axis=0), axis=1).sum())
        changed.append(e['episode_id'])
    d['stairs_straight_source'] = {'args': vars(args), 'changed': changed}
    p_out = Path(args.paths_json).with_name(Path(args.paths_json).stem.rsplit('_v', 1)[0] + f'_{args.out_tag}.json')
    p_out.write_text(json.dumps(d))
    E = json.loads(Path(args.eps_json).read_text())
    init = r2r_initial(args.r2r_raw)
    by = {str(e['episode_id']): e for e in d['episodes']}
    sub = []
    for i, x in enumerate(E['episodes']):
        k = x['key'].split(':')[1]
        if int(k) in changed:
            T = np.asarray(by[k]['trajectory'], np.float64)
            E['episodes'][i] = dict(x, waypoints=resample(T).tolist(), start_yaw=start_yaw(by[k], T, args.start_seed, init))
            sub.append(E['episodes'][i])
    e_out = Path(args.eps_json).with_name(f'eps_noeun_{args.out_tag}.json')
    e_out.write_text(json.dumps(E))
    Path(args.eps_json).with_name(f'eps_noeun_{args.out_tag}_changed.json').write_text(json.dumps(dict(E, episodes=sub)))
    for e in d['episodes']:
        if 'stairs_straight' in e and (ids is None or e['episode_id'] in ids):
            print(f"[04g] ep {e['episode_id']}: {e['stairs_straight']}")
    print(f'[04g] 바뀐 ep {changed} -> {p_out} · {e_out} · eps_noeun_{args.out_tag}_changed.json')


if __name__ == '__main__':
    main()
