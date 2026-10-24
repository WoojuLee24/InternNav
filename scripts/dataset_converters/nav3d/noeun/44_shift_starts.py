"""04f — 출발 자리 검사(U6)에 걸린 ep 의 **출발점만** 경로를 따라 앞으로 옮긴다 (경로·도착은 그대로, ep 를 지우거나 바꾸지 않는다).

왜: 06k v5 (200/200 성공) 에서 출발 직후 경로 이탈 상위 5 ep 가 모두 U6 실패 출발점이었다
(U6 통과 169 ep 최대 0.31 m vs 실패 31 ep 최대 0.65 m). 출발점 옆 턱·물체에 몸이 밀려 제자리 회전 중 옆으로 미끄러진다.

규칙
- U6 (04d `start_flat`): 출발점 중심 반지름 `--disk_m` 원 안 바닥이 출발 높이 ± `--dz_m`.
- U6 실패 ep: 경로 점을 처음부터 따라가며 U6 를 통과하는 첫 점을 찾아 그 점을 새 출발점으로 하고, 그 앞 점들을 잘라낸다.
  `--max_shift_m` (경로 길이) 안에서 못 찾으면 그대로 두고 기록한다.
- 계단 구간(segments)은 잘린 만큼 인덱스를 옮긴다. 원래 출발점·잘린 길이는 `start_shift` 에 남긴다.
- 06k 입력 json 도 같이 만든다 (0.5 m 경유점 · 출발 yaw = 06 start_yaw_for r2r_train 규칙, 같은 episode_id seed → 경로 첫 방향 기준 상대각 그대로).

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/44_shift_starts.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
sys.path.insert(0, str(HERE))
from mesh_probe import start_flat  # noqa: E402
from walk_inputs import r2r_initial, resample, start_yaw  # noqa: E402
D04 = 'scripts/dataset_converters/nav3d/logs/noeun/paths'
D06K = 'scripts/dataset_converters/nav3d/logs/noeun/walk'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths_json', default=f'{D04}/noeun_station_paths_meshfloor_straight_v4.json')
    ap.add_argument('--eps_json', default=f'{D06K}/eps_noeun_v4.json', help='출발 yaw 를 다시 뽑지 않는 ep 의 원본 (U6 통과 ep 는 그대로 복사)')
    ap.add_argument('--mesh', default='data/scene_data/noeun_station/noeun_station_mid.ply')
    ap.add_argument('--disk_m', type=float, default=0.6)
    ap.add_argument('--dz_m', type=float, default=0.1)
    ap.add_argument('--max_shift_m', type=float, default=3.0)
    ap.add_argument('--start_seed', type=int, default=0)
    ap.add_argument('--r2r_raw', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r/train/train.json.gz')
    ap.add_argument('--out_tag', default='v5')
    return ap.parse_args()


def main():
    import trimesh
    args = parse()
    mesh = trimesh.load(args.mesh, force='mesh')
    d = json.loads(Path(args.paths_json).read_text())
    E = json.loads(Path(args.eps_json).read_text())
    eps_by_id = {e['key'].split(':')[1]: e for e in E['episodes']}
    init = r2r_initial(args.r2r_raw)
    moved, kept, stuck = [], 0, []
    for e in d['episodes']:
        T = np.asarray(e['trajectory'], np.float64)
        ok0, bad0, _ = start_flat(mesh, T[0], args.disk_m, args.dz_m)
        if ok0:
            kept += 1
            continue
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T[:, :2], axis=0), axis=1))]
        j = next((i for i in range(1, len(T)) if L[i] <= args.max_shift_m and start_flat(mesh, T[i], args.disk_m, args.dz_m)[0]), None)
        if j is None:
            stuck.append(e['episode_id'])
            e['start_shift'] = {'applied': False, 'why': f'{args.max_shift_m} m 안에 U6 통과 점 없음', 'bad0': bad0}
            continue
        e['start_shift'] = {'applied': True, 'from': T[0].tolist(), 'cut_points': j, 'cut_m': float(L[j]), 'bad0': bad0}
        e['trajectory'] = T[j:].tolist()
        e['trajectory_z'] = T[j:, 2].tolist()
        e['start'] = T[j].tolist()
        for key in ('trajectory_navmesh_z', 'floor_offset'):
            if key in e:
                e[key] = e[key][j:]
        segs = []
        for s in e['segments']:
            i0, i1 = int(s['i0']) - j, int(s['i1']) - j
            if i1 < 0:
                continue
            segs.append(dict(s, i0=max(0, i0), i1=i1))
        e['segments'] = segs
        e['path_length_m'] = float(e['path_length_m'] - L[j])
        moved.append((e['episode_id'], round(float(L[j]), 2), bad0))
    d['start_shift_source'] = {'args': vars(args), 'moved': len(moved), 'stuck': stuck}
    p_out = Path(args.paths_json).with_name(Path(args.paths_json).stem.replace('_v4', '') + f'_{args.out_tag}.json')
    p_out.write_text(json.dumps(d))
    # 06k 입력: 옮긴 ep 만 새 경유점·출발 yaw, 나머지는 원본 그대로
    out = []
    moved_ids = {str(i) for i, _, _ in moved}
    for e in d['episodes']:
        k = str(e['episode_id'])
        if k in moved_ids:
            T = np.asarray(e['trajectory'], np.float64)
            out.append(dict(eps_by_id[k], waypoints=resample(T).tolist(), start_yaw=start_yaw(e, T, args.start_seed, init)))
        else:
            out.append(eps_by_id[k])
    e_out = Path(args.eps_json).with_name(f'eps_noeun_{args.out_tag}.json')
    e_out.write_text(json.dumps(dict(E, episodes=out)))
    Path(args.eps_json).with_name(f'eps_noeun_{args.out_tag}_moved.json').write_text(
        json.dumps(dict(E, episodes=[x for x in out if x['key'].split(':')[1] in moved_ids])))
    print(f'[04f] U6 통과 {kept} · 출발점 옮김 {len(moved)} (옮긴 길이 중앙 {np.median([m for _, m, _ in moved]) if moved else 0:.2f} m, 최대 '
          f'{max([m for _, m, _ in moved], default=0):.2f}) · 못 옮김 {stuck}')
    print(f'  -> {p_out} · {e_out} · 옮긴 ep 만: eps_noeun_{args.out_tag}_moved.json')


if __name__ == '__main__':
    main()
