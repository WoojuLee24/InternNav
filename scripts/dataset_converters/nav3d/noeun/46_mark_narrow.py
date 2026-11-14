"""noeun 46 — 걷기 입력(41/44 eps json)의 경유점마다 '좁은 통로 구간' 표시(`narrow_wp`)를 붙인다.

왜: RBQ(몸 1.0 x 0.5 m)가 개찰구(폭 ≈ 0.85 m) 안에서 15° 제자리 보정 회전을 하다 다리가 걸려 넘어진다
(2026-11-14, 개찰구 통과 경로 성공 59% vs 다른 경로 91%). 걷기(walk_rbq_worker `--narrow_only`)가 이 표시가 있는
경유점에서만 다른 규칙(걸으며 방향 보정 또는 더 큰 회전 기준)을 쓰게 해서, 나머지 구간의 이산 동작 라벨은 그대로 둔다.

규칙: 경유점에서 진행 방향 좌·우로 수평 광선(바닥 + `--probe_h`) → 폭 = 좌 + 우 (32 와 같은 side_dists).
폭 < `--narrow_m` 인 경유점 = 좁음. 앞뒤 `--pad_m` 만큼 넓힌다 (몸이 통로를 다 빠져나올 때까지).

사용 (habitat 컨테이너, 저장소 루트):
    python3 scripts/dataset_converters/nav3d/noeun/46_mark_narrow.py --eps_json <eps> --out <eps_narrow.json>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'utils'))
from mesh_probe import side_dists  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eps_json', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mesh', default='data/scene_data/noeun_station/noeun_station_mid.ply')
    ap.add_argument('--narrow_m', type=float, default=1.0, help='32 와 같음 (개찰구 0.87 m)')
    ap.add_argument('--probe_h', default='0.15,0.4')
    ap.add_argument('--pad_m', type=float, default=1.5, help='좁은 곳 앞뒤로 넓힐 길이 (32 --pre_m/--post_m 와 같게)')
    ap.add_argument('--max_ray_m', type=float, default=3.0)
    return ap.parse_args()


def main():
    import trimesh
    a = parse()
    hs = [float(x) for x in a.probe_h.split(',')]
    mesh = trimesh.load(a.mesh, force='mesh')
    d = json.loads(Path(a.eps_json).read_text())
    n_ep = n_wp = 0
    for e in d['episodes']:
        W = np.asarray(e['waypoints'], float)
        Lw, Rw, _ = side_dists(mesh, W, W[:, 2], hs, a.max_ray_m)
        nar = (Lw + Rw) < a.narrow_m
        if nar.any():
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(W[:, :2], axis=0), axis=1))]
            idx = np.flatnonzero(nar)
            m = np.zeros(len(W), bool)
            for i in idx:
                m |= np.abs(s - s[i]) <= a.pad_m
            e['narrow_wp'] = m.astype(int).tolist()
            n_ep += 1; n_wp += int(m.sum())
        else:
            e.pop('narrow_wp', None)
    Path(a.out).write_text(json.dumps(d))
    tot = sum(len(e['waypoints']) for e in d['episodes'])
    print(f'[46] 좁은 구간 있는 ep {n_ep}/{len(d["episodes"])} · 표시된 경유점 {n_wp}/{tot} ({n_wp / max(tot, 1):.1%}) -> {a.out}')


if __name__ == '__main__':
    main()
