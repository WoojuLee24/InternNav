"""06k 출발 지점 검사 — 06k walk 결과의 모든 ep 에 대해 스폰 자리를 mesh 로 재고, 실패 ep 와 성공 ep 를 비교한다.

재는 것 (출발점 W[0] · 출발 yaw 기준)
- tilt0      : 06k frame 0 (스폰 + warmup 직후) 몸 기울기 [°]. 걷기 시작 전에 이미 기울었는지.
- floor_rng  : 몸 아래 (앞뒤 ±`--half_len` · 좌우 ±`--half_wid`, 0.1 m 격자) 아래로 쏜 광선의 첫 hit 높이 − W[0].z 의 [최소, 최대].
               최대 > 0.1 → 몸 아래에 바닥이 아닌 물체 (스폰 시 다리/몸이 물체에 박힘), 최소 < −0.1 → 몸 아래가 꺼짐 (턱·계단 끝).
- clear      : 바닥 + 0.15 / 0.4 m 에서 16 방향 수평 광선 최소 거리 [m] (몸 중심 기준).
- turn0      : 경로 첫 방향 − 출발 yaw [°] (출발 직후 제자리 회전량).

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/noeun_walk_start_check.py --walk_json <walk json> --episodes_json <eps json>
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--walk_json', default='scripts/dataset_converters/nav3d/logs/noeun/walk/walk_noeun_v3.json')
    ap.add_argument('--episodes_json', default='scripts/dataset_converters/nav3d/logs/noeun/walk/eps_noeun_all_v2.json')
    ap.add_argument('--mesh', default='data/scene_data/noeun_station/noeun_station_mid.ply')
    ap.add_argument('--half_len', type=float, default=0.5, help='몸 앞뒤 반길이 (RBQ10 ≈ 1.0 m)')
    ap.add_argument('--half_wid', type=float, default=0.25)
    ap.add_argument('--out', default='scripts/dataset_converters/nav3d/logs/noeun/walk/start_check_v3.json')
    ap.add_argument('--png', default='scripts/dataset_converters/nav3d/logs/noeun/walk/start_check_v3_fail.png')
    return ap.parse_args()


def tilt_deg(q):
    w, x, y, z = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def main():
    import trimesh
    args = parse()
    mesh = trimesh.load(args.mesh, force='mesh')
    W = json.loads(Path(args.walk_json).read_text())['results']
    E = {e['key']: e for e in json.loads(Path(args.episodes_json).read_text())['episodes']}
    ang = np.radians(np.arange(0, 360, 22.5))
    D16 = np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)]
    gx, gy = np.meshgrid(np.arange(-args.half_len, args.half_len + 1e-6, 0.1), np.arange(-args.half_wid, args.half_wid + 1e-6, 0.1))
    G = np.c_[gx.ravel(), gy.ravel()]
    rows = []
    for r in W:
        e = E[r['key']]
        P = np.asarray(e['waypoints'], np.float64)
        p0, yaw = P[0], float(e['start_yaw'])
        c, s = math.cos(yaw), math.sin(yaw)
        XY = p0[:2] + np.c_[c * G[:, 0] - s * G[:, 1], s * G[:, 0] + c * G[:, 1]]
        O = np.c_[XY, np.full(len(XY), p0[2] + 0.6)]
        loc, idx, _ = mesh.ray.intersects_location(O, np.tile([0, 0, -1.0], (len(O), 1)), multiple_hits=False)
        dz = np.full(len(O), -9.9)
        dz[idx] = loc[:, 2] - p0[2]
        cl = np.inf
        for h in (0.15, 0.4):
            Oc = np.tile([p0[0], p0[1], p0[2] + h], (len(D16), 1))
            lc, ic, _ = mesh.ray.intersects_location(Oc, D16, multiple_hits=False)
            if len(ic):
                cl = min(cl, float(np.linalg.norm(lc[:, :2] - Oc[ic, :2], axis=1).min()))
        k = next((i for i in range(1, len(P)) if np.linalg.norm(P[i, :2] - p0[:2]) > 0.3), len(P) - 1)
        h0 = math.atan2(P[k, 1] - p0[1], P[k, 0] - p0[0])
        rows.append({'key': r['key'], 'success': r['success'], 'why': r['why'], 'n_actions': len(r['frames']) - 1,
                     'tilt0': tilt_deg(r['frames'][0]['quat_wxyz']), 'floor_min': float(dz.min()), 'floor_max': float(dz.max()),
                     'clear': cl, 'turn0': math.degrees((h0 - yaw + math.pi) % (2 * math.pi) - math.pi),
                     'p0': p0.tolist(), 'yaw': yaw, 'grid_dz': dz.round(3).tolist()})
    Path(args.out).write_text(json.dumps(rows))
    F = [x for x in rows if not x['success']]
    S = [x for x in rows if x['success']]
    print(f'{"key":24s} {"actions":>7s} {"tilt0":>6s} {"floor_min":>9s} {"floor_max":>9s} {"clear":>6s} {"turn0":>6s}  why')
    for x in F:
        print(f'{x["key"]:24s} {x["n_actions"]:7d} {x["tilt0"]:6.1f} {x["floor_min"]:9.2f} {x["floor_max"]:9.2f} {x["clear"]:6.2f} {x["turn0"]:6.0f}  {x["why"]}')
    for name, f in (('tilt0 > 5°', lambda x: x['tilt0'] > 5), ('floor_max > 0.1', lambda x: x['floor_max'] > 0.1),
                    ('floor_min < -0.1', lambda x: x['floor_min'] < -0.1), ('clear < 0.3', lambda x: x['clear'] < 0.3),
                    ('|turn0| > 90°', lambda x: abs(x['turn0']) > 90)):
        print(f'{name:18s} 실패 {sum(map(f, F))}/{len(F)} · 성공 {sum(map(f, S))}/{len(S)}')
    print('성공 tilt0 분위수 (50/90/99/max):', np.percentile([x['tilt0'] for x in S], [50, 90, 99, 100]).round(1))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(F), figsize=(4 * len(F), 4))
    for ax, x in zip(np.atleast_1d(axs), F):
        p0, yaw = np.array(x['p0']), x['yaw']
        sec = mesh.section(plane_origin=[0, 0, p0[2] + 0.15], plane_normal=[0, 0, 1])
        if sec is not None:
            for seg in sec.discrete:
                ax.plot(seg[:, 0], seg[:, 1], 'k-', lw=0.6)
        c, s = math.cos(yaw), math.sin(yaw)
        XY = p0[:2] + np.c_[c * G[:, 0] - s * G[:, 1], s * G[:, 0] + c * G[:, 1]]
        sc = ax.scatter(XY[:, 0], XY[:, 1], c=np.clip(x['grid_dz'], -0.3, 0.3), cmap='coolwarm', vmin=-0.3, vmax=0.3, s=12)
        P = np.asarray(E[x['key']]['waypoints'])
        ax.plot(P[:30, 0], P[:30, 1], 'g-', lw=1)
        ax.arrow(p0[0], p0[1], 0.6 * c, 0.6 * s, color='m', width=0.03)
        ax.set_xlim(p0[0] - 2, p0[0] + 2)
        ax.set_ylim(p0[1] - 2, p0[1] + 2)
        ax.set_aspect('equal')
        ax.set_title(f'{x["key"]}\ntilt0 {x["tilt0"]:.0f}° floor {x["floor_min"]:.2f}~{x["floor_max"]:.2f}', fontsize=8)
    fig.colorbar(sc, ax=axs, label='hit z − start z [m]')
    fig.savefig(args.png, dpi=110)
    print('->', args.out, args.png)


if __name__ == '__main__':
    main()
