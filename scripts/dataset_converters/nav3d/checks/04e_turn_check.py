"""04e — 06k 입력(경로 + 출발 yaw) 전체에 대해 RBQ 제자리 회전 자리를 검사한다 (turn_space.py).

## 판정 기준 (결과를 보기 전에 정함)
| | 묻는 것 | 기준 | 근거 |
|---|---|---|---|
| T0 재생 확인 | 운동학 재생 회전 자리가 실제 06k 회전 자리를 잡는가 | 실제 회전 자리(06k frame 중 action 2/3 직전 위치) 중 재생 자리 0.3 m 안에 있는 비율 ≥ 90 % | 재생이 틀리면 U7 이 엉뚱한 곳을 잰다 |
| U7 회전 자리 | 회전 자리마다 몸이 쓸 원(반지름 `--disk_m`) 안이 비고 평평한가 | 원 안 바닥 ±0.1 m · 높이 0.15/0.4 m 수평 여유 ≥ `--disk_m` | RBQ 1.0 × 0.5 m → 몸 끝 0.56 m |
출발점(첫 회전 자리)도 U7 에 포함된다 (04d U6 와 같은 자리 + 수평 여유).

출력: ep 마다 {key, n_turn, n_bad, bad: [{p, floor_bad, clear}]} json · 실패 ep 그림.

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/04e_turn_check.py --eps_json <06k 입력> --walk_json <06k 결과>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/ (이 파일은 nav3d/checks/ 에 있다)
sys.path.insert(0, str(HERE))
from turn_space import turn_points, turn_space_ok  # noqa: E402

D06K = 'scripts/dataset_converters/nav3d/logs/06k_rbq_walk'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eps_json', default=f'{D06K}/eps_noeun_v4.json')
    ap.add_argument('--walk_json', default=f'{D06K}/walk_noeun_v4.json', help='T0 용 실제 회전 자리 (없으면 T0 생략)')
    ap.add_argument('--mesh', default='data/scene_data/noeun_station/noeun_station_mid.ply')
    ap.add_argument('--disk_m', type=float, default=0.6, help='몸 끝 0.56 m + 여유')
    ap.add_argument('--thr_deg', type=float, default=30.0, help='06k --turn_threshold_deg 와 같게')
    ap.add_argument('--out', default=f'{D06K}/turn_check_v4.json')
    ap.add_argument('--png', default=f'{D06K}/turn_check_v4_bad.png')
    return ap.parse_args()


def main():
    import trimesh
    args = parse()
    mesh = trimesh.load(args.mesh, force='mesh')
    E = json.loads(Path(args.eps_json).read_text())['episodes']
    Wk = {r['key']: r for r in json.loads(Path(args.walk_json).read_text())['results']} if args.walk_json else {}
    rows, hit, tot = [], 0, 0
    for e in E:
        W = np.asarray(e['waypoints'])
        T = turn_points(W, e['start_yaw'], args.thr_deg)
        r = Wk.get(e['key'])
        if r is not None and r['success']:
            fr = r['frames']
            A = np.array([fr[i - 1]['pos'] for i in range(1, len(fr)) if fr[i]['action'] in (2, 3)])
            if len(A) and len(T):
                d = np.linalg.norm(A[:, None, :2] - T[None, :, :2], axis=2).min(1)
                hit += int((d <= 0.3).sum())
            tot += len(A)
        bad = []
        for p in T:
            ok, info = turn_space_ok(mesh, p, args.disk_m)
            if not ok:
                bad.append({'p': p.tolist(), **info})
        rows.append({'key': e['key'], 'n_turn': len(T), 'n_bad': len(bad), 'start_bad': bool(bad and np.allclose(bad[0]['p'], T[0])),
                     'bad': bad, 'turns': T.tolist()})
    Path(args.out).write_text(json.dumps({'args': vars(args), 'rows': rows}))
    if tot:
        print(f'T0 재생 확인: 실제 회전 자리 {tot} 중 재생 0.3 m 안 {hit} ({hit / tot:.1%}) -> {"PASS" if hit / tot >= 0.9 else "FAIL"}')
    nb = [x for x in rows if x['n_bad']]
    print(f'U7: 회전 자리 {sum(x["n_turn"] for x in rows)} · 걸리는 자리 {sum(x["n_bad"] for x in rows)} · '
          f'걸리는 ep {len(nb)}/{len(rows)} (그중 출발점 {sum(x["start_bad"] for x in nb)})')
    cl = np.array([b['clear'] for x in rows for b in x['bad']])
    fb = np.array([b['floor_bad'] for x in rows for b in x['bad']])
    if len(cl):
        print(f'  걸린 자리: 수평 여유 < {args.disk_m} {int((cl < args.disk_m).sum())} · 바닥 나쁨 {int((fb > 0).sum())} · 여유 분위 5/50/95 {np.percentile(cl, [5, 50, 95]).round(2)}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = nb[:12]
    if show:
        fig, axs = plt.subplots(2, 6, figsize=(24, 8))
        for ax, x in zip(axs.ravel(), show):
            b = np.array(x['bad'][0]['p'])
            sec = mesh.section(plane_origin=[0, 0, b[2] + 0.15], plane_normal=[0, 0, 1])
            if sec is not None:
                for seg in sec.discrete:
                    ax.plot(seg[:, 0], seg[:, 1], 'k-', lw=0.6)
            W = np.asarray(next(e for e in E if e['key'] == x['key'])['waypoints'])
            ax.plot(W[:, 0], W[:, 1], 'g-', lw=1)
            for q in x['bad']:
                ax.add_patch(plt.Circle(q['p'][:2], args.disk_m, color='r', fill=False))
            ax.set_xlim(b[0] - 2.5, b[0] + 2.5)
            ax.set_ylim(b[1] - 2.5, b[1] + 2.5)
            ax.set_aspect('equal')
            ax.set_title(f"{x['key']} 걸림 {x['n_bad']}/{x['n_turn']}", fontsize=8)
        fig.savefig(args.png, dpi=90)
        print('->', args.out, args.png)


if __name__ == '__main__':
    main()
