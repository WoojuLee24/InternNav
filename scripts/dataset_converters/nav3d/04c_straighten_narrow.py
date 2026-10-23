"""04c — 좁은 통로(개찰구 등) 안에서는 경로가 꺾이지 않게 곧게 편다 (04b paths json → 새 json).

왜: 노은역 RBQ 200 ep (06k) 에서 넘어짐 52 건 중 35 건이 대합실 한 곳 (x 14.5~17.5, y ≈ −2) 에서 났다.
그곳은 바닥 +0.05~0.5 m 에서 양옆 mesh 가 0.51 / 0.36 m — 폭 ≈ 0.87 m 통로 (개찰구로 보임) 이고,
경로가 그 안에서 꺾여 RBQ 가 통로 안에서 제자리 회전하다 다리가 걸려 넘어졌다.

규칙
1. 경로 점마다 진행 방향의 왼쪽·오른쪽으로 수평 광선 (바닥 + `--probe_h` 높이들) → 통로 폭 = 왼쪽 + 오른쪽 거리 (높이 중 최소).
2. 폭 < `--narrow_m` 인 점이 이어진 구간 = 좁은 구간. 구간 축 = 시작점 → 끝점.
3. 구간을 축 방향으로 앞 `--pre_m` · 뒤 `--post_m` 늘린 직선으로 바꾼다 (회전은 통로 앞·뒤에서).
   새 직선의 모든 점이 mesh 에서 `--min_clear_m` 이상 떨어져 있을 때만 바꾸고, 아니면 원래 경로를 둔다.
계단 구간(segments kind=stairs)과 겹치는 좁은 구간은 건드리지 않는다.

출력 episode 마다 `narrow` = [{i0, i1, width_min, applied, why}] 기록.

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/04c_straighten_narrow.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from floor_mesh import stairs_mask_from_segments  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths_json', default='scripts/dataset_converters/nav3d/logs/04_export_paths/noeun_station/noeun_station_paths_meshfloor.json')
    ap.add_argument('--mesh', default='data/scene_data/noeun_station/noeun_station_mid.ply')
    ap.add_argument('--out', default='scripts/dataset_converters/nav3d/logs/04_export_paths/noeun_station/noeun_station_paths_meshfloor_straight.json')
    ap.add_argument('--narrow_m', type=float, default=1.0, help='이 폭보다 좁으면 좁은 구간 (개찰구 0.87 m)')
    ap.add_argument('--probe_h', default='0.15,0.4', help='바닥 위 광선 높이 (다리 높이)')
    ap.add_argument('--pre_m', type=float, default=1.0, help='통로 앞 직선 길이 (그 전에 돌아서 들어가게)')
    ap.add_argument('--post_m', type=float, default=0.5)
    ap.add_argument('--min_clear_m', type=float, default=0.2, help='새 직선이 mesh 에서 떨어져야 할 최소 거리 (다리 높이)')
    ap.add_argument('--max_ray_m', type=float, default=3.0)
    return ap.parse_args()


def side_dists(mesh, P, H, hs, max_ray):
    """경로 점마다 (왼쪽, 오른쪽) 수평 거리 — 높이들 중 최소."""
    d = np.gradient(P[:, :2], axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(n, 1e-9)
    left = np.c_[-d[:, 1], d[:, 0], np.zeros(len(P))]
    L = np.full(len(P), max_ray)
    R = np.full(len(P), max_ray)
    for h in hs:
        O = np.c_[P[:, :2], H + h]
        for D, out in ((left, L), (-left, R)):
            loc, idx, _ = mesh.ray.intersects_location(O, D, multiple_hits=False)
            dist = np.linalg.norm(loc[:, :2] - O[idx, :2], axis=1)
            out[idx] = np.minimum(out[idx], np.minimum(dist, max_ray))
    return L, R, left[:, :2]


def clearance(mesh, Q, hs):
    """점마다 다리 높이에서 mesh 까지 수평 최소 거리 (16 방향 광선)."""
    ang = np.radians(np.arange(0, 360, 22.5))
    D = np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)]
    out = np.full(len(Q), np.inf)
    for i, q in enumerate(Q):
        for h in hs:
            O = np.tile([q[0], q[1], q[2] + h], (len(D), 1))
            loc, idx, _ = mesh.ray.intersects_location(O, D, multiple_hits=False)
            if len(idx):
                out[i] = min(out[i], float(np.linalg.norm(loc[:, :2] - O[idx, :2], axis=1).min()))
    return out


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def main():
    import trimesh
    args = parse()
    hs = [float(x) for x in args.probe_h.split(',')]
    mesh = trimesh.load(args.mesh, force='mesh')
    d = json.loads(Path(args.paths_json).read_text())
    n_ep, n_app, n_skip = 0, 0, 0
    for e in d['episodes']:
        T = np.asarray(e['trajectory'], dtype=np.float64)
        st = stairs_mask_from_segments(len(T), e['segments'])
        Lw, Rw, LEFT = side_dists(mesh, T, T[:, 2], hs, args.max_ray_m)
        width = Lw + Rw
        rec = []
        Tn = T.copy()
        keep_idx = np.ones(len(T), bool)
        extra = []                                            # (삽입 위치, 점들)
        for i0, i1 in runs(width < args.narrow_m):
            if st[max(0, i0 - 5):i1 + 6].any():
                rec.append({'i0': i0, 'i1': i1, 'width_min': float(width[i0:i1 + 1].min()), 'applied': False, 'why': 'stairs'})
                continue
            # 통로 가운데 선: 점마다 왼·오른 벽의 중간으로 옮긴 뒤 직선 맞춤 (주성분), 진행 방향으로
            C = T[i0:i1 + 1, :2] + LEFT[i0:i1 + 1] * ((Lw[i0:i1 + 1] - Rw[i0:i1 + 1]) / 2.0)[:, None]
            mu = C.mean(0)
            if len(C) >= 3:
                ax = np.linalg.svd(C - mu)[2][0]
            else:
                ax = T[min(i1 + 3, len(T) - 1), :2] - T[max(i0 - 3, 0), :2]
            ax = ax / max(np.linalg.norm(ax), 1e-9)
            if np.dot(ax, T[min(i1 + 1, len(T) - 1), :2] - T[max(i0 - 1, 0), :2]) < 0:
                ax = -ax
            t = (C - mu) @ ax
            pa, pb = mu + (t.min() - args.pre_m) * ax, mu + (t.max() + args.post_m) * ax
            # 원래 경로에 다시 붙일 점: pa 보다 앞 · pb 보다 뒤에서 가장 가까운 점
            ja = int(np.argmin(np.linalg.norm(T[:i0 + 1, :2] - pa, axis=1)))
            jb = i1 + int(np.argmin(np.linalg.norm(T[i1:, :2] - pb, axis=1)))
            z = float(np.median(T[ja:jb + 1, 2]))
            knots = np.array([[*T[ja, :2], T[ja, 2]], [*pa, z], [*pb, z], [*T[jb, :2], T[jb, 2]]])
            seg = [knots[0]]
            for u, v in zip(knots[:-1], knots[1:]):
                m = max(1, int(np.ceil(np.linalg.norm(v[:2] - u[:2]) / 0.1)))
                seg += [u + (v - u) * (k / m) for k in range(1, m + 1)]
            seg = np.asarray(seg)
            c = clearance(mesh, seg, hs)
            if c.min() < args.min_clear_m:
                rec.append({'i0': i0, 'i1': i1, 'width_min': float(width[i0:i1 + 1].min()), 'applied': False,
                            'why': f'직선 여유 {c.min():.2f} < {args.min_clear_m}'})
                n_skip += 1
                continue
            keep_idx[ja:jb + 1] = False
            extra.append((ja, seg))
            rec.append({'i0': i0, 'i1': i1, 'ja': ja, 'jb': jb, 'width_min': float(width[i0:i1 + 1].min()), 'applied': True,
                        'clear_min': float(c.min())})
            n_app += 1
        if extra:
            # 원래 점 순서대로 가면서, 바뀐 구간(ja..jb)은 그 자리에 직선 seg 를 넣는다
            out, k = [], 0
            starts = {ja: seg for ja, seg in extra}
            while k < len(T):
                if k in starts:
                    out.extend(starts[k].tolist())
                    k += 1
                    while k < len(T) and not keep_idx[k]:
                        k += 1
                    continue
                if keep_idx[k]:
                    out.append(T[k].tolist())
                k += 1
            Tn = np.asarray(out)
            e['trajectory_before_04c'] = T.round(5).tolist()
            e['trajectory'] = Tn.tolist()
            e['trajectory_z'] = Tn[:, 2].tolist()
            e['segments'] = [s for s in e['segments']]      # 계단 구간은 안 건드렸으므로 i0/i1 을 새 인덱스로 옮긴다
            for s in e['segments']:
                for key in ('i0', 'i1'):
                    p = T[int(s[key])]
                    s[key] = int(np.argmin(np.linalg.norm(Tn - p, axis=1)))
            n_ep += 1
        e['narrow'] = rec
    d['narrow_source'] = {'from': args.paths_json, 'args': vars(args)}
    Path(args.out).write_text(json.dumps(d))
    tot = sum(len(e['narrow']) for e in d['episodes'])
    print(f'[04c] 좁은 구간 {tot} (계단 겹침 {sum(r["why"] == "stairs" for e in d["episodes"] for r in e["narrow"] if not r["applied"])}) · '
          f'곧게 편 구간 {n_app} · 여유 부족으로 둔 구간 {n_skip} · 바뀐 경로 {n_ep}/{len(d["episodes"])} -> {args.out}')


if __name__ == '__main__':
    main()
