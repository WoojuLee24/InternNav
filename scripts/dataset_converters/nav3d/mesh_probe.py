"""충돌 mesh 에 광선을 쏴서 경로 주변을 재는 함수 (04c · 04d · 04f · 04g 가 같이 쓴다).

`04c_straighten_narrow.py` (side_dists) 와 `04d_replace_episodes.py` (start_flat) 에서 옮겼다 (코드 그대로).
"""

import numpy as np


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


def start_flat(mesh, p, disk_m, dz_m, probe_up=1.0):
    """U6: 출발점 중심 반지름 disk_m 원 안 0.1 m 격자의 아래 광선 hit 높이 − 중심 hit 높이. (통과, 나쁜 점 수, 점 수)."""
    g = np.arange(-disk_m, disk_m + 1e-6, 0.1)
    G = np.array([(x, y) for x in g for y in g if x * x + y * y <= disk_m * disk_m + 1e-9])
    O = np.c_[p[0] + G[:, 0], p[1] + G[:, 1], np.full(len(G), p[2] + probe_up)]
    loc, idx, _ = mesh.ray.intersects_location(O, np.tile([0, 0, -1.0], (len(O), 1)), multiple_hits=False)
    z = np.full(len(O), np.nan)
    z[idx] = loc[:, 2]
    c = int(np.argmin(np.linalg.norm(G, axis=1)))
    if not np.isfinite(z[c]):
        return False, len(G), len(G)
    bad = int((~np.isfinite(z) | (np.abs(z - z[c]) > dz_m)).sum())
    return bad == 0, bad, len(G)
