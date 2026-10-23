"""촘촘한 3D 경로 -> labeling 이 받는 '꺾인 점'(R2R reference_path 같은 waypoint) 만들기 (07 공용).

labeling 의 경로 단어(route_words)는 waypoint 사이 구간 방향 차로 회전을 말한다. R2R 경로는 2 m 남짓 간격의 꺾인 점이라 잘 맞지만,
nav3d 경로는 모퉁이를 곡선으로 도는 10 cm 간격 점이다. 그래서 점을 줄이는 방법이 경로 단어를 정한다.

| 방법 | 규칙 | 문제 |
|---|---|---|
| `rdp` (기존) | 직선에서 eps 이상 벗어나는 점만 남김 | 완만한 굽이는 점이 안 남아 회전이 사라지고, 큰 굽이는 점이 여럿 남아 "bear slightly" 여러 번으로 쪼개진다 |
| `turn` (새) | 0.25 m 간격으로 다시 찍고, `window_m` 동안 누적된 방향 변화가 `turn_deg` 이상인 구간을 회전 하나로 묶어 그 가운데에 점 하나. 계단(높이 변화) 시작·끝에도 점 | |

두 방법 모두 끝점은 그대로 둔다.
"""

import numpy as np


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


def resample_xy(P: np.ndarray, step: float) -> np.ndarray:
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
    if L[-1] < 1e-6:
        return P[[0, -1]]
    s = np.r_[np.arange(0, L[-1], step), L[-1]]
    return np.stack([np.interp(s, L, P[:, k]) for k in range(3)], 1)


def turn_vertices(P: np.ndarray, step: float = 0.25, window_m: float = 2.0, turn_deg: float = 25.0,
                  stairs_slope: float = 0.25) -> np.ndarray:
    """곡선 경로 -> 꺾인 점. 반환 (K,3), 첫·끝 점 포함."""
    Q = resample_xy(np.asarray(P, dtype=np.float64), step)
    if len(Q) < 3:
        return Q
    d = np.diff(Q[:, :2], axis=0)
    h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))          # 구간 방향 (len n-1)
    w = max(1, int(round(window_m / step)))
    n = len(h)
    # 구간 i 를 가운데로 하는 창의 누적 방향 변화
    lo, hi = np.clip(np.arange(n) - w // 2, 0, n - 1), np.clip(np.arange(n) + w // 2, 0, n - 1)
    cum = h[hi] - h[lo]
    turning = np.abs(cum) >= np.radians(turn_deg)
    keep = {0, len(Q) - 1}
    i = 0
    while i < n:                                          # 회전 구간마다: 방향 변화가 가장 큰 곳 = 회전 중심
        if turning[i]:
            j = i
            while j + 1 < n and turning[j + 1] and np.sign(cum[j + 1]) == np.sign(cum[i]):
                j += 1
            seg = np.arange(i, j + 1)
            dh = np.abs(np.gradient(h))[seg] if n > 1 else np.zeros(len(seg))
            c = int(seg[np.argmax(dh)]) if dh.max() > 0 else (i + j) // 2
            keep.add(c + 1)
            i = j + 1
        else:
            i += 1
    # 계단: 높이 기울기가 stairs_slope 이상인 구간의 시작·끝
    slope = np.abs(np.diff(Q[:, 2])) / np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    st = slope >= stairs_slope
    for k in range(1, n):
        if st[k] != st[k - 1]:
            keep.add(k)
    return Q[sorted(keep)]


def merge_turns(V: np.ndarray, merge_m: float = 2.0, min_deg: float = 10.0) -> np.ndarray:
    """rdp 꺾인 점에서, 같은 방향으로 도는 이웃 꺾인 점 사이가 merge_m 보다 짧으면 둘을 하나로 합친다
    (모퉁이를 둥글게 돈 −44°, −28° -> −72° 한 번). 합친 점 = 앞 구간 직선과 뒤 구간 직선의 교점(없으면 가운데)."""
    V = [np.asarray(v, dtype=np.float64) for v in V]
    changed = True
    while changed and len(V) > 3:
        changed = False
        for i in range(1, len(V) - 2):
            a, b, c, d = V[i - 1], V[i], V[i + 1], V[i + 2]
            if abs(c[2] - b[2]) > 0.3 or np.linalg.norm(c[:2] - b[:2]) >= merge_m:   # 계단 구간·먼 점은 합치지 않음
                continue
            h = [np.arctan2(*(q - p)[1::-1]) for p, q in ((a, b), (b, c), (c, d))]
            t1 = (h[1] - h[0] + np.pi) % (2 * np.pi) - np.pi
            t2 = (h[2] - h[1] + np.pi) % (2 * np.pi) - np.pi
            if np.sign(t1) != np.sign(t2) or min(abs(t1), abs(t2)) < np.radians(min_deg):
                continue
            u, w = b[:2] - a[:2], d[:2] - c[:2]
            den = u[0] * w[1] - u[1] * w[0]
            if abs(den) > 1e-9:
                s_ = ((c[0] - a[0]) * w[1] - (c[1] - a[1]) * w[0]) / den
                m = a[:2] + s_ * u
                if np.linalg.norm(m - (b[:2] + c[:2]) / 2) > merge_m:      # 거의 평행하면 교점이 멀다 -> 가운데
                    m = (b[:2] + c[:2]) / 2
            else:
                m = (b[:2] + c[:2]) / 2
            V = V[:i] + [np.r_[m, (b[2] + c[2]) / 2]] + V[i + 2:]
            changed = True
            break
    return np.stack(V)


def simplify(P: np.ndarray, method: str, rdp_eps: float = 0.3, merge_m: float = 2.0, **kw) -> np.ndarray:
    if method == 'rdp':
        return rdp(np.asarray(P, dtype=np.float64), rdp_eps)
    elif method == 'rdp_merge':
        return merge_turns(rdp(np.asarray(P, dtype=np.float64), rdp_eps), merge_m)
    elif method == 'turn':
        return turn_vertices(P, **kw)
    else:
        assert False, f'unreachable method={method!r}'
