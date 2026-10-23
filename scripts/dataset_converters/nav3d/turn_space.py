"""RBQ 제자리 회전 자리 검사 (04d U6·U7 · 04e 공용).

RBQ 몸 ≈ 1.0 × 0.5 m → 제자리에서 돌면 중심에서 몸 끝까지 √(0.5² + 0.25²) ≈ 0.56 m 원을 쓸고 지나간다.
navmesh r_b (0.25 m) 는 옆 폭만 보장하므로, 회전하는 자리(출발점 · 경로 중간 회전 지점)는 따로 잰다.

- `turn_points`: 06k 의 action 규칙(경유점 방향과 `thr` 넘게 어긋나면 15° 제자리 회전, 아니면 0.25 m 전진 + steer_gain 방향 보정)을
  운동학으로 재생해 제자리 회전이 일어나는 자리를 낸다 (물리 없음).
- `turn_space_ok`: 그 자리에서 (1) 반지름 `disk_m` 원 안 바닥이 평평한가 (아래 광선 hit 가 모두 있고 중심 ± `dz_m`)
  (2) 다리·몸 높이 (`probe_h`) 16 방향 수평 광선이 `disk_m` 안에서 mesh 에 닿지 않는가.
"""

import math

import numpy as np

FWD, TURN = 0.25, math.radians(15.0)


def turn_points(W, start_yaw, thr_deg=30.0, steer_gain=1.5, fwd_speed=0.5, max_actions=600):
    """W (N,3) 0.5 m 경유점 · start_yaw rad -> 제자리 회전이 일어나는 자리 (K,3) (같은 자리 연속 회전은 한 번)."""
    W = np.asarray(W, np.float64)
    pos, yaw, wi, out = W[0].copy(), float(start_yaw), 1, []
    thr = math.radians(thr_deg)
    k = 1.0 - math.exp(-steer_gain * FWD / fwd_speed) if steer_gain > 0 else 0.0
    for _ in range(max_actions):
        while wi < len(W) and np.linalg.norm(W[wi, :2] - pos[:2]) < FWD * 0.6:
            wi += 1
        if wi >= len(W):
            break
        dp = W[wi, :2] - pos[:2]
        d = (math.atan2(dp[1], dp[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(d) > thr:
            if not out or np.linalg.norm(out[-1][:2] - pos[:2]) > 1e-6:
                out.append(pos.copy())
            yaw += TURN * (1 if d > 0 else -1)
            continue
        y_mid = yaw + 0.5 * k * d
        yaw += k * d
        pos = pos + np.array([FWD * math.cos(y_mid), FWD * math.sin(y_mid), 0.0])
        j = min(wi, len(W) - 1)
        pos[2] = W[j, 2]
    return np.asarray(out).reshape(-1, 3)


def turn_space_ok(mesh, p, disk_m=0.6, dz_m=0.1, probe_h=(0.15, 0.4), probe_up=1.0):
    """p = 바닥 위 점 (x, y, 바닥 z) -> (통과, {'floor_bad': 원 안 바닥 나쁜 점 수, 'clear': 수평 최소 거리 m})."""
    g = np.arange(-disk_m, disk_m + 1e-6, 0.1)
    G = np.array([(x, y) for x in g for y in g if x * x + y * y <= disk_m * disk_m + 1e-9])
    O = np.c_[p[0] + G[:, 0], p[1] + G[:, 1], np.full(len(G), p[2] + probe_up)]
    loc, idx, _ = mesh.ray.intersects_location(O, np.tile([0, 0, -1.0], (len(O), 1)), multiple_hits=False)
    z = np.full(len(O), np.nan)
    z[idx] = loc[:, 2]
    c = int(np.argmin(np.linalg.norm(G, axis=1)))
    zc = z[c] if np.isfinite(z[c]) else p[2]
    floor_bad = int((~np.isfinite(z) | (np.abs(z - zc) > dz_m)).sum())
    ang = np.radians(np.arange(0, 360, 22.5))
    D = np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)]
    clear = np.inf
    for h in probe_h:
        Oc = np.tile([p[0], p[1], zc + h], (len(D), 1))
        lc, ic, _ = mesh.ray.intersects_location(Oc, D, multiple_hits=False)
        if len(ic):
            clear = min(clear, float(np.linalg.norm(lc[:, :2] - Oc[ic, :2], axis=1).min()))
    return floor_bad == 0 and clear >= disk_m, {'floor_bad': floor_bad, 'clear': clear}
