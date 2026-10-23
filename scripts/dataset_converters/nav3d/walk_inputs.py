"""06k RBQ 걷기 입력(episode 마다 0.5 m 경유점 + 출발 yaw) 만들기 (04d · 04f · 04g · 06k_make_walk_inputs 가 같이 쓴다).

`04d_replace_episodes.py` 에서 옮겼다 (코드 그대로). 출발 yaw 규칙은 06 `start_yaw_for(r2r_train)` 과 같다.
"""

import gzip
import hashlib
import json
import math

import numpy as np


def resample(T, s=0.5):
    """06k 경유점: 경로를 수평 길이 0.5 m 간격으로 (eps_noeun_all_v2 와 같음 — 200/200 일치 확인)."""
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T[:, :2], axis=0), axis=1))]
    u = np.r_[np.arange(0, L[-1], s), L[-1]]
    return np.c_[[np.interp(u, L, T[:, k]) for k in range(3)]].T


def r2r_initial(raw_json):
    """06 start_yaw_for(r2r_train) 와 같은 분포 (R2R train 시작 방향 − 첫 구간, 도)."""
    out = []
    for e in json.load(gzip.open(raw_json))['episodes']:
        p, q = e['reference_path'][0], e['reference_path'][1]
        seg = math.degrees(math.atan2(q[0] - p[0], -(q[2] - p[2])))
        st = -math.degrees(2 * math.atan2(e['start_rotation'][1], e['start_rotation'][3]))
        out.append((seg - st + 180) % 360 - 180)
    return np.asarray(out)


def start_yaw(ep, T, seed, init):
    d = T[min(len(T) - 1, 5), :2] - T[0, :2]
    rng = np.random.default_rng(int(hashlib.sha1(f"{seed}:{ep['episode_id']}".encode()).hexdigest()[:8], 16))
    return float(np.arctan2(d[1], d[0]) + np.radians(float(rng.choice(init))))
