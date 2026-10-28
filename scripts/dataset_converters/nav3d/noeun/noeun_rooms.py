"""노은역 구역 순서 — instruction_labeling `label/rooms.room_sequence` 의 노은역판 (route.rooms=noeun-v1).

rooms.room_sequence 는 MP3D .house 방 다각형을 읽는다. 노은역에는 .house 가 없으므로 61_build_zones.py 가 만든
구역 점(경로 프레임 world 좌표 + 구역)을 쓰고, 경로 점마다 가장 가까운 구역 점의 구역을 고른다.
연속 구간 · 짧은 통과 제거 · 이름 매기기는 rooms.room_sequence 와 같은 규칙이다 (처음·끝 구역은 짧아도 둔다).

62_label.py 가 `rooms.room_sequence` 를 이 함수로 바꿔 끼운다 (기존 파일은 고치지 않는다).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[4]


@lru_cache(maxsize=4)
def load_zones(zones_file: str):
    import json
    d = json.loads((REPO / zones_file).read_text())
    P = np.asarray(d['points'], float)
    return cKDTree(P[:, :3]), P[:, 3].astype(int), d['zones']


def densify(path, step_m):
    out = []
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_m)))
        out += [a + (b - a) * k / n for k in range(n)]
    out.append(np.asarray(path[-1], float))
    return out


def room_sequence(ep: dict, house_path: str, rc: dict) -> dict:
    """rooms.room_sequence 와 같은 반환: {'rooms': [이름, ...], 'coverage': 구역이 잡힌 점 비율, 'raw': 점별 구역}. house_path 는 쓰지 않는다."""
    tree, zid, zones = load_zones(rc['zones_file'])
    pts = np.asarray(densify(ep['reference_path'], rc['step_m']))
    d, j = tree.query(pts)
    labels = [zones[zid[k]] if dist <= rc['max_dist_m'] else None for dist, k in zip(d, j)]
    cover = sum(x is not None for x in labels) / max(1, len(labels))
    runs = []  # [label, n_points]
    for x in labels:
        if x is None:
            continue
        if runs and runs[-1][0] == x:
            runs[-1][1] += 1
        else:
            runs.append([x, 1])
    by_label = rc.get('min_visit_by_label') or {}

    def min_n(lab):
        return max(1, int(round(by_label.get(lab, rc['min_visit_m']) / rc['step_m'])))

    kept = [r for i, r in enumerate(runs) if r[1] >= min_n(r[0]) or i in (0, len(runs) - 1)]
    names = []
    for lab, _ in kept:
        nm = rc['names'].get(lab)
        if nm and (not names or names[-1] != nm):
            names.append(nm)
    return {'rooms': names, 'coverage': round(cover, 3), 'raw': labels}
