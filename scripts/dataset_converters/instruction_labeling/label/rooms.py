"""MP3D `.house` 방(region) 정보로 경로가 지나는 방 순서를 구한다 (route.rooms 옵션).

GT 지시문은 쓰지 않는다 — scene annotation(방 종류·바닥 다각형)과 reference_path 만 쓴다.
.house 형식: `R region level 0 0 label px py pz xlo ylo zlo xhi yhi zhi height ...`
             `S surface region 0 F px py pz ...` (바닥면) / `V vertex surface F px py pz ...` (바닥 다각형 꼭짓점, 순서대로)
좌표는 mesh Z-up. habitat(Y-up) 점 (x, y, z) -> mesh (x, -z, y).
점이 속한 방 = 바닥 다각형이 (x, y) 를 포함하고 바닥 높이가 점의 바닥 높이와 `max_dz_m` 안인 방 중 가장 작은 것.
"""

from functools import lru_cache

import numpy as np
from matplotlib.path import Path as MplPath


@lru_cache(maxsize=64)
def load_regions(house_path: str) -> list:
    regions, surf_region, verts = {}, {}, {}
    for line in open(house_path):
        t = line.split()
        if not t:
            continue
        if t[0] == 'R':
            regions[int(t[1])] = {'label': t[5], 'level': int(t[2])}
        elif t[0] == 'S':
            surf_region[int(t[1])] = int(t[2])
        elif t[0] == 'V':
            verts.setdefault(int(t[2]), []).append((float(t[4]), float(t[5]), float(t[6])))
    out = []
    for s, ri in surf_region.items():
        v = np.array(verts.get(s, []))
        if ri not in regions or len(v) < 3:
            continue
        xy = v[:, :2]
        area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))
        out.append({'region': ri, 'label': regions[ri]['label'], 'floor_z': float(v[:, 2].mean()),
                    'poly': MplPath(xy), 'area': area})
    return out


def room_at(regions: list, p_hab, max_dz_m: float):
    x, z_up, y = p_hab[0], p_hab[1], -p_hab[2]
    best = None
    for r in regions:
        if abs(r['floor_z'] - z_up) <= max_dz_m and r['poly'].contains_point((x, y)):
            if best is None or r['area'] < best['area']:
                best = r
    return best


def densify(path, step_m):
    out = []
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_m)))
        out += [a + (b - a) * k / n for k in range(n)]
    out.append(np.asarray(path[-1], float))
    return out


def room_sequence(ep: dict, house_path: str, rc: dict) -> dict:
    """{'rooms': [이름, ...] (연속 중복·짧은 통과 제거), 'coverage': 방이 잡힌 점 비율, 'raw': 점별 label}"""
    regions = load_regions(house_path)
    pts = densify(ep['reference_path'], rc['step_m'])
    labels = []
    for p in pts:
        r = room_at(regions, p, rc['max_dz_m'])
        labels.append((r['region'], r['label']) if r else None)
    cover = sum(x is not None for x in labels) / max(1, len(labels))
    runs = []  # (region, label, n_points)
    for x in labels:
        if x is None:
            continue
        if runs and runs[-1][0] == x[0]:
            runs[-1][2] += 1
        else:
            runs.append([x[0], x[1], 1])
    min_n = max(1, int(round(rc['min_visit_m'] / rc['step_m'])))
    kept = [r for i, r in enumerate(runs) if r[2] >= min_n or i in (0, len(runs) - 1)]  # 처음·끝 방은 짧아도 둔다
    names = []
    for _, lab, _ in kept:
        nm = rc['names'].get(lab)
        if nm and (not names or names[-1] != nm):
            names.append(nm)
    return {'rooms': names, 'coverage': round(cover, 3), 'raw': [x[1] if x else None for x in labels]}


def rooms_line(ep: dict, house_path: str, rc: dict) -> tuple:
    seq = room_sequence(ep, house_path, rc)
    if not seq['rooms']:
        return '', seq
    return rc['line'].format(rooms=rc['joiner'].join(seq['rooms']), goal=seq['rooms'][-1]), seq
