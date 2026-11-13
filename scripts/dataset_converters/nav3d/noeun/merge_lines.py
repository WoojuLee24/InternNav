"""noeun · 한 줄로 이어진 물체(스크린도어)를 **줄마다 물체 하나**로 합친다 (사용자 결정 2026-11-13).

왜: 승강장 스크린도어는 벽처럼 이어져 있는데 69 는 조각 30 여 개로 잡았다 ("몇 번째 스크린도어" 로 못 가리킨다).
그 줄의 열리는 부분(doorway)은 건드리지 않는다 — 이름 그대로 따로 남는다 (사용자 결정).

corrections.json:
    "merge_lines": {"screen doors": {"gap_m": 12, "perp_m": 1.5, "mask_points": true}}
  gap_m       줄 방향으로 이만큼 떨어져도 같은 줄 (검출이 빠진 구간이 최대 11 m)
  perp_m      줄과 직각 방향 차이가 이 안이어야 같은 줄 (섬식 승강장 양쪽 줄은 약 7 m 떨어져 따로)
  mask_points true 면 상자 점 = 구성원 근처 69 마스크 픽셀 전부를 메쉬에 쏜 점 (검출 중심만 쓰면 높이가 가운데만 잡힌다)
  mask_perp_m 그 점 중 줄 면에서 이 안만 (기본 0.5 m)
줄 방향 = 그 층 그 이름 랜드마크마다 가장 가까운 이웃 방향을 모은 주축 (line_axis). 합친 물체 점 = 구성원들의 검출 점(70 과 같은 규칙) + 사람이 지정한 점.
"""

import math
from pathlib import Path

import numpy as np

ID0 = 300000          # 합친 물체 번호 (69 0~ · 화면 추가 100000~ · 계단 끝 200000~ 와 안 겹침)


# 69 마스크를 3D 점으로 (69_scene_landmarks.py 와 같은 카메라: 640x480, 수평 시야 68.7°)
W69, H69, HFOV69 = 640, 480, 68.7
_MK = {}


def mask_points(names, near, masks='logs/scene_landmarks_masks', snaps='logs/scene_scan/snaps.jsonl',
                mesh='data/data/noeun_eval/navmesh/noeun_station/noeun_station.glb', merge_m=1.5, per_det=200):
    """69 --save_masks 검출 중 이름이 names 이고 near 점들 merge_m 안인 것의 **마스크 픽셀 전부**를 메쉬에 쏜 3D 점.
    검출 xyz 는 마스크 중앙값 하나라서 높이가 물체 가운데만 잡힌다 (스크린도어 상자가 바닥 위 0.1~1.5 m 로 나왔다)."""
    import json
    from PIL import Image
    if 'ri' not in _MK:
        import trimesh
        _MK['ri'] = trimesh.load(mesh, force='mesh', process=False, skip_materials=True).ray
        _MK['snap'] = {json.loads(l)['id']: json.loads(l) for l in open(snaps)}
        _MK['det'] = [d for d in json.load(open(Path(masks) / 'detections.json')) if d.get('mask')]
    fx = (W69 / 2) / math.tan(math.radians(HFOV69) / 2)
    near = np.asarray(near, float).reshape(-1, 3)
    out = []
    for d in _MK['det']:
        if d['name'] not in names or np.linalg.norm(near - np.asarray(d['xyz']), axis=1).min() > merge_m:
            continue
        vs, us = np.nonzero(np.asarray(Image.open(Path(masks) / 'masks' / d['mask'])) > 0)
        if len(us) > per_det:
            k = np.random.RandomState(0).choice(len(us), per_det, replace=False)
            us, vs = us[k], vs[k]
        c2w = np.asarray(_MK['snap'][d['snap']]['c2w'])
        ray = np.stack([(us - (W69 - 1) / 2) / fx, (vs - (H69 - 1) / 2) / fx, np.ones(len(us))], 1) @ c2w[:3, :3].T
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        loc, _, _ = _MK['ri'].intersects_location(np.repeat([c2w[:3, 3]], len(ray), 0), ray, multiple_hits=False)
        out += list(loc)
    return np.array(out).reshape(-1, 3)


def line_axis(X):
    """평면 점들 -> 줄 방향. 전체 PCA 는 평행한 두 줄(섬식 승강장)이 길이가 다르면 기운다 ->
    점마다 **가장 가까운 이웃으로 가는 방향**을 모아 (방향 텐서) 가장 큰 축."""
    if len(X) < 2:
        return np.array([1.0, 0.0])
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    np.fill_diagonal(D, np.inf)
    V = X[D.argmin(1)] - X
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    w, E = np.linalg.eigh(V.T @ V)
    return E[:, -1]


def inject(lms, c, dets, storey_of, merge_m=1.5, log=print):
    """corrections "merge_lines" 가 있으면 줄마다 합친 **새 목록**. 없으면 lms 그대로 (기존 동작)."""
    cfg = (c or {}).get('merge_lines') or {}
    if not cfg:
        return lms
    by_name = {}
    for d in dets:
        by_name.setdefault(d['name'], []).append(np.asarray(d['xyz'], float))
    out, k = list(lms), 0
    for name, p in sorted(cfg.items()):
        gap, perp = float(p.get('gap_m', 12.0)), float(p.get('perp_m', 1.5))
        mem = [l for l in out if l['name'] == name]
        groups = []
        for st in sorted({storey_of(l['xyz'][2]) for l in mem}):
            L = [l for l in mem if storey_of(l['xyz'][2]) == st]
            X = np.array([l['xyz'][:2] for l in L], float)
            ax = line_axis(X)
            nv = np.array([-ax[1], ax[0]])
            par = list(range(len(L)))

            def find(i):
                while par[i] != i:
                    par[i] = par[par[i]]
                    i = par[i]
                return i
            for i in range(len(L)):
                for j in range(i + 1, len(L)):
                    d = X[j] - X[i]
                    if abs(d @ ax) <= gap and abs(d @ nv) <= perp:
                        par[find(i)] = find(j)
            g = {}
            for i in range(len(L)):
                g.setdefault(find(i), []).append(L[i])
            groups += [(st, ax, sorted(v, key=lambda l: l['xyz'][0] * ax[0] + l['xyz'][1] * ax[1])) for v in g.values()]
        merged = []
        for st, ax, G in sorted(groups, key=lambda t: (t[0], t[2][0]['xyz'][0])):
            if len(G) < 2:
                continue
            pts = []
            for l in G:
                c0 = np.asarray(l['xyz'], float)
                pts += [q for q in by_name.get(l.get('orig_name', l['name']), []) if np.linalg.norm(q - c0) <= merge_m]
                pts += [np.asarray(q, float) for q in l.get('points') or []]
            P = np.array(pts) if pts else np.array([l['xyz'] for l in G], float)
            if p.get('mask_points'):          # 검출 중심 대신 마스크 전체 -> 높이·길이가 실제대로
                Q = mask_points({l.get('orig_name', l['name']) for l in G}, [l['xyz'] for l in G], merge_m=merge_m)
                nv = np.array([-ax[1], ax[0]])
                off = float(np.median(np.array([l['xyz'][:2] for l in G]) @ nv))
                Q = Q[np.abs(Q[:, :2] @ nv - off) <= float(p.get('mask_perp_m', 0.5))]   # 줄 면 밖(유리 너머 선로 등) 빼기
                if len(Q) >= 20:
                    P = np.r_[Q, np.array([q for l in G for q in l.get('points') or []]).reshape(-1, 3)]
            names = []
            for l in G:
                names += [n for n in l.get('names', [l['name']]) if n not in names]
            new = {'id': ID0 + k, 'name': name, 'names': names, 'xyz': np.median(P, 0).round(3).tolist(),
                   'points': P.round(3).tolist(), 'views': sum(l.get('views') or 0 for l in G),
                   'score': max(l.get('score') or 0 for l in G), 'n_det': sum(l.get('n_det') or 0 for l in G),
                   'merged_ids': [l['id'] for l in G], 'line_axis': ax.round(4).tolist(), 'storey': st}
            if any(l.get('common') for l in G):
                new['common'] = next(l['common'] for l in G if l.get('common'))
            k += 1
            merged.append((new, G))
        gone = {l['id'] for _, G in merged for l in G}
        out = [l for l in out if l['id'] not in gone] + [m for m, _ in merged]
        log(f'[줄] {name}: {len(gone)} 개 -> 줄 {len(merged)} 개 (' + ', '.join(f"#{m['id']} {len(G)}개" for m, G in merged)
            + ')')
    return out


def _selftest():
    """두 줄 (y=0 · y=7) 에 띄엄띄엄 문 + 줄 위 doorway -> 줄 2 개, doorway 는 그대로."""
    L = [{'id': i, 'name': 'screen doors', 'xyz': [x, y, 0.0]} for i, (x, y) in
         enumerate([(0, 0), (2, 0.2), (13, -0.1), (20, 0), (0, 7), (9, 7.3)])]
    L += [{'id': 10, 'name': 'doorway', 'xyz': [5, 0.3, 0.0]}, {'id': 11, 'name': 'doorway', 'xyz': [5, 3.5, 0.0]}]
    out = inject(L, {'merge_lines': {'screen doors': {'gap_m': 12, 'perp_m': 1.5}}}, [],
                 lambda z: 0, log=lambda *x: None)
    M = [l for l in out if l.get('merged_ids')]
    assert sorted(len(m['merged_ids']) for m in M) == [2, 4], M
    d = {l['id']: l for l in out}
    assert d[10] == L[6] and d[10]['name'] == 'doorway' and 'names' not in d[10], d[10]
    assert not any(l['name'] == 'screen doors' and not l.get('merged_ids') for l in out)
    print('merge_lines selftest ok')


if __name__ == '__main__':
    _selftest()
