"""noeun 73·74 공통 — 목적지가 **말로 하나로 가리켜지는가** 를 정하는 규칙 (코드로 계산, 사람이 정한 기준).

용어
- 종류(key): 공통 class 가 있으면 그것, 없으면 대표 이름. "같은 종류" 는 이 값이 같은 것.
- 물체(object): 같은 종류 랜드마크끼리 link_m 안에 이어진 것을 하나로 묶은 것.
  69 는 점을 1.5 m 안에서만 묶어서, 긴 카운터가 랜드마크 여러 개로 쪼개져 있다 (marble counter 41).
- 긴 물체: 물체 길이 > long_m. 반대쪽 끝에 서면 목적지에서 3 m 를 넘으므로 목적지를 **양 끝**에 둔다.
- 반복 물체: 목표 rep_r 안에 같은 종류 물체가 rep_n 개 넘게 있다 (screen doors). "몇 번째" 로 못 가리키므로
  근처의 **하나뿐인 물체(기준점, anchor)** 와 같이 쓴다. 기준점이 없으면 그 목표는 못 쓴다.

실패 이유 코드 (REASONS) 는 74 의 fail_reasons.json · 73 의 dropped 와 같다.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np

REASONS = {
    'NO_PATH': '출발점에서 목적지까지 길이 없음 (다른 섬 등)',
    'DIST_RANGE': '출발점~목적지 직선거리가 범위 밖',
    'TOO_LONG': '걷는 길이 max_path_m 초과 (600 걸음 여유)',
    'OFF_NAVMESH': '경로가 걸을 수 있는 바닥 밖 (U2)',
    'CLEARANCE': '벽에 로봇 몸 반지름보다 가까움 (U3)',
    'HOLE': '바닥 구멍 옆을 지남 (U4)',
    'FLAT_SHORT': '평지 경로인데 짧거나 높이차 큼 (U5)',
    'START_ON_STAIRS': '출발점이 계단 한가운데',
    'GOAL_ON_STAIRS': '목적지가 계단 한가운데',
    'STOREY_MISMATCH': '층 이동/평지 요구와 출발 층이 안 맞음 (뽑기 단계)',
    'START_TOO_CLOSE': '같은 목적지의 다른 경로 출발점과 너무 가까움',
    'DISTRACTOR_MANY': '가는 길에 같은 종류를 max_pass 개 넘게 지남',
    'NO_ANCHOR': '반복 물체인데 근처에 하나뿐인 기준점이 없음',
    'NO_SPOT': '설 자리 없음 (가림 · 화면 밖 · 다른 층)',
    'GOAL_HOLE': '물체가 구멍 위/근처',
    'STAIR_SIDE': '계단 끝인데 입구 바깥 바닥(층 높이)이 아님 — 디딤판 위 등',
}
JUDGE_CODE = {'U1': 'NO_PATH', 'U2': 'OFF_NAVMESH', 'U3': 'CLEARANCE', 'U4': 'HOLE', 'U5': 'FLAT_SHORT'}


def key(l):
    return (l.get('common') or l['name']).lower()


def _m70():
    p = Path(__file__).with_name('70_make_house.py')      # .house 와 같은 축·반지름 계산을 쓴다
    sp = importlib.util.spec_from_file_location('make_house70', p)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def build_objects(lms, dets, storey_of, link_m=1.5, merge_m=1.5, min_r=0.10):
    """같은 종류 · 같은 층 랜드마크를 link_m(평면) 안으로 이어 물체로 묶는다.
    -> objs [{oid, key, ids, storey, c(3), dir(2), half_len}], lm_obj {랜드마크 id: oid}"""
    m70 = _m70()
    n = len(lms)
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    X = np.array([l['xyz'] for l in lms], float).reshape(-1, 3)
    K = [key(l) for l in lms]
    S = [storey_of(x[2]) for x in X]
    for i in range(n):
        for j in range(i + 1, n):
            if K[i] == K[j] and S[i] == S[j] and math.hypot(*(X[i, :2] - X[j, :2])) <= link_m:
                par[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    by_name = {}
    for d in dets:
        by_name.setdefault(d['name'], []).append(np.asarray(d['xyz'], float))
    objs, lm_obj = [], {}
    for g in groups.values():
        pts = [p for i in g for p in by_name.get(lms[i].get('orig_name', lms[i]['name']), [])
               if np.linalg.norm(p - X[i]) <= merge_m]
        manual = any(lms[i].get('points') for i in g)        # 사람이 지정한 점 · 계단 끝: 그 점만 쓰고 70 과 같이 똑바로 선 상자
        if manual:
            pts = [np.asarray(p, float) for i in g for p in lms[i].get('points') or []]
        P = np.array(pts) if pts else X[g]
        c, A, r = m70.points_box(P) if manual else m70.pca_box(P)
        r = np.maximum(r, min_r)
        # 평면에서 가장 긴 축
        h = [(r[k] * np.linalg.norm(A[k, :2]), A[k, :2] / max(np.linalg.norm(A[k, :2]), 1e-9)) for k in range(3)]
        half, d2 = max(h, key=lambda t: t[0])
        # 랜드마크 중심들 사이 거리도 본다 (점 구름이 적어도 이어진 길이는 잡히게)
        if len(g) > 1:
            proj = (X[g, :2] - c[:2]) @ d2
            half = max(half, float(np.abs(proj).max()))
        o = {'oid': len(objs), 'key': K[g[0]], 'ids': [lms[i]['id'] for i in g], 'storey': S[g[0]],
             'c': np.asarray(c, float), 'dir': np.asarray(d2, float), 'half_len': float(half)}
        objs.append(o)
        for i in g:
            lm_obj[lms[i]['id']] = o['oid']
    return objs, lm_obj


def targets(o, long_m):
    """목적지를 둘 자리. 짧으면 중심 하나, 길면 양 끝 둘 -> [(part, xyz)]"""
    if 2 * o['half_len'] <= long_m:
        return [('whole', o['c'])]
    e = np.r_[o['dir'] * o['half_len'], 0.0]
    return [('end', o['c'] + e), ('end', o['c'] - e)]


def _seg_dist(p, o):
    """평면에서 점 p 와 물체(중심 ± dir·half_len 선분) 사이 거리."""
    v = p[:2] - o['c'][:2]
    t = float(np.clip(v @ o['dir'], -o['half_len'], o['half_len']))
    return float(np.linalg.norm(v - t * o['dir']))


def same_near(objs, o, p, r):
    """p 에서 r 안(평면, 같은 층) 의 같은 종류 물체 (o 포함)."""
    return [q for q in objs if q['key'] == o['key'] and q['storey'] == o['storey'] and _seg_dist(p, q) <= r]


def find_anchor(objs, o, p, rep_r, rep_n, anchor_r, succ_r):
    """반복 물체면 기준점을 찾는다. -> (반복인가, anchor dict 또는 None)
    기준점 = p 에서 anchor_r 안, 다른 종류, 자기 주변 rep_r 안에 같은 종류가 자기뿐인 물체.
    그리고 그 기준점에서 가장 가까운 o 종류 물체가 o 자신(또는 o 에서 succ_r 안)이어야 말이 하나로 가리킨다."""
    if len(same_near(objs, o, p, rep_r)) <= rep_n:
        return False, None
    cands = []
    for q in objs:
        if q['key'] == o['key'] or q['storey'] != o['storey']:
            continue
        d = _seg_dist(p, q)
        if d > anchor_r or len(same_near(objs, q, q['c'], rep_r)) != 1:
            continue
        mine = [x for x in objs if x['key'] == o['key'] and x['storey'] == o['storey']]
        near = min(mine, key=lambda x: _seg_dist(q['c'], x))
        if near['oid'] != o['oid'] and _seg_dist(o['c'], near) > succ_r:
            continue
        cands.append((d, q))
    if not cands:
        return True, None
    d, q = min(cands, key=lambda t: t[0])
    return True, {'oid': q['oid'], 'key': q['key'], 'ids': q['ids'],
                  'xyz': [round(float(t), 2) for t in q['c']], 'dist_m': round(d, 2)}


def passed_same(objs, oid, X, goal, pass_r, succ_r):
    """경로 X 가 지나는 같은 종류 물체 (목표 자신 · 목표에서 succ_r 안 것은 뺀다 — 거기서 멈춰도 성공).
    -> 지나는 순서대로 [{oid, ids, min_dist_m}]"""
    o = objs[oid]
    out = []
    for q in objs:
        if q['oid'] == oid or q['key'] != o['key']:
            continue
        if _seg_dist(goal, q) <= succ_r:
            continue
        d = np.array([_seg_dist(x, q) if abs(x[2] - q['c'][2]) < 2.0 else 1e9 for x in X])
        if d.min() <= pass_r:
            out.append((int(d.argmin()), {'oid': q['oid'], 'ids': q['ids'], 'min_dist_m': round(float(d.min()), 2)}))
    return [x for _, x in sorted(out, key=lambda t: t[0])]


def _selftest():
    lms = [{'id': 0, 'name': 'counter', 'xyz': [0, 0, 0]}, {'id': 1, 'name': 'counter', 'xyz': [1.4, 0, 0]},
           {'id': 2, 'name': 'counter', 'xyz': [2.8, 0, 0]}, {'id': 3, 'name': 'door', 'xyz': [0, 5, 0]},
           {'id': 4, 'name': 'door', 'xyz': [2.3, 5, 0]}, {'id': 5, 'name': 'door', 'xyz': [4.6, 5, 0]},
           {'id': 6, 'name': 'door', 'xyz': [6.9, 5, 0]}, {'id': 7, 'name': 'stairs', 'xyz': [2.3, 7, 0]}]
    objs, lm_obj = build_objects(lms, [], lambda z: 0)
    assert lm_obj[0] == lm_obj[1] == lm_obj[2], '이어진 카운터는 하나'
    oc = objs[lm_obj[0]]
    assert 2 * oc['half_len'] > 2.7 and len(targets(oc, 2.5)) == 2, '긴 물체는 양 끝'
    od = objs[lm_obj[4]]
    rep, anc = find_anchor(objs, od, od['c'], rep_r=10, rep_n=3, anchor_r=5, succ_r=3)
    assert rep and anc and anc['key'] == 'stairs', '반복 물체 -> 기준점 계단'
    od0 = objs[lm_obj[6]]
    rep, anc = find_anchor(objs, od0, od0['c'], rep_r=10, rep_n=3, anchor_r=3, succ_r=1)
    assert rep and anc is None, '기준점에서 가장 가까운 문이 아니면 못 쓴다'
    X = np.array([[x, 4.0, 0] for x in np.linspace(-2, 7, 40)])
    ps = passed_same(objs, lm_obj[6], X, np.array([6.9, 4.0, 0]), pass_r=3, succ_r=3)
    assert [p['ids'][0] for p in ps] == [3, 4], f'지나는 순서 {ps}'
    print('goal_rules selftest ok')


if __name__ == '__main__':
    _selftest()
