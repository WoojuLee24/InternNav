"""noeun · 계단을 **위 끝 · 아래 끝** 랜드마크로 바꾼다 (사용자 결정 2026-11-13).

왜: 계단은 길고 기울어서 상자(.house 물체 하나)로 덮으면 엉뚱한 곳까지 들어간다. 69 검출로는 55 조각이 됐다.
목적지는 "계단 바로 전 · 바로 후" 만 허용하므로 (계단 한가운데 금지) 계단을 그 두 곳으로 대신한다.

어디서: navmesh 의 계단 삼각형 (nav3d_path.NavScene.stairs_tri — 74 의 계단 판정과 같은 라벨) 을 이은 덩어리마다
  - 경사 방향 = 덩어리 꼭짓점에 평면 z = ax + by + c 를 맞춘 기울기
  - 입구(mouth) = 가장 낮은 꼭짓점들(아래) · 가장 높은 꼭짓점들(위)
  - 끝으로 인정 = 입구 바깥 search_m 안에 **계단 아닌 바닥**이 있고, 그 높이가 navmesh 층 높이(levels) 중 하나와
    level_tol 안. 층 사이 높이의 바닥 = 계단 중간 참(landing) -> 끝 아님 (계단 중간 목적지 금지와 같은 이유).
69 의 계단 랜드마크는 쓰지 않는다 (corrections.json "stairs": "ends" 일 때). navmesh 에 없는 계단
(걸을 수 없는 에스컬레이터 등) 은 그래서 빠진다.

    python3 scripts/dataset_converters/nav3d/noeun/stair_ends.py     # 자기 점검 + 끝 목록 (habitat 컨테이너)
"""
import math
import sys
from pathlib import Path

import numpy as np

STAIR_NAMES = ('stairs',)         # 69 랜드마크 중 이 이름(이름 고치기 뒤)은 끝 랜드마크로 대신한다
NAVMESH = 'data/data/noeun_eval/navmesh/noeun_station/noeun_station.navmesh'   # 73 과 같은 navmesh
ID0 = 200000                      # 끝 랜드마크 번호 = ID0 + 2·덩어리 + (0 아래 · 1 위). 69(0~) · 화면 추가(100000~) 와 안 겹침


def find_ends(sc, level_tol=0.5, search_m=1.2, near_m=1.5):
    """NavScene -> 끝 목록 [{sid, end, mouth, floor, out, rise, width, points, ok, why}] (ok=False 도 이유와 함께)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
    import nav3d_utils as nu
    T, st = sc.tris, sc.stairs_tri
    comp = nu.tri_components(T, st)
    lz = np.array([l['z'] for l in sc.levels], float)
    out = []
    for c in np.unique(comp[comp >= 0]):
        V = T[comp == c].reshape(-1, 3)
        g = np.linalg.lstsq(np.c_[V[:, :2], np.ones(len(V))], V[:, 2], rcond=None)[0][:2]
        up = g / max(np.linalg.norm(g), 1e-9)
        z0, z1 = V[:, 2].min(), V[:, 2].max()
        for end, sel, o in (('bottom', V[:, 2] < z0 + 0.15, -up), ('top', V[:, 2] > z1 - 0.15, up)):
            M = V[sel]
            m = M.mean(0)
            side = np.array([-o[1], o[0]])
            s = (M[:, :2] - m[:2]) @ side
            floor, why = None, set()
            # 입구를 따라 5 곳 × 방향 (바깥쪽 먼저) × 거리 — 계단이 꺾여 올라와도 찾게
            for t in np.linspace(s.min(), s.max(), 5):
                for ang in (0.0, -0.5, 0.5, -1.05, 1.05, -1.6, 1.6, -2.1, 2.1, -2.6, 2.6, 3.14):   # 바깥쪽부터, 꺾인 계단은 옆·뒤까지
                    d2 = np.array([math.cos(ang) * o[0] - math.sin(ang) * o[1], math.sin(ang) * o[0] + math.cos(ang) * o[1]])
                    for d in np.arange(0.3, search_m + 1e-6, 0.3):
                        q = m[:2] + side * t + d2 * d
                        _, idx = sc.surf.track_idx(q.reshape(1, 2), z0=m[2])
                        i = int(idx[0])
                        if i < 0 or abs(T[i][:, 2].mean() - m[2]) > 0.45:
                            why.add('바닥 없음')
                            continue
                        if st[i]:
                            why.add('아직 계단')
                            continue
                        zf = float(T[i][:, 2].mean())
                        if np.abs(lz - zf).min() > level_tol:
                            why.add(f'층 사이 참 z{zf:.2f}')
                            break
                        floor = np.array([q[0], q[1], zf])
                        break
                    if floor is not None:
                        break
                if floor is not None:
                    break
            # 상자용 점: 입구 꼭짓점 + 입구 near_m 안의 계단 꼭짓점 (첫 몇 칸)
            P = np.r_[M, V[np.linalg.norm(V[:, :2] - m[:2], axis=1) <= near_m]]
            out.append({'sid': int(c), 'end': end, 'mouth': m.round(3).tolist(),
                        'floor': None if floor is None else floor.round(3).tolist(), 'out': o.round(4).tolist(),
                        'rise': round(float(z1 - z0), 2), 'width': round(float(np.ptp(s)), 2),
                        'points': np.unique(P.round(3), axis=0).tolist(), 'ok': floor is not None,
                        'why': '' if floor is not None else ','.join(sorted(why))})
    return out


_SC = {}


def scene(navmesh):
    """navmesh -> NavScene (한 번만 읽는다)."""
    if navmesh not in _SC:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
        import nav3d_path as npth
        import nav3d_utils as nu
        _SC[navmesh] = npth.NavScene(nu.load_pathfinder(navmesh))
    return _SC[navmesh]


def inject(lms, c, navmesh, log=print):
    """corrections "stairs": "ends" 이면 남은 랜드마크에서 계단을 빼고 끝 랜드마크를 넣은 **새 목록**을 돌려준다.
    없으면 lms 그대로 (기존 동작). 끝 랜드마크 xyz = 입구 + 0.3 m (사람 눈에 첫 계단 높이 · 70 의 바닥 높이 거르기 통과)."""
    mode = (c or {}).get('stairs', 'landmarks')
    if mode == 'landmarks':
        return lms
    elif mode == 'ends':
        pass
    else:
        assert False, f'unreachable stairs={mode!r}'
    gone = [l for l in lms if l['name'] in STAIR_NAMES]
    keep = [l for l in lms if l['name'] not in STAIR_NAMES]
    E = [e for e in find_ends(scene(navmesh)) if e['ok']]
    al = [x for x in ((c.get('aliases') or {}).get('stairs') or []) if x != 'stairs']   # 다른 이름·공통 class 는 corrections 의 'stairs' 것
    cm = (c.get('common') or {}).get('stairs')
    for e in E:
        keep.append({'id': ID0 + 2 * e['sid'] + (e['end'] == 'top'), 'name': 'stairs', 'names': ['stairs'] + al,
                     **({'common': cm} if cm else {}),
                     'xyz': [e['mouth'][0], e['mouth'][1], round(e['mouth'][2] + 0.3, 3)], 'points': e['points'], 'views': 0, 'score': 1.0, 'n_det': 0,
                     'stair_end': e['end'], 'stair_floor': e['floor'], 'stair_out': e['out']})
    log(f'[계단] 69 계단 랜드마크 {len(gone)} 개 -> navmesh 계단 끝 {len(E)} 개 '
        f'(아래 {sum(e["end"] == "bottom" for e in E)} · 위 {sum(e["end"] == "top" for e in E)})')
    return keep


def _selftest():
    """경사판 하나: 아래(z 0 바닥) · 위(z 1 바닥) 끝이 모두 잡히고, 위쪽 바닥이 층 사이면 위 끝은 안 잡힌다."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
    import nav3d_utils as nu

    def quad(x0, x1, z0, z1, y0=0.0, y1=2.0):
        a, b, c_, d = [x0, y0, z0], [x1, y0, z1], [x1, y1, z1], [x0, y1, z0]
        return [[a, b, c_], [a, c_, d]]

    def scene(levels):
        tris = np.array(quad(-5, 0, 0, 0) + quad(0, 3, 0, 1.5) + quad(3, 8, 1.5, 1.5), float)
        st = np.zeros(len(tris), bool)
        st[2:4] = True

        class S:
            pass
        s = S()
        s.tris, s.stairs_tri = tris, st
        s.surf = nu.NavSurface(tris)
        s.levels = [{'z': z} for z in levels]
        return s
    E = {e['end']: e for e in find_ends(scene([0.0, 1.5]))}
    assert E['bottom']['ok'] and E['top']['ok'], E
    assert abs(E['bottom']['mouth'][0]) < 0.01 and abs(E['top']['mouth'][0] - 3) < 0.01
    assert E['bottom']['out'][0] < 0 < E['top']['out'][0], '바깥 방향'
    E = {e['end']: e for e in find_ends(scene([0.0]))}         # 위쪽 바닥이 층 높이가 아니다 -> 참(계단 중간)
    assert E['bottom']['ok'] and not E['top']['ok'] and '참' in E['top']['why'], E['top']
    print('stair_ends selftest ok')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
    _selftest()
    for e in find_ends(scene(NAVMESH)):
        print(f"계단{e['sid']:2d} {e['end']:6s} 오름 {e['rise']:4.1f} m 입구 {np.round(e['mouth'], 2)} 폭 {e['width']:.1f} -> "
              + ('끝 OK 바닥 ' + str(np.round(e['floor'], 2)) if e['ok'] else '아님: ' + e['why']))
