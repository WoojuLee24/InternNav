"""노은역 목적지 거르기 — 쓸 수 있는 에피소드만 남긴다.

왜: 성공 기준이 목적지 3 m 안인데, 문장이 가리키는 A 가 멀거나(ep23 16.7 m) 서술 범위가 넓으면
    (ep21 "vending machines and seating area" 8.6 m) **문장대로 해도 실패**한다 — 애초에 풀 수 없는 문제다.
    노은역 문장은 Gemma 3 1B 가 목적지 프레임만 보고 지었고 (`stop_cue: null`), 거리·가림 검사가 없었다.

거르는 기준 (사용자 결정, 2026-11-12):
  a 가깝다 : A 의 3D 중앙점 ↔ 목적지 수평거리 <= dist_max (2.0 m)
             근거 = 로봇이 A 옆(약 1 m)에 서면 목적지에서 최대 3 m -> 성공 반경 안
  b 크다   : 마스크 면적 >= area_min (0.5 % = 40x40 px). 더 작으면 사람도 못 알아본다
  c 좁다   : 서술이 가리키는 범위(3D 점들의 p90) <= spread_max (3.0 m)
             "the street" · "ticket gate area" 처럼 **점이 안 정해지는 구역**을 거른다
  d 계단 중간 아님 : 목적지가 navmesh 연결부(verify.json, kind=stairs) bbox 안이면 제외.
             계단 바로 전/후는 허용한다 (사용자 결정)

허용하는 서술 유형 (사용자 결정): 1 단일 물체 · 2 두 물체 사이 · 3 물체의 특정 끝 · 5 면의 중심.
  4 경계 통과("just past the gates") · 6 방향 기준("in front of")은 **거리 표현이 모호**해서 제외한다
  (퍼짐 기준으로는 안 걸러진다 — 물체는 작은데 "얼마나 앞"이 안 정해지기 때문).

    python3 scripts/apptainer/noeun_goal_filter.py --landmarks logs/landmark40/landmarks_sam.json --out logs/landmark40/filtered.json
"""
import argparse, gzip, json

DEF_VAL = 'data/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1/raw_data/r2r/val_seen/val_seen.json.gz'
DEF_NAV = 'data/data/noeun_eval/navmesh/noeun_station/verify.json'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--landmarks', required=True, help='noeun_landmark_sam.py 결과')
    ap.add_argument('--val', default=DEF_VAL)
    ap.add_argument('--verify', default=DEF_NAV)
    ap.add_argument('--breakdown', default='', help='있으면 결과(성공/지나침)까지 붙인다')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dist_max', type=float, default=2.0)
    ap.add_argument('--area_min', type=float, default=0.5)
    ap.add_argument('--spread_max', type=float, default=3.0)
    ap.add_argument('--stair_pad_z', type=float, default=0.5)
    return ap.parse_args()


def main():
    a = parse()
    conns = json.load(open(a.verify))['connectors']
    h2o = lambda q: (q[0], -q[2], q[1])

    def on_stairs(p):
        for c in conns:
            (x0, y0), (x1, y1) = c['bbox_xy']; z0, z1 = c['z_range']
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and z0 - a.stair_pad_z <= p[2] <= z1 + a.stair_pad_z:
                return True
        return False

    eps = {str(e['episode_id']): e for e in json.load(gzip.open(a.val, 'rt'))['episodes']}
    L = {r['ep']: r for r in json.load(open(a.landmarks))}
    bd = {r['eid']: r for r in json.load(open(a.breakdown))} if a.breakdown else {}

    rec = []
    for ep, e in eps.items():
        r = L.get(ep, {}); b = bd.get(ep, {})
        g = h2o(e['goals'][0]['position'])
        d = r.get('dist_to_goal_m')
        why = []
        if on_stairs(g):
            why.append('계단중간')
        if d is None:
            why.append(r.get('verdict', '측정불가'))
        else:
            if d > a.dist_max:              why.append('멀다')
            if r['area_pct'] < a.area_min:  why.append('작다')
            if r['spread_m'] > a.spread_max: why.append('퍼짐')
        rec.append({'ep': ep, 'keep': not why, 'why': '·'.join(why), 'A': r.get('landmark'),
                    'd': d, 'area_pct': r.get('area_pct'), 'spread': r.get('spread_m'),
                    'n_inst': r.get('n_inst'), 'stair': on_stairs(g),
                    'res': b.get('res'), 'success': b.get('res') == 'success', 'mind': b.get('mind')})

    json.dump(rec, open(a.out, 'w'), ensure_ascii=False, indent=1)
    keep = [x for x in rec if x['keep']]
    print(f'전체 {len(rec)} -> 남김 {len(keep)} / 버림 {len(rec)-len(keep)}')
    import collections
    for w, c in collections.Counter(x['why'] for x in rec if not x['keep']).most_common():
        print(f'   {w:30s} {c}')
    print('남김:', ','.join('ep' + x['ep'] for x in keep))
    if bd:
        def st(rs, nm):
            n = len(rs); s = sum(1 for x in rs if x['success'])
            ov = sum(1 for x in rs if not x['success'] and (x['mind'] or 99) <= 3.0)
            cap = sum(1 for x in rs if x['res'] == 'exceed_total_max_step')
            print(f'  {nm:12s} n={n:2d}  SR={s/max(n,1):.3f} ({s})  지나침 {ov}  걸음상한 {cap}')
        print()
        st(rec, '전체'); st(keep, '남김'); st([x for x in rec if not x['keep']], '버림')
    print(f'저장 {a.out}')


if __name__ == '__main__':
    main()
