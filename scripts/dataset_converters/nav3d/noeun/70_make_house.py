"""noeun 70 · 랜드마크 목록 -> **MP3D `.house` 파일** 로 변환.

왜 이게 최종 목표인가:
    MP3D 의 instruction_labeling 은 `.house`(사람이 붙인 물체 주석)를 읽어 "멈출 대상"(stop_cue)을 고른다.
        build_stop_cues.py -> stop_cues.object_candidates(ep, house_path, ...) -> load_objects(.house)
        -> config.py 가 route['stop_cue'] 에 꽂음 -> e2e.generate 가 VLM 프롬프트에 "You stop next to the {name}{side}."
    노은역·ETRI 데이터는 GS 스캔이라 `.house` 가 없다 -> `stop_cue: null` -> VLM 이 멈출 대상을 **지어냈다**
    (val_seen 40 중 21 개가 "A 가 목적지 2 m 밖" 또는 "범위 3 m 초과").
    -> **`.house` 를 만들어 주면 기존 MP3D 파이프라인을 한 줄도 고치지 않고 그대로 쓸 수 있다.**

입력: 69_scene_landmarks.py 의 landmarks.json  ({name, xyz, views, area_pct, spread_m, ...})
      + detections.json (물체마다 3D 점 구름을 다시 모아 축·반지름을 구한다)
출력: <scene>.house  (ASCII 1.1 형식 — load_objects 가 읽는 C/O 줄)

`.house` 줄 형식 (MP3D 실제 파일에서 확인):
    C  idx  mapidx  raw_name  cat_idx  mpcat40_name  0 0 0 0 0
    O  idx  region  cat  px py pz  a0x a0y a0z  a1x a1y a1z  r0 r1 r2  0 0 0 0 0 0 0 0
      중심(px,py,pz) · 직교축 2 개(a0,a1; a2 = a0 x a1) · 반지름 3 개(r0,r1,r2)

축·반지름은 마스크 3D 점 구름의 **주성분 분석**으로 구한다 (`.house` 는 직육면체 근사이고 우리도 같게).
높이 조건(-0.2 ~ 2.2 m)은 `route.yaml` A-grounded-v1 과 같은 값 — 천장등·환풍구를 뺀다.

좌표: `.house` 는 mesh Z-up. 노은역 ours 좌표가 같은 규약이라 변환 없이 쓴다
      (stop_cues.py 주석: "habitat (x, y_up, z) <-> mesh (x, -z, y)").

    python3 scripts/dataset_converters/nav3d/noeun/70_make_house.py --landmarks logs/scene_landmarks --out data/scene_data/noeun_station/noeun_station.house
"""
import argparse, collections, json, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import corrections  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--landmarks', required=True, help='69 결과 폴더 (landmarks.json · detections.json)')
    ap.add_argument('--out', required=True, help='<scene>.house')
    ap.add_argument('--scene', default='noeun_station')
    ap.add_argument('--zones', default='', help='61_build_zones 결과 (region 번호용). 없으면 층으로 나눈다')
    ap.add_argument('--verify', default='data/data/noeun_eval/navmesh/noeun_station/verify.json',
                    help='층 높이 (region 대용)')
    ap.add_argument('--min_h', type=float, default=-0.2, help='바닥 대비 높이 하한 (route.yaml 과 같음)')
    ap.add_argument('--max_h', type=float, default=2.2, help='상한')
    ap.add_argument('--merge_m', type=float, default=1.5)
    ap.add_argument('--min_r', type=float, default=0.10, help='반지름 하한 (너무 납작하면 투영이 깨진다)')
    ap.add_argument('--corrections', default='',
                    help='리포트의 "고친 내용" 파일(corrections.json). 이름 고치기 + 제외를 반영한다. '
                         '73_goal_candidates.py 와 **같은 파일**을 줘야 .house 와 목적지가 어긋나지 않는다.')
    return ap.parse_args()


def pca_box(P):
    """3D 점 구름 -> (중심, 축 3 개, 반지름 3 개). `.house` 의 직육면체 근사와 같은 방식."""
    c = np.median(P, axis=0)
    X = P - c
    if len(P) < 4:
        return c, np.eye(3), np.full(3, 0.2)
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return c, np.eye(3), np.full(3, 0.2)
    A = Vt[:3]
    if np.linalg.det(A) < 0:
        A[2] = -A[2]
    proj = X @ A.T
    r = np.percentile(np.abs(proj), 90, axis=0)
    return c, A, r


def main():
    a = parse()
    L = Path(a.landmarks)
    lms = json.load(open(L / 'landmarks.json'))
    for i, l in enumerate(lms):          # id = landmarks.json 안의 순서. 리포트가 쓰는 번호와 같다
        l.setdefault('id', i)
    dets = json.load(open(L / 'detections.json'))
    print(f'[70] 랜드마크 {len(lms)} · 검출 {len(dets)}')

    if a.corrections:
        lms = corrections.apply(lms, corrections.load(a.corrections), log=print)

    # 층 높이 -> region 번호
    storeys = json.load(open(a.verify))['storeys'] if Path(a.verify).exists() else []
    def region_of(z):
        best, bi = 1e9, 0
        for i, s in enumerate(storeys):
            d = abs(z - s['z'])
            if d < best:
                best, bi = d, i
        return bi
    def floor_z(z):
        return storeys[region_of(z)]['z'] if storeys else 0.0

    # 랜드마크마다 자기 검출들의 3D 점을 다시 모아 축·반지름 계산
    by_name = collections.defaultdict(list)
    for d in dets:
        by_name[d['name']].append(d)

    cats, objs = {}, []
    cat_common, cat_alias = {}, {}
    n_low, n_high = 0, 0
    for lm in lms:
        c0 = np.array(lm['xyz'], float)
        dz = c0[2] - floor_z(c0[2])
        if dz < a.min_h:
            n_low += 1; continue
        if dz > a.max_h:
            n_high += 1; continue          # 천장등 · 환풍구 · 간판 상단
        # 이름을 고쳤어도 점 구름은 **원래 이름**으로 묶여 있다
        pts = np.array([d['xyz'] for d in by_name.get(lm.get('orig_name', lm['name']), [])
                        if math.dist(d['xyz'], lm['xyz']) <= a.merge_m], float)
        if len(pts) == 0:
            pts = c0.reshape(1, 3)
        c, A, r = pca_box(pts)
        r = np.maximum(r, a.min_r)
        if lm['name'] not in cats:
            cats[lm['name']] = len(cats)
        # 공통 class 가 있으면 mpcat40 로 쓴다 (없으면 대표 이름). 다른 이름은 옆 파일로 나간다
        cat_common[lm['name']] = lm.get('common') or lm['name']
        for a_ in lm.get('names', [lm['name']])[1:]:
            cat_alias.setdefault(lm['name'], [])
            if a_ not in cat_alias[lm['name']]:
                cat_alias[lm['name']].append(a_)
        objs.append({'cat': cats[lm['name']], 'region': region_of(c[2]),
                     'center': c, 'axes': A, 'radii': r,
                     'views': lm['views'], 'score': lm['score']})
    print(f'[70] 높이로 제외: 바닥 아래 {n_low} · {a.max_h} m 위 {n_high} -> 물체 {len(objs)} · 범주 {len(cats)}')

    lo = np.min([o['center'] - o['radii'].max() for o in objs], axis=0) if objs else np.zeros(3)
    hi = np.max([o['center'] + o['radii'].max() for o in objs], axis=0) if objs else np.zeros(3)
    n_reg = max(1, len(storeys))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        f.write('ASCII 1.1\n')
        f.write(f'H  {a.scene}  {a.scene}  0 0 0 {n_reg} 0 {len(objs)} {len(cats)} 0 0  0  0 0 0 0 0 '
                f'{lo[0]:.5f} {lo[1]:.5f} {lo[2]:.5f}  {hi[0]:.5f} {hi[1]:.5f} {hi[2]:.5f}  0 0 0 0 0\n')
        for name, idx in sorted(cats.items(), key=lambda kv: kv[1]):
            safe = name.replace(' ', '_')
            # C idx mapidx raw_name cat_idx mpcat40_name — load_objects 는 t[1](idx) 과 t[5](mpcat40) 만 쓴다.
            # raw_name = 대표 이름, mpcat40_name = 공통 class (없으면 대표 이름과 같다)
            common = cat_common.get(name, name).replace(' ', '_')
            f.write(f'C  {idx}  {idx + 1} {safe}  {idx + 1} {common}  0 0 0 0 0 \n')
        for i, o in enumerate(objs):
            c, A, r = o['center'], o['axes'], o['radii']
            f.write(f"O  {i} {o['region']} {o['cat']}  {c[0]:.5f} {c[1]:.5f} {c[2]:.5f}  "
                    f"{A[0,0]:.6f} {A[0,1]:.6f} {A[0,2]:.6f}  {A[1,0]:.6f} {A[1,1]:.6f} {A[1,2]:.6f}  "
                    f"{r[0]:.5f} {r[1]:.5f} {r[2]:.5f}  0 0 0 0 0 0 0 0 \n")
    # 다른 이름은 .house 에 못 들어간다 (범주마다 이름 두 칸뿐) -> 옆 파일로
    if cat_alias:
        side = out.with_suffix('.aliases.json')
        side.write_text(json.dumps({'raw_name': {k: v for k, v in sorted(cat_alias.items())},
                                    'common': {k: v for k, v in sorted(cat_common.items()) if v != k}},
                                   ensure_ascii=False, indent=1))
        print(f'[70] 다른 이름 {sum(len(v) for v in cat_alias.values())} 개 / 범주 {len(cat_alias)} -> {side}')

    print(f'[70] 저장 {out}  (물체 {len(objs)} · 범주 {len(cats)} · region {n_reg})')

    # 바로 되읽어 검증 — 기존 load_objects 로
    import sys
    sys.path.insert(0, 'scripts/dataset_converters/instruction_labeling/label')
    from stop_cues import load_objects
    got = load_objects(str(out))
    print(f'[70] 검증: load_objects 가 {len(got)} 개 읽음 (기대 {len(objs)})')
    assert len(got) == len(objs), '읽은 개수가 다르다'
    cnt = collections.Counter(g['mpcat40'] for g in got)
    print('[70] 범주별:', ', '.join(f'{k} {v}' for k, v in cnt.most_common(10)))


if __name__ == '__main__':
    main()
