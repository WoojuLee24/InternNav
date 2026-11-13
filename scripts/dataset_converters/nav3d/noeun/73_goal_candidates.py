"""noeun 73 · 랜드마크 거르기 + **목적지 후보** 찾기 (한 번에).

왜 한 번에 하나:
    "쓸 수 있는 랜드마크인가" 와 "목적지를 어디에 둘까" 는 사실 같은 질문이다 —
    둘 다 **랜드마크 2 m 안에 설 수 있는 자리가 있나** 를 묻는다. 따로 하면 같은 계산을 두 번 한다.

왜 목적지를 랜드마크 옆에 두나:
    평가 성공 기준이 목적지 3 m 안이다. 문장은 "A 옆에서 멈춰라" 라고 말한다.
        A 가 목적지 2 m 안  +  로봇이 A 옆(약 1 m)에 섬  =  목적지에서 최대 3 m  -> 성공 보장
    지금까지는 목적지를 무작위로 찍고 나중에 A 를 찾았다 -> val_seen 40 중 21 개가 "A 가 멀다" 로 탈락.

거르는 조건
  (1) 랜드마크가 **바닥 구멍** 위/근처면 버린다 — 로봇이 빠지거나(mesh) 사진에 바닥이 비친다(gs).
      등급은 CRITICAL 만 (사용자 결정 2026-11-13). minor 1,993 개까지 빼면 역을 거의 못 쓴다.
  (2) 랜드마크 2 m 안에 **navmesh 위 + 구멍 밖** 인 자리가 없으면 버린다 — 설 데가 없다.
  (3) 그 자리에서 랜드마크가 **보여야** 한다 — 화면 안 + 메시에 가리지 않음.
      기둥 바로 뒤에 서면 기둥이 화면 밖이라 단서가 못 된다.

출력: goals.json  [{landmark, xyz, spots:[{xyz, yaw, dist_m, u, v}], ...}]
      spots 하나가 **목적지 후보** 하나다 (서는 자리 + 바라보는 방향).

    python3 scripts/dataset_converters/nav3d/noeun/73_goal_candidates.py --landmarks logs/scene_landmarks --out logs/goal_candidates
"""
import argparse, collections, csv, json, math
from pathlib import Path

import numpy as np
import re
import sys

import trimesh

sys.path.insert(0, str(Path(__file__).parent))
import corrections  # noqa: E402

W, H, HFOV_DEG = 640, 480, 68.7
FX = (W / 2) / math.tan(math.radians(HFOV_DEG) / 2)
HALF_W = math.radians(HFOV_DEG) / 2


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--landmarks', required=True, help='69 결과 폴더')
    ap.add_argument('--holes', default='scripts/dataset_converters/nav3d/logs/noeun/11_check_assets/noeun_station/holes.csv')
    ap.add_argument('--navmesh', default='data/data/noeun_eval/navmesh/noeun_station/noeun_station.navmesh')
    ap.add_argument('--mesh', default='data/data/noeun_eval/navmesh/noeun_station/noeun_station.glb')
    ap.add_argument('--verify', default='data/data/noeun_eval/navmesh/noeun_station/verify.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--flags', default='CRITICAL', help='거를 구멍 등급 (쉼표). 사용자 결정: CRITICAL 만')
    ap.add_argument('--r_b', type=float, default=0.25, help='로봇 몸 반지름 (01 navmesh 와 같음)')
    ap.add_argument('--goal_r', type=float, default=2.0, help='랜드마크에서 목적지까지 최대 거리')
    ap.add_argument('--min_r', type=float, default=0.6, help='너무 붙으면 화면에 안 들어온다')
    ap.add_argument('--n_dir', type=int, default=12, help='랜드마크 둘레를 몇 방향으로 훑을지')
    ap.add_argument('--max_spots', type=int, default=3, help='랜드마크당 목적지 후보 수')
    ap.add_argument('--cam_h', type=float, default=0.60, help='카메라 높이 (바닥 위)')
    ap.add_argument('--cam_pitch', type=float, default=15.0)
    ap.add_argument('--yaw_max', type=float, default=25.0, help='랜드마크가 화면 중심에서 이 각도 안에 들어와야')
    ap.add_argument('--exclude_types', default='',
                    help='쓰지 않을 랜드마크 종류 (쉼표, 머리 명사 기준 부분일치). '
                         "예: 'pillar,canopy,arrow' -> white pillar · blue arrow 등 전부 제외. "
                         '기둥처럼 같은 것이 여러 개라 "어느 기둥?"이 되는 종류를 뺄 때 쓴다.')
    ap.add_argument('--exclude_names', default='',
                    help='쓰지 않을 랜드마크 **종류 이름 그대로** (세미콜론 구분, 띄어쓰기 포함 · 대소문자 무시). '
                         "예: 'white sign;blue arrow' -> 그 두 종류만 빼고 green sign 은 남긴다. "
                         '--exclude_types 가 머리 명사로 뭉뚱그리는 것과 달리 **한 종류씩** 집어낸다. '
                         '.txt/.json 경로를 주면 그 파일에서 읽는다 (txt=한 줄에 하나, json=문자열 배열).')
    ap.add_argument('--exclude_ids', default='',
                    help='쓰지 않을 **랜드마크 하나하나**의 번호 (쉼표). landmarks.json 안의 순서(0 부터)이고 '
                         '편집기·리포트가 보여주는 id 와 같다. 같은 종류 안에서 특정 물체만 뺄 때 쓴다. '
                         '.txt/.json 경로를 주면 그 파일에서 읽는다.')
    ap.add_argument('--corrections', default='',
                    help='리포트의 "고친 내용" 파일(corrections.json). 이름 고치기 + 제외를 한꺼번에 반영한다. '
                         '70_make_house.py 와 **같은 파일**을 줘야 .house 와 목적지가 어긋나지 않는다.')
    ap.add_argument('--keep_json', default='',
                    help='편집기에서 내려받은 landmarks_edited.json. 주면 그 안의 id 만 쓴다.')
    return ap.parse_args()


def main():
    a = parse()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    flags = {x.strip() for x in a.flags.split(',')}

    lms = json.load(open(Path(a.landmarks) / 'landmarks.json'))
    for i, l in enumerate(lms):          # id = landmarks.json 안의 순서. 걸러도 번호는 안 바뀐다
        l.setdefault('id', i)
    n_all = len(lms)

    if a.corrections:
        lms = corrections.apply(lms, corrections.load(a.corrections), log=print)

    # 사람이 정한 제외 — 종류 이름 부분일치 (단·복수 모두)
    ex = [w.strip().lower() for w in a.exclude_types.split(',') if w.strip()]
    if ex:
        pat = re.compile(r'\b(' + '|'.join(re.escape(w) for w in ex) + r')s?\b', re.I)
        dropped_types = collections.Counter(l['name'] for l in lms if pat.search(l['name']))
        lms = [l for l in lms if not pat.search(l['name'])]
        print(f'[73] 종류 제외 ({a.exclude_types}): {n_all - len(lms)} 개 '
              f'/ 종류 {len(dropped_types)} -> ' + ', '.join(f'{k} {v}' for k, v in dropped_types.most_common()))

    # 이름 그대로 제외 — 한 종류씩
    if a.exclude_names:
        src = Path(a.exclude_names)
        if src.suffix in ('.txt', '.json') and src.exists():
            raw = json.load(open(src)) if src.suffix == '.json' else src.read_text().splitlines()
        else:
            raw = a.exclude_names.split(';')
        names = {w.strip().lower() for w in raw if w and w.strip()}
        before = len(lms)
        gone = collections.Counter(l['name'] for l in lms if l['name'].lower() in names)
        lms = [l for l in lms if l['name'].lower() not in names]
        unknown = names - {l.lower() for l in gone}
        print(f'[73] 이름 제외: {before - len(lms)} 개 / 종류 {len(gone)} -> '
              + ', '.join(f'{k} {v}' for k, v in gone.most_common()))
        if unknown:
            print(f'[73]   경고 — 없는 이름 {len(unknown)}: ' + ', '.join(sorted(unknown)))

    # 번호로 제외 — 랜드마크 하나하나
    if a.exclude_ids:
        src = Path(a.exclude_ids)
        if src.suffix in ('.txt', '.json') and src.exists():
            raw = json.load(open(src)) if src.suffix == '.json' else src.read_text().replace(',', ' ').split()
        else:
            raw = a.exclude_ids.replace(',', ' ').split()
        ids = {int(x) for x in raw if str(x).strip()}
        before = len(lms)
        gone = collections.Counter(l['name'] for l in lms if l['id'] in ids)
        lms = [l for l in lms if l['id'] not in ids]
        print(f'[73] 번호 제외: {before - len(lms)} 개 / 종류 {len(gone)} -> '
              + ', '.join(f'{k} {v}' for k, v in gone.most_common(8)))
        missing = ids - set(range(n_all))
        if missing:
            print(f'[73]   경고 — 범위 밖 번호 {len(missing)}: ' + ', '.join(map(str, sorted(missing)[:10])))

    # 편집기 결과가 있으면 그 목록만
    if a.keep_json:
        ed = json.load(open(a.keep_json))
        keep_ids = {l['id'] for l in ed['landmarks']}
        by_ed = {l['id']: l for l in ed['landmarks']}
        before = len(lms)
        lms = [l for l in lms if l['id'] in keep_ids]
        for l in lms:                      # 사람이 고친 이름·좌표를 반영
            e = by_ed.get(l.get('id'))
            if e:
                l['name'], l['xyz'], l['storey'] = e['name'], e['xyz'], e['storey']
        print(f'[73] 편집기 목록 적용: {before} -> {len(lms)}')
    holes = [r for r in csv.DictReader(open(a.holes)) if r['flag'] in flags]
    print(f'[73] 랜드마크 {len(lms)} · 구멍 {len(holes)} ({",".join(sorted(flags))})')
    hk = collections.Counter(r['kind'] for r in holes)
    print(f'[73]   구멍 종류: ' + ', '.join(f'{k} {v}' for k, v in hk.most_common()))

    HX = np.array([[float(r['x']), float(r['y']), float(r['z']),
                    float(r['inscribed_diam_m']) / 2] for r in holes], float)

    def hole_clear(p):
        """구멍에서 (로봇반지름 + 구멍반지름) 밖인가. 같은 층(높이 1.5 m 안)만 본다."""
        if len(HX) == 0:
            return True, 99.0
        m = np.abs(HX[:, 2] - p[2]) < 1.5
        if not m.any():
            return True, 99.0
        d = np.hypot(HX[m, 0] - p[0], HX[m, 1] - p[1]) - HX[m, 3] - a.r_b
        return bool(d.min() > 0), float(d.min())

    import importlib.util as iu
    sp = iu.spec_from_file_location('ns', 'scripts/eval/noeun_navmesh_server.py')
    NS = iu.module_from_spec(sp); sp.loader.exec_module(NS)
    import habitat_sim
    pf = habitat_sim.PathFinder()
    assert pf.load_nav_mesh(a.navmesh), a.navmesh
    mesh = trimesh.load(a.mesh, force='mesh', process=False, skip_materials=True)
    RI = mesh.ray

    def snap(p):
        q = pf.snap_point(NS.w2h(np.asarray(p, float)).astype(np.float32))
        if not np.all(np.isfinite(q)):
            return None
        return np.asarray(NS.h2w(np.asarray(q, float)), float)

    storeys = json.load(open(a.verify))['storeys']
    def storey_of(z):
        return int(np.argmin([abs(z - s['z']) for s in storeys]))

    kept, dropped = [], collections.Counter()
    for lm in lms:
        c = np.asarray(lm['xyz'], float)
        ok, d = hole_clear(c)
        if not ok:
            dropped['구멍 위/근처'] += 1
            continue

        spots = []
        for k in range(a.n_dir):
            th = 2 * math.pi * k / a.n_dir
            for rr in (a.goal_r, (a.goal_r + a.min_r) / 2, a.min_r):
                p = np.array([c[0] + math.cos(th) * rr, c[1] + math.sin(th) * rr, c[2]])
                s = snap(p)
                if s is None:
                    continue
                if abs(s[2] - c[2]) > 1.2:          # 다른 층으로 snap 된 것
                    continue
                dist = float(math.hypot(s[0] - c[0], s[1] - c[1]))
                if not (a.min_r * 0.7 <= dist <= a.goal_r):
                    continue
                ok2, hd = hole_clear(s)
                if not ok2:
                    continue
                # 랜드마크 쪽을 봤을 때 화면 안에 드는가 + 가리지 않는가
                yaw = math.atan2(c[1] - s[1], c[0] - s[0])
                cam = np.array([s[0], s[1], s[2] + a.cam_h])
                v = c - cam
                rng = float(np.linalg.norm(v))
                if rng < 0.2:
                    continue
                # 화면 좌우 각도
                dth = abs(((math.atan2(v[1], v[0]) - yaw) + math.pi) % (2 * math.pi) - math.pi)
                if dth > math.radians(a.yaw_max):
                    continue
                # 가림 — 카메라에서 랜드마크로 광선을 쏴 더 가까운 면에 먼저 맞으면 가려진 것
                dirv = v / rng
                loc, _, _ = RI.intersects_location(np.array([cam]), np.array([dirv]), multiple_hits=False)
                if len(loc) and float(np.linalg.norm(loc[0] - cam)) < rng - 0.35:
                    continue
                spots.append({'xyz': [round(float(t), 2) for t in s],
                              'yaw': round(yaw, 4), 'dist_m': round(dist, 2),
                              'hole_clear_m': round(hd, 2), 'yaw_off_deg': round(math.degrees(dth), 1)})
                break                                # 이 방향은 하나면 충분
        if not spots:
            dropped['설 자리 없음'] += 1
            continue
        # 고르게 퍼진 것부터 (방향이 다양하도록 yaw 로 정렬 후 솎기)
        spots.sort(key=lambda s: s['yaw'])
        pick = spots[:: max(1, len(spots) // a.max_spots)][:a.max_spots]
        kept.append({**lm, 'storey': storey_of(c[2]), 'spots': pick, 'n_spot_all': len(spots)})

    n_goal = sum(len(l['spots']) for l in kept)
    print(f"\n[73] 랜드마크 {len(lms)} -> 쓸 수 있는 것 {len(kept)}")
    for k, v in dropped.most_common():
        print(f'       버림 {k}: {v}')
    print(f'[73] 목적지 후보 {n_goal} 개 (랜드마크당 평균 {n_goal/max(len(kept),1):.1f})')
    cs = collections.Counter(l['storey'] for l in kept)
    for i, s in enumerate(storeys):
        print(f'       층 {i}: 랜드마크 {cs.get(i,0)} · 목적지 {sum(len(l["spots"]) for l in kept if l["storey"]==i)}')
    cn = collections.Counter(l['name'] for l in kept)
    print('[73] 상위 종류: ' + ', '.join(f'{n}({k})' for n, k in cn.most_common(8)))
    json.dump({'landmarks': kept, 'dropped': dict(dropped),
               'params': {'flags': sorted(flags), 'goal_r': a.goal_r, 'min_r': a.min_r,
                          'yaw_max': a.yaw_max, 'r_b': a.r_b}},
              open(out / 'goals.json', 'w'), ensure_ascii=False)
    print(f'[73] 저장 {out}/goals.json')


if __name__ == '__main__':
    main()
