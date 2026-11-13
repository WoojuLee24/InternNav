"""노은역 랜드마크 걸어 보기 — teleop_view.py(Isaac, RBQ10 걷기) 앞에 붙는 페이지 서버 (CPU).

브라우저 http://localhost:<port> :
  왼쪽 = 위에서 본 지도 + 로봇 점 + 시야 부채꼴 + **랜드마크 전부** (.house 에 들어갈 중심·크기 타원)
  가운데 = 로봇 카메라 RGB + 그 위에 랜드마크를 되찍은 표시
  오른쪽 = 고른 랜드마크의 .house 정보 (고칠 수 있음) + 69 스캔에서 그 물체가 찍힌 사진
지도·카메라에서 랜드마크에 마우스를 올리면 이름, 누르면 정보가 열린다.

고친 것은 corrections.json 에 쓴다 (70 .house · 73 목적지가 같이 읽는 파일):
  이름 rename_ids · 다른 이름 alias_ids · 공통 class common_ids · 빼기 exclude_ids · 되살리기 include_ids
  · 중심/반지름/다시 맞춘 3D 점 override_ids · 화면에서 추가한 물체 add (카메라 화면을 눌러 분할 -> 메쉬에 광선 -> 3D 점). 처음 저장할 때 원본을 <파일>.bak_<시각> 으로 남긴다.

걷기·렌더는 teleop_view.py 가 한다 (수정 없이 --log_root 로 노은역 지도를 준다). 이 서버는 그쪽 /meta /bg /state
/frame.jpg /cmd /teleport /reset /layer 를 그대로 전달한다.

    # 1) isaac 컨테이너 (GPU): 걷기 + 렌더
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/tools/teleop_view.py --scene noeun_station \
      --log_root scripts/dataset_converters/nav3d/logs/noeun_walk --usd /ws/src/InternNav/data/GS_USDZ/noeun_station_collision.usdz --start 0 0 0
    # 2) habitat 컨테이너 (CPU): 이 페이지
    python3 scripts/dataset_converters/nav3d/noeun/tools/noeun_landmark_walk.py
"""
import argparse
import collections
import copy
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # noeun/tools
NOEUN = HERE.parent
sys.path[:0] = [str(NOEUN), str(NOEUN.parent / 'utils')]
import corrections  # noqa: E402
import stair_ends  # noqa: E402
import merge_lines  # noqa: E402
from render_poses import trunk_to_cam_cv  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8766)
    ap.add_argument('--teleop', default='http://127.0.0.1:8765', help='teleop_view.py 주소')
    ap.add_argument('--seg', default='http://127.0.0.1:8768', help='noeun_seg_server.py 주소 (실시간 분할)')
    ap.add_argument('--landmarks', default='logs/scene_landmarks', help='69 결과 (landmarks.json · detections.json)')
    ap.add_argument('--corrections', default='logs/landmark_final/corrections.json', help='읽고 고쳐 쓰는 파일')
    ap.add_argument('--goals', default='logs/goal_candidates_final/goals.json', help='73 결과 (없으면 목적지 표시 안 함)')
    ap.add_argument('--tool_data', default='logs/landmark_tool/data.json', help='72 결과 — 랜드마크별 스캔 사진 위치(u, v)')
    ap.add_argument('--snaps', default='logs/scene_scan/rgb')
    ap.add_argument('--masks', default='logs/scene_landmarks_masks',
                    help='69 --save_masks 결과 (detections.json + masks/). 없으면 분할 사진 없이')
    ap.add_argument('--verify', default='data/data/noeun_eval/navmesh/noeun_station/verify.json')
    ap.add_argument('--navmesh', default='data/data/noeun_eval/navmesh/noeun_station/noeun_station.navmesh',
                    help='corrections "stairs": "ends" 일 때 계단 끝 (70·73 과 같은 navmesh)')
    ap.add_argument('--mesh', default='data/data/noeun_eval/navmesh/noeun_station/noeun_station.glb',
                    help='카메라에서 물체까지 가리는지 볼 메쉬 (73 의 가림 판정과 같은 메쉬)')
    # 70_make_house.py 와 같은 값이어야 화면의 중심·반지름이 .house 와 같다
    ap.add_argument('--merge_m', type=float, default=1.5)
    ap.add_argument('--min_r', type=float, default=0.10)
    ap.add_argument('--min_h', type=float, default=-0.2)
    ap.add_argument('--max_h', type=float, default=2.2)
    ap.add_argument('--cam_w', type=int, default=640, help='teleop_view --width 와 같게')
    ap.add_argument('--cam_h', type=int, default=480)
    return ap.parse_args()


def load70():
    sp = importlib.util.spec_from_file_location('make_house70', NOEUN / '70_make_house.py')
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


class Model:
    """landmarks + corrections -> 화면에 그릴 물체 목록 (70 과 같은 계산)."""

    def __init__(self, a):
        self.a = a
        self.L = json.load(open(Path(a.landmarks) / 'landmarks.json'))
        self.dets = json.load(open(Path(a.landmarks) / 'detections.json'))
        self.by_name = collections.defaultdict(list)
        for d in self.dets:
            self.by_name[d['name']].append(np.asarray(d['xyz'], float))
        self.storeys = json.load(open(a.verify))['storeys']
        m70 = load70()
        self.pca_box, self.points_box = m70.pca_box, m70.points_box
        self.shots = {}
        if Path(a.tool_data).exists():
            for l in json.load(open(a.tool_data))['landmarks']:
                self.shots[l['id']] = sorted(l.get('shots') or [], key=lambda s: -s['score'])[:12]
        self.spots = collections.defaultdict(list)
        if Path(a.goals).exists():
            for g in json.load(open(a.goals))['landmarks']:
                for i in g.get('ids', [g['id']]):
                    self.spots[i] += [{k: s.get(k) for k in ('xyz', 'yaw', 'dist_m', 'part', 'anchor')} for s in g['spots']]
        # 분할 사진: 69 --save_masks 를 다시 돌린 결과. 검출 순서가 원래 것과 같다는 보장이 없어
        # 70 과 같은 규칙(원래 이름 · 랜드마크 중심 merge_m 안)으로 랜드마크에 붙인다.
        self.mdets = collections.defaultdict(list)
        fm = Path(a.masks) / 'detections.json'
        if fm.exists():
            rgb = {json.loads(l)['id']: json.loads(l)['rgb'] for l in open(Path(a.snaps).parent / 'snaps.jsonl')}
            for d in json.load(open(fm)):
                if d.get('mask'):
                    self.mdets[d['name']].append({**d, 'rgb': rgb.get(d['snap'], '')})
        self.c = corrections.load(a.corrections) if Path(a.corrections).exists() else corrections.load('')
        self.backed_up = False
        self.lock = threading.Lock()
        self.build()

    def storey_of(self, z):
        return int(np.argmin([abs(z - s['z']) for s in self.storeys]))

    def build(self):
        a, c = self.a, self.c
        lms = [dict(l, id=i) for i, l in enumerate(self.L)]
        kept_l = corrections.apply(lms, c, log=lambda *x: None)   # lms 의 이름 고치기는 제자리에서 된다
        kept_l = stair_ends.inject(kept_l, c, a.navmesh, log=lambda *x: None)   # 70·73 과 같게: 계단 -> 위·아래 끝
        kept_l = merge_lines.inject(kept_l, c, self.dets, self.storey_of, merge_m=a.merge_m, log=lambda *x: None)
        kept = {l['id']: l for l in kept_l}
        lms += [l for l in kept_l if l.get('stair_end') or l.get('merged_ids')]
        merged_of = {i: l['id'] for l in kept_l for i in l.get('merged_ids') or []}
        ex_ids = {int(x) for x in c.get('exclude_ids') or []}
        ex_nm = {str(w).strip().lower() for w in c.get('exclude_names') or []}
        ex_ty = [w.strip().lower() for w in c.get('exclude_types') or [] if str(w).strip()]
        pat = re.compile(r'\b(' + '|'.join(map(re.escape, ex_ty)) + r')s?\b', re.I) if ex_ty else None
        objs = []
        for l in lms:
            i, k = l['id'], kept.get(l['id'])
            o = {'id': i, 'name': l['name'], 'orig_name': l.get('orig_name', l['name']),
                 'xyz': [round(float(t), 3) for t in l['xyz']], 'storey': self.storey_of(l['xyz'][2]),
                 'views': l.get('views'), 'n_det': l.get('n_det'), 'score': l.get('score'),
                 'kept': k is not None, 'shots': self.shots.get(i, []), 'spots': self.spots.get(i, []),
                 'masks': sorted([{x: d[x] for x in ('snap', 'rgb', 'mask', 'score', 'cam_m', 'area_pct')}
                                  for d in self.mdets.get(l.get('orig_name', l['name']), [])
                                  if math.dist(d['xyz'], l['xyz']) <= a.merge_m], key=lambda d: -d['score'])[:9]}
            if l.get('merged_ids'):              # 한 줄로 합친 물체 (스크린도어) — 구성원 분할 사진은 각자에
                o.update(merged_ids=l['merged_ids'], masks=[])
            if l.get('stair_end'):               # navmesh 계단 끝 — 69 분할 사진은 이 물체 것이 아니다
                o.update(stair_end=l['stair_end'], masks=[])
            if k is None:
                o['why'] = (f'한 줄로 합침 -> #{merged_of[i]}' if i in merged_of
                            else '계단은 위·아래 끝으로 대신함' if c.get('stairs') == 'ends' and l['name'] in stair_ends.STAIR_NAMES
                            else '번호로 뺌' if i in ex_ids else '이름으로 뺌' if l['name'].lower() in ex_nm
                            else '종류로 뺌' if pat and pat.search(l['name']) else '뺌')
                objs.append(o)
                continue
            # 70_make_house.py 와 같은 계산: 높이 거르기 → 원래 이름 검출 점 → PCA 상자 → (사람이 고친 값)
            c0 = np.asarray(l['xyz'], float)
            dz = float(c0[2] - self.storeys[o['storey']]['z'])
            house = {'in_house': a.min_h <= dz <= a.max_h,
                     'why': '' if a.min_h <= dz <= a.max_h else f'바닥에서 {dz:.2f} m (허용 {a.min_h}~{a.max_h})'}
            pts = np.array([p for p in self.by_name.get(o['orig_name'], []) if np.linalg.norm(p - c0) <= a.merge_m])
            if l.get('points'):                  # 화면에서 추가한 물체 (70 과 같게)
                pts = np.asarray(l['points'], float)
            o['added'] = bool(l.get('added'))
            cc, A, r = (self.points_box(pts) if l.get('points')       # 70 과 같게: 사람이 지정한 점은 똑바로 선 상자
                        else self.pca_box(pts if len(pts) else c0.reshape(1, 3)))
            r = np.maximum(r, a.min_r)
            ov = k.get('override') or {}
            if ov.get('center') is not None:
                cc = np.asarray(ov['center'], float)
            if ov.get('radii') is not None:
                r = np.maximum(np.asarray(ov['radii'], float), a.min_r)
            # 지도 타원: 평면 성분이 큰 두 축
            hz = sorted(range(3), key=lambda j: -r[j] * np.linalg.norm(A[j, :2]))
            j0, j1 = hz[0], hz[1]
            house.update(raw_name=k['name'], mpcat40=k.get('common') or k['name'], aliases=k.get('names', [k['name']])[1:],
                         common=k.get('common') or '', center=[round(float(t), 3) for t in cc],
                         axes=[[round(float(t), 4) for t in A[j]] for j in range(3)], radii=[round(float(t), 3) for t in r],
                         region=o['storey'], n_pts=int(len(pts)), override=ov,
                         ell={'a': float(r[j0] * np.linalg.norm(A[j0, :2])), 'b': float(r[j1] * np.linalg.norm(A[j1, :2])),
                              'ang': float(math.atan2(A[j0, 1], A[j0, 0]))})
            o['house'] = house
            objs.append(o)
        self.objs = objs
        n_k = sum(o['kept'] for o in objs)
        n_h = sum(1 for o in objs if o['kept'] and o['house']['in_house'])
        print(f'[walk] 랜드마크 {len(objs)} · 남김 {n_k} · .house 에 들어감 {n_h}', flush=True)

    def box(self, P):
        """3D 점 -> 70 과 같은 상자 (중심·축·반지름·지도 타원)."""
        c, A, r = self.points_box(P)
        r = np.maximum(r, self.a.min_r)
        hz = sorted(range(3), key=lambda j: -r[j] * np.linalg.norm(A[j, :2]))
        st = self.storey_of(float(np.median(P[:, 2])))
        dz = float(c[2] - self.storeys[st]['z'])
        return {'center': [round(float(t), 3) for t in c], 'radii': [round(float(t), 3) for t in r],
                'storey': st, 'dz': round(dz, 2), 'in_house': self.a.min_h <= dz <= self.a.max_h,
                'ell': {'a': float(r[hz[0]] * np.linalg.norm(A[hz[0], :2])), 'b': float(r[hz[1]] * np.linalg.norm(A[hz[1], :2])),
                        'ang': float(math.atan2(A[hz[0], 1], A[hz[0], 0]))}}

    def add(self, b):
        """화면에서 추가한 물체 -> corrections.json 'add' (+ 다른 이름·공통 class 는 같은 id 로)."""
        nm = str(b['name']).strip().lower()
        assert nm, '이름이 비었다'
        P = np.asarray(b['points'], float)
        assert P.ndim == 2 and P.shape[1] == 3 and len(P) >= 3, '3D 점이 3 개 미만'
        c = self.c
        c.setdefault('add', [])
        i = max([100000] + [int(x['id']) + 1 for x in c['add']])     # 69 번호(0~)와 안 겹치게
        c['add'].append({'id': i, 'name': nm, 'xyz': [round(float(t), 3) for t in np.median(P, 0)],
                         'points': [[round(float(t), 3) for t in q] for q in P[:400]], 'views': 1, 'score': 1.0,
                         'source': b.get('source') or {}, 'added_at': time.strftime('%Y-%m-%dT%H:%M:%S')})
        rest = {k: b[k] for k in ('aliases', 'common') if k in b}
        return self.edit({'id': i, **rest})

    def edit(self, b):
        """한 랜드마크 고치기 -> corrections.json 저장."""
        i = int(b['id'])
        s = str(i)
        c = self.c
        for k in ('rename_ids', 'alias_ids', 'common_ids', 'override_ids'):
            c.setdefault(k, {})
        for k in ('exclude_ids', 'include_ids'):
            c.setdefault(k, [])
        if 'name' in b:
            nm = str(b['name']).strip().lower()
            if nm:
                c['rename_ids'][s] = nm
        if 'aliases' in b:
            al = [str(x).strip().lower() for x in b['aliases'] if str(x).strip()]
            c['alias_ids'][s] = al
            if not al:
                c['alias_ids'].pop(s)
        if 'common' in b:
            cm = str(b['common'] or '').strip().lower()
            c['common_ids'][s] = cm
            if not cm:
                c['common_ids'].pop(s)
        if 'center' in b or 'radii' in b or 'points' in b:
            ov = dict(c['override_ids'].get(s) or {})
            if b.get('points'):                  # 화면에서 다시 맞춤: 점이 새 상자를 정하므로 손으로 넣은 중심·반지름은 지운다
                P = np.asarray(b['points'], float)
                assert P.ndim == 2 and P.shape[1] == 3 and len(P) >= 3, '3D 점이 3 개 미만'
                ov = {'points': [[round(float(t), 3) for t in q] for q in P[:400]], 'refit_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                      'source': b.get('source') or {}}
            elif 'points' in b:
                ov.pop('points', None); ov.pop('refit_at', None); ov.pop('source', None)
            for k in ('center', 'radii'):
                if k in b:
                    ov[k] = None if b[k] is None else [round(float(t), 3) for t in b[k]]
            ov = {k: v for k, v in ov.items() if v is not None}
            c['override_ids'][s] = ov
            if not ov:
                c['override_ids'].pop(s)
        if 'exclude' in b:
            exl = {int(x) for x in c['exclude_ids']}
            inc = {int(x) for x in c['include_ids']}
            if b['exclude']:
                exl.add(i); inc.discard(i)
            else:
                exl.discard(i); inc.add(i)    # 종류·이름째 빠진 것도 이 하나는 살린다
            c['exclude_ids'], c['include_ids'] = sorted(exl), sorted(inc)
        p = Path(self.a.corrections)
        if not self.backed_up and p.exists():
            bak = p.with_name(p.name + time.strftime('.bak_%Y%m%d_%H%M%S'))
            bak.write_bytes(p.read_bytes())
            self.backed_up = True
        c['saved_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        c['saved_by'] = 'noeun_landmark_walk'
        p.write_text(json.dumps(c, ensure_ascii=False, indent=1))
        try:
            os.chown(p, 1000, 1000)    # 컨테이너 root → 호스트 사용자
        except OSError:
            pass
        self.build()
        return next(o for o in self.objs if o['id'] == i), str(p)


def main():
    a = parse()
    M = Model(a)
    html = HERE / 'noeun_landmark_walk.html'     # 요청마다 읽는다 (페이지 고쳐도 재시작 없이)
    M_cam = trunk_to_cam_cv().tolist()
    import trimesh
    RI = trimesh.load(a.mesh, force='mesh', process=False, skip_materials=True).ray

    def visible(x, y, z, yaw, cam_xyz=None):
        """로봇 자세 -> 카메라에서 광선을 쏴 가려지지 않은 랜드마크 id (73 과 같은 규칙: 물체보다 0.35 m 앞에서 맞으면 가림)."""
        c, s_ = math.cos(yaw), math.sin(yaw)
        T = np.array([[c, -s_, 0, x], [s_, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        cam = (T @ np.asarray(M_cam))[:3, 3] if cam_xyz is None else np.asarray(cam_xyz, float)   # 카메라를 옮겼으면 그 자리
        with M.lock:
            P = [(o['id'], np.asarray(o['house']['center'] if o.get('house') else o['xyz'], float)) for o in M.objs]
        P = [(i, p) for i, p in P if 0.3 < np.linalg.norm(p - cam) < 25]
        if not P:
            return []
        ids, X = zip(*P)
        D = np.asarray(X) - cam
        rng = np.linalg.norm(D, axis=1)
        loc, ri, _ = RI.intersects_location(np.tile(cam, (len(D), 1)), D / rng[:, None], multiple_hits=False)
        hit = np.full(len(D), np.inf)
        hit[ri] = np.linalg.norm(loc - cam, axis=1)
        return [int(i) for i, h, r in zip(ids, hit, rng) if h >= r - 0.35]

    def raycast(b):
        """멈춘 화면의 마스크 픽셀 -> 메쉬에 광선 -> 3D 점. 가장자리에서 뒤 벽에 맞은 점은 거리 사분위로 뺀다."""
        s = b['pose']
        c, s_ = math.cos(s['yaw']), math.sin(s['yaw'])
        T = np.array([[c, -s_, 0, s['x']], [s_, c, 0, s['y']], [0, 0, 1, s['z']], [0, 0, 0, 1]])
        if s.get('R'):                         # 몸 기울기까지 — 렌더 카메라와 같은 자세 (yaw 만 쓰면 4 m 앞이 10~20 cm 어긋난다)
            T[:3, :3] = np.asarray(s['R'], float)
        T = np.asarray(s['C'], float) if s.get('C') else T @ np.asarray(M_cam)   # C = 실제로 찍은 카메라 (시점 옮김 포함)
        R, cam = T[:3, :3], T[:3, 3]
        W, H = b['W'], b['H']
        f = W / 2 / math.tan(math.radians(b['hfov_deg']) / 2)
        px = np.asarray(b['pixels'], float)
        uv = px[:, :2]
        inner = px[:, 2] > 0 if px.shape[1] > 2 else np.ones(len(px), bool)   # 분할 서버가 준 '안쪽' 표시 (없으면 전부)
        d = np.c_[(uv[:, 0] + .5 - W / 2) / f, (uv[:, 1] + .5 - H / 2) / f, np.ones(len(uv))] @ R.T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        loc, ri, _ = RI.intersects_location(np.tile(cam, (len(d), 1)), d, multiple_hits=False)
        if len(loc) < 3:
            raise RuntimeError('메쉬에 맞은 점이 3 개 미만')
        hit = np.full(len(d), np.nan)
        hit[ri] = np.linalg.norm(loc - cam, axis=1)
        rng = hit[np.isfinite(hit) & inner]
        if len(rng) < 10:                    # 안쪽이 거의 없는 가는 물체 -> 전부로
            rng = hit[np.isfinite(hit)]
        q1, q3 = np.percentile(rng, [25, 75])   # 물체 거리 범위 = 안쪽 픽셀 (가는 다리 사이로 뒤 바닥에 맞은 광선은 섞지 않는다)
        med = float(np.median(rng))
        k = (hit >= q1 - 1.5 * (q3 - q1) - 0.2) & (hit <= q3 + 1.5 * (q3 - q1) + 0.2)
        k &= inner | (np.abs(hit - med) <= 0.15)   # 가는 부분·가장자리는 물체 거리 ±0.15 m 일 때만 (다리 바로 뒤 바닥을 막는다)
        # 물체 거리 밖(뒤 바닥·벽에 맞음 = 메쉬에 없는 가는 다리 · 가장자리, 또는 안 맞음)은 버리지 않고
        # 그 광선 위 **물체 거리(중앙값)** 에 둔다 — 분할 모양은 맞으니 위치만 물체 면으로
        r_use = np.where(k, hit, med)
        P = cam + d * r_use[:, None]
        out = M.box(P)
        cc = np.asarray(out['center'])
        with M.lock:   # 이미 있는 랜드마크가 가까우면 알려준다 (중복 추가 막기)
            near = sorted([(round(float(np.linalg.norm(np.asarray(o['house']['center'] if o.get('house') else o['xyz']) - cc)), 2),
                            o['id'], o['house']['raw_name'] if o.get('house') else o['name'], o['kept']) for o in M.objs])
        out.update(points=[[round(float(t), 3) for t in q] for q in P], n_hit=int(len(loc)), n_kept=int(k.sum()),
                   n_moved=int((~k).sum()), range_m=round(med, 2),
                   near=[{'dist': d_, 'id': i, 'name': n, 'kept': kp} for d_, i, n, kp in near if d_ <= 2.0][:5])
        return out

    def fwd(path, body=None, base=None, timeout=3):
        req = urllib.request.Request((base or a.teleop) + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'} if body is not None else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), r.headers.get('Content-Type', 'application/octet-stream')

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), 'application/json; charset=utf-8')

        def _proxy(self, path, body=None):
            try:
                self._send(*fwd(path, body))
            except Exception as e:          # teleop 이 아직 안 떴거나 죽었다
                self._json({'error': f'teleop_view ({a.teleop}) 연결 안 됨: {e}'}, 502)

        def do_GET(self):
            p = self.path.split('?')[0]
            if p == '/':
                self._send(200, html.read_bytes(), 'text/html; charset=utf-8')
            elif p == '/nmeta':
                self._json({'storeys': M.storeys, 'M_cam': M_cam, 'cam_w': a.cam_w, 'cam_h': a.cam_h,
                            'corrections': a.corrections})
            elif p == '/objects':
                with M.lock:
                    self._json(M.objs)
            elif p == '/visible':
                q = dict(kv.split('=') for kv in self.path.split('?', 1)[1].split('&'))
                cx = [float(q[k]) for k in ('cx', 'cy', 'cz')] if 'cx' in q else None
                self._json(visible(*(float(q[k]) for k in ('x', 'y', 'z', 'yaw')), cam_xyz=cx))
            elif p.startswith('/mask/'):
                f = Path(a.masks) / 'masks' / Path(p[6:]).name
                self._send(200, f.read_bytes(), 'image/png') if f.exists() else self._send(404, b'no', 'text/plain')
            elif p.startswith('/scan/'):
                f = Path(a.snaps) / Path(p[6:]).name
                self._send(200, f.read_bytes(), 'image/jpeg') if f.exists() else self._send(404, b'no', 'text/plain')
            elif p.startswith('/seg/'):
                try:
                    self._send(*fwd(self.path, base=a.seg, timeout=30))
                except Exception as e:
                    self._send(502, f'분할 서버 ({a.seg}) 연결 안 됨: {e}'.encode(), 'text/plain; charset=utf-8')
            elif p in ('/meta', '/state', '/frame.jpg') or p.startswith('/bg/'):
                self._proxy(self.path)
            else:
                self._send(404, b'not found', 'text/plain')

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            except json.JSONDecodeError:
                return self._json({'error': 'json 이 아니다'}, 400)
            if self.path == '/edit':
                try:
                    with M.lock:
                        o, path = M.edit(body)
                    self._json({'ok': True, 'obj': o, 'path': path})
                except Exception as e:
                    self._json({'error': f'{type(e).__name__}: {e}'}, 400)
            elif self.path in ('/add_preview', '/add'):
                try:
                    if self.path == '/add_preview':
                        return self._json(raycast(body))
                    with M.lock:
                        o, path = M.add(body)
                    self._json({'ok': True, 'obj': o, 'path': path})
                except Exception as e:
                    self._json({'error': f'{type(e).__name__}: {e}'}, 400)
            elif self.path.startswith('/seg/'):
                try:
                    self._send(*fwd(self.path, body, base=a.seg, timeout=180))
                except urllib.error.HTTPError as e:
                    self._send(e.code, e.read(), 'application/json; charset=utf-8')
                except Exception as e:
                    self._json({'error': f'분할 서버 ({a.seg}) 연결 안 됨: {e}'}, 502)
            elif self.path in ('/cmd', '/teleport', '/reset', '/layer', '/cam'):
                self._proxy(self.path, body)
            else:
                self._json({'error': 'not found'}, 404)

    print(f'[walk] http://localhost:{a.port}  (teleop {a.teleop} · 고친 것 → {a.corrections})', flush=True)
    ThreadingHTTPServer(('0.0.0.0', a.port), H).serve_forever()


if __name__ == '__main__':
    main()
