"""noeun 72 · 랜드마크 편집 도구가 읽을 데이터 묶음을 만든다.

왜: 69 가 낸 landmarks.json 은 사람이 눈으로 확인·수정해야 한다 (같은 기둥을 여럿으로 셌나, 이름이 맞나,
    크기가 맞나). 웹 도구 한 파일이 바로 읽을 수 있게 **필요한 것만** 한 json 으로 모은다.

넣는 것
  - 랜드마크: 이름 · 3D 위치 · 층 · 관측수 · score · **PCA 크기(반지름 3)**  <- .house 의 r0 r1 r2
  - 그 랜드마크가 보이는 사진 목록 (최대 n_views 장) + 그 사진에서의 검출 상자
  - 사진 메타 (카메라 위치 · 보는 방향) — 평면도에 '지금 어디서 보고 있나' 를 그리려고
  - 층 정보 (verify.json storeys)

    python3 scripts/dataset_converters/nav3d/noeun/72_landmark_tool_data.py --landmarks logs/scene_landmarks --snaps logs/scene_scan --out logs/landmark_tool/data.json
"""
import argparse, collections, json, math
from pathlib import Path

import numpy as np

W, H, HFOV_DEG = 640, 480, 68.7
FX = (W / 2) / math.tan(math.radians(HFOV_DEG) / 2)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--landmarks', required=True)
    ap.add_argument('--snaps', required=True)
    ap.add_argument('--verify', default='data/data/noeun_eval/navmesh/noeun_station/verify.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_views', type=int, default=6, help='랜드마크마다 보여줄 사진 수')
    ap.add_argument('--merge_m', type=float, default=1.5, help='69 가 쓴 묶기 거리 (도구에서 다시 조절)')
    return ap.parse_args()


def pca_box(P):
    """3D 점 구름 -> 반지름 3 개 (`.house` 의 r0 r1 r2). 점이 적으면 기본값."""
    if len(P) < 4:
        return [0.3, 0.3, 0.3]
    c = np.median(P, axis=0); X = P - c
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return [0.3, 0.3, 0.3]
    proj = X @ Vt[:3].T
    return [round(float(max(x, 0.1)), 2) for x in np.percentile(np.abs(proj), 90, axis=0)]


def main():
    a = parse()
    L = Path(a.landmarks)
    lms = json.load(open(L / 'landmarks.json'))
    dets = json.load(open(L / 'detections.json'))
    snaps = {json.loads(l)['id']: json.loads(l) for l in (Path(a.snaps) / 'snaps.jsonl').open()}
    storeys = json.load(open(a.verify))['storeys']
    print(f'[72] 랜드마크 {len(lms)} · 검출 {len(dets)} · 사진 {len(snaps)}')

    by_name = collections.defaultdict(list)
    for d in dets:
        by_name[d['name']].append(d)

    def storey_of(z):
        return int(np.argmin([abs(z - s['z']) for s in storeys]))

    out_lm = []
    for i, lm in enumerate(lms):
        mine = [d for d in by_name.get(lm['name'], [])
                if math.dist(d['xyz'], lm['xyz']) <= a.merge_m]
        P = np.array([d['xyz'] for d in mine], float) if mine else np.array([lm['xyz']], float)
        views = []
        for d in sorted(mine, key=lambda d: -d['score'])[:a.n_views]:
            s = snaps.get(d['snap'])
            if s is None:
                continue
            # 검출 상자를 다시 투영해 그린다 (저장해 둔 3D 위치 -> 그 사진 화면 좌표)
            c2w = np.asarray(s['c2w']); R = c2w[:3, :3].T
            p = (np.asarray(d['xyz']) - c2w[:3, 3]) @ R.T
            u = v = None
            if p[2] > 0.05:
                u, v = float(p[0] / p[2] * FX + (W - 1) / 2), float(p[1] / p[2] * FX + (H - 1) / 2)
            views.append({'snap': d['snap'], 'rgb': s['rgb'], 'score': d['score'],
                          'cam_m': d['cam_m'], 'area_pct': d['area_pct'],
                          'u': None if u is None else round(u), 'v': None if v is None else round(v)})
        out_lm.append({
            'id': i, 'name': lm['name'], 'xyz': lm['xyz'], 'storey': storey_of(lm['xyz'][2]),
            'views': lm['views'], 'n_det': lm['n_det'], 'score': lm['score'],
            'area_pct': lm['area_pct'], 'spread_m': lm['spread_m'],
            'radii': pca_box(P), 'shots': views, 'keep': True, 'edited': False,
        })

    out_snaps = [{'id': s['id'], 'rgb': s['rgb'], 'pos': s['robot_pos'],
                  'yaw': round(float(math.atan2(np.asarray(s['c2w'])[1, 2], np.asarray(s['c2w'])[0, 2])), 4),
                  'storey': storey_of(s['robot_pos'][2])}
                 for s in snaps.values()]

    data = {'scene': 'noeun_station',
            'storeys': [{'i': i, 'z': round(s['z'], 2), 'z_range': [round(x, 2) for x in s['z_range']],
                         'name': ['지하 2층 (승강장)', '지하 1층 (대합실)', '지상 (출구)'][i] if i < 3 else f'층 {i}'}
                        for i, s in enumerate(storeys)],
            'landmarks': out_lm, 'snaps': out_snaps,
            'vocab': json.load(open(L / 'vocab.json'))['vocab'],
            'merge_m': a.merge_m, 'img_base': 'scene_scan/'}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out, 'w'), ensure_ascii=False)
    print(f'[72] 저장 {out}  ({out.stat().st_size/1e6:.1f} MB)')
    c = collections.Counter(r['name'] for r in out_lm)
    print(f'[72] 종류 {len(c)} · 상위: ' + ', '.join(f'{n}({k})' for n, k in c.most_common(6)))


if __name__ == '__main__':
    main()
