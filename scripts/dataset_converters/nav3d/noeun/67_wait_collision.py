"""noeun 67 · 대기 프레임 "사람과 충돌 가능" 자동 판정 v4 — oracle 사람 위치(hunav 궤적) + 사람 검출(사진).

기준 (2026-10-29 사용자): 충돌 가능 = WAIT. 사람이 보여도 부딪힐 가능성이 없으면 제외, 일부만 보여도 가능성이 있으면 WAIT. 위험 거리 4 m.
v4 규칙 (사용자 라벨 97 개와 97/97 일치, wait_review_v3.json):
  충돌 가능 = [oracle: t 또는 t-1 에 같은 층(높이 차 <= 1.5 m) 사람이 4 m 안, 카메라 정면 ±45° 안]
              또는 [검출: t 사진의 사람 박스(score >= 0.3) 아래 끝 y2 >= 280 px (480 중, 가까움)]
  oracle = raw_data/r2r/humans_noeun_v3.npz (0.05 s) 를 sidecar 시각으로 읽어 시작 기준 로컬로 (65 와 같은 변환).
  검출기 = torchvision Faster R-CNN ResNet50-FPN v2 (COCO person), CPU. 다리만 보이는 가까운 사람은 자주 놓친다 -> oracle 이 주 판정.
이전: v1 (65 투영, t 또는 t-1) 85/97 · v3 (검출 y2>=240 | 투영 t) 90/97.

출력 (--out): {'frames': [{split, episode, frame, visible (= 충돌 가능 자동), oracle, det_t, near_t, near_prev, boxes_t, boxes_prev}], 'params', 'summary'}
66_wait_keep.py --visibility 로 이 파일을 넣는다.
사용 (저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/67_wait_collision.py
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='/home/irteam/data-vol1/nav3d/noeun/vln_ce_noeun_nav3d_rbq_hunav_v1')
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--frames', default=str(HERE / 'wait' / 'wait_visibility_v1.json'), help='대기 프레임 목록 (65 출력의 split/episode/frame 만 쓴다)')
    ap.add_argument('--danger_m', type=float, default=4.0, help='위험 거리 (사용자 2026-10-29)')
    ap.add_argument('--danger_deg', type=float, default=45.0, help='카메라 정면 기준 각도 (화면 가장자리 = 34.4°)')
    ap.add_argument('--score', type=float, default=0.3)
    ap.add_argument('--y2_min', type=float, default=280.0, help='t 사진 사람 박스 아래 끝 (480 px 중) — 클수록 가까움')
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--out', default=str(HERE / 'wait' / 'wait_collision_v1.json'))
    return ap.parse_args()


def main():
    a = parse()
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    torch.set_num_threads(a.threads)
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, box_score_thresh=a.score).eval()
    root = Path(a.root)
    frames = json.load(open(a.frames))['frames']
    hz = np.load(root / 'raw_data' / 'r2r' / 'humans_noeun_v3.npz')
    HX, HY, HZ, hdt = hz['x'], hz['y'], hz['z'], float(hz['t'][1] - hz['t'][0])
    cache = {}

    def episode(split, ep):
        if (split, ep) not in cache:
            tdir = 'r2r' if split == 'train' else 'r2r_val_seen'
            sc = root / 'traj_data' / tdir / 'noeunstation'
            side = {int(r['episode']): r for r in map(json.loads, open(sc / 'meta' / 'hunav_sidecar.jsonl'))}
            e = json.load(gzip.open(root / 'raw_data' / 'r2r' / split / f'{split}.json.gz'))['episodes'][ep]
            col = f'pose.{a.rig}'
            T = pq.read_table(sc / 'data' / 'chunk-000' / f'episode_{ep:06d}.parquet', columns=[col]).to_pydict()[col]
            cache[(split, ep)] = (np.array([[np.array(r) for r in x] for x in T], dtype=np.float64), e, side[ep]['t'], tdir)
        return cache[(split, ep)]

    def oracle_near(split, ep, f):
        """t=f 에서 같은 층 사람 중 danger_m 안 · 정면 ±danger_deg 안이 있는가 + [(거리, 각도)] (65 와 같은 좌표 변환)."""
        if f < 0:
            return False, []
        T, e, ts, _ = episode(split, ep)
        sx, sy, sz = e['start_position']
        q = e['start_rotation']
        phi = 2 * math.atan2(q[1], q[3])
        fwd = np.array([-math.sin(phi), -math.cos(phi)])
        left = np.array([-math.cos(phi), math.sin(phi)])
        k = min(max(int(round(ts[f] / hdt)), 0), len(HX) - 1)
        hab = np.c_[HX[k], HZ[k], -HY[k]]                                    # nav3d -> habitat (nav3d_utils.ours_to_hab)
        d = hab[:, [0, 2]] - np.array([sx, sz])
        loc = np.c_[d @ fwd, d @ left, hab[:, 1] - sy]                       # 시작 기준 로컬 (x 전방, y 왼쪽, z 위; 바닥 = 0)
        cam = T[f][:3, 3]
        ax = T[f][:3, 2][:2] / np.linalg.norm(T[f][:3, 2][:2])                 # 카메라 광축 수평 방향
        rel = loc[:, :2] - cam[:2]
        dist = np.linalg.norm(rel, axis=1)
        ang = np.degrees(np.arccos(np.clip((rel @ ax) / np.maximum(dist, 1e-6), -1, 1)))
        same = np.abs(loc[:, 2] - (cam[2] - T[0, 2, 3])) <= 1.5
        m = same & (dist < a.danger_m)
        return bool((m & (ang < a.danger_deg)).any()), [[round(float(x), 2), round(float(y), 1)] for x, y in zip(dist[m], ang[m])]

    def boxes(split, ep, f):
        if f < 0:
            return []
        tdir = episode(split, ep)[3]
        p = root / 'traj_data' / tdir / 'noeunstation' / 'videos' / 'chunk-000' / f'observation.images.rgb.{a.rig}' / f'episode_{ep:06d}_{f}.jpg'
        with torch.no_grad():
            o = model([TF.to_tensor(Image.open(p).convert('RGB'))])[0]
        return [[round(float(s), 3)] + [round(float(v), 1) for v in b] for b, l, s in zip(o['boxes'], o['labels'], o['scores']) if int(l) == 1]

    out = []
    for r in frames:
        sp, ep, f = r['split'], r['episode'], r['frame']
        ot, nt = oracle_near(sp, ep, f)
        op, npv = oracle_near(sp, ep, f - 1)
        bt, bp = boxes(sp, ep, f), boxes(sp, ep, f - 1)
        det = any(s >= a.score and y2 >= a.y2_min for s, x1, y1, x2, y2 in bt)
        out.append({'split': sp, 'episode': ep, 'frame': f, 'visible': bool(ot or op or det), 'oracle': bool(ot or op), 'det_t': det,
                    'near_t': nt, 'near_prev': npv, 'boxes_t': bt, 'boxes_prev': bp})
    summ = {s: {'wait_frames': sum(o['split'] == s for o in out), 'collision_auto': sum(o['split'] == s and o['visible'] for o in out)}
            for s in ('train', 'val_seen')}
    Path(a.out).write_text(json.dumps({'frames': out, 'summary': summ,
                                       'params': {'rule': 'oracle(t|t-1: d < danger_m & ang < danger_deg) | det(t: y2 >= y2_min)', 'danger_m': a.danger_m,
                                                  'danger_deg': a.danger_deg, 'score': a.score, 'y2_min': a.y2_min,
                                                  'detector': 'fasterrcnn_resnet50_fpn_v2 COCO_V1', 'frames': Path(a.frames).name}}, indent=1))
    print(json.dumps(summ), '->', a.out)


if __name__ == '__main__':
    main()
