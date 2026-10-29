"""noeun 65 · 대기 프레임 사람 보임 판정 — hunav 사람 궤적을 그 프레임 카메라에 투영해 "사진에 사람이 보이는 대기 프레임" 을 고른다.

대기 프레임 = 로더 시프트 후 actions[t] == 4 (= sidecar waiting[t]). 사람 = raw_data/r2r/humans_noeun_v3.npz (nav3d 좌표, z 위, 0.05 s 간격),
그 프레임 시각 = sidecar t[t]. 카메라 = parquet pose.60cm_15deg (시작 기준 로컬, OpenCV c2w) + raw json start_position/rotation.
판정 (사람마다 발 0.1 · 허리 0.9 · 머리 1.6 m 세 점):
  보임 = 한 점이라도 카메라 앞(z > 0.2 m) 이고 화면(640 x 480, hfov 68.7) 안, 수평 거리 <= --max_dist_m, 높이 차 <= 1.5 m (같은 층)
  벽에 가려짐(occlusion)은 모른다 -> 사람 눈 확인(검토 페이지)으로 고친다.
최종 visible = 현재 프레임 보임 또는 직전 --prev 프레임 안에 보임 (기본 1, 2026-10-29 사용자 결정).
  S2 의 지난 프레임은 linspace(0, t-1, 8) 이라 항상 들어가는 최근 프레임은 t-1 하나뿐이다 -> margin 은 t-1 까지만.
지난 --hist 프레임 안에 보였는지도 참고로 적는다.

출력 (--out): {'frames': [{split, episode, frame, visible, visible_cur, visible_prev, hist_visible, n_visible, min_dist_m, points}], 'params': ...}
사용 (저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/65_wait_visibility.py
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
W, H, HFOV = 640, 480, 68.7                      # gs_vlnpe/apply_real/camera_profiles.py 60cm rig
FX = (W / 2.0) / math.tan(math.radians(HFOV) / 2)
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0
BODY_Z = (0.1, 0.9, 1.6)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='/home/irteam/data-vol1/nav3d/noeun/vln_ce_noeun_nav3d_rbq_hunav_v1')
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--max_dist_m', type=float, default=10.0, help='6 m 는 화면에 보이는 먼 사람을 놓쳤다 (ep21 · ep48 눈 확인)')
    ap.add_argument('--v_margin', type=int, default=200, help='화면 아래 여유 px: 가까운(<1.5 m) 사람은 투영이 실제보다 아래로 어긋나 발이 화면 밖으로 나간다 (ep182 · val ep24 눈 확인)')
    ap.add_argument('--near_m', type=float, default=0.0, help='0 = 끔. >0 이면 이 거리 안 + 카메라 앞 ±near_deg 를 투영 없이 보임으로 (1.5 로 시험: 카메라 아래·옆 사람까지 보임이 돼 ep76 등 오판 -> 끔)')
    ap.add_argument('--near_deg', type=float, default=60.0)
    ap.add_argument('--prev', type=int, default=1, help='visible 에 넣는 직전 프레임 수 (S2 입력에 항상 있는 것은 t-1)')
    ap.add_argument('--hist', type=int, default=5, help='참고용')
    ap.add_argument('--out', default=str(HERE / 'wait' / 'wait_visibility_v1.json'))
    return ap.parse_args()


def project(M, hum_xyz_local):
    """M: 4x4 c2w (로컬). hum: [N, 3] 로컬 점 -> (u, v, 깊이) [N, 3]."""
    c = (np.linalg.inv(M) @ np.c_[hum_xyz_local, np.ones(len(hum_xyz_local))].T).T[:, :3]
    z = c[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u, v = FX * c[:, 0] / z + CX, FX * c[:, 1] / z + CY
    return np.c_[u, v, z]


def main():
    a = parse()
    root = Path(a.root)
    hz = np.load(root / 'raw_data' / 'r2r' / 'humans_noeun_v3.npz')
    ht, HX, HY, HZ = hz['t'], hz['x'], hz['y'], hz['z']
    dt = float(ht[1] - ht[0])
    out = []
    for split, tdir in [('train', 'r2r'), ('val_seen', 'r2r_val_seen')]:
        sc = root / 'traj_data' / tdir / 'noeunstation'
        side = {int(r['episode']): r for r in map(json.loads, open(sc / 'meta' / 'hunav_sidecar.jsonl'))}
        eps = json.load(gzip.open(root / 'raw_data' / 'r2r' / split / f'{split}.json.gz'))['episodes']
        for i, e in enumerate(eps):
            act = [int(x) for x in pq.read_table(sc / 'data' / 'chunk-000' / f'episode_{i:06d}.parquet', columns=['action']).to_pydict()['action']]
            acts = act[1:] + [0]                                    # 로더와 같은 시프트
            waits = [t for t, x in enumerate(acts) if x == 4]
            if not waits:
                continue
            assert all(side[i]['waiting'][t] for t in waits), f'{split} ep {i}: sidecar waiting 과 action 4 가 다르다'
            col = f'pose.{a.rig}'
            T = pq.read_table(sc / 'data' / 'chunk-000' / f'episode_{i:06d}.parquet', columns=[col]).to_pydict()[col]
            T = np.array([[np.array(r) for r in x] for x in T], dtype=np.float64)
            sx, sy, sz = e['start_position']
            q = e['start_rotation']
            phi = 2 * math.atan2(q[1], q[3])
            fwd = np.array([-math.sin(phi), -math.cos(phi)])         # habitat (x, z) 평면
            left = np.array([-math.cos(phi), math.sin(phi)])
            z0 = T[0, 2, 3]                                          # 로컬 원점 = 시작 바닥 + 카메라 높이

            def vis(f):
                k = int(round(side[i]['t'][f] / dt))
                hab = np.c_[HX[k], HZ[k], -HY[k]]                     # nav3d -> habitat (nav3d_utils.ours_to_hab)
                d = hab[:, [0, 2]] - np.array([sx, sz])
                loc = np.c_[d @ fwd, d @ left, hab[:, 1] - sy]       # 시작 기준 로컬 (x 전방, y 왼쪽, z 위; 바닥 = 0)
                cam = T[f][:3, 3]
                pts, dists = [], []
                for j in range(len(loc)):
                    horiz = float(np.linalg.norm(loc[j, :2] - cam[:2]))
                    if horiz > a.max_dist_m or abs(loc[j, 2] - (cam[2] - z0)) > 1.5:
                        continue
                    body = np.array([[loc[j, 0], loc[j, 1], loc[j, 2] + bz] for bz in BODY_Z])   # 로컬 z 0 = 시작 바닥 (pose 의 카메라 z0 ≈ 0.52)
                    uvz = project(T[f], body)
                    fw = T[f][:3, 2][:2] / max(1e-6, np.linalg.norm(T[f][:3, 2][:2]))           # 카메라 광축의 수평 방향 (로컬)
                    rel = loc[j, :2] - cam[:2]
                    near = horiz <= a.near_m and float(rel @ fw) > 0 and math.degrees(math.acos(max(-1.0, min(1.0, float(rel @ fw) / max(horiz, 1e-6))))) <= a.near_deg
                    if near:
                        pts.append([-1.0, -1.0])                                                    # 가까움: 위치 대신 표시만
                        dists.append(horiz)
                        continue
                    ok = (uvz[:, 2] > 0.2) & (uvz[:, 0] >= 0) & (uvz[:, 0] < W) & (uvz[:, 1] >= 0) & (uvz[:, 1] < H + a.v_margin)
                    if ok.any():
                        pts += [[round(float(u), 1), round(float(v), 1)] for u, v, _ in uvz[ok]]
                        dists.append(horiz)
                return pts, dists

            for t in waits:
                pts, dists = vis(t)
                prev = any(vis(f)[0] for f in range(max(0, t - a.prev), t))
                hist = any(vis(f)[0] for f in range(max(0, t - a.hist), t))
                out.append({'split': split, 'episode': i, 'frame': t, 'visible': bool(pts) or prev, 'visible_cur': bool(pts),
                            'visible_prev': bool(prev), 'hist_visible': bool(hist),
                            'n_visible': len(dists), 'min_dist_m': round(min(dists), 2) if dists else None, 'points': pts})
    n = len(out)
    summ = {s: {'wait_frames': sum(r['split'] == s for r in out), 'visible': sum(r['split'] == s and r['visible'] for r in out),
                'visible_cur': sum(r['split'] == s and r['visible_cur'] for r in out),
                'prev_only': sum(r['split'] == s and r['visible'] and not r['visible_cur'] for r in out)} for s in ('train', 'val_seen')}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({'frames': out, 'summary': summ, 'params': {'max_dist_m': a.max_dist_m, 'v_margin': a.v_margin, 'near_m': a.near_m, 'near_deg': a.near_deg, 'prev': a.prev, 'hist': a.hist, 'hfov': HFOV,
                                                                                  'rule': 'visible = 현재 또는 직전 prev 프레임에 사람이 화면 안',
                                                                                  'body_z': BODY_Z, 'rig': a.rig, 'occlusion': 'not checked'}}, indent=1))
    print(json.dumps(summ, ensure_ascii=False), '->', a.out, f'({n} frames)')


if __name__ == '__main__':
    main()
