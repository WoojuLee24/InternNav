"""다듬은 경로 평가와 사용 가능 판정 U1~U5 (03 · 04 · 04d 가 같이 쓴다).

`03_refine_paths.py` (one · gt_metrics · stairs_points) 와 `04_export_paths.py` (load_holes · near_hole · segments · judge)
에서 옮겼다 (코드 그대로). 두 파일은 이 이름들을 다시 import 해서 그대로 노출한다.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gs_vlnpe'))
import nav3d_path as npth  # noqa: E402
from esdf_utils import discrete_frechet_distance  # noqa: E402


DOOR_WIDTH_M = 0.8


def gt_metrics(X, gt):
    fr = discrete_frechet_distance(X[:, :2], gt[:, :2])
    return {'frechet_m': fr, 'same_route': bool(fr < DOOR_WIDTH_M)}


def stairs_points(sc, X):
    """평가용 계단 라벨 — 다듬는 방식과 무관하게 기본 NavScene 라벨(점 아래 삼각형)로 잰다(split · none 같은 자)."""
    _, idx = sc.surf.track_idx(X[:, :2], z0=X[0, 2])
    return np.where(idx >= 0, sc.stairs_tri[np.maximum(idx, 0)], False)


def one(sc, P, md, r_b, cfg, gt=None, sc_ref=None):
    """sc = 평가용(기본 계단 라벨), sc_ref = 다듬기용(없으면 sc). --stairs_mode none 이면 sc_ref 의 계단이 비어 있다."""
    t = time.time()
    r = npth.refine(sc_ref or sc, P, cfg)
    e0 = npth.evaluate(sc, r['raw'], stairs_points(sc, r['raw']), r['raw_on'], md, r_b)
    e1 = npth.evaluate(sc, r['traj'], stairs_points(sc, r['traj']), r['traj_on'], md, r_b)
    row = {'sec': time.time() - t, 'n_stairs_runs': r['n_stairs_runs'], 'n_fallback': r['n_fallback'],
           **{f'raw_{k}': v for k, v in e0.items()}, **{f'ref_{k}': v for k, v in e1.items()}}
    if gt is not None:
        row.update({f'raw_gt_{k}': v for k, v in gt_metrics(r['raw'], gt).items()})
        row.update({f'ref_gt_{k}': v for k, v in gt_metrics(r['traj'], gt).items()})
        row['gt_turn_per_m_deg'] = npth.turn_per_m(gt)
    return row, r


def load_holes(path):
    out = []
    for r in csv.DictReader(open(path)):
        if r['flag'] == 'CRITICAL' and r['kind'].startswith('mesh'):
            out.append((float(r['x']), float(r['y']), float(r['z']), float(r['inscribed_diam_m']) / 2))
    return np.array(out) if out else np.zeros((0, 4))


def near_hole(X, holes, r_b):
    if len(holes) == 0:
        return 0
    d = np.linalg.norm(X[:, None, :2] - holes[None, :, :2], axis=2)
    dz = np.abs(X[:, None, 2] - holes[None, :, 2])
    hit = (d < r_b + holes[None, :, 3]) & (dz < 0.5)
    return int(hit.any(1).sum())


def segments(stairs_mask, floor_z):
    """점별 계단 표시 -> [{kind, i0, i1, floor_z}] (gs_vlnpe 03mf 와 같은 키)."""
    segs, i0 = [], 0
    for i in range(1, len(stairs_mask) + 1):
        if i == len(stairs_mask) or stairs_mask[i] != stairs_mask[i0]:
            segs.append({'kind': 'stairs' if stairs_mask[i0] else 'floor', 'i0': int(i0), 'i1': int(i - 1),
                         'floor_z': float(floor_z[i0])})
            i0 = i
    return segs


def judge(row, X, kind, holes, r_b, args):
    why = []
    if not row.get('found', True):
        return ['U1 경로 없음']
    if row['ref_on_frac'] < 1.0:
        why.append('U2 navmesh 밖')
    if row['ref_mesh_clear_lt_rb_frac'] > 0 or row['ref_mesh_clear_min_m'] < 0.02:
        why.append('U3 r_b 미달')
    if near_hole(X, holes, r_b) > 0:
        why.append('U4 큰 mesh 구멍 옆')
    if kind == 'flat' and (np.ptp(X[:, 2]) >= 0.5 or row['ref_length_m'] < args.flat_min_m):
        why.append('U5 평지 아님/짧음')
    return why
