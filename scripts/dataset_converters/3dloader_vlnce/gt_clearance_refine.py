"""GT 여유 매칭 refine — **벽에 안 붙으면서 GT를 잘 따라가는** 경로용 refiner.

## 왜
우리 ladder가 생성하는 경로(`refine_min_move`)는 여유가 `r_b`에 붙는다(여유<0.10 비율이 GT의 2~4배 —
[[260820_gt_wall_clearance_result]]). 원본 vln_ce GT는 벽에서 넉넉하게 가는데(추가 여유 median
0.36/0.24 m) 이것이 **학습 라벨의 스타일**이라, r_b 증강 시 스타일 차이가 학습 분포에 들어간다.

## 아이디어 — "GT를 따라간다"에 여유 스타일까지 포함
점마다 목표 여유 = **GT가 그 근처에서 확보했던 여유**(clamp: 하한 r_b · 상한 r_b+cap):
- 좁은 문(GT도 여유 없음) → 목표 낮음 → **안 밈 → 우회 없음** (고정 마진 방식의 약점 회피)
- 넓은 방(GT 넉넉) → 목표 높음 → 여유 높은 영역엔 GT 자신이 있으므로 **GT 쪽으로 이동**

## 기존 양극단과의 관계
`refine_min_move`(목표 = r_b 고정, 벽에 붙음) ↔ **refine_match_gt (목표 = GT 로컬 여유)** ↔
`greedy_refine`(argmax = 최대, GT에서 멀어짐 — N1식, 기각됨). A* 단계에서 미는 `clearance_weight`는
경로 자체가 바뀌어 기각됐다(03d) — refine 단계는 route를 유지하고 점만 민다.

self-check(씬·habitat 불필요): /usr/bin/python scripts/dataset_converters/3dloader_vlnce/gt_clearance_refine.py
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / 'gs_vlnpe')); sys.path.insert(0, str(_HERE))
from esdf_utils import world_to_cell  # noqa: E402

CAP_M = 0.30      # 목표 여유 상한 = r_b + CAP_M. GT가 아주 넓은 곳에 있어도 횡 이동을 이 이상 안 만든다


def refine_match_gt(path_xy, esdf, origin, cell_m, gt_xy, r_b,
                    cap_m=CAP_M, radius_m=0.30, fix_endpoints=True, corr=None, fixed_m=None):
    """점마다 목표 여유 = clamp(GT 최근접점의 esdf, r_b, r_b+cap_m)로 **최소 이동**. -> (N,2)

    `refine_min_move`(목표 = r_b 고정)의 일반화 — 목표만 GT의 로컬 여유로 바꾼 것.
    이동 실패 시 폴백: `esdf >= r_b`(유효성 우선) → 창 argmax. `corr`가 있으면 후보를 회랑 안으로 제한.
    `fixed_m`을 주면 목표 = r_b + fixed_m 상수 (ablation용 — GT 매칭 없이 고정 마진).
    """
    from scipy.spatial import cKDTree
    p = np.asarray(path_xy, dtype=np.float64)[:, :2].copy()
    gt = np.asarray(gt_xy, dtype=np.float64)[:, :2]
    ny, nx = esdf.shape
    rad = max(1, int(round(radius_m / cell_m)))

    if fixed_m is not None:
        c_tgt = np.full(len(p), float(r_b) + float(fixed_m))
    else:
        # 목표 여유: GT 최근접점의 esdf (격자 밖 GT점은 r_b로 폴백)
        _d, idx = cKDTree(gt).query(p)
        gij = world_to_cell(gt[idx], origin, cell_m)
        ok_g = (gij[:, 0] >= 0) & (gij[:, 0] < nx) & (gij[:, 1] >= 0) & (gij[:, 1] < ny)
        c_gt = np.full(len(p), float(r_b))
        c_gt[ok_g] = esdf[gij[ok_g][:, 1], gij[ok_g][:, 0]]
        c_tgt = np.clip(c_gt, float(r_b), float(r_b) + float(cap_m))

    ij = world_to_cell(p, origin, cell_m)
    lo, hi = (0, len(p)) if not fix_endpoints else (1, len(p) - 1)
    for i in range(lo, hi):
        ix, iy = ij[i]
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        if esdf[iy, ix] >= c_tgt[i]:
            continue                                   # 이미 목표 충족 — 안 건드림 (최소 이동)
        y0, y1 = max(0, iy - rad), min(ny, iy + rad + 1)
        x0, x1 = max(0, ix - rad), min(nx, ix + rad + 1)
        win = esdf[y0:y1, x0:x1]
        allow = np.ones_like(win, dtype=bool) if corr is None else corr[y0:y1, x0:x1]
        cand = np.argwhere((win >= c_tgt[i]) & allow)
        if not len(cand):
            cand = np.argwhere((win >= float(r_b)) & allow)     # 폴백 1: 유효성(r_b)만이라도
        if len(cand):
            d2 = (cand[:, 1] + x0 - ix) ** 2 + (cand[:, 0] + y0 - iy) ** 2
            by, bx = cand[np.argmin(d2)]
        else:
            by, bx = np.unravel_index(np.argmax(np.where(allow, win, -1.0)), win.shape)  # 폴백 2
        p[i] = [origin[0] + (x0 + bx + 0.5) * cell_m, origin[1] + (y0 + by + 0.5) * cell_m]
    return p


# ---------------------------------------------------------------------------
# self-check — 합성 esdf로 목표-여유 기제를 검사한다 (씬·habitat 불필요).
# ---------------------------------------------------------------------------
def _selfcheck():
    from esdf_utils import compute_esdf_2d
    cell, ny, nx = 0.05, 60, 100
    origin = np.zeros(3)
    obstacle = np.zeros((ny, nx), dtype=bool)
    obstacle[0, :] = True                     # 아래쪽 긴 벽 (y=0.025)
    obstacle[:, 60:63] = False
    obstacle[0:14, 60] = True                 # 좁은 문: x=3.0에 벽 기둥(문폭 위쪽만 열림)
    esdf = compute_esdf_2d(obstacle, cell)
    r_b = 0.10

    def line(y, x0=0.5, x1=4.5, n=80):
        return np.stack([np.linspace(x0, x1, n), np.full(n, y)], axis=1)

    # GT는 벽에서 0.60 m 떨어져 간다(여유 큼) → 벽에 붙은 경로(0.15)가 GT 쪽으로 밀려야 한다.
    # 단 문 기둥(x≈3.0) 근처는 GT 자신의 여유도 낮으므로 **안 미는 게 맞다** — 그게 이 refiner의 요점.
    gt_far = line(0.60)
    hug = line(0.15)
    out = refine_match_gt(hug, esdf, origin, cell, gt_far, r_b, cap_m=0.30, radius_m=0.60)
    open_area = np.abs(out[:, 0] - 3.0) > 0.6
    mid = out[10:70][open_area[10:70]]
    assert (mid[:, 1] > 0.30).all(), f'개방 구간에서 안 밀림 (y min {mid[:,1].min():.2f})'
    assert (mid[:, 1] <= 0.60 + 0.10).all(), 'cap을 넘어 과도하게 밀림'
    near_door = out[np.abs(out[:, 0] - 3.0) < 0.15]
    assert (near_door[:, 1] < 0.30).any(), '문 기둥 근처(GT 여유 낮음)까지 밀어버림'
    assert np.allclose(out[0], hug[0]) and np.allclose(out[-1], hug[-1]), 'endpoints가 움직임'

    # GT 자신도 벽에 붙어 있으면(좁은 문 스타일) 목표가 낮아 **안 밀어야** 한다 (r_b 충족 구간 한정 —
    # 문 기둥 바로 옆은 여유 < r_b라 유효성 하한이 미는 게 맞다)
    gt_hug = line(0.15)
    out2 = refine_match_gt(line(0.15), esdf, origin, cell, gt_hug, r_b)
    far_door = np.abs(line(0.15)[:, 0] - 3.0) > 0.6
    assert np.allclose(out2[far_door], line(0.15)[far_door]), 'GT 여유가 낮은 곳을 불필요하게 밀었다'

    # 여유 < r_b인 점은 GT가 어떻든 r_b까지는 민다 (유효성 하한)
    out3 = refine_match_gt(line(0.04), esdf, origin, cell, gt_hug, r_b, radius_m=0.40)
    ij = world_to_cell(out3[10:70], origin, cell)
    assert (esdf[ij[:, 1], ij[:, 0]] >= r_b).all(), 'r_b 하한이 안 지켜짐'

    # corr 제한: 회랑 밖 셀로는 못 민다
    corr = np.zeros((ny, nx), dtype=bool); corr[:8, :] = True          # y < 0.40만 허용
    out4 = refine_match_gt(line(0.15), esdf, origin, cell, gt_far, r_b, radius_m=0.60, corr=corr)
    assert (out4[10:70, 1] < 0.42).all(), 'corr 제한이 안 걸림'

    # 폴백: 창 안에 목표 충족 셀이 없으면 r_b 충족 셀로라도 (좁은 문 근처)
    door = np.stack([np.linspace(2.9, 3.1, 9), np.full(9, 0.85)], axis=1)
    out5 = refine_match_gt(door, esdf, origin, cell, gt_far, r_b, cap_m=0.30, radius_m=0.15)
    ij5 = world_to_cell(out5, origin, cell)
    assert (esdf[ij5[:, 1], ij5[:, 0]] > 0).all(), '폴백이 유효성을 깨뜨림'

    print('[selfcheck] gt_clearance_refine 6/6 통과 '
          '(GT쪽으로밈·cap상한·endpoints·저여유GT안밈·r_b하한·corr제한·폴백)')
    return 0


if __name__ == '__main__':
    sys.exit(_selfcheck())
