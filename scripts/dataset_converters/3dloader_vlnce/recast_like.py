"""recast-like — **칸마다 바닥을 찾는** 2D 지도 생성기 (occ npz만 사용, habitat 불필요).

## 왜
#04 실측: 우리 밴드 맵과 habitat navmesh의 차이는 두께가 아니라 **질문의 구조**다. 밴드는
"높이 `floor_y+0.2~1.5`에 장애물이 없나"만 묻고 **"발 디딜 바닥이 있나"를 묻지 않는다** — 다층 씬에서
다른 층 허공이 통행 가능이 된다(s8 거짓 승인의 66.6%). recast는 칸(voxel 기둥)마다 실제 바닥면(span)을
찾고 그 위 머리 공간·이웃 단차·작은 섬을 검사한다. 여기서 그 논리를 우리 voxel 격자 위에 흉내 낸다.

## 단계 (recast 대응)
| 단계 | 우리 구현 | recast 원본 |
|---|---|---|
| ① 바닥 | `[floor_y−0.5, +0.2]` 창에서 가장 높은 점유 voxel = `z_f`. 없으면 차단 | span top (하단 0.5 = ref 래스터 `eps`, 상단 0.2 = `agent_max_climb`) |
| ② 머리 공간 | `(z_f+0.2, z_f+1.5]`에 점유 있으면 장애물 — **자기 바닥에 앵커** | span 위 `agent_height` clearance |
| ③ 단차 | 4-이웃과 `|Δz_f| > 0.2`면 장애물 경계선 | `agent_max_climb` step 연결성 + ledge 제거 |
| ④ 침식 | `esdf(장애물) ≥ r_b`. ⚠️ **장애물에서만** 침식, 바닥 유무는 침식 **후** AND | `agent_radius` erosion |
| ⑤ 섬 | 연결 성분 < 400셀(1.0 m², `regionMinSize=20`²) 삭제 | `region_min_size` |

④가 핵심 설계점이다 — skin 샘플링(3M점)이라 바닥에 드문 구멍이 있는데, 구멍에서 침식하면 구멍 하나가
주변 ~13칸을 지운다(스펙클 증폭). 바닥 구멍은 `binary_closing(3×3)`으로만 메운다.

## 안 하는 것 (근거)
- **45° 경사 검사**: 칸당 0.05 m 기준으로 걸면 계단 디딤판(단차 ~0.17)을 전부 죽인다 — ③이 절벽 분리만 담당.
- **`cell_height=0.2` 양자화**: ref navmesh에 이미 구워져 있고 우리 0.05가 더 정밀 — 맞출 메커니즘이 없다.
- **경계 1칸(R 버킷)**: 격자 반올림 자체라 어떤 방법으로도 못 없앤다(17DRP 거짓 승인의 80%).

self-check(씬·habitat 불필요): /usr/bin/python scripts/dataset_converters/3dloader_vlnce/recast_like.py
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / 'gs_vlnpe')); sys.path.insert(0, str(_HERE))
from esdf_utils import compute_esdf_2d, truncate_navigable  # noqa: E402

# 전부 배포 navmesh 직독값 (#04와 동일). BELOW_M만 ref 래스터의 `eps=0.5`에서 온다.
H_NAV_M, H_OBS_M = 0.20, 1.50        # agent_max_climb / agent_height
BELOW_M, TOP_M = 0.50, 0.20          # 바닥 탐색 창: floor_y − BELOW_M ~ floor_y + TOP_M
MIN_REGION_CELLS = 400               # habitat regionMinSize=20, doc: regionMinSize=sqrt(regionMinArea)


def column_floor(occ, origin, cell, floor_y, below_m=BELOW_M, top_m=TOP_M):
    """칸마다 바닥 voxel을 찾는다. -> (kf (Ny,Nx) int32 · 바닥 없으면 −1, floor_ok (Ny,Nx) bool)

    skin voxelization이라 바닥면도 점유 voxel로 존재한다 — 창에서 **가장 높은** 점유가 그 칸의 바닥이다
    (러그·낮은 단의 윗면 = recast의 span top 의미론).
    """
    nz = occ.shape[2]
    k_lo = max(int(np.ceil((floor_y - below_m - origin[2]) / cell)), 0)
    k_hi = min(int(np.floor((floor_y + top_m - origin[2]) / cell)), nz - 1)
    if k_lo > k_hi:
        shp = occ.shape[1::-1]
        return np.full(shp, -1, dtype=np.int32), np.zeros(shp, dtype=bool)
    win = occ[:, :, k_lo:k_hi + 1]
    ok = win.any(axis=2)
    idx = win.shape[2] - 1 - win[:, :, ::-1].argmax(axis=2)     # 마지막 True = 가장 높은 점유
    kf = np.where(ok, k_lo + idx, -1)
    return kf.T.astype(np.int32), ok.T


def recast_mask(grid, floor_y, r_b, ledge=True, region=True, top_m=TOP_M, below_m=BELOW_M,
                despike=True, start_xy=None):
    """recast-like navigable 마스크. -> (Ny,Nx) bool. 단계는 모듈 docstring 표.

    `ledge`/`region`/`despike`/`start_xy`는 ablation 스위치. v2 추가 둘:
    `despike` — 바닥 voxel이 빠진 칸이 창 아래 면으로 새면(kf 스파이크) 이웃 중앙값으로 보간.
    이게 없으면 s8처럼 바닥이 고르지 않은 씬에서 ledge seam이 방 한가운데를 지나가 GT를 차단한다.
    `start_xy` — 주어지면 **그 점이 속한 연결 성분만** 남긴다(recast의 도달 가능성 근사).
    "바닥처럼 생긴 면"(navmesh가 걸을 수 없다고 판정한 낮은 면)은 seam 너머라 여기서 떨어진다.
    """
    import warnings
    from scipy.ndimage import binary_closing, label
    occ, origin, cell = grid['occ'], grid['origin'], float(grid['cell'])
    nx, ny, nz = occ.shape
    kf, ok = column_floor(occ, origin, cell, floor_y, below_m, top_m)

    if despike:
        climb = int(round(H_NAV_M / cell))
        kfn = np.where(ok, kf.astype(np.float32), np.nan)
        p = np.pad(kfn, 1, constant_values=np.nan)
        views = [p[1 + dy:1 + dy + ok.shape[0], 1 + dx:1 + dx + ok.shape[1]]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')            # 전-이웃 NaN 칸의 nanmedian 경고
            med = np.nanmedian(np.stack(views), axis=0)
        spike = ok & np.isfinite(med) & (np.abs(kf - med) > climb)
        kf = np.where(spike, np.rint(np.nan_to_num(med)).astype(np.int32), kf)

    # ② 머리 공간: (z_f + h_nav, z_f + h_obs]에 점유가 있나 — cumsum으로 칸별 구간 카운트.
    # cumsum은 floor_y·r_b와 무관하므로 **씬당 1회**만 만들어 grid dict에 캐시한다(s8 96 MB, ~180 ms).
    C = grid.get('_recast_csum')
    if C is None:
        C = np.zeros((ny, nx, nz + 1), dtype=np.int32)
        C[:, :, 1:] = occ.transpose(1, 0, 2).cumsum(axis=2)
        grid['_recast_csum'] = C
    gap, top = int(round(H_NAV_M / cell)), int(round(H_OBS_M / cell))
    ka = np.clip(np.where(ok, kf + gap + 1, 0), 0, nz)
    kb = np.clip(np.where(ok, kf + top + 1, 0), 0, nz)
    cnt = (np.take_along_axis(C, kb[..., None], axis=2)
           - np.take_along_axis(C, ka[..., None], axis=2))[..., 0]
    obstacle = ok & (cnt > 0)

    # ③ 단차: 이웃 바닥과 |Δz_f| > max_climb면 절벽 경계 (계단 디딤판 ~0.17은 통과)
    if ledge:
        climb = int(round(H_NAV_M / cell))
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            nb_kf, nb_ok = np.roll(kf, sh, axis=ax), np.roll(ok, sh, axis=ax)
            edge = np.zeros_like(ok)                # roll이 감아온 반대편 가장자리는 비교 제외
            if ax == 0:
                edge[0 if sh == 1 else -1, :] = True
            else:
                edge[:, 0 if sh == 1 else -1] = True
            obstacle |= ok & nb_ok & ~edge & (np.abs(kf - nb_kf) > climb)

    # ④ 침식은 장애물에서만 — 바닥 구멍에서 침식하면 구멍 하나가 ~13칸을 지운다(스펙클 증폭)
    nav = truncate_navigable(compute_esdf_2d(obstacle, cell), float(r_b))
    # closing은 pad-edge로 — 기본 border_value=0이면 배열 가장자리 1칸 링이 침식돼 사라진다
    cl = binary_closing(np.pad(ok, 1, mode='edge'), structure=np.ones((3, 3), dtype=bool))[1:-1, 1:-1]
    nav &= cl & grid['coverage']

    # ⑤ 작은 섬 제거 (8-연결)
    if region:
        lab, n = label(nav, structure=np.ones((3, 3), dtype=int))
        if n:
            sizes = np.bincount(lab.ravel())
            nav &= (sizes >= MIN_REGION_CELLS)[lab] & (lab > 0)

    # ⑥ 시작점 연결성: 시작점이 속한 성분만 (0.5 m 안에서 스냅). 성분을 못 찾으면 그대로 둔다.
    if start_xy is not None and nav.any():
        lab, _n = label(nav, structure=np.ones((3, 3), dtype=int))
        ij = np.floor((np.asarray(start_xy, dtype=np.float64)[:2] - origin[:2]) / cell).astype(int)
        sid = lab[ij[1], ij[0]] if (0 <= ij[1] < ny and 0 <= ij[0] < occ.shape[0]) else 0
        if sid == 0:
            ys, xs = np.nonzero(nav)
            d2 = (ys - ij[1]) ** 2 + (xs - ij[0]) ** 2
            k = int(np.argmin(d2))
            if d2[k] <= (0.5 / cell) ** 2:
                sid = lab[ys[k], xs[k]]
        if sid:
            nav = nav & (lab == sid)
    return nav


# ---------------------------------------------------------------------------
# self-check — 합성 occ로 단계별 동작을 검사한다 (씬·habitat 불필요).
# ---------------------------------------------------------------------------
def _selfcheck():
    nx, ny, nz, cell = 80, 40, 45, 0.05
    origin = np.zeros(3)
    fy = origin[2] + (12 + 0.5) * cell           # 바닥 voxel k=12의 중심 높이 = 0.625. 창 = k 3..16
    occ = np.zeros((nx, ny, nz), dtype=bool)
    occ[:, :, 12] = True                         # 평평한 바닥
    occ[5:10, 10:30, 12] = False                 # 빈 기둥 패치(허공) — F. 전폭 띠로 만들면 왼쪽이 섬이 된다
    occ[20, :, :] = True                         # 벽 (모든 높이)
    occ[30:34, 10:14, 27] = True                 # 테이블 상판 (+0.75 m) — 머리 공간 실패
    occ[40:44, :, 14] = True                     # 러그 (+0.10 m, 창 안·단차 2칸≤climb) — 윗면이 새 바닥
    occ[50, 20, 12] = False                      # 바닥 구멍 1칸 — closing이 메워야 함
    occ[60:72, 10:22, 12] = False                # 침하 테라스(12×12=144칸 < 400): 바닥을 k=3으로 내림
    occ[60:72, 10:22, 3] = True                  #   Δ=9칸 > climb 4칸 → ledge 경계
    occ[55, 30, 12] = False                      # kf 스파이크: 바닥 voxel이 빠지고 아래 면(k=5)으로 샘
    occ[55, 30, 5] = True                        #   despike가 이웃 중앙값(12)으로 보간해야 함

    grid = dict(occ=occ, origin=origin, cell=cell, coverage=np.ones((ny, nx), dtype=bool))
    kf, ok = column_floor(occ, origin, cell, fy)
    assert ok[15, 2] and kf[15, 2] == 12, '평바닥 탐지 실패'
    assert not ok[15, 7], '빈 기둥이 F로 안 잡힘'                                  # (1) F
    assert kf[15, 42] == 14, '러그 윗면이 바닥으로 안 잡힘'                         # span-top 의미론

    nav = recast_mask(grid, fy, 0.10)
    assert nav[30, 2], '평바닥이 통행 불가'                                        # (2)
    assert not nav[15, 7], '허공이 통행 가능'                                      # (3)
    # `esdf >= r_b` 규약이라 벽(1칸)+양옆 1칸이 지워지고 거리 정확히 0.10인 2칸째는 남는다
    # — recast는 ceil(r/cell)=2칸을 지우므로 이 반 칸이 R 버킷(경계 1칸)의 한 원인이다.
    assert not nav[15, 19:22].any() and nav[15, 18] and nav[15, 22], '벽 침식 규약이 예상과 다름'
    assert not nav[12, 31], '테이블 아래가 통행 가능'                              # (5) 머리 공간
    assert nav[30, 42], '러그 위가 통행 불가'                                      # (6)
    assert nav[20, 50], '바닥 구멍 1칸이 closing으로 안 메워짐'                    # (7) 스펙클
    assert not nav[10:22, 60:72].any(), '침하 테라스(144칸)가 안 지워짐'            # (8) ledge+region

    # ablation: region을 끄면 단의 내부(seam 1 + 침식 2 = 양쪽 3칸 안쪽)가 살아난다
    nav_nr = recast_mask(grid, fy, 0.10, region=False)
    assert nav_nr[13:19, 63:69].any(), 'region=False인데 테라스 내부가 없다 — ledge가 과도'
    nav_nl = recast_mask(grid, fy, 0.10, ledge=False, region=False)
    assert nav_nl[10:22, 60:72].sum() > nav_nr[10:22, 60:72].sum(), 'ledge 스위치가 무효과'

    # despike: 스파이크 칸이 보간돼 걷게 되고, 끄면 ledge seam이 주변을 차단한다
    assert nav[30, 55], 'kf 스파이크가 despike로 보간되지 않음'
    nav_nd = recast_mask(grid, fy, 0.10, despike=False)
    assert not nav_nd[30, 54:57].all(), 'despike=False인데 스파이크 seam이 없다 — 테스트 무효'

    # 시작점 연결성: 본층 시작점을 주면 (ledge로 분리된) 테라스가 region=False여도 떨어진다
    sx = origin[0] + (15 + 0.5) * cell
    sy = origin[1] + (35 + 0.5) * cell
    nav_s = recast_mask(grid, fy, 0.10, region=False, start_xy=(sx, sy))
    assert not nav_s[10:22, 60:72].any(), '시작점 연결성이 테라스를 못 떨어뜨림'
    assert nav_s[30, 2], '시작점 연결성이 본층을 지움'

    print('[selfcheck] recast_like 13/13 통과 '
          '(평바닥·허공F·러그span·벽침식·테이블·러그통행·구멍closing·섬제거·region끔·ledge끔'
          '·despike·despike끔·시작점연결성)')
    return 0


if __name__ == '__main__':
    sys.exit(_selfcheck())
