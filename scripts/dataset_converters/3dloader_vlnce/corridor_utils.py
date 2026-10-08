"""GT 폴리라인 주변 **회랑(corridor) 마스크** — leg별 A*가 사람 주석 루트를 벗어나지 못하게 막는다.

## 왜 필요한가
`start→goal` 순수 A*는 **instruction을 모른다**. R2R GT는 사람이 고른 경유지를 지나도록 만들어졌는데
(원본: AMT 작업자가 경로를 보고 지시문 작성) A*는 기하적 최단만 찾으므로 다른 방으로 돌아버리고,
그러면 instruction이 거짓 라벨이 된다. 회랑으로 탐색 공간을 GT 주변으로 제한하면
**"같은 방 안에서 가구를 우회하는 것"은 허용하고 "다른 방으로 새는 것"은 구조적으로 막는다.**

## 구현
새로 만드는 것은 거리장 하나뿐이고 나머지는 기존 부품이다:
`path_cell_sequence`(`esdf_utils.py`, 인접 셀 보장)로 폴리라인을 격자에 찍고
→ `distance_transform_edt`로 **폴리라인까지의 거리장** → 임계로 자르면 회랑.
같은 거리장을 우회량(`detour_max_m`) 측정에도 그대로 쓴다.

⚠️ **회랑 폭은 2셀 이상**이어야 한다. `astar`는 8-이웃이고 대각 코너컷 방지 검사
(`if dx and dy and not (navigable[iy,nx] and navigable[ny,ix]): continue`)가 있어서, 1셀 폭 대각
회랑에서는 대각 이동이 전부 막혀 경로가 끊긴다.

배열 규약은 `esdf_utils` 모듈 docstring을 따른다 — 2D는 `(Ny, Nx)`, 인덱싱은 `arr[iy, ix]`, y 뒤집지 않음.

self-check: /usr/bin/python scripts/dataset_converters/3dloader_vlnce/corridor_utils.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / 'gs_vlnpe'))
from esdf_utils import path_cell_sequence  # noqa: E402

MIN_CORRIDOR_M = 0.10          # 2셀 @ 0.05 m — 대각 코너컷 방지 때문에 이보다 좁으면 안 된다


def dist_to_path_field(path_xy, origin, cell_m, shape):
    """폴리라인까지의 거리장 [m]. -> (Ny, Nx) float32.

    `shape`는 `(Ny, Nx)`로 esdf/navigable과 같아야 한다. 격자 밖으로 나가는 폴리라인 셀은 버린다
    (전부 밖이면 거리장이 전부 inf가 되므로 호출부가 `np.isfinite`로 확인해야 한다).
    """
    ny, nx = shape
    seed = np.ones((ny, nx), dtype=bool)          # True = 아직 폴리라인 아님 (EDT는 0인 곳까지의 거리)
    ij = path_cell_sequence(np.asarray(path_xy, dtype=np.float64)[:, :2], origin, cell_m)
    keep = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    if not keep.any():
        return np.full((ny, nx), np.inf, dtype=np.float32)
    seed[ij[keep][:, 1], ij[keep][:, 0]] = False
    return (distance_transform_edt(seed) * cell_m).astype(np.float32)


def corridor_mask(path_xy, origin, cell_m, shape, width_m):
    """폴리라인 주변 반경 `width_m` 회랑. -> ((Ny,Nx) bool, 거리장)

    거리장을 함께 돌려주는 이유: 호출부가 우회량(경로가 GT에서 얼마나 벗어났나)을 재는 데 같은 장을
    쓴다. 두 번 계산하면 EDT를 두 번 도는 낭비다.
    """
    w = max(float(width_m), MIN_CORRIDOR_M)
    field = dist_to_path_field(path_xy, origin, cell_m, shape)
    return field <= w, field


def detour_amount_m(traj_xy, field, origin, cell_m, with_point=False):
    """경로가 GT 폴리라인에서 벗어난 **최대** 거리 [m]. 격자 밖 점은 무시한다.

    `with_point=True`면 `(최대거리, 그 지점 xy)`를 준다 — 리포트에서 **우회가 어디서 일어났는지**
    표시하는 데 쓴다. 우회량만으로는 그림에서 detour를 구분할 수 없다(선이 GT에서 10 cm = 2셀만
    벌어져도 detour인데 육안으로 안 보인다).
    """
    ny, nx = field.shape
    t = np.asarray(traj_xy, dtype=np.float64)[:, :2]
    ij = np.floor((t - np.asarray(origin)[:2]) / cell_m).astype(int)
    keep = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    if not keep.any():
        return (float('nan'), None) if with_point else float('nan')
    d = np.where(keep, np.nan, np.nan)
    d[keep] = field[ij[keep][:, 1], ij[keep][:, 0]]
    ok = np.isfinite(d)
    if not ok.any():
        return (float('nan'), None) if with_point else float('nan')
    k = int(np.nanargmax(np.where(ok, d, -np.inf)))
    return (float(d[k]), t[k]) if with_point else float(d[k])


# ---------------------------------------------------------------------------
# self-check — 씬·렌더러 없이 합성 격자로 회랑의 성질을 확인한다.
# ---------------------------------------------------------------------------
def _selfcheck():
    cell, n = 0.05, 80
    origin = np.zeros(3)
    # y=1.0 m 에 가로로 뻗은 직선 폴리라인
    line = np.stack([np.linspace(0.5, 3.0, 40), np.full(40, 1.0)], axis=1)

    mask, field = corridor_mask(line, origin, cell, (n, n), 0.30)
    # 폴리라인 위는 거리 0
    assert field[int(1.0 / cell), int(1.5 / cell)] < 1e-6, '폴리라인 위 거리가 0이 아니다'
    # 수직 거리가 실제 거리와 맞나 (0.5 m 위)
    d = field[int(1.5 / cell), int(1.5 / cell)]
    assert abs(d - 0.5) < cell * 1.5, f'수직 거리 {d:.3f} != 0.5'
    # 회랑 폭: 0.30 m 안은 포함, 0.5 m 밖은 제외
    assert mask[int(1.2 / cell), int(1.5 / cell)], '0.20 m 지점이 회랑에서 빠졌다'
    assert not mask[int(1.6 / cell), int(1.5 / cell)], '0.60 m 지점이 회랑에 들어왔다'
    # 폴리라인 밖(왼쪽 끝 이전)은 끝점으로부터의 거리라 회랑에서 빠져야 한다
    assert not mask[int(1.0 / cell), int(0.1 / cell)], '폴리라인 시작 전이 회랑에 들어왔다'

    # 최소 폭 강제 — 0으로 줘도 2셀은 확보돼야 대각 이동이 막히지 않는다
    narrow, _ = corridor_mask(line, origin, cell, (n, n), 0.0)
    band = narrow[:, int(1.5 / cell)].sum()
    assert band >= 3, f'최소 폭 강제 실패: 세로 폭 {band}셀'

    # 우회량: 폴리라인 그대로면 ~0, 0.4 m 띄우면 ~0.4
    assert detour_amount_m(line, field, origin, cell) < cell * 1.5, 'GT 자신의 우회량이 0이 아니다'
    off = line + np.array([0.0, 0.4])
    assert abs(detour_amount_m(off, field, origin, cell) - 0.4) < cell * 1.5, '우회량 측정이 틀렸다'
    # 최대 이탈 지점도 함께 — 중간만 0.6 m 튀어나온 경로에서 그 위치를 찾아야 한다
    bump = line.copy(); bump[20, 1] += 0.6
    dmax, pt = detour_amount_m(bump, field, origin, cell, with_point=True)
    assert abs(dmax - 0.6) < cell * 2, f'최대 이탈량 {dmax:.3f} != 0.6'
    assert pt is not None and abs(pt[0] - line[20, 0]) < cell * 2, f'최대 이탈 지점이 틀림 {pt}'

    # 격자 밖 폴리라인 → 전부 inf (호출부가 감지해야 하는 실패 모드)
    far, _ = corridor_mask(line + 100.0, origin, cell, (n, n), 0.30)
    assert not far.any(), '격자 밖 폴리라인이 회랑을 만들었다'

    print('[selfcheck] corridor_utils 10/10 통과 '
          '(거리0·수직거리·폭포함·폭제외·시작전제외·최소폭·우회량·이탈량·이탈지점·격자밖)')


if __name__ == '__main__':
    _selfcheck()
