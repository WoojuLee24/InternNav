"""S3 default 맵 — 캐시된 r_b별 VLN-CE navmesh를 우리 격자의 `esdf`/`navigable`로 바꾼다.

## 왜 이게 필요한가
#04 결론(`reports.md` #04): 밴드 투영(`derive_obstacle_2d`)으로는 navmesh를 못 맞춘다. GT 주변 2 m 안의
셀 단위 일치율이 s8 **75.8%** / 17DRP 95.6%이고, 어긋나는 방향이 **항상 과소 장애물**이라(s8 거짓 승인
44,219셀 vs 거짓 기각 164셀) "갈 수 없는 길"을 정답으로 가르친다. 그래서 default 맵을 navmesh
래스터화로 바꾼다 — habitat 일치 **100%**, IoU **1.0000**.
원인은 #04가 분해했다: s8 거짓 승인의 **66.6%가 "이 높이에 바닥이 없다"**(다층 씬에서 다른 층 공간이
새어 들어온다). `& floor_exists` 수정은 일치율을 90.7%로 올리지만 거짓 기각이 9배가 돼 **채택하지 않았다**.

## 핵심 트릭 — 하류를 하나도 안 고친다
navmesh는 **이미 `r_b`로 깎인 configuration space**다. 그래서 "clearance ≥ r_b"가 아니라 "마스크 안"이
올바른 질문인데, 기존 소비자(`refine_min_move`/`truncate_navigable`/`check_path_navigable`/`_path_ok`)는
전부 clearance 규약을 쓴다. **esdf에 `r_b`를 더해 두면** 두 규약이 정확히 같아진다:

    esdf = (마스크 안) ? 경계까지거리 + r_b : 0

- `hard_ok`(clearance > 0) ⟺ 마스크 안
- `segments_ok`(clearance ≥ r_b) ⟺ 마스크 안
- `truncate_navigable(esdf, r_b + PLAN_MARGIN_M)` ⟺ 마스크 경계에서 1셀 이상 안쪽
  → 스플라인 코너컷 여유(PLAN_MARGIN_M)가 의도 그대로 남는다

## 알아둘 것
- **파이프라인이 habitat에 묶인다.** 캐시(21 KB/맵)라 dataloader worker는 habitat 없이 돌지만,
  Isaac/VLN-PE는 별도 맵이 필요하다.
- `get_topdown_view`는 **단일 높이 절단**이다. `floor_y`(= 에피소드 `reference_path` y = mesh z)를 줘야
  하고, 계단 에피소드는 호출부가 제외해야 한다.
- baseline r_b=0.10에서 GT의 0.78%가 마스크를 벗어난다(s8 66/8459). **최대 깊이가 정확히 1셀(0.050 m)**
  이라 ladder의 `nudge`가 2~3 cm로 흡수한다 — 샘플 손실이 아니다(#04 실험결과 5).

self-check(씬·habitat 불필요): /usr/bin/python scripts/dataset_converters/3dloader_vlnce/navmesh_grid.py
"""

import importlib
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / 'gs_vlnpe')); sys.path.insert(0, str(_HERE))
from esdf_utils import compute_esdf_2d, truncate_navigable  # noqa: E402

DEFAULT_NAVMESH_DIR = 'data/embodiment_aug/navmesh'


def cache_path(navmesh_dir, scene, r_b):
    """S1이 저장한 파일 경로. 이름 규약을 한 곳에만 둔다."""
    return Path(navmesh_dir) / f'{scene}_rb{float(r_b):.2f}.navmesh'


def load_mask(navmesh_dir, scene, r_b, floor_y, origin, cell, shape, _cache={}):
    """캐시된 r_b navmesh -> 우리 격자 bool 마스크. `PathFinder`는 (씬,r_b)당 1회만 로드한다."""
    from habitat_sim.nav import PathFinder
    navmesh_to_our_grid = importlib.import_module('04_calibrate_map').navmesh_to_our_grid
    key = (str(navmesh_dir), scene, round(float(r_b), 4))
    pf = _cache.get(key)
    if pf is None:
        f = cache_path(navmesh_dir, scene, r_b)
        assert f.exists(), (f'{f} 없음 — S1(03_verify_navmesh_gt.py)을 먼저 돌려 r_b={r_b} navmesh를 '
                            f'캐시해라')
        pf = PathFinder(); pf.load_nav_mesh(str(f))
        if len(_cache) >= 16:
            _cache.pop(next(iter(_cache)))
        _cache[key] = pf
    return navmesh_to_our_grid(pf, float(floor_y), origin, cell, shape)


def esdf_from_mask(mask, cell, r_b):
    """configuration-space 마스크 -> clearance 규약 esdf. 모듈 docstring의 트릭.

    `+ r_b`가 핵심이다 — 이것 없이는 `segments_ok`(≥ r_b)가 마스크 안에서도 거짓이 된다.
    """
    base = compute_esdf_2d(~np.asarray(mask, dtype=bool), float(cell))
    return np.where(mask, base + float(r_b), 0.0).astype(np.float32)


def leg_grids(ctx, navmesh_dir, scene, r_b, floor_y, plan_margin_m):
    """`EmbodimentAugmenter._leg_grids`의 navmesh 판. -> (ctx, cell, esdf, navigable)

    반환 규약을 기존 판과 **동일하게** 맞춘다 — 호출부(`follow_waypoints`)는 분기를 모른다.
    """
    cell = float(ctx['args'].cell_m)
    mask = load_mask(navmesh_dir, scene, r_b, floor_y, ctx['origin'], cell, ctx['coverage'].shape)
    esdf = esdf_from_mask(mask, cell, r_b)
    navigable = truncate_navigable(esdf, float(r_b) + float(plan_margin_m)) & ctx['coverage']
    return ctx, cell, esdf, navigable


# ---------------------------------------------------------------------------
# self-check — esdf 변환이 clearance 규약과 정확히 맞는지 (씬·habitat 불필요).
# ---------------------------------------------------------------------------
def _selfcheck():
    from esdf_utils import check_path_navigable
    cell, n, r_b = 0.05, 60, 0.30
    mask = np.zeros((n, n), dtype=bool)
    mask[10:50, 10:50] = True                    # 2 m x 2 m 통행 가능 영역
    esdf = esdf_from_mask(mask, cell, r_b)

    # (1) 마스크 밖은 0 → hard_ok가 막는다
    assert esdf[5, 5] == 0.0, f'마스크 밖 esdf가 0이 아니다 {esdf[5, 5]}'
    # (2) 마스크 경계 셀도 clearance >= r_b (= "마스크 안"과 동치)
    assert esdf[10, 30] >= r_b, f'경계 셀이 r_b 미달 {esdf[10, 30]} < {r_b}'
    # (3) 안쪽으로 갈수록 커진다
    assert esdf[30, 30] > esdf[12, 30] > esdf[10, 30], 'esdf가 안쪽으로 증가하지 않는다'
    # (4) 계획 margin의 실제 효과. `compute_esdf_2d`는 경계 마스크 셀에 **정확히 1셀(0.05)** 을 주므로
    # margin 0.05로는 경계가 안 깎인다(`>=`). 1셀을 실제로 비우려면 2셀(0.10)이 필요하다.
    assert truncate_navigable(esdf, r_b + 0.05)[10, 30], 'margin 0.05는 경계를 남겨야 한다(=)'
    nav2 = truncate_navigable(esdf, r_b + 0.10)
    assert not nav2[10, 30] and nav2[12, 30], 'margin 0.10이 경계 1셀을 비우지 못한다'
    # (5) `check_path_navigable`이 마스크 안 경로를 통과시키고 밖 경로를 막는다
    origin = np.zeros(3)
    inside = np.stack([np.linspace(0.7, 2.2, 40), np.full(40, 1.5)], axis=1)
    chk = check_path_navigable(inside, esdf, origin, cell, r_b)
    assert chk['hard_ok'] and chk['segments_ok'], f'마스크 안 경로가 막혔다 {chk}'
    outside = np.stack([np.linspace(0.1, 2.2, 40), np.full(40, 1.5)], axis=1)
    chk2 = check_path_navigable(outside, esdf, origin, cell, r_b)
    assert not chk2['hard_ok'], '마스크 밖 경로가 통과했다'
    # (6) r_b > 0이면 두 조건이 **모두** "마스크 안"과 동치다 — 이 트릭의 전부다
    for rb in (0.1, 0.30, 0.45):
        e = esdf_from_mask(mask, cell, rb)
        assert np.array_equal(e > 0, mask), f'r_b={rb}: hard_ok가 마스크와 어긋난다'
        assert np.array_equal(e >= rb, mask), f'r_b={rb}: segments_ok가 마스크와 어긋난다'
    # (7) 이름 규약
    assert cache_path('d', 's8pcmisQ38h', 0.1).name == 's8pcmisQ38h_rb0.10.navmesh', '캐시 이름 규약 불일치'

    print('[selfcheck] navmesh_grid 7/7 통과 '
          '(밖=0·경계=r_b·안쪽증가·계획margin·경로통과·경로차단·r_b불변·이름규약)')


if __name__ == '__main__':
    _selfcheck()
