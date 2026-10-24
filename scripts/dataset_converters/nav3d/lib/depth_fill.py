"""vln_ce 릴리스와 같은 depth 구멍 채우기 (2026-10-15 실측으로 맞춤).

habitat 은 표면이 없는 방향의 depth 를 0 으로 둔다(mesh 구멍). vln_ce 릴리스 depth 는 이 구멍을 **가장자리에서
약 20 px 까지만** 채우고 그보다 안쪽은 0 으로 남긴다(구멍 RGB 는 릴리스도 검정 — 뒷면을 그린 게 아니다).

재현 규칙: 0 인 픽셀을 3×3 이웃 중 **가장 작은 유효 값**으로 채우기를 19 번 반복.
- 채운 영역: 릴리스와 IoU 0.954 / 0.994 (1LXt ep119 f78 / 1pXnu ep95 f54).
- 채운 값: mm 까지 같지는 않다(중앙 오차 4~7 mm) — 릴리스 생성기의 정확한 방법은 미확인.
- 효과(같은 두 프레임): 1 m 넘게 틀린 픽셀 14.75 → 0.71 % / 9.77 → 0.12 %.

입력·출력은 uint16 mm (0 = 무효) — 릴리스 png 와 같은 단위에서 채운다.
self-check: `/usr/bin/python scripts/dataset_converters/nav3d/lib/depth_fill.py`
"""

import numpy as np

FILL_ITERS = 19
FILL_KERNEL = 3


def fill_depth_holes(depth_mm: np.ndarray, iters: int = FILL_ITERS, k: int = FILL_KERNEL) -> np.ndarray:
    """depth_mm (H,W) 정수 mm, 0 = 구멍 -> 구멍을 가장자리부터 iters 번 3×3 최소 이웃값으로 채운 사본."""
    from scipy import ndimage
    d = np.asarray(depth_mm).astype(np.int64)
    big = 10 ** 9                       # 센티널 (int64 최댓값은 scipy grey_erosion 에서 깨진다)
    fp = np.ones((k, k), bool)
    for _ in range(iters):
        hole = d == 0
        if not hole.any():
            break
        nb = ndimage.grey_erosion(np.where(hole, big, d), footprint=fp, mode='constant', cval=big)
        new = hole & (nb < big)
        if not new.any():
            break
        d[new] = nb[new]
    return d.astype(depth_mm.dtype)


if __name__ == '__main__':
    a = np.full((40, 40), 1000, np.uint16)
    a[5:15, 5:15] = 0          # 작은 구멍: 다 채워진다
    a[:, 30:] = 0              # 넓은 구멍(폭 10 이지만 테두리 쪽은 경계 밖이 없음)
    b = np.zeros((60, 60), np.uint16)
    b[:, :5] = 2000            # 왼쪽만 유효 -> 19 칸까지만 채운다
    fa, fb = fill_depth_holes(a), fill_depth_holes(b)
    assert (fa[5:15, 5:15] == 1000).all()
    assert (fb[:, 5:24] == 2000).all() and (fb[:, 24:] == 0).all(), (fb[0, 20:30])
    print('[depth_fill] self-check OK')
