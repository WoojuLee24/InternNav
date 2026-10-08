"""2dloader 검증용 그리기 헬퍼. 색·범례 규약은 3dloader `validate_augment.py`를 그대로 승계한다
(`RB_COLORS`, `chip`, BEV 3색). BEV는 **로봇 전방이 위**인 이미지 좌표 그대로 그린다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent / '3dloader_vlnce'), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from validate_augment import RB_COLORS, chip  # noqa: E402,F401  (재수출 — 색/범례 단일 출처)

GT_COLOR = (255, 230, 60)       # 원본 GT path (노랑)
NEW_COLOR = (0, 255, 90)        # 재계획 path (초록)
START_COLOR = (255, 255, 255)
GOAL_COLOR = (255, 0, 255)      # 저장된 원본 goal
ADJ_COLOR = (90, 190, 255)      # 조정된 goal
OBST_COLOR = (255, 140, 0)      # 합성 장애물 footprint


CSPACE_COLOR = (140, 40, 40)     # r_b 때문에 못 가는 free 칸 (C-space 확장분)
FLOORFREE_COLOR = (60, 75, 60)   # 바닥 반환으로 판 free (raycast free와 구분)


def bev_rgb(bev: np.ndarray) -> np.ndarray:
    """3-state BEV -> uint8 RGB. 0=unknown 검정 / 0.5=free 회색 / 1=occupied 빨강."""
    img = np.zeros((*bev.shape, 3), np.uint8)
    img[(bev >= 0.25) & (bev < 0.75)] = (90, 90, 90)
    img[bev >= 0.75] = (255, 70, 70)
    return img


def bev_rgb_planning(m: dict, nav_coarse: np.ndarray = None) -> np.ndarray:
    """**A*가 실제로 경로를 고른 그 격자**를 그린다.

    `bev_rgb_cspace`는 4.46 cm 세밀 격자를 그리지만, A*는 그것을 4×4로 묶은
    **17.9 cm 조립 격자**(`plan_path`의 `nav_coarse`)에서 돈다. 두 격자는 다르다 —
    `nav_coarse`는 "여유 ≥ r_b인 칸이 하나라도 있고(any) 16칸 전부가 관측된 free(all)"인 칸만
    통행 가능으로 본다. 경로를 판단한 근거를 보이려면 **이 격자**를 그려야 한다.

    `nav_coarse`가 없으면(계획 실패 등) 세밀 격자 버전으로 물러난다.
    """
    if nav_coarse is None:
        return bev_rgb_cspace(m)
    S = m['bev_size']
    f = int(round(S / nav_coarse.shape[0]))
    up = np.kron(nav_coarse, np.ones((f, f), dtype=bool))[:S, :S]
    if up.shape != (S, S):                       # 나누어떨어지지 않으면 가장자리를 채운다
        pad = np.zeros((S, S), bool); pad[:up.shape[0], :up.shape[1]] = up; up = pad
    img = np.zeros((S, S, 3), np.uint8)
    img[m['free']] = CSPACE_COLOR                # 비었지만 A* 격자에서 빠진 칸
    img[up] = (90, 90, 90)                       # A*가 실제로 쓸 수 있었던 칸
    img[m['occupied']] = (255, 70, 70)           # 점유(참고용으로 항상 위에 그린다)
    return img


def bev_rgb_cspace(m: dict) -> np.ndarray:
    """**embodiment별로 달라지는 지도**. 점유/free/unknown에 더해 `r_b` 팽창분을 따로 칠한다.

    점유칸 자체는 `r_b`와 무관하지만(높이 밴드만으로 정해짐), 로봇이 **실제로 갈 수 있는 곳**은
    `r_b`만큼 좁아진다. 그 차이(=C-space 확장분)를 어두운 빨강으로 칠하면 r_b별 패널이
    눈으로 구분된다 — 그러지 않으면 네 패널이 똑같아 보인다.
    """
    img = bev_rgb(m['bev'])
    img[m['free']] = (90, 90, 90)                       # 바닥 carving으로 늘어난 free 포함
    img[m['free'] & ~m['navigable']] = CSPACE_COLOR     # 여유 부족 = 이 로봇은 못 감
    return img


def draw_pts(img: np.ndarray, ij, color, radius: int = 1) -> np.ndarray:
    """BEV 픽셀 (i=row, j=col) 목록을 찍는다. 격자 밖 점은 건너뛴다."""
    h, w = img.shape[:2]
    for i, j in np.atleast_2d(np.asarray(ij)).astype(int):
        if 0 <= i < h and 0 <= j < w:
            cv2.circle(img, (int(j), int(i)), radius, tuple(int(c) for c in color), -1)
    return img


def draw_polyline(img: np.ndarray, ij, color, thickness: int = 1) -> np.ndarray:
    a = np.atleast_2d(np.asarray(ij)).astype(np.int32)
    if len(a) >= 2:
        cv2.polylines(img, [a[:, ::-1].reshape(-1, 1, 2)], False,
                      tuple(int(c) for c in color), thickness)
    return img


def draw_marker(img: np.ndarray, uv, color, radius: int = 7, filled: bool = True) -> np.ndarray:
    """이미지 좌표 (u=col, v=row)에 원. `filled=False`면 빈 원(원본 goal 규약)."""
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    h, w = img.shape[:2]
    if 0 <= u < w and 0 <= v < h:
        cv2.circle(img, (u, v), radius, tuple(int(c) for c in color), -1 if filled else 2)
    return img


def content_box(mask: np.ndarray, margin: int = 8):
    """관측된 셀만 담는 사각창 (i0, i1, j0, j1). BEV의 90%는 미관측 검정이라 그대로 보면 안 보인다.

    한 프레임의 모든 패널에 **같은 창**을 써야 e별 비교가 성립한다 — 호출부가 하나의 mask로 구한 뒤
    모든 패널에 적용할 것.
    """
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    i0, j0 = idx.min(0); i1, j1 = idx.max(0) + 1
    return (max(int(i0) - margin, 0), min(int(i1) + margin, mask.shape[0]),
            max(int(j0) - margin, 0), min(int(j1) + margin, mask.shape[1]))


def crop(img: np.ndarray, box) -> np.ndarray:
    i0, i1, j0, j1 = box
    return img[i0:i1, j0:j1]


def upscale(img: np.ndarray, min_px: int = 448) -> np.ndarray:
    f = max(1, int(np.ceil(min_px / max(img.shape[:2]))))
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) if f > 1 else img


def hstrip(images, height: int = 480, gap: int = 6) -> np.ndarray:
    """높이를 맞춰 가로로 잇는다 (간격은 흰 줄)."""
    out = []
    for k, im in enumerate(images):
        if im.ndim == 2:
            im = np.dstack([im] * 3)
        s = height / im.shape[0]
        out.append(cv2.resize(im, (int(round(im.shape[1] * s)), height), interpolation=cv2.INTER_NEAREST))
        if k < len(images) - 1:
            out.append(np.full((height, gap, 3), 255, np.uint8))
    return np.hstack(out)


def label(img: np.ndarray, text: str, org=(8, 24), color=(255, 255, 255)) -> np.ndarray:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return img
