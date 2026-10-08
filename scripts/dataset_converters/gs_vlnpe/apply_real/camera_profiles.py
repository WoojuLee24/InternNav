"""Synthetic observation camera profiles for Stage 04.

Dataset GT cameras remain owned by ``dataset_utils``.  Profiles here are used
only when a scene has no GT camera, so changing one cannot silently alter the
legacy InternNav-N1 path.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraProfile:
    name: str
    width: int
    height: int
    k: np.ndarray
    depth_scale_m: float
    depth_min_m: float
    depth_max_m: float
    render_near_m: float
    render_far_m: float
    # vln_ce rig 전용 — 기존 프로파일은 None이라 동작이 바뀌지 않는다.
    # vln_ce는 카메라를 "바닥 위 고정 높이 + 고정 pitch"로 정의하는데(rig 이름이 `125cm_30deg`),
    # 기존 프로파일은 그 축이 없다(에피소드마다 h_b/pitch를 따로 받는다).
    mount_height_m: float = None
    pitch_down_deg: float = None

    def to_dict(self) -> dict:
        out = {
            'name': self.name,
            'width': self.width,
            'height': self.height,
            'intrinsic': self.k.tolist(),
            'depth_scale_m': self.depth_scale_m,
            'depth_min_m': self.depth_min_m,
            'depth_max_m': self.depth_max_m,
            'render_near_m': self.render_near_m,
            'render_far_m': self.render_far_m,
        }
        if self.mount_height_m is not None:
            out['mount_height_m'] = self.mount_height_m
        if self.pitch_down_deg is not None:
            out['pitch_down_deg'] = self.pitch_down_deg
        return out


def square_pixel_k_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Return a zero-distortion, square-pixel pinhole intrinsic matrix."""
    focal_px = (width * 0.5) / np.tan(np.deg2rad(hfov_deg) * 0.5)
    return np.array([
        [focal_px, 0.0, width * 0.5],
        [0.0, focal_px, height * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# Nominal synthetic profile, not a calibration dump from a particular device.
# A real deployment must replace K with librealsense get_intrinsics() output and
# verify depth_scale_m with get_depth_scale().  0.001 m/raw is the D400 default
# Z16 unit; 10 m is this synthetic dataset's storage cutoff, not a claim about
# D455 measurement accuracy at 10 m.
D455_NOMINAL = CameraProfile(
    name='d455_nominal',
    width=480,
    height=270,
    k=square_pixel_k_from_hfov(480, 270, 90.0),
    depth_scale_m=0.001,
    depth_min_m=0.1,
    depth_max_m=10.0,
    render_near_m=0.05,
    render_far_m=12.0,
)

# 30 m 저장 범위 변형. K/해상도는 d455_nominal과 동일하게 두어 "범위만 바꿨을 때의 차이"를
# 분리해서 볼 수 있게 한다. render_far는 저장 상한보다 커야 한다 — generate_episode의 valid
# 조건이 depth < render_far*0.99 이므로 30 m를 담으려면 far가 최소 30/0.99 = 30.3 m다.
D455_30M = CameraProfile(
    name='d455_30m',
    width=480,
    height=270,
    k=square_pixel_k_from_hfov(480, 270, 90.0),
    depth_scale_m=0.001,
    depth_min_m=0.1,
    depth_max_m=30.0,
    render_near_m=0.05,
    render_far_m=35.0,
)

# ---------------------------------------------------------------------------
# vln_ce rig — 릴리스 `InternData-N1-v0.5-mini/vln_ce`와 같은 세팅
# ---------------------------------------------------------------------------
#
# 릴리스 스트림 이름이 `observation.images.{rgb,depth}.<H>cm_<P>deg`이고, `<H>`는 바닥 위
# 카메라 높이[cm], `<P>`는 아래로 기울인 각도[도]다. 실측으로 확인한 것:
#
#   - `pose.<rig>` 0행의 translation이 정확히 `(0, 0, H/100)` -> 높이가 이름 그대로다.
#   - 이미지 640x480, rgb는 **jpg**, depth는 **uint16 PNG 밀리미터**(0 = 무효).
#   - **hfov가 rig 높이마다 다르다** — `3dloader_vlnce/pixel_goal_utils.py:35`의 `_RIG_HFOV`.
#     125 cm는 79.0도(Habitat 기본), 60 cm는 68.7도(그쪽에서 실측 피팅한 값).
#
# ⚠️ **주점(cx, cy)이 위 `square_pixel_k_from_hfov`와 다르다.** 기존 함수는 `width * 0.5`
# (= 320.0)를 쓰는데 vln_ce는 `(width - 1) / 2`(= 319.5)다. 0.5 px 차이라 조용히 어긋나므로
# 별도 빌더를 둔다 — 픽셀 goal 투영이 이 주점을 쓰기 때문에 섞으면 goal이 0.5 px씩 밀린다.
VLNCE_W, VLNCE_H = 640, 480
VLNCE_RIG_HFOV_DEG = {125: 79.0, 60: 68.7}   # pixel_goal_utils._RIG_HFOV 와 같은 값
# 릴리스 depth는 uint16 밀리미터다. 학습은 224로 줄인 뒤 /1000 하고 5.0 m에서 자른다
# (`internvla_n1_lerobot_dataset.py:1052-1065`). 저장은 그보다 넓게 둔다 — 릴리스 실측 최대 8.2 m.
VLNCE_DEPTH_SCALE_M = 0.001


def vlnce_k(hfov_deg: float) -> np.ndarray:
    """vln_ce 주점 규약(`(W-1)/2`)을 쓰는 intrinsic. `pixel_goal_utils.intrinsics_for_rig`와 일치."""
    fx = (VLNCE_W / 2.0) / np.tan(np.deg2rad(hfov_deg) * 0.5)
    return np.array([
        [fx, 0.0, (VLNCE_W - 1) / 2.0],
        [0.0, fx, (VLNCE_H - 1) / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _vlnce_rig(height_cm: int, pitch_deg: int) -> CameraProfile:
    hfov = VLNCE_RIG_HFOV_DEG[height_cm]
    return CameraProfile(
        name=f'{height_cm}cm_{pitch_deg}deg',
        width=VLNCE_W,
        height=VLNCE_H,
        k=vlnce_k(hfov),
        depth_scale_m=VLNCE_DEPTH_SCALE_M,
        depth_min_m=0.1,
        depth_max_m=10.0,
        render_near_m=0.05,
        render_far_m=12.0,
        mount_height_m=height_cm / 100.0,
        pitch_down_deg=float(pitch_deg),
    )


# 릴리스에 실제로 존재하는 5개 조합 (학습 preset이 이 중 (height, pitch_1, pitch_2)를 고른다)
VLNCE_RIGS = tuple(_vlnce_rig(h, p) for h, p in
                   ((125, 0), (125, 30), (125, 45), (60, 15), (60, 30)))

PROFILES = {D455_NOMINAL.name: D455_NOMINAL, D455_30M.name: D455_30M}
PROFILES.update({r.name: r for r in VLNCE_RIGS})


def vlnce_rig_names() -> tuple:
    """vln_ce rig 이름만 (기존 프로파일 제외)."""
    return tuple(r.name for r in VLNCE_RIGS)


def names() -> tuple:
    return tuple(PROFILES)


def get(name: str) -> CameraProfile:
    if name not in PROFILES:
        raise KeyError(f'unknown camera profile: {name!r}')
    return PROFILES[name]
