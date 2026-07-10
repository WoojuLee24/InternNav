"""Isaac-Sim topdown ground-truth debug visualization — no Isaac Sim dependency.

Pure numpy/cv2, given a topdown RGB array + yaw (radians). The only Isaac-only piece
(extracting yaw from Isaac's quaternion via
``omni.isaac.core.utils.rotations.quat_to_euler_angles``) lives in
``internvla_n1_policy_unified.py``'s ``_save_isaac_topdown_debug``, not here — so this
module is importable/testable on any machine (CPU-only, no simulator), the same way
``depth_rgb_to_bev_torch.py`` is a standalone module reused by both Habitat and Isaac
code.

The rotation SIGN (``_rotate_by_yaw``) was derived analytically from
``topdown_camera_500``'s fixed mount quaternion + this codebase's yaw convention, not
copied from ``scripts/visualization/validate_depth_to_bev.py`` — that script's own
``rot_deg = degrees(yaw_rad) + rot_offset_deg`` convention turned out to never actually
run through its own validation path (``run_validation`` only calls
``save_comparison_pair``, not the differently-signed ``save_panel``), so it wasn't a
trustworthy reference. See ``_rotate_by_yaw``'s docstring for the derivation.

Reuses the agent-marker drawing style of
``internnav/habitat_extensions/vln/habitat_vln_evaluator_unified.py``'s ``_draw_agent``.
File naming (``..._b00_4_gt_topdown_world.jpg``) is chosen so this file sorts immediately
after ``UnifiedImageProvider``'s own ``..._b00_3_bev_gt_cur.jpg`` in a directory listing,
for direct side-by-side comparison — no separate raw-frame or BEV-copy files are saved.
"""

import math
import os

import cv2
import numpy as np

# topdown_camera_500 is an ORTHOGRAPHIC camera (not perspective) — verified by parsing
# the USD asset directly (no simulator needed): `pxr.Usd.Stage.Open(".../h1_internvla.usd")`,
# prim `/h1_description/topdown_camera_500`, `UsdGeom.Camera(prim).GetProjectionAttr().Get()
# == 'orthographic'`, `GetHorizontalApertureAttr().Get() == 200.0`,
# `UsdGeom.GetStageMetersPerUnit(stage) == 1.0`. USD aperture is in tenths of a stage unit,
# so this camera's frustum is a FIXED 200.0/10 = 20m-wide square in world space, regardless
# of the configured pixel resolution (VLNCameraCfg.resolution). This is why comparing the
# raw/world-aligned gt_cam jpg directly against a BEV jpg (which only covers
# `2*bev_range` metres, default 10m) looked scale-mismatched — gt_cam's frame is 2x wider.
_TOPDOWN_CAMERA_WORLD_WIDTH_M = 20.0


def _crop_center_to_range_m(img_bgr: np.ndarray, range_m: float) -> np.ndarray:
    """Center-crop to a ``2*range_m``-metre-wide square, using
    ``_TOPDOWN_CAMERA_WORLD_WIDTH_M`` to convert metres -> pixels for THIS image's actual
    width (not a hardcoded resolution, so it stays correct if VLNCameraCfg.resolution
    changes). The image center is the robot's position (topdown_camera_500 tracks robot
    XY), so this crop is valid regardless of any rotation already applied.
    """
    h, w = img_bgr.shape[:2]
    px_per_m = w / _TOPDOWN_CAMERA_WORLD_WIDTH_M
    crop_px = min(int(round(2 * range_m * px_per_m)), h, w)
    cy, cx = h // 2, w // 2
    half = crop_px // 2
    return img_bgr[cy - half : cy - half + crop_px, cx - half : cx - half + crop_px]


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    out = img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if out.shape[-1] == 4:
        out = out[..., :3]
    return np.ascontiguousarray(out)


def _draw_agent(img_bgr: np.ndarray) -> np.ndarray:
    """Draw a forward-facing arrow + center dot at image center.

    After ``rotate_topdown_world`` rotates the frame, the agent's forward direction is
    always "up" by construction, so the arrow is drawn straight up from center —
    matches Habitat's ``_draw_agent`` / ``validate_depth_to_bev.py``'s forward-arrow.
    """
    h, w = img_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    fa = int(2.5 * (w / 10.0))
    cv2.arrowedLine(img_bgr, (cx, cy), (cx, cy - fa), (0, 220, 0), 3, tipLength=0.25, line_type=cv2.LINE_AA)
    cv2.circle(img_bgr, (cx, cy), 8, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img_bgr, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def _rotate_by_yaw(img_bgr: np.ndarray, yaw_rad: float, rot_offset_deg: float = 0.0) -> np.ndarray:
    """Rotate a BGR image so the robot's forward direction points up. No agent marker.

    Sign: ``topdown_camera_500``'s fixed mount quaternion (``h1.py``'s
    ``orientation_quat``) resolves to world "image-up" = world +X (this codebase's
    yaw=0/forward direction, matching ``depth_rgb_to_bev_torch.py``'s "World: X=forward"
    convention) and "image-right" = world -Y. Projecting the robot's forward vector
    ``(cos(yaw), sin(yaw), 0)`` through that camera shows it drifting counter-clockwise
    (up -> left) in the RAW frame as yaw increases (standard extrinsic yaw: +X rotates
    toward +Y). ``cv2.getRotationMatrix2D``'s positive angle = counter-clockwise, so
    cancelling that drift needs a CLOCKWISE rotation of ``yaw`` degrees, i.e. a NEGATIVE
    angle — hence ``-degrees(yaw_rad)`` below (not ``+``, which would double the drift
    instead of cancelling it). ``rot_offset_deg`` is an additive extra correction (e.g. if
    the camera's fixed-mount assumption above turns out to need a constant fudge).
    """
    rot_deg = -math.degrees(yaw_rad) + rot_offset_deg
    if abs(rot_deg % 360) > 0.5:
        h, w = img_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h))
    return img_bgr


def rotate_topdown_world(
    topdown_rgb: np.ndarray,
    yaw_rad: float,
    rot_offset_deg: float = 0.0,
    crop_range_m: float = None,
    resize_to: int = None,
) -> np.ndarray:
    """World-align a topdown RGB frame to the robot's forward direction + draw an agent marker.

    Args:
        topdown_rgb: [H, W, 3] (or [H, W, 4]) uint8 or float [0, 1] RGB frame from
            Isaac's ``topdown_camera_500`` sensor obs (``obs['topdown_rgb']``).
        yaw_rad: robot yaw in radians (world frame), e.g. from
            ``omni.isaac.core.utils.rotations.quat_to_euler_angles(obs['globalrotation'])[2]``.
        rot_offset_deg: extra constant correction (degrees), additive on top of the
            analytically-derived ``-yaw`` (see ``_rotate_by_yaw``'s docstring). 0.0 unless
            that derivation's camera-mount assumption turns out to need a fudge.
        crop_range_m: if given, center-crop to a ``2*crop_range_m``-metre-wide square
            AFTER rotating (so no black-corner artifacts) — pass the BEV provider's
            ``bev_range`` here to make this image directly scale-comparable to a BEV jpg
            (``_TOPDOWN_CAMERA_WORLD_WIDTH_M``'s docstring explains why gt_cam's raw frame
            is otherwise ~2x wider than BEV's coverage). ``None`` = no crop (default,
            preserves prior behaviour).
        resize_to: if given, resize the (possibly cropped) result to this many pixels per
            side after cropping — pass the BEV provider's ``bev_size`` alongside
            ``crop_range_m`` for a pixel-for-pixel-comparable output. ``None`` = no resize.

    Returns:
        uint8 BGR image, rotated (+ cropped/resized) + agent-marker overlay drawn.
    """
    img_bgr = cv2.cvtColor(_to_uint8_rgb(topdown_rgb), cv2.COLOR_RGB2BGR)
    img_bgr = _rotate_by_yaw(img_bgr, yaw_rad, rot_offset_deg)
    if crop_range_m is not None:
        img_bgr = _crop_center_to_range_m(img_bgr, crop_range_m)
    if resize_to is not None:
        img_bgr = cv2.resize(img_bgr, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    return _draw_agent(img_bgr)


def save_isaac_topdown_debug(
    debug_dir: str,
    prefix: str,
    step: int,
    topdown_rgb: np.ndarray,
    yaw_rad: float,
    rot_offset_deg: float = 0.0,
    crop_range_m: float = None,
    resize_to: int = None,
) -> None:
    """Save the world-aligned topdown debug jpg for one step.

    Writes ``{prefix}_{step:06d}_b00_4_gt_topdown_world.jpg`` (world-aligned + agent
    marker, cropped to ``crop_range_m``/resized to ``resize_to`` when given — pass the
    BEV provider's ``bev_range``/``bev_size`` so this scales right against
    ``UnifiedImageProvider``'s own ``..._3_bev_gt_cur.jpg``). The ``_b00_4_`` prefix (not
    bare ``_4_``) is deliberate: it sorts immediately after ``..._b00_3_bev_gt_cur.jpg``
    in a directory listing (a bare ``_4_...`` would sort before ``_b00_...`` entirely,
    since ``'4' < 'b'`` in ASCII) — for direct side-by-side comparison, no separate BEV
    copy needed. The raw (un-rotated, full-frame) sensor capture is no longer saved — not
    needed for isaac-eval comparison. No-ops if ``topdown_rgb`` is ``None``.
    """
    if topdown_rgb is None:
        return
    os.makedirs(debug_dir, exist_ok=True)

    world = rotate_topdown_world(
        topdown_rgb, yaw_rad, rot_offset_deg=rot_offset_deg,
        crop_range_m=crop_range_m, resize_to=resize_to,
    )
    cv2.imwrite(f"{debug_dir}/{prefix}_{step:06d}_b00_4_gt_topdown_world.jpg", world)
