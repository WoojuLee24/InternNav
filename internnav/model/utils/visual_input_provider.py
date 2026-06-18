"""Pluggable S1/S2 visual-input providers (FPV passthrough or BEV injection).

This module is the single shared abstraction for BEV projection across:
  - Habitat eval        (``HabitatVLNEvaluatorBEV``, eval_type='habitat_vln_bev')
  - Isaac Sim / H1 eval (``InternVLAN1AgentBEV``,    model_name='internvla_n1_bev')
  - Training            (``BEVProcessor.bev_chw`` / ``occupancy`` are batch APIs
                         equivalent to the helpers inside internvla_n1_bev.py)

Design (see /root/.claude/plans/feature-bev-immutable-ritchie.md):
  - Existing files are NOT modified. New subclasses own a ``VisualInputProvider``
    and are selected purely via config (``eval_type`` / ``model_name``).
  - ``FPVProvider`` is a no-op: every method returns "use the original input",
    so selecting it reproduces the existing code path exactly.
  - ``BEVImageProvider`` converts (RGB, metric depth) frames into coloured BEV
    images via :func:`internnav.model.utils.depth_rgb_to_bev_torch.depth_rgb_to_bev`.
  - ``BEVFeatureProvider`` is a Phase-2 stub (DINOv2 bypass via precomputed
    BEV feature tokens).

Depth-unit contract:
  - ``get_s1_input`` / ``BEVProcessor.for_s1``  : depth is METRIC (metres).
  - ``get_s2_extra`` / ``BEVProcessor.for_s2``  : depth is raw sensor output;
    it is multiplied by ``depth_scale`` to obtain metres (H1 obs depth is
    normalised [0, 1] with depth_scale=10.0; pass metres with depth_scale=1.0).
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from internnav.model.utils.depth_rgb_to_bev_torch import (
    depth_rgb_to_bev,
    depth_to_bev_occ,
    depth_to_bev_occ_ros2,
    bev_3state_to_binary,
    bev_3state_to_prob,
    bev_3state_to_dist,
    bev_3state_to_dist_sep,
)


def _load_dav2_full(max_depth: float = 10.0):
    """Load full DepthAnythingV2 (ViT-S + depth head) for metric depth estimation."""
    from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2
    model_cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'max_depth': max_depth}
    model = DepthAnythingV2(**model_cfg)
    ckpt = 'checkpoints/depth_anything_v2_metric_hypersim_vits.pth'
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location='cpu'), assign=True)
    model.eval().requires_grad_(False)
    return model


def _estimate_depth_dav2_batch(
    rgb_flat: torch.Tensor,   # [N, H, W, 3] float [0,1] RGB HWC
    dav2_model,
) -> torch.Tensor:            # [N, H, W] metric depth in metres
    N, H, W, _ = rgb_flat.shape
    x = rgb_flat.permute(0, 3, 1, 2).float()
    mean_t = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t  = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean_t) / std_t
    H14 = max((H // 14) * 14, 14)
    W14 = max((W // 14) * 14, 14)
    if H14 != H or W14 != W:
        x = F.interpolate(x, (H14, W14), mode='bilinear', align_corners=False)
    with torch.no_grad():
        depth = dav2_model(x)
    if H14 != H or W14 != W:
        depth = F.interpolate(depth.unsqueeze(1), (H, W), mode='bilinear', align_corners=True).squeeze(1)
    return depth


def _load_udv2():
    """Load UniDepthV2 for metric depth estimation."""
    import sys, os as _os
    _enc = _os.path.normpath(_os.path.join(_os.path.dirname(__file__), '..', 'encoder'))
    if _enc not in sys.path:
        sys.path.insert(0, _enc)
    from internnav.model.encoder.unidepth.models import UniDepthV2
    model = UniDepthV2.from_pretrained("checkpoints/unidepth-v2-vitl14")
    model.eval().requires_grad_(False)
    return model


def _estimate_depth_udv2_batch(
    rgb_flat: torch.Tensor,  # [N, H, W, 3] float [0, 1] RGB HWC
    udv2_model,
    fx: float = None, fy: float = None, cx: float = None, cy: float = None,
) -> torch.Tensor:           # [N, H, W] metric depth in metres
    # UniDepthV2 BatchedCamera is broken for N>1 with explicit intrinsics:
    # unproject passes full-batch data to each per-image camera → reshape fails.
    # Process one image at a time to avoid this.
    N, H, W, _ = rgb_flat.shape
    K = (rgb_flat.new_tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]).unsqueeze(0)
         if fx is not None and fy is not None and cx is not None and cy is not None
         else None)
    depths = []
    for i in range(N):
        x_i = rgb_flat[i:i+1].permute(0, 3, 1, 2).float() * 255.0  # [1, 3, H, W]
        K_i = K.clone() if K is not None else None
        with torch.no_grad():
            pred = udv2_model.infer(x_i, K_i)
        depths.append(pred['depth'].squeeze(1))  # [1, H, W]
    return torch.cat(depths, dim=0)  # [N, H, W]


def _cfg_get(config, key, default=None):
    """Read a key from a dict, argparse.Namespace, or pydantic/attr object."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    value = getattr(config, key, default)
    return default if value is None else value


@dataclass
class S1VisualInput:
    """Return type of ``VisualInputProvider.get_s1_input``.

    ``None`` fields mean "keep the original input" — the FPV path sets all
    fields to None so existing behaviour is preserved bit-for-bit.
    """

    images: Optional[torch.Tensor] = None    # [B, T, H, W, 3] float [0, 1] — replaces images_dp
    features: Optional[torch.Tensor] = None  # [B, N, C] — Phase 2: DINOv2 bypass tokens
    depths: Optional[torch.Tensor] = None    # [B, T, H, W, 1] metres — replaces depths_dp


class BEVProcessor:
    """Stateless (RGB, depth) → BEV converter shared by eval and training.

    Intrinsics are given at a reference resolution (``ref_width`` × ``ref_height``)
    and automatically rescaled to whatever resolution the inputs arrive at
    (e.g. 224×224 NavDP frames vs 640×480 sensor frames).
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        cam_height: float = 1.25,
        cam_pitch_deg: float = 0.0,
        ref_width: int = 640,
        ref_height: int = 480,
        bev_size: int = 224,
        bev_range: float = 5.0,
        depth_scale: float = 1.0,
        z_min: float = -0.2,
        z_max: float = 2.5,
        cam_x_offset: float = 0.0,
        cam_z_offset: float = 0.0,
        device: str = 'cuda:0',
        depth_source: str = 'gt',
        dav2_max_depth: float = 10.0,
    ):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.cam_height = cam_height
        self.cam_pitch_deg = cam_pitch_deg
        self.ref_width, self.ref_height = ref_width, ref_height
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.depth_scale = depth_scale
        self.z_min, self.z_max = z_min, z_max
        self.cam_x_offset, self.cam_z_offset = cam_x_offset, cam_z_offset
        self.device = torch.device(device)
        self.depth_source = depth_source
        self.dav2_max_depth = dav2_max_depth
        self._dav2_model = None
        self._udv2_model = None

    # ------------------------------------------------------------------ utils

    def _scaled_intrinsics(self, H: int, W: int):
        """Rescale reference intrinsics to the actual input resolution."""
        return (
            self.fx * W / self.ref_width,
            self.fy * H / self.ref_height,
            self.cx * W / self.ref_width,
            self.cy * H / self.ref_height,
        )

    @staticmethod
    def _to_float_rgb(rgb: torch.Tensor) -> torch.Tensor:
        """uint8 [0, 255] or float [0, 1] → float32 [0, 1]."""
        rgb = rgb.float()
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        return rgb

    # ------------------------------------------------------------ core (batch)

    def _estimate_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        """Lazy monocular metric depth estimation (DAV2 or UniDepth).

        Args:
            rgb: [B, H, W, 3] float [0, 1].
        Returns:
            [B, H, W] metric depth in metres.
        """
        if self.depth_source == 'dav2':
            if self._dav2_model is None:
                self._dav2_model = _load_dav2_full(max_depth=self.dav2_max_depth).to(rgb.device)
            depth = _estimate_depth_dav2_batch(rgb, self._dav2_model)
        elif self.depth_source == 'udv2':
            if self._udv2_model is None:
                self._udv2_model = _load_udv2().to(rgb.device)
            _, H, W, _ = rgb.shape
            fx, fy, cx, cy = self._scaled_intrinsics(H, W)
            depth = _estimate_depth_udv2_batch(rgb, self._udv2_model, fx=fx, fy=fy, cx=cx, cy=cy)
        else:
            raise ValueError(f"_estimate_depth called with non-estimation depth_source={self.depth_source!r}")
        self._last_estimated_depth = depth
        return depth

    def bev_chw(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        depth_in_meters: bool = False,
        cam_pitch_deg: Optional[float] = None,
        cam_height: Optional[float] = None,
        z_filter: bool = True,
    ) -> torch.Tensor:
        """Coloured BEV for a batch of frames.

        Args:
            rgb       : [B, H, W, 3] float [0, 1] or uint8.
            depth     : [B, H, W] raw (× depth_scale → metres) or metres.
                        Ignored when depth_source=='dav2' (estimated from rgb).
            cam_height: overrides self.cam_height for per-sample use.
            z_filter  : if False, project ALL valid depth pixels (no height filter).
        Returns:
            [B, 3, bev_size, bev_size] float [0, 1].
        """
        rgb = self._to_float_rgb(rgb.to(self.device))
        if self.depth_source in ('dav2', 'udv2'):
            depth = self._estimate_depth(rgb)
            depth_in_meters = True
        else:
            depth = depth.to(self.device).float()
        _, H, W = depth.shape
        fx, fy, cx, cy = self._scaled_intrinsics(H, W)
        pitch = self.cam_pitch_deg if cam_pitch_deg is None else cam_pitch_deg
        height = self.cam_height if cam_height is None else cam_height
        scale = 1.0 if depth_in_meters else self.depth_scale
        return depth_rgb_to_bev(
            depth, rgb,
            cam_height=height,
            cam_pitch_deg=pitch,
            cam_x_offset=self.cam_x_offset,
            cam_z_offset=self.cam_z_offset,
            fx=fx, fy=fy, cx=cx, cy=cy,
            bev_range=self.bev_range,
            bev_size=self.bev_size,
            depth_scale=scale,
            z_min=self.z_min, z_max=self.z_max,
            z_filter=z_filter,
        )

    def occupancy(
        self,
        depth: Optional[torch.Tensor] = None,
        depth_in_meters: bool = False,
        cam_pitch_deg: Optional[float] = None,
        rgb: Optional[torch.Tensor] = None,
        cam_height: Optional[float] = None,
    ) -> torch.Tensor:
        """Occupancy BEV for a batch of depth frames.

        Args:
            depth     : [B, H, W] raw or metres. Ignored when depth_source=='dav2'.
            rgb       : [B, H, W, 3] float [0, 1], required when depth_source=='dav2'.
            cam_height: overrides self.cam_height for per-sample use.
        Returns:
            [B, bev_size, bev_size] float [0, 1].
        """
        if self.depth_source in ('dav2', 'udv2'):
            rgb = self._to_float_rgb(rgb.to(self.device))
            depth = self._estimate_depth(rgb)
            depth_in_meters = True
        depth = depth.to(self.device).float()
        _, H, W = depth.shape
        fx, fy, cx, cy = self._scaled_intrinsics(H, W)
        pitch = self.cam_pitch_deg if cam_pitch_deg is None else cam_pitch_deg
        height = self.cam_height if cam_height is None else cam_height
        scale = 1.0 if depth_in_meters else self.depth_scale
        return depth_to_bev_occ(
            depth,
            cam_height=height,
            cam_pitch_deg=pitch,
            cam_x_offset=self.cam_x_offset,
            cam_z_offset=self.cam_z_offset,
            fx=fx, fy=fy, cx=cx, cy=cy,
            bev_range=self.bev_range,
            bev_size=self.bev_size,
            depth_scale=scale,
            z_min=self.z_min, z_max=self.z_max,
        )

    # --------------------------------------------------------------- S2 / S1

    def for_s2(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        depth_in_meters: bool = False,
        cam_pitch_deg: Optional[float] = None,
    ) -> Image.Image:
        """Single frame → BEV PIL image for the S2 (LLM) prompt.

        Args:
            rgb  : [H, W, 3] uint8 (or float [0, 1]).
            depth: [H, W] (or [H, W, 1]) raw sensor depth or metres.
                   Optional when depth_source=='dav2'.
        """
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0)
        if depth is not None:
            depth = np.asarray(depth)
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth_t = torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0)
        else:
            depth_t = None
        bev = self.bev_chw(rgb_t, depth_t, depth_in_meters=depth_in_meters, cam_pitch_deg=cam_pitch_deg)
        bev_np = (bev[0].permute(1, 2, 0).clamp(0, 1).cpu().float().numpy() * 255).astype(np.uint8)
        return Image.fromarray(bev_np)

    def _bev_batch(self, rgb_flat: torch.Tensor, depth_flat: Optional[torch.Tensor],
                   pitch, image_type: str,
                   cam_height=None) -> torch.Tensor:
        """[N, H, W, 3] → [N, 3, S, S] BEV; pitch/cam_height may be scalar or [N] tensor."""
        if image_type in ('occ', 'occ.binary', 'occ.prob', 'occ.dist', 'occ.dist.sep'):
            if self.depth_source in ('dav2', 'udv2'):
                rgb_dev = self._to_float_rgb(rgb_flat.to(self.device))
                depth_m = self._estimate_depth(rgb_dev)
            else:
                depth_m = depth_flat.to(self.device).float()
            _, H, W = depth_m.shape
            fx, fy, cx, cy = self._scaled_intrinsics(H, W)
            height = self.cam_height if cam_height is None else cam_height
            p = self.cam_pitch_deg if pitch is None else pitch
            bev_3s = depth_to_bev_occ_ros2(
                depth_m,
                cam_height=height, cam_pitch_deg=p,
                cam_x_offset=self.cam_x_offset, cam_z_offset=self.cam_z_offset,
                fx=fx, fy=fy, cx=cx, cy=cy,
                bev_range=self.bev_range, bev_size=self.bev_size,
                depth_scale=1.0, z_min=self.z_min, z_max=self.z_max,
            )  # [N, S, S]: 0=unknown, 0.5=free, 1.0=occupied
            if image_type == 'occ':
                return bev_3s.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
            if image_type == 'occ.binary':
                return bev_3state_to_binary(bev_3s)   # [N, 3, S, S]: [free, occ, unk]
            if image_type == 'occ.prob':
                return bev_3state_to_prob(bev_3s).unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
            if image_type == 'occ.dist':
                return bev_3state_to_dist(bev_3s).unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
            if image_type == 'occ.dist.sep':
                return bev_3state_to_dist_sep(bev_3s)  # [N, 3, S, S]: [dist, occ, unk]
        if image_type == 'occ_old':
            bev_hw = self.occupancy(depth_flat, depth_in_meters=True,
                                    cam_pitch_deg=pitch, rgb=rgb_flat, cam_height=cam_height)
            return bev_hw.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        return self.bev_chw(rgb_flat, depth_flat, depth_in_meters=True,
                            cam_pitch_deg=pitch, cam_height=cam_height, z_filter=False)

    def for_s1(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        cam_pitch_deg=None,     # float | [B] tensor | None
        image_type: str = 'rgb',
        cam_height=None,        # float | [B] tensor | None
    ) -> torch.Tensor:
        """NavDP frame stack → BEV stack in the SAME layout/dtype as the input.

        Args:
            rgb          : [B, T, H, W, 3] float [0, 1] (NavDP images_dp layout).
            depth        : [B, T, H, W] or [B, T, H, W, 1] METRES. Optional when depth_source=='dav2'.
            cam_pitch_deg: scalar float, None, or [B] tensor for per-sample pitch.
            cam_height   : scalar float, None, or [B] tensor for per-sample camera height.
            image_type   : 'rgb' (coloured BEV) | 'occ' (occupancy BEV, replicated to 3ch).
        Returns:
            [B, T, H, W, 3] BEV frames, resized to (H, W), dtype of ``rgb``.
        """
        B, T, H, W = rgb.shape[:4]
        rgb_flat = rgb.flatten(0, 1)                                       # [B*T, H, W, 3]
        if depth is not None:
            d = depth.flatten(0, 1)
            depth_flat = d[..., 0] if d.ndim == 4 else d                  # [B*T, H, W]
        else:
            depth_flat = None

        # Expand [B] pitch/height to [B*T] so each frame in the stack gets its sample's value.
        if isinstance(cam_pitch_deg, torch.Tensor) and cam_pitch_deg.ndim > 0:
            cam_pitch_deg = cam_pitch_deg.repeat_interleave(T)  # [B] -> [B*T]
        if isinstance(cam_height, torch.Tensor) and cam_height.ndim > 0:
            cam_height = cam_height.repeat_interleave(T)        # [B] -> [B*T]
        bev = self._bev_batch(rgb_flat, depth_flat, cam_pitch_deg, image_type, cam_height=cam_height)

        if bev.shape[-2:] != (H, W):
            bev = F.interpolate(bev, size=(H, W), mode='bilinear', align_corners=False)
        bev = bev.permute(0, 2, 3, 1).reshape(B, T, H, W, 3)             # back to HWC stack
        return bev.to(device=rgb.device, dtype=rgb.dtype)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class VisualInputProvider(ABC):
    """Strategy object deciding what S1/S2 actually see (FPV, BEV, ...).

    ``s1_mode`` / ``s2_mode`` ('fpv' | 'bev' | 'fpv_bev') tell call sites how
    to combine the provider output with the original input; the FPV default
    means "no change".
    """

    s1_mode: str = 'fpv'
    s2_mode: str = 'fpv'

    def reset(self):
        """Called at episode start."""

    def set_goal(self, rgb, depth):
        """Called when a pixel goal is confirmed (goal-frame snapshot hook)."""

    @abstractmethod
    def get_s1_input(self, rgb, depth) -> S1VisualInput:
        """Visual input for S1 (NavDP). ``None`` fields = keep original."""

    @abstractmethod
    def get_s2_extra(self, rgb, depth, is_lookdown: bool = False) -> List[Image.Image]:
        """Extra images to append to the S2 (LLM) prompt. Empty list = none."""


class FPVProvider(VisualInputProvider):
    """No-op provider: existing FPV behaviour, untouched."""

    def get_s1_input(self, rgb, depth) -> S1VisualInput:
        return S1VisualInput()

    def get_s2_extra(self, rgb, depth, is_lookdown: bool = False) -> List[Image.Image]:
        return []


_VALID_MODES = ('fpv', 'bev', 'fpv_bev')


class BEVImageProvider(VisualInputProvider):
    """Image-level BEV injection for S1 and/or S2, with per-system modes.

    Modes (independent for S1 and S2):
        'fpv'     : original input only — identical to the existing code path.
        'bev'     : BEV replaces the FPV frame(s).
        'fpv_bev' : FPV and BEV are both fed.
                    S1: BEV frames concatenated along T → [B, 2T, H, W, 3]
                        (same convention as the fpv_concat_gt training mode of
                        internvla_n1_bev.py; depths are duplicated to match).
                    S2: BEV image appended after the look-down FPV image.

    Stateless w.r.t. the goal frame: S1 call sites already pass both the
    pixel-goal frame and the current frame stacked along T, so both are
    converted to BEV per call (``set_goal`` stays a no-op hook).

    Args:
        processor   : shared BEVProcessor.
        s1_mode     : 'fpv' | 'bev' | 'fpv_bev' for NavDP images_dp.
        s2_mode     : 'fpv' | 'bev' | 'fpv_bev' for the LLM look-down turn
                      (S2 history frames always stay FPV — only the look-down
                      pixel-goal turn is affected).
        s1_pitch_deg: camera pitch of the frames S1 consumes
                      (Habitat: look-down frames → base + 60°).
        s2_pitch_deg: camera pitch of the look-down frame S2 consumes.
        s2_depth_in_meters: whether depth handed to get_s2_extra is already
                      metric (Habitat evaluator) or raw (H1 obs, × depth_scale).
    """

    def __init__(
        self,
        processor: BEVProcessor,
        s1_mode: str = 'bev',
        s2_mode: str = 'fpv',
        s1_pitch_deg: Optional[float] = None,
        s2_pitch_deg: Optional[float] = None,
        s2_depth_in_meters: bool = False,
        image_type: str = 'rgb',
        debug_dir: Optional[str] = None,
    ):
        if s1_mode not in _VALID_MODES or s2_mode not in _VALID_MODES:
            raise ValueError(f"s1_mode/s2_mode must be one of {_VALID_MODES}, got {s1_mode!r}/{s2_mode!r}")
        self.processor = processor
        self.s1_mode = s1_mode
        self.s2_mode = s2_mode
        self.s1_pitch_deg = s1_pitch_deg
        self.s2_pitch_deg = s2_pitch_deg
        self.s2_depth_in_meters = s2_depth_in_meters
        self.image_type = image_type
        self.debug_dir = debug_dir
        self._debug_step = 0

    # ------------------------------------------------------------------ debug

    def _save_s1_debug(self, rgb, depth, bev, pitch, cam_height):
        """Save all batch samples frame[0]: FPV | GT-depth | BEV-out | [GT-BEV when dav2].

        Args:
            rgb   : [B, T, H, W, 3] float [0,1]
            depth : [B, T, H, W, 1] metres, or None
            bev   : [B, T, H, W, 3] float [0,1]  — provider output
            pitch : scalar or [B] tensor (degrees)
            cam_height: scalar or [B] tensor (metres)
        """
        import os
        from internnav.model.utils.depth_rgb_to_bev_torch import depth_rgb_to_bev, depth_to_bev_occ_ros2, save_image
        os.makedirs(self.debug_dir, exist_ok=True)
        s = self._debug_step
        B = rgb.shape[0]

        depth_src = self.processor.depth_source  # 'gt' | 'dav2' | 'udv2'
        for b in range(B):
            prefix = f"{self.debug_dir}/step_{s:06d}_b{b:02d}"
            fpv  = rgb[b, 0]   # [H, W, 3]
            bev0 = bev[b, 0]   # [H, W, 3]
            save_image(fpv, f"{prefix}_1_fpv.jpg")
            save_image(bev0, f"{prefix}_3_bev_{depth_src}.jpg")

            if depth is not None:
                gt_d = depth[b, 0, ..., 0]   # [H, W] metres
                depth_vis = (gt_d / gt_d.max().clamp(min=0.1)).clamp(0, 1)
                save_image(depth_vis, f"{prefix}_2_gt_depth.jpg")

                # ── occupancy comparison (training torch vs ROS2 numpy) ────────
                def _raycast_free(grid_np):
                    """BEV 중심 → occupied 셀 방향으로 raycasting해 free(0) 셀 표시."""
                    g = grid_np.copy()
                    S2 = g.shape[0]
                    ci = cj = S2 // 2
                    occ_r, occ_c = np.where(g == 100)
                    for r, c in zip(occ_r, occ_c):
                        n = max(abs(int(r) - ci), abs(int(c) - cj))
                        if n == 0:
                            continue
                        ts = np.arange(n) / n          # [0, 1) — exclude endpoint
                        rs = (ci + ts * (r - ci)).astype(np.int32).clip(0, S2 - 1)
                        cs = (cj + ts * (c - cj)).astype(np.int32).clip(0, S2 - 1)
                        free_mask = g[rs, cs] == -1
                        g[rs[free_mask], cs[free_mask]] = 0
                    return g

                def _vis_occ(grid_np):
                    """[-1/0/100] int8 [S,S] → [S,S,3] uint8 BGR (ROS2 rviz 색상)."""
                    img = np.full((*grid_np.shape, 3), 128, dtype=np.uint8)  # unknown → gray
                    img[grid_np == 0]   = (255, 255, 255)   # free    → white
                    img[grid_np == 100] = (0,   0,   0  )   # occupied → black
                    return img

                H2, W2 = gt_d.shape
                fx2, fy2, cx2, cy2 = self.processor._scaled_intrinsics(H2, W2)
                p0 = pitch[b].item()      if isinstance(pitch,      torch.Tensor) else (float(pitch)      if pitch      is not None else self.processor.cam_pitch_deg)
                h0 = cam_height[b].item() if isinstance(cam_height, torch.Tensor) else (float(cam_height) if cam_height is not None else self.processor.cam_height)
                # 3_train_occ: _vis_occ visualization from bev0 per image_type
                _bev0_np = bev0.float().cpu().numpy()  # [H, W, 3]
                _101 = None
                if self.image_type == 'occ':
                    _101 = np.where(_bev0_np[..., 0] >= 0.75, 100,
                           np.where(_bev0_np[..., 0] >= 0.25, 0, -1)).astype(np.int8)
                elif self.image_type == 'occ.binary':
                    _free_ch, _occ_ch = _bev0_np[..., 0], _bev0_np[..., 1]
                    _101 = np.where(_occ_ch > 0.5, 100,
                           np.where(_free_ch > 0.5, 0, -1)).astype(np.int8)
                elif self.image_type == 'occ.prob':
                    _ch = _bev0_np[..., 0]
                    _101 = np.where(_ch >= 0.75, 100,
                           np.where(_ch <= 0.25, 0, -1)).astype(np.int8)
                if _101 is not None:
                    save_image(_vis_occ(_101), f"{prefix}_3_train_occ_{depth_src}.jpg")
                # 2. ROS2-style OCC: numpy pinhole + R_c2w (occgrid_publisher.py 로직 재현)
                gt_d_np = gt_d.float().cpu().numpy()
                vv, uu = np.mgrid[0:H2, 0:W2]
                zc = gt_d_np
                xc = (uu - cx2) * zc / fx2
                yc = (vv - cy2) * zc / fy2
                cos_p = np.cos(np.radians(p0))
                sin_p = np.sin(np.radians(p0))
                # R_c2w: cam(right,down,fwd) → world(X=fwd, Y=left, Z=up)
                Xw = -sin_p * yc + cos_p * zc
                Yw = -xc
                Zw = -cos_p * yc - sin_p * zc + h0
                height_mask = (Zw > self.processor.z_min) & (Zw < self.processor.z_max) & (zc > 0.1)
                S = self.processor.bev_size
                bev_scale = S / (2.0 * self.processor.bev_range)
                ii = np.floor(S / 2.0 - Xw * bev_scale).astype(np.int32)
                jj = np.floor(S / 2.0 - Yw * bev_scale).astype(np.int32)
                in_bounds = (ii >= 0) & (ii < S) & (jj >= 0) & (jj < S)
                valid = height_mask & in_bounds
                ros2_101 = np.full((S, S), -1, dtype=np.int8)
                ros2_101[ii[valid], jj[valid]] = 100
                save_image(_vis_occ(_raycast_free(ros2_101)), f"{prefix}_6_ros2_occ_gt.jpg")
                # 3. ROS2 package build_occupancy_grid (same z filter, 224×224)
                try:
                    import sys as _sys
                    _occ_pub_dir = os.path.normpath(os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', '..', 'scripts', 'realworld'))
                    if _occ_pub_dir not in _sys.path:
                        _sys.path.insert(0, _occ_pub_dir)
                    from occgrid_publisher import build_occupancy_grid as _build_occ_grid
                    from builtin_interfaces.msg import Time as _RosTime
                    pcloud_xy_v = np.stack([Xw.ravel()[valid.ravel()],
                                            Yw.ravel()[valid.ravel()]], axis=-1)
                    _resolution = 2.0 * self.processor.bev_range / S   # 10/224 m/cell
                    _msg = _build_occ_grid(
                        pcloud_xy_v,
                        stamp=_RosTime(sec=0, nanosec=0),
                        frame_id='debug',
                        resolution=_resolution,
                        grid_size=S,
                        center_x=0.0, center_y=0.0,
                    )
                    ros2_pkg_grid = np.array(_msg.data, dtype=np.int8).reshape(S, S)
                    # build_occupancy_grid: grid[cy, cx] where cx=X(fwd)→col, cy=Y(left)→row
                    # training BEV: bev[i,j] = bev[S/2-X*s, S/2-Y*s] (forward=top, left=left)
                    # alignment: bev[i,j] = ros2[S-1-j, S-1-i]
                    ros2_pkg_aligned = np.fliplr(np.flipud(ros2_pkg_grid)).T
                    save_image(_vis_occ(_raycast_free(ros2_pkg_aligned)), f"{prefix}_7_ros2_pkg_occ_gt.jpg")
                except Exception:
                    import traceback as _tb; _tb.print_exc()

                if self.processor.depth_source in ('dav2', 'udv2'):
                    with torch.no_grad():
                        est_d = self.processor._estimate_depth(fpv.unsqueeze(0))[0]
                    est_vis = (est_d / est_d.max().clamp(min=0.1)).clamp(0, 1)
                    save_image(est_vis, f"{prefix}_2b_{self.processor.depth_source}_depth.jpg")

                    p0 = pitch[b].item()      if isinstance(pitch,      torch.Tensor) else (pitch      or self.processor.cam_pitch_deg)
                    h0 = cam_height[b].item() if isinstance(cam_height, torch.Tensor) else (cam_height or self.processor.cam_height)
                    S = self.processor.bev_size
                    fx, fy, cx, cy = self.processor._scaled_intrinsics(S, S)
                    bev_kwargs = dict(
                        cam_height=h0, cam_pitch_deg=p0,
                        cam_x_offset=self.processor.cam_x_offset,
                        cam_z_offset=self.processor.cam_z_offset,
                        fx=fx, fy=fy, cx=cx, cy=cy,
                        bev_range=self.processor.bev_range,
                        bev_size=S,
                        depth_scale=1.0,
                        z_min=self.processor.z_min, z_max=self.processor.z_max,
                    )
                    if self.image_type == 'occ':
                        gt_bev = depth_to_bev_occ_ros2(gt_d.unsqueeze(0), **bev_kwargs)[0]  # [S, S] 3-state
                    else:
                        fpv_s = F.interpolate(fpv.permute(2, 0, 1).unsqueeze(0), size=(S, S), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).unsqueeze(0)
                        gt_bev = depth_rgb_to_bev(gt_d.unsqueeze(0), fpv_s, **bev_kwargs, z_filter=False)[0]  # [3, S, S]
                    save_image(gt_bev, f"{prefix}_4_gt_bev.jpg")

    # ------------------------------------------------------------------ API

    def get_s1_input(self, rgb, depth, cam_pitch_deg=None, cam_height=None) -> S1VisualInput:
        """Per-call overrides accept float or [B] tensor (for per-sample train batches)."""
        if self.s1_mode == 'fpv':
            return S1VisualInput()
        if not (isinstance(rgb, torch.Tensor) and (depth is None or isinstance(depth, torch.Tensor))):
            # Legacy 'sync' path hands raw numpy frames straight to generate_traj;
            # BEV substitution is only defined for the stacked-tensor layout.
            print('[BEVImageProvider] non-tensor S1 input — falling back to FPV for this call')
            return S1VisualInput()
        pitch = cam_pitch_deg if cam_pitch_deg is not None else self.s1_pitch_deg
        bev = self.processor.for_s1(rgb, depth, cam_pitch_deg=pitch,
                                     image_type=self.image_type, cam_height=cam_height)
        if self.debug_dir:
            try:
                self._save_s1_debug(rgb, depth, bev, pitch, cam_height)
            except Exception:
                import traceback; traceback.print_exc()
        self._debug_step += 1

        if self.s1_mode == 'bev':
            return S1VisualInput(images=bev)
        # 'fpv_bev': [fpv_goal, fpv_cur, bev_goal, bev_cur] along T (fpv_concat_gt
        # convention). NOTE: T doubles — requires a checkpoint trained with the
        # matching fpv_concat mode.
        images = torch.cat([rgb, bev], dim=1)
        depths = torch.cat([depth, depth], dim=1)  # BEV frames reuse the FPV depth
        return S1VisualInput(images=images, depths=depths)

    def save_train_tdmap_debug(self, tdmap: torch.Tensor, world_heading: Optional[float] = None, batch_idx: int = 0) -> None:
        """Save tdmap + world-aligned BEV for training debug. Call AFTER get_s1_input.

        Agent-frame (order: bev → gt_bev → gt_topdown):
          step_XXXXXX_bNN_3_bev_{src}.jpg       — BEV (agent frame)         [saved by _save_s1_debug]
          step_XXXXXX_bNN_4_gt_bev.jpg          — GT BEV (agent frame)      [saved by _save_s1_debug]
          step_XXXXXX_bNN_5_gt_topdown.jpg      — topdown (agent forward = up)

        World-frame (order: bev → gt_bev → gt_topdown), only when world_heading is not None:
          step_XXXXXX_bNN_w3_bev_world_{src}.jpg   — BEV rotated to world frame
          step_XXXXXX_bNN_w4_gt_bev_world.jpg      — GT BEV rotated to world frame
          step_XXXXXX_bNN_w5_gt_topdown_world.jpg  — topdown rotated to world + FWD arrow

        {src} = processor.depth_source ('gt' | 'dav2' | 'udv2')
        """
        if not self.debug_dir:
            return
        try:
            import cv2
            import math
            import numpy as np
            from internnav.model.utils.depth_rgb_to_bev_torch import save_image

            s         = self._debug_step - 1
            debug_dir = self.debug_dir
            S         = self.processor.bev_size
            bev_range = self.processor.bev_range
            os.makedirs(debug_dir, exist_ok=True)
            prefix = f"step_{s:06d}_b{batch_idx:02d}"

            td_np = (tdmap.clamp(0, 1).cpu().float().numpy() * 255).astype(np.uint8)  # [H,W,3] uint8

            # agent-frame topdown
            save_image(tdmap, f"{debug_dir}/{prefix}_5_gt_topdown.jpg")

            if world_heading is None:
                return

            agent_angle = float(world_heading)
            rot_deg     = math.degrees(math.pi + agent_angle)
            c_px        = S // 2
            fa          = int(2.5 * S / (2.0 * bev_range))

            def _rotate_world(im):
                if abs(rot_deg % 360) > 0.5:
                    M  = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), rot_deg, 1.0)
                    im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))
                return im

            # world-aligned BEV and GT-BEV
            _dsrc = self.processor.depth_source
            for src_sfx, dst_sfx in [(f'_3_bev_{_dsrc}.jpg',       f'_w3_bev_world_{_dsrc}.jpg'),
                                      ('_4_gt_bev.jpg',              '_w4_gt_bev_world.jpg')]:
                img = cv2.imread(f"{debug_dir}/{prefix}{src_sfx}")
                if img is not None:
                    cv2.imwrite(f"{debug_dir}/{prefix}{dst_sfx}", _rotate_world(img))

            # world-aligned topdown + FWD arrow
            td_world = _rotate_world(cv2.cvtColor(
                cv2.resize(td_np, (S, S), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_RGB2BGR,
            ))
            ac = int(c_px + fa * math.sin(agent_angle))
            ar = int(c_px + fa * math.cos(agent_angle))
            cv2.arrowedLine(td_world, (c_px, c_px), (ac, ar), (0, 220, 0), 3, tipLength=0.25,
                            line_type=cv2.LINE_AA)
            cv2.circle(td_world, (c_px, c_px), 8, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(td_world, (c_px, c_px), 8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(f"{debug_dir}/{prefix}_w5_gt_topdown_world.jpg", td_world)
        except Exception:
            import traceback; traceback.print_exc()

    def get_s2_extra(self, rgb, depth, is_lookdown: bool = False) -> List[Image.Image]:
        if self.s2_mode == 'fpv' or not is_lookdown:
            return []
        bev = self.processor.for_s2(
            rgb, depth,
            depth_in_meters=self.s2_depth_in_meters,
            cam_pitch_deg=self.s2_pitch_deg,
        )
        return [bev]


class BEVFeatureProvider(VisualInputProvider):
    """Phase 2 stub — feature-level BEV (DINOv2 bypass via S1VisualInput.features).

    Requires InternVLAN1ForCausalLMBEV.generate_traj(precomputed_features=...);
    see the plan file. Not implemented in Phase 1.
    """

    def __init__(self, processor: BEVProcessor):
        self.processor = processor

    def get_s1_input(self, rgb, depth) -> S1VisualInput:
        raise NotImplementedError('bev_feature provider is a Phase-2 stub — use bev_image instead')

    def get_s2_extra(self, rgb, depth, is_lookdown: bool = False) -> List[Image.Image]:
        return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_visual_provider(config, device: str = 'cuda:0') -> VisualInputProvider:
    """Build a provider from model_settings (dict / Namespace / pydantic).

    Recognised keys (all prefixed ``bev_`` except the selector):
        visual_provider : 'fpv' (default) | 'bev_image' | 'bev_feature'
        bev_fx, bev_fy, bev_cx, bev_cy : intrinsics at the reference resolution
        bev_ref_width, bev_ref_height  : reference resolution (default 640×480)
        bev_cam_height, bev_cam_pitch_deg, bev_cam_x_offset, bev_cam_z_offset
        bev_size, bev_range, bev_depth_scale, bev_z_min, bev_z_max
        bev_s1_mode, bev_s2_mode       : 'fpv' | 'bev' | 'fpv_bev' per system
                                         (defaults: s1='bev', s2='fpv_bev')
        bev_s1, bev_s2                 : legacy booleans, used only when the
                                         mode key is absent (True → default
                                         mode, False → 'fpv')
        bev_s1_pitch_deg, bev_s2_pitch_deg : per-system pitch overrides
        bev_s2_depth_in_meters         : depth unit handed to get_s2_extra
        bev_depth_source               : 'gt' | 'dav2' | 'udv2' (default 'gt')
    """
    ptype = _cfg_get(config, 'visual_provider', 'fpv')
    if ptype == 'fpv':
        return FPVProvider()
    if ptype not in ('bev_image', 'bev_feature'):
        raise ValueError(f"Unknown visual_provider '{ptype}' (expected fpv | bev_image | bev_feature)")

    processor = BEVProcessor(
        fx=_cfg_get(config, 'bev_fx', 585.0),
        fy=_cfg_get(config, 'bev_fy', 585.0),
        cx=_cfg_get(config, 'bev_cx', 320.0),
        cy=_cfg_get(config, 'bev_cy', 240.0),
        cam_height=_cfg_get(config, 'bev_cam_height', 1.25),
        cam_pitch_deg=_cfg_get(config, 'bev_cam_pitch_deg', 0.0),
        ref_width=_cfg_get(config, 'bev_ref_width', 640),
        ref_height=_cfg_get(config, 'bev_ref_height', 480),
        bev_size=_cfg_get(config, 'bev_size', 224),
        bev_range=_cfg_get(config, 'bev_range', 5.0),
        depth_scale=_cfg_get(config, 'bev_depth_scale', 1.0),
        z_min=_cfg_get(config, 'bev_z_min', -0.2),
        z_max=_cfg_get(config, 'bev_z_max', 2.5),
        cam_x_offset=_cfg_get(config, 'bev_cam_x_offset', 0.0),
        cam_z_offset=_cfg_get(config, 'bev_cam_z_offset', 0.0),
        device=device,
        depth_source=_cfg_get(config, 'bev_depth_source', 'gt'),
        dav2_max_depth=_cfg_get(config, 'bev_dav2_max_depth', 10.0),
    )
    if ptype == 'bev_feature':
        return BEVFeatureProvider(processor)

    # mode keys win; legacy booleans (bev_s1/bev_s2) map to default-mode/'fpv'
    s1_mode = _cfg_get(config, 'bev_s1_mode', 'bev' if _cfg_get(config, 'bev_s1', True) else 'fpv')
    s2_mode = _cfg_get(config, 'bev_s2_mode', 'fpv_bev' if _cfg_get(config, 'bev_s2', True) else 'fpv')
    return BEVImageProvider(
        processor,
        s1_mode=s1_mode,
        s2_mode=s2_mode,
        s1_pitch_deg=_cfg_get(config, 'bev_s1_pitch_deg', None),
        s2_pitch_deg=_cfg_get(config, 'bev_s2_pitch_deg', None),
        s2_depth_in_meters=_cfg_get(config, 'bev_s2_depth_in_meters', False),
        image_type=_cfg_get(config, 'bev_image_type', 'rgb'),
        debug_dir=_cfg_get(config, 'debug_dir', None),
    )
