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
)


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

    def bev_chw(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_in_meters: bool = False,
        cam_pitch_deg: Optional[float] = None,
    ) -> torch.Tensor:
        """Coloured BEV for a batch of frames.

        Args:
            rgb  : [B, H, W, 3] float [0, 1] or uint8.
            depth: [B, H, W] raw (× depth_scale → metres) or metres.
        Returns:
            [B, 3, bev_size, bev_size] float [0, 1].

        Training note: with metric ``traj_depths`` flattened to [B*T, H, W] and
        ``traj_images`` flattened to [B*T, H, W, 3], this matches the
        ``rgb_gt`` bev_mode of internvla_n1_bev.py.
        """
        rgb = self._to_float_rgb(rgb.to(self.device))
        depth = depth.to(self.device).float()
        _, H, W = depth.shape
        fx, fy, cx, cy = self._scaled_intrinsics(H, W)
        pitch = self.cam_pitch_deg if cam_pitch_deg is None else cam_pitch_deg
        # depth_rgb_to_bev multiplies depth by its depth_scale argument to get metres
        scale = 1.0 if depth_in_meters else self.depth_scale
        return depth_rgb_to_bev(
            depth, rgb,
            cam_height=self.cam_height,
            cam_pitch_deg=pitch,
            cam_x_offset=self.cam_x_offset,
            cam_z_offset=self.cam_z_offset,
            fx=fx, fy=fy, cx=cx, cy=cy,
            bev_range=self.bev_range,
            bev_size=self.bev_size,
            depth_scale=scale,
            z_min=self.z_min, z_max=self.z_max,
        )

    def occupancy(
        self,
        depth: torch.Tensor,
        depth_in_meters: bool = False,
        cam_pitch_deg: Optional[float] = None,
    ) -> torch.Tensor:
        """Occupancy BEV for a batch of depth frames.

        Args:
            depth: [B, H, W] raw or metres.
        Returns:
            [B, bev_size, bev_size] float [0, 1].

        Training note: equivalent to the ``occ_gt`` bev_mode of internvla_n1_bev.py.
        """
        depth = depth.to(self.device).float()
        _, H, W = depth.shape
        fx, fy, cx, cy = self._scaled_intrinsics(H, W)
        pitch = self.cam_pitch_deg if cam_pitch_deg is None else cam_pitch_deg
        scale = 1.0 if depth_in_meters else self.depth_scale
        return depth_to_bev_occ(
            depth,
            cam_height=self.cam_height,
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
        depth: np.ndarray,
        depth_in_meters: bool = False,
        cam_pitch_deg: Optional[float] = None,
    ) -> Image.Image:
        """Single frame → BEV PIL image for the S2 (LLM) prompt.

        Args:
            rgb  : [H, W, 3] uint8 (or float [0, 1]).
            depth: [H, W] (or [H, W, 1]) raw sensor depth, or metres.
        """
        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[..., 0]
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0)
        depth_t = torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0)
        bev = self.bev_chw(rgb_t, depth_t, depth_in_meters=depth_in_meters, cam_pitch_deg=cam_pitch_deg)
        bev_np = (bev[0].permute(1, 2, 0).clamp(0, 1).cpu().float().numpy() * 255).astype(np.uint8)
        return Image.fromarray(bev_np)

    def for_s1(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        cam_pitch_deg: Optional[float] = None,
    ) -> torch.Tensor:
        """NavDP frame stack → BEV stack in the SAME layout/dtype as the input.

        Args:
            rgb  : [B, T, H, W, 3] float [0, 1] (NavDP images_dp layout).
            depth: [B, T, H, W, 1] METRES (NavDP depths_dp layout).
        Returns:
            [B, T, H, W, 3] BEV frames, resized to (H, W), dtype of ``rgb``.
        """
        B, T, H, W = rgb.shape[:4]
        rgb_flat = rgb.flatten(0, 1)                    # [B*T, H, W, 3]
        depth_flat = depth.flatten(0, 1)[..., 0]        # [B*T, H, W]
        bev = self.bev_chw(rgb_flat, depth_flat, depth_in_meters=True, cam_pitch_deg=cam_pitch_deg)
        if bev.shape[-2:] != (H, W):
            bev = F.interpolate(bev, size=(H, W), mode='bilinear', align_corners=False)
        bev = bev.permute(0, 2, 3, 1).reshape(B, T, H, W, 3)  # back to HWC stack
        return bev.to(device=rgb.device, dtype=rgb.dtype)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class VisualInputProvider(ABC):
    """Strategy object deciding what S1/S2 actually see (FPV, BEV, ...)."""

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


class BEVImageProvider(VisualInputProvider):
    """Image-level BEV injection for S1 and/or S2.

    Stateless w.r.t. the goal frame: S1 call sites already pass both the
    pixel-goal frame and the current frame stacked along T, so both are
    converted to BEV per call (``set_goal`` stays a no-op hook).

    Args:
        processor   : shared BEVProcessor.
        use_s1      : replace NavDP images_dp with BEV frames.
        use_s2      : append a BEV image to the LLM look-down turn.
        s1_pitch_deg: camera pitch of the frames S1 consumes
                      (Habitat: look-down frames → base + 60°).
        s2_pitch_deg: camera pitch of the look-down frame S2 consumes.
        s2_depth_in_meters: whether depth handed to get_s2_extra is already
                      metric (Habitat evaluator) or raw (H1 obs, × depth_scale).
    """

    def __init__(
        self,
        processor: BEVProcessor,
        use_s1: bool = True,
        use_s2: bool = True,
        s1_pitch_deg: Optional[float] = None,
        s2_pitch_deg: Optional[float] = None,
        s2_depth_in_meters: bool = False,
    ):
        self.processor = processor
        self.use_s1 = use_s1
        self.use_s2 = use_s2
        self.s1_pitch_deg = s1_pitch_deg
        self.s2_pitch_deg = s2_pitch_deg
        self.s2_depth_in_meters = s2_depth_in_meters

    def get_s1_input(self, rgb, depth) -> S1VisualInput:
        if not self.use_s1:
            return S1VisualInput()
        if not (isinstance(rgb, torch.Tensor) and isinstance(depth, torch.Tensor)):
            # Legacy 'sync' path hands raw numpy frames straight to generate_traj;
            # BEV substitution is only defined for the stacked-tensor layout.
            print('[BEVImageProvider] non-tensor S1 input — falling back to FPV for this call')
            return S1VisualInput()
        images = self.processor.for_s1(rgb, depth, cam_pitch_deg=self.s1_pitch_deg)
        return S1VisualInput(images=images)

    def get_s2_extra(self, rgb, depth, is_lookdown: bool = False) -> List[Image.Image]:
        if not (self.use_s2 and is_lookdown):
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
        bev_s1, bev_s2                 : enable injection per system (default True)
        bev_s1_pitch_deg, bev_s2_pitch_deg : per-system pitch overrides
        bev_s2_depth_in_meters         : depth unit handed to get_s2_extra
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
    )
    if ptype == 'bev_feature':
        return BEVFeatureProvider(processor)
    return BEVImageProvider(
        processor,
        use_s1=_cfg_get(config, 'bev_s1', True),
        use_s2=_cfg_get(config, 'bev_s2', True),
        s1_pitch_deg=_cfg_get(config, 'bev_s1_pitch_deg', None),
        s2_pitch_deg=_cfg_get(config, 'bev_s2_pitch_deg', None),
        s2_depth_in_meters=_cfg_get(config, 'bev_s2_depth_in_meters', False),
    )
