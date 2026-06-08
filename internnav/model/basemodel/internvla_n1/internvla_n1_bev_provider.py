"""Training-time BEV via the SAME module the inference path uses.

Baseline ``internvla_n1.py`` and the existing ``internvla_n1_bev.py`` are NOT
modified. This file gives training the inference-side abstraction:

  - BEV pixels are produced by ``visual_input_provider.BEVProcessor`` — the very
    same class S1/S2 eval uses (``InternVLAN1NetBEV`` / ``HabitatVLNEvaluatorBEV``).
    A unit test asserts bit-for-bit parity with the legacy training helpers, so
    a model trained here sees exactly what it will see at inference.
  - The mode selector mirrors the eval ``bev_s1_mode`` key:
        'fpv'     → no change (baseline training)
        'bev'     → BEV replaces traj_images   (≡ legacy rgb_gt / occ_gt)
        'fpv_bev' → FPV ++ BEV along T          (≡ legacy fpv_concat_gt)
    ``bev_image_type`` ∈ {rgb, occ} and ``bev_depth_source`` ∈ {gt, depthanythingv2}
    together cover all five legacy ``bev_mode`` values.

Only the training path (``labels is not None``) substitutes; inference passes
through untouched.

S2 training injection (BEV into the LLM prompt during training) is out of scope
here — it needs dataset/prompt-pipeline changes. This file covers S1.
"""

import argparse
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.visual_input_provider import BEVProcessor

# Habitat base camera intrinsics at 640×480, hfov≈79° — identical to the
# constants baked into internvla_n1_bev.py so parity holds.
_BASE_FX = 388.19
_BASE_FY = 388.19
_BASE_CX = 319.5
_BASE_CY = 239.5
_BASE_W = 640
_BASE_H = 480
_DEFAULT_CAM_HEIGHT = 1.25
_DEFAULT_CAM_PITCH = 0.0
_BEV_SIZE = 224
_BEV_RANGE = 5.0
_Z_MIN = -0.2
_Z_MAX = 2.5

# bev_s1_mode → legacy bev_mode (documentation / parity reference only)
_MODE_EQUIV = {
    ('fpv', 'rgb', 'gt'): 'none',
    ('bev', 'rgb', 'gt'): 'rgb_gt',
    ('bev', 'occ', 'gt'): 'occ_gt',
    ('fpv_bev', 'rgb', 'gt'): 'fpv_concat_gt',
    ('bev', 'rgb', 'depthanythingv2'): 'rgb_depthanythingv2',
    ('bev', 'occ', 'depthanythingv2'): 'occ_depthanythingv2',
}


def _cam_scalar(tensor: Optional[torch.Tensor], default: float) -> float:
    if tensor is None:
        return default
    return float(tensor.float().mean().item())


def make_train_bev_processor(cam_height: float, device, depth_scale: float = 1.0) -> BEVProcessor:
    """BEVProcessor configured to match internvla_n1_bev.py exactly.

    Intrinsics are given at the 640×480 reference and auto-rescaled to the
    incoming frame size; traj_depths are already metric so depth_scale=1.0.
    """
    return BEVProcessor(
        fx=_BASE_FX, fy=_BASE_FY, cx=_BASE_CX, cy=_BASE_CY,
        cam_height=cam_height, cam_pitch_deg=_DEFAULT_CAM_PITCH,
        ref_width=_BASE_W, ref_height=_BASE_H,
        bev_size=_BEV_SIZE, bev_range=_BEV_RANGE,
        depth_scale=depth_scale, z_min=_Z_MIN, z_max=_Z_MAX,
        device=device,
    )


def _resize_to_hw3(bev_chw: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """[N, 3, bev, bev] → [N, H, W, 3] (bilinear, matching legacy helper)."""
    if bev_chw.shape[-2:] != (H, W):
        bev_chw = F.interpolate(bev_chw, size=(H, W), mode='bilinear', align_corners=False)
    return bev_chw.permute(0, 2, 3, 1).contiguous()


def apply_bev_to_traj(
    traj_images: torch.Tensor,        # [B, T, H, W, 3] float [0, 1] HWC
    traj_depths: torch.Tensor,        # [B, T, H, W] metric metres
    processor: BEVProcessor,
    s1_mode: str = 'bev',
    image_type: str = 'rgb',
    cam_pitch_deg: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure, test-friendly core of the BEV substitution.

    Returns (new_traj_images, new_traj_depths). For 'fpv_bev' the BEV frames are
    concatenated AFTER the FPV frames along T (T → 2T, matching the legacy
    fpv_concat_gt mode); traj_depths is left unchanged so the base model keeps
    its original [B, T, H, W] depth tensor (legacy training semantics).
    """
    if s1_mode == 'fpv':
        return traj_images, traj_depths

    B, T, H, W = traj_images.shape[:4]
    fpv_flat = traj_images.flatten(0, 1)        # [B*T, H, W, 3]
    depth_flat = traj_depths.flatten(0, 1)      # [B*T, H, W]

    if image_type == 'occ':
        bev_hw = processor.occupancy(depth_flat, depth_in_meters=True, cam_pitch_deg=cam_pitch_deg)
        bev_chw = bev_hw.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    else:  # 'rgb'
        bev_chw = processor.bev_chw(fpv_flat, depth_flat, depth_in_meters=True, cam_pitch_deg=cam_pitch_deg)

    bev_hw3 = _resize_to_hw3(bev_chw, H, W).reshape(B, T, H, W, 3).to(traj_images)

    if s1_mode == 'bev':
        return bev_hw3, traj_depths
    if s1_mode == 'fpv_bev':
        return torch.cat([traj_images, bev_hw3], dim=1), traj_depths
    raise ValueError(f"bev_s1_mode must be fpv|bev|fpv_bev, got {s1_mode!r}")


def parse_bev_cli_args(argv: List[str]) -> Tuple[dict, List[str]]:
    """Extract ``--bev_*`` flags from argv without touching the base parser.

    Returns (bev_settings_dict, remaining_argv). Lets the new trainer add BEV
    knobs while leaving internnav/trainer/internvla_n1_argument.py untouched.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--bev_s1_mode', choices=['fpv', 'bev', 'fpv_bev'], default='bev')
    p.add_argument('--bev_image_type', choices=['rgb', 'occ'], default='rgb')
    p.add_argument('--bev_depth_source', choices=['gt', 'depthanythingv2'], default='gt')
    p.add_argument('--bev_dav2_max_depth', type=float, default=10.0)
    known, remaining = p.parse_known_args(argv)
    return vars(known), remaining


# Pending settings stashed by the launcher before from_pretrained() runs, since
# the HF Trainer constructs the model without forwarding our custom kwargs.
_PENDING_BEV_SETTINGS: dict = {}


def set_pending_bev_settings(settings: dict) -> None:
    _PENDING_BEV_SETTINGS.clear()
    _PENDING_BEV_SETTINGS.update(settings)


class InternVLAN1BEVProviderForCausalLM(InternVLAN1ForCausalLM):
    """S1 BEV via shared BEVProcessor, selected by bev_s1_mode (training only)."""

    def __init__(self, config):
        super().__init__(config)
        # Persist pending CLI settings onto config so they land in the saved
        # checkpoint's config.json and survive resume.
        for key, value in _PENDING_BEV_SETTINGS.items():
            if not hasattr(config, key) or getattr(config, key) is None:
                setattr(config, key, value)
        self._bev_processor = None
        self._dav2_model = None

    # --------------------------------------------------------------- helpers

    def _get_processor(self, cam_height: float, device) -> BEVProcessor:
        if self._bev_processor is None or self._bev_processor.cam_height != cam_height:
            self._bev_processor = make_train_bev_processor(cam_height, device)
        return self._bev_processor

    def _estimate_depth(self, fpv_flat: torch.Tensor) -> torch.Tensor:
        """Lazy DepthAnythingV2 path — reuses internvla_n1_bev.py helpers if present."""
        from internnav.model.basemodel.internvla_n1.internvla_n1_bev import (
            _estimate_depth_dav2_batch,
            _load_dav2_full,
        )
        if self._dav2_model is None:
            max_depth = getattr(self.config, 'bev_dav2_max_depth', 10.0)
            self._dav2_model = _load_dav2_full(max_depth=max_depth).to(fpv_flat.device)
        return _estimate_depth_dav2_batch(fpv_flat, self._dav2_model)

    # --------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        t_s_pos: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        traj_images: Optional[torch.Tensor] = None,
        traj_depths: Optional[torch.Tensor] = None,
        traj_cam_heights: Optional[torch.Tensor] = None,
        traj_cam_pitch_1: Optional[torch.Tensor] = None,
        traj_cam_pitch_2: Optional[torch.Tensor] = None,
        video_frame_num: Optional[torch.Tensor] = None,
        traj_poses: Optional[torch.Tensor] = None,
        traj_tdmaps: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        s1_mode = getattr(self.config, 'bev_s1_mode', 'fpv')
        image_type = getattr(self.config, 'bev_image_type', 'rgb')
        depth_source = getattr(self.config, 'bev_depth_source', 'gt')
        is_dav2 = depth_source == 'depthanythingv2'

        if labels is not None and s1_mode != 'fpv' and (is_dav2 or traj_depths is not None):
            cam_height = _cam_scalar(traj_cam_heights, _DEFAULT_CAM_HEIGHT)
            cam_pitch = _cam_scalar(traj_cam_pitch_2, _DEFAULT_CAM_PITCH)
            processor = self._get_processor(cam_height, traj_images.device)

            if is_dav2:
                B, T, H, W = traj_images.shape[:4]
                depth_est = self._estimate_depth(traj_images.flatten(0, 1)).reshape(B, T, H, W)
                traj_depths_used = depth_est
            else:
                traj_depths_used = traj_depths

            traj_images, traj_depths = apply_bev_to_traj(
                traj_images, traj_depths_used, processor,
                s1_mode=s1_mode, image_type=image_type, cam_pitch_deg=cam_pitch,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            t_s_pos=t_s_pos,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            traj_images=traj_images,
            traj_depths=traj_depths,
            video_frame_num=video_frame_num,
            traj_poses=traj_poses,
        )
