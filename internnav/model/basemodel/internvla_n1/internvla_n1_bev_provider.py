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
    ``bev_image_type`` ∈ {rgb, occ} and ``bev_depth_source`` ∈ {gt, dav2, udv2}
    together cover all five legacy ``bev_mode`` values.

Only the training path (``labels is not None``) substitutes; inference passes
through untouched.

S2 training injection (BEV into the LLM prompt during training) is out of scope
here — it needs dataset/prompt-pipeline changes. This file covers S1.
"""

import argparse
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.visual_input_provider import BEVImageProvider, BEVProcessor

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
    ('bev', 'rgb', 'dav2'): 'rgb_depthanythingv2',
    ('bev', 'occ', 'dav2'): 'occ_depthanythingv2',
}



def make_train_bev_processor(
    cam_height: float,
    device,
    depth_scale: float = 1.0,
    depth_source: str = 'gt',
    dav2_max_depth: float = 10.0,
    z_min: float = _Z_MIN,
    z_max: float = _Z_MAX,
) -> BEVProcessor:
    """BEVProcessor configured to match internvla_n1_bev.py exactly.

    Intrinsics are given at the 640×480 reference and auto-rescaled to the
    incoming frame size; traj_depths are already metric so depth_scale=1.0.
    """
    return BEVProcessor(
        fx=_BASE_FX, fy=_BASE_FY, cx=_BASE_CX, cy=_BASE_CY,
        cam_height=cam_height, cam_pitch_deg=_DEFAULT_CAM_PITCH,
        ref_width=_BASE_W, ref_height=_BASE_H,
        bev_size=_BEV_SIZE, bev_range=_BEV_RANGE,
        depth_scale=depth_scale, z_min=z_min, z_max=z_max,
        device=device,
        depth_source=depth_source,
        dav2_max_depth=dav2_max_depth,
    )



def apply_bev_to_traj(
    traj_images: torch.Tensor,                   # [B, T, H, W, 3] float [0, 1] HWC
    traj_depths: Optional[torch.Tensor],          # [B, T, H, W] metric metres, or None for dav2
    processor: BEVProcessor,
    s1_mode: str = 'bev',
    image_type: str = 'rgb',
    cam_pitch_deg: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure, test-friendly core of the BEV substitution.

    Uses the same ``processor.for_s1()`` path as eval (BEVImageProvider.get_s1_input),
    so train and eval BEV generation are identical.

    Returns (new_traj_images, new_traj_depths). For 'fpv_bev' the BEV frames are
    concatenated AFTER the FPV frames along T (T → 2T); traj_depths is left
    unchanged (legacy training semantics: [B, T, H, W], not doubled).

    traj_depths may be None when processor.depth_source=='dav2'.
    """
    if s1_mode == 'fpv':
        return traj_images, traj_depths

    bev_hw3 = processor.for_s1(
        traj_images, traj_depths,
        cam_pitch_deg=cam_pitch_deg, image_type=image_type,
    )  # [B, T, H, W, 3]

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
    p.add_argument('--bev_image_type', choices=['rgb', 'occ', 'occ.binary', 'occ.prob', 'occ.dist', 'occ.dist.sep'], default='rgb')
    p.add_argument('--bev_depth_source', choices=['gt', 'dav2', 'udv2'], default='gt')
    p.add_argument('--bev_dav2_max_depth', type=float, default=10.0)
    p.add_argument('--bev_z_min', type=float, default=_Z_MIN)
    p.add_argument('--bev_z_max', type=float, default=_Z_MAX)
    # --debug_dir is NOT parsed here: it belongs to ModelArguments in the base
    # trainer (internvla_n1_argument.py). Peeling it here would leave
    # model_args.debug_dir==None, causing trainer.py line 165 to overwrite
    # config.debug_dir with None after __init__ sets it via _PENDING_BEV_SETTINGS.
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
        self._bev_provider = None

    # --------------------------------------------------------------- helpers

    def _get_provider(self, device) -> BEVImageProvider:
        """Lazy-build BEVImageProvider from config — same class as eval path.

        cam_height is NOT a cache key: it is passed per-call via get_s1_input()
        so each sample in the batch uses its own camera height.
        """
        depth_source = getattr(self.config, 'bev_depth_source', 'gt')
        dav2_max_depth = getattr(self.config, 'bev_dav2_max_depth', 10.0)
        z_min = getattr(self.config, 'bev_z_min', _Z_MIN)
        z_max = getattr(self.config, 'bev_z_max', _Z_MAX)
        s1_mode = getattr(self.config, 'bev_s1_mode', 'bev')
        image_type = getattr(self.config, 'bev_image_type', 'rgb')
        debug_dir = getattr(self.config, 'debug_dir', None)
        p = self._bev_provider
        if (p is None
                or p.processor.depth_source != depth_source
                or p.s1_mode != s1_mode
                or p.image_type != image_type
                or p.processor.z_min != z_min
                or p.processor.z_max != z_max):
            processor = make_train_bev_processor(
                _DEFAULT_CAM_HEIGHT, device,
                depth_source=depth_source, dav2_max_depth=dav2_max_depth,
                z_min=z_min, z_max=z_max,
            )
            self._bev_provider = BEVImageProvider(
                processor, s1_mode=s1_mode, image_type=image_type,
                debug_dir=debug_dir,
            )
        else:
            p.debug_dir = debug_dir  # sync in case config was updated after first build
        return self._bev_provider

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
        traj_world_headings: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        s1_mode = getattr(self.config, 'bev_s1_mode', 'fpv')
        depth_source = getattr(self.config, 'bev_depth_source', 'gt')

        if labels is not None:
            # Stashed for subclasses (e.g. pixel_goal) that need per-sample camera pitch to
            # project a pixel goal to metric coordinates, regardless of bev_s1_mode.
            # .float(): HF Trainer casts batch to bfloat16; bfloat16(0.6)=0.6016 (7-bit mantissa loss)
            self._pixel_goal_cam_pitch = traj_cam_pitch_2.float() if traj_cam_pitch_2 is not None else None

        # Same BEVImageProvider.get_s1_input() path as eval.
        # depth shape: train [B,T,H,W] ↔ provider [B,T,H,W,1] — adapt around the call.
        if labels is not None and s1_mode != 'fpv' and (depth_source in ('dav2', 'udv2') or traj_depths is not None):
            # pass [B] tensors so each sample uses its own pitch and cam_height
            # .float(): HF Trainer casts batch to bfloat16; bfloat16(0.6)=0.6016 (7-bit mantissa loss)
            cam_pitch  = traj_cam_pitch_2.float()  if traj_cam_pitch_2  is not None else _DEFAULT_CAM_PITCH
            cam_height = traj_cam_heights.float()  if traj_cam_heights  is not None else _DEFAULT_CAM_HEIGHT
            provider = self._get_provider(traj_images.device)
            depths_5d = traj_depths.unsqueeze(-1) if traj_depths is not None else None
            s1v = provider.get_s1_input(traj_images, depths_5d,
                                        cam_pitch_deg=cam_pitch, cam_height=cam_height)
            traj_images = s1v.images if s1v.images is not None else traj_images
            # depths_5d may be doubled for fpv_bev; squeeze last dim back to [B,T,H,W]
            traj_depths = s1v.depths[..., 0] if s1v.depths is not None else traj_depths
            # training tdmap + world-aligned BEV debug (mirrors _save_eval_tdmap_debug)
            if traj_tdmaps is not None and provider.debug_dir:
                B_td = traj_tdmaps.shape[0]
                for b in range(B_td):
                    wh_b = float(traj_world_headings[b, 0]) if traj_world_headings is not None else None
                    provider.save_train_tdmap_debug(traj_tdmaps[b, 0], wh_b, batch_idx=b)

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
