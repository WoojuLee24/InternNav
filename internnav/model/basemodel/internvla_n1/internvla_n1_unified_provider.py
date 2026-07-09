"""Unified S1 image input via UnifiedImageProvider — training only.

Generalizes ``internvla_n1_bev_provider.py`` into four independent axes
(view/type/mode/combine — see ``unified_image_provider.py``). Selected by
``s1_combine_mode`` (independent of the legacy ``bev_s1_mode``):

    combine='none'          -> no change (delegates to
                                InternVLAN1BEVProviderForCausalLM, which
                                itself no-ops when bev_s1_mode=='fpv')
    combine='replace', view='bev'
                             -> 100% delegation to the existing, unmodified
                                BEV pipeline (same code InternVLAN1BEVProviderForCausalLM
                                already runs, reached via super().forward())
    combine='concat', view='bev'
                             -> BEV is an ADDITIONAL slot, not a substitution:
                                UnifiedImageProvider.get_s1_input returns it via
                                S1VisualInput.extra_images, forwarded here as
                                traj_bev_images through internvla_n1_bev_provider.py's
                                passthrough into internvla_n1.py's forward(), which
                                stacks [goal, cur, bev] as a 3rd token-dim slot
                                (see internvla_n1.py's _encode_s1_memory_tokens).
    combine='replace', view='fpv', type='depth'
                             -> UnifiedImageProvider converts traj_depths into
                                a depth image; ImageInputAdapter maps it to 3
                                channels before it reaches the (untouched)
                                rgb_model / base forward.
    type='panorama'         -> stub, raises NotImplementedError
    combine='concat', view='fpv', type='depth'
                             -> raises NotImplementedError (T-doubling requires
                                re-deriving traj_poses/video_frame_num alignment,
                                out of scope — see task doc risk #2)

S2 image injection during training lives in the dataset
(``internvla_n1_lerobot_dataset.py``), not here — this file only covers S1.
S2 image injection during eval (Habitat) lives in ``visual_input_provider.py``'s
``create_visual_provider`` / ``habitat_vln_evaluator_bev.py``.
"""

import argparse
import os
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from internnav.model.basemodel.internvla_n1.internvla_n1_bev_provider import (
    InternVLAN1BEVProviderForCausalLM,
)
from internnav.model.utils.unified_image_provider import ImageInputAdapter
from internnav.model.utils.visual_input_provider import BEVProcessor
from internnav.model.utils.unified_image_provider import UnifiedImageProvider

# Habitat base camera intrinsics at 640x480 — same constants used by
# make_train_bev_processor() in internvla_n1_bev_provider.py, so a dav2/udv2
# depth_source estimates depth with the same camera model as BEV training.
_BASE_FX = 388.19
_BASE_FY = 388.19
_BASE_CX = 319.5
_BASE_CY = 239.5

# (config attr, default) pairs that fully determine the built UnifiedImageProvider.
_PROVIDER_CONFIG_KEYS = [
    ('s1_image_view', 'fpv'), ('s1_image_type', 'rgb'), ('s1_image_mode', 'raw'), ('s1_combine_mode', 'none'),
    ('s2_image_view', 'fpv'), ('s2_image_type', 'rgb'), ('s2_image_mode', 'raw'), ('s2_combine_mode', 'none'),
    ('bev_depth_source', 'gt'), ('bev_dav2_max_depth', 10.0), ('bev_z_min', -0.2), ('bev_z_max', 2.5),
]


def parse_unified_cli_args(argv: List[str]) -> Tuple[dict, List[str]]:
    """Extract ``--s1_image_*``/``--s1_combine_mode``/``--bev_depth_*`` flags
    without touching the base HfArgumentParser.

    ``--s2_image_*``/``--s2_combine_mode`` are deliberately NOT peeled here:
    they are plain ``DataArguments`` fields (see internvla_n1_argument.py),
    parsed directly by the base HfArgumentParser and read by the dataset —
    S2 training image injection has no CLI-peel/pending-settings need since
    HF's data collator (unlike ``from_pretrained``) does forward normal argv.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--s1_image_view', choices=['fpv', 'bev'], default='fpv')
    p.add_argument('--s1_image_type', choices=['rgb', 'depth', 'panorama'], default='rgb')
    p.add_argument('--s1_image_mode', default='raw')
    p.add_argument('--s1_combine_mode', choices=['none', 'replace', 'concat'], default='none')
    p.add_argument('--bev_depth_source', choices=['gt', 'dav2', 'udv2'], default='gt')
    p.add_argument('--bev_dav2_max_depth', type=float, default=10.0)
    p.add_argument('--bev_z_min', type=float, default=-0.2)
    p.add_argument('--bev_z_max', type=float, default=2.5)
    p.add_argument('--depth_adapter_mode', choices=['repeat', 'conv'], default='repeat')
    known, remaining = p.parse_known_args(argv)
    return vars(known), remaining


_PENDING_UNIFIED_SETTINGS: dict = {}


def set_pending_unified_settings(settings: dict) -> None:
    _PENDING_UNIFIED_SETTINGS.clear()
    _PENDING_UNIFIED_SETTINGS.update(settings)


class InternVLAN1UnifiedProviderForCausalLM(InternVLAN1BEVProviderForCausalLM):
    """S1 unified (view/type/mode/combine) image input; BEV path untouched."""

    def __init__(self, config):
        super().__init__(config)
        for key, value in _PENDING_UNIFIED_SETTINGS.items():
            if not hasattr(config, key) or getattr(config, key) is None:
                setattr(config, key, value)
        self._unified_provider = None
        self._unified_provider_key = None
        self._depth_adapter = None
        self._depth_adapter_key = None
        self._debug_step = 0

    # --------------------------------------------------------------- helpers

    def _get_unified_provider(self, device) -> UnifiedImageProvider:
        key = tuple(getattr(self.config, k, d) for k, d in _PROVIDER_CONFIG_KEYS)
        if self._unified_provider is None or self._unified_provider_key != key:
            processor = BEVProcessor(
                fx=_BASE_FX, fy=_BASE_FY, cx=_BASE_CX, cy=_BASE_CY,
                device=device,
                depth_source=getattr(self.config, 'bev_depth_source', 'gt'),
                dav2_max_depth=getattr(self.config, 'bev_dav2_max_depth', 10.0),
                z_min=getattr(self.config, 'bev_z_min', -0.2),
                z_max=getattr(self.config, 'bev_z_max', 2.5),
            )
            self._unified_provider = UnifiedImageProvider(
                processor,
                s1_view=getattr(self.config, 's1_image_view', 'fpv'),
                s1_type=getattr(self.config, 's1_image_type', 'rgb'),
                s1_image_mode=getattr(self.config, 's1_image_mode', 'raw'),
                s1_combine=getattr(self.config, 's1_combine_mode', 'none'),
                s2_view=getattr(self.config, 's2_image_view', 'fpv'),
                s2_type=getattr(self.config, 's2_image_type', 'rgb'),
                s2_image_mode=getattr(self.config, 's2_image_mode', 'raw'),
                s2_combine=getattr(self.config, 's2_combine_mode', 'none'),
                depth_adapter_mode=getattr(self.config, 'depth_adapter_mode', 'repeat'),
                debug_dir=getattr(self.config, 'debug_dir', None),
            )
            self._unified_provider_key = key
        return self._unified_provider

    def _get_depth_adapter(self, in_channels: int) -> ImageInputAdapter:
        mode = getattr(self.config, 'depth_adapter_mode', 'repeat')
        key = (in_channels, mode)
        if self._depth_adapter is None or self._depth_adapter_key != key:
            self._depth_adapter = ImageInputAdapter(in_channels, out_channels=3, mode=mode).to(self.device)
            self._depth_adapter_key = key
        return self._depth_adapter

    def _save_debug(self, traj_images_orig, depth_image, traj_images_adapted):
        from internnav.model.utils.depth_rgb_to_bev_torch import save_image

        debug_dir = getattr(self.config, 'debug_dir', None)
        if not debug_dir:
            return
        try:
            os.makedirs(debug_dir, exist_ok=True)
            s = self._debug_step
            for b in range(min(traj_images_orig.shape[0], 4)):
                prefix = f"{debug_dir}/unified_step_{s:06d}_b{b:02d}"
                save_image(traj_images_orig[b, 0], f"{prefix}_1_fpv.jpg")
                depth_vis = depth_image[b, 0]
                if depth_vis.shape[-1] == 1:
                    depth_vis = depth_vis.repeat_interleave(3, dim=-1)
                # display-only per-frame normalization: 'raw' mode is metric depth
                # (metres), which .clamp(0,1) alone washes out to near-white — this
                # does not affect the tensor actually fed to the model, only the jpg.
                depth_vis = (depth_vis / depth_vis.max().clamp(min=1e-3)).clamp(0, 1)
                save_image(depth_vis, f"{prefix}_2_depth_raw.jpg")
                adapted_vis = traj_images_adapted[b, 0]
                adapted_vis = (adapted_vis / adapted_vis.max().clamp(min=1e-3)).clamp(0, 1)
                save_image(adapted_vis, f"{prefix}_3_depth_adapted.jpg")
        except Exception:
            import traceback
            traceback.print_exc()
        self._debug_step += 1

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

        s1_combine = getattr(self.config, 's1_combine_mode', 'none')
        debug_dir = getattr(self.config, 'debug_dir', None)
        traj_bev_images = None

        if labels is not None and s1_combine != 'none':
            provider = self._get_unified_provider(traj_images.device)
            cam_pitch = traj_cam_pitch_2.float() if traj_cam_pitch_2 is not None else None
            cam_height = traj_cam_heights.float() if traj_cam_heights is not None else None
            depths_5d = traj_depths.unsqueeze(-1) if traj_depths is not None else None
            out = provider.get_s1_input(traj_images, depths_5d, cam_pitch_deg=cam_pitch, cam_height=cam_height)
            if out.extra_images is not None:
                traj_bev_images = out.extra_images.to(dtype=traj_images.dtype)

            # tdmap debug is only aligned with provider._debug_step on the bev
            # delegation path (BEVImageProvider.get_s1_input increments it); the
            # fpv+depth path uses a separate unified_step_ counter (_save_debug
            # below) and isn't wired here.
            if traj_tdmaps is not None and debug_dir and provider.s1_view == 'bev':
                try:
                    wh = float(traj_world_headings[0, 0]) if traj_world_headings is not None else None
                    provider.save_train_tdmap_debug(traj_tdmaps[0, 0], world_heading=wh)
                except Exception:
                    import traceback; traceback.print_exc()

            traj_images_orig = traj_images
            if out.images is not None:
                in_channels = out.images.shape[-1]
                if in_channels == 3:
                    traj_images = out.images.to(dtype=traj_images_orig.dtype)
                else:
                    adapter = self._get_depth_adapter(in_channels)
                    traj_images = adapter(out.images).to(dtype=traj_images_orig.dtype)
                if debug_dir and provider.s1_view == 'fpv':
                    self._save_debug(traj_images_orig, out.images, traj_images)
            if out.depths is not None:
                traj_depths = out.depths[..., 0]
        elif labels is not None and debug_dir and traj_images is not None:
            # s1_combine=='none' (true no-op) — still visualize the raw S1 input
            # + tdmap for comparison against the replace/concat runs. Debug-only:
            # traj_images/traj_depths passed to super().forward() below are untouched.
            provider = self._get_unified_provider(traj_images.device)
            depths_5d = traj_depths.unsqueeze(-1) if traj_depths is not None else None
            provider.save_none_debug(traj_images, depths_5d, traj_tdmaps, traj_world_headings)

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
            traj_cam_heights=traj_cam_heights,
            traj_cam_pitch_1=traj_cam_pitch_1,
            traj_cam_pitch_2=traj_cam_pitch_2,
            video_frame_num=video_frame_num,
            traj_poses=traj_poses,
            traj_tdmaps=traj_tdmaps,
            traj_world_headings=traj_world_headings,
            traj_bev_images=traj_bev_images,
        )
