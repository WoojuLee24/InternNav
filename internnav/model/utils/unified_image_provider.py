"""Unified S1/S2 image provider — view x type x mode x combine.

Generalizes ``BEVImageProvider`` (BEV-only) into four independent axes per
system (S1, S2):

    view    : 'fpv' | 'bev'                     — spatial viewpoint
    type    : 'rgb' | 'depth' | 'panorama'       — signal encoded
    mode    : value-processing, vocabulary depends on (view, type):
                (fpv, rgb)   -> 'raw'
                (bev, rgb)   -> 'raw'                                  (today's colored BEV)
                (bev, depth) -> 'raw'|'binary'|'prob'|'dist'|'dist_sep' (today's occupancy BEV)
                (fpv, depth) -> 'raw'|'normalized'|'colormap'          (new)
    combine : 'none' | 'replace' | 'concat'      — relation to the original FPV frame(s)/prompt

``UnifiedImageProvider`` inherits the bare ``VisualInputProvider`` ABC, NOT
``BEVImageProvider``. ``BEVImageProvider`` is frozen (no longer modified) and
represents a narrower BEV-only concept; ``UnifiedImageProvider`` also covers
view=='fpv' configurations that never touch BEV at all, so being an
``isinstance`` of a BEV-specific class would misrepresent those instances
(and did — see ``habitat_vln_evaluator_unified.py``'s ``s1_view``/``s2_view``
checks, which replaced earlier ``isinstance(provider, BEVImageProvider)``
checks that were always true regardless of view). The ``view=='bev'`` branches
of ``get_s1_input``/``get_s2_extra`` below call ``self.processor.for_s1`` /
``self.processor.for_s2`` directly (``BEVProcessor`` is the shared, unfrozen
geometry engine both this class and ``BEVImageProvider`` depend on) and own a
trimmed copy of the debug-image saving (``_save_s1_debug``,
``save_train_tdmap_debug``) — core fpv/depth/bev visualization only; the
ROS2-occupancy-grid and dav2-depth comparison debug code in
``BEVImageProvider._save_s1_debug`` is intentionally not duplicated here.

Panorama (``type=='panorama'``) is a stub: no panorama data exists anywhere
in the pipeline yet (see ``.claude/tasks/260707_s1s2_bev_depth_panorama.md``,
risk #1) — it raises ``NotImplementedError`` when actually invoked
(``combine != 'none'``).
"""

from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from internnav.model.utils.visual_input_provider import (
    BEVProcessor,
    S1VisualInput,
    VisualInputProvider,
    _cfg_get,
)

_VALID_VIEWS = ('fpv', 'bev', 'bev_ld')  # bev=FPV→BEV, bev_ld=lookdown→BEV
_VALID_TYPES = ('rgb', 'depth', 'panorama')
_VALID_COMBINE = ('none', 'replace', 'concat')
_FPV_DEPTH_MODES = ('raw', 'normalized', 'colormap')

# (view=bev, type=depth) mode -> legacy BEVProcessor image_type string.
_OCC_MODE_MAP = {
    'raw': 'occ',
    'binary': 'occ.binary',
    'prob': 'occ.prob',
    'dist': 'occ.dist',
    'dist_sep': 'occ.dist.sep',
}

_S2_LABELS = {
    ('fpv', 'rgb'):      "the front view of your surroundings",
    ('fpv', 'depth'):    "the depth view of your surroundings",
    ('bev', 'rgb'):      "the bird's-eye view of your surroundings",
    ('bev', 'depth'):    "the occupancy map of your surroundings",
    ('bev_ld', 'rgb'):   "the bird's-eye view of your surroundings",
    ('bev_ld', 'depth'): "the occupancy map of your surroundings",
    # 'panorama' intentionally excluded: get_s1_input/get_s2_extra raise
    # NotImplementedError for it before this label would ever be used.
}


def _per_frame_normalize(depth: torch.Tensor) -> torch.Tensor:
    """[N, H, W] metric depth -> [N, H, W] in [0, 1], per-frame min-max."""
    flat = depth.flatten(1)
    dmin = flat.min(dim=1, keepdim=True).values
    dmax = flat.max(dim=1, keepdim=True).values
    scale = (dmax - dmin).clamp(min=1e-6)
    return ((flat - dmin) / scale).view_as(depth)


def _depth_to_colormap(depth_01: torch.Tensor) -> torch.Tensor:
    """[N, H, W] depth in [0, 1] -> [N, H, W, 3] float [0, 1] JET colormap.

    ponytail: per-frame cv2 loop (CPU round-trip) — fine for the batch sizes
    this trains with; revisit with a batched GPU colormap if profiling shows
    this as a bottleneck.
    """
    device, dtype = depth_01.device, depth_01.dtype
    depth_np = (depth_01.clamp(0, 1).float().detach().cpu().numpy() * 255).astype(np.uint8)
    frames = [cv2.applyColorMap(depth_np[i], cv2.COLORMAP_JET)[..., ::-1].copy() for i in range(depth_np.shape[0])]
    return torch.from_numpy(np.stack(frames)).to(device=device, dtype=dtype) / 255.0


class ImageInputAdapter(nn.Module):
    """Match an arbitrary-channel image to the channel count ``rgb_model`` expects.

    'repeat' (default) has no learnable params — safe to drop in front of a
    pretrained rgb_model. 'conv' adds a learnable 1x1 projection when a
    trained channel-mixing adapter is wanted instead.
    """

    def __init__(self, in_channels: int, out_channels: int = 3, mode: str = 'repeat'):
        super().__init__()
        if mode not in ('repeat', 'conv'):
            raise ValueError(f"mode must be 'repeat' or 'conv', got {mode!r}")
        if mode == 'repeat' and in_channels != 1 and in_channels != out_channels:
            raise ValueError(f"'repeat' adapter only supports in_channels==1 (got {in_channels})")
        self.in_channels, self.out_channels, self.mode = in_channels, out_channels, mode
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if mode == 'conv' else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., H, W, C] channels-last -> [..., H, W, out_channels]."""
        if x.shape[-1] == self.out_channels:
            return x
        if self.mode == 'repeat':
            return x.repeat_interleave(self.out_channels, dim=-1)
        shape = x.shape
        chw = x.reshape(-1, *shape[-3:]).permute(0, 3, 1, 2).float()
        out = self.proj(chw).permute(0, 2, 3, 1).reshape(*shape[:-1], self.out_channels)
        return out.to(x.dtype)


class UnifiedImageProvider(VisualInputProvider):
    """view x type x mode x combine, independently for S1 and S2.

    ``view=='bev'`` calls the shared ``BEVProcessor`` directly (same math
    ``BEVImageProvider`` uses, but not inherited from it — see module
    docstring). ``view=='fpv', type=='depth'`` is new logic added here.
    ``type=='panorama'`` is a stub.
    """

    def __init__(
        self,
        processor: BEVProcessor,
        s1_view: str = 'fpv', s1_type: str = 'rgb', s1_image_mode: str = 'raw', s1_combine: str = 'none',
        s2_view: str = 'fpv', s2_type: str = 'rgb', s2_image_mode: str = 'raw', s2_combine: str = 'none',
        s1_pitch_deg: Optional[float] = None,
        s2_pitch_deg: Optional[float] = None,
        s2_depth_in_meters: bool = False,
        depth_adapter_mode: str = 'repeat',
        debug_dir: Optional[str] = None,
    ):
        for name, v in (('s1_view', s1_view), ('s2_view', s2_view)):
            if v not in _VALID_VIEWS:
                raise ValueError(f"{name} must be one of {_VALID_VIEWS}, got {v!r}")
        for name, v in (('s1_type', s1_type), ('s2_type', s2_type)):
            if v not in _VALID_TYPES:
                raise ValueError(f"{name} must be one of {_VALID_TYPES}, got {v!r}")
        for name, v in (('s1_combine', s1_combine), ('s2_combine', s2_combine)):
            if v not in _VALID_COMBINE:
                raise ValueError(f"{name} must be one of {_VALID_COMBINE}, got {v!r}")

        self.processor = processor
        self.s1_pitch_deg = s1_pitch_deg
        self.s2_pitch_deg = s2_pitch_deg
        self.s2_depth_in_meters = s2_depth_in_meters
        # legacy occ-type string BEVProcessor.for_s1/for_s2 expect ('rgb'|'occ'|'occ.binary'|...)
        self.image_type = _OCC_MODE_MAP.get(s1_image_mode, 'rgb') if s1_type == 'depth' else 'rgb'
        self.debug_dir = debug_dir
        self._debug_step = 0
        self._s2_debug_step = 0

        self.s1_view, self.s1_type, self.s1_image_mode, self.s1_combine = s1_view, s1_type, s1_image_mode, s1_combine
        self.s2_view, self.s2_type, self.s2_image_mode, self.s2_combine = s2_view, s2_type, s2_image_mode, s2_combine
        self.depth_adapter_mode = depth_adapter_mode
        self.s2_image_label = _S2_LABELS.get((s2_view, s2_type), _S2_LABELS[('bev', 'rgb')])

    # ------------------------------------------------------------------ debug

    def _save_s1_debug(self, rgb, depth, bev):
        """Save frame[0] (and frame[1] for the eval [goal, cur] pair) per batch sample:
        FPV | GT-depth | BEV-out. Core visualization only (no ROS2/dav2 comparison —
        see module docstring).

        Args:
            rgb   : [B, T, H, W, 3] float [0,1]
            depth : [B, T, H, W, 1] metres, or None
            bev   : [B, T, H, W, 3] float [0,1]  — provider output
        """
        import os
        from internnav.model.utils.depth_rgb_to_bev_torch import save_image
        os.makedirs(self.debug_dir, exist_ok=True)
        s = self._debug_step
        depth_src = self.processor.depth_source  # 'gt' | 'dav2' | 'udv2'

        def _save_frame(prefix, b, t, suffix):
            save_image(rgb[b, t], f"{prefix}_1_fpv{suffix}.jpg")
            save_image(bev[b, t], f"{prefix}_3_bev_{depth_src}{suffix}.jpg")
            if depth is not None:
                gt_d = depth[b, t, ..., 0]
                depth_vis = (gt_d / gt_d.max().clamp(min=0.1)).clamp(0, 1)
                save_image(depth_vis, f"{prefix}_2_gt_depth{suffix}.jpg")

        for b in range(rgb.shape[0]):
            prefix = f"{self.debug_dir}/step_{s:06d}_b{b:02d}"
            _save_frame(prefix, b, 0, '')
            # T=1 (current frame) — only for the eval re-call pattern, which always
            # stacks exactly 2 frames ([pixel_goal, cur]); training passes
            # num_history frames instead, so this only fires for the eval pair.
            if rgb.shape[1] == 2:
                _save_frame(prefix, b, 1, '_cur')

    def save_train_tdmap_debug(
        self, tdmap: torch.Tensor, world_heading: Optional[float] = None, batch_idx: int = 0,
        prefix: Optional[str] = None,
    ) -> None:
        """Save tdmap + world-aligned BEV for training/eval debug. Call AFTER get_s1_input.

        Agent-frame (order: bev → gt_topdown):
          step_XXXXXX_bNN_3_bev_{src}.jpg       — BEV (agent frame)  [saved by _save_s1_debug]
          step_XXXXXX_bNN_5_gt_topdown.jpg      — topdown (agent forward = up)

        World-frame (order: bev → gt_topdown), only when world_heading is not None:
          step_XXXXXX_bNN_w3_bev_world_{src}.jpg   — BEV rotated to world frame
          step_XXXXXX_bNN_w5_gt_topdown_world.jpg  — topdown rotated to world + FWD arrow

        {src} = processor.depth_source ('gt' | 'dav2' | 'udv2')
        """
        if not self.debug_dir:
            return
        try:
            import math
            import os
            from internnav.model.utils.depth_rgb_to_bev_torch import save_image

            debug_dir = self.debug_dir
            S         = self.processor.bev_size
            bev_range = self.processor.bev_range
            os.makedirs(debug_dir, exist_ok=True)
            if prefix is None:
                s = self._debug_step - 1
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

            # world-aligned BEV
            _dsrc = self.processor.depth_source
            src_sfx, dst_sfx = f'_3_bev_{_dsrc}.jpg', f'_w3_bev_world_{_dsrc}.jpg'
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

    # ------------------------------------------------------------ internals

    def _resolve_depth(self, rgb: torch.Tensor, depth: Optional[torch.Tensor]) -> torch.Tensor:
        """[B, T, H, W, 3] rgb, [B, T, H, W(,1)] depth|None -> [B, T, H, W] metric depth."""
        if self.processor.depth_source in ('dav2', 'udv2'):
            B, T, H, W = rgb.shape[:4]
            est = self.processor._estimate_depth(rgb.flatten(0, 1))
            return est.view(B, T, H, W)
        if depth is None:
            raise ValueError("depth is required when depth_source == 'gt'")
        return depth[..., 0] if depth.dim() == 5 else depth

    def _depth_images(self, depth: torch.Tensor, mode: str) -> torch.Tensor:
        """[B, T, H, W] metric depth -> [B, T, H, W, C] float image."""
        if mode not in _FPV_DEPTH_MODES:
            raise ValueError(f"image_mode must be one of {_FPV_DEPTH_MODES} for view=fpv,type=depth, got {mode!r}")
        B, T, H, W = depth.shape
        flat = depth.flatten(0, 1)  # [B*T, H, W]
        if mode == 'raw':
            img = flat.unsqueeze(-1)
        elif mode == 'normalized':
            img = _per_frame_normalize(flat).unsqueeze(-1)
        else:  # colormap
            img = _depth_to_colormap(_per_frame_normalize(flat))
        return img.view(B, T, H, W, img.shape[-1]).to(dtype=depth.dtype)

    # ------------------------------------------------------------------ API

    def get_s1_input(self, rgb, depth=None, cam_pitch_deg=None, cam_height=None) -> S1VisualInput:
        if self.s1_combine == 'none':
            return S1VisualInput()
        if self.s1_type == 'panorama':
            # Checked before the view/combine dispatch below so panorama raises
            # regardless of view/combine — it's a stub, no data yet.
            raise NotImplementedError(
                "s1_image_type='panorama' needs a data-collection pipeline extension "
                "(see .claude/tasks/260707_s1s2_bev_depth_panorama.md, risk #1) — not available yet."
            )

        if self.s1_view == 'bev':
            if self.s1_combine == 'replace':
                return self._get_s1_bev_replace(rgb, depth, cam_pitch_deg, cam_height)
            elif self.s1_combine == 'concat':
                # Unlike 'replace', BEV does NOT substitute the FPV frame here — it's
                # returned as an ADDITIONAL slot (S1VisualInput.extra_images) that the
                # caller stacks alongside [goal, cur] as a 3rd image, folded into the
                # token dim (not the batch dim) by internvla_n1.py's
                # _encode_s1_memory_tokens. (A prior attempt instead doubled
                # traj_images' T axis directly, which desynced memory_tokens' batch dim
                # from traj_hidden_states' — see internvla_n1.py's forward()/generate_traj()
                # for how the token-dim approach avoids that.)
                return self._get_s1_bev_concat(rgb, depth, cam_pitch_deg, cam_height)
            else:
                assert False, f"unreachable s1_combine={self.s1_combine!r}"

        elif self.s1_view == 'fpv':
            if self.s1_type == 'rgb':
                raise ValueError(
                    f"s1_combine_mode={self.s1_combine!r} has nothing to {self.s1_combine} "
                    "s1_image_view='fpv' with when s1_image_type='rgb' too — that would substitute "
                    "the FPV frame with itself. Use s1_image_type='depth' or s1_image_view='bev' instead."
                )
            assert self.s1_type == 'depth'
            if self.s1_combine == 'concat':
                # Same nextdit_async T-doubling issue as the view='bev' branch above.
                raise NotImplementedError(
                    "s1_combine_mode='concat' is not implemented for s1_image_type='depth': it requires "
                    "re-deriving video_frame_num / traj_poses alignment for the doubled T dimension."
                )
            assert self.s1_combine == 'replace'
            return self._get_s1_fpv_depth_replace(rgb, depth)

        else:
            assert False, f"unreachable s1_view={self.s1_view!r} ('bev_ld' is S2-only, not valid for S1)"

    def _get_s1_fpv_depth_replace(self, rgb, depth) -> S1VisualInput:
        depth_m = self._resolve_depth(rgb, depth)
        images = self._depth_images(depth_m, self.s1_image_mode)
        if self.debug_dir:
            try:
                # reuses _save_s1_debug's 3rd slot (labeled "_3_bev_{depth_src}.jpg" —
                # name kept as-is for that shared helper) for the depth-encoded image
                # that actually replaces the FPV frame here, so it's directly comparable
                # to _1_fpv.jpg / _2_gt_depth.jpg for the same step.
                self._save_s1_debug(rgb, depth, images)
            except Exception:
                import traceback; traceback.print_exc()
        self._debug_step += 1
        return S1VisualInput(images=images, depths=depth_m.unsqueeze(-1))

    def _get_s1_bev_replace(self, rgb, depth, cam_pitch_deg, cam_height) -> S1VisualInput:
        if not (isinstance(rgb, torch.Tensor) and (depth is None or isinstance(depth, torch.Tensor))):
            # Legacy 'sync' path hands raw numpy frames straight to generate_traj;
            # BEV substitution is only defined for the stacked-tensor layout.
            print('[UnifiedImageProvider] non-tensor S1 input — falling back to FPV for this call')
            return S1VisualInput()
        pitch = cam_pitch_deg if cam_pitch_deg is not None else self.s1_pitch_deg
        bev = self.processor.for_s1(rgb, depth, cam_pitch_deg=pitch,
                                     image_type=self.image_type, cam_height=cam_height)
        if self.debug_dir:
            try:
                self._save_s1_debug(rgb, depth, bev)
            except Exception:
                import traceback; traceback.print_exc()
        self._debug_step += 1
        return S1VisualInput(images=bev)

    def _get_s1_bev_concat(self, rgb, depth, cam_pitch_deg, cam_height) -> S1VisualInput:
        """Same BEV computation as _get_s1_bev_replace (pitch defaulting, debug save,
        non-tensor fallback all reused as-is) — returned as an EXTRA slot instead of a
        substitution, so the caller can stack it alongside [goal, cur] as a 3rd image."""
        replaced = self._get_s1_bev_replace(rgb, depth, cam_pitch_deg, cam_height)
        if replaced.images is None:
            return S1VisualInput()
        return S1VisualInput(extra_images=replaced.images)

    def save_none_debug(self, rgb, depth=None, tdmap=None, world_heading=None) -> None:
        """Debug-only visualization for s1_combine=='none' (true no-op).

        get_s1_input() returns immediately for combine='none' (nothing to save on
        that path, by design — it must stay byte-identical to image_provider=False).
        This is a separate, debug-only call so the no-op baseline still produces
        step_XXXXXX_bNN debug jpgs (fpv / gt depth / gt topdown) comparable to the
        combine='replace'/'concat' runs. Never touches the tensors fed to the model.
        """
        if not self.debug_dir:
            return
        import os
        from internnav.model.utils.depth_rgb_to_bev_torch import save_image
        os.makedirs(self.debug_dir, exist_ok=True)
        s = self._debug_step
        for b in range(rgb.shape[0]):
            prefix = f"{self.debug_dir}/step_{s:06d}_b{b:02d}"
            save_image(rgb[b, 0], f"{prefix}_1_fpv.jpg")
            if depth is not None:
                d = depth[b, 0, ..., 0]
                d_vis = (d / d.max().clamp(min=0.1)).clamp(0, 1)
                save_image(d_vis, f"{prefix}_2_gt_depth.jpg")
        self._debug_step += 1
        if tdmap is not None:
            try:
                wh = float(world_heading[0, 0]) if world_heading is not None else None
                self.save_train_tdmap_debug(tdmap[0, 0], world_heading=wh)
            except Exception:
                import traceback; traceback.print_exc()

    def get_s2_extra(self, rgb, depth, is_lookdown: bool = False, cam_pitch_deg: Optional[float] = None) -> List[Image.Image]:
        if self.s2_combine == 'none' or not is_lookdown:
            return []
        if self.s2_combine == 'replace':
            # No training-side implementation: internnav/dataset/internvla_n1_lerobot_dataset.py
            # only ever APPENDS S2 images (pitch_1 FPV + pitch_2 lookdown + BEV all kept — see its
            # get_s2_extra call site) regardless of s2_combine_mode; it never drops the FPV frame.
            # Evaluating/training with s2_combine_mode='replace' would exercise a code path with no
            # matching training data. Use s2_combine_mode='concat' instead.
            raise NotImplementedError(
                "s2_combine_mode='replace' has no training-side implementation (the LeRobot dataset "
                "always appends S2 images, never replaces) — use s2_combine_mode='concat' instead."
            )
        if self.s2_type == 'panorama':
            raise NotImplementedError(
                "s2_image_type='panorama' needs a data-collection pipeline extension — not available yet."
            )
        if self.s2_view in ('bev', 'bev_ld'):
            # S2's own occ mode, independent of s1_image_mode (unlike self.image_type,
            # which __init__ derives from s1_image_mode for S1's use).
            s2_occ_type = _OCC_MODE_MAP.get(self.s2_image_mode, 'rgb') if self.s2_type == 'depth' else 'rgb'
            bev = self.processor.for_s2(
                rgb, depth,
                depth_in_meters=self.s2_depth_in_meters,
                cam_pitch_deg=self.s2_pitch_deg if cam_pitch_deg is None else cam_pitch_deg,
                image_type=s2_occ_type,
            )
            if self.debug_dir:
                try:
                    import os
                    os.makedirs(self.debug_dir, exist_ok=True)
                    prefix = f"{self.debug_dir}/s2_step_{self._s2_debug_step:06d}"
                    # matches get_s1_input's _1_fpv.jpg / _N_bev_{src}.jpg naming (_save_s1_debug
                    # above) so the source frame and derived BEV image are directly comparable.
                    Image.fromarray(np.asarray(rgb).astype(np.uint8)).save(f"{prefix}_1_fpv.jpg")
                    type_sfx = '' if s2_occ_type == 'rgb' else f'_{s2_occ_type}'
                    bev.save(f"{prefix}_2_bev{type_sfx}_{self.processor.depth_source}.jpg")
                except Exception:
                    import traceback; traceback.print_exc()
                self._s2_debug_step += 1
            return [bev]
        if self.s2_type != 'depth':
            raise ValueError(f"Unknown s2_image_type {self.s2_type!r} (expected rgb|depth|panorama)")

        if self.processor.depth_source in ('dav2', 'udv2'):
            rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0).float() / 255.0
            depth_t = self.processor._estimate_depth(rgb_t)[0]
        else:
            depth_np = np.asarray(depth)
            if depth_np.ndim == 3:
                depth_np = depth_np[..., 0]
            depth_t = torch.from_numpy(np.ascontiguousarray(depth_np)).float()

        img = self._depth_images(depth_t.view(1, 1, *depth_t.shape[-2:]), self.s2_image_mode)[0, 0]  # [H, W, C]
        if img.shape[-1] == 1:
            img = img.repeat_interleave(3, dim=-1)
        img_np = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        if self.debug_dir:
            try:
                import os

                os.makedirs(self.debug_dir, exist_ok=True)
                prefix = f"{self.debug_dir}/s2_unified_step_{self._s2_debug_step:06d}"
                # matches S1's _1_fpv.jpg / _2_....jpg naming (internvla_n1_unified_provider.py
                # _save_debug) so the source frame and derived image are directly comparable.
                Image.fromarray(np.asarray(rgb).astype(np.uint8)).save(f"{prefix}_1_fpv.jpg")
                # display-only per-frame normalization: 'raw' mode is metric depth, which
                # clamp(0,1) alone washes out to near-white — does not affect img_np /
                # the returned image (the actual S2 model input), only this debug copy.
                img_vis = (img / img.max().clamp(min=1e-3)).clamp(0, 1)
                img_vis_np = (img_vis.cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(img_vis_np).save(f"{prefix}_2_{self.s2_image_mode}.jpg")
            except Exception:
                import traceback

                traceback.print_exc()
            self._s2_debug_step += 1

        return [Image.fromarray(img_np)]


def create_unified_provider(config, device: str = 'cuda:0') -> UnifiedImageProvider:
    """Build a UnifiedImageProvider from model_settings (dict / Namespace / pydantic).

    Reuses the same ``bev_*``-prefixed geometry keys as ``create_visual_provider``'s
    ``bev_image``/``bev_feature`` branches, plus the new ``s{1,2}_image_*``/
    ``s{1,2}_combine_mode`` axis keys.
    """
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
    return UnifiedImageProvider(
        processor,
        s1_view=_cfg_get(config, 's1_image_view', 'fpv'),
        s1_type=_cfg_get(config, 's1_image_type', 'rgb'),
        s1_image_mode=_cfg_get(config, 's1_image_mode', 'raw'),
        s1_combine=_cfg_get(config, 's1_combine_mode', 'none'),
        s2_view=_cfg_get(config, 's2_image_view', 'fpv'),
        s2_type=_cfg_get(config, 's2_image_type', 'rgb'),
        s2_image_mode=_cfg_get(config, 's2_image_mode', 'raw'),
        s2_combine=_cfg_get(config, 's2_combine_mode', 'none'),
        s1_pitch_deg=_cfg_get(config, 'bev_s1_pitch_deg', None),
        s2_pitch_deg=_cfg_get(config, 'bev_s2_pitch_deg', None),
        s2_depth_in_meters=_cfg_get(config, 'bev_s2_depth_in_meters', False),
        depth_adapter_mode=_cfg_get(config, 'depth_adapter_mode', 'repeat'),
        debug_dir=_cfg_get(config, 'debug_dir', None),
    )
