#!/usr/bin/env python3
"""Standalone System-1 (S1) inference for InternVLA-N1, decoupled from the
dual-system agent / threading / evaluator.

S1 is the trajectory generator. It is INDEPENDENT of the Qwen LLM: given the
`traj_latents` produced by S2 (`generate_latents`, shape [B, n_query=4, 3584])
plus the pixel-goal frame and current frame, it denoises a continuous trajectory
and converts it to discrete actions. The Qwen2.5-VL backbone is NOT needed here.

Two backends (pick with `backend=`):
  * "lite"  (default) - builds ONLY the S1 submodules (traj_dit, cond_projector,
              action_encoder/decoder, rgb_model=DAv2-DINO, memory_encoder,
              rgb_resampler) and loads ONLY the System-1 weights from the
              checkpoint. No 7B Qwen is allocated -> lightweight (~1 GB).
  * "full"  - loads the whole InternVLAN1ForCausalLM and calls its production
              `generate_traj`. Heavy (loads Qwen too) but bit-for-bit the
              reference. Use it to validate the lite backend (`--verify`).

The `generate_traj` math in the lite backend is copied verbatim from
internnav/model/basemodel/internvla_n1/internvla_n1.py:generate_traj
(nextdit_async branch) so the two backends stay in lock-step.

Run from the repo root (loads checkpoints/depth_anything_v2_metric_hypersim_vits.pth):
    convenv/bin/python quantization_kit/s1_inference.py --smoke
    convenv/bin/python quantization_kit/s1_inference.py --verify
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file

from internnav.model.basemodel.internvla_n1.internvla_n1_arch import (
    MemoryEncoder,
    QFormer,
    SinusoidalPositionalEncoding,
    build_traj_dit,
)
from internnav.model.utils.vln_utils import chunk_token, traj_to_actions


def _build_dav2(dav2_ckpt="checkpoints/depth_anything_v2_metric_hypersim_vits.pth"):
    """Build the DepthAnythingV2-vits backbone (rgb_model) WITHOUT triggering
    internnav.model.encoder.__init__ (which eagerly imports CLIP/gym/etc. that the
    lite S1 path does not need). We stub the encoder package with just a __path__ so
    Python can still resolve the depth_anything submodule.
    """
    import types

    repo = str(Path(__file__).resolve().parents[1])
    enc_pkg = "internnav.model.encoder"
    if not getattr(sys.modules.get(enc_pkg), "_lite_stub", False):
        m = types.ModuleType(enc_pkg)
        m.__path__ = [os.path.join(repo, "internnav/model/encoder")]
        m._lite_stub = True
        sys.modules[enc_pkg] = m

    from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2

    dav2 = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    dav2.load_state_dict(torch.load(dav2_ckpt, map_location="cpu"), assign=True)
    return dav2.pretrained

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

# S1 submodule prefixes (after stripping the leading "model.") that live in the
# checkpoint. latent_queries is excluded: it belongs to S2's generate_latents path.
_S1_SUBMODULES = (
    "cond_projector",
    "traj_dit",
    "action_encoder",
    "action_decoder",
    "rgb_model",
    "memory_encoder",
    "rgb_resampler",
)

_SYS1_DEPTH_THRESHOLD = 5.0  # metres; matches internvla_n1_agent.py


class _ConfigLike:
    """Minimal stand-in so build_traj_dit can read `_name_or_path` for FFN autodetect."""

    def __init__(self, model_path):
        self._name_or_path = model_path


class S1LiteModel(nn.Module):
    """Holds only the System-1 nextdit_async submodules and runs generate_traj."""

    def __init__(self, model_path, async_mode=True):
        super().__init__()
        cfg = _ConfigLike(model_path)
        self.async_mode = async_mode

        self.traj_dit, _ = build_traj_dit(cfg)             # NextDiTCrossAttn + scheduler (scheduler unused here)
        self.action_encoder = nn.Linear(3, 384, bias=True)
        self.pos_encoding = SinusoidalPositionalEncoding(384)
        self.action_decoder = nn.Linear(384, 3, bias=True)
        self.cond_projector = nn.Sequential(
            nn.Linear(3584, 768), nn.GELU(approximate="tanh"), nn.Linear(768, 768)
        )
        if async_mode:
            self.rgb_model = _build_dav2()                  # DepthAnythingV2 vits .pretrained (DINOv2)
            self.memory_encoder = MemoryEncoder()
            self.rgb_resampler = QFormer()

        self.register_buffer("_resnet_mean", torch.FloatTensor(_RESNET_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("_resnet_std", torch.FloatTensor(_RESNET_STD).view(1, 1, 3, 1, 1))

        # goal/memory-frame DINOv2 feature cache (frame 0). DINOv2 is a per-image function
        # (LayerNorm-only, no cross-frame attention), so the goal frame's features are
        # independent of the current frame -> reusing them while the goal is unchanged is
        # mathematically exact. Off by default (set via enable_rgb_cache()).
        self._rgb_cache_enabled = False
        self._rgb_cache_key = None
        self._rgb_cache_feat = None
        self._rgb_cache_misses = 0

    def enable_rgb_cache(self, enabled=True):
        """Reuse the goal-frame (frame 0) DINOv2 features across calls with the same cache_key.
        Invalidate by passing a new key (i.e. when S2 delivers a new plan/goal frame)."""
        self._rgb_cache_enabled = bool(enabled)
        self._rgb_cache_key = None
        self._rgb_cache_feat = None
        self._rgb_cache_misses = 0
        return self

    def _encode_rgb(self, x_flat, n_frames, cache_key=None):
        """x_flat: [n_frames, 3, H, W] (frame 0 = goal/memory). Returns feat [1, n_frames, N, C].
        With the cache on, the goal frame is encoded once per cache_key and reused; only the
        current frame(s) are re-encoded each call."""
        backbone = self.rgb_model.get_intermediate_layers
        if not self._rgb_cache_enabled or cache_key is None or n_frames < 2:
            return backbone(x_flat)[0].unflatten(0, (1, n_frames))
        if self._rgb_cache_key == cache_key and self._rgb_cache_feat is not None:
            goal_feat = self._rgb_cache_feat                       # exact reuse (no recompute)
        else:
            goal_feat = backbone(x_flat[:1])[0]                    # encode goal once, cache it
            self._rgb_cache_feat, self._rgb_cache_key = goal_feat, cache_key
            self._rgb_cache_misses += 1
        cur_feat = backbone(x_flat[1:])[0]                         # encode current frame(s)
        return torch.cat([goal_feat, cur_feat], dim=0).unflatten(0, (1, n_frames))

    @torch.no_grad()
    def load_s1_weights(self, model_path):
        sd = {}
        for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            for k, v in load_file(shard).items():
                if not k.startswith("model."):
                    continue
                stripped = k[len("model."):]
                if stripped.split(".")[0] in _S1_SUBMODULES:
                    sd[stripped] = v
        missing, unexpected = self.load_state_dict(sd, strict=False)
        # pos_encoding/buffers carry no checkpoint weights -> filter those out of `missing`
        real_missing = [m for m in missing if not m.startswith(("pos_encoding", "_resnet"))]
        print(f"[lite] loaded {len(sd)} S1 tensors  | missing={len(real_missing)} unexpected={len(unexpected)}")
        if real_missing:
            print("  missing:", real_missing[:8], "..." if len(real_missing) > 8 else "")
        if unexpected:
            print("  unexpected:", unexpected[:8], "..." if len(unexpected) > 8 else "")
        return self

    # --- verbatim port of internvla_n1.py generate_traj (nextdit_async branch) ---
    @torch.no_grad()
    def generate_traj(self, traj_latents, images_dp, depths_dp=None, predict_step_nums=32,
                      guidance_scale=1.0, num_inference_steps=10, num_sample_trajs=32,
                      rgb_cache_key=None):
        scheduler = FlowMatchEulerDiscreteScheduler()
        device, dtype = traj_latents.device, traj_latents.dtype

        traj_latents = self.cond_projector(traj_latents)
        if self.async_mode:
            images_dp = images_dp.permute(0, 1, 4, 2, 3)
            images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std
            self.rgb_model.to(dtype)
            images_dp_feat = self._encode_rgb(
                images_dp_norm.flatten(0, 1).to(dtype), images_dp_norm.shape[1], cache_key=rgb_cache_key
            )
            memory_feat = self.memory_encoder(images_dp_feat.flatten(1, 2))
            memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
            memory_tokens = self.rgb_resampler(memory_feat)
            hidden_states = torch.cat([memory_tokens, traj_latents], dim=1)
        else:
            hidden_states = traj_latents

        hidden_states_null = torch.zeros_like(hidden_states, device=device, dtype=dtype)
        hidden_states_input = torch.cat([hidden_states_null, hidden_states], 0)
        batch_size = traj_latents.shape[0]

        latents = randn_tensor(
            shape=(batch_size * num_sample_trajs, predict_step_nums, 3),
            generator=None, device=device, dtype=dtype,
        )
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)

        for t in scheduler.timesteps:
            latent_features = self.action_encoder(latents)
            pos_ids = torch.arange(latent_features.shape[1]).reshape(1, -1).repeat(batch_size, 1).to(device)
            latent_features = latent_features + self.pos_encoding(pos_ids)
            latent_model_input = latent_features.repeat(2, 1, 1)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.traj_dit(
                x=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(device, torch.long),
                z_latents=hidden_states_input,
            )
            noise_pred = self.action_decoder(noise_pred)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        return latents


def prep_images(rgb_memory, depth_memory, rgb, depth, device, dtype=torch.float32):
    """Reproduce internvla_n1_agent.py async S1 image prep -> rgbs[1,2,224,224,3], depths[1,2,224,224,1]."""
    from PIL import Image

    def _rgb(x):
        return np.array(Image.fromarray(x).resize((224, 224))) / 255.0

    def _depth(x):
        d = np.array(Image.fromarray(x).resize((224, 224))) * 10.0
        d[d > _SYS1_DEPTH_THRESHOLD] = _SYS1_DEPTH_THRESHOLD
        return d

    if depth_memory.ndim == 3:
        depth_memory = depth_memory[:, :, 0]
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    rgbs = torch.stack([torch.from_numpy(_rgb(rgb_memory)), torch.from_numpy(_rgb(rgb))]).unsqueeze(0)
    depths = torch.stack([torch.from_numpy(_depth(depth_memory)), torch.from_numpy(_depth(depth))]).unsqueeze(0).unsqueeze(-1)
    return rgbs.to(device, dtype), depths.to(device, dtype)


def prep_images_gpu(rgb_memory, depth_memory, rgb, depth, device, dtype=torch.float32):
    """Same output as prep_images ([1,2,224,224,3] / [1,2,224,224,1]) but resizes on the GPU
    (torch.interpolate) instead of CPU PIL -> ~5x faster prep. NOTE: GPU bicubic is NOT bit-identical
    to PIL bicubic, so this is an APPROXIMATION; verify action fidelity (verify_s1_opt.py) before
    relying on it. (The async DualVLN S1 path ignores depth, so only RGB resampling matters.)"""
    import torch.nn.functional as F

    def _rgb(x):
        t = torch.as_tensor(np.ascontiguousarray(x), device=device).permute(2, 0, 1)[None].to(dtype)
        t = F.interpolate(t, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        return (t / 255.0).clamp_(0, 1)[0].permute(1, 2, 0)            # [224,224,3]

    def _depth(x):
        if x.ndim == 3:
            x = x[:, :, 0]
        t = torch.as_tensor(np.ascontiguousarray(x), device=device, dtype=dtype)[None, None]
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False) * 10.0
        return t.clamp_(max=_SYS1_DEPTH_THRESHOLD)[0, 0]               # [224,224]

    rgbs = torch.stack([_rgb(rgb_memory), _rgb(rgb)])[None]            # [1,2,224,224,3]
    depths = torch.stack([_depth(depth_memory), _depth(depth)])[None].unsqueeze(-1)  # [1,2,224,224,1]
    return rgbs.to(dtype), depths.to(dtype)


class S1Inferencer:
    def __init__(self, model_path="checkpoints/InternVLA-N1-DualVLN", device="cuda:0",
                 dtype=torch.float32, backend="lite", continuous_traj=True):
        # NOTE: float32 is the safe default for the lite backend. The production stack
        # runs S1 in bf16, but that relies on the older diffusers it was trained with;
        # newer diffusers upcast inside traj_dit.norm_out and break bf16. S1 is tiny
        # (~1.2 GB in fp32) so fp32 is cheap and avoids the mixed-dtype landmine.
        self.device = device
        self.dtype = dtype
        self.backend = backend
        self.continuous_traj = continuous_traj

        if backend == "lite":
            self.model = S1LiteModel(model_path).load_s1_weights(model_path).to(device, dtype).eval()
        elif backend == "full":
            from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
            self.model = InternVLAN1ForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, attn_implementation="sdpa", device_map={"": device},
            ).eval()
        else:
            raise ValueError(f"backend must be 'lite' or 'full', got {backend!r}")

    @torch.no_grad()
    def infer(self, traj_latents, rgb_memory, depth_memory, rgb, depth,
              num_inference_steps=10, num_sample_trajs=32, guidance_scale=1.0):
        """traj_latents: [B, 4, 3584] from S2.  Returns dict(trajectory, actions)."""
        traj_latents = traj_latents.to(self.device, self.dtype)
        rgbs, depths = prep_images(rgb_memory, depth_memory, rgb, depth, self.device, self.dtype)

        dp_actions = self.model.generate_traj(
            traj_latents=traj_latents, images_dp=rgbs, depths_dp=depths,
            num_inference_steps=num_inference_steps, num_sample_trajs=num_sample_trajs,
            guidance_scale=guidance_scale,
        )
        if self.continuous_traj:
            actions = traj_to_actions(dp_actions.clone())
        else:
            choice = np.random.choice(dp_actions.shape[0])
            actions = chunk_token(dp_actions[choice])
        actions = [a for a in actions if a != 0]
        return {"trajectory": dp_actions, "actions": actions[:4]}


def _dummy_inputs(device):
    g = torch.Generator(device="cpu").manual_seed(0)
    latent = torch.randn(1, 4, 3584, generator=g)
    rgb = (torch.rand(480, 640, 3, generator=g).numpy() * 255).astype(np.uint8)
    depth = (torch.rand(480, 640, generator=g).numpy() * 3.0).astype(np.float32)
    return latent, rgb, depth, rgb, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="checkpoints/InternVLA-N1-DualVLN")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--backend", default="lite", choices=["lite", "full"])
    ap.add_argument("--smoke", action="store_true", help="build + run one dummy latent")
    ap.add_argument("--verify", action="store_true",
                    help="sanity-check the lite backend: all S1 weights load, output finite & well-shaped")
    args = ap.parse_args()

    latent, rgb_m, depth_m, rgb, depth = _dummy_inputs(args.device)

    if args.verify:
        # In this host env (diffusers > training version) the bf16 `full` backend's
        # traj_dit is dtype-broken, and fp32 `full` would OOM (7B). So we verify the
        # lite extraction is self-consistent: complete weight load + sane output.
        # (bf16 bit-parity vs production should be checked inside the docker env.)
        inf = S1Inferencer(args.model_path, args.device, backend="lite")
        out = inf.infer(latent, rgb_m, depth_m, rgb, depth)
        traj = out["trajectory"]
        ok = (
            tuple(traj.shape[1:]) == (32, 3)
            and torch.isfinite(traj).all().item()
            and all(a in (0, 1, 2, 3, 5) for a in out["actions"])
        )
        print(f"[verify] trajectory={tuple(traj.shape)} finite={torch.isfinite(traj).all().item()} "
              f"actions={out['actions']}")
        print("[verify] PASS" if ok else "[verify] FAIL")
        return

    inf = S1Inferencer(args.model_path, args.device, backend=args.backend)
    if args.smoke:
        out = inf.infer(latent, rgb_m, depth_m, rgb, depth)
        print(f"[{args.backend}] trajectory {tuple(out['trajectory'].shape)}  actions={out['actions']}")


if __name__ == "__main__":
    main()
