"""Export System-1 sub-networks of an InternVLA-N1 DualVLN checkpoint to ONNX (TensorRT rebuild source).

Reproduces quantization_kit/onnx/* (the original export tool was not shipped with the bundle). The full
diffusion loop is not a static graph (scheduler, CFG, data-dependent timestep list), so only the per-step
denoise network is exported for TensorRT, plus the once-per-call conditioning encoders (optional):

  s1_denoise_step.onnx   latents[B,32,3] i64 timestep[B2] z_latents[B2,N,768] i64 pos_ids[1,32] -> noise_pred[B2,32,3]
                         = action_decoder(traj_dit(cat2(action_encoder(latents) + pos_encoding(pos_ids)), t, z))
                         (exactly one iteration of S1LiteModel.generate_traj's loop; CFG combine + Euler step
                          stay in host code, see quantization_kit/trt_s1.py)
  rgb_model_dinov2.onnx  img[B,3,224,224] -> feat[B,256,384]          (DepthAnythingV2-S DINOv2 backbone)
  memory_encoder.onnx    memory[B,N,384] -> enc[1,N,384]
  rgb_resampler.onnx     visual[B,N,768] -> tokens[1,32,768]          (QFormer)
  cond_projector.onnx / action_encoder.onnx / action_decoder.onnx      (small heads, not TRT'd)

I/O names, dynamic axes and opsets (18 for denoise / DINOv2, 17 for the rest) follow the original files.
Weights: the checkpoint's bf16 System-1 tensors loaded into the fp32 S1LiteModel (same as the lite S1 service).

  quantization_kit/convenv_x86/bin/python deploy/tools/export_s1_onnx.py \
      --ckpt checkpoints/InternVLA-N1-DualVLN --out quantization_kit/out/onnx [--only s1_denoise_step]
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT), str(_ROOT / "quantization_kit")]
from quantization_kit.s1_inference import S1LiteModel  # noqa: E402


class DenoiseStep(nn.Module):
    """One denoise iteration of S1LiteModel.generate_traj (CFG batch = cond/uncond pair)."""

    def __init__(self, m):
        super().__init__()
        self.action_encoder, self.pos_encoding = m.action_encoder, m.pos_encoding
        self.traj_dit, self.action_decoder = m.traj_dit, m.action_decoder

    def forward(self, latents, timestep, z_latents, pos_ids):
        f = self.action_encoder(latents) + self.pos_encoding(pos_ids)
        x = f.repeat(2, 1, 1)
        return self.action_decoder(self.traj_dit(x=x, timestep=timestep, z_latents=z_latents))


class Dino(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.rgb_model = m.rgb_model

    def forward(self, img):
        return self.rgb_model.get_intermediate_layers(img)[0]


def specs(m):
    B, B2, N = 32, 64, 36
    return {
        "s1_denoise_step": (DenoiseStep(m), 18,
                            (torch.randn(B, 32, 3), torch.full((B2,), 500, dtype=torch.long), torch.randn(B2, N, 768),
                             torch.arange(32).reshape(1, 32)),
                            ["latents", "timestep", "z_latents", "pos_ids"], ["noise_pred"],
                            {"latents": {0: "B"}, "timestep": {0: "B2"}, "z_latents": {0: "B2", 1: "N"},
                             "noise_pred": {0: "B2"}}),
        "rgb_model_dinov2": (Dino(m), 18, (torch.randn(2, 3, 224, 224),), ["img"], ["feat"],
                             {"img": {0: "B"}, "feat": {0: "B"}}),
        "memory_encoder": (m.memory_encoder, 17, (torch.randn(1, 512, 384),), ["memory"], ["enc"],
                           {"memory": {0: "B", 1: "N"}, "enc": {1: "N"}}),
        "rgb_resampler": (m.rgb_resampler, 17, (torch.randn(1, 512, 768),), ["visual"], ["tokens"],
                          {"visual": {0: "B", 1: "N"}}),
        "cond_projector": (m.cond_projector, 17, (torch.randn(1, 4, 3584),), ["latent"], ["proj"],
                           {"latent": {0: "B", 1: "n"}, "proj": {0: "B", 1: "n"}}),
        "action_encoder": (m.action_encoder, 17, (torch.randn(32, 32, 3),), ["x"], ["feat"],
                           {"x": {0: "B", 1: "T"}, "feat": {0: "B", 1: "T"}}),
        "action_decoder": (m.action_decoder, 17, (torch.randn(64, 32, 384),), ["feat"], ["xyt"],
                           {"feat": {0: "B", 1: "T"}, "xyt": {0: "B", 1: "T"}}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(_ROOT / "checkpoints/InternVLA-N1-DualVLN"))
    ap.add_argument("--out", default=str(_ROOT / "quantization_kit/out/onnx"))
    ap.add_argument("--only", default="", help="comma list of component names (default: all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # nn.TransformerEncoder's eval fast path (aten::_transformer_encoder_layer_fwd) has no ONNX symbolic;
    # the regular path computes the same function
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(0)
    m = S1LiteModel(a.ckpt).load_s1_weights(a.ckpt).float().eval()
    todo = specs(m)
    names = [n for n in todo if not a.only or n in a.only.split(",")]
    for name in names:
        mod, opset, args, inames, onames, dyn = todo[name]
        path = os.path.join(a.out, f"{name}.onnx")
        with torch.no_grad():
            torch.onnx.export(mod, args, path, input_names=inames, output_names=onames, dynamic_axes=dyn,
                              opset_version=opset, do_constant_folding=True, dynamo=False,
                              external_data=True)
        print(f"[export] {name:18s} opset {opset} -> {path}", flush=True)


if __name__ == "__main__":
    main()
