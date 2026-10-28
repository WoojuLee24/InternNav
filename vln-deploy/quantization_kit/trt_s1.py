"""Native TensorRT FP16 acceleration for S1's diffusion denoise loop, hot-swappable into
S1LiteModel.generate_traj (and thus InternVLAN1LlamaCppPolicy.s1).

Builds/caches a native FP16 TRT engine for the per-step DiT (s1_denoise_step =
action_encoder -> traj_dit -> action_decoder) and runs the FlowMatchEuler sampling loop +
CFG combine in host code (standard diffusion-on-TRT pattern). Conditioning (cond_projector,
rgb_model, memory_encoder, rgb_resampler) stays PyTorch (runs once). Numerically faithful to
the fp32 PyTorch generate_traj up to fp16 rounding.

  attach_trt_denoise(policy.s1)        # monkeypatches policy.s1.generate_traj
self-test (native TRT build + correctness + speed vs PyTorch):
  PYTHONPATH=/usr/lib/python3.12/dist-packages \
    quantization_kit/convenv/bin/python quantization_kit/trt_s1.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.append("/usr/lib/python3.12/dist-packages")  # native tensorrt (python3-libnvinfer); append => venv wins
import tensorrt as trt  # noqa: E402

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from diffusers.utils.torch_utils import randn_tensor  # noqa: E402

_QK = os.path.dirname(os.path.abspath(__file__))
_LOG = trt.Logger(trt.Logger.ERROR)
_T2T = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32, trt.DataType.INT64: torch.int64}
DENOISE_SHAPES = {"latents": (32, 32, 3), "timestep": (64,), "z_latents": (64, 36, 768), "pos_ids": (1, 32)}
RGB_SHAPES = {"img": (2, 3, 224, 224)}      # DINOv2 backbone: memory + current frame (batch-2)
RGB_SHAPES_B1 = {"img": (1, 3, 224, 224)}   # batch-1 variant for the goal-frame feature cache
MEMENC_SHAPES = {"memory": (1, 512, 384)}   # memory_encoder: 2 frames x 256 patches = 512 tokens
RESAMPLER_SHAPES = {"visual": (1, 512, 768)}  # QFormer rgb_resampler: 512 tokens -> 32 query tokens


def _build_engine(onnx_path, shapes, fp16=True):
    b = trt.Builder(_LOG)
    net = b.create_network(0)
    parser = trt.OnnxParser(net, _LOG)
    if not parser.parse_from_file(onnx_path):       # resolves external .onnx.data next to the file
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = b.create_builder_config()
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    prof = b.create_optimization_profile()
    for nm, shp in shapes.items():
        prof.set_shape(nm, shp, shp, shp)           # static engine for the fixed S1 workload
    cfg.add_optimization_profile(prof)
    ser = b.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError("engine build returned None")
    return ser


class _Engine:
    def __init__(self, serialized):
        self.eng = trt.Runtime(_LOG).deserialize_cuda_engine(serialized)
        self.ctx = self.eng.create_execution_context()
        names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.inp = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.out = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    def __call__(self, feeds):
        keep = []
        for n in self.inp:
            t = feeds[n].to(_T2T[self.eng.get_tensor_dtype(n)]).contiguous()
            keep.append(t)
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr())
        outs = {}
        for n in self.out:
            o = torch.empty(tuple(self.ctx.get_tensor_shape(n)),
                            dtype=_T2T[self.eng.get_tensor_dtype(n)], device="cuda")
            outs[n] = o
            self.ctx.set_tensor_address(n, o.data_ptr())
        # enqueue on PyTorch's current stream -> later PyTorch ops are correctly ordered, no per-iter sync
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return outs[self.out[0]]


def _sm_tag():                                              # [fix] Thor/5090 자동 분기: 엔진 파일명에 실제 GPU compute capability 반영
    cc = torch.cuda.get_device_capability()                 # (12, 0) for RTX 5090; (8, 7) for Thor (legacy 이름은 sm110)
    return f"sm{cc[0]}{cc[1]}"


def _engine_for(tag, onnx_name, shapes, onnx_dir, engine_dir):
    ep = os.path.join(engine_dir, f"{tag}.engine")
    if os.path.exists(ep):
        return _Engine(open(ep, "rb").read())
    ser = _build_engine(os.path.join(onnx_dir, onnx_name), shapes)
    open(ep, "wb").write(ser)
    return _Engine(ser)


def attach_trt_denoise(s1, onnx_dir=None, engine_dir=None, trt_rgb=True, trt_cond=False):
    """Monkeypatch s1.generate_traj to run the per-step DiT (and, if trt_rgb, the DINOv2 rgb
    backbone; if trt_cond, the memory_encoder + QFormer resampler) on native TRT FP16 engines.
    Scheduler loop + CFG + small heads stay PyTorch."""
    onnx_dir = onnx_dir or os.path.join(_QK, "onnx")
    engine_dir = engine_dir or os.path.join(_QK, "trt_engines")
    os.makedirs(engine_dir, exist_ok=True)
    sm = _sm_tag()                                          # [fix] Thor/5090 자동 분기
    eng = _engine_for(f"s1_denoise_native_fp16_{sm}", "s1_denoise_step.onnx", DENOISE_SHAPES, onnx_dir, engine_dir)
    rgb_eng = rgb_eng_b1 = None
    if trt_rgb:
        try:
            rgb_eng = _engine_for(f"rgb_dinov2_native_fp16_{sm}", "rgb_model_dinov2.onnx", RGB_SHAPES, onnx_dir, engine_dir)
            # batch-1 engine: lets the goal-frame DINOv2 cache re-encode only the current frame
            rgb_eng_b1 = _engine_for(f"rgb_dinov2_b1_native_fp16_{sm}", "rgb_model_dinov2.onnx", RGB_SHAPES_B1, onnx_dir, engine_dir)
        except Exception as e:
            print(f"[trt_s1] rgb_model TRT skipped ({type(e).__name__}: {str(e)[:80]}); using PyTorch rgb")
    memenc_eng = resampler_eng = None
    if trt_cond:
        try:   # the eager memory_encoder(TransformerEncoder) + QFormer were ~5.4 ms/tick in PyTorch
            memenc_eng = _engine_for(f"memenc_native_fp16_{sm}", "memory_encoder.onnx", MEMENC_SHAPES, onnx_dir, engine_dir)
            resampler_eng = _engine_for(f"resampler_native_fp16_{sm}", "rgb_resampler.onnx", RESAMPLER_SHAPES, onnx_dir, engine_dir)
        except Exception as e:
            print(f"[trt_s1] cond TRT skipped ({type(e).__name__}: {str(e)[:80]}); using PyTorch memory_encoder/QFormer")

    def _rgb_feat(images_dp_norm, dtype, cache_key):
        """DINOv2 features [1, F, N, C] with optional goal-frame (frame 0) reuse (see
        S1LiteModel._encode_rgb). Uses the batch-1 TRT engine for the per-frame path."""
        xf = images_dp_norm.flatten(0, 1).to(torch.float32).contiguous()   # [F,3,224,224]
        F = images_dp_norm.shape[1]

        def encode(t):
            if rgb_eng_b1 is not None:
                return rgb_eng_b1({"img": t.contiguous()}).to(dtype)
            s1.rgb_model.to(dtype)
            return s1.rgb_model.get_intermediate_layers(t.to(dtype))[0]

        if not s1._rgb_cache_enabled or cache_key is None or F < 2:
            if rgb_eng is not None:
                return rgb_eng({"img": xf}).to(dtype).unflatten(0, (1, F))
            s1.rgb_model.to(dtype)
            return s1.rgb_model.get_intermediate_layers(xf.to(dtype))[0].unflatten(0, (1, F))
        if s1._rgb_cache_key == cache_key and s1._rgb_cache_feat is not None:
            goal_feat = s1._rgb_cache_feat
        else:
            goal_feat = encode(xf[:1])
            s1._rgb_cache_feat, s1._rgb_cache_key = goal_feat, cache_key
            s1._rgb_cache_misses += 1
        cur_feat = encode(xf[1:])
        return torch.cat([goal_feat, cur_feat], dim=0).unflatten(0, (1, F))

    @torch.no_grad()
    def trt_generate_traj(traj_latents, images_dp, depths_dp=None, predict_step_nums=32,
                          guidance_scale=1.0, num_inference_steps=10, num_sample_trajs=32,
                          rgb_cache_key=None):
        scheduler = FlowMatchEulerDiscreteScheduler()
        device, dtype = traj_latents.device, traj_latents.dtype
        # ---- conditioning (runs once) ----
        tl = s1.cond_projector(traj_latents)
        images_dp = images_dp.permute(0, 1, 4, 2, 3)
        images_dp_norm = (images_dp - s1._resnet_mean) / s1._resnet_std
        feat = _rgb_feat(images_dp_norm, dtype, rgb_cache_key)
        ff = feat.flatten(1, 2)                                         # [1, 512, 384]
        if memenc_eng is not None:
            mem = memenc_eng({"memory": ff.to(torch.float32).contiguous()}).to(dtype)
        else:
            mem = s1.memory_encoder(ff)
        mem = torch.cat([ff, mem], dim=-1)                             # [1, 512, 768]
        if resampler_eng is not None:
            resampled = resampler_eng({"visual": mem.to(torch.float32).contiguous()}).to(dtype)
        else:
            resampled = s1.rgb_resampler(mem)
        hidden = torch.cat([resampled, tl], dim=1)
        hin = torch.cat([torch.zeros_like(hidden), hidden], 0)
        bs = traj_latents.shape[0]
        latents = randn_tensor((bs * num_sample_trajs, predict_step_nums, 3), device=device, dtype=dtype)
        scheduler.set_timesteps(num_inference_steps, sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps))
        hin = hin.repeat_interleave(num_sample_trajs, dim=0)                 # [2*bs*32, 36, 768]
        pos_ids = torch.arange(predict_step_nums).reshape(1, -1).to(device)  # [1, 32]
        # ---- TRT denoise loop ----
        for t in scheduler.timesteps:
            ts = t.unsqueeze(0).expand(hin.shape[0]).to(device, torch.long)  # [2*bs*32]
            noise = eng({"latents": latents, "timestep": ts, "z_latents": hin, "pos_ids": pos_ids}).to(dtype)
            nu, nc = noise.chunk(2)
            noise = nu + guidance_scale * (nc - nu)
            latents = scheduler.step(noise, t, latents).prev_sample
        return latents

    s1._trt_denoise = eng
    s1._trt_rgb = rgb_eng
    s1.generate_traj = trt_generate_traj
    return s1


def _selftest():
    sys.path[:0] = [os.path.dirname(_QK), _QK]
    from quantization_kit.s1_inference import S1LiteModel, prep_images, traj_to_actions
    dev = "cuda:0"
    g = torch.Generator(device="cpu").manual_seed(0)
    latent = torch.randn(1, 4, 3584, generator=g).to(dev)
    rgb = (torch.rand(480, 640, 3, generator=g).numpy() * 255).astype(np.uint8)
    depth = np.zeros((480, 640), np.float32)
    rgbs, depths = prep_images(rgb, depth, rgb, depth, dev, torch.float32)

    m = S1LiteModel("checkpoints/InternVLA-N1-DualVLN").load_s1_weights("checkpoints/InternVLA-N1-DualVLN").to(dev).eval()

    def timed(fn, n=20, w=3):
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    pt_ms = timed(lambda: m.generate_traj(traj_latents=latent, images_dp=rgbs, depths_dp=depths))
    torch.manual_seed(0)  # same initial noise -> clean fp16-vs-fp32 comparison
    traj_pt = m.generate_traj(traj_latents=latent, images_dp=rgbs, depths_dp=depths)

    print("building native TRT FP16 denoise engine ...")
    attach_trt_denoise(m)
    trt_ms = timed(lambda: m.generate_traj(traj_latents=latent, images_dp=rgbs, depths_dp=depths))
    torch.manual_seed(0)  # identical seed -> isolates fp16 rounding from RNG
    traj_trt = m.generate_traj(traj_latents=latent, images_dp=rgbs, depths_dp=depths)

    a_pt = traj_to_actions(traj_pt.clone())[:4]
    a_trt = traj_to_actions(traj_trt.clone())[:4]
    print("\n================ native TRT S1 self-test (Jetson Thor) ================")
    print(f"  generate_traj  PyTorch fp32 : {pt_ms:7.1f} ms")
    print(f"  generate_traj  TRT FP16 loop: {trt_ms:7.1f} ms   ({pt_ms/trt_ms:.1f}x)")
    print(f"  trajectory mean |Δ| (fp16 vs fp32): {(traj_pt-traj_trt).abs().mean().item():.4f}")
    print(f"  actions PyTorch={a_pt}  TRT={a_trt}  {'MATCH' if a_pt==a_trt else 'differ (fp16 rounding)'}")


if __name__ == "__main__":
    _selftest()
