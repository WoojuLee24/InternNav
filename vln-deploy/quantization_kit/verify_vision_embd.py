"""Vision encoder parity: llama.cpp mmproj output vs HF model.visual() for the exact S2 input images.

Images (VLN-CE, first S2 frame of each verify_s2_gguf episode + its 30 deg look-down frame) are prepared exactly as
the HF-parity GGUF policy feeds them: front 640x480 -> PIL 384x384 -> smart_resize BICUBIC 392x392, look-down
640x480 -> 644x476.
  gguf (isaacsim)  runner EMBD for each image                         -> out/vision_gguf_<tag>.npz
  hf   (gd_vln)    processor + model.visual (bf16 and fp32) on (a) the same pre-resized image and
                   (b) the HF policy path (384 image, processor resizes)  -> out/vision_hf.npz
  cmp              per-token cosine / relative L2
  quantization_kit/convenv_x86/bin/python quantization_kit/verify_vision_embd.py gguf --text <text.gguf>
  /usr/bin/python3 quantization_kit/verify_vision_embd.py hf ; python3 quantization_kit/verify_vision_embd.py cmp
"""
import argparse
import base64
import sys
from pathlib import Path

import numpy as np

QK = Path(__file__).resolve().parent
ROOT = QK.parent
INTERNNAV = ROOT.parent
OUT = QK / "verify_s2_gguf_out"


def images():
    sys.path[:0] = [str(QK), str(INTERNNAV)]
    from PIL import Image

    from internnav.model.basemodel.internvla_n1.internvla_n1_policy_llamacpp_hfparity import hf_processor_resize
    from verify_s2_gguf import episodes, load

    out = []
    for ep in episodes():
        f = Image.fromarray(load(ep["fpv"][0])).convert("RGB")
        out.append(("front", f.resize((384, 384)), hf_processor_resize(f.resize((384, 384)))))
        lk = Image.fromarray(load(ep["look"][0])).convert("RGB")
        out.append(("look", lk, hf_processor_resize(lk)))
    return out   # (kind, HF policy input, pre-resized input fed to llama.cpp)


def phase_gguf(a):
    sys.path[:0] = [str(ROOT), str(INTERNNAV)]
    from internnav.model.basemodel.internvla_n1.internvla_n1_policy_llamacpp_hfparity import (
        InternVLAN1LlamaCppPolicyHFParity,
    )

    b = QK / "llama.cpp/build_x86/bin"
    pol = InternVLAN1LlamaCppPolicyHFParity(runner_bin=str(b / "llama-s2-runner"), model_gguf=a.text, mmproj_gguf=a.mmproj,
                                            lib_dir=str(b), resize=384, load_s1=False, embd_cache=64)
    res = {}
    for i, (kind, _, pre) in enumerate(images()):
        p = pol._save(pre)
        r = pol._cmd(f"EMBD\t{p}")
        pol._unlink([p])
        shape, data = r.split("\t")
        nx, ny, ne = map(int, shape.split(","))
        res[f"{i}"] = np.frombuffer(base64.b64decode(data), np.float32).reshape(-1, ne)
        print(f"[gguf] {i} {kind} {pre.size} tokens {nx}x{ny}", flush=True)
    np.savez(OUT / f"vision_gguf_{a.tag}.npz", **res)


def phase_hf(a):
    sys.path[:0] = [str(INTERNNAV / "scripts/sim"), str(INTERNNAV)]
    import torch
    from transformers import AutoProcessor

    from quant import load_internvla

    proc = AutoProcessor.from_pretrained(a.ckpt)
    model = load_internvla(a.ckpt, quant="bf16")
    vis = model.visual
    res = {}
    for dt_name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        vis.to(dt)
        for i, (kind, hfin, pre) in enumerate(images()):
            for path, im in (("pre", pre), ("policy", hfin)):
                x = proc.image_processor(images=[im], return_tensors="pt")
                with torch.no_grad():
                    e = vis(x["pixel_values"].to("cuda:0", dt), grid_thw=x["image_grid_thw"].to("cuda:0"))
                res[f"{dt_name}_{path}_{i}"] = e.float().cpu().numpy()
            print(f"[hf] {dt_name} {i} {kind} grid {x['image_grid_thw'].tolist()}", flush=True)
    np.savez(OUT / "vision_hf.npz", **res)


def phase_cmp(a):
    hf = np.load(OUT / "vision_hf.npz")
    n = len([k for k in hf if k.startswith("bf16_pre_")])

    def stats(x, y):
        c = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
        return float(np.median(c)), float(c.min()), float(np.linalg.norm(x - y) / np.linalg.norm(y))

    rows = []
    for name, get in [("HF bf16 vs HF fp32 (pre)", lambda i: (hf[f"bf16_pre_{i}"], hf[f"fp32_pre_{i}"])),
                      ("HF bf16: policy path vs pre-resized", lambda i: (hf[f"bf16_policy_{i}"], hf[f"bf16_pre_{i}"]))]:
        rows.append((name, [stats(*get(i)) for i in range(n)]))
    for tag in a.tags:
        g = np.load(OUT / f"vision_gguf_{tag}.npz")
        rows.append((f"GGUF {tag} vs HF fp32 (pre)", [stats(g[str(i)], hf[f"fp32_pre_{i}"]) for i in range(n)]))
        rows.append((f"GGUF {tag} vs HF bf16 (pre)", [stats(g[str(i)], hf[f"bf16_pre_{i}"]) for i in range(n)]))
        rows.append((f"GGUF {tag} vs HF bf16 (policy path)", [stats(g[str(i)], hf[f"bf16_policy_{i}"]) for i in range(n)]))
    for name, st in rows:
        fr = [s for i, s in enumerate(st) if i % 2 == 0]
        lk = [s for i, s in enumerate(st) if i % 2 == 1]
        f = lambda v: f"cos med {np.median([s[0] for s in v]):.5f} min {min(s[1] for s in v):.4f} relL2 {np.median([s[2] for s in v]):.4f}"  # noqa: E731
        print(f"{name:42s} front: {f(fr)} | look: {f(lk)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gguf", "hf", "cmp"])
    ap.add_argument("--text", default=str(QK / "out/gguf/InternVLA-S2-text-f16-latent.gguf"))
    ap.add_argument("--mmproj", default=str(QK / "out/gguf/InternVLA-S2-mmproj-f16.gguf"))
    ap.add_argument("--tag", default="f16")
    ap.add_argument("--tags", nargs="+", default=["f16"])
    ap.add_argument("--ckpt", default="/checkpoints/InternVLA-N1-DualVLN")
    a = ap.parse_args()
    if a.phase == "gguf":
        phase_gguf(a)
    elif a.phase == "hf":
        phase_hf(a)
    elif a.phase == "cmp":
        phase_cmp(a)
    else:
        assert False, a.phase


if __name__ == "__main__":
    main()
