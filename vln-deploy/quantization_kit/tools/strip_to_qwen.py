"""InternVLA-N1 checkpoint -> plain Qwen2.5-VL HF checkpoint (input of llama.cpp convert_hf_to_gguf.py).

Keeps the System-2 tensors only (LLM: model.embed_tokens / model.layers / model.norm / lm_head, vision: visual.*),
drops System-1 (traj_dit, rgb_model, memory_encoder, rgb_resampler, cond_projector, action_*, navdp) and saves
model.latent_queries [1, n_query, 3584] separately (latent_queries.npy, fp32 of the bf16 values) so that
gguf_add_latent_tokens.py can inject them verbatim as the <|latent_i|> token embeddings. config.json is rewritten
to Qwen2_5_VLForConditionalGeneration; tokenizer / processor / chat template files are copied unchanged.

  quantization_kit/convenv_x86/bin/python quantization_kit/tools/strip_to_qwen.py <InternVLA ckpt> <out dir>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

KEEP = ("model.embed_tokens.", "model.layers.", "model.norm.", "lm_head.", "visual.")
DROP_CFG = ("system1", "n_query", "model_cfg", "navdp", "navdp_version", "auto_map")
COPY = ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "added_tokens.json",
        "special_tokens_map.json", "chat_template.json", "chat_template.jinja", "preprocessor_config.json",
        "generation_config.json")


def main(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    for k in DROP_CFG:
        cfg.pop(k, None)
    cfg["architectures"] = ["Qwen2_5_VLForConditionalGeneration"]
    cfg["model_type"] = "qwen2_5_vl"
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in COPY:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    new_map, kept, dropped, total = {}, 0, [], 0
    for i, shard in enumerate(shards):
        out = {}
        with safe_open(src / shard, "pt") as f:
            for k in f.keys():
                if k == "model.latent_queries":
                    lq = f.get_tensor(k).float().numpy()
                    np.save(dst / "latent_queries.npy", lq)
                    print(f"[strip] latent_queries {tuple(lq.shape)} -> {dst / 'latent_queries.npy'}")
                elif k.startswith(KEEP):
                    out[k] = f.get_tensor(k).contiguous()
                else:
                    dropped.append(k)
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(out, dst / name, metadata={"format": "pt"})
        for k, v in out.items():
            new_map[k] = name
            total += v.numel() * v.element_size()
        kept += len(out)
        print(f"[strip] {shard}: kept {len(out)} -> {name}", flush=True)
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": new_map}, indent=1))
    prefixes = sorted({".".join(k.split(".")[:2]) for k in dropped})
    print(f"[strip] kept {kept} tensors, dropped {len(dropped)} (System-1: {prefixes})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
