"""InternVLA-N1 loader with optional bitsandbytes quantization of the LLM decoder only.

Only `model.layers.*` (the Qwen2.5 text decoder, ~7B of the weights) is quantized. These stay
bf16: the vision tower, lm_head, latent queries / embeddings, and the whole System-1
(traj_dit / NavDP / DepthAnything memory encoder). `generate_latents()` reads HF hidden
states directly, so latent extraction needs no extra code (unlike the GGUF path in vln-deploy).
"""
import os

import torch

# Substring match against full module names (transformers' bnb replacement checks
# `key in current_key_name_str`). Covers both DualVLN (nextdit) and w-NavDP checkpoints.
SKIP_MODULES = [
    "visual",
    "lm_head",
    "traj_dit",
    "action_encoder",
    "action_decoder",
    "cond_projector",
    "rgb_model",
    "rgb_resampler",
    "memory_encoder",
    "navdp",
]

QUANT_CHOICES = ("bf16", "nf4", "int8", "fp16")


def bnb_config(quant):
    if quant in ("bf16", "fp16"):   # fp16: unquantized, float16 weights/compute (numeric-sensitivity check)
        return None
    from transformers import BitsAndBytesConfig

    if quant == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=SKIP_MODULES,
        )
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=SKIP_MODULES)
    raise ValueError(f"unknown quant {quant!r}, choose from {QUANT_CHOICES}")


def legacy_config(model_path):
    """Config for the legacy InternVLA-N1 release (now 'InternVLA-N1-wo-dagger'), else None.

    That checkpoint predates the `system1` field (system1=None, navdp='NavDP_Policy_DPT_CriticSum_DAT',
    n_query=16) and InternVLAN1MetaModel cannot build it. Its model.navdp.* tensors have exactly the same keys
    and shapes as InternVLA-N1-w-NavDP (only latent_queries differs: 16 vs 4 queries, taken from n_query), so
    it is loaded as system1='navdp_async' (RGB-D NavDP S1, same as w-NavDP). Current checkpoints are untouched.
    """
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ModelConfig

    cfg = InternVLAN1ModelConfig.from_pretrained(model_path)
    if getattr(cfg, "system1", None) is None and getattr(cfg, "navdp", None):
        cfg.system1 = "navdp_async"
        print(f"[quant] legacy checkpoint {model_path}: system1=None, navdp={cfg.navdp} -> system1='navdp_async' "
              f"(n_query={cfg.n_query})", flush=True)
        return cfg
    return None


def load_internvla(model_path, device="cuda:0", quant="bf16", attn_implementation="flash_attention_2"):
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

    kwargs = dict(
        torch_dtype=torch.float16 if quant == "fp16" else torch.bfloat16,
        # VLN_HF_ATTN (e.g. sdpa / eager) overrides the attention kernel; unset = flash_attention_2 as before
        attn_implementation=os.environ.get("VLN_HF_ATTN", attn_implementation),
        device_map={"": device},
    )
    cfg_legacy = legacy_config(model_path)
    if cfg_legacy is not None:
        kwargs["config"] = cfg_legacy
    cfg = bnb_config(quant)
    if cfg is not None:
        kwargs["quantization_config"] = cfg
    model = InternVLAN1ForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    if cfg is None:
        model.to(device)  # bnb models must not be moved
    return model


def describe(model):
    """Count parameters per storage dtype (sanity check that only the decoder got quantized)."""
    stats = {}
    for name, p in model.named_parameters():
        key = str(p.dtype)
        n, names = stats.get(key, (0, set()))
        stats[key] = (n + p.numel(), names | {".".join(name.split(".")[:2])})
    return {k: (n, sorted(v)[:12]) for k, (n, v) in stats.items()}
