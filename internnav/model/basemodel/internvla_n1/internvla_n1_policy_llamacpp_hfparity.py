"""llama.cpp System-2 policy with exactly the HF model's image resolution (InternVLAN1LlamaCppPolicy subclass).

The HF path (internvla_n1_agent_realworld.py + Qwen2VLImageProcessor) sees:
  * planning / history frames: PIL resize to resize x resize (384), then processor smart_resize to a multiple of 28
    (384 -> 392x392, BICUBIC, preprocessor_config.json: resample=3, min/max_pixels 3136 / 12845056)
  * look-down frame: NOT resized by the agent -> processor smart_resize of the camera frame (640x480 -> 644x476)
Here every image is brought to exactly that size with the same PIL BICUBIC resampling before it is handed to
llama-s2-runner; with MTMD_QWEN25VL_HF_ALIGN=1 (llama.cpp tools/mtmd patch) clip.cpp aligns to 28 px like HF, so the
already-aligned image is copied unchanged (no second resample) and edge attention windows are masked as in HF.
S2_LATENT_WITH_EOS=1 (llama-s2-runner patch) decodes the end-of-turn token before the latent tokens, as HF
generate_latents does (it appends the latent queries to generate()'s sequences, which end with <|im_end|>).
S2_HF_CHAT=1 (runner patch) adds the Qwen2-VL default system message HF's chat template inserts and keeps each
answer's <|im_end|> in the KV (HF look-down turn: "↓<|im_end|>\n<|im_start|>user ...").
repeat_last_n = n_ctx turns the repetition penalty (1.05, HF generation_config) on over the whole context.
_format_prompt applies HF's split_and_clean() so the text around the images is tokenized like HF (no space before
an image, no newlines between history images, no leading space in the look-down turn).
The parent's 112-aligned path (resize 336/448) is untouched.
"""
import math
import os

import numpy as np
from PIL import Image

from internnav.model.basemodel.internvla_n1.internvla_n1_policy_llamacpp import MEDIA, InternVLAN1LlamaCppPolicy
from internnav.model.utils.vln_utils import split_and_clean

FACTOR, MIN_PIXELS, MAX_PIXELS = 28, 3136, 12845056   # InternVLA-N1 preprocessor_config.json


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize (verbatim logic)."""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def hf_processor_resize(pil):
    w, h = pil.size
    hb, wb = smart_resize(h, w)
    return pil if (hb, wb) == (h, w) else pil.resize((wb, hb), Image.BICUBIC)


class InternVLAN1LlamaCppPolicyHFParity(InternVLAN1LlamaCppPolicy):
    def __init__(self, *args, resize=384, **kwargs):
        os.environ["MTMD_QWEN25VL_HF_ALIGN"] = "1"      # inherited by the runner process (parent copies os.environ)
        os.environ["S2_LATENT_WITH_EOS"] = "1"          # latent queries after <|im_end|>, as HF generate_latents
        os.environ["S2_HF_CHAT"] = "1"                  # default system message + answer EOG kept in the KV, as HF
        kwargs["lookdown_full_res"] = True             # agent passes the look-down frame unresized
        # HF generate(): repetition_penalty 1.05 over the whole input_ids. The runner's "-1" leaves the penalty off
        # (not expanded to the context size in this llama.cpp build) -> pass the context size explicitly
        kwargs.setdefault("repeat_last_n", kwargs.get("n_ctx", 8192))
        super().__init__(*args, resize=resize, **kwargs)

    def _format_prompt(self, prompt):
        # HF policy: split_and_clean() around <image> (drops newlines, strips each text piece) and the chat template
        # concatenates the pieces without separators -> "...observations:<img><img>...<img>. you can see<img>."
        parts = split_and_clean(prompt.replace(MEDIA, "<image>"))
        return "".join(MEDIA if p == "<image>" else p for p in parts)

    def _resize(self, rgb):
        # agent: PIL resize to (resize, resize); processor: smart_resize to a multiple of 28
        return hf_processor_resize(super()._resize(rgb))

    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        if look_down:   # parent uses the frame as-is (lookdown_full_res) -> apply the processor resize here
            rgb = np.asarray(hf_processor_resize(Image.fromarray(rgb).convert("RGB")))
        return super().s2_step(rgb, depth, pose, instruction, intrinsic, look_down=look_down)
