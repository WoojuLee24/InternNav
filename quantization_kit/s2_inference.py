"""
Standalone S2 (System 2) inference for InternVLA-N1.

This module wraps the Qwen2.5-VL based `InternVLAN1ForCausalLM` planner so that
quantization experiments can be run with the smallest possible surface area:

    from s2_inference import S2Inferencer
    infer = S2Inferencer(model_path="checkpoints/InternVLA-N1-System2")
    result = infer.infer(current_rgb, history_pil_images, "Walk to the kitchen.")
    print(result["llm_output_text"], result["output_pixel"], result["output_action"])

To benchmark a quantized model, change ONLY the `_load_model` / `_load_processor`
methods (or pass `model` / `processor` in the constructor). Everything else --
prompt construction, image preprocessing, output parsing, latency measurement --
stays identical so results are directly comparable to the baseline.

The chat prompt and parsing logic are kept byte-for-byte equivalent to
`internnav/model/basemodel/internvla_n1/internvla_n1_policy.py:s2_step` so that
the baseline (bf16) reference outputs match what the agent produces in
production.
"""

import copy
import itertools
import random
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

# These imports rely on the InternNav repo being on PYTHONPATH (`pip install -e .`).
# The model class is a standard HuggingFace subclass of Qwen2_5_VLForConditionalGeneration,
# so AutoModel.from_pretrained also works once the custom class has been imported once.
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import split_and_clean
from transformers import AutoProcessor, AutoTokenizer


DEFAULT_IMAGE_TOKEN = "<image>"

# Exact prompt template used by the production agent (do not change).
SYSTEM_PROMPT_TEMPLATE = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? "
    "Please output the next waypoint's coordinates in the image. "
    "Please output STOP when you have successfully completed the task."
)
# Conjunction list mirrors habitat_vln_evaluator.py; use random.choice at inference time.
CONJUNCTIONS = [
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
]
CONJUNCTION = CONJUNCTIONS[0]  # kept for backward-compat; infer() uses CONJUNCTIONS

ACTIONS2IDX = OrderedDict({
    "STOP": [0],
    "↑": [1],
    "←": [2],
    "→": [3],
    "↓": [5],
})


@dataclass
class S2Result:
    """Structured S2 output. One of `output_pixel` or `output_action` is set."""
    llm_output_text: str = ""
    output_pixel: Optional[List[int]] = None       # [y, x] in image coordinates
    output_action: Optional[List[int]] = None      # discrete action sequence
    latency_s: float = 0.0
    new_token_count: int = 0
    input_token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "llm_output_text": self.llm_output_text,
            "output_pixel": list(self.output_pixel) if self.output_pixel is not None else None,
            "output_action": list(self.output_action) if self.output_action is not None else None,
            "latency_s": float(self.latency_s),
            "new_token_count": int(self.new_token_count),
            "input_token_count": int(self.input_token_count),
        }


class S2Inferencer:
    def __init__(
        self,
        model_path: str = "checkpoints/InternVLA-N1-System2",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        num_history: int = 8,
        resize_w: int = 384,
        resize_h: int = 384,
        max_new_tokens: int = 128,
        model=None,
        processor=None,
        tokenizer=None,
    ):
        """
        Args:
            model_path: HuggingFace checkpoint path. The bf16 baseline lives at
                `checkpoints/InternVLA-N1-System2`.
            device: torch device string, e.g. "cuda:0".
            torch_dtype: dtype for `from_pretrained`. Ignored when `model` is given.
            attn_implementation: passed to `from_pretrained`. If your environment
                lacks flash-attention, pass "sdpa" or "eager".
            num_history: number of historical frames included in the prompt.
                Must match how the dataset samples were prepared (default 8).
            resize_w / resize_h: target image size before tokenization (default 384).
            max_new_tokens: generation cap. Default 1024 matches the production agent (habitat_s2_cfg.py).
            model / processor / tokenizer: pass pre-loaded objects to swap in a
                quantized model without touching the rest of this class.
        """
        self.device = device
        self.num_history = num_history
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.max_new_tokens = max_new_tokens

        if model is None:
            self.model = self._load_model(model_path, torch_dtype, attn_implementation, device)
        else:
            self.model = model

        if processor is None:
            self.processor = self._load_processor(model_path)
        else:
            self.processor = processor

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        else:
            self.tokenizer = tokenizer

        self.processor.tokenizer = self.tokenizer
        self.processor.tokenizer.padding_side = "left"

        self.model.eval()

    # ------------------------------------------------------------------
    # Loading helpers (override these to swap in a quantized backbone).
    # ------------------------------------------------------------------
    @staticmethod
    def _load_model(model_path, torch_dtype, attn_implementation, device):
        return InternVLAN1ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map={"": device},
        )

    @staticmethod
    def _load_processor(model_path):
        return AutoProcessor.from_pretrained(model_path)

    # ------------------------------------------------------------------
    # Inference.
    # ------------------------------------------------------------------
    def _preprocess_image(self, rgb_or_pil) -> Image.Image:
        if isinstance(rgb_or_pil, np.ndarray):
            img = Image.fromarray(rgb_or_pil).convert("RGB")
        elif isinstance(rgb_or_pil, Image.Image):
            img = rgb_or_pil.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(rgb_or_pil)}")
        return img.resize((self.resize_w, self.resize_h))

    def _build_prompt(self, instruction: str, n_history: int) -> str:
        # Mirrors policy.s2_step: first turn carries instruction + history placeholders + current.
        sources_value = SYSTEM_PROMPT_TEMPLATE.replace("<instruction>.", instruction)
        if n_history > 0:
            placeholder = (DEFAULT_IMAGE_TOKEN + "\n") * n_history
            sources_value += f" These are your historical observations: {placeholder}."
        sources_value += f" {CONJUNCTION}{DEFAULT_IMAGE_TOKEN}."
        return sources_value

    def _make_conversation(self, prompt: str, input_images: List[Image.Image]) -> list:
        parts = split_and_clean(prompt)
        content = []
        img_id = 0
        for p in parts:
            if p == DEFAULT_IMAGE_TOKEN:
                content.append({"type": "image", "image": input_images[img_id]})
                img_id += 1
            else:
                content.append({"type": "text", "text": p})
        assert img_id == len(input_images), (
            f"prompt has {img_id} image slots but {len(input_images)} were provided"
        )
        return [{"role": "user", "content": content}]

    @staticmethod
    def parse_actions(output_text: str) -> List[int]:
        regex = re.compile("|".join(re.escape(a) for a in ACTIONS2IDX))
        matches = regex.findall(output_text)
        return list(itertools.chain.from_iterable(ACTIONS2IDX[m] for m in matches))

    def _run_generate(self, inputs, measure_cuda_sync: bool):
        """Run model.generate; return (output_ids, new_tokens, text_out, latency, input_len)."""
        if measure_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            past_key_values=None,
            return_dict_in_generate=True,
        )
        if measure_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - t0

        output_ids = out.sequences
        input_len = inputs.input_ids.shape[1]
        new_tokens = output_ids[0][input_len:]
        text_out = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return output_ids, new_tokens, text_out, latency, input_len

    @staticmethod
    def _parse_pixel(text_out: str):
        """Return [y, x] if text contains ≥2 integers, else None."""
        coord = [int(c) for c in re.findall(r"\d+", text_out)]
        return [int(coord[1]), int(coord[0])] if len(coord) >= 2 else None

    @torch.no_grad()
    def infer(
        self,
        current_rgb,
        history_images: Optional[List] = None,
        instruction: str = "",
        measure_cuda_sync: bool = True,
        lookdown_image=None,
    ) -> S2Result:
        """Two-step S2 inference matching habitat_vln_evaluator.py behaviour.

        Step 1 — normal forward pass (history + current frame).
        Step 2 — triggered when step 1 outputs the look-down action ↓ (action 5):
          append look-down frame, continue conversation as multi-turn
          [user, assistant(↓), user(look-down image)], re-run generate.
          When lookdown_image is None (VLN-PE), current frame is reused as proxy.
          When lookdown_image is provided (VLN-CE), the actual look-down frame is used.

        Args:
            current_rgb: np.ndarray (H,W,3) uint8, OR PIL.Image.
            history_images: list of frames, length ≤ num_history.
            instruction: natural-language navigation instruction.
            measure_cuda_sync: sync CUDA for accurate latency measurement.
            lookdown_image: optional actual look-down frame (VLN-CE). If None,
                current_rgb is reused as proxy (VLN-PE behaviour).

        Returns:
            S2Result.  latency_s and token counts cover both steps when step 2 ran.
        """
        history_images = list(history_images) if history_images is not None else []

        cur_pil = self._preprocess_image(current_rgb)
        history_pil = [self._preprocess_image(h) for h in history_images]
        input_images = history_pil + [cur_pil]

        # ── Step 1 ───────────────────────────────────────────────────────────
        conjunction = random.choice(CONJUNCTIONS)
        prompt = self._build_prompt(instruction, n_history=len(history_pil))
        # replace fixed CONJUNCTION in _build_prompt with the randomly chosen one
        prompt = prompt.replace(CONJUNCTION, conjunction, 1)
        conversation = self._make_conversation(prompt, input_images)

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.device)

        _, new_tokens_1, text_out_1, latency_1, input_len_1 = self._run_generate(
            inputs, measure_cuda_sync
        )
        print(f"[s2 step1] {text_out_1!r}")

        # ── Step 2: look-down iteration ──────────────────────────────────────
        action_seq_1 = self.parse_actions(text_out_1)
        if not bool(re.search(r"\d", text_out_1)) and 5 in action_seq_1:
            lookdown_pil = self._preprocess_image(lookdown_image) if lookdown_image is not None else cur_pil
            step2_images = input_images + [lookdown_pil]

            # New user turn: " {conjunction}<image>." using the appended (last) image.
            # Mirrors evaluator: sources[0]["value"] starts as "" then += f" {prompt}."
            step2_conjunction = random.choice(CONJUNCTIONS)
            step2_user_content = []
            for part in split_and_clean(f" {step2_conjunction}{DEFAULT_IMAGE_TOKEN}."):
                if part == DEFAULT_IMAGE_TOKEN:
                    step2_user_content.append({"type": "image", "image": step2_images[-1]})
                else:
                    step2_user_content.append({"type": "text", "text": part})

            step2_messages = [
                conversation[0],  # original user message (instruction + history)
                {"role": "assistant", "content": [{"type": "text", "text": text_out_1}]},
                {"role": "user", "content": step2_user_content},
            ]
            step2_text = self.processor.apply_chat_template(
                step2_messages, tokenize=False, add_generation_prompt=True
            )
            step2_inputs = self.processor(
                text=[step2_text], images=step2_images, return_tensors="pt"
            ).to(self.device)

            _, new_tokens_2, text_out_2, latency_2, input_len_2 = self._run_generate(
                step2_inputs, measure_cuda_sync
            )
            print(f"[s2 step2] {text_out_2!r}")

            result = S2Result(
                llm_output_text=text_out_2,
                latency_s=latency_1 + latency_2,
                new_token_count=int(new_tokens_1.shape[0]) + int(new_tokens_2.shape[0]),
                input_token_count=int(input_len_1) + int(input_len_2),
            )
            if bool(re.search(r"\d", text_out_2)):
                result.output_pixel = self._parse_pixel(text_out_2)
            else:
                result.output_action = self.parse_actions(text_out_2)
            return result

        # ── Single-step result ───────────────────────────────────────────────
        result = S2Result(
            llm_output_text=text_out_1,
            latency_s=latency_1,
            new_token_count=int(new_tokens_1.shape[0]),
            input_token_count=int(input_len_1),
        )
        if bool(re.search(r"\d", text_out_1)):
            result.output_pixel = self._parse_pixel(text_out_1)
        else:
            result.output_action = self.parse_actions(text_out_1)
        return result
