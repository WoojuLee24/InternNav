"""Make every saved checkpoint self-contained.

HF Trainer checkpoints hold model + tokenizer but not preprocessor_config.json (final save
only) nor chat_template.json (never saved), so AutoProcessor.from_pretrained(<checkpoint>)
fails. This callback writes both files (~1.6 KB) into each checkpoint-<step> as it is saved.
"""

import json
import os
from typing import Optional

from transformers import TrainerCallback


def resolve_chat_template(model_name_or_path: str) -> Optional[str]:
    """Chat-template string of the base model (local dir or HF hub id); None if unavailable."""
    try:
        from transformers import AutoProcessor

        return getattr(AutoProcessor.from_pretrained(model_name_or_path), "chat_template", None)
    except Exception:
        return None


def write_chat_template(chat_template: Optional[str], dst_dir: str) -> None:
    """Write chat_template.json in the same format transformers uses ({"chat_template": ...})."""
    if not chat_template:
        return
    with open(os.path.join(dst_dir, "chat_template.json"), "w") as f:
        json.dump({"chat_template": chat_template}, f, indent=4)


class SaveAuxFilesCallback(TrainerCallback):
    """Save preprocessor_config.json + chat_template.json into each checkpoint-<step>."""

    def __init__(self, image_processor, model_name_or_path: str):
        self.image_processor = image_processor  # dataset mutates max/min_pixels on this object
        self.chat_template = resolve_chat_template(model_name_or_path)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            self.image_processor.save_pretrained(ckpt_dir)
            write_chat_template(self.chat_template, ckpt_dir)
