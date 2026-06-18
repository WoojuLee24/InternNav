"""
Generate the baseline (bf16) reference outputs that `evaluate.py` compares against.

Run this ONCE when building the kit. The resulting `reference_outputs.jsonl`
is what the quantization team will use as ground truth, so it ships with the kit.

Usage:
    python generate_reference.py \
        --model_path checkpoints/InternVLA-N1-System2 \
        --samples_dir quantization_kit/data/samples \
        --output quantization_kit/data/reference_outputs.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from s2_inference import S2Inferencer


def load_sample(sample_dir: Path):
    instruction = (sample_dir / "instruction.txt").read_text().strip()
    meta = json.loads((sample_dir / "meta.json").read_text())
    current = np.array(Image.open(sample_dir / "current.jpg").convert("RGB"))
    history = []
    i = 0
    while True:
        p = sample_dir / f"history_{i}.jpg"
        if not p.exists():
            break
        history.append(np.array(Image.open(p).convert("RGB")))
        i += 1
    lookdown_path = sample_dir / "lookdown.jpg"
    lookdown = np.array(Image.open(lookdown_path).convert("RGB")) if lookdown_path.exists() else None
    return instruction, current, history, meta, lookdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="checkpoints/InternVLA-N1-System2")
    ap.add_argument("--dataset", choices=["vln-pe", "vln-ce"], default="vln-pe")
    ap.add_argument("--samples_dir", default=None,
                    help="Override samples dir (default: quantization_kit/data/<dataset>/samples)")
    ap.add_argument("--output", default=None,
                    help="Override output path (default: quantization_kit/data/<dataset>/reference_outputs.jsonl)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn_implementation", default="flash_attention_2",
                    help='Pass "sdpa" if flash-attention is not installed.')
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    data_root = Path(f"quantization_kit/data/{args.dataset}")
    samples_dir = Path(args.samples_dir) if args.samples_dir else data_root / "samples"
    out_path = Path(args.output) if args.output else data_root / "reference_outputs.jsonl"
    sample_dirs = sorted([p for p in samples_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    if args.limit > 0:
        sample_dirs = sample_dirs[: args.limit]
    print(f"[reference] {len(sample_dirs)} samples found")

    print(f"[reference] loading model {args.model_path} on {args.device} (bf16) ...")
    infer = S2Inferencer(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    with open(out_path, "w") as f:
        for i, sd in enumerate(sample_dirs):
            instruction, current, history, meta, lookdown = load_sample(sd)
            result = infer.infer(current, history, instruction, lookdown_image=lookdown)
            row = {
                "sample_id": meta["sample_id"],
                "instruction": instruction,
                "scene": meta.get("scene"),
                "episode_index": meta.get("episode_index"),
                "current_step": meta.get("current_step"),
                "look_down": meta.get("look_down", False),
                **result.to_dict(),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or i == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(sample_dirs)}] {row['sample_id']} "
                      f"text={row['llm_output_text'][:40]!r} "
                      f"({elapsed:.1f}s elapsed, {(i+1)/elapsed:.2f} samples/s)")

    print(f"[reference] wrote {out_path}")


if __name__ == "__main__":
    main()
