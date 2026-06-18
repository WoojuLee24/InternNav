"""
Evaluate a (possibly quantized) S2 model against the bf16 reference outputs.

This script measures two things in one pass:
  1. Output fidelity vs. the baseline produced by `generate_reference.py`:
        - text exact-match rate
        - output-type match rate (did the model still produce pixel vs. action?)
        - pixel L2 distance (mean / median / within-10px rate) for pixel outputs
        - action match rate for action outputs
  2. Runtime cost:
        - per-sample latency (mean / p50 / p95) measured around model.generate()
          with torch.cuda.synchronize() boundaries (handled inside S2Inferencer).
        - throughput (samples / second).
        - peak GPU memory allocated (torch.cuda.max_memory_allocated()).
        - steady-state GPU memory allocated after warm-up.

Usage:
    python evaluate.py \
        --model_path checkpoints/InternVLA-N1-System2-quantized \
        --samples_dir quantization_kit/data/samples \
        --reference quantization_kit/data/reference_outputs.jsonl \
        --output_dir quantization_kit/results/my_quant_run \
        --warmup 5

The quantization team typically only needs to change `--model_path`. If their
quantization framework requires a custom load function, override
`S2Inferencer._load_model` or pass `model=...` into the constructor.
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median

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
    return instruction, current, history, meta


def percentile(xs, q):
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def compare(pred: dict, ref: dict):
    """Return per-sample comparison fields (None when not applicable)."""
    out = {
        "text_exact_match": pred["llm_output_text"] == ref["llm_output_text"],
        "output_type_match": None,
        "pixel_l2": None,
        "pixel_within_10": None,
        "action_match": None,
    }
    pred_kind = "pixel" if pred.get("output_pixel") is not None else (
        "action" if pred.get("output_action") is not None else "none"
    )
    ref_kind = "pixel" if ref.get("output_pixel") is not None else (
        "action" if ref.get("output_action") is not None else "none"
    )
    out["output_type_match"] = pred_kind == ref_kind

    if pred_kind == "pixel" and ref_kind == "pixel":
        p = np.asarray(pred["output_pixel"], dtype=np.float64)
        r = np.asarray(ref["output_pixel"], dtype=np.float64)
        d = float(np.linalg.norm(p - r))
        out["pixel_l2"] = d
        out["pixel_within_10"] = d <= 10.0
    elif pred_kind == "action" and ref_kind == "action":
        out["action_match"] = pred["output_action"] == ref["output_action"]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--samples_dir", default="quantization_kit/data/samples")
    ap.add_argument("--reference", default="quantization_kit/data/reference_outputs.jsonl")
    ap.add_argument("--output_dir", default="quantization_kit/results/run")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn_implementation", default="flash_attention_2",
                    help='Pass "sdpa" if flash-attention is not installed.')
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--limit", type=int, default=-1,
                    help="Only run on the first N samples (dry-run).")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    samples_dir = Path(args.samples_dir)
    sample_dirs = sorted([p for p in samples_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    if args.limit > 0:
        sample_dirs = sample_dirs[: args.limit]

    # Load reference outputs.
    ref_map = {}
    with open(args.reference) as f:
        for line in f:
            row = json.loads(line)
            ref_map[row["sample_id"]] = row
    missing = [p.name for p in sample_dirs if p.name not in ref_map]
    if missing:
        print(f"[warn] {len(missing)} samples missing from reference (first 3: {missing[:3]})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"

    print(f"[eval] loading model {args.model_path} (dtype={args.dtype}) ...")
    infer = S2Inferencer(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )

    # Warm-up: GPU memory and JIT-style first-pass costs should not pollute metrics.
    if args.warmup > 0:
        print(f"[eval] warm-up on {min(args.warmup, len(sample_dirs))} samples ...")
        for sd in sample_dirs[: args.warmup]:
            instruction, current, history, _ = load_sample(sd)
            _ = infer.infer(current, history, instruction)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    mem_steady_mb = None
    if torch.cuda.is_available():
        mem_steady_mb = torch.cuda.memory_allocated() / 1024**2

    latencies = []
    text_matches = []
    type_matches = []
    pixel_l2s = []
    pixel_within10s = []
    action_matches = []
    missing_in_ref = 0

    t_wall_start = time.time()
    with open(pred_path, "w") as f:
        for i, sd in enumerate(sample_dirs):
            instruction, current, history, meta = load_sample(sd)
            result = infer.infer(current, history, instruction)
            pred_row = {
                "sample_id": meta["sample_id"],
                "scene": meta.get("scene"),
                "episode_index": meta.get("episode_index"),
                "current_step": meta.get("current_step"),
                **result.to_dict(),
            }

            ref = ref_map.get(meta["sample_id"])
            if ref is None:
                missing_in_ref += 1
                cmp_row = {"text_exact_match": None, "output_type_match": None,
                           "pixel_l2": None, "pixel_within_10": None, "action_match": None}
            else:
                cmp_row = compare(pred_row, ref)
                text_matches.append(bool(cmp_row["text_exact_match"]))
                type_matches.append(bool(cmp_row["output_type_match"]))
                if cmp_row["pixel_l2"] is not None:
                    pixel_l2s.append(cmp_row["pixel_l2"])
                    pixel_within10s.append(bool(cmp_row["pixel_within_10"]))
                if cmp_row["action_match"] is not None:
                    action_matches.append(bool(cmp_row["action_match"]))

            pred_row.update({"compare": cmp_row})
            f.write(json.dumps(pred_row) + "\n")
            f.flush()
            latencies.append(result.latency_s)

            if (i + 1) % 25 == 0 or i == 0:
                cur_mean = mean(latencies)
                print(f"  [{i+1}/{len(sample_dirs)}] {pred_row['sample_id']} "
                      f"text_match={cmp_row['text_exact_match']} "
                      f"latency={result.latency_s*1000:.1f}ms (mean {cur_mean*1000:.1f}ms)")

    wall_s = time.time() - t_wall_start

    mem_peak_mb = None
    if torch.cuda.is_available():
        mem_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    def rate(xs):
        return float(sum(xs) / len(xs)) if xs else None

    metrics = {
        "n_samples_eval": len(sample_dirs),
        "n_samples_missing_in_reference": missing_in_ref,
        "wall_time_s": wall_s,
        "throughput_samples_per_sec": (len(sample_dirs) / wall_s) if wall_s > 0 else None,
        "latency_ms_mean": (mean(latencies) * 1000) if latencies else None,
        "latency_ms_median": (median(latencies) * 1000) if latencies else None,
        "latency_ms_p95": (percentile(latencies, 95) * 1000) if latencies else None,
        "latency_ms_p99": (percentile(latencies, 99) * 1000) if latencies else None,
        "gpu_mem_steady_mb": mem_steady_mb,
        "gpu_mem_peak_mb": mem_peak_mb,
        "text_exact_match_rate": rate(text_matches),
        "output_type_match_rate": rate(type_matches),
        "pixel_l2_mean": (mean(pixel_l2s) if pixel_l2s else None),
        "pixel_l2_median": (median(pixel_l2s) if pixel_l2s else None),
        "pixel_within_10px_rate": rate(pixel_within10s),
        "action_match_rate": rate(action_matches),
        "n_pixel_pairs": len(pixel_l2s),
        "n_action_pairs": len(action_matches),
        "model_path": args.model_path,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "warmup": args.warmup,
    }

    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n[eval] wrote {pred_path}")
    print(f"[eval] wrote {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
