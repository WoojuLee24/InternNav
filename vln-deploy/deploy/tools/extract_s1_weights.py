"""Extract ONLY the System-1 tensors from the DualVLN checkpoint into one safetensors file.

The full InternVLA-N1-DualVLN checkpoint is ~16 GB, but on-device only System-1 loads from
it (System-2 runs from GGUF; `InternVLAN1LlamaCppPolicy(load_s1=False)` never touches it).
Both checkpoint readers keep working on the extracted file because they just glob
`*.safetensors` in the directory:
  * `S1LiteModel.load_s1_weights`  — filters keys `model.<sub>` with <sub> in _S1_SUBMODULES
  * `_detect_traj_dit_ffn_dim`     — reads only safetensors metadata (endswith match)

Usage:
    convenv/bin/python deploy/tools/extract_s1_weights.py <src_ckpt_dir> <dst_file.safetensors>
"""
import glob
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# the single source of truth for what System-1 owns
from quantization_kit.s1_inference import _S1_SUBMODULES  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    from safetensors.torch import load_file, save_file

    sd, per_sub = {}, {}
    for shard in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        for k, v in load_file(shard).items():   # one ~4.7 GB shard in RAM at a time
            if not k.startswith("model."):
                continue
            sub = k[len("model."):].split(".")[0]
            if sub in _S1_SUBMODULES:
                sd[k] = v                        # KEEP the original key (both readers expect it)
                per_sub[sub] = per_sub.get(sub, 0) + 1
        print(f"[extract] scanned {os.path.basename(shard)}  (kept so far: {len(sd)})", flush=True)

    assert sd, f"no S1 tensors found under {src}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    save_file(sd, dst, metadata={"source": os.path.basename(src.rstrip('/')),
                                 "subset": "system-1", "submodules": json.dumps(sorted(per_sub))})
    gb = os.path.getsize(dst) / 1e9
    print(f"[extract] wrote {len(sd)} S1 tensors ({gb:.2f} GB) -> {dst}")
    print(f"[extract] per submodule: {per_sub}")


if __name__ == "__main__":
    main()
