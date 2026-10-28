#!/bin/bash
# InternVLA-N1 System-2 -> llama.cpp GGUF: text decoder blocks Q8_0, token embedding / output F16,
# mmproj (vision) F16, latent query tokens injected verbatim. Run from vln-deploy/
# (isaacsim image on x86, natively on Thor; PY/BIN default by `uname -m`, env overrides them):
#   bash quantization_kit/tools/build_s2_gguf.sh [ckpt] [out_dir]
# Intermediates (Qwen split, f16 GGUFs, ~45 GB) go to $B and are deleted at the end; only the 2 deploy files stay.
set -euo pipefail
CK=${1:-checkpoints/InternVLA-N1-DualVLN}; OUT=${2:-quantization_kit/out/gguf}
ARCH=$([ "$(uname -m)" = aarch64 ] && echo aarch64 || echo x86)
B=quantization_kit/out/gguf_build; PY=${PY:-quantization_kit/convenv_$ARCH/bin/python}; LQ=quantization_kit/llama.cpp
BIN=${BIN:-$LQ/$([ $ARCH = aarch64 ] && echo build || echo build_x86)/bin}; export LD_LIBRARY_PATH=$BIN:${LD_LIBRARY_PATH:-}
Q8=$OUT/InternVLA-S2-text-Q8_0-latent-embf16.gguf; MM=$OUT/InternVLA-S2-mmproj-f16.gguf
[ -f $Q8 ] && [ -f $MM ] && { echo "[build_s2_gguf] $OUT already built"; ls -la $OUT; exit 0; }
mkdir -p $B $OUT
t0=$(date +%s); step() { echo "[build_s2_gguf] $(( $(date +%s) - t0 ))s  $*"; }
step "1/5 strip InternVLA -> plain Qwen2.5-VL ($CK)"
[ -f $B/qwen/model.safetensors.index.json ] || $PY quantization_kit/tools/strip_to_qwen.py $CK $B/qwen
step "2/5 text model -> GGUF F16"
[ -f $B/text-f16.gguf ] || $PY $LQ/convert_hf_to_gguf.py $B/qwen --outtype f16 --outfile $B/text-f16.gguf
step "3/5 vision projector (mmproj) -> GGUF F16"
[ -f $MM ] || $PY $LQ/convert_hf_to_gguf.py $B/qwen --mmproj --outtype f16 --outfile $MM
step "4/5 inject latent query tokens (verbatim latent_queries rows)"
[ -f $B/text-f16-latent.gguf ] || $PY quantization_kit/tools/gguf_add_latent_tokens.py $B/text-f16.gguf $B/qwen/latent_queries.npy $B/text-f16-latent.gguf
step "5/5 quantize decoder blocks Q8_0 (token_embd / output stay F16)"
$BIN/llama-quantize --token-embedding-type f16 --output-tensor-type f16 $B/text-f16-latent.gguf $B/q8.gguf Q8_0
mv $B/q8.gguf $Q8   # a half-written Q8 never passes the skip check above
rm -rf $B
step "done"; ls -la $OUT
