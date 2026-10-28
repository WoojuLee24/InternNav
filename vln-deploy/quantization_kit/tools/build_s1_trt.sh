#!/bin/bash
# InternVLA-N1 System-1 -> TensorRT FP16: ONNX export (7 sub-networks) + native engines (5, static shapes as the originals).
# Engine names carry the GPU tag (sm120 = RTX 5090, sm110 = Thor). Existing files are reused. Run from vln-deploy/ (isaacsim image on x86, natively on Thor; PY by `uname -m`):
#   bash quantization_kit/tools/build_s1_trt.sh [ckpt] [onnx_dir] [engine_dir]
set -euo pipefail
CK=${1:-checkpoints/InternVLA-N1-DualVLN}; ONNX=${2:-quantization_kit/out/onnx}; ENG=${3:-quantization_kit/out/trt_engines}
PY=${PY:-quantization_kit/convenv_$([ "$(uname -m)" = aarch64 ] && echo aarch64 || echo x86)/bin/python}
mkdir -p $ONNX $ENG
t0=$(date +%s); step() { echo "[build_s1_trt] $(( $(date +%s) - t0 ))s  $*"; }
step "1/2 S1 -> ONNX ($CK -> $ONNX)"
[ "$(ls $ONNX/*.onnx 2>/dev/null | wc -l)" -ge 7 ] || $PY deploy/tools/export_s1_onnx.py --ckpt $CK --out $ONNX
step "2/2 ONNX -> TensorRT FP16 engines ($ENG)"
$PY - "$ONNX" "$ENG" <<'EOF'
import sys; sys.path.insert(0, ".")
from quantization_kit.trt_s1 import _engine_for, _sm_tag, DENOISE_SHAPES, RGB_SHAPES, RGB_SHAPES_B1, MEMENC_SHAPES, RESAMPLER_SHAPES
onnx_dir, eng_dir = sys.argv[1:]
sm = _sm_tag()
for tag, onnx, shp in [("s1_denoise", "s1_denoise_step.onnx", DENOISE_SHAPES), ("rgb_dinov2", "rgb_model_dinov2.onnx", RGB_SHAPES),
                       ("rgb_dinov2_b1", "rgb_model_dinov2.onnx", RGB_SHAPES_B1), ("memenc", "memory_encoder.onnx", MEMENC_SHAPES),
                       ("resampler", "rgb_resampler.onnx", RESAMPLER_SHAPES)]:
    _engine_for(f"{tag}_native_fp16_{sm}", onnx, shp, onnx_dir, eng_dir)
print("engines ->", sm)
EOF
n=$(ls $ENG/*.engine | wc -l); [ "$n" -ge 5 ] || { echo "[build_s1_trt] expected 5 engines, got $n" >&2; exit 1; }
step "done"; ls -la $ONNX $ENG
