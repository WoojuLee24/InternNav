#!/usr/bin/env bash
# Build a FULLY SELF-CONTAINED deployment bundle: every runtime artifact is COPIED in
# (code, python venv, llama.cpp incl. built runner, GGUF models, TRT engines, S1 weights)
# so the bundle runs with zero references back to this development tree.
#
#   deploy/package.sh [TARGET] [--full-gguf] [--full-ckpt]
#
#   TARGET        bundle directory (default: ~/vln-deploy)
#   --full-gguf   also copy the f16-latent reference GGUF (+15 GB; default ships Q4 only)
#   --full-ckpt   copy the full DualVLN checkpoint shards (16 GB) instead of extracting
#                 the ~1-2 GB System-1-only safetensors (S2 runs from GGUF, so the S2
#                 safetensors weights are dead weight on-device)
#
# Remaining SYSTEM requirements on the target Jetson (not bundle-able):
#   JetPack (CUDA + TensorRT: /usr/lib/python3.12/dist-packages + libnvinfer),
#   /usr/bin/python3.12 (the venv symlinks to it), a V4L2 camera (optional).
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # repo root
TARGET="${HOME}/vln-deploy"
FULL_GGUF=0; FULL_CKPT=0
for a in "$@"; do
    case "$a" in
        --full-gguf) FULL_GGUF=1 ;;
        --full-ckpt) FULL_CKPT=1 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) TARGET="$a" ;;
    esac
done
TARGET="$(mkdir -p "$TARGET" && cd "$TARGET" && pwd)"
for d in internnav quantization_kit deploy checkpoints; do
    case "$TARGET/" in "$SRC/$d/"*) echo "ERROR: TARGET may not live inside $SRC/$d"; exit 1 ;; esac
done
QK="$SRC/quantization_kit"
PYSRC="$QK/convenv/bin/python"
step() { echo; echo "==> $*"; }
RS="rsync -a --info=stats1 --exclude __pycache__"

step "[1/8] code: internnav/ + deploy/ + quantization_kit python"
$RS "$SRC/internnav/" "$TARGET/internnav/"
$RS --exclude logs --exclude run "$SRC/deploy/" "$TARGET/deploy/"
mkdir -p "$TARGET/quantization_kit"
$RS "$QK/async_zmq/" "$TARGET/quantization_kit/async_zmq/"
rsync -a "$QK"/*.py "$QK/run.sh" "$QK/RUN.md" "$TARGET/quantization_kit/" 2>/dev/null || \
    rsync -a "$QK"/*.py "$QK/run.sh" "$TARGET/quantization_kit/"
mkdir -p "$TARGET/quantization_kit/dbg"
rsync -a "$QK/dbg/test448.png" "$TARGET/quantization_kit/dbg/"
rsync -a "$SRC/mps.sh" "$TARGET/"

step "[2/8] python venv (convenv, ~5.4 GB — symlinks point only at the system python)"
$RS "$QK/convenv/" "$TARGET/quantization_kit/convenv/"

step "[3/8] llama.cpp — the WHOLE folder (source + built runners), minus VCS metadata"
$RS --exclude .git "$QK/llama.cpp/" "$TARGET/quantization_kit/llama.cpp/"

step "[4/8] GGUF models (Q4_K_M-latent + mmproj$([ $FULL_GGUF = 1 ] && echo ' + f16-latent'))"
mkdir -p "$TARGET/quantization_kit/gguf"
rsync -a --info=stats1 "$QK/gguf/InternVLA-S2-text-Q4_K_M-latent.gguf" \
         "$QK/gguf/InternVLA-S2-mmproj-f16.gguf" "$TARGET/quantization_kit/gguf/"
[ $FULL_GGUF = 1 ] && rsync -a --info=stats1 "$QK/gguf/InternVLA-S2-text-f16-latent.gguf" \
         "$TARGET/quantization_kit/gguf/"

step "[5/8] TensorRT engines (native fp16) + onnx/ (for engine rebuild on a fresh box)"
mkdir -p "$TARGET/quantization_kit/trt_engines"
rsync -a --info=stats1 "$QK/trt_engines/"*_native_fp16_*.engine "$TARGET/quantization_kit/trt_engines/"
$RS "$QK/onnx/" "$TARGET/quantization_kit/onnx/"

step "[6/8] checkpoints: DepthAnythingV2 + DualVLN ($([ $FULL_CKPT = 1 ] && echo 'FULL 16 GB shards' || echo 'System-1-only extract'))"
CKPT_SRC="$SRC/checkpoints/InternVLA-N1-DualVLN"
CKPT_DST="$TARGET/checkpoints/InternVLA-N1-DualVLN"
mkdir -p "$CKPT_DST"
rsync -a "$SRC/checkpoints/depth_anything_v2_metric_hypersim_vits.pth" "$TARGET/checkpoints/"
# small sidecar files (config/tokenizer/processor) — everything except weights, logs, index
rsync -a --exclude '*.safetensors' --exclude 'model.safetensors.index.json' --exclude 'eval_*.log' \
    "$CKPT_SRC/" "$CKPT_DST/"
if [ $FULL_CKPT = 1 ]; then
    rsync -a --info=stats1 "$CKPT_SRC/"*.safetensors "$CKPT_SRC/model.safetensors.index.json" "$CKPT_DST/"
elif [ ! -s "$CKPT_DST/model-s1only.safetensors" ]; then
    "$PYSRC" "$SRC/deploy/tools/extract_s1_weights.py" "$CKPT_SRC" "$CKPT_DST/model-s1only.safetensors"
else
    echo "    (model-s1only.safetensors already extracted — skipping)"
fi

step "[7/8] bundle entry point + README + stamp"
cat > "$TARGET/run.sh" <<'EOF'
#!/usr/bin/env bash
# Self-contained InternVLA-N1 deployment — everything runs from inside this folder.
#   ./run.sh start --wait | status | test | demo | camera | prompt "<text>" | stop | clean
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy/run_manager.sh" "$@"
EOF
chmod +x "$TARGET/run.sh" "$TARGET/deploy/run_manager.sh" "$TARGET/deploy/"*/run.sh \
         "$TARGET/quantization_kit/run.sh" 2>/dev/null || true
cat > "$TARGET/BUNDLE.md" <<EOF
# vln-deploy — self-contained InternVLA-N1 field bundle
Built: $(date -Iseconds) on $(hostname) ($(uname -m)) from $SRC

Everything needed at runtime is INSIDE this folder (code, venv, llama.cpp+runner,
GGUF, TRT engines, S1 weights). System deps: JetPack CUDA/TensorRT + /usr/bin/python3.12.

    ./run.sh test --suite protocol     # no-GPU sanity
    ./run.sh start --wait && ./run.sh status
    ./run.sh test --keep-up            # protocol+s1+s2+e2e (+cam if a camera is present)
    ./run.sh camera --view mjpeg       # LIVE: camera -> S2/S1, watch at http://<ip>:8089/
    ./run.sh prompt "go to the door"   # change the instruction live
    ./run.sh stop

Details: deploy/README.md. Checkpoint note: model-s1only.safetensors carries ONLY the
System-1 tensors (S2 runs from GGUF); rebuild with deploy/tools/extract_s1_weights.py.
EOF

step "[8/8] size report"
du -sh "$TARGET"/quantization_kit/convenv "$TARGET"/quantization_kit/llama.cpp \
       "$TARGET"/quantization_kit/gguf "$TARGET"/quantization_kit/trt_engines \
       "$TARGET"/quantization_kit/onnx "$TARGET"/checkpoints "$TARGET"/internnav "$TARGET"/deploy
echo "----------------------------------------"
du -sh "$TARGET"
echo
echo "BUNDLE READY: $TARGET   (verify:  cd $TARGET && ./run.sh test)"
