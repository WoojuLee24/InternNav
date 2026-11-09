#!/bin/bash
# Plain Qwen2.5-VL System-2 (노은 S2 등, InternVLA latent 없음) -> llama.cpp GGUF:
# text decoder blocks Q8_0 (token embedding / output F16) + mmproj (vision) F16.
# build_s2_gguf.sh 와 같은 레시피에서 strip(1/5) · latent 주입(4/5)만 뺀 것. Run from vln-deploy/ (isaacsim image, x86):
#   bash quantization_kit/tools/build_qwen_s2_gguf.sh <ckpt> <out_dir> [QTYPE=Q8_0]
# 중간 산출물(text f16 ~15 GB)은 $OUT/_build 에 두고 끝나면 지운다. 남는 파일: text-<QTYPE>.gguf, mmproj-f16.gguf
set -euo pipefail
CK=$1; OUT=$2; QT=${3:-Q8_0}
PY=${PY:-quantization_kit/convenv_x86/bin/python}; LQ=quantization_kit/llama.cpp
BIN=${BIN:-$LQ/build_x86/bin}; export LD_LIBRARY_PATH=$BIN:${LD_LIBRARY_PATH:-}
TXT=$OUT/text-$QT-embf16.gguf; MM=$OUT/mmproj-f16.gguf; B=$OUT/_build
[ -f $TXT ] && [ -f $MM ] && { echo "[build_qwen_s2_gguf] $OUT already built"; ls -la $OUT; exit 0; }
mkdir -p $B
t0=$(date +%s); step() { echo "[build_qwen_s2_gguf] $(( $(date +%s) - t0 ))s  $*"; }
step "1/3 text model -> GGUF F16 ($CK)"
[ -f $B/text-f16.gguf ] || $PY $LQ/convert_hf_to_gguf.py $CK --outtype f16 --outfile $B/text-f16.gguf
step "2/3 vision projector (mmproj) -> GGUF F16"
[ -f $MM ] || $PY $LQ/convert_hf_to_gguf.py $CK --mmproj --outtype f16 --outfile $MM
step "3/3 quantize decoder blocks $QT (token_embd / output stay F16)"
$BIN/llama-quantize --token-embedding-type f16 --output-tensor-type f16 $B/text-f16.gguf $B/q.gguf $QT
mv $B/q.gguf $TXT   # a half-written file never passes the skip check above
echo "$CK" > $OUT/SOURCE_CKPT
rm -rf $B
step "done"; ls -la $OUT
