#!/bin/bash
# GGUF replays for verify_s2_gguf.py with the HF-parity policy (HF resolution, chat format, prompt spacing,
# repetition penalty) and the vision-embedding cache. isaacsim image, from vln-deploy/.
P=quantization_kit/convenv_x86/bin/python; V=quantization_kit/verify_s2_gguf.py; R=quantization_kit/out/gguf
export PYTHONDONTWRITEBYTECODE=1
for m in f16 q8 q4; do
  case $m in
    f16) T=$R/InternVLA-S2-text-f16-latent.gguf ;;
    q8)  T=$R/InternVLA-S2-text-Q8_0-latent-embf16.gguf ;;
    q4)  T=$R/InternVLA-S2-text-Q4_K_M-latent-embf16.gguf ;;
    *) echo "unknown $m"; exit 2 ;;
  esac
  $P $V gguf --hf_parity --resize 384 --embd_cache 512 --tag hfp_gguf_$m --text $T --mmproj $R/InternVLA-S2-mmproj-f16.gguf
done
