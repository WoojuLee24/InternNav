#!/bin/bash
# Vision-embedding cache check (isaacsim image, from vln-deploy/): same replay as run_verify_s2_gguf.sh
# (HF resolution) with --embd_cache 512; must be bit-identical to the uncached run of the same model.
P=quantization_kit/convenv_x86/bin/python; V=quantization_kit/verify_s2_gguf.py; R=quantization_kit/out/gguf
export PYTHONDONTWRITEBYTECODE=1
for m in q4 f16; do
  T=$R/InternVLA-S2-text-Q4_K_M-latent-embf16.gguf; [ $m = f16 ] && T=$R/InternVLA-S2-text-f16-latent.gguf
  $P $V gguf --hf_parity --resize 384 --tag hfres_gguf_${m}_repro_nocache2 --text $T --mmproj $R/InternVLA-S2-mmproj-f16.gguf
  $P $V gguf --hf_parity --resize 384 --embd_cache 512 --tag hfres_gguf_${m}_repro_cache --text $T --mmproj $R/InternVLA-S2-mmproj-f16.gguf
done
