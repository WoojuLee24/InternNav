#!/bin/bash
# Closed-loop SR / latency / FPS of System-2 variants (HOST script):
#   hf_bf16   : PyTorch HF bf16 (gd_vln container)
#   gguf_orig : vln-deploy original GGUF (text Q4_K_M incl. token_embd Q4_K / output Q6_K, mmproj bf16-as-f16)
#   gguf_repro: reproduced GGUF (decoder blocks Q4_K_M, token_embd/output F16, mmproj F16, latent rows verbatim)
#   gguf_f16  : reproduced GGUF, text F16 (no quantization), mmproj F16
#   gguf_f16_pt: gguf_f16 with S1 = original PyTorch bf16 (S1=hf)
#   hf_bf16_pt: baseline, S2 HF bf16 + S1 original PyTorch bf16 (S1=hf)
#   gguf_q8   : reproduced GGUF, decoder Q8_0, token_embd/output F16, mmproj F16
# S1 = TensorRT FP16 10 step for all, dual_server async, Mode B (gdm ./lpp-sim-vln), clear-area episodes x REPS.
#   scripts/sim/verify/run_s2_sr.sh [reps] [configs...]   (e.g. 4 gguf_q8 gguf_f16 gguf_f16_pt gguf_repro)
set -u
REPS=${1:-2}; shift || true
CFGS=${*:-hf_bf16 gguf_orig gguf_repro}
cd "$(dirname "$0")/../../.."      # InternNav (host path)
GD=internnav-habitat
EPS=scripts/sim/verify/e2e_episodes_etri_indoor_clear.json
for cfg in $CFGS; do
  case $cfg in
    hf_bf16)    S2=hf scripts/sim/run_vln_stack.sh start ;;
    hf_bf16_pt) S1=hf S2=hf scripts/sim/run_vln_stack.sh start ;;
    gguf_orig)  S2=gguf GGUF_TEXT=quantization_kit/gguf/InternVLA-S2-text-Q4_K_M-latent.gguf \
                GGUF_MMPROJ=quantization_kit/gguf/InternVLA-S2-mmproj-f16.gguf scripts/sim/run_vln_stack.sh start ;;
    gguf_repro) S2=gguf GGUF_TEXT=quantization_kit/out/gguf/InternVLA-S2-text-Q4_K_M-latent-embf16.gguf scripts/sim/run_vln_stack.sh start ;;
    gguf_f16)   S2=gguf GGUF_TEXT=quantization_kit/out/gguf/InternVLA-S2-text-f16-latent.gguf scripts/sim/run_vln_stack.sh start ;;
    gguf_f16_pt) S1=hf S2=gguf GGUF_TEXT=quantization_kit/out/gguf/InternVLA-S2-text-f16-latent.gguf scripts/sim/run_vln_stack.sh start ;;
    gguf_q8)    S2=gguf GGUF_TEXT=quantization_kit/out/gguf/InternVLA-S2-text-Q8_0-latent-embf16.gguf scripts/sim/run_vln_stack.sh start ;;
    *) echo "unknown $cfg"; exit 2 ;;
  esac
  docker exec $GD bash -c "cp /ws/src/InternNav/scripts/sim/verify/out/live/stack_s2.log /ws/src/InternNav/scripts/sim/verify/out/live/s2sr_${cfg}_s2.log"
  for r in $(seq 1 $REPS); do
    docker exec $GD bash -c "cd /ws/src/InternNav && source scripts/sim/setup_ros_ws.sh >/dev/null && python3 scripts/sim/verify/run_e2e.py --mode trajectory --tag s2sr_${cfg}_rep$r --episodes $EPS" \
      | grep '^{' | python3 -c "import sys,json;[print('   $cfg',d['id'],'OK' if d.get('success') else 'x','%.1f m'%d['final_dist_to_goal_m']) for d in map(json.loads,sys.stdin)]"
  done
  docker exec $GD bash -c "cp /ws/src/InternNav/scripts/sim/verify/out/live/stack_dual.log /ws/src/InternNav/scripts/sim/verify/out/live/s2sr_${cfg}_dual.log"
done
scripts/sim/run_vln_stack.sh stop
echo "[s2sr] done"
