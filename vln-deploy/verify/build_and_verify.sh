#!/usr/bin/env bash
# docs/command.md 2장(경량화) + 3장(검증)을 한 번에. 같은 스크립트가 두 장치에서 돈다:
#   x86 (5090) : 호스트에서 실행. [isaacsim] / [gd_vln] 단계는 docker run 으로 띄운다 (HF 기준값은 gd_vln 에서만)
#   aarch64 (Thor): 장치에서 바로 실행. HF 기준값은 5090 에서 만든 파일을 복사해 쓴다 (command.md 4장)
#
#   vln-deploy/verify/build_and_verify.sh [--stages build,dataset,rgbd,sim] [--tag TAG] [--frames DIR] [--rgbd-ref RUN]
#                                         [--sim-reps N] [--ckpt PATH]
#   --stages   build   : S1 ONNX -> TRT FP16 (build_s1_trt.sh), S2 GGUF Q8_0 + mmproj (build_s2_gguf.sh)
#              dataset : open-loop 데이터셋 검증 (S2 replay, S2+S1) + 합격 판정 (명령 일치·ADE<5cm >= 0.99)
#              rgbd    : RGB-D 프레임 폴더(--frames)로 S2·S1·S2+S1 실행 + 이상 여부 확인, --rgbd-ref 가 있으면 그 실행과 비교
#              sim     : Isaac Sim closed-loop (x86 전용, sim·gdm 이 떠 있어야 함)
#              기본 build,dataset
#   --tag      S2 replay 이름 (기본 x86 hfp_gguf_q8, Thor thor_q8)
#   이미 있는 빌드 결과물은 재사용한다 (다시 만들려면 command.md 2-0 으로 먼저 치운다).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # vln-deploy/
ARCH=$([ "$(uname -m)" = aarch64 ] && echo aarch64 || echo x86)
STAGES=build,dataset; TAG=$([ $ARCH = aarch64 ] && echo thor_q8 || echo hfp_gguf_q8)
FRAMES=""; RGBD_REF=""; SIM_REPS=4; CKPT=checkpoints/InternVLA-N1-DualVLN
while [ $# -gt 0 ]; do
    case "$1" in
        --stages) STAGES="$2"; shift 2 ;;  --tag) TAG="$2"; shift 2 ;;
        --frames) FRAMES="$2"; shift 2 ;;  --rgbd-ref) RGBD_REF="$2"; shift 2 ;;
        --sim-reps) SIM_REPS="$2"; shift 2 ;;  --ckpt) CKPT="$2"; shift 2 ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
has() { [[ ",$STAGES," == *",$1,"* ]]; }

QK=quantization_kit; R=$QK/out/gguf; O=$QK/verify_s2_gguf_out
Q8=$R/InternVLA-S2-text-Q8_0-latent-embf16.gguf; MM=$R/InternVLA-S2-mmproj-f16.gguf
PY=$QK/convenv_$ARCH/bin/python
LOG="$ROOT/verify/build_out/$(date +%Y%m%d-%H%M%S)-$ARCH"; mkdir -p "$LOG"

if [ $ARCH = x86 ]; then   # 컨테이너 안 경로 = /ws/src/InternNav/vln-deploy
    W="$(cd "$ROOT/../../.." && pwd)"
    dk() { docker run --rm -u 0 --entrypoint bash --gpus all --runtime=nvidia --network=host --ipc=host \
        -v "$W":/ws -v /media/LTDataset/checkpoints:/ws/src/InternNav/checkpoints:ro -v /media/TrainDataset:/ws/src/InternNav/data:ro \
        -e PYTHONDONTWRITEBYTECODE=1 -w /ws/src/InternNav/vln-deploy "$1" -c "$2"; }
    IS() { dk "${IS_IMG:-dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090}" "$1"; }
    GD() { dk "${GD_IMG:-dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-5090}" "$1"; }
    NP=/usr/bin/python3   # 비교(numpy) 단계는 gd_vln 에서
else                       # Thor: 전부 장치에서 바로. Jetson TensorRT 파이썬 바인딩은 시스템 dist-packages
    export PYTHONPATH="/usr/lib/python3.12/dist-packages${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
    IS() { (cd "$ROOT" && bash -c "$1"); }
    GD() { echo "[skip] HF 원본 단계는 x86(gd_vln)에서만: $1"; }
    NP=$PY
fi

HOST() { bash -c "$1"; }
RESULTS=()
run() {   # run "이름" 실행기 "명령"
    local name="$1" t0=$SECONDS; echo; echo "==> $name" | tee -a "$LOG/run.log"
    "$2" "$3" 2>&1 | tee -a "$LOG/run.log"; local rc=${PIPESTATUS[0]}
    RESULTS+=("$([ $rc = 0 ] && echo "  PASS" || echo "  FAIL(rc=$rc)")  $name  ($((SECONDS - t0))s)"); return $rc
}
must() { run "$@" || { printf '%s\n' "${RESULTS[@]}"; echo "중단: $1"; exit 1; }; }
need() { for f in "$@"; do [ -e "$ROOT/$f" ] || { echo "없음: $f  (5090 에서 만들어 복사, command.md Thor 절)"; exit 1; }; done; }
echo "ARCH=$ARCH STAGES=$STAGES TAG=$TAG CKPT=$CKPT  log=$LOG/run.log"

if has build; then
    # Isaac Sim 이 떠 있으면 TRT 엔진 빌드가 멈춘다 (command.md 2-1). 엔진이 이미 있으면 빌드하지 않으므로 괜찮다
    if [ $ARCH = x86 ] && docker ps --format '{{.Names}}' | grep -qx isaac-sim && [ "$(ls "$ROOT"/$QK/out/trt_engines/*.engine 2>/dev/null | wc -l)" -lt 5 ]; then
        echo "Isaac Sim 이 떠 있어 TRT 엔진 빌드가 멈춘다 -> docker stop isaac-sim 후 다시 실행"; exit 2
    fi
    must "2-1 S1 ONNX -> TRT FP16" IS "bash $QK/tools/build_s1_trt.sh $CKPT"
    must "2-2 S2 GGUF -> Q8_0"     IS "bash $QK/tools/build_s2_gguf.sh $CKPT $R"
fi

if has dataset; then
    if [ $ARCH = aarch64 ]; then need $O/hf_bf16_384.npy $O/hf_bf16_384.json $O/s1s2_full_pt__hf_bf16_384.npz
    elif [ ! -f "$ROOT/$O/hf_bf16_384.npy" ]; then must "3-0 [gd_vln] HF S2 기준값" GD "/usr/bin/python3 $QK/verify_s2_gguf.py hf --resize 384 --tag hf_bf16_384"; fi
    must "3-3a S2 Q8_0 replay"  IS "$PY $QK/verify_s2_gguf.py gguf --hf_parity --resize 384 --embd_cache 512 --tag $TAG --text $Q8 --mmproj $MM"
    if [ $ARCH = x86 ]; then
        run  "3-3b [gd_vln] S2 vs HF"            GD "/usr/bin/python3 $QK/verify_s2_gguf.py cmp --ref hf_bf16_384"
        must "3-4a [gd_vln] S1 PyTorch 기준"     GD "/usr/bin/python3 $QK/verify_s1s2_full.py ref --test_tags $TAG"
    else
        run  "3-3b S2 vs HF (S1 ADE 제외)"       IS "$PY $QK/verify_s2_gguf.py cmp --ref hf_bf16_384 --no_s1"
    fi
    must "3-4b S1 TRT FP16"     IS "$PY $QK/verify_s1s2_full.py trt --test_tags $TAG"
    must "3-4c S2+S1 비교"      $([ $ARCH = x86 ] && echo GD || echo IS) "$NP $QK/verify_s1s2_full.py cmp --test_tags $TAG"
    run  "3-4d 합격 판정"        $([ $ARCH = x86 ] && echo GD || echo IS) "$NP -c \"
import json, sys; r = json.load(open('$O/report_s1s2_full.json')); ok = True
for k in ('$TAG+TRT', 'HF+TRT'):
    c, t = r[k]['command_agree(first action)'], r[k]['traj_agree(ADE<5cm)']; ok &= c >= 0.99 and t >= 0.99
    print(f'{k:<18} 명령 일치 {c:.3f}  ADE<5cm {t:.3f}  ', 'PASS' if c >= 0.99 and t >= 0.99 else 'FAIL')
sys.exit(0 if ok else 1)\""
fi

if has rgbd; then
    [ -n "$FRAMES" ] || { echo "--stages rgbd 에는 --frames DIR 이 필요 (verify_rgbd.py export 로 만들 수 있음)"; exit 2; }
    OUT=verify/rgbd_out/$(basename "$FRAMES")-$ARCH
    run "R-1 RGB-D S2+S1 실행" IS "$PY verify/verify_rgbd.py run --frames $FRAMES --mode dual --out $OUT"
    [ -n "$RGBD_REF" ] && run "R-2 RGB-D S1 단독 (기준 실행의 S2 출력 사용)" IS "$PY verify/verify_rgbd.py run --frames $FRAMES --mode s1 --latents $RGBD_REF --out $OUT-s1"
    [ -n "$RGBD_REF" ] && run "R-3 RGB-D 기준 실행과 비교 (S2+S1)" IS "$PY verify/verify_rgbd.py cmp $RGBD_REF $OUT"
    [ -n "$RGBD_REF" ] && run "R-4 RGB-D 기준 실행과 비교 (S1 단독)" IS "$PY verify/verify_rgbd.py cmp $RGBD_REF $OUT-s1"
fi

if has sim; then
    [ $ARCH = x86 ] || { echo "sim 은 x86 전용"; exit 2; }
    run "3-5 Isaac Sim closed-loop x$SIM_REPS" HOST "cd '$ROOT/..' && bash scripts/sim/verify/run_s2_sr.sh $SIM_REPS gguf_q8"
fi

echo; echo "================ SUMMARY ($LOG/run.log) ================"; printf '%s\n' "${RESULTS[@]}"
printf '%s\n' "${RESULTS[@]}" | grep -q FAIL && exit 1 || exit 0
