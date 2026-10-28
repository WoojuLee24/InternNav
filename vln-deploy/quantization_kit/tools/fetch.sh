#!/usr/bin/env bash
# git 에 올리지 않는 외부 자산을 받는다 (docs/command.md 0-3, 1-3). vln-deploy/ 에서 실행:
#   bash quantization_kit/tools/fetch.sh llama           llama.cpp upstream 389ff61d7 + 우리 patch -> quantization_kit/llama.cpp
#   bash quantization_kit/tools/fetch.sh ckpt  [DEST]    원본 체크포인트 InternVLA-N1-DualVLN (공개, 16 GB)  기본 DEST=checkpoints/InternVLA-N1-DualVLN
#   bash quantization_kit/tools/fetch.sh data  [DEST]    InternData-N1 vln_ce R2R 검증용 6 장면 (gated: HF 승인 + HF_TOKEN 필요)
#                                                        기본 DEST=../data/InternData-N1-v0.5-mini
# ckpt/data 는 huggingface_hub 가 있는 python 을 쓴다 (기본 convenv, 없으면 python3; PYH=... 로 지정).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."      # vln-deploy/
ARCH=$([ "$(uname -m)" = aarch64 ] && echo aarch64 || echo x86)

LLAMA_REPO=https://github.com/ggml-org/llama.cpp.git
LLAMA_BASE=389ff61d77b5c71cec0cf92fe4e5d01ace80b797     # 2026-05-10, vendored 소스와 tools/mtmd 8 파일 외 동일
CKPT_REPO=InternRobotics/InternVLA-N1-DualVLN; CKPT_REV=a698a9e898b4001621a319e1bc89f02ec715cc86
DATA_REPO=InternRobotics/InternData-N1;        DATA_REV=cd04ff65aa5f8647e70ae66eba737c0454883211
SCENES="17DRP5sb8fy 1LXtFkjw3qL 1pXnuDYAj8r 29hnd4uzFmX 2n8kARJN3HM 5LpN3gDmAk7"   # verify_s2_gguf.py episodes(): 이름순 앞 6개

pyh() {   # huggingface_hub 가 import 되는 python
    for p in "${PYH:-}" quantization_kit/convenv_$ARCH/bin/python python3; do
        [ -n "$p" ] && "$p" -c "import huggingface_hub" 2>/dev/null && { echo "$p"; return; }
    done
    echo "huggingface_hub 가 없음: pip install -U huggingface_hub 후 PYH=<python> 로 다시" >&2; exit 1
}

case "${1:-}" in
llama)
    D=quantization_kit/llama.cpp
    [ -e "$D" ] && { echo "$D 가 이미 있음 — 다시 받으려면 먼저 옮기거나 지운다"; exit 1; }
    git init -q "$D" && git -C "$D" remote add origin "$LLAMA_REPO"
    git -C "$D" fetch -q --depth 1 origin "$LLAMA_BASE" && git -C "$D" checkout -q FETCH_HEAD   # 커밋 하나만 (~33 MB)
    git -C "$D" apply "$PWD/quantization_kit/tools/llama.cpp-vln.patch"   # tools/mtmd: s2-runner · guidedog-cli · HF parity · embd cache
    git -C "$D" status --short | head -20
    echo "[fetch] llama.cpp @ ${LLAMA_BASE:0:9} + patch -> $D  (빌드: command.md 1-3)" ;;
ckpt)
    D=${2:-checkpoints/InternVLA-N1-DualVLN}
    "$(pyh)" -c "from huggingface_hub import snapshot_download as s; print(s('$CKPT_REPO', revision='$CKPT_REV', local_dir='$D'))"
    n=$(ls "$D"/*.safetensors | wc -l); [ "$n" -ge 4 ] || { echo "safetensors $n 개 (4 개여야 함)"; exit 1; }
    echo "[fetch] $CKPT_REPO @ ${CKPT_REV:0:7} -> $D" ;;
data)
    D=${2:-../data/InternData-N1-v0.5-mini}
    pats=$(for s in $SCENES; do printf "'vln_ce/traj_data/r2r/%s.tar.gz'," "$s"; done)
    "$(pyh)" -c "from huggingface_hub import snapshot_download as s; s('$DATA_REPO', repo_type='dataset', revision='$DATA_REV', local_dir='$D', allow_patterns=[$pats])"
    for s in $SCENES; do
        [ -d "$D/vln_ce/traj_data/r2r/$s" ] || tar xzf "$D/vln_ce/traj_data/r2r/$s.tar.gz" -C "$D/vln_ce/traj_data/r2r/"
        [ -f "$D/vln_ce/traj_data/r2r/$s/meta/episodes.jsonl" ] || { echo "$s 압축 해제 결과가 예상과 다름"; exit 1; }
    done
    echo "[fetch] $DATA_REPO @ ${DATA_REV:0:7} 6 장면 -> $D/vln_ce/traj_data/r2r" ;;
*)
    sed -n '2,7p' "$0"; exit 2 ;;
esac
