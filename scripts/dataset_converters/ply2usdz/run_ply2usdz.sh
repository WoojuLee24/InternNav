#!/usr/bin/env bash
# 원본 3DGS PLY + 충돌 mesh → <name>_collision.usdz. 호스트에서 실행한다 (GPU 필요, ply2usdz.py 설명 참고).
#   run_ply2usdz.sh <gs.ply> <mesh.obj|mesh.ply> <out.usdz> [--nurec-name NAME] [--near-clip 1e-8]
#   예) run_ply2usdz.sh NoeunStation/point_cloud.ply mesh.ply out/noeun_station_collision.usdz --nurec-name noeun_station
# 3DGRUT 환경 = usd2usdz setup_3dgrut_venv.sh 로 만든 venv(기본 ~/git/3dgrut-venv, 컨테이너 안 /opt/3dgrut 고정)
set -euo pipefail
[[ $# -ge 3 ]] || { sed -n 2,5p "$0" >&2; exit 2; }
VENV=${VENV:-$HOME/git/3dgrut-venv}
IMAGE=${IMAGE:-dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090}
HERE="$(cd "$(dirname "$0")" && pwd)"
GS=$(realpath "$1"); MESH=$(realpath "$2"); OUT=$(realpath -m "$3"); shift 3
mkdir -p "$(dirname "$OUT")"

# 입력·출력 폴더를 같은 경로로 마운트한다 (같은 폴더는 한 번만 — docker 는 중복 마운트를 거부한다)
MOUNTS=(-v "$(dirname "$OUT")":"$(dirname "$OUT")")
for d in $(printf '%s\n' "$(dirname "$GS")" "$(dirname "$MESH")" | sort -u); do
    [[ $d == "$(dirname "$OUT")" ]] || MOUNTS+=(-v "$d":"$d":ro)
done
# venv 의 python 은 /isaac-sim(isaac-sim 사용자 전용)을 가리켜 root 로 돌리고 출력 소유자만 되돌린다
docker run --rm --gpus all --memory "${MEM:-80g}" --entrypoint bash -e HOME=/tmp -e OWNER="$(id -u):$(id -g)" \
    -e PATH=/opt/3dgrut/venv/bin:/usr/local/cuda-12.8/bin:/usr/bin:/bin \
    -v "$VENV":/opt/3dgrut:ro -v "$HERE":/ply2usdz:ro "${MOUNTS[@]}" \
    "$IMAGE" -c 'cd /tmp && python /ply2usdz/ply2usdz.py "$@"; rc=$?; chown "$OWNER" "$3" 2>/dev/null; exit $rc' _ "$GS" "$MESH" "$OUT" "$@"
