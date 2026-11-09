#!/usr/bin/env bash
# 노은역 val 을 sim(RBQ10 물리 보행)으로 평가한다 — ep 마다 sim 을 ep 시작 자세(GUIDEDOG_INIT_*)로 다시 띄우고
# gd_vln 컨테이너에서 scripts/sim/verify/run_noeun_s2_sim_eval.py --only <ep> 를 부른다. [호스트] 에서 실행.
#
#   scripts/sim/run_noeun_sim_eval.sh <policy gt|model> <tag> <episodes(호스트 경로 .json.gz)> [ep id 쉼표 목록]
#   예) GT 보행 검증:  scripts/sim/run_noeun_sim_eval.sh gt gt_v6 /media/TrainDataset/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1/raw_data/r2r/val_seen/val_seen.json.gz
#   예) 모델(경량화): 먼저 agent server 를 띄운다 (internnav-isaacsim, 같은 config):
#       docker exec -d -w /ws/src/InternNav internnav-isaacsim bash -c 'CUDA_VISIBLE_DEVICES=1 NOEUN_AGENT=internvla_n1_s2_gguf NOEUN_GGUF_DIR=<gguf 폴더> NOEUN_TASK=noeun_sim_<tag> /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/start_server.py --config scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py > logs/noeun_sim_server.log 2>&1'
#       NOEUN_AGENT=internvla_n1_s2_gguf NOEUN_GGUF_DIR=<gguf 폴더> scripts/sim/run_noeun_sim_eval.sh model <tag> <episodes>
#
# env: SIM_GPU (sim 렌더 · 물리 GPU, 0 — GUIDEDOG_ACTIVE_GPU. 컨테이너가 privileged 라 NVIDIA_VISIBLE_DEVICES 로는 못 막는다) · SIM_READY_S (sim 로딩 대기, 600) · EVAL_ARGS (run_noeun_s2_sim_eval.py 추가 인자) · NOEUN_* (agent config, 컨테이너로 넘긴다)
# 결과: scripts/sim/verify/out/noeun_s2_sim/<tag>/<ep>/{episode.json,steps.jsonl,path.csv}, sim 로그 .../<ep>/sim.log
set -euo pipefail
POLICY=${1:?policy gt|model}; TAG=${2:?tag}; EPS_HOST=${3:?episodes .json.gz (host path)}; ONLY=${4:-}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"           # InternNav (host)
SIM_DIR="$ROOT/../../../../sim_docker"                              # ~/gd_project/modules/sim_docker
SIM_DIR="$(cd "$SIM_DIR" && pwd)"
USDZ_SRC=/media/TrainDataset/GS_USDZ/noeun_station_collision.usdz
OUT="$ROOT/scripts/sim/verify/out/noeun_s2_sim/$TAG"
GD=internnav-habitat
EPS_CTR=${EPS_HOST/#\/media\/TrainDataset/\/ws\/src\/InternNav\/data}  # gd_vln 컨테이너: /media/TrainDataset -> data
mkdir -p "$OUT"

# 1) 노은역 장면 (VLN-PE 와 같은 collision usdz: 화면 = GS, 충돌 · lidar · depth = mesh). sim 컨테이너가 읽는 곳에 둔다.
mkdir -p "$SIM_DIR/workspace/NoeunStation"
[[ -f "$SIM_DIR/workspace/NoeunStation/noeun_station_collision.usdz" ]] || cp "$USDZ_SRC" "$SIM_DIR/workspace/NoeunStation/"
# 2) 컨테이너
# sim_docker/run.sh 의 준비(캐시 폴더 · 센서 설정 링크)를 먼저 한다. 없으면 docker 가 root 소유 폴더로 만들어 sim 이 죽는다.
(cd "$SIM_DIR" && mkdir -p isaac-sim/cache/main/ov isaac-sim/cache/main/warp isaac-sim/cache/computecache isaac-sim/config \
    isaac-sim/data/documents isaac-sim/data/Kit isaac-sim/logs isaac-sim/pkg \
 && { [[ -e sensor_configuration.yaml ]] || ln -sfn "$( [[ -f ../gdm_docker/workspace/src/gdm_planning/resource/sensor_configuration.yaml ]] && echo ../gdm_docker/workspace/src/gdm_planning/resource/sensor_configuration.yaml || echo _sensor_configuration.yaml )" sensor_configuration.yaml; })
docker ps --format '{{.Names}}' | grep -qx isaac-sim || (cd "$SIM_DIR" && GID=$(id -g) USER_NAME=$(id -un) docker compose -f compose.yml -f "$ROOT/scripts/sim/sim_docker_a5000.override.yml" up -d)
"$ROOT/docker/run-torch2.9.0-cuda13.0-habitat-jazzy-a5000.sh" -d >/dev/null

# 3) ep 목록과 시작 자세 (habitat [x,y,z] Y-up -> sim world [x,-z,y], yaw = run_r2r_eval.hab_yaw_to_world)
mapfile -t ROWS < <(python3 - "$EPS_HOST" "$ONLY" <<'PY'
import gzip, json, math, sys
eps = json.load(gzip.open(sys.argv[1]))["episodes"]
only = [s for s in sys.argv[2].split(",") if s]
for e in eps:
    if only and str(e["episode_id"]) not in only:
        continue
    x, y, z = e["start_position"]
    r = e["start_rotation"]
    yaw = math.degrees(2.0 * math.atan2(r[1], r[3]) + math.pi / 2)
    print(e["episode_id"], f"{x:.3f}", f"{-z:.3f}", f"{y:.3f}", f"{yaw:.2f}")
PY
)
echo "[noeun_sim] ${#ROWS[@]} ep · policy=$POLICY tag=$TAG -> $OUT"

stop_sim() {
    docker exec isaac-sim pkill -INT -f run_rbq10_sim.py 2>/dev/null || true
    for _ in $(seq 60); do docker exec isaac-sim pgrep -f run_rbq10_sim.py >/dev/null 2>&1 || return 0; sleep 2; done
    docker exec isaac-sim pkill -KILL -f run_rbq10_sim.py 2>/dev/null || true
}
trap stop_sim EXIT

ENV_FWD=$(env | grep '^NOEUN_' | sed 's/^/export /' | tr '\n' ';' || true)
for row in "${ROWS[@]}"; do
    read -r EID X Y FLOOR YAW <<<"$row"
    [[ -f "$OUT/$EID/episode.json" ]] && { echo "[noeun_sim] ep $EID 있음 -> 건너뜀"; continue; }
    mkdir -p "$OUT/$EID"
    stop_sim
    echo "[noeun_sim] ep $EID 시작 ($X, $Y) 바닥 z $FLOOR yaw $YAW"
    # 노은역 런처(run_rbq10_etri_sim.sh noeun)와 같은 값 + 시작 자세. 사람 없음(HUNAV=0), 창 없음.
    (cd "$SIM_DIR" && env GUIDEDOG_HEADLESS=1 GUIDEDOG_HUNAV=0 GUIDEDOG_ACTIVE_GPU="${SIM_GPU:-0}" \
        GUIDEDOG_BACKGROUND_USD=/isaac-sim/workspace/NoeunStation/noeun_station_collision.usdz \
        GUIDEDOG_BACKGROUND_COLLISION=1 GUIDEDOG_BACKGROUND_MESH_VISIBLE=0 \
        GUIDEDOG_LIDAR_TYPE=physx GUIDEDOG_DEPTH_TYPE=physx GUIDEDOG_VIEWPORT_LIGHT=camera GUIDEDOG_FOLLOW_CAM=0 \
        GUIDEDOG_INIT_X="$X" GUIDEDOG_INIT_Y="$Y" GUIDEDOG_INIT_FLOOR_Z="$FLOOR" GUIDEDOG_INIT_Z=-100 GUIDEDOG_INIT_YAW="$YAW" \
        LAUNCHER_TAG=noeun_sim_eval ./_rbq10_common.sh </dev/null >"$OUT/$EID/sim.log" 2>&1 &)
    docker exec -w /ws/src/InternNav $GD bash -c "$ENV_FWD source scripts/sim/setup_ros_ws.sh >/dev/null && python3 scripts/sim/verify/run_noeun_s2_sim_eval.py --policy $POLICY --tag $TAG --episodes $EPS_CTR --only $EID --ready_timeout_s ${SIM_READY_S:-600} ${EVAL_ARGS:-}" \
        2>&1 | tee "$OUT/$EID/eval.log" | grep --line-buffered '\[noeun_sim\]\|Error\|error' || true
done
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
rs = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("*/episode.json"))]
n = max(len(rs), 1)
agg = {k: round(sum(float(r[k]) for r in rs) / n, 4) for k in ("success", "oracle_success", "spl", "ne_m", "fall")} | {"blocked": round(sum(float(r.get("blocked", False)) for r in rs) / n, 4)}
(Path(sys.argv[1]) / "summary.json").write_text(json.dumps({"n": len(rs), "mean": agg, "episodes": rs}, indent=1, ensure_ascii=False))
print(f"[noeun_sim] {len(rs)} ep · {agg}")
PY
