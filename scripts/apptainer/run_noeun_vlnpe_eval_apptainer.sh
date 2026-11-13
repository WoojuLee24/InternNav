#!/usr/bin/env bash
# 노은역 VLN-PE(Isaac H1 flash) 평가 — apptainer 판. [호스트]
#
# scripts/eval/run_noeun_vlnpe_eval.sh (docker 판) 를 그대로 옮긴 것. 바뀐 곳만:
#   docker exec internnav-isaacsim  ->  apptainer exec instance://$INST
#   docker exec -d ...              ->  nohup apptainer exec ... &   (apptainer 에는 -d 가 없다)
#   컨테이너가 root 가 아니라 호스트 사용자로 돌므로 logs 권한 조작이 필요 없다.
#
#   bash scripts/apptainer/run_noeun_vlnpe_eval_apptainer.sh <task> <ckpt (컨테이너 경로)>
#
# env: SG (모델 서버 GPU, 1) · EG (Isaac 렌더·물리 GPU, 0 — ISAAC_ACTIVE_GPU. 렌더러는 CUDA_VISIBLE_DEVICES 를 안 따른다)
#      INST (apptainer 인스턴스 이름, internnav) · PORT (8031) · STALL_MIN (10) · MAX_RESTART (20) · N_EPS (40)
#      NOEUN_* 는 그대로 넘어간다 (NOEUN_MAX_FOLLOW 로 질의 간격을 바꾼다)
set -euo pipefail
TASK=${1:?task}; M=${2:?ckpt}
INST=${INST:-internnav}
SG=${SG:-1}; EG=${EG:-0}; PORT=${PORT:-8031}; STALL_MIN=${STALL_MIN:-10}; MAX_RESTART=${MAX_RESTART:-20}; N_EPS=${N_EPS:-40}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
L="$ROOT/logs/$TASK"
mkdir -p "$L"
CFG=scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py
P=/workspace/isaaclab/_isaac_sim/python.sh
AX="apptainer exec instance://$INST"

FWD=$(env | grep '^NOEUN_' | grep -v '^NOEUN_TASK=\|^NOEUN_MODEL=\|^NOEUN_PORT=' | sed 's/^/export /' | tr '\n' ';' || true)
C="cd /ws/src/InternNav && export PYTHONPATH=/ws/src/InternNav NOEUN_MODEL=$M NOEUN_TASK=$TASK NOEUN_PORT=$PORT EVAL_OUTPUT_DIR=/ws/src/InternNav/logs/$TASK; $FWD true"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L/watchdog.log"; }

# 이 task 의 <script> 프로세스 (environ 의 NOEUN_TASK 로 가린다 — 호스트에서 바로 보인다, 같은 PID 공간)
task_pids() {
    for p in $(pgrep -f "$1 --config $CFG" 2>/dev/null || true); do
        tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "NOEUN_TASK=$TASK" && echo "$p"
    done | tr '\n' ' '
}

start_server() {  # 서버는 /agent/init 마다 모델을 새로 올린다 -> eval 재시작 때 서버도 다시 (아니면 GPU OOM)
    local p; p=$(task_pids start_server.py)
    if [[ -n "${p// }" ]]; then kill -9 $p 2>/dev/null || true; sleep 10; fi
    log "agent server :$PORT (GPU $SG)"
    nohup $AX bash -c "$C && CUDA_VISIBLE_DEVICES=$SG $P scripts/eval/start_server.py --config $CFG >> logs/$TASK/server.log 2>&1" >/dev/null 2>&1 &
    until curl -s -o /dev/null "http://127.0.0.1:$PORT/docs"; do sleep 5; done
}

n_done() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['val_seen']['Count'])" "$L/result.json" 2>/dev/null || echo 0; }

for r in $(seq 0 "$MAX_RESTART"); do
    start_server
    log "eval 시작 (재시작 $r, 끝난 ep $(n_done))"
    nohup $AX bash -c "$C && CUDA_VISIBLE_DEVICES=$EG ISAAC_ACTIVE_GPU=$EG $P scripts/eval/eval.py --config $CFG >> logs/$TASK/eval.log 2>&1" >/dev/null 2>&1 &
    sleep 60
    last_size=-1; last_change=$(date +%s)
    while true; do
        sleep 30
        pids=$(task_pids eval.py)
        if [[ -z "${pids// }" ]]; then break; fi
        size=$(stat -c %s "$L/eval.log" 2>/dev/null || echo 0)
        if [[ "$size" != "$last_size" ]]; then last_size=$size; last_change=$(date +%s); continue; fi
        if (( $(date +%s) - last_change > STALL_MIN * 60 )); then
            log "eval.log 가 ${STALL_MIN} 분 멈춤 -> pid $pids 종료 후 다시 (끝난 ep $(n_done))"
            kill -9 $pids 2>/dev/null || true
            sleep 10
            break
        fi
    done
    if [[ "$(n_done)" -ge "$N_EPS" ]]; then
        log "끝 — $(cat "$L/result.json")"
        p=$(task_pids start_server.py); [[ -n "${p// }" ]] && kill $p 2>/dev/null || true
        exit 0
    fi
done
log "재시작 ${MAX_RESTART} 번 넘음 — 중단"
exit 1
