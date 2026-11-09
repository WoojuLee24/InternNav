#!/usr/bin/env bash
# 노은역 VLN-PE(Isaac H1 flash) 평가 — agent server + eval 을 internnav-isaacsim 에서 띄우고, eval 이 멈추면 다시 띄운다. [호스트]
#
#   scripts/eval/run_noeun_vlnpe_eval.sh <task> <ckpt (컨테이너 경로)>
#   예) SG=1 EG=0 NOEUN_FOLLOWER=navmesh NOEUN_DATA=/ws/src/InternNav/data/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1/raw_data/r2r scripts/eval/run_noeun_vlnpe_eval.sh noeun_v01_bothb1wait_v6b1_navmesh /ws/src/InternNav/checkpoints/relabel_half_noeun-v01/s1.noeun-both-b1_wait-wait_internvla-n1-system2_20261029_104936
#
# env: SG (서버 GPU, 1) · EG (Isaac 렌더 · 물리 GPU, 0 — ISAAC_ACTIVE_GPU 로 넘긴다. 렌더러는 CUDA_VISIBLE_DEVICES 를 따르지 않는다)
#      PORT (8031) · STALL_MIN (eval.log 가 이 분 동안 안 늘면 멈춘 것으로 보고 다시 띄움, 10) · MAX_RESTART (20)
#      NOEUN_* (config 값, 그대로 넘긴다; follower=navmesh 면 noeun_navmesh_server.py 가 떠 있어야 한다)
# 왜: Isaac 렌더(rep.orchestrator.step)가 시작 ~1 시간 뒤 프레임을 안 내고 멈춘다 (2026-11-07 세 번, 위치 무관). 평가기는
#     끝난 ep 을 lmdb 에 남겨 같은 task 로 다시 띄우면 건너뛴다 (ResumablePathKeyEpisodeloader) — 멈춘 ep 만 처음부터 다시 돈다.
# 결과: logs/<task>/ (result.json · progress/ · agent/ · eval.log · server.log · watchdog.log)
set -euo pipefail
TASK=${1:?task}; M=${2:?ckpt}
SG=${SG:-1}; EG=${EG:-0}; PORT=${PORT:-8031}; STALL_MIN=${STALL_MIN:-10}; MAX_RESTART=${MAX_RESTART:-20}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
L="$ROOT/logs/$TASK"
docker exec internnav-isaacsim bash -c "mkdir -p /ws/src/InternNav/logs/$TASK && chmod a+rwx /ws/src/InternNav/logs/$TASK"  # 컨테이너(root)가 만든 폴더에 호스트가 watchdog.log 를 쓴다
CFG=scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py
P=/workspace/isaaclab/_isaac_sim/python.sh
FWD=$(env | grep '^NOEUN_' | grep -v '^NOEUN_TASK=\|^NOEUN_MODEL=\|^NOEUN_PORT=' | sed 's/^/export /' | tr '\n' ';' || true)
C="cd /ws/src/InternNav && export PYTHONPATH=/ws/src/InternNav NOEUN_MODEL=$M NOEUN_TASK=$TASK NOEUN_PORT=$PORT EVAL_OUTPUT_DIR=/ws/src/InternNav/logs/$TASK; $FWD true"  # FWD 가 ; 로 끝나므로 뒤 && 앞에 true
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L/watchdog.log"; }
task_pids() {  # 이 task 의 <script> 프로세스들 (bash -c · python.sh · python3 모두 — environ 의 NOEUN_TASK 로 가린다)
    docker exec internnav-isaacsim bash -c "for p in \$(pgrep -f '$1 --config $CFG'); do tr '\\0' '\\n' < /proc/\$p/environ 2>/dev/null | grep -qx 'NOEUN_TASK=$TASK' && echo \$p; done; true" 2>/dev/null | tr '\n' ' ' || true
}
start_server() {  # 서버는 /agent/init 마다 모델을 새로 올린다 -> eval 을 다시 띄울 때 서버도 다시 (아니면 GPU OOM)
    local p; p=$(task_pids start_server.py)
    if [[ -n "${p// }" ]]; then docker exec internnav-isaacsim kill -9 $p || true; sleep 10; fi
    log "agent server :$PORT (GPU $SG)"
    docker exec -d internnav-isaacsim bash -c "$C && CUDA_VISIBLE_DEVICES=$SG $P scripts/eval/start_server.py --config $CFG >> logs/$TASK/server.log 2>&1"
    until curl -s -o /dev/null "http://127.0.0.1:$PORT/docs"; do sleep 5; done
}
eval_pids() {  # 이 task 의 eval 프로세스들 (bash -c · python.sh · python3 모두 — environ 의 NOEUN_TASK 로 가린다)
    docker exec internnav-isaacsim bash -c "for p in \$(pgrep -f 'eval.py --config $CFG'); do tr '\\0' '\\n' < /proc/\$p/environ 2>/dev/null | grep -qx 'NOEUN_TASK=$TASK' && echo \$p; done; true" 2>/dev/null | tr '\n' ' ' || true
}
n_done() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['val_seen']['Count'])" "$L/result.json" 2>/dev/null || echo 0; }


for r in $(seq 0 "$MAX_RESTART"); do
    start_server
    log "eval 시작 (재시작 $r, 끝난 ep $(n_done))"
    # 2026-11-09: EG=2 · 3 은 이 컨테이너에서 렌더 장치를 못 만든다 ("No device could be created", CUDA_VISIBLE_DEVICES 유무 무관) — EG=0 을 같이 쓴다
    docker exec -d internnav-isaacsim bash -c "$C && CUDA_VISIBLE_DEVICES=$EG ISAAC_ACTIVE_GPU=$EG $P scripts/eval/eval.py --config $CFG >> logs/$TASK/eval.log 2>&1"
    sleep 60
    last_size=-1; last_change=$(date +%s)
    while true; do
        sleep 30
        pids=$(eval_pids)
        if [[ -z "${pids// }" ]]; then break; fi        # eval 끝남 (정상 종료 또는 죽음)
        size=$(stat -c %s "$L/eval.log")
        if [[ "$size" != "$last_size" ]]; then last_size=$size; last_change=$(date +%s); continue; fi
        if (( $(date +%s) - last_change > STALL_MIN * 60 )); then
            log "eval.log 가 ${STALL_MIN} 분 멈춤 -> pid $pids 종료 후 다시 (끝난 ep $(n_done))"
            docker exec internnav-isaacsim kill -9 $pids || true
            sleep 10
            break
        fi
    done
    if [[ "$(n_done)" -ge 40 ]]; then
        log "끝 — $(cat "$L/result.json")"
        p=$(task_pids start_server.py); [[ -n "${p// }" ]] && docker exec internnav-isaacsim kill $p || true  # 서버가 GPU 를 계속 잡지 않게
        exit 0
    fi
done
log "재시작 ${MAX_RESTART} 번 넘음 — 중단"
exit 1
