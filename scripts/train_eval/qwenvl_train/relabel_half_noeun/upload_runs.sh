#!/usr/bin/env bash
# relabel_half_noeun-v01 학습 결과를 Drive 로 올린다 — rclone gdrive 가 설정된 노드(node1)에서 한 번 띄워 두면 된다.
# ckpt 는 공유 스토리지(data-vol2)에 쌓이므로 학습한 노드와 상관없이 여기서 올린다 (다른 노드의 rclone 설정 불필요).
#   끝난 run = 최상위에 model.safetensors.index.json + trainer_state.json 이 있고 10 분 넘게 안 바뀜
#   올리는 것 = 최종 모델 · tokenizer · config · logs/ (checkpoint-*/ 의 optimizer 와 wandb/ 는 뺌)
#   올린 run 에는 UPLOADED 표시를 남겨 다시 올리지 않는다. 모든 run(기본 14) 이 올라가거나 MAX_HOURS 가 지나면 끝.
# 사용 (저장소 루트):
#   setsid nohup bash scripts/train_eval/qwenvl_train/relabel_half_noeun/upload_runs.sh > /home/irteam/data-vol2/checkpoints/relabel_half_noeun-v01/upload.log 2>&1 &
set -u
SRC=${SRC:-/home/irteam/data-vol2/checkpoints/relabel_half_noeun-v01}
DST=${DST:-gdrive:InternNav/checkpoints/relabel_half_noeun-v01}
N_EXPECT=${N_EXPECT:-14}
MAX_HOURS=${MAX_HOURS:-36}
POLL_S=${POLL_S:-300}
start=$(date +%s)
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log "upload_runs 시작 SRC=$SRC DST=$DST N_EXPECT=$N_EXPECT"
while true; do
    n_done=0
    for run in "$SRC"/s1.*_internvla-n1-system2_*/; do
        [ -d "$run" ] || continue
        run=${run%/}
        if [ -f "$run/UPLOADED" ]; then n_done=$((n_done + 1)); continue; fi
        [ -f "$run/model.safetensors.index.json" ] && [ -f "$run/trainer_state.json" ] || continue
        if [ -n "$(find "$run" -maxdepth 1 -newermt '-10 minutes' -print -quit)" ]; then continue; fi   # 아직 쓰는 중일 수 있음
        name=$(basename "$run")
        log "올림: $name"
        if rclone copy "$run" "$DST/$name" --exclude "checkpoint-*/**" --exclude "wandb/**" --exclude "UPLOADED" --transfers 4 --drive-chunk-size 64M; then
            rclone check "$run" "$DST/$name" --one-way --exclude "checkpoint-*/**" --exclude "wandb/**" --exclude "UPLOADED" >/dev/null 2>&1 \
                && { date '+%Y-%m-%d %H:%M:%S' > "$run/UPLOADED"; log "완료: $name"; n_done=$((n_done + 1)); } \
                || log "check 실패 (다음 회차에 다시): $name"
        else
            log "rclone copy 실패 (다음 회차에 다시): $name"
        fi
    done
    log "올림 $n_done / $N_EXPECT"
    [ "$n_done" -ge "$N_EXPECT" ] && { log "전부 올림 — 끝"; exit 0; }
    [ $(( $(date +%s) - start )) -gt $(( MAX_HOURS * 3600 )) ] && { log "MAX_HOURS 지남 — 끝"; exit 1; }
    sleep "$POLL_S"
done
