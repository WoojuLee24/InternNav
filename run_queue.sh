#!/bin/bash
# run_queue.sh

QUEUE_FILE="queue.txt"
LOG_FILE="run_queue.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "===== run_queue.sh started (PID=$$) ====="

while true; do
    # 줄 번호로 가져와서 sed 특수문자 문제 회피
    LINE_NUM=$(grep -n -m1 '^\s*[^#]' "$QUEUE_FILE" 2>/dev/null | cut -d: -f1)

    if [ -z "$LINE_NUM" ]; then
        log "Queue empty, waiting 30s..."
        sleep 30
        continue
    fi

    NEXT=$(sed -n "${LINE_NUM}p" "$QUEUE_FILE")

    log "========================================"
    log "Running: $NEXT"
    log "========================================"

    # 줄 번호 기반으로 comment out (특수문자 안전)
    sed -i "${LINE_NUM}s|.*|# [done] ${NEXT}|" "$QUEUE_FILE"

    eval "$NEXT" 2>&1 | tee -a "$LOG_FILE"

    log "Finished: $NEXT"
    log "========================================"
done
