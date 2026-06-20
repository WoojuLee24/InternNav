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
        printf "\r[$(date '+%Y-%m-%d %H:%M:%S')] Queue empty, waiting 30s..."
        sleep 30
        continue
    fi

    NEXT=$(sed -n "${LINE_NUM}p" "$QUEUE_FILE")

    # 남은 non-comment 라인 수 확인
    REMAINING=$(grep -c '^\s*[^#]' "$QUEUE_FILE" 2>/dev/null || echo 0)

    log "========================================"
    if [ "$REMAINING" -gt 1 ]; then
        log "Running: $NEXT"
        log "========================================"

        # 줄 번호 기반으로 comment out (특수문자 안전)
        sed -i "${LINE_NUM}s|.*|# ${NEXT}|" "$QUEUE_FILE"
    else
        log "Running (last, repeating): $NEXT"
        log "========================================"
        # 마지막 라인은 done 처리 없이 반복 실행
    fi

    eval "$NEXT" 2>&1 | tee -a "$LOG_FILE"

    log "Finished: $NEXT"
    log "========================================"
done
